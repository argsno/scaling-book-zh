---
layout: distill
title: "Profiling TPUs（如何理解 TPU 性能剖析）"
permalink: /profiling-zh/
description: "到目前为止，本系列完全是理论性的：基于硬件屋顶线（roofline）的粗略估算。这种理解能让你走得很远，但大量优化归根结底要落实到实际细节上：XLA 编译器是如何工作的，以及如何使用像 JAX/TensorBoard Profiler 这样的性能剖析工具，来弄清楚出问题时该怎么办。我们在这里讨论这些内容。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

sitemap: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "Part 8: Serving LLaMA"

next_section_url: ../jax-stuff
next_section_name: "Part 10: JAX"

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "TPU 软件栈概览"
  - name: "JAX Profiler：一款多用途 TPU 性能分析器"
  - subsections:
    - name: "Trace Viewer（追踪视图）"
    - name: "如何阅读一个 XLA 操作"
    - name: "Graph Viewer（图视图）"
    - name: "看一个（接近）真实的性能剖析示例"
    - name: "Memory Profile（内存分析）"
  - name: "习题"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

## TPU 软件栈概览 {#tpu-软件栈概览}

Google 提供了一系列用于编程 TPU 的 API，从高级的 JAX 代码到低级的 Pallas 或 HLO 都有。大多数程序员只写 JAX 代码，它让你编写抽象的、NumPy 风格的线性代数程序，这些程序会被自动编译，从而在 TPU 上高效运行。

下面是一个简单的例子，一个将两个矩阵相乘的 JAX 程序：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用 `jax.jit`，我们告诉 JAX 跟踪（trace）这个函数，并发出一个更低层级的 IR，称为 [StableHLO](https://openxla.org/stablehlo)，这是一种与平台无关的 ML 计算 IR，它进而被 XLA 编译器降级（lower）为 HLO。编译器会执行许多遍（pass）来确定融合（fusion）、布局（layout）以及其他因素，最终生成可以在 JAX profile 中观察到的 HLO。这种 HLO 以 LLVM 风格的图视图来表示 JAX 代码中的所有核心线性代数操作（矩阵乘、逐点操作、卷积等）。例如，下面是上述程序经删节后的 HLO 版本<d-footnote>要得到这段 HLO，你可以运行 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`。</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们稍后会解释 HLO 的语法，但现在只需注意，它其实与上面的 JAX 代码相当吻合。例如，

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

就是上面那个真正的矩阵乘，它分别沿第 0 维和第 1 维对两个 f32 矩阵做乘法。

**要将这段 HLO 转换为可在 TPU 上执行的代码，XLA 编译器首先会把它降级（lower）为 LLO**（low-level optimizer，底层优化器）IR。LLO 直接对 TPU 编程，安排各内存之间的拷贝、将数组推入脉动阵列（systolic array）等。LLO 代码包含这样的原语：把缓冲区推入脉动阵列、把结果取出来，以及调度在不同 TPU 内存之间通信的 DMA。一旦降到 LLO 之后，它就会被编译为机器码，加载进 TPU 的 IMEM 并执行。

当某个程序运行得比我们期望的慢时，我们主要是在 JAX 这一层来做性能优化。然而，这么做往往要求我们理解 HLO 的一些语义，以及代码在 TPU 上是如何真正运行的。当更底层出现问题时，我们会再拉一道"逃生舱口"，用 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html) 编写自定义 kernel。要查看某个程序的 HLO 及其运行时统计信息，我们使用 JAX profiler。

## JAX Profiler：一款多用途 TPU 性能分析器 {#jax-profiler-一款多用途-tpu-性能分析器}

JAX 提供了一个多用途的 TPU profiler，带有一系列有用的工具，用于理解程序运行时 TPU 上正在发生什么。你可以使用 `jax.profiler` 模块在程序运行时对它做 trace，并记录从各个子组件的耗时、每个程序的 HLO、内存使用量等方方面面。例如，下面这段代码会把一份 trace 转储到 `/tmp/tensorboard` 下的文件中，该文件可在 TensorBoard 中查看（[这里](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling) 有分步指南）。

```py
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# Now you can load TensorBoard in a Google Colab with
#
# !pip install -U xprof
# !pip install -U protobuf
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# or externally with
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

下面是你在 profiler 中可以做的操作概览：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

一旦进入 TensorBoard，profiler 有几个关键标签页可以帮助你理解自己的程序：

1. **Trace Viewer** 展示 TPU 上实际正在发生的事情的详细时间线。
2. **Graph Viewer** 展示 HLO 图，让你看到程序的哪些部分相互喂入（feed into），以及各部分是怎样分片（shard）的。
3. **Memory Profile 与 Memory Viewer**：它们展示你的程序占用了多少内存。

虽然分享 profile 多少有点麻烦，但[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) 有一个 Perfetto 链接，其中至少包含了某个简单 Transformer 的 Trace Viewer 部分。[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 让你可以生成完整的 JAX/TensorBoard trace 并随意把玩。

### Trace Viewer（追踪视图） {#trace-viewer-追踪视图}

**Trace Viewer 大概是 profiler 中最有用的部分。** 下面的例子展示了一个标注了各部分的简单 Transformer。名称来自代码中提供的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer 展示每个 TPU 核心上所有操作按时间顺序排列的时间线。这里我们只看 TPU:0，因为通常所有 TPU 都执行相同的指令。几个要点：

1. 最上面一行（XLA Ops）展示的是真正的 TPU 操作（名称即 HLO 名称）。其余部分是基于 `jax.named_scope`、`jax.named_call` 以及 Python 栈追踪得到的近似 trace。
2. 注意到那些重复出现的块，我们就可以在这里单独拎出某一层。我们也可以看出（通过查看代码/理解 Transformer 的工作原理）哪些部分是 attention，哪些部分是 MLP。
3. 点击某个 XLA 操作，我们可以查看它来自代码中的哪个位置（这对理解 trace 很有用），并看到指向 Graph Viewer 的链接。

<p markdown=1 class="takeaway">**小贴士：** 你可以用"电子游戏"风格的操作方式来浏览 Trace Viewer，用 A/D 左右平移，用 W/S 放大缩小。这些操作让浏览轻松很多。</p>

### 如何阅读一个 XLA 操作 {#如何阅读一个-xla-操作}

HLO 其实并不难读，而且它对于理解上面 trace 中某一部分对应什么非常有帮助。下面是一个名为 fusion.3 的示例操作。

```c
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

我们把它拆开来逐条说明。

* **操作名（Op Name）**：fusion.3
  * 一个 dot 或 fusion 操作是一组操作，其中最多包含 1 次矩阵乘法，以及可能的一堆相关逐点 VPU 操作。
* **形状（Shape）**：`bf16[32,32,4096]`
  * 这是该操作的输出形状。可以看出 dtype 是 bf16（每个元素 2 字节），而 `[32,32,4096]` 是形状。
* **布局（Layout）：** `{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}` 告诉我们各轴在内存中的排列顺序（列主序、行主序等）以及数组的填充（padding）。详见下文。
* **内存位置（Memory location）：** S(1)
  * S(1) 表示这个数组位于 VMEM 中。S(0)（有时会省略）是 HBM。S(2) 和 S(3) 是其他内存空间。
* **参数（Arguments）：** `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 这个操作有一个输入，是一个名为 fusion.32、具有特定形状的 bf16 数组。这告诉我们是哪个函数喂入到这个操作里。

我们再来稍微深入理解一下这套记法。我们以这个作为简单例子：

`f32[3,5]{1,0:T(2,2)}`

它同样告诉我们，这个操作返回一个 float32 数组，形状为 `[3, 5]`，并带有特定的分块（tiling）`{1,0:T(2,2)}`。虽然分块不算*特别*重要，但简要地说，分块告诉我们一个 N 维数组在内存中是怎样顺序排布的。下面这张图展示了这个数组的排布方式：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在 `{1,0:T(2,2)}` 中，`1,0` 这部分告诉我们数组各维度在物理内存中的顺序，从最次（minor）到最主（major）。你可以从右向左读这部分，并在 `f32[3,5]` 中挑出对应的维度，从而弄清数组的物理布局。在这个例子中，物理布局是 `[3,5]`，与逻辑形状相同。

接着，`T(2,2)` 告诉我们数组以 `(2, 2)` 为块做分块，在每个块内部，数组先排行（**row-major**，行主序），再排列，即 `(0, 0)` 之后是 `(0, 1)`，然后是 `(1, 0)` 和 `(1, 1)`。因为 `T(2, 2)` 分块，数组被填充（pad）到 `[4, 6]`，内存占用因此增大约 1.6 倍。对于上面那个大的 bf16 数组 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，我们用的是 `T(8,128)(2,1)`，这意味着数组有两级分块：外层 `(8, 128)` 分块，以及在该单元内部的内层 `(2, 1)` 分块（用于 bf16，以便我们的加载总是 4 字节的整数倍）。例如，下面是 `bf16[4,8]{1,0:T(2,4)(2,1)}`（颜色标出的是 (2,4) 块，红框是 (2,1) 块）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

分块会影响张量块被加载进 VMEM 的效率，而 XLA 有时会引入一些拷贝，用于在程序内部对张量做"重分块（retile）"或"重布局（re-layout）"，这有时会带来不小的开销。<d-footnote>JAX 提供了一个<a href="https://docs.jax.dev/en/latest/notebooks/layout.html">实验性特性</a>来解决此问题：它允许 XLA 为程序输入计算其"偏好（preferred）"的布局。当你用 `jax.jit` "即时（just-in-time）"编译一个程序时，通常会传入"mock"（模拟）输入，告诉 JAX 预期的形状和 dtype。这些输入通常也带有分块信息，而那未必是最优的。作为替代，你可以把输入布局指定为 AUTO，于是 `jax.jit` 会返回一个被 jit 的程序所偏好的布局。随后你可以用该布局显式加载张量，从而避免程序内部产生拷贝。</d-footnote>

### Graph Viewer（图视图） {#graph-viewer-图视图}

尽管上面有些 fusion 看着挺复杂，但 XLA Graph Viewer 让它们更易读。例如，下面是一个相当复杂的 fusion 的视图：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

盯着一堆 HLO 图看，并尝试把 HLO 操作映射到你所剖析的代码上，是非常有用的。把鼠标悬停在某个方框上，你常常能看到定义该函数的代码行。

### 看一个（接近）真实的性能剖析示例 {#看一个-接近-真实的性能剖析示例}

[这个 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 里有一个假 Transformer 的示例 profile。[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a) 是一个 Perfetto 链接，如果你赶时间，至少可以看看 Trace Viewer。我比平时花了更多功夫，用 `jax.named_scope` 调用对 trace 做了标注，这样你就能辨认出发生了什么。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

看看这个 profile，试着真正理解每个部分在做什么。我们稍微拆解一下，先从 FFW 块说起：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

这里我们放大到了 FFW 块。你会看到 up-projection（上投影）操作是一个 fusion（matmul，矩阵乘），其输入为 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`，输出为 `bf16[8, 1024, 16384]`。我知道（因为我写了这段代码）这是一次 4 路 DP、2 路 MP 分片矩阵乘的局部视图，所以我们实际上在做的是

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们预期这要花多长时间？** 首先，我们每个数据并行分片（data parallel shard）的批大小为 `8 * 1024 = 8192`，所以我们应该是稳稳地算力受限（compute-bound）。这是在 8 个 TPU v2 核心上，所以我们预期它大约要花 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`，而这几乎正好就是它实际花费的时间（96ms）。太棒了！这意味着我们获得了极好的 FLOPs 利用率！

注意，Google Colab 不再发放 TPU v2-8 切片（slice）了。想要一个真正的 8 核心切片来跟着做，你可以用 [Kaggle](https://www.kaggle.com/)，它仍然免费提供，或者在 GCP 上配置一个 8 核心切片。<d-footnote>如果你只是想用假问题玩玩分片，也可以在 CPU 上伪造 8 个设备：`import jax; jax.config.update("jax_num_cpu_devices", 8)`（需要 jax >= 0.4.27 左右），然后 `print(jax.devices())`。这只对玩具问题有效，不能反映真实性能。</d-footnote>

**那通信呢？** 你会注意到第二个矩阵乘末尾藏着一个小 fusion。如果我们点击它，你会看到

```c
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上就是一个小的 ReduceScatter（这里是 Graph Viewer）；

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

**我们预期这要花多长时间？** 嗯，我们在 TPU v2 4x2 上做一次 ReduceScatter，这只需要在 1.2e11 的双向带宽上跳一跳（one hop）。数组大小为 `2*32*1024*8192`，批次轴被分成 4 份，所以每个分片是 `2*8*1024*8192=128MB`。因此这大约要花 1.1ms。**它实际花了多长时间？** profile 中报告的是 1.13ms。所以我们非常接近屋顶线（roofline）！

**我们也来看看 attention！** 下面是 attention 组件的 profile：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了 Q projection（Q 投影）操作，它使用一个形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256] 的矩阵 $$W_Q$$。我们沿 head（头）维度做 Megatron 分片。试着做同样的练习，算一算它们应该花多长时间。

### Memory Profile（内存分析） {#memory-profile-内存分析}

Memory Profile 让我们可以很方便地以时间为函数查看程序内存（as a function of time）。这对调试 OOM 很有帮助。你可以看到这里大约 7.5GB 分配给了模型参数，还有约 8.5GB 空闲。所以我们还能往内存里塞很多东西。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 习题 {#习题}

**问题 1**：看看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/profile，搞清楚哪里看着可疑，以及这里到底发生了什么。你能准确说出正在进行哪些计算、每个操作在做什么吗？其中涉及的每个矩阵的真实形状是什么、它们是怎样分片的？*试着先只看 profile，不要读代码。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details Click here for the answer. %}

这是两次矩阵乘法，具体来说是：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

你能看到一次 reduce、两个大 fusion，以及一次 all-reduce。第一个大 fusion 是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

这告诉我们，每个分片的形状是 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（沿 8192 维）。通过观察最后那次 AllReduce 中的{% raw %}`replica_groups={{0,16,32,48,64,80,96,112}, ...}`{% endraw %}，我们可以判断我们做的是 8 路模型并行（model parallelism），因此真实形状是 `bf16[8, 8192] * bf16[32768, 8192] -> bf16[8, 32768]`。

{% enddetails %}

**问题 2：** [前面那个 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 实现了一个简单的 mock Transformer。由于 Colab 不再提供 TPU v2-8 切片，你需要在 [Kaggle](https://www.kaggle.com/) 或一个 8 核心 GCP 切片上运行它来跟着做。按照 Colab 里的说明，跑出一个使用 GSPMD 划分的朴素（naive）Transformer 的基准。每个部分花了多长时间？它本该花多长时间？用的是哪种分片？试着修一下分片！*提示：用 `jax.lax.with_sharding_constraint` 来约束行为。用这个修法，你能得到的最好 MFU 是多少？*

作为参考，初始版本大约是 184ms/层，优化后的 profile 是 67ms/层。做完之后，试着盯着 profile 看，看看你能否单凭 profile 回答这些问题：

- 这是什么分片策略？
- 批大小、$$d_\text{model}$$、$$d_\text{ff}$$ 各是多少？
- 花在 attention 上的时间占多大比例，花在 MLP 块上的又占多少？
- 在屋顶线（roofline）下，每个操作本应花的时间占多大比例？

**注：** 自从写下这个问题以来，XLA 编译器变得更强了。初始版本现在大约是 90ms/层，而优化后的 profile 仅好了约 10ms/层（80ms/层）。不过，它仍然值得把玩，看看你能不能做得更好。

<h3 markdown=1 class="next-section">第 9 部分到此结束。关于第 10 部分——深入探究 JAX 并行，请点击[这里](../jax-stuff)。</h3>
