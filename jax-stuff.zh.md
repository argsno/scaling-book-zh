---
layout: distill
title: "Programming TPUs in JAX（用 JAX 编写 TPU 程序）"
# permalink: /main/
description: "如何使用 JAX 高效地编写 TPU 程序！本节大部分内容改编自<a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>此处</a>。你可以借助 <a href='https://colab.sandbox.google.com/'>Google Colab</a> 上免费的 TPU，运行本节中的代码示例。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling-zh"
previous_section_name: "第 9 部分. 如何理解 TPU 性能剖析"

next_section_url: "../conclusion-zh"
next_section_name: "第 11 部分. 总结与延伸阅读"

permalink: /jax-stuff-zh/

sitemap: false

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "JAX 中的并行是如何工作的？"
  - subsections:
    - name: "自动分片模式（Auto sharding mode）"
    - name: "显式分片模式（Explicit sharding mode）"
    - name: "通过 shard_map 的手动分片模式（Manual sharding mode via shard_map）"
  - name: "练习题（Worked Problems）"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

## JAX 中的并行是如何工作的？ {#jax-中的并行是如何工作的}

JAX 支持三种多设备编程的"流派"：

1. **编译器，你来掌舵！** 让 XLA 编译器自动划分数组，并决定需要添加哪些通信来支撑给定的程序。这样，你可以用一个在单设备上运行的程序，在不改动任何代码的情况下，自动在数千个设备上运行。
2. **JAX，你来掌舵！** 自动并行很棒，但有时候编译器会做出一些离谱的事情。显式分片让你像往常一样编写单设备代码，但由 JAX（而不是编译器）来处理分片传播。这意味着当你的意图不明确时，JAX 可以请你澄清。
3. **就让我写我想做的，见鬼！** 虽说编译器很好用，但它们有时也会做错事，加入你本不打算进行的通信。有时候我们希望精确地指定自己想要运行哪些通信。

| Mode | View? | Explicit sharding? | Explicit Collectives? |
|:---:|:---:|:---:|:---:|
| Auto | Global | ❌ | ❌ |
| Explicit | Global | ✅ | ❌ |
| Manual | Per-device | ✅ | ✅ |

相应地，JAX 为每种模式都提供了相应的 API：

1. `jax.jit`（使用 `Auto` 网格轴）可以让你接收任意已有的 JAX 函数，并以分片后的输入来调用它。JAX 随后会使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器自动对程序进行并行化。在需要支撑现有运算时，XLA 会为你添加通信（AllGather、ReduceScatter、AllReduce 等）。虽然它并不完美，但通常能在不修改代码的前提下，很好地将你的程序自动扩展到任意数量的芯片上。
2. 使用 `Explicit` 网格轴的 `jax.jit` 与 (1) 类似，但由 JAX（而非 XLA）来处理分片传播。这意味着数组的分片实际上成为了 JAX 类型系统的一部分；当 JAX 检测到有歧义的通信时，它会报错，并交由用户来解决。
3. `jax.shard_map` 是更偏向手动的对应方案。你看到的是设备的局部视图，必须显式地写出任何你想要的通信。手里有一个分片数组，想把整体都放到每个设备上？加一个 `jax.lax.all_gather`。想对所有设备上的数组求和？加一个 `jax.lax.psum`（即 AllReduce）。编程更费力，但几乎不可能做出你不想要的事情。

<h3 id="自动分片模式-auto-sharding-mode">自动分片模式（Auto sharding mode）</h3>

`jax.jit` 在 JAX 中扮演着两种角色。顾名思义，它"即时（just-in-time）"地把一个函数从 Python 编译成字节码（经由 XLA/HLO/LLO），从而运行得更快。但是，如果输入是分片的，或者用户指定了 `in_sharding` 或 `out_sharding`，它还会让 XLA 把计算分布到多个设备上，并按需添加通信。举例来说，下面是你如何用 `jax.jit` 编写一个分片矩阵乘：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

# This creates a fake set of 8 CPU devices so you can run this on a CPU without TPUs.
jax.config.update("jax_num_cpu_devices", 8)

# This creates a 2D 4x2 mesh with axis names X and Y that JAX uses by default.
# We explicitly tell JAX to let the XLA compiler infer sharding along these axes.
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'), axis_types=(Auto, Auto))
jax.set_mesh(mesh)

# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# We can explicitly compile the sharded matmul function here. This adds all the
# necessary comms (e.g. an AllReduce after the matmul).
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这段代码会以任意分片方式自动运行，并把计算分布到我们的各个设备上。**但在硬件层面，实际发生了什么？**

1. 首先，我们创建 In 和 W，让它们在我们的设备上分片<d-footnote>注意我们是怎么做的。这是创建具有特定分片的数组的一种方式（即在创建函数中添加 device 参数）。另一种方式是先用 `jnp.array(....)` 正常创建数组，然后再执行例如 `jax.device_put(..., jax.P('X', 'Y'))`。还有一种是编写一个创建所需数组的函数，并用你想要的 `out_shardings` 对它做 jit 编译。</d-footnote>。W 沿收缩维度被分成 2 份，而 In 被分成 8 份：沿输入维度分 4 份，沿收缩维度分 2 份。这对应于一种分片方式 W[D<sub>Y</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>]，也就是一种模型并行与数据并行的组合。
2. 如果我们是在本地（即单个设备上）运行，`matmul_square` 会简单地对输入平方并执行一次普通的矩阵乘。但因为我们把 `out_shardings` 指定为 `P('X', None)`，输出将沿批次维度分片，而在模型维度上被复制，因此需要一次 AllReduce 才能计算出来。

使用前面章节的记号，它大概会做如下操作：

1. Out[B<sub>X</sub>, F] = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F])

`jax.jit` 会自动为我们加上这一步！我们可以用 `jit_matmul.as_text()` 打印出 HLO，看到如下（大幅精简后的）HLO：

```py
# This fusion is the actual matmul of the sharded inputs and matrix
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# We reduce the partially summed results across devices
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以看到上面的矩阵乘（即 fusion）和 AllReduce。请特别注意其中的形状。`bf16[2, 1024]` 是激活值的局部视图，因为我们的 `batch_size=8` 被分到了 4 个设备上，而 `d_model=2048` 同样被分成 2 份。

**这相当神奇！** 无论我们的程序有多复杂，[Shardy](https://openxla.org/shardy) 和 jit 都会尝试为所有中间激活值找到分片方式，并按需添加通信。话虽如此，Shardy 也有它的缺陷。它可能会犯错。有时你查看一个 profile，会发现有什么地方出了问题。一个巨大的 AllGather 占了 profile 的 80%，而它本不需要。发生这种情况时，我们可以通过用 `jax.lax.with_sharding_constraint` 显式地标注中间张量来纠正编译器。例如，对于两个矩阵乘，我可以强制让中间激活值沿 `y` 维度分片（这并不是个好主意），代码如下：

```py
import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X', 'Y'), (Auto, Auto))
jax.set_mesh(mesh)

def matmul(x, W_in, W_out):
  hidden = jnp.einsum('bd,df->bf', x, W_in)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, W_out)
```

在自动划分的世界里，通过 `jax.lax.with_sharding_constraint` 来控制中间分片的方式，大概占了 JAX 并行编程的 60%。但"哄编译器"显然不是一种令人愉快的编程模型。你可能会给每个中间变量都加上标注，却仍然不知道能否得到正确的结果。那么，如果由 JAX 自己来处理和控制分片传播呢？

<h3 id="显式分片模式-explicit-sharding-mode">显式分片模式（Explicit sharding mode）</h3>

显式分片（或称"类型中的分片"）看起来与自动分片非常相似，但分片传播发生在 JAX 层面！每个 JAX 运算都有一条分片规则，它接收该运算各参数的分片方式，并得出结果的分片方式。你可以通过 `jax.typeof` 查看结果的分片：

```py
import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# Running on a TPU v5e 2x2. This assigns names to the two physical axes of the hardware.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))

# This tells JAX to use this mesh for all operations, so you can just specify the PartitionSpec P.
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

如你所见，JAX 把分片从输入（`x`）传播到了输出（`out`），并且可以通过 `jax.typeof` 在 trace 阶段查看。对于大多数运算，这些规则简单且显而易见，因为只有一个合理的选择（例如逐元素运算会保留相同的分片）。但对于某些运算，结果该如何分片是模糊的，此时 JAX 会抛出一个 trace 阶段的错误，并要求程序员显式地提供一个 `out_sharding` 参数（例如 jnp.einsum、jnp.reshape 等）。我们来看另一个出现冲突的例子：

```py
# We create a matrix W and input activations In sharded across our devices.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # This will error
```

这段代码会报如下错误：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

这很棒，因为 einsum 的输出该如何分片是模糊的。输出分片可以是：
* P('X', 'Y')，这会引入一个 ReduceScatter；或者
* P('X', None)，这会引入一个 AllReduce

与 Auto 模式不同，显式模式在检测到有歧义的通信时会报错，并要求用户来解决。所以在这里你可以这样做：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

Auto 模式和 Explicit 模式可以通过 `jax.sharding.auto_axes` 与 `jax.sharding.explicit_axes` 这两个 API 组合使用。想了解更多，这篇[文档很值得一读](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)。

<h3 id="通过-shard-map-的手动分片模式-manual-sharding-mode-via-shard-map">通过 shard_map 的手动分片模式（Manual sharding mode via shard_map）</h3>

虽然 Shardy 是"编译器来掌舵"的模式，但 jax 的 [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 把一切都交到你手中。你像在 jax.jit 中一样指定输入的分片方式，但之后你必须显式地写出所有通信。与 `jax.jit` 给你一个全局的跨设备程序视图不同，`shard_map` 给你的是局部的逐设备视图。

来看一个例子。试着想想这个函数做了什么：<d-footnote>如果你想在 colab 中通过模拟一个网格自己试玩，可以用下面这个 cell 来做到：`import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp

Explicit = jax.sharding.AxisType.Explicit

mesh = jax.make_mesh((2, 4), ('x', 'y'), (Explicit, Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# This function will operate on 1/8th of the array.
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**这段代码做了什么？** `slice_and_average` 在每个 TPU 上用数组的 1/8 运行，从中切出前 4 个元素，并在整个网格上对它们求平均。这意味着我们实际上在做 `mean(x[:4], x[64:68], x[128:132], …)`。这相当酷，因为否则在 JAX 中很难表达这样的操作。

**为什么不用 jax.jit 而要用它？** 如果我们用了 `jax.jit`，`slice_and_average` 会看到一个数组的全局视图（完整的 `[512,]` 数组）。我们就得切出这块不均匀的切片，然后做一次平均，而 XLA 必须正确地理解它。XLA 可能会加入错误的通信，或者感到困惑。在这里我们看到了局部视图，并且只写出我们真正需要的通信。

**示例 [集合矩阵乘法]：** 举一个更贴近实际的例子，假设我们要实现模型并行，其中激活值最初是按模型分片的，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。朴素的做法是，先对 A 做一次 AllGather，再做一个本地矩阵乘：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

遗憾的是，这并不好，因为它无法让通信与计算重叠。使用"集合矩阵乘法"可以让二者重叠，正如 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) 所描述。该算法基本如下：

* 对于每一个 Y 分片，用 A 的本地块与 W 的本地块做一次矩阵乘，产生一个形状为 `[B / X, F / Y]` 的结果。与此同时，对 A 做置换，使你能拿到下一个本地块，做矩阵乘，并把结果累加。

我们可以用 `jax.shard_map` 相当轻松地实现它：

```py
import functools

import jax
import jax.numpy as jnp
import numpy as np

Explicit = jax.sharding.AxisType.Explicit

# This is intended to run on a TPU v5e-8 runtime. If you can't get this,
# try setting jax.config.update('jax_num_cpu_devices', 8).
#
mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'), axis_types=(Explicit, Explicit))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, jax.P('X', 'Y'))
W = jax.device_put(W, jax.P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=jax.P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.axis_size('Y')  # axis_size = 4 for this example
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # Matmul for a chunk
    update = lhs @ rhs_chunk
    # Circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum = jax.lax.pcast(accum, ('X', 'Y'), to='varying')
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # Compute the last chunk after the final permute to leave lhs in the state we found it
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(jax.shard_map(
  collective_matmul_allgather_lhs_contracting,
  in_specs=(jax.P('X', 'Y'), jax.P(None, 'Y')), out_specs=jax.P('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

这相当巧妙！我们可以对它做基准测试，发现它也快了很多！[这是](https://imgur.com/a/e9I6SrM) 默认 jit 矩阵乘的 profile，它在开头有一个很大的阻塞式 AllGather，耗时 311us：

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

而[这是](https://imgur.com/a/21iy0Sv) 上面那个版本，耗时 244us。你可以看到 profile 里没有 AllGather。全都是有效的工作！我们的 FLOPs 利用率也高了很多。

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

同样值得注意的是，在收缩维度上不做分片时，矩阵乘耗时为 [224us](https://imgur.com/a/i3gNKfq)，所以我们已经非常接近未分片的基线了。这是一个很好的例子，说明你可能最终会做哪类性能工程来提升 TPU 利用率。想看更多 `shard_map` 示例，[这篇笔记很不错](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side)。

现在来看几个有用的练习题，试着用 `jax.jit` 或 `shard_map` 实现它们！

## 练习题（Worked Problems） {#练习题-worked-problems}

这里有一些随机的 JAX 相关题目。我以后再补充更多。所有这些题目你都需要一定数量的 TPU。Colab 已经不再提供 TPU v2-8 切片了，所以请使用 [Kaggle](https://www.kaggle.com/)（它仍然免费提供）或者一个 8 核的 GCP 切片。<d-footnote>如果你只是想在假想的题目上模拟一个网格，也可以用 `import jax; jax.config.update('jax_num_cpu_devices', 8)`（需要 jax >= 0.4.27 左右）在 CPU 上伪造出 8 个设备，不过这并不能反映真实的性能。</d-footnote> 从现在起，我们假设你有 N 个可用的设备。

**问题 1：** 令 **A** 为一个形状为 float32[S<sub>X</sub>, D<sub>Y</sub>]、满足 `X * Y = N` 的激活值数组。完成以下任务：

1. 在 JAX 中写一个函数，计算每个 `(X, Y)` 分片内部的平均值，即返回一个大小为 [X, Y] 的数组，其中 `arr[i, j]` 是分片 `(i, j)` 上的平均值。分别用 `jax.jit` 和 `shard_map` 实现。对二者做 profile，看看各自耗时多少。有加入任何通信吗？*提示：本不应该有，但有时 XLA 还是会加上。*

2. 在 JAX 中写一个函数，对每个分片**沿 X 方向**做某个 shift，返回 `roll(x, shift, axis=0) - x`。我还没自虐到要你在 jax.jit 里做这个，所以只用 `shard_map` 实现即可。

{% details 点击此处查看答案。 %}

第 1 部分：这里是第 1 部分的一种解法。注意为了 `jax.jit` 的解法，我们不得不做一些相当复杂的 reshape。

```py
import numpy as np

import jax
import jax.numpy as jnp

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

第 2 部分：这里是第 2 部分的一种类似解法。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

Auto = jax.sharding.AxisType.Auto

mesh = jax.make_mesh((4, 2), ('X','Y'), (Auto, Auto))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**问题 2：** 这里我们一起来做一个基础的"混合专家（MoE）"模型。令 **W**：float32[E<sub>X</sub>, D, F] 为一组 E 个"专家"矩阵。令 **A**：float32[S<sub>X</sub>, D]（我们的激活值），令 **B**：int32[S<sub>X</sub>] 为一组"路由分配"，其中 B[i] 是一个落在 `[0, E)` 范围内的整数，告诉我们想要用哪个矩阵来处理该激活值。我们想在 JAX 中写一个函数，返回 `Out[i] = A[i] @ W[B[i]]`。

1. 我们一开始先完全忽略分片。把所有这些张量都做小到能放进一个设备。写一个这个函数的本地实现。*确保你不要实体化出一个形状为 `[S, D, F]` 的数组！提示：试着把词元排序到一个形状为 `[E, S, D]` 的新缓冲区里，并注意掩码（为什么我们需要第二个维度大小为 S？）。*

2. 如果你直接对上面的方法做 `jax.jit`，会发生一些事情。对它做 profile，看看它决定做哪些通信。耗时多少？

3. 你会发现上面的做法有一个问题：它很可能会在本地把完整的激活值集合 **A** 收集起来，即 AllGather<sub>X</sub>([S<sub>X</sub>, D])。这不仅在通信上代价高昂，而且如果我们无法把完整激活值集合放进本地，在内存上也会极其昂贵。请用 `shard_map` 和显式通信来实现上面的功能。

      1. 作为第一遍尝试，最简单的方式可能是用 `jax.lax.all_gather` 并按第 1 步那样重排。

      2. 作为第二遍尝试，试着避免实体化任何大小为 `[E, S, D]` 的数组，也就是尝试在 `jax.lax.while_loop` 内部使用 `jax.lax.all_to_all`，以不规则（ragged）的方式执行计算。这样，你就能避免实体化完整的激活值，并避免在填充（padding）上浪费算力。这比你最初的实现快了多少？

4. 大多数 MoE 会路由到多个（k 个）专家，然后对结果求平均。重构上面的代码来实现这一点。在这种情况下，令 **B**：int32[S<sub>X</sub>, k]，表示要路由到的 k 个专家。

{% details 点击此处查看（部分）答案。 %}

1/2. 对于第 (1) 部分，你有很多选择。下面是一种借助掩码遍历各个专家的做法。

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - this expert's transform of ALL tokens
        output = output + expert_result * mask  # Only keep results for assigned tokens
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

你也可以用 `jax.lax.ragged_dot`，它的功能类似，但更高效。

3. 这里我只给出伪代码的框架（如果你有干净的解法，欢迎补充）：

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # sort x according to assignments
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
     i += chunk_size
  return concat(out)
```

基本思路是：遍历数组的各个块，对它们排序并做一次 all_to_all，然后再做本地 FLOPs。

{% enddetails %}

**问题 3：** 上面的集合矩阵乘例子，其实和真实的 LLM 高度相关。我们来改造一下这个例子，实现完整的 Transformer 堆栈。

1. 作为练习，我们先来实现一个 AllReduce 集合矩阵乘，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。注意输出并不是复制的。朴素的算法上文已经讨论过，基本上就是本地矩阵乘之后再接一个 AllReduce。试着做一个通信重叠的"集合"版本。*提示：沿输出维度分块（tile），并且可以用 `jax.lax.psum`（即 AllReduce）。* *注意：由于 XLA 处理这件事的方式，它实际上可能并不会比基线更快。*

2. 与上面的 AllReduce 集合矩阵乘互补的是 ReduceScatter 集合矩阵乘，如 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。它出现在 Transformer 的降维投影矩阵中。在 JAX 中实现一个集合的、重叠的版本。注意只传递你所需的最小数据量。*提示：在累加结果的同时对结果做置换。*

3. 把这两个组合起来，形成一个端到端的 Transformer 块，执行 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]，并使用重叠的通信。<d-footnote>和之前一样，我们不能先做 $W_{in} \cdot W_{out}$，因为这里省略了一个非线性项。</d-footnote> 这比 `jax.jit` 的实现快了多少？

**问题 4：** 上面实现的所有集合矩阵乘都是单向的：它们只在一个方向上做置换。重写集合 AllReduce 矩阵乘和集合 ReduceScatter 矩阵乘，让它们使用双向通信。这能快多少？

### 第 10 部分到此结束。基本就这些了！想看最终结论和延伸阅读，请点击[这里](../conclusion)。 {#第-10-部分到此结束-基本就这些了-想看最终结论和延伸阅读-请点击-这里-conclusion}
