---
layout: distill
title: "All About Rooflines（关于屋顶线分析）"
permalink: /roofline-zh/
description: "当我们在硬件上运行算法时，会受到三件事的约束：计算机做数学运算的速度（OPs/秒）、搬运数据可用的带宽（字节/秒），以及用于存储数据的总内存容量（字节）。这些“屋顶线（roofline）”约束让我们能够对一次给定计算的耗时给出上下界。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "Part 0: Introduction"

next_section_url: ../tpus
next_section_name: "Part 2: TPUs"

bibliography: main.bib

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

  - name: 时间花在了哪里？
  - subsections:
    - name: "可视化屋顶线（图）"
    - name: "矩阵乘法"
    - name: 网络通信屋顶线
  - name: 几道练习题

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

## 时间花在了哪里？ {#时间花在了哪里}

让我们从一个极其简单的问题开始：*为什么一个算法耗时 50 毫秒，而不是 50 秒或 5 毫秒*？模型内部到底发生了什么，才占用了可观的时间？我们又应该预期它耗时多久？

**计算：** 一个深度学习模型本质上是一堆矩阵乘法，每一个都由浮点乘法和加法"操作"（FLOPs）组成。我们的加速器速度决定了这些运算需要多久才能完成：

$$\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}$$

例如，一块 NVIDIA H100 可以进行约 9.89e14 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，一种常用于机器学习的 16 位浮点格式。</d-footnote> FLOPs/s，而一块 TPU v6e 可以进行 9.1e14 FLOPs/s。<d-footnote>H100 和 B200 通常只能达到标称峰值 FLOPs 的约 80-85%，而 TPU 在正常使用中可接近 95%。</d-footnote> 这意味着，在 H100 上做 1e12 次 FLOPs 将耗时（大致）`1e12 / 9.89e14 = 1.01ms`，而在 TPU v6e 上为 `1e12 / 9.1e14 = 1.1ms`。<d-footnote>注意这些芯片定价不同，且本对比未对成本做归一化。</d-footnote>

**芯片内通信：** *在单个加速器内部*，张量需要在加速器内存（HBM）与计算核心之间传输。你会看到这条链路的带宽被称为"HBM 带宽"。<d-footnote>NVIDIA 也称其为 "memory bandwidth（内存带宽）"。</d-footnote> 在 H100 上，[这一数值约为 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，而在 TPU v6e 上 [约为 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间通信：** 当我们将一个模型分布到*多个*加速器上时，张量经常需要在它们之间传输。在我们的硬件上通常有多种可选方案（ICI、DCN 和 PCIe），各自的带宽不同。

无论通信发生在芯片内还是芯片间，我们都以字节/秒来衡量，并用下式估算总通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}$$

通常（但并非总是如此），单个芯片内的计算可以与芯片内及芯片间的通信相重叠。这意味着**我们可以用计算时间与通信时间的最大值作为训练和推理时间的下界**。我们也可以**用两者之和作为上界**。在实践中，我们以最大值为目标进行优化，因为代数运算更简单，而且我们通常可以通过让通信与计算重叠来逼近这一上界。如果我们以最大值为目标来优化，那么上下界最多相差 2 倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。在此之外，我们还通过建模"重叠区域"和开销来提高精度，而这可以借助对你具体模型与目标系统的性能剖析来获得。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设能让通信与计算完美重叠，那么当 $T_\text{math} > T_\text{comms}$ 时，我们的硬件就达到了完全利用率。我们称之为"算力受限（compute-bound）"。当 $T_\text{comms} > T_\text{math}$ 时，我们往往处于"通信受限（communication-bound）"<d-footnote>本书中我们将 "communication-bound"、"comms-bound"、"memory-bound（内存受限）" 和 "bandwidth-bound（带宽受限）" 互换使用。</d-footnote>，至少我们的加速器有一部分 FLOPs/s 在等待数据传来传去时被白白浪费。判断一个操作是算力受限还是通信受限的一种方法是看它的"**算术强度（arithmetic intensity）**"或"**运算强度（operational intensity）**"。

**定义：** 一个算法的算术强度，等于它所执行的总 FLOPs 与它需要通信的字节数之比——通信可以发生在芯片内，也可以发生在芯片间。

$$\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}$$

算术强度衡量的是某个给定操作的"每字节 FLOPs"。在一级近似下，当我们的算术强度较高时，$T_\text{math}$ 相对于 $T_\text{comms}$ 较大，我们通常会用满可用的 FLOPs。当情况相反时，我们会在通信上花费更多时间，从而浪费 FLOPs。这一交叉点发生之处，就是硬件的"峰值算术强度"，即加速器峰值 FLOPs/s 与加速器带宽之比。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{Computation FLOPs}} {\text{Accelerator FLOPs/s}} > \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{Intensity}(\text{Computation}) > \text{Intensity}(\text{Accelerator}) & \\
\end{align*}$$

量 $\text{Intensity}(\text{Accelerator})$ 是加速器达到其峰值 FLOPs/s 时所对应的算术强度。**对于 TPU v5e 的 MXU（矩阵乘法单元，matrix multiply unit），这一数值约为 240 FLOPs/字节**，因为该 TPU 可以执行 `1.97e14` FLOPs/s，并从 HBM 加载 `8.2e11` 字节/秒。<d-footnote>MXU 是 TPU 上的矩阵乘法单元。我们在此特别说明，是因为 TPU 还有其他加速器（如 VPU，即向量处理单元），负责逐元素运算，其峰值 FLOPs/s 不同。</d-footnote> 这意味着，如果一个算法的算术强度低于 240 FLOPs/字节，它就会受字节加载所限，从而我们无法很好地利用硬件。<d-footnote>这仅在算法从 HBM 加载权重并在 MXU 中运行时成立。正如下一节将讨论的，我们有时可将参数存储在 VMEM（向量内存）中，其带宽要高得多。许多算法也在 VPU（向量处理单元）中运行，具有不同的性能特征。</d-footnote> 我们来看这样一个例子：

**<span style="color:#7ab5ff">示例（点积）</span>：** 要计算两个向量在 bfloat16 精度下的点积 `x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从内存加载 $x$ 和 $y$，各自占用 $2 * N = 2N$ 字节，执行 $N$ 次乘法和 $N-1$ 次加法，并将 $2$ 字节写回 HBM。

$$\begin{equation}
\text{Intensity}(\text{dot product}) = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当 $N\rightarrow\infty$ 时。所以点积的算术强度为 $\frac{1}{2}$，换句话说，点积每加载一个字节就执行 0.5 次浮点运算。这意味着我们的算术强度低于硬件的算术强度，从而我们将受通信限制。<d-footnote>上面的 240 这个数在此并不适用，因为如你在下一节将看到的，点积是在 VPU 上而非 MXU 上执行的。TPU v5p 的 VPU 每核约可做 7e12 FLOPs/秒，因此其临界强度约为 3，这意味着我们在此仍在一定程度上受通信限制。无论如何，我们的强度较低且恒定，这意味着在大多数硬件上很难达到算力受限。</d-footnote>

### 可视化屋顶线（图） {#可视化屋顶线-图}

我们可以用**屋顶线图（roofline plot）**来可视化内存与计算之间的权衡，它将算法在我们的硬件上可达到的峰值 FLOPs/s（吞吐量，即图中的 y 轴）相对于该算法的算术强度（x 轴）绘制出来。下面是一个双对数坐标图示例：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b>一个示例屋顶线图，展示了两种具有不同算术强度的算法（Algo 1 与 Algo 2）以及它们在不同带宽（BW1 与 BW2）下的对应理论峰值吞吐量。在红色区域，算法在两种带宽下均受带宽限制，并浪费了硬件峰值 FLOPs/s 的一部分。黄色区域仅在较低带宽（BW1）下受带宽限制。绿色区域在所有带宽下均受算力限制。此处我们使用加速器的峰值 FLOPs/s，提高带宽或改善强度均无法带来收益。"%}

如上图，随着强度增大（从左向右移动），我们最初看到算法性能（以 FLOPs/s 计）线性提升，直到达到硬件的临界算术强度——对于 TPU v5e 而言是 240。任何强度更低的算法都会受带宽（BW）限制，并受峰值内存带宽所限（如红色所示）。任何位于右侧的算法都会用满我们的 FLOPs（如绿色所示）。这里，Algo 1 受通信限制，只使用了硬件总 FLOPs/s 的一小部分。Algo 2 是算力受限的。一般而言，我们可以通过提高其算术强度，或增加可用的内存带宽（从 BW1 移动到 BW2）来改善算法性能。

### 矩阵乘法 {#矩阵乘法}

让我们来看看我们日后最钟爱的算法：矩阵乘法（aka matmul）。我们记 $X * Y \rightarrow Z$，其中 $X$ 的形状为 $\text{bf16}[B, D]$，$Y$ 的形状为 $\text{bf16}[D, F]$，$Z$ 的形状为 $\text{bf16}[B, F]$。要做这个 matmul，我们需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ 次 FLOPs，并将 $2BF$ 字节写回。<d-footnote>严格来说我们执行 $BF \times (2D - 1)$ 次 FLOPs，但这样近似已足够接近。这来自 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>虽然 matmul 的输出严格来说是 float32，但我们通常在复制回 HBM 之前将其降精度为 bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设"批大小" $B$ 相对于 $D$ 和 $F$ 较小，就可以得到一个漂亮的简化。于是我们有

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{Intensity}(\text{matmul}) > \text{Intensity}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对于 Transformer 的 matmul 来说，这是一个合理的假设，因为我们通常具有局部的（每副本的）批大小 $B < 1024$ 个词元（*而非序列*），但 $D$ 和 $F > 8000$。因此我们一般在每副本<d-footnote>我们说"每副本"，是因为如果我们采用某种模型分片以增加 matmul 中使用的芯片数量，我们可用的算力和内存带宽会按相同比例扩展。因此临界批大小对每个独立的模型权重副本成立。</d-footnote>批大小大于 240 个词元时达到算力受限——这是一条非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 要使一个 bfloat16 matmul 在大多数 TPU 上达到算力受限，我们需要每副本的词元批大小大于 240。<d-footnote>注意这_不是_通常意义上的批大小（后者指序列的批大小）。事实证明，大多数屋顶线分析纯粹取决于词元数量，无论它们属于相同或不同的序列。例如，若在 2048 个 GPU 上有 512 个序列、每个序列 4096 个词元的批大小，则总批大小为 `512 * 4096 = 2M` 个词元，本地批大小为 1k 个词元。</d-footnote></p>

这其中有几个值得注意的例外情况，我们将在下面的习题中探讨，尤其是涉及量化时（例如，如果我们对激活值做量化，但仍执行全精度 FLOPs），但它仍是一条值得记住的好规则。对于 GPU，这个数字略高一些（接近 300），但同样的结论大体成立。当我们[将一个大 matmul 分解为更小的 matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，分块大小也很重要。<d-footnote>当我们进行大型矩阵乘法时，需要将其分解为适合 VMEM/SMEM/TMEM（更高带宽的片上内存）的小分块。这导致我们会多次加载数据块，因此"只加载 $O(N^2)$ 字节"不再完全成立。考虑一个 $(m, k) \cdot (k, n)$ 的 matmul，分块大小为 $bm$、$bk$、$bn$。令 $tm = m / bm$，依此类推。则总 FLOPs 为 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总字节数为 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + bm \cdot bn)$。忽略最后一项，我们得到强度为 $bm \cdot bn / (bm + bn)$，与上面类似。</d-footnote> 我们将在[下一节](../tpus)讨论 GPU 和 TPU 的底层细节。

### 网络通信屋顶线 {#网络通信屋顶线}

到目前为止我们讨论过的所有屋顶线都是内存带宽屋顶线，且_全部位于单个芯片内部_。但这不应被视为一条铁律。事实上，本书中我们关心的大多数屋顶线都涉及芯片间的通信：通常是那些矩阵被分片到多个 TPU 上的矩阵乘法。

举一个略显刻意的例子：假设我们想将两个大矩阵 $X\sim \text{bf16}[B, D]$ 和 $Y \sim \text{bf16}[D, F]$ 相乘，它们沿 $D$ 维度被均匀切分到 2 块 TPU/GPU 上。要做这个乘法（正如我们将在[第 3 节](../sharding)看到的），我们可以在每块 TPU 上各自乘一半矩阵（TPU 0 上 `Z0 = X[:, :D // 2] @ Y[:D // 2, :]`，TPU 1 上 `Z1 = X[:, D // 2:] @ Y[D // 2:, :]`），然后将得到的"部分和（partial sums）"复制到另一块 TPU 上并相加。假设我们可以在每方向上以 `4.5e10` 字节/秒的速度复制，且每块芯片上可执行 `1.97e14` FLOPs/s。那么 $T_\text{math}$ 和 $T_\text{comms}$ 各是多少？

$T_\text{math}$ 显然是从前的一半，因为每块 TPU 只做了一半的工作，即<d-footnote>我们忽略了将两个部分和相加所需的 FLOPs（又一次 BF 加法），但这基本可忽略不计。</d-footnote>

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}} = \frac{BDF}{1.97e14}$$

那么 $T_\text{comms}$ 呢？它现在指的是芯片间的通信时间！这不过是发送的总字节数除以网络带宽，即

$$T_\text{comms} = \frac{2BF}{\text{Network Bandwidth}} = \frac{2BF}{4.5e10}$$

因此，当我们满足 $$\text{Intensity}(\text{matmul (2-chips)}) > \text{Intensity}(\text{TPU w.r.t. inter-chip network})$$，也就是等价地当 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$ 或 $D > 8755$ 时，我们达到算力受限（此时是相对于芯片间网络而言）。注意，与之前不同，现在的临界阈值取决于 $D$ 而非 $B$！试着想想这是为什么。这只是一个例子，但我们想强调，这类屋顶线对于判断"何时可以将一个操作并行化到多个 TPU 上"至关重要。

## 几道练习题 {#几道练习题}

**问题 1 [int8 matmul]：** 假设我们想以 int8 精度（每个参数 1 字节）而非 bfloat16（每个参数 2 字节）来做 matmul $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$<d-footnote>此处及全书我们将使用记号 $A \cdot_D B$ 表示乘法在 D 维度上执行收缩（contraction）。这是对 einsum 记号的借用。</d-footnote>，因为 TPU/GPU 可以在更低精度下更快地完成 matmul。

1. 需要从内存加载多少字节？又需要写回多少字节？
2. 总共执行多少次 OPs？
3. 算术强度是多少？
4. $T_\text{math}$ 和 $T_\text{comms}$ 的屋顶线估算是多少？整个操作的运行时间合理的上下界是什么？

假设我们的 HBM 带宽为 `8.2e11` 字节/秒，int8 峰值 OPs/s 为 `3.94e14`（约为 bfloat16 的 2 倍）。

{% details 点击此处查看答案。 %}

1. 由于我们将参数以 int8 存储，每个参数占 1 字节，因此我们从 HBM 加载 $$BD + DF$$ 字节，并写回 $$BF$$ 字节。
2. 这与 bfloat16 下相同，但理论上 int8 的 OPs/s 应该更快。所以仍然是 $2BDF$ 次 OPs。
3. 算术强度为 $$2BDF / (BD + DF + BF)$$。如果我们沿用上面关于 $$B \ll D$$ 和 $$B \ll F$$ 的假设，就得到算术强度 $$2B$$，这意味着我们的规则变为 $B > \text{HBM int8 算术强度} / 2$。代入给定数字，这个 int8 强度为 `3.94e14 / 8.2e11 = 480`，所以规则是 $B > 480 / 2 = 240$。注意这基本没变！
4. $$T_\text{math} = 2BDF / 3.94e14$$，$$T_\text{comms} = (BD + DF + BF) / 8.2e11$$，因此合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 matmul]：** 在实践中，我们经常对权重和激活值采用不同的量化，因此可能以极低精度存储权重，但将激活值（及计算）保持在较高精度。假设我们想将权重以 int8 量化，但将激活值（及计算）保持在 bfloat16。我们在多大的批大小下会达到算力受限？假设 bfloat16 FLOPs/s 为 `1.97e14`。

*提示：这具体指 `bf16[B, D] * int8[D, F] -> bf16[B, F]`，其中 $B$ 是"批大小"。*

{% details 点击此处查看答案。 %}

再次假设 B 较小，我们有 2BDF 次 bfloat16 FLOPs，但只有 DF 个权重（而非 bfloat16 下的 2DF）。这意味着我们在 $$2B > 240$$ 或 $$B > 120$$ 时达到算力受限。这低了很多，意味着如果我们能做 int8 权重量化（这相当容易做到），同时仍执行 bfloat16 FLOPs，我们就能在效率上得到可观的提升（尽管 int8 OPs 会更好）。

{% enddetails %}

**问题 3：** 沿用问题 2 的设置，对 $F = D = 4096$ 和 $F = D = 1024$ 两种情况，画出峰值 FLOPs/s 相对于 $B$ 的屋顶线图。*请使用加载字节数的精确值，而非近似值。*

{% details 点击此处查看答案。 %}

下面就是相关的图：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

注意两个模型最终都达到了硬件峰值 FLOPs/s，但更大的 D/F 更早达到。D=F=1024 几乎使临界批大小翻倍。生成该图的代码如下：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**问题 4：** 如果我们想执行 $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$，即设想每个批元素各有一个不同的矩阵，情况会怎样？这个操作的算术强度是多少？

{% details 点击此处查看答案。 %}

让我们先看看总的 FLOPs 和通信量。

1. 总 FLOPs：FLOPs 基本相同，因为我们在做 $$B$$ 个相互独立的 $$[D] \times [D, F]$$ 乘积，其总工作量与单个 $$[B, D] \times [D, F]$$ matmul 相同（这将在第 4 节进一步讨论）。所以就是 $$2BDF$$。
2. 总通信量：这里的通信量要大得多：$$BD + BDF + BF$$。
3. 因此，我们现在实际的算术强度是 $$2BDF / (BD + BDF + BF)$$。由于 $$BDF$$ 在分母中占主导，这大约为 $$2$$。所以它与批大小无关，而是基本恒定。这很糟糕，因为它意味着无论怎样我们都将基本上始终受通信限制。

{% enddetails %}

**问题 5 [GPU 的内存屋顶线]：** 使用 [NVIDIA 提供的 H100 SXM 规格表](https://www.nvidia.com/en-us/data-center/h100/)，计算一个 bfloat16 矩阵乘法达到算力受限时的批大小。*注意，Tensor Core 的 FLOPs 数值是真实值的两倍，因为它们只有在结构化稀疏性下才能达到。*

{% details 点击此处查看答案。 %}

从规格表可见，标称的 bfloat16 FLOPs 值为 `1.979e15` FLOPs/s，并带星号注明"含稀疏性（with sparsity）"。不使用稀疏性时的真实值是其一半，即 `9.89e14` FLOPs/s。内存带宽为 3.35TB/s，即 `3.35e12` 字节/秒。因此 $B_\text{crit}$ 为 `9.89e14 / 3.35e12 = 295`，与 TPU 颇为接近。

{% enddetails %}

<h3 markdown=1 class="next-section">第一部分到此结束！进入第二部分，了解真实 TPU 如何处理 FLOPs 与通信，[点击此处](../tpus)。</h3>
