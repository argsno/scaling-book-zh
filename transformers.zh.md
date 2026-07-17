---
layout: distill
title: "All the Transformer Math You Need to Know（你需要了解的 Transformer 数学）"
# permalink: /main/
description: "这里我们会对 Transformer 架构做一个快速回顾，具体聚焦于如何计算 FLOPs、字节数，以及其他我们关心的量。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding-zh"
previous_section_name: "第 3 部分. 分片矩阵与分片矩阵乘法"

next_section_url: "../training-zh"
next_section_name: "第 5 部分. 如何对 Transformer 进行训练并行化"

permalink: /transformers-zh/

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
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "数点（Counting Dots）"
  - subsections:
    - name: "前向与反向 FLOPs"
  - name: "Transformer 计算量核算"
  - name: "全局 FLOPs 与参数量计算"
  - name: "其他数学"
  - subsections:
    - name: "稀疏性与混合专家（Mixture-of-Experts）"
    - name: "梯度检查点"
    - name: "键值（KV）缓存（Key-Value caching）"
  - name: "本节要点"
  - name: "几道练习题"
  - name: "附录"
  - subsections:
    - name: "附录 A：Flash Attention 是如何工作的？"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

## 数点（Counting Dots） {#数点-counting-dots}

让我们从形状如下的向量 $$x$$、$$y$$ 和矩阵 $$A$$、$$B$$ 开始：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{array}  & \textrm{shape} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end{array}
$$

- $$x \cdot y$$ 的点积需要 $$P$$ 次 _adds_ 和 _multiplies_，总共 $$2P$$ 次浮点运算。
- 矩阵-向量乘积 $$Ax$$ 沿 $$A$$ 的各行做 $$N$$ 次点积，共 $$2NP$$ 次 FLOPs。
- 矩阵-矩阵乘积 $$AB$$ 对 $$B$$ 的 $$M$$ 个列分别做一次矩阵-向量乘积，共 $$2NPM$$ 次 FLOPs。
- 一般地，如果我们有两个高维数组 $$C$$ 和 $$D$$，其中某些维度是 <span style="color:red">CONTRACTING</span>，某些维度是 <span style="color:blue">BATCHING</span>（例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$），那么该收缩的 FLOPs 代价是 $$C$$ 和 $$D$$ 所有维度的乘积的两倍，但批处理维度和收缩维度只计一次（例如 $$2\blue{GH}IJMN\red{KL}$$）。注意，一个维度只有当它同时出现在两个乘数中时才是批处理的。（另请注意，如果没有收缩维度、仅仅是按元素相乘，则系数 2 不适用。）<d-footnote><b>Contracting</b>（收缩）维度是在运算中被求和的轴（它们同时出现在两个输入中，但不在输出中），例如矩阵乘法中的内部维度。<b>Batching</b>（批处理）维度是同时出现在两个输入中、并原样传递到输出的共享轴；它们为相互独立的子问题建立索引，在 FLOPs 计数中不会被相乘。用 einsum 的术语来说：同时出现在两个输入和输出中的标签即为 batching；同时出现在两个输入中、但在输出中缺失的标签即为 contracting。</d-footnote>

$$
\begin{array}{ccc}
\textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end{array}
$$

请注意，对于矩阵-矩阵乘法，*计算量*随 $$O(N^3)$$ 三次方增长，而数据搬运量仅随 $$O(N^2)$$ 二次方增长——这意味着随着我们将 matmul 的规模扩大，达到算力饱和上限变得*更容易*。这是极为反常的，也在很大程度上解释了为什么我们采用以矩阵乘法为主的架构——它们天然适合被扩展！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向与反向 FLOPs {#前向与反向-flops}

在训练过程中，我们并不特别关心某次矩阵乘法的结果；我们真正关心的是它的导数。事实证明，计算该导数的代价大约是仅执行矩阵乘法本身的 3 倍。

假设 **B** 只是一个更大网络中的一个矩阵，而 **A** 是我们的输入激活值，**C = A B**，那么损失 **L** 对 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

计算它需要 $2NPM$ 次 FLOPs（因为它在 $N$ 维度上进行收缩）。类似地，损失对 **A** 的导数为

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

同样需要 $2NPM$ 次 FLOPs，因为 **dL/dC** 是一个大小为 $$[N, M]$$ 的矩阵。虽然这个值并不是对某个参数的导数，但它被用来计算网络前几层的导数（例如，正如上面用 dL/dC 来计算 dL/dB 一样）。

把这些加总，我们看到**在训练时，我们总共需要 6NPM 次 FLOPs**，而推理时只需 2NPM：前向传播 2NPM，反向传播 4NPM。由于 PM 是矩阵中的参数数量，这就是著名的训练时 Transformer FLOPs 近似公式 $$6 * \text{num parameters} * \text{num tokens}$$ 的最简形式：每个词元需要 $$6 * \text{num parameters}$$ 次 FLOPs。我们会在下文给出更严谨的推导。

## Transformer 计算量核算 {#transformer-计算量核算}

Transformer 就是未来。好吧，至少它们已经是现在了。也许几年前，它们还只是众多架构之一。但时至今日，值得去了解该架构几乎每一个细节。我们不会重新介绍该架构，不过 [这篇博客](https://jalammar.github.io/illustrated-transformer/) 和 [原始的 Transformer 论文](https://arxiv.org/abs/1706.03762) 或许是不错的参考资料。

下面是一个 Transformer 解码器架构的基本示意图：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>Figure:</b> 该图展示了一个标准 Transformer 的一层，数据流自上而下。我们用单字母约定来描述 Transformer 中数组的形状与布局，再次以红色表示收缩维度，以蓝色表示批处理维度。在给定运算中，输入形状标注在左上方，参数形状标注在右上方，结果形状标注在下方，例如 BTD 是门控 einsum 的输入形状，DF 是权重形状。"%}

**注 [门控 einsum]**：上面的图使用了"[门控 einsum](https://arxiv.org/abs/2002.05202)"<d-cite key="glu"></d-cite>，我们将上投影矩阵拆分为两个矩阵（即上图中的 $W_\text{In1}$ 和 $W_\text{In2}$），它们的输出以逐元素相乘的方式作为一种"门控函数"。并非所有大语言模型都采用这种方式，所以你有时会看到一个单独的 $W_\text{In}$ 矩阵，此时 MLP 的总参数量为 2DF 而非 3DF。通常在这种情况下，D 和 F 会被相应放大，以使得参数量与三矩阵情形保持一致。尽管如此，LLaMA、DeepSeek 以及许多其他模型都采用了某种形式的门控 einsum。

**注 2 [MHA 注意力]**：在自注意力中，T 和 S 是相同的，但在交叉注意力中它们可能不同。在原始的 Multi-Head Attention（MHA）中，N 和 K 相同；而在 [Multi-Query Attention](https://arxiv.org/abs/1911.02150)（MQA）<d-cite key="mqa"></d-cite> 中 K=1，在 [Grouped MQA](https://arxiv.org/abs/2305.13245)（GMQA）<d-cite key="gmqa"></d-cite> 中，K 只需能整除 N。

**注 3 [前置归一化 vs. 后置归一化]**：上图展示的是所谓的"前置归一化（pre-norm）"架构，其中归一化发生在残差连接之前，通常写作 `x + attn(norm(x))`。如今像 LLaMA-3 这样的模型就采用这种方式。原始的 Transformer 论文则采用了"后置归一化（post-norm）"架构，其中层归一化发生在残差连接之后，即 `norm(x + attn(x))`。

## 全局 FLOPs 与参数量计算 {#全局-flops-与参数量计算}

让我们来计算 Transformer 每一层的 FLOPs（这样就不必处处都带着 **L** 这个因子）。请注意，下文的训练 FLOPs 几乎总是推理 FLOPs 的 3 倍，因此你可以用任意总量除以 3 得到仅前向传播的代价。

### MLP {#mlp}

Transformer 的 MLP 通常由两个逐元素组合的输入 matmul 和一个输出 matmul 组成：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意力 {#注意力}

对于 **Q** 与 **KV** 头数不同的通用分组查询注意力情形，我们假设 **Q**、**K**、**V** 投影具有相同的头维度 H，并估算 **QKVO** 各项 matmul 的代价：

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

点积注意力运算更为微妙，它实质上是在 $$B$$、$$K$$ 维度上分块的 $$TH \cdot HS$$ matmul、一个 softmax，以及同样在 $$B$$、$$K$$ 维度上分块的 $$TS \cdot SH$$ matmul。我们用蓝色标出分块的维度：

$$
\begin{array}{cc}
\textrm{operation} & \textrm{train FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**注 [因果掩码]**：近期的 Transformer 大多使用因果掩码，而非完全的双向注意力。在这种情况下，点积运算的有效 FLOPs 减半。要在实践中实现这一降低，我们需要借助注意力 kernel，而非朴素的 einsum。

### 其他运算 {#其他运算}

Transformer 中还有若干其他运算。层归一化的代价相对较低，在一阶代价估计中可以忽略。请注意，每层通常有两个层归一化（一个在注意力之前，一个在 MLP 之前）。此外还有最后那个庞大的（尽管不是逐层的）反嵌入矩阵乘法。

$$
\begin{array}{ccc}
\textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\
\hline \\
2 \times \textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{2D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOPs 经验法则 {#transformer-flops-经验法则}

如果我们忽略点积注意力的代价（对于较短上下文的训练这是合理的），那么所有层的总 FLOPs 为

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

由此得到了一个著名的经验法则，用于估算稠密 Transformer 的 FLOP 数，并忽略注意力 FLOPs。（反嵌入是另一个简单的 matmul，具有 $6BTDV$ 次 FLOPs 和 $DV$ 个参数，并遵循同样的经验法则。）

### 注意力随上下文长度的代价占比 {#注意力随上下文长度的代价占比}

如果我们确实考虑上面的点积注意力，并假设 $$F=4D$$、$$D=NH$$（通常如此）且 $$N=K$$，那么点积注意力 FLOPs 与所有 matmul FLOPs（含注意力投影）的比值为：

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

结论是**只有当 T>8D 时，点积注意力 FLOPs 才会在训练中占据主导**。当 D ≈ 8k 时，这对应约 64K 个词元。这有一定道理，因为它意味着随着 MLP 规模增大，注意力 FLOPs 的重要性下降。对于大型模型，注意力的二次方代价实际上并不是长上下文训练的巨大障碍。然而，对于较小的模型，例如 D=4608 的 Gemma-27B，注意力大约在 37k 的序列长度时变得主导。<d-footnote>请注意，一些现代的开源（OSS）模型引入了局部注意力或其他优化手段，降低了注意力的代价，从而改变了这条屋顶线（roofline）。</d-footnote> Flash Attention 也有助于缓解长上下文的代价，我们将在 [附录 A](#附录-a-flash-attention-是如何工作的) 中简要讨论。

## 其他数学 {#其他数学}

### 稀疏性与混合专家（Mixture-of-Experts） {#稀疏性与混合专家-mixture-of-experts}

我们不应略过对混合专家（Mixture of Experts，MoE）模型的简要讨论<d-cite key="moe"></d-cite>，它们用一组可动态路由的独立 MLP 替换了标准 Transformer 中单一的稠密 MLP 块。粗略地说，**MoE 只是一个每层有 E 个 MLP 块的普通稠密模型**，而非仅一个。每个词元会激活其中 $k$ 个专家，通常 $k \ll E$。比值 $E / k$ 被称为稀疏性（sparsity），通常介于 8 到 64 之间（例如 [DeepSeek v3](https://arxiv.org/pdf/2412.19437) 实际上有 $k=8$、$E=256$）。与稠密版本相比，这将参数量增加了 $O(E)$，同时将每个词元被激活的参数总数乘以 $k$。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>Figure:</b> 一个包含 $n$ 个专家的 MoE 层示例。门控专家将每个词元路由到其中 $k$ 个，这 $k$ 个 MLP 的输出被求和。我们的参数量是每个专家大小的 $n$ 倍，但每个词元仅使用其中的 $k$ 个。<a href='https://deepgram.com/learn/mixture-of-experts-ml-model-guide'>来源</a>。"%}

与稠密模型相比，MoE 引入了新的通信，主要是两个 AllToAll（分别位于 MoE 块之前和之后），它们将词元路由到正确的专家，并将其送回原先所在设备。<d-footnote>严格来说，这只有在我们沿与专家相同的轴进行数据或序列分片时才会发生。</d-footnote> 不过，正如我们在上一节所见，每个 AllToAll 的代价仅相当于沿单一轴的同类 AllGather 的 1/4（对于双向环而言）。

### 梯度检查点 {#梯度检查点}

反向传播作为一种算法，是以计算换取内存。前向传播不再需要 $$O(n_\text{layers}^2)$$ 次 FLOPs 的反向传播，而是**需要 $$O(n_\text{layers})$$ 的内存**，并保存前向传播中产生的所有中间激活值。虽然这比二次方的计算要好，但在内存方面代价极其高昂：对于一个 $$B * T=4M$$（每批次共 4M 个词元）、L=64、D=8192 的模型，如果要避免所有不必要的反向传播计算，就必须在 bfloat16 中保存大约 $$2 * 20 * B * T * D * L = 84TB$$ 的激活值。其中 20 大致来自对上图 Transformer 中每个中间节点的计数，例如：

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

因此，为了避免重新计算，我们需要从前向传播中保存 $$g(x)$$ 和 $$\exp(g(x))$$。为了避免保存如此大的内存，我们可以选择只保存部分中间激活值。以下是我们使用的几种策略。

* **整块重计算（Block remat）**：只保存每层的输入。这是我们采用的最激进的方法，每层仅保存 1 个检查点，意味着在上例中我们只需保存 4.2TB。这会迫使我们在反向传播中重复几乎全部的前向传播 FLOPs，也就是说，我们的 FLOPs 会从 $$6 \cdot \text{num params} \cdot \text{num tokens}$$ 增加到大约 $$8 \cdot \text{num params} \cdot \text{num tokens}$$。
* **仅保存大 matmul（Big matmuls only）**：另一种简单的策略是只保存大型 matmul 的输出。这样我们就能在反向传播中避免重新计算任何大型 matmul，但仍需重新计算其他的激活函数以及注意力的部分内容。这将上面每层 20 的计数降低到接近每层 7。

这绝非全部。在使用 JAX 时，这些通常由 `jax.remat`/`jax.checkpoint` 控制（你可以[在此](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html) 阅读更多内容）。

### 键值（KV）缓存（Key-Value caching） {#键值-kv-缓存-key-value-caching}

正如我们将在 [第 7 节](../inference) 中看到的，大语言模型推理有两个关键部分：预填充（prefill）和生成（generation）。

* **预填充（Prefill）** 处理长提示词，并将其注意力激活值保存在键值缓存（KV Cache）中，供生成阶段使用，具体指的是注意力块中的键值投影。
* **生成（Generation）** 将若干这样的 KV cache 批量组合在一起，并从每一个中采样词元。

因此，每个 KV cache 实际上是一个大小为 $[2, S, L, K, H]$ 的数组，其中 2 对应键和值。这个数组相当大！在 int8 下，键值缓存的总大小为 $2SLKH$。对于一个上下文长度为 8k、64 层、且 $KH = NH = D = 8192$ 的中等规模模型，其大小为 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$。这也就解释了为什么我们希望使用 $K \ll N$ 的 GMQA。

## 本节要点 {#本节要点}

* Transformer 的总参数量和 FLOPs 相当容易计算，这里在假设 MHA（批大小 B、词表大小 V、长度为 T 的序列、D=d<sub>model</sub>、F=d<sub>ff</sub>）的前提下进行了汇总：


<!-- $$
\begin{array}{ccc}
\textrm{Component} & \textrm{Params per layer} & \textrm{Training FLOPs per layer} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{ (total, not per-layer)} & 12BTDV \\[10pt]
\end{array}
$$ -->

| 组件     | 每层参数量          | 每层训练 FLOPs             |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **Attention** | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **Other**     | 2D                        | BTD                           |
| **Vocab**     | DV（总计，非逐层） | 12BTDV                        |

* MLP 块的参数量在总参数量中占主导，并且只要序列长度 $T < 8D$，MLP 块在 FLOPs 预算中也占主导。
* 对于合理的上下文长度，训练时的总 FLOPs 预算可以很好地用 $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$ 来近似。
* 在推理时，我们的 KV cache 每个大约占用 $$2 \cdot S \cdot L \cdot K \cdot H$$（其中 K 为 KV 头数），不过架构上的修改往往能够减小这一数值。

## 几道练习题 {#几道练习题}

**问题 1：** 一个 $D=4096$、$F=4 \cdot D$、$V=32,000$、$L=64$ 的模型有多少参数？其中注意力参数占多大比例？我们每个词元的 KV cache 有多大？*你可以假设 $N\cdot H=D$，并使用 int8 的 KV 多头注意力。*

{% details Click here for the answer. %}

1. 总参数量大约为 $$L \cdot (3DF + 4DNH + 2D) + 2DV$$（计入每层的两次层归一化）。对于给定的数值，这是 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 2 \cdot 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，即 16B 参数。
2. 注意力参数与总参数之比一般为 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。这意味着大约 1/4 的参数用于注意力。
3. 每个词元，我们的 KV cache 在 int8 下为 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$，即 `512 KiB / token`。

{% enddetails %}

**问题 2：** 在 `{'X': 4, 'Y': 8, 'Z': 4}` 上执行 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 需要多少总 FLOPs？每个 TPU 执行多少 FLOPs？

{% details Click here for the answer. %}

该运算的总"理论"FLOPs 为 $$2 \cdot B \cdot D \cdot F$$。然而，由于计算并未沿 Z 维度分片，我们实际上多做了 Z 倍的 FLOPs，即总 FLOPs 为 $$2 \cdot B \cdot D \cdot F \cdot Z$$。由于计算沿其他维度分片，每个设备的总量大约为 $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$。

{% enddetails %}

**问题 3：** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少 FLOPs？

{% details Click here for the answer. %}

根据上述规则，I 和 J 是收缩维度，K、L、M、N、O 是非收缩维度。我们没有"批处理维度"，因此这仅仅是 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$，即所有轴的乘积。如果存在一个共享轴，它只会被计数一次。

{% enddetails %}

**问题 4：** 自注意力的算术强度是多少（忽略 Q/K/V/O 投影）？*请将答案表示为 Q 和 KV 长度 T 与 S 的函数。* 在什么上下文长度下注意力会受 FLOPs 限制（算力受限）？给定我们 TPU 的 HBM 带宽，请画出随着上下文长度增长，注意力相对于 FFW 块的有效相对代价。

{% details Click here for the answer. %}

自注意力需要先加载 $$Q$$、$$K$$、$$V$$ 激活值，然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$，再将结果写回 HBM。这会用 Flash Attention 来完成，因此这套计算存在一些注意事项，但基本上在 bf16 下自注意力执行的是

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

因此我们的总字节数为 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总 FLOPs 为 $$4BTSNH + O(BTSN)$$，算术强度为 $$4BTSKGH / (4BHK * (TG + S))$$。

所以基本上，在预填充阶段我们有 $$S=T$$，因此算术强度为 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成阶段，$$T=1$$，因此假设 $$S$$ 非常大，则有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$。取决于你对问题的理解，在预填充或训练时，如果不进行序列分片，自注意力在 S=240 处达到算力受限。在生成阶段，我们永远不会达到算力受限，因为 $$G$$ 很小。尽管如此，可以看出增大 $$G$$ 会使我们更接近算力受限。

{% enddetails %}

**问题 5：** 在多大的序列长度下，自注意力的 FLOPs 与 QKVO 投影的 FLOPs 相等？

{% details Click here for the answer. %}

这纯粹是问 $$24BTDNH = 12BT^2NH$$ 何时成立。化简得到 $$2D = T$$，例如对于 $$D=4096$$，这就是 $$8192$$。这说明对于大多数合理的上下文长度，matmul 的 FLOPs 更大。

{% enddetails %}

**问题 6：** 假设我们在前向传播中只保存 Transformer 层中 7 个主要 matmul 的输出（Q、K、V、O \+ 三个 FFW 矩阵）。我们在反向传播中需要额外"重计算（rematerialize）"多少 FLOPs？

{% details Click here for the answer. %}

仅保存七个 matmul 的输出（Q、K、V、O、W₁、W₂、W₃）意味着反向传播必须重新计算两个注意力 matmul

$$QK^{\top} \quad\text{and}\quad \operatorname{softmax}(QK^{\top})V$$

in order to obtain $\frac{\partial L}{\partial W_\text{O}}$.

每个都是在 $B$ 个序列和 $N$ 个头上分块的 $T \times T$ matmul，因此额外的 FLOPs 为

$$4 \; B \, T^{2} \, N \, H.$$

其他需要重新计算的运算包括：
1. 计算 $\frac{\partial L}{\partial W_\text{In1}}$ 和 $\frac{\partial L}{\partial W_\text{In2}}$ 需要 $O(BTD)$。
2. 计算 $\frac{\partial L}{\partial W_\text{Out}}$ 需要 $O(BTF)$。

{% enddetails %}

**问题 7：** DeepSeek v3 宣称在 14.8T 个词元上训练了 2.79M 个 H800 小时（[来源](https://arxiv.org/pdf/2412.19437v1)）。已知其有 37B 个被激活的参数，他们大致达到了怎样的硬件利用率？*提示：注意他们使用的是不带结构化稀疏性的 FP8 FLOPs。*

{% details Click here for the answer. %}

根据[这里](https://lenovopress.lenovo.com/lp1814.pdf) 的规格表，我们查到 FP8 性能在带稀疏性时为 3,026 TFLOPs/s，不带稀疏性时通常为其一半（`1.513e15` FLOPs/s）。2.79M 个 H800 小时意味着总 FLOPs 为 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25`。给定 37B 的激活参数量，这次训练本应使用大约 `6 * 37e9 * 14.8e12 = 3.3e24` 次 FLOPs。这意味着 FLOPs 利用率约为 `3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题 8：** 混合专家（MoE）模型拥有 $E$ 份标准稠密 MLP 块的副本，每个词元会激活其中 $k$ 个专家。对于在 TPU v5e 上、权重为 int8 的 MoE，要达到算力受限需要多少词元的批大小？对于拥有 256 个（路由）专家且 $k=8$ 的 DeepSeek，这个数值是多少？

{% details Click here for the answer. %}

由于每个专家有 $E$ 份副本，在 int8 下，对于每个权重矩阵，我们需要加载 $E \cdot D \cdot F$ 字节。由于每个词元激活 $k$ 个专家，对于每个权重矩阵我们有 $2\cdot k \cdot B \cdot D \cdot F$ 次 FLOPs。要在 int8 权重和 bfloat16 FLOPs 下达到算力受限，我们需要算术强度（每加载 1 字节对应的 FLOPs）超过 TPU 约 240 FLOPs/字节的水平，这发生在 $(2\cdot k \cdot BDF) / EDF > 240$，即 $k \cdot B / E > 120$ 之时。

因此，要达到算力受限，我们需要 $B > 120 \cdot E / k$。对于 DeepSeek，这给出 $B > 120 \cdot 256 / 8 = 3840$。在生成阶段，这是一个相当惊人的大批量大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第 4 部分到此结束！关于第 5 部分（扩展 Transformer 训练），[请点击这里](../training)！</h3>

## 附录 {#附录}

### 附录 A：Flash Attention 是如何工作的？ {#附录-a-flash-attention-是如何工作的}

将 Transformer 扩展到超长上下文的传统反对意见是：注意力的 FLOPs 和内存占用会随上下文长度呈二次方增长。虽然注意力 QK 乘积的形状确实为 $[B, T, S, N]$（其中 B 是批大小，T 和 S 是 Q 和 K 的序列维度，N 是头数），但这一说法伴随着一些重要的注意事项：

1. 正如我们之前指出的，尽管这是二次方的，但注意力 FLOPs 只有在 $$T > 8 \cdot D$$ 时才会占主导；而且在训练时，单个注意力矩阵所占的内存相对于内存中所有权重和激活检查点而言很小，尤其是在分片之后。
2. 为了计算注意力，我们并不需要具现化（materialize）完整的注意力矩阵！我们可以计算局部的求和与最大值，从而避免具现化超过数组的一小个分块。虽然总 FLOPs 仍是二次方的，但我们可以大幅降低内存压力。

这第二个观察最早由 [Rabe 等人 2021](https://arxiv.org/abs/2112.05682) 提出，随后出现在 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)（Dao 等人，2022）中。其基本思想是分块计算 K/V 上的注意力：我们先计算局部 softmax 和一些辅助统计量，再将它们传给下一个分块，由后者与自身的局部分块进行合并。具体而言，我们计算：

1. **M：** 沿序列维度的 $$q \cdot k$$ 的滚动最大值
2. **O：** 沿序列维度的滚动完整注意力 softmax
3. **L：** 滚动分母 $$\sum_i \exp(q \cdot k_i - \text{running max})$$

有了这些，我们就可以仅用常数级别的内存来计算新的最大值、新的滚动求和以及新的输出。为了粗略地描述其工作原理，注意力大致是如下运算：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

减去最大值是为了数值稳定性，并且可以在不影响结果的前提下减去它，因为 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$。仅看上面的分母，如果我们设想有两个相邻的键向量分块 $$K^1$$ 和 $$K^2$$，并分别计算每个分块的局部 softmax 求和 $$L^1$$ 和 $$L^2$$

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

那么，我们可以通过下式将这两个分块合并为完整的 softmax 求和：

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

对于完整的 softmax 也同样可以这样做，从而为我们提供了一种累积任意大 softmax 求和的方法。下面是 Flash Attention 论文中的完整算法。

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件角度看，这让我们可以将 Q 的分块放入向量内存（VMEM，即上述算法所称的片上 SRAM），从而每次迭代只需加载 KV 分块，提高了算术强度。我们还可以将滚动统计量保留在向量内存（VMEM）中。

最后还有一个值得强调的微妙之处，即注意力 softmax 的一个性质，它被用来使 Flash VJP（反向模式导数）的计算在训练中可行。我们定义一个中间的 softmax 数组：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}$$

在注意力中，我们从反向模式的 *dO* 和 *V* 数组得到 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在将这个梯度反向传播到 Q 和 K 的过程中

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，用沿特征**深度（depth）**维度的局部收缩，来替换沿较大的键**长度（length）**维度的收缩。

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这一替换对于实现 VJP 的序列块*局部（local）*计算至关重要，并且使得诸如环形注意力（ring attention）等更巧妙的分片方案成为可能。
