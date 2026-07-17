---
layout: distill
title: "How to Parallelize a Transformer for Training（如何对 Transformer 进行训练并行化）"
permalink: /training-zh/
sitemap: false
# permalink: /main/
description: "这里我们讨论 LLM 训练中使用的四种主要并行方案：数据并行、全分片数据并行（FSDP）、张量并行与流水线并行。对每一种，我们都会算出它在什么情况下会受到通信的瓶颈限制。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers-zh"
previous_section_name: "第 4 部分. 你需要了解的 Transformer 数学"

next_section_url: "../applied-training-zh"
next_section_name: "第 6 部分. 在 TPU 上训练 LLaMA 3"

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
  - name: "我们所说的\"扩展\"是什么意思？"
  - subsections:
    - name: "数据并行"
    - name: "全分片数据并行（FSDP）"
    - name: "张量并行"
    - name: "结合 FSDP 与张量并行"
    - name: "流水线"
    - name: "跨 Pod 扩展"
  - name: "TPU 上 LLM 训练的要点"
  - name: "一些练习题"
  - name: "附录"
  - subsections:
    - name: "附录 A：推导反向传播的通信"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

## 我们所说的"扩展"是什么意思？ {#我们所说的-扩展-是什么意思}

“模型扩展”（model scaling）的目标是能够在增加用于训练或推理的芯片数量的同时，获得成比例的、线性的吞吐量提升（我们称之为*强扩展*）。而单芯片上的性能取决于内存带宽与 FLOPs 之间的权衡，集群层面的性能则取决于能否把芯片间通信与有用的 FLOPs 重叠，从而隐藏通信开销。这并非易事，因为增加芯片数量会增大通信负载，同时减少我们可用于隐藏通信的每设备计算量。正如我们在 [Section 3](../sharding) 中看到的，分片矩阵乘法常常需要代价高昂的 AllGather 或 ReduceScatter，这会阻碍 TPU 执行有用的工作。本节的目标就是弄清楚这些通信何时会变得*过于昂贵*。

在本节中，我们将讨论四种常见的并行方案：（纯）**数据并行**、**全分片数据并行**（FSDP / ZeRO 分片）、**张量并行**（也称为模型并行），以及（简要地）**流水线并行**。对每一种方案，我们都会说明它带来了多少通信开销，以及在什么情况下这一开销开始成为我们计算开销的瓶颈。<d-footnote>我们将聚焦于通信约束——因为尽管内存容量约束很重要，但在预训练中使用重计算（梯度检查点）以及海量芯片时，它通常并不会成为我们的瓶颈。我们在此也不讨论 MoE 的专家并行——那会大幅扩展设计空间——而只讨论稠密 Transformer 的基本情况。</d-footnote> 在本节中，你只需关注芯片间通信开销，因为只要我们的单芯片批大小足够大，数据从 HBM 传输到矩阵乘法单元（MXU）的过程就已经与计算重叠了。

我们将使用下列记号来简化本节的计算。

| Notation | Meaning (model parameters)                                             |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub>（隐藏维度 / 残差流维度）                         |
| F        | **d**<sub>ff</sub>（前馈维度）                                         |
| B        | 批次维度（批中的词元数量；是总量，非每设备）                           |
| T        | Sequence length（序列长度）                                            |
| L        | Number of layers in the model（模型的层数）                            |

| Notation | Meaning (hardware characteristic)                                                                 |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 每芯片 FLOPs/s（FLOPS/s per chip）                                                                |
| W        | 网络带宽（双向，常以下标表示，例如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$）                         |
| X        | 沿网格轴 X 的芯片数量（Number of chips along mesh axis X）                                        |
| Y        | 沿另一条网格轴（标记为 Y）的芯片数量（Number of chips along an alternate mesh axis, labeled Y）  |
| Z        | 沿第三条网格轴（标记为 Z）的芯片数量（Number of chips along a third mesh axis, labeled Z）       |

为简化起见，**我们把 Transformer 近似为一堆 MLP 块**——正如我们在 [Section 4](../transformers) 中看到的，对于较大的模型，注意力所占的 FLOPs 比例相对较小。我们还会忽略门控 matmul，从而得到每一层如下的简化结构：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>Figure:</b> 一个简化的 Transformer 层。我们把每个 FFW 块视为两个矩阵的堆叠：<b>W<sub>in</sub></b>: <code>bf16[D, F]</code>（升维投影）与 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code>（降维投影），输入为 <b>In</b>: <code>bf16[B, D]</code>。"%}

{% details 以下是我们这个无并行的小 Transformer 的完整算法。 %}

<div markdown=1 class="algorithm">

**Forward pass:** need to compute Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**Backward pass:** need to compute dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] （上一层需要用到）

</div>

我们提供这个，是为了与加入通信后的算法做对比。

{% enddetails %}

以下是我们将讨论的 4 种并行方案。每一种方案都可以看作由上面图示中对 **In**、**W<sub>in</sub>、W<sub>out</sub>、Out** 的分片方式唯一定义。

**1. 数据并行：** *激活值沿批次维度分片，参数和优化器状态在每个设备上复制。通信只发生在反向传播阶段。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 全分片数据并行（FSDP 或 ZeRO-3）：** *激活值沿批次维度分片（与纯数据并行相同），参数沿同一网格轴分片，并在前向传播中使用前即时 AllGather。优化器状态同样沿批次维度分片。减少了重复的内存占用。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为 Megatron 分片或模型并行）：** *激活值沿 D（$d_\text{model}$）分片，参数沿 F（$d_{ff}$）分片。在每个块前后对激活值做 AllGather 与 ReduceScatter。可与 FSDP 兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维度分片，激活值微批次化并沿层维度滚动。流水线阶段之间的通信极小（仅将激活值在单跳之间移动）。为兼容记号：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行 {#数据并行}

**Syntax:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当你的模型即使只用一个很小的批大小（>240 个词元，从而算力受限）也能放下到单芯片上时，**你应该始终使用简单的数据并行。** 纯数据并行把我们的激活值分到任意数量的 TPU 上，只要 TPU 数量小于我们的批大小即可。前向传播不涉及任何通信，但在每一步结束时，**每个 TPU 都会对它的本地梯度执行一次 AllReduce，以在更新参数之前将它们同步。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 纯数据并行（前向传播）示意图。我们的激活值（左）完全沿批次维度分片，而我们的权重完全复制，因此每个 TPU 都持有相同的一份权重副本。这意味着我们权重占用的总内存放大了 N 倍，但前向传播不需要任何通信。"%}

{% details 以下是前向与反向传播的完整算法。我们滥用记号，把 dL/dOut 写作 dOut，纯粹是为了紧凑。 %}

<div markdown=1 class="algorithm">

**Pure Data Parallelism Algorithm:**

**Forward pass:** need to compute Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**Backward pass:** need to compute dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D]) （不在关键路径上，可异步执行）
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F]) （不在关键路径上，可异步执行）
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] （上一层需要用到）

</div>

我们忽略了损失函数的细节，并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 简写。注意，尽管我们最终的损失是 **AllReduce**(Loss[B<sub>X</sub>]) 的均值，但我们只需要在反向传播对权重梯度求平均时计算这个 AllReduce。

{% enddetails %}

注意，前向传播没有通信——**全部都在反向传播里**！反向传播还有一个很好的性质：AllReduce 不在“关键路径”上，这意味着每次 AllReduce 都可以在方便时执行，而不会阻塞你进行后续操作。如果总通信开销超过我们的总计算开销，它_仍然会成为我们的瓶颈_，但从实现角度看要宽容得多。我们将看到，模型/张量并行不具备这一性质。

**为什么要这么做？** 纯数据并行通过将激活值沿批次维度拆分，减轻了激活值内存压力，使我们可以在拥有更多芯片来拆分批次维度时，几乎任意地增大批大小。尤其是在训练时，激活值常常占据我们的大部分内存用量，这一点非常有用。

**为什么不这么做？** 纯数据并行对来自模型参数或优化器状态的内存压力毫无帮助，这意味着对于参数加上优化器状态放不进单个 TPU 的大规模模型，纯数据并行很少有用武之地。为了有个量级概念，如果我们用 bf16 存储参数、用 fp32 存储优化器状态，并使用 Adam<d-footnote>Adam 存储参数、一阶与二阶累加器。由于参数采用 bfloat16、优化器状态采用 float32，每个参数共占用 `2 + 8 = 10` 字节。</d-footnote>，那么我们能放下的最大模型的参数量为 $$\text{TPU memory} / 10$$，例如在使用 96GB HBM 的 TPUv5p 芯片、且采用纯数据并行时，这大约是 9B 参数。

<p markdown=1 class="takeaway">**要点**：使用 Adam 和纯数据并行训练时，我们能训练的最大模型满足 $$\text{num_params} = \text{HBM per device} / 10$$。对于 TPU v5p，这大约是 9B 参数。<d-footnote>注意这里没有包含梯度检查点，所以实际上并不实用。这是一个 batch 为 1 个词元时的绝对下界。</d-footnote></p>

*为了让它在训练真实模型时真正有用，我们至少需要把模型参数或优化器部分地分片。*

**我们何时会因通信而成为瓶颈？** 正如上面所见，我们每层有两个 AllReduce，每个大小为 $$2DF$$（针对 bf16 权重）。数据并行在什么情况下会让我们通信受限？

如上表所示，令 $C$ = 每芯片 FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 批被划分到的分片数量<d-footnote>我们假设这一划分是在 ICI 网格上进行的，因此相关的网络带宽是 $W_\text{ici}$</d-footnote>。我们来计算执行相关 matmul 所需的时间 $$T_\text{math}$$，以及所需的通信时间 $$T_\text{comms}$$。由于这一并行方案在前向传播中不需要通信，我们只需针对反向传播计算这两个量。

*通信时间：* 从前一节我们知道，在一维网格上执行一次 AllReduce 所需的时间只取决于被 AllReduce 的数组总字节数以及 ICI 带宽 $W_\text{ici}$；具体来说，AllReduce 的时间为 $2 \cdot \text{total bytes} / W_\text{ici}$。由于我们需要对 $W_\text{in}$ 和 $W_\text{out}$ 都做 AllReduce，我们每层有 2 个 AllReduce。每个 AllReduce 针对一个权重矩阵，即一个包含 $DF$ 个参数的数组，也就是 $2DF$ 字节。综上，单层内 AllReduce 的总时间为

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*Matmul 时间：* 每层在前向传播中包含两个 matmul，在反向传播中包含四个 matmul，每个需要 $2(B/X)DF$ FLOPs。因此，对于反向传播中的单层，我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们进行重叠，每层的總时间为这两个量中的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$，即

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

时，我们保持算力受限。

结论是，要在数据并行下保持算力受限，我们需要每设备的批大小 $$B / X$$ 超过 ICI 运算强度 $C / W_\text{ici}$。这归根结底是因为：计算时间随每设备批大小缩放，而通信时间与这一量无关（因为我们传输的是模型权重）。注意 $B/X > C/W_\text{ici}$ 这个条件与单设备算力受限规则 $B > 240$ 的相似之处；在那种情况下，规则同样源于：计算时间随批大小缩放，而数据传输量（在 $B \ll F, D$ 区间）与批大小无关。

我们代入一些真实数字来获得量级概念。对于 TPUv5p，在 ICI 上做一维数据并行时 `C=4.6e14`、`W=2 * 9e10`，因此**我们每芯片的批大小至少要达到 2,550，才能避免通信受限**。由于我们可以跨多个轴做数据并行，如果我们把 TPUv5p Pod 的全部三个轴都用于纯数据并行，我们的带宽 $W_\text{ici}$ 就变为原来的 3 倍，从而可以把每 TPU 的批大小降到仅 BS=850，或者每个 Pod（8960 个芯片）每批 7.6M 个词元！**由此可见，纯数据并行相当难以成为瓶颈！**

<p markdown=1 class="takeaway">**注 [上下文并行]：** 在本节中，$B$ 始终指**以词元计**的总批大小。然而显然，我们的批是由许多不同的序列组成的，那么这是如何运作的呢？就 MLP 而言，**词元就是词元**！它们属于同一序列还是两个不同序列并不重要。因此，我们基本可以自由地对批次维度和序列维度同时做数据并行：我们称之为上下文并行或序列并行，但你可以把它看作另一种数据并行。注意力比 MLP 更棘手，因为我们要做一些跨序列的计算，但这可以通过在注意力过程中收集 KV 或 Q、并仔细地重叠 FLOPs 与通信来解决（通常借助一种称为“ring attention”的技术）。在本节中，我们将完全忽略序列维度，并假设存在某种程度的批次或序列并行。</p>

**关于多网格轴的说明：** 我们应当快速说明多个轴如何影响可用带宽。当我们将多个网格轴用于某个并行策略时，我们会获得更大的带宽。

* **定义：** $M_X$（$M_Y$、$M_Z$ 等）是某个并行策略所跨的硬件网格轴数量。
* **效果（带宽受限）：** 使用 $M$ 个轴提供（约 $M$ 倍）的聚合链路带宽，因此集合通信的时间按 $\propto 1/M_X$ 缩放。

### 全分片数据并行（FSDP） {#全分片数据并行-fsdp}

**Syntax:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

全分片数据并行（常被称为 FSDP 或 ZeRO 分片<d-cite key="zero"></d-cite>）把模型的优化器状态和权重分片到数据并行的各个分片上，并按需高效地收集和散布它们。**相比纯数据并行，FSDP 大幅降低了每设备内存占用，并节省了反向传播的 FLOPs，而开销极小。**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>Figure:</b> FSDP 沿数据维度对 Win 的收缩维度和 Wout 的输出维度做分片。这减少了内存，但（根据第 3 节）要求我们在执行 matmul 之前先收集 W 的权重。注意激活值（左）<i>并没有沿收缩维度分片</i>，这正是迫使我们做收集的原因。<b>注意我们的权重优化器状态同样沿收缩维度分片。</b>"%}

你大概还记得（见 [Section 3](../sharding)），一个 AllReduce 可以分解成一个 AllGather 和一个 ReduceScatter。这意味着，我们不必像标准数据并行那样做完整的梯度 AllReduce，而是可以把权重和优化器状态在芯片间分片，在前向传播每层时 AllGather 它们，在反向传播时对权重做 ReduceScatter，而且不增加额外开销。

{% details 以下是 FSDP 的完整算法。 %}

<div markdown=1 class="algorithm">

**Fully-Sharded Data Parallelism (FSDP):**

**Forward pass:** need to compute Loss[B<sub>X</sub>]

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) （不在关键路径上，可在上一层进行）
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] （现在可以丢弃 W<sub>in</sub>[D, F]）
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) （不在关键路径上，可在上一层进行）
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**Backward pass:** need to compute dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D]) （不在关键路径上，可异步执行）
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) （可提前进行）
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] （这里可以丢弃 W<sub>out</sub>[F, D]）
6.  dW<sub>in</sub>[D,F] = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F]) （不在关键路径上，可异步执行）
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) （可提前进行）
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] （上一层需要用到）（这里可以丢弃 W<sub>in</sub>[D, F]）

</div>

{% enddetails %}

这也被称为“ZeRO 分片”（Zero Redundancy Optimizer，零冗余优化器），因为我们不做任何不必要的计算，也不存储任何不必要的状态。ZeRO-{1,2,3} 分别用来指以这种方式对优化器状态、梯度和权重做分片。由于它们的通信开销都相同<d-footnote>严格来说，FSDP 在前向传播中增加了纯 DP 所没有的通信，但其比例与反向传播相同，因此不会对通信屋顶线（roofline）产生影响。关键在于，ZeRO-3 把一个反向传播中的 AllReduce 变成了一个 AllGather 和一个 ReduceScatter，而二者总通信量相同。</d-footnote>，我们基本上随时都可以采用 ZeRO-3 分片，即把参数、梯度和优化器状态在一组设备上分片。

**我们为什么要这么做？** 标准数据并行涉及大量重复工作。每个 TPU 都对完整梯度做 AllReduce，然后更新完整的优化器状态（所有 TPU 上完全相同的工作），再更新参数（同样是完全重复的）。对于 ZeRO 分片（对梯度/优化器状态分片），你可以用 ReduceScatter 来分散梯度，只更新你那一份优化器状态分片、更新一份参数分片，然后在需要时为前向传播 AllGather 参数，而不必做 AllReduce。

**我们何时会因通信而成为瓶颈？** 我们的相对 FLOPs 与通信开销与纯数据并行完全相同，因为反向传播中的每个 AllReduce 都变成了一个 AllGather + ReduceScatter。回想一下，一个 AllReduce 是作为一个 AllGather 和一个 ReduceScatter 实现的，二者各占一半开销。这里我们对前向传播建模，因为它与反向传播有相同的 FLOPs 与通信之比：

$$\begin{aligned}
T_\text{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_\text{comms} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

因此，与纯数据并行一样，当 $$B / X > C / W_\text{ici}$$，即当每设备批大小 $B/X$ 超过“ICI 运算强度” $C/W_\text{ici}$（v5p 上为 `4.59e14 / 1.8e11 = 2550`）时，我们算力受限。这对我们很有利，因为它意味着：如果我们的每设备批大小大到足以让纯数据并行保持算力受限，那么我们可以在不必担心脱离算力受限区间的情况下，直接升级到 FSDP，从而为自己省下大量的参数和优化器状态内存！尽管我们确实给前向传播增加了通信，但这一开销无关紧要，因为它只是与前向传播的 FLOPs 重叠。

<p markdown=1 class="takeaway">**要点：** 当每设备批大小小于 $2550 / M_X$（其中 $M_X$ 为网格轴数量）时，FSDP 与纯数据并行在 TPUv5 上都会变为带宽受限。</p>

例如，DeepSeek-V2（近来少数公开训练批大小信息的强模型之一）使用了约 40M 个词元的批大小。**这能让我们在触及带宽上限之前，扩展到大约 47,000 个芯片，即约 5 个 TPUv5 Pod。**

对于 LLaMA-3 70B，其训练量约为 `6.3e24 (15e12 * 70e9 * 6)` FLOPs，我们可以把 16M 个词元的批大小分配到大约 `16e6 / (2550 / 3) = 18,823` 个芯片上（约 2 个 8960 芯片的 Pod），每个芯片以峰值 FLOPs 利用率（常称为 MFU）的 50%、即 `4.59e14` FLOPs 运行，并**在大约 17 天内完成训练**。不错！但我们来看看如何做得更好。

<p markdown=1 class="takeaway">**关于临界批大小的说明**：有些反直觉的是，随着总批大小减小（芯片数量固定），我们会更易受通信瓶颈限制。数据并行和 FSDP 让我们可以扩展到任意多的芯片，只要我们能够持续增大批大小！然而在实践中，随着批大小增大，我们往往会看到训练收益递减，因为梯度几乎不再含噪声。我们有时也会看到训练不稳定。因此，在“无限算力区间”中寻找最优分片方案，通常从一个由缩放定律确定的固定批大小和一个已知（很大）的芯片数量出发，然后目标是找到一种划分方式，让我们能在这许多芯片上放下那个小批大小。</p>

### 张量并行 {#张量并行}

**Syntax:** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$（我们使用 $$Y$$ 以便最终与 FSDP 结合）

在完全分片的数据并行 AllReduce 中，我们在芯片间搬运权重。我们也可以对模型的前馈维度分片，并在层内搬运激活值——这被称为“一维模型并行”或 Megatron 分片<d-cite key="megatron"></d-cite>。这可以解锁每个 Pod 更小的有效批大小。下图展示了一个矩阵以此方式分片的例子：

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 一个基础张量并行的例子。因为我们只在 Y 上对我们的激活值分片（而不像 FSDP 那样在 X 上分片），我们在 X 上复制激活值。用我们的标准记号，即为 <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]。因为我们只在一个收缩维度上分片，我们通常会在 matmul 之前对激活值 <b>A</b> 做 AllGather。"%}

如上所述，**In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\] 意味着我们必须要在第一个 matmul 之前收集激活值。当激活值比权重更小时，这比 ZeRO 分片更省。这通常在叠加了一定程度的 ZeRO 分片（这会减少收集的大小）时才成立。这也是我们倾向于把 ZeRO 分片与张量并行混合使用的原因之一。

{% details 以下是张量并行的算法！ %}

<div markdown=1 class="algorithm">

**Tensor Parallelism:**

**Forward pass:** need to compute Loss[B]

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) （在关键路径上）
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] （未沿收缩维度分片，故无通信）
3.  Out[B, D] = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D]) （在关键路径上）
5.  Loss[B] = ...

**Backward pass:** need to compute dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) （在关键路径上）
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] （这里可以丢弃 dOut[B, D]）
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) （这可以与前向传播的 (1) 共享，从而跳过）
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B, D] \*<sub>B</sub> dTmp[B, F<sub>Y</sub>]
7.  dIn[B, D] = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] （上一层需要用到）
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D]) （在关键路径上）

</div>

{% enddetails %}

张量并行的一个好处是，它与我们 Transformer 前向传播中的两个矩阵配合得很好。朴素地，我们会在两个矩阵之后各做一次 AllReduce。但这里我们先做 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**，再做 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**。这意味着我们在开头 AllGather **In**，在结尾 ReduceScatter **Out**，而不是做 AllReduce。

**这有多昂贵？** 我们只对前向传播建模——反向传播只是这里每个操作的转置。在一维张量并行中，我们在第一个 matmul 之前 AllGather 激活值，在第二个之后 ReduceScatter 它们，每次发送两字节（bf16）。我们来看何时会因通信而受限。

$$\begin{align}
T_\text{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_\text{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

注意到我们希望计算开销大于通信开销，于是得到：

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

因此例如，在 bf16 下 TPUv5p 的 $C / W_{ici} = 2550$，所以我们最多只能做到 $Y < F / 2550$ 的张量并行。当我们有多个 ICI 轴时，我们的 $T_\text{comms}$ 会减少 $M_Y$ 倍，于是得到 $Y < M_Y \cdot F / 2550$。

<p markdown=1 class="takeaway">**要点**：当 $Y > M_Y \cdot F / 2550$ 时，张量并行会变为通信受限。对大多数模型而言，这大约在 8 路到 16 路张量并行之间。</p>

**注意，这与计算的精度无关**，因为例如在 TPUv5p 上，int8 的 $$C_\text{int8} / W_{ici}$$ 是 $$5100$$ 而非 $$2550$$，但通信量也减半了，所以两个二倍因子相互抵消。

**让我们看几个例子：**

* 在 TPUv5p 上，对于 LLaMA 3-70B（$$D = 8192$$、$$F \approx 30,000$$），我们可以从容地做 8 路张量并行，但在 16 路张量并行下会通信受限。8 路模型分片所需的 F 为 20k。
* 对于 Gemma 7B，$$F \approx 50k$$，因此我们在 19 路张量并行时会通信受限。这意味着我们很可能可以做 16 路，并且仍有不错的性能。

### 结合 FSDP 与张量并行 {#结合-fsdp-与张量并行}

**Syntax:** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP 与张量并行好的一点是它们可以结合。通过同时沿两个轴对 **W<sub>in</sub>** 和 **W<sub>out</sub>** 分片，我们既能节省内存，又能节省计算。因为我们沿 X 对 B 分片，所以减小了模型并行 AllGather 的大小；又因为我们沿 Y 对 F 分片，所以减小了 FSDP 的通信开销。这意味着二者结合能让我们达到比上面更低的每副本有效批大小。

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 结合 FSDP 与张量并行的示意图。与其他情况不同，这里模型参数没有重复。"%}

{% details 以下是混合 FSDP + 张量并行的完整算法。尽管我们有大量通信，但所有的 AllGather 和 ReduceScatter 都更小，因为我们对激活值做了批次分片、对权重做了更多张量分片！ %}

<div markdown=1 class="algorithm">

**Forward pass:** need to compute Loss[B]

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) （在关键路径上）
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) （可提前进行）
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) （可提前进行）
5.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D]) （在关键路径上）
7.  Loss[B<sub>X</sub>] = ...

**Backward pass:** need to compute dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) （在关键路径上）
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D])
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) （可提前进行）
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] （这里可以丢弃 dOut[B, D]）
7.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) （不在关键路径上 + 可与上一层的 (2) 共享）
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F<sub>Y</sub>]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>])
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) （可提前进行）
11. dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] （上一层需要用到）
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D]) （在关键路径上）

</div>

{% enddetails %}

**FSDP 与 TP 怎样组合才合适？** 一条简单而关键的准则是：FSDP 搬运权重，张量并行搬运激活值。这意味着随着批大小缩小（尤其是我们做更多数据并行时），张量并行变得更便宜，因为我们的每份分片激活值更小。

* 张量并行执行 $$\mathbf{AllGather}_Y([B_X, D_Y])$$，它随 $$X$$ 增大而缩小。
* FSDP 执行 $$\mathbf{AllGather}_X([D_X, F_Y])$$，它随 $$Y$$ 增大而缩小。

因此，结合二者可以进一步降低我们的每副本最小批大小。我们可以像上面一样计算 FSDP 与 TP 的最优配比：

令 $$X$$ 为专用于 FSDP 的芯片数，$$Y$$ 为专用于张量并行的芯片数。令 $$N$$ 为我们切片中的芯片总数，且 $$N=XY$$。令 $$M_X$$ 和 $$M_Y$$ 分别为我们做 FSDP 和 TP 所跨的网格轴数（二者之和大致为 3）。我们只对前向传播建模，因为它每 FLOP 的通信量最大。把上面算法中的通信量加起来，我们有

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{TP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

同样地，我们的总 FLOPs 时间为

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

为简化分析，我们做两个假设：第一，我们允许 $X$ 和 $Y$ 取非整数值（只要它们为正且满足 $XY=N$）；第二，我们假设可以完全重叠 $X$ 轴与 $Y$ 轴上的通信。在第二个假设下，总通信时间为

$$T_\text{comms} = \max\left(T_\text{FSDP comms}, T_\text{TP comms}\right)$$

在我们追问何种条件下算力受限之前，先来找使总通信最小的 $X$ 和 $Y$ 的最优值。由于我们的 FLOPs 与 $X$ 和 $Y$ 无关，最优设置就是单纯让通信最小化的设置。为此，我们把上面的 $T_\text{comms}$ 用 $X$ 和 $N$（后者固定，因为它是系统中的芯片数）表示，而非用 $X$ 和 $Y$ 表示：

$$T_\text{comms} (X) = \frac{4D}{W_\text{ici}} \max\left(\frac{F \cdot X}{N \cdot M_X}, \frac{B}{X \cdot M_Y}\right)$$

由于 $T_\text{FSDP comms}$ 随 $X$ 单调递增，而 $T_\text{TP comms}$ 随 $X$ 单调递减，最大值必然在 $T_\text{FSDP comms} = T_\text{TP comms}$ 时取到最小，这发生在

$$\begin{align*}
\frac{FX_{opt}}{M_X} = \frac{BN}{X_{opt} M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

这非常有用！它告诉我们，对于给定的 $B$、$F$ 和 $N$，最优的 FSDP 配比是多少。我们来看看量级。代入真实数值，即 $N = 64$（对应 4x4x4 的芯片阵列）、$B=48,000$、$F=32768$，得到大约 $X\approx 13.9$。因此我们会取 $X=16$、$Y=4$，接近我们计算出的最优值。

<p markdown=1 class="takeaway">**要点：** 一般而言，在训练中，最优的 FSDP 配比为 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$。</p>

现在让我们回到对所有并行策略都提出过的问题：**在什么条件下我们会保持算力受限？** 由于我们可以重叠 FLOPs 与通信，当

$$\max\left(T_\text{FSDP comms}, T_\text{TP comms}\right) < T_\text{math}$$

时，我们算力受限。

令 $\alpha \equiv C / W_\text{ici}$（即 ICI 运算强度），我们可以化简为：

$$\max\left(\frac{F}{Y \cdot M_X}, \frac{B}{X \cdot M_Y}\right) < \frac{B \cdot F}{N \cdot \alpha}$$

由于我们算出的 $X_{opt}$ 使左边的最大值相等，我们可以把它代入任意一边（注意 $Y_{opt} = N/X_{opt}$），即

$$\frac{F}{N \cdot W_\text{ici} \cdot M_X} \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N} < \frac{B \cdot F}{N \cdot C}$$

进一步化简，我们得到

$$ \sqrt{\frac{B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

其中左边与通信时间成正比，右边与计算时间成正比。注意，计算时间随批大小线性缩放（无论并行方式如何都是如此），而通信时间随批大小的平方根缩放。因此计算时间与通信时间之比也随批大小的平方根缩放：

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{\alpha \sqrt{N}}. $$

为了确保这个比值大于 1，从而保持算力受限，我们要求

$$ \frac{B}{N} > \frac{\alpha^2}{M_X M_Y F}$$

为了估算数值，再次代入 $F=32,768$、$\alpha=2550$ 和 $M_X M_Y=2$（三维网格必然如此）。这给出大约 $B/N > 99$。相比纯数据并行（或 FSDP）的情形——在三维网格下我们算得 $B/N$ 必须超过约 $850$ 才能算力受限——这大约为我们赢得了 8 倍的改善。

<p markdown=1 class="takeaway">**要点：** 将张量并行与 FSDP 结合，能让我们的 $B/N$ 降到 $$2550^2 / 2F$$。这使我们可以处理每芯片低至 100 的批大小，大约比仅用 FSDP 所能达到的小 8 倍。</p>

下面我们在有代表性的 4x4x4 芯片阵列上，绘制混合 FSDP + TP 的 FLOPs 与通信时间之比，并与仅张量并行（TP）和仅数据并行（FSDP）做对比。虽然纯 FSDP 并行在超大批大小下占优，但在批大小除以芯片数介于约 100 到 850 的区间，需要采用混合 FSDP + TP 策略才能保持算力受限。

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>Figure:</b> 在 F=30k 的 TPUv5p 4x4x4 切片上，最优混合 FSDP/TP 的 FLOPs 与通信时间之比。正如预期，张量并行与批大小呈固定比值；理想的混合 FSDP + TP 按 $\sqrt{B}$ 缩放，而 FSDP 按 $B$ 缩放。然而，在中等批大小区间，只有 FSDP + TP 能达到大于 1 的比值。"%}

下面是另一个 TPU v5p 16x16x16 的例子，展示不同分片方案下 FLOPs 与通信时间随批大小的变化。

{% include figure.liquid path="assets/img/math-comms-time.png" class="img-fluid" caption="<b>Figure:</b> 不同并行方案的通信耗时。黑色虚线是矩阵乘法 FLOPs 所花的时间，因此任何位于该线之上的曲线都是通信受限的。我们注意到，所有策略在批大小低于 6e5 时都会变为通信受限，这与我们预期的 4096 * 2550^2 / (2 * 8192 * 4) = 4e5 一致。" %}

黑色曲线是模型 FLOPs 所花的时间，这意味着任何使该值低于所有通信开销的批大小都严格是通信受限的。你会注意到黑色曲线大约在 `4e5` 处与绿色曲线相交，正如预测所示。

下面是一个可交互的动画，展示不同批大小下的总计算时间与通信时间：

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

你会注意到这大体上与上面一致（最小值在 FSDP=256、TP=16 附近），再加减一些由于各方案轴数略有差异带来的波动因子。

### 流水线 {#流水线}

你可能已经注意到，我们在前面的章节中完全避开了流水线的话题。流水线是 GPU 并行的一种主导策略，但在 TPU 上则不那么关键。简而言之，流水线化训练指的是把模型的各层拆分到多个设备上，并在前向与反向传播中于各流水线阶段之间传递激活值。算法大致如下：

1. 在 TPU 0 上初始化你的数据，权重沿层维度分片（对结合 FSDP 与张量并行的流水线，为 $W_\text{in}[L_Z, D_X, F_Y]$）。
2. 在 TPU 0 上执行第一层，然后将得到的激活值拷贝到 TPU 1，重复此过程直到最后一个 TPU。
3. 计算损失函数及其导数 $\partial L / \partial x_L$。
4. 对最后一个流水线阶段，计算导数 $\partial L / \partial W_L$ 和 $\partial L / \partial x_{L-1}$，然后将 $\partial L / \partial x_{L-1}$ 拷贝到上一个流水线阶段，重复此过程直到回到 TPU 0。

{% details 以下是一些（可运行的）Python 伪代码 %}

这段伪代码应能在 Cloud TPU VM 上运行。虽然它效率不高、也不够真实，但能让你感受到数据是如何在设备间传播的。

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**为什么这是个好主意？** 流水线很好，原因有很多：流水线阶段之间的通信开销很低，这意味着即使互联带宽很低，你也能训练非常大的模型。这在 GPU 上常常非常有用，因为 GPU 不像 TPU 那样由 ICI 密集连接。

**为什么这困难/麻烦？** 你可能已经在上面的伪代码中注意到，TPU 0 几乎总是空闲的！它只在整个流水线的最开始和最后一步才工作。这段空闲期被称为流水线气泡，非常难处理。通常我们首先用微批次来缓解：向流水线中送入多个小批次，使 TPU 0 在总步时间里至少被占用更大比例。

第二种方法是仔细地重叠前向 matmul $W_i @ x_i$、反向 $dx$ matmul $W_i @ \partial L / \partial x_{i+1}$，以及 $dW$ matmul $\partial L / \partial x_{i+1} @ x_i$。由于每一个都需要一些 FLOPs，我们可以重叠它们以完全隐藏气泡。下面是近期 DeepSeek v3 论文<d-cite key="DeepSeek3"></d-cite>中的一张图，展示了他们的“无气泡”流水线调度：

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>Figure:</b> DeepSeek v3 的流水线调度（来自其<a href='https://arxiv.org/pdf/2412.19437'>近期论文</a>）。橙色为前向 matmul，绿色为 dL/dx matmul，蓝色为 dL/dW matmul。通过优先安排反向的 dL/dx 乘法，我们可以避免 FLOPs 被“搁浅”。"%}

因为这对 TPU（拥有更大的互联 Pod）而言不那么关键，我们不会深入探讨，但理解流水线的主要瓶颈是一个很好的练习。

### 跨 Pod 扩展 {#跨-pod-扩展}

可能的最大 TPU 切片是一个拥有 8960 个芯片（以及 2240 个主机）的 TPU v5p SuperPod。当我们要扩展到超过这一规模时，就需要跨越数据中心网络（DCN）边界。每个 TPU 主机都配备了一个或多个网卡（NIC，Network Interface Card），通过以太网把主机连接到其他 TPU v5p Pod。正如 [TPU 章节](../tpus) 所述，每个主机约有 200Gbps（25GB/s）的全双工 DCN 带宽，即每 TPU 约 6.25GB/s 的全双工（出口）带宽。

通常，当扩展到单个 Pod 之外时，我们在 ICI 域内做某种形式的模型并行或 FSDP，然后在多个 Pod 之间做纯数据并行。令 $N$ 为我们想扩展到的 TPU 数量，$M$ 为每个 ICI 连接切片的 TPU 数量。为了在 DCN 上做 AllReduce，我们可以对一组 Pod 做环形归约，得到（在反向传播中）：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{M \cdot W_\text{dcn}}$$

通信带宽随 $M$ 缩放，因为不同于 ICI，总带宽会随着我们扩大 ICI 域、获得更多网卡而增长。化简后，我们发现当

$$\frac{B}{\text{slice}} > \frac{C}{W_\text{dcn}}$$

时，有 $T_\text{math} > T_\text{comms}$。

对于 TPU v5p，$\frac{C}{W_\text{dcn}}$ 约为 `4.59e14 / 6.25e9 = 73,440`。这告诉我们，为了高效地跨 DCN 扩展，每个 ICI 域需要有一个最小批大小，才能把数据送出每个节点。

**这有多大问题？** 举一个具体的例子：假设我们想在 TPU v5p 上以 2M 个词元的 BS 训练 LLaMA-3 70B。LLaMA-3 70B 的 $F\approx 30,000$。根据上面的章节，我们知道以下几点：

* 我们最多可以做 $Y = M_Y \cdot F / 2550 \approx 11 \cdot M_Y$ 的张量并行。
* 只要 $B / N > 2550 / M_X$，我们就能做 FSDP。这意味着如果我们想以 BS=2M、3 个轴的数据并行来训练，我们最多只能用约 $\approx 2400$ 个芯片，大约是一个 TPU v5p Pod 的四分之一。
* 当我们结合 FSDP + 张量并行时，若 $B / N < 2550^2 / (2 \cdot 30000) = 108$ 就会通信受限，因此这能让我们扩展到大约 18k 个芯片！然而，TPU v5p Pod 的最大规模是 8k 个芯片，超过之后我们就必须使用 DCN。

总而言之，对于 BS=1M 的训练，我们有一个不错的方案：大致取 X（FSDP）= 1024、Y（TP）= 8；但 BS=2M 时我们就需要使用 DCN。如上所述，我们的 DCN 运算强度为 $\text{73,440}$，因此我们只需确保每 ICI 域的批大小大于此值。这对我们来说轻而易举，因为用 2 个 Pod 时，每 Pod 的 BS 为 1M，每 TPU 批大小为 111，这很理想（也许稍显临界，但在理论上成立）。

<p markdown=1 class="takeaway">**要点：** 只要每 Pod 批大小至少为 73k 个词元，使用纯数据并行跨多个 TPU Pod 扩展就相当直接。</p>

## TPU 上 LLM 训练的要点 {#tpu-上-llm-训练的要点}

* 增加并行度或减小批大小，都会使我们更易受通信限制，因为它们减少了每芯片执行的计算量。
* 在合理的上下文长度（~32k）以内，我们可以把 Transformer 近似为一堆 MLP 块来建模，并通过每种并行方案如何对每层两到三个主要 matmul 做分片来定义它们。
* 训练时我们考虑 4 种主要并行方案，每种都有其自身的带宽与计算需求（数据并行、FSDP、张量并行，以及混合 FSDP + 张量并行）。

| **策略**                                 | **描述**                                                                                                                                                                            |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **数据并行**                             | 激活值按批次分片，其余全部复制，我们在反向传播中对梯度做 AllReduce（全归约）。                                                                                                     |
| **FSDP**                                 | 激活值、权重和优化器按批次分片，权重在使用前才收集，梯度做 reduce-scatter。                                                                                                        |
| **张量并行（又称 Megatron、模型并行）** | 激活值沿 $$d_\text{model}$$ 分片，权重沿 $$d_{ff}$$ 分片，激活值在 W<sub>in</sub> 之前被收集，结果在 W<sub>out</sub> 之后被 reduce-scatter。                                      |
| **混合 FSDP + 张量并行**                | 上述二者结合，其中 FSDP 收集模型分片权重。                                                                                                                                         |

以下是每种方法的“公式”：

$$\small
\begin{array}{cc}
\text{Strategy} & \text{Formula}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 每一种策略都有一个会因每设备计算与通信而变为网络/通信受限的界限。下面是每层的计算量与通信量，假设 $$X$$ 为 FSDP、$$Y$$ 为张量并行。

$$
\small
\begin{array}{ccc}
\text{Strategy} & \text{Compute per layer} & \text{Comms per layer} \\
& \text{(ignoring gating einsum)} & \text{(bytes, forward + backward pass)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少有用，因为模型及其优化器状态占用的字节数 = 参数量的 10 倍。这意味着我们很少能在内存中放下超过几十亿个参数。
* 当 $$\text{batch size per shard} < C / W$$（网络的运算强度）时，数据并行和 FSDP 会变为通信受限。对 ICI 而言是 2,550，对 DCN 而言约为 71,000。这可以通过更多并行轴来提高。
* 当 $$\lvert Y\rvert > F / 2550$$ 时，张量并行会变为通信受限。**对大多数模型而言这大约在 8-16 路。** 这与批大小无关。
* 混合 FSDP + 张量并行让我们可以把批大小降到低至 $$2550^2 / 2F \approx 100$$。这低得惊人。
* 跨 Pod 的数据并行在变为 DCN 受限之前，要求每 Pod 最小批大小约为 71,000。
* 基本上，如果你的批大小很大或模型很小，事情就很简单。你要么做数据并行，要么跨 DCN 做 FSDP + 数据并行。中间区段才是耐人寻味之处。

## 一些练习题 {#一些练习题}

我们本节以 LLaMA-2 13B 作为基础模型。以下是模型细节：

| hyperparam | value  |
| ---------- | ------ |
| L          | 40     |
| D          | 5,120  |
| F          | 13824  |
| N          | 40     |
| K          | 40     |
| H          | 128    |
| V          | 32,000 |

LLaMA-2 拥有独立的嵌入矩阵和输出矩阵，以及一个带门控的 MLP 块。

**问题 1：** LLaMA-2 13B 有多少个参数（我知道这问题有点傻，但请算一算）？*注意，正如 [Transformer Math](../transformers) 所述，LLaMA-3 有 3 个大 FFW 矩阵，两个升维投影、一个降维投影。本节我们忽略了两个“门控” einsum 矩阵，但它们的行为与本节中的 W<sub>in</sub> 相同。*

{% details 点击此处查看答案。 %}

* FFW 参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词表参数：$$2VD$$ = `0.33e9`
* 总计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，正如预期！

{% enddetails %}

**问题 2：** 假设我们以 BS=16M 个词元训练，并使用 Adam。暂时忽略并行，模型参数、优化器状态和激活值共占用多少内存？*假设我们以 bf16 存储参数、以 fp32 存储优化器状态，并在每层对激活值做三次检查点（在三个大 matmul 之后）。*

{% details 点击此处查看答案。 %}

参数（bf16）与两个优化器状态（fp32，即一阶与二阶矩累加器）所占的总内存为 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两个 matmul 之后的激活值形状为 $BF$，最后一个之后为 $BD$（见上面的 Transformer 图示），因此 bf16 的总内存为 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$，即 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`（因为 `B=16e6`）。其余所有激活值多少可忽略不计。

{% enddetails %}

**问题 3：** 假设我们想在一个 TPUv5p 16x16x16 切片上，以 32k 序列长度、总计 3M 个词元的批大小训练。假设我们想使用 bfloat16 权重和一个 float32 优化器，如上所述。

1. 我们能使用纯数据并行吗？为什么能或为什么不能？
2. 我们能使用纯 FSDP 吗？为什么能或为什么不能？在纯 FSDP 下，每设备会占用多少内存（假设我们只在 3 个大 FFW 矩阵之后做梯度检查点）？
3. 我们能使用混合 FSDP + 张量并行吗？为什么能或为什么不能？如果可以，$X$ 和 $Y$ 应当取多少？每设备会存储多少内存？仅用屋顶线（roofline）FLOPs 估计、并忽略注意力，在 40% MFU 下每个训练步需要多久？

{% details 点击此处查看答案。 %}

首先，我们写下一些数字。在 32k 序列长度、3M 批大小下，我们的序列批大小为 96。在 TPU v5p 16x16x16 切片上，我们有 `393TB` 的 HBM。

1. 我们不能使用纯数据并行，因为它在每个芯片上复制参数和优化器状态，而它们已经有大约 130GB（来自问题 2），超出了我们每芯片（96GB）的 HBM。

2. 我们先纯粹看内存。在问题 2 中把 BS=16M 换成 3M，得到总计 `~7.86e12` 的检查点激活值，加上 1.3e11 的优化器状态，总共几乎正好是 8e12 = 8TB。TPUv5p 切片总共拥有 `393TB` 的 HBM，因此我们安全地低于 HBM 上限。接下来我们看看会是通信受限还是计算受限。在 4096 个芯片、3 个并行轴下，我们的最小批大小为 `850 * 4096 = 3.48M` 个词元。这略高于我们的 3M 批大小。所以我们实际上是通信受限的，这有点遗憾。因此总的答案是**不能，我们无法单独使用 FSDP**。

3. 现在我们知道主要顾虑是通信受限，所以代入一些数字。首先，我们从上面知道，在混合 FSDP + 张量并行下，我们每芯片批大小需要高于 $2550^2 / 2F = 235$。这意味着理论上我们可以这么做！我们来算算各自取多少。

我们有公式 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，因此这里得到 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，意味着我们会做大约 1024 路 DP 和 4 路 TP。每 TPU 的内存如 (2) 所示，步时间则为 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">Part 5 到此结束！对于把本节内容应用到真实 LLaMA 模型的 Part 6，[请点击此处](../applied-training)！</h3>

## 附录 {#附录}

### 附录 A：推导反向传播的通信 {#附录-a-推导反向传播的通信}

在上面，我们把 Transformer 层的前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。我们如何推导反向传播所需的通信？

这相当自然地源自前一节对单个 matmul **Y = X * A** 的规则：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

运用这一点，我们得到下列公式（令 Tmp[B, F] 表示 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

注意，这些公式是数学陈述，未提及分片。反向传播的任务就是计算这四个量。因此，要确定所需的通信，我们只需取出上面四个方程中将要被 matmul 的所有量（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>）的分片方式——它们由我们的并行方案指定——并运用分片 matmul 的规则，来推知我们需要做哪些通信。注意 dOut 的分片方式与 Out 相同。
