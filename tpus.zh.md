---
layout: distill
title: "How to Think About TPUs（如何理解 TPU）"
permalink: /tpus-zh/
sitemap: false
# permalink: /main/
description: "This section is all about how TPUs work, how they're networked together to enable multi-chip training and inference, and how this affects the performance of our favorite algorithms. There's even some good stuff for GPU users too!"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "Part 1: Rooflines"

next_section_url: ../sharding
next_section_name: "Part 3: Sharding"

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
  - name: 什么是 TPU？
  - name: TPU 网络
  - name: 核心要点
  - subsections:
    - name: TPU 规格
  - name: 例题
  - name: 附录
  - subsections:
    - name: "附录 A：TPU 内部机制详解"
    - name: "附录 B：脉动阵列是如何工作的？"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

<p markdown=1 class="announce">你可能也会喜欢阅读关于 NVIDIA GPU 的新[第 12 节](../gpus)！</p>

## 什么是 TPU？ {#什么是-tpu}

**TPU 本质上是一个专门做矩阵乘法（称为 TensorCore）的计算核心，连接着一堆高速内存（称为高带宽内存，即 HBM）<d-cite key="tpu_paper"></d-cite>。** 下面是示意图：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图：</b> TPU 芯片的基本组成。TensorCore 是左侧的灰色方框，包含矩阵乘法单元（MXU）、向量处理单元（VPU）和向量内存（VMEM）。"%}

你可以把 TensorCore 理解为本质上是一台非常出色的矩阵乘法机器，但它还有几个值得一提的其他功能。TensorCore 有三个关键单元：

* **MXU**（矩阵乘法单元，Matrix Multiply Unit）是 TensorCore 的核心。对于大多数 TPU 代次，它每 8 个周期使用脉动阵列执行一次 `bf16[8,128] @ bf16[128,128] -> f32[8,128]` 的矩阵乘法<d-footnote>TPU v6e（Trillium）的 MXU 为 256x256，而之前所有代次都使用 128x128。</d-footnote>（详见 <a href="#附录-b-脉动阵列是如何工作的">附录 B</a>）。
  * 在 TPU v5e 上，按 1.5GHz 计，每个 MXU 约为 `5e13` bf16 FLOPs/s。大多数 TensorCore 拥有 2 或 4 个 MXU，因此例如 TPU v5e 的总 bf16 FLOPs/s 为 `2e14`。
  * TPU 还支持更高吞吐量的低精度 matmul（例如，每个 TPU v5e 芯片可执行 `4e14` int8 OPs/s）。

* **VPU**（向量处理单元，Vector Processing Unit）执行一般的数学运算，例如 ReLU 激活值，或向量之间的逐元素加法或乘法。归约（求和）也在这里进行。<a href="#附录-a-tpu-内部机制详解">附录 A</a> 提供了更多细节。
* **VMEM**（向量内存，Vector Memory）是位于 TensorCore 内部、靠近计算单元的片上暂存器。它比 HBM 小得多（例如，TPU v5e 上为 128 MiB），但与 MXU 之间的带宽高得多。VMEM 的运作方式有点像 CPU 上的 L1/L2 缓存，但容量更大且由程序员控制。HBM 中的数据需要先复制到 VMEM，TensorCore 才能对其进行计算。

**TPU 在矩阵乘法上非常、非常快**。这基本上是它们的本职工作，而且干得很漂亮。迄今为止最强大的 TPU 之一 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)，每个核心可做 `2.5e14` bf16 FLOPs / 秒，或每芯片 `5e14` bf16 FLOPs / 秒。一个由 8960 颗芯片组成的 Pod 可达到 4 bf16 exaFLOPs/s。这*相当可观*。这是世界上最强大的超级计算机之一。而 Google 拥有很多这样的机器。<d-footnote>TPU 及其脉动阵列之所以是这么强大的硬件加速器，是因为矩阵乘法是为数不多的、用 $O(n^3)$ 的算力换取 $O(n^2)$ 字节的算法之一。这使得普通的 ALU 很容易受算力而非内存带宽所制约。</d-footnote>

上面的示意图还包含一些其他组件，例如 SMEM 和标量单元，它们用于控制流处理，并在 <a href="#附录-a-tpu-内部机制详解">附录 A</a> 中简要讨论，但理解它们并非关键。另一方面，HBM 很重要，而且相当简单：

* **HBM**（高带宽内存，High Bandwidth Memory）是一大块高速内存，用于存储供 TensorCore 使用的张量。HBM 的容量通常约为数十吉字节（例如，[TPU v5e 拥有 16GiB 的 HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。

  * 当计算需要时，张量从 HBM 经 VMEM（见下文）流式送入 MXU，结果再从 VMEM 写回 HBM。

  * HBM 与 TensorCore（经 VMEM）之间的带宽被称为"HBM 带宽"（通常约为 1-2TB/秒），它限制了内存受限工作负载中计算的快慢。

**一般来说，所有 TPU 操作都是流水线化并相互重叠的。** 要执行一次 matmul $X \cdot A \to Y$，TPU 首先需要把矩阵 $A$ 和 $X$ 的分块从 HBM 复制到 VMEM，然后加载进 MXU，由它将 8x128（对应 $X$）和 128x128（对应 $A$）的分块相乘，再把结果逐块复制回 HBM。为了高效完成这一过程，matmul 被流水线化，使得与 VMEM 之间的复制和 MXU 的工作相互重叠。这让 MXU 能够持续工作，而不是等待内存传输，从而让 matmul 保持算力受限而非内存受限。

下面是如何从 HBM 执行逐元素乘积的一个例子：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图：</b> 一段展示在 TPU 上执行逐元素乘积的动画，其中字节从 HBM 载入。注意字节如何以分块方式从内存中流式读出、并将部分和流水线式写回，而无需等待整个数组生成完毕。"%}

matmul 看起来几乎一模一样，只是它会载入 MXU 而不是 VPU/向量单元，并且加载和存储的顺序不同，因为同一权重分块会被用于多个激活值分块。你可以看到数据分块流入 VMEM，然后进入 VREGs（向量寄存器），再进入向量单元，最后回到 VMEM 和 HBM。正如我们即将看到的，如果从 HBM 到 VMEM 的载入比向量单元（或 MXU）中的 FLOPs 更慢，我们就会陷入"带宽受限"，因为我们让 VPU 或 MXU 无活可干（饥饿）。

<p markdown=1 class="takeaway">**核心要点：** TPU 非常简单。它们将权重从 HBM 载入 VMEM，再从 VMEM 载入脉动阵列，后者每秒可执行约 200 万亿次乘加运算。HBM $\leftrightarrow$ VMEM 与 VMEM $\leftrightarrow$ 脉动阵列的带宽，为 TPU 能高效完成哪些计算设定了根本性的限制。</p>

**VMEM 与算术强度：** VMEM 比 HBM 小得多，但与 MXU 之间的带宽高得多。正如我们在 [第 1 节](../roofline) 中所见，这意味着如果一个算法能将其所有输入/输出放进 VMEM，它就不太可能遇到通信瓶颈。当某个计算的算术强度较低时，这一点尤其有用：VMEM 带宽约为 HBM 带宽的 22 倍，这意味着从 VMEM 读取或向 VMEM 写入的 MXU 操作只需 10-20 的算术强度即可达到峰值 FLOPs 利用率。也就是说，如果我们能把权重放进 VMEM 而非 HBM，我们的矩阵乘法就能在更小的批大小下达到 FLOPs 受限。这也意味着那些本质上算术强度较低的算法仍然可以很高效。只是 VMEM 太小了，所以这往往是个挑战。<d-footnote>我们有时会谈到 VMEM 预取，即在 VMEM 中提前载入权重，从而掩盖 matmul 的载入开销。例如，在普通的 Transformer 中，我们有时可以在注意力计算期间把较大的前馈权重载入 VMEM，若处于内存带宽受限状态，这能隐藏权重载入的开销。这要求权重足够小或分片足够细，以便单层能装入 VMEM 并留有富余空间。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**一个 TPU 芯片通常（但不总是）由两颗共享内存、可被视作单一大型加速器的 TPU 核心组成**，其 FLOPs 翻倍（这被称为"大核"（megacore）配置）。v4、v5 和 v6 代 TPU 都是如此（TPU v7 取消了大核，改为在两颗核心之间使用高带宽链路）。较老的 TPU 芯片拥有独立内存，被视为两个独立的加速器（TPU v3 及更早代次）。像 TPU v5e 这样的推理优化芯片每颗芯片只有一个 TPU 核心。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**以**每 4 颗放在一个"托盘"上**的方式排列，并通过 PCIe 网络连接到 **CPU 主机**。这是大多数读者最熟悉的形态：通过 Colab 或单个 TPU-VM 暴露出来的 4 颗芯片（8 个核心，但通常被视为 4 个逻辑大核）。对于像 TPU v5e 这样的推理芯片，每个主机有 2 个托盘而非 1 个，但每颗芯片也只有 1 个核心，因此得到 8 颗芯片 = 8 个核心。<d-footnote>在 Cloud TPU VM 上，每个托盘作为独立 VM 的一部分暴露出来，因此可见的仍是 4 个核心。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 带宽是受限的：** 和 HBM $\leftrightarrow$ VMEM 链路一样，CPU $\leftrightarrow$ HBM 的 PCIe 连接拥有特定的带宽，限制了你从主机内存载入 HBM 或反向载入的快慢。例如，TPU v4 的 PCIe 带宽为每方向 16GB / 秒，因此比 HBM 慢了近 100 倍。我们*可以*把数据载入/卸载到主机（CPU）内存，但速度并不快。

## TPU 网络 {#tpu-网络}

**芯片通过 Pod 内的 ICI 网络彼此相连**。在较老的代次（TPU v2 和 TPU v3）、推理芯片（例如 TPU v5e）以及 Trillium（TPU v6e）上，ICI（"芯片间互连"，inter-chip interconnects）连接 4 个最近邻（通过边缘链路构成二维环面）。TPU v4 和 TPU v5p 则连接到最近的 6 个邻居（构成三维环面）。注意这些连接**不**经过它们的主机，而是芯片之间的直接链路。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面结构将任意两个节点之间的最大距离从 $N$ 减小到 $N / 2$，使通信快得多。TPU 还有一种"扭转环面"配置，将环面以类似莫比乌斯带的方式包裹起来，以进一步缩短节点之间的平均距离。

**TPU Pod（由 ICI 连接）可以非常庞大：** 最大 Pod 规模（称为 **SuperPod**）在 TPU v4 上为 `16x16x16`，在 TPU v5p 上为 `16x20x28`。这些大型 Pod 由 `4x4x4` 的可重构立方体芯片组成，通过 [光学回环链路](https://arxiv.org/pdf/2208.10041) 连接<d-footnote>光交换机本质上只是一个具有相同 ICI 带宽的可重构连接。它只是让我们在连接立方体时仍能保留回环链路。</d-footnote>，我们可以重新配置它们以连接非常庞大的拓扑。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

更小的拓扑（例如 `2x2x1`、`2x2x2`）也可以申请，只是没有回环。这是一个重要的注意事项，因为它通常会使大多数通信的时间翻倍。任何完整立方体的整数倍（例如 `4x4x4` 或 `4x4x8`）都会有光交换机提供的回环。<d-footnote>注意，`2x2x4` 不会有任何回环，因为回环由光交换机提供，而光交换机仅在完整的立方体上可用。不过，TPU v5e 的 8x16 在较长轴上*会*有回环，因为它不使用可重构的光网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e 和 Trillium 的 Pod 由单一的 `16x16` 二维环面组成，在任一大小为 16 的轴上都有回环（即 `8x16` 在长轴上有回环）。TPU v5e 和 v6e（Trillium）无法扩展到 16x16 环面之外，但 Pod 之间仍可以通过标准的数据中心网络（DCN）相互通信，DCN 将各 TPU 主机彼此相连。同样，也可以申请更小的拓扑，在小于 16 的维度上不带回环。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接是 TPU 与 GPU 之间的一个关键区别**。GPU 通过分层的交换机连接，近似于每颗 GPU 之间的点对点连接，而不是像 TPU 那样使用局部连接。通常，节点内的 GPU（H100 为 8 颗，B200 NVL72 多达 72 颗）是直接相连的，而更大的拓扑需要每颗 GPU 之间经过 O(log(N)) 跳。一方面，这意味着 GPU 可以在很少的跳数内发送任意数据。另一方面，TPU 要便宜得多（因为 NVLink 交换机很昂贵）、连线和组合也更简单，并且能扩展到更大的拓扑，因为每设备的链路数和每设备的带宽是恒定的。详见 [此处](../gpus#networking)。

**相对于 DCN，ICI 非常快，但仍慢于 HBM 带宽。** 例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 拥有：

* 每芯片 `2.8e12` 字节/秒（2.8 TB/s）的 HBM 带宽。
* 每轴 `9e10` 字节/秒（90 GB/s）的 ICI 带宽，每芯片 3 个轴。<d-footnote>上述页面列出的带宽为 100 GB/s，与这里列出的略有不同。TPU 的 ICI 链路会根据所执行的操作而具有略有不同的带宽。一般来说，你可以放心使用本文档中的数字。</d-footnote>
* 每颗 TPU `6.25e9` 字节/秒（6.25 GB/s）的 DCN（出口）带宽（经由每个主机上的 1-2 个网卡（NIC））。<d-footnote>TPU v6e 和 TPU7x 为 12.5e9 字节/秒，v5e 为 3.125e9 字节/秒。</d-footnote>

这意味着，当我们把模型分片到多颗芯片上时，需要小心避免让较慢的跨设备通信成为 MXU 的瓶颈。

**多切片训练：** 一组由 ICI 相连的 TPU 被称为一个 **切片**。不同的切片之间可以通过 DCN 相互连接，例如连接不同 Pod 上的切片。由于 DCN 是比 ICI 慢得多的连接，我们应当尽量限制计算需要等待 DCN 数据的程度。DCN 是主机到主机的，因此要通过 DCN 将缓冲区从一颗 TPU 传输到另一颗 TPU，我们首先需要通过 PCIe 传输到主机，然后通过网络出口，再通过目标主机网络入口，最后通过 PCIe 进入 HBM。

## 核心要点 {#核心要点}

* TPU 很简单，在大多数情况下可以被看作一个连接到内存（极快）、通过 ICI 连接到其他芯片（相当快）、并通过 DCN 连接到数据中心其余部分（还算快）的矩阵乘法单元。

* 通信受限于我们各种网络带宽，按速度排序为：
  * HBM 带宽：介于 TensorCore 与其相关联的 HBM 之间。
  * ICI 带宽：介于一颗 TPU 芯片与其最近的 4 或 6 个邻居之间。
  * PCIe 带宽：介于 CPU 主机与其相关联的芯片托盘之间。
  * DCN 带宽：介于多个 CPU 主机之间，通常是那些未由 ICI 相连的主机。

* **在切片内部，TPU 仅通过 ICI 与最近邻相连。** 这意味着切片内相距较远的芯片之间经由 ICI 的通信，需要先经过中间的芯片逐跳转发。

* **权重矩阵需要在两个维度上都填充到至少 128** 的大小（TPU v6e 上为 256），以填满 MXU（实际上，较小的轴会被填充到 128）。

* **更低精度的矩阵乘法往往更快。** 在支持该特性的代次上，TPU 执行 int8 或 int4 OPs 的速度比 bfloat16 FLOPs 大约快 2 倍 / 4 倍。VPU 操作仍以 fp32 执行。

* 为了避免让 TPU 计算单元成为瓶颈，我们需要**确保每条通道上的通信量与通道速度成正比**。

### TPU 规格 {#tpu-规格}

以下是我们各芯片的一些具体数字：

| 型号                                       | Pod 规模  | 主机规模 | 每芯片 HBM 容量 | 每芯片 HBM 带宽（字节/秒） | 每芯片 FLOPs/s（bf16） | 每芯片 FLOPs/s（int8） |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.2e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |
| <span class="nowrap-header">TPU7x</span>   | 4x4x576  |   2x2x1   |       192GB       |        7.4e12         |       2.30e15       |       4.61e15       |

主机规模指的是连接到单个主机的 TPU 拓扑（例如，TPU v5e 有一个 CPU 主机以 4x2 拓扑连接 8 颗 TPU）。有关最新代次的更多细节，请参阅 [TPU7x 文档](https://docs.cloud.google.com/tpu/docs/tpu7x)。以下是互连数据：

| 型号        | ICI 每链路带宽（单向，字节/秒） | ICI 每链路带宽（双向，字节/秒） |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |             1.0e11             |           2.0e11            |
| **TPU v4p** |             4.5e10             |           9.0e10            |
| **TPU v5p** |             9.0e10             |           1.8e11            |
| **TPU v5e** |             4.5e10             |           9.0e10            |
| **TPU v6e** |             9.0e10             |           1.8e11            |
| **TPU7x**   |             9.0e10             |           1.8e11            |

我们同时列出单向（unidirectional）带宽和 bidi（双向）带宽，因为单向带宽更贴近硬件实际情况，而双向带宽更频繁地出现在涉及完整环面的方程中。<d-footnote>所谓 bidi（双向）带宽，是指沿单条链路两个方向总共可发送的字节数，或者等价地说，是在能有效利用两条链路的前提下，单个 TPU 沿某一特定轴总共可发出的字节数。当我们拥有一个正常运作的环（也就是在该轴上拥有回环连接）时，这一条件成立。在推理芯片上，当某轴为完整的 16 时成立；在训练芯片（v*p）上，当某轴为 4 的倍数时成立。我们更倾向于使用双向带宽，因为它频繁出现在涉及双向通信的计算中。</d-footnote>

PCIe 带宽通常约为每颗 TPU `1.6e10` 字节 / 秒（TPU v6e 为 `3.2e10`），而 DCN 带宽通常约为每颗 TPU `6.25e9` 字节 / 秒（TPU v6e 和 TPU7x 为 `12.5e9`，TPU v5e 为 `3.125e9`）。

## 例题 {#例题}

这些数字有点枯燥，但它们能让你对模型性能做出基本的屋顶线估计。我们做几道题，来解释为什么这很有用。你会在第 3 部分看到更多例子。

**问题 1 [为 LLM 延迟定界]：** 假设你想从一个参数量为 200B、以 bf16 存储、分片在 32 颗 TPU v4p 上的模型中采样。将所有参数从 HBM 载入脉动阵列需要多长时间？*提示：使用上面的数字。*

{% details 点击此处查看答案。 %}

**答案：** 我们要在 32 颗芯片上载入 `sizeof(bf16) * 200e9 = 400e9` 字节，即每芯片 12.5e9 字节，每颗芯片的 HBM 带宽为 1.23e12。因此载入耗时约 10 毫秒。

这相当有意思，因为*这是从该模型采样延迟的一个合理下界*。每个采样步都需要从 HBM 载入所有参数，因此耗时不可能少于 10 毫秒。在实践中，在较小的批大小下，这一下界是接近可达的。

{% enddetails %}

**问题 2 [TPU 细节]：** 考虑一个完整的 TPU v5e Pod。总共有多少个 CPU 主机？有多少个 TPU TensorCore？整个 Pod 的总 FLOPs/s 是多少？总 HBM 是多少？对 TPU v5p 的 Pod 做同样的练习。

{% details 点击此处查看答案。 %}

**答案：** 对于 TPU v5e，每个 Pod 为 `16x16`，每个主机是一个 4x2 的切片，因此我们有 `16*16 / 8 = 32` 个主机。对于 TPU v5e，每颗 TPU 只有一个核心，因此我们共有 256 个 TensorCore。总 FLOPs/s 在 bfloat16 下为 `16*16*2e14 = 5.1e16`。每颗芯片有 16GB 的 HBM，因此总内存为 `256 * 16 = 4TB`。

对于一个完整的 TPU v5p Pod，我们有 `16x20x28` 颗芯片，每个主机为 2x2x1，因此我们有 `(16*20*28) / (2*2) = 2,240` 个主机。对于 TPU v5p，每颗 TPU 有两个 TensorCore，因此我们共有 `8960 * 2 = 17,920` 个核心。总 FLOPs/s 在 bfloat16 下为 `8960 * 4.59e14 = 4.1e18`。每颗芯片有 96GB 的 HBM，因此总内存为 `8960 * 96 = 860TB`。

{% enddetails %}

**问题 3 [PCIe 运算强度]：** 假设我们被迫把一个类型为 $\text{bf16}[D, F]$ 的大权重矩阵 $A$，以及一批类型为 $\text{bf16}[B, D]$ 的激活值 $x$ 存储在主机 DRAM 中，并想对它们做矩阵乘法。这运行在单个主机上，并且我们使用连接到它的单颗 TPU v6e 芯片。你可以假设 $B \ll D$，且 $F = 4D$（在后面的章节中你会看到这些为何是合理的假设）。为了使我们在 PCIe 上保持 FLOPs 受限，所需的最小批大小 $B$ 是多少？假设 PCIe 带宽为 1.6e10 字节 / 秒。

{% details 点击此处查看答案。 %}

**答案：** 我们需要执行 $2BDF$ 次浮点运算，每颗芯片每秒可执行 `9.2e14` 次浮点运算。因此执行需要 $2BDF / 9.2e14$ 秒。我们需要从 DRAM 载入 $2DF + 2BD$ 字节，并将 $2BF$ 字节写回。我们受限于 PCIe 传输速度，因此需要 $2 \cdot (BD + DF + BF) / 1.6e10$ 秒来与 TPU 之间传输数据。由于我们希望计算耗时超过权重载入（假设我们能让所有权重载入与计算重叠），我们需要 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.6e10$。利用我们的假设 $B \ll D$ 以及 $F = 4D$，可以将其简化为

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.6 \times 10^{10}}$$

或

$$B > \frac{9.2 \times 10^{14}}{1.6 \times 10^{10}} \simeq 57{,}500$$

{% enddetails %}

**问题 4 [通用 matmul 延迟]：** 假设我们想把一个 int8[16384, 4096] 的权重矩阵乘以一个大小为 int8[B, 4096] 的激活值矩阵，其中 B 是某个未知的批大小。假设我们从 1 颗 TPU v5e 开始。

1. 这次乘法作为 B 的函数需要多长时间？*提示：分别计算从 HBM 载入数组所需的时间，以及乘法实际执行所需的时间可能会有帮助。哪个是瓶颈？*
2. 如果我们想从 VMEM 执行这个操作呢？作为 B 的函数，它需要多长时间？

{% details 点击此处查看答案。 %}

**答案：**（1）我们需要执行的操作数为 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。因此 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们需要从 HBM 向 VMEM 载入 $16384 \cdot 4096 + 4096 \cdot B$ 字节，并将 $16384 \cdot B$ 字节从 VMEM 写回 HBM。这意味着 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.2 \times 10^{11}$ 秒。假设通信与计算尽可能重叠，整个乘法大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}, \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}}\right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$，等价地即 $B > 267$ 时，我们将处于 FLOPs 受限。这比我们在 [第 1 节](../roofline) 中推导出的 240 略大，因为我们计入了 $D$ 和 $F$ 的完整影响。

（2）如果我们改为从 VMEM 载入，不妨将 VMEM 到 MXU 的带宽视为 HBM $\leftrightarrow$ VMEM 带宽的 22 倍。这将我们的数据载入分母从 8.2e11 变为 1.80e13，于是得到 $B > 11$。注意，在实践中我们无法把全部 VMEM 带宽都用于载入权重矩阵，因此实际上会更接近 20。

{% enddetails %}

**问题 5 [ICI 带宽]：** 假设我们有一个 TPU v5e 的 `4x4` 切片。假设我们想把一个类型为 `bf16[8, 128, 8192]` 的数组从 `TPU{0,0}` 发送到 `TPU{3, 3}`。假设 TPU v5e 的每跳延迟为 $1\mu s$。

1. 第一个字节要多久才能到达目的地？
2. 整个传输需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** 在 TPU v5e 上我们拥有二维连接。因为我们只有一个 `4x4` 切片（没有大小为 16 的轴），所以没有回环连接。因此，我们的目标芯片有两个可接收数据的端口，同样地，源芯片也有两个可发送数据的端口。我们需要传输的数据量为 `2 * 8 * 128 * 8192 = 1.7e7` 字节。我们可以同时从两个端口传输（即一半数组向右发送，一半向下发送），因此我们得到每秒传输 `2 * 4.5e10 = 9e10` 字节，这意味着传完整个数组大约需要 `1.7e7 / 9e10 = 188us`（假设我们带宽受限）。在 `4x4` 切片中，芯片 $(0, 0)$ 和 $(3, 3)$ 之间有六跳，因为对于少于 16 颗芯片的轴不存在回环链路。由于每跳的延迟约为 $1\mu s$，第一个字节大约会在 `6us` 后到达，而整个传输大约需要 `188 + 6 = 194us`，因为最后一个字节在离开源芯片后同样必须穿越六跳（一般来说，延迟项和带宽项是相加的，尽管这里的延迟只是一个很小的修正量）。

{% enddetails %}

**问题 6 [综合应用，较难]：** 假设你有一个大矩阵 **A**：`int8[128 * 1024, 128 * 1024]`，它均匀分片在 TPU v5e 的 4x4 切片上，但被卸载到了每颗芯片的主机 DRAM 中。假设你想把整个数组复制到 TPU{0, 0} 并与一个向量 `bf16[8, 128 * 1024]` 相乘。这需要多长时间？*提示：使用上面的数字。*

{% details 点击此处查看答案。 %}

**答案：** 让我们先列出需要执行的操作。我们的数组约为 16GB。根据上表，一个 TPU v5e 主机采用 4x2 拓扑，因此一个 4x4 有 2 个主机。于是，由于数组是均匀分片的，每个主机实际上包含数组的 1/2，即 8GB。我们需要把这些分块全部复制到 TPU{0,0}，这给了我们两种选择：

1. 我们可以通过 DCN 复制，然后通过 PCIe 将整个未分片的数组载入 HBM。
2. 我们可以把分片后的数组载入到各自对应的 TPU 上，然后通过 ICI 执行一次 gather，再在 TPU{0,0} 上执行 matmul。

显然方案（2）更好。与 ICI 相比，DCN 很慢，我们更希望通过多条 PCIe 链路而非少数几条（主机 0 上的 8 条）来载入一个大数组。下面是该系统一部分的示意图。如上所述，注意 TPU 通过 ICI 与邻居相连（即使跨主机也是如此），所有 TPU 都连接到其主机 CPU（经由 PCIe），而主机之间由 DCN 相连。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="实际上每个芯片都有自己的 PCIe 链路连接到其主机，但为了清晰起见，这里只显示了一个。"%}

现在我们来计算每一部分需要多长时间：

1. **PCIe 载入**：我们要通过 16 条 PCIe 链路载入 16GB 的分块，每条链路的带宽为 `1.6e10` 字节/秒。因此这大约需要 63 毫秒。

2. **ICI 复制：** 现在每颗 TPU 拥有我们数组的 16GB / 16 = 1GB。我们的 ICI 带宽为每链路 `9e10` 字节/秒*双向*，并且你会从上面示意图中注意到，在此拓扑中，TPU v5e 的 4 条 ICI 链路里只有 2 条在 TPU{0,0} 上被使用。由于 TPU{0,0} 需要沿 2 个轴、以 `4.5e10` 字节/秒/链路 的速率总共接收 15GB，我们可以把时间下界估计为 `15e9 / (4.5e10 * 2) = 167ms`。在实践中这可能难以达到，因为负载非常不均衡，但大概在 2 倍误差范围内。正如你将在第 3 节中看到的，执行一次完整的 AllGather 也大约需要 `16e9 / (4.5e10 * 2)`，因此这已接近最优。

3. **HBM $\rightarrow$ MXU 载入：** 为了执行我们最终的 matmul，我们需要通过 HBM 带宽将这些 16e9 字节加上 bf16[8, 128 \* 1024] 数组（另外 2MB，可忽略不计）载入 MXU，这需要 `16e9 / 8.2e11 = 20ms`。

4. **FLOPs：** 我们总共执行 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ 次 FLOPs，由于我们能执行 `1.97e14` bf16 FLOPs/s，因此得到 1.4ms。

总时间的一个上界是上述所有时间之和，但由于 TPU 通常能让这些操作相互重叠，我们可以将其视为一个由最慢一环决定瓶颈的流水线问题。假设确实如此，那么答案至少为 167ms，在不完美重叠的情况下可能更接近 200ms。

{% enddetails %}

<h3 markdown=1 class="next-section">第 2 部分到此结束！关于划分与跨 TPU 通信的第 3 部分，[点击此处](../sharding)。</h3>

## 附录 {#附录}

### 附录 A：TPU 内部机制详解 {#附录-a-tpu-内部机制详解}

在这里我们将更深入地探究 TPU 的内部操作。除非另有说明，我们给出的规格都针对 TPU v5p。

### VPU {#vpu}

VPU 是 TPU 的向量算术核心。VPU 由一个二维 SIMD 向量机（即 **VPU**）和一组称为 **VREGs**（向量寄存器）的向量寄存器组成，前者执行逐元素算术运算，例如 vadd（向量加法）或 vmax（逐元素最大值），后者为 VPU 和 MXU 保存数据。

**VREGs：** 每个 TPU v5p 核心有 64 个 32 位 VREGs（TPU v4 中为 32 个），使得每个核心约有 `64 * 8 * 128 * 4 = 256kB` 的 VREG 内存（或整颗芯片的 2 倍，因为我们有两个核心）。一个 TPU v5p 每周期能从 VMEM 载入 3 个寄存器，并每周期向 VMEM 写入 1 个寄存器。

**VPU：** VPU 是一个形状为 `(8, 128)` 的二维向量算术单元，其中 128 这个维度称为通道轴（lane axis），8 这个维度称为子通道轴（sublane axis）。v5 上每一对（通道，子通道）包含 4 个相互独立的标准浮点 ALU。VPU 在其每个 ALU 中用一个周期执行大多数算术指令（如 vadd 或向量加法），延迟为 2 个周期，因此例如在 v5 上，你每周期可以从 VREGs 中把 4 对 f32 值相加。一条典型的 VPU 指令可能形如 `{v2 = vadd.8x128.f32 v0, v1}`，其中 v0 和 v1 是输入 VREGs，v2 是输出 VREG。

所有通道和子通道都以纯 SIMD 方式在每个周期执行同一程序，但每个 ALU 可以执行不同的操作。因此我们可以例如在单个周期内处理 1 个 vadd 和 1 个 vsub，二者各自操作两个完整的 VREGs 并将输出写入第三个 VREG。

**随堂小测 [计算 VPU 吞吐量]：** 利用上述信息，计算一个 TPU v5p 能执行多少向量 FLOPs/s。TPU v5p 的时钟频率约为 1.75GHz。

{% details 点击此处查看答案。 %}

*答案*：每个周期，每个核心可以在 `8 * 128` 个 ALU 上执行 4 条向量指令。这给我们每核心 `8 * 128 * 4` FLOPs/周期，即 `8 * 128 * 4 * 1.75e9 = 7e12 FLOPs/s`。注意这与每核心约 `2e14` 的 MXU FLOPs/s（大约 30 倍）相比小了多少。

{% enddetails %}

**归约：** 一般来说，跨子通道维度的通信或归约比跨通道维度更容易。例如，VPU 支持一种通道内洗牌（intra-lane shuffle）操作，可以在大约一个周期内沿大小为 8 的轴滚动。这可用于在子通道维度上执行高效的归约（只需按 4、2、1 洗牌，再做 3 对逐元素求和）。

跨通道的归约要困难得多，它涉及一个称为转置单元（XLU，cross lane unit）的独立硬件单元，它既慢又相当昂贵。

**与 GPU 的对比：** 对于熟悉 NVIDIA GPU 的读者，VPU 中的每个 ALU 类似于一个 CUDA Core，而单个 VPU 通道类似于一个"线程束调度器"（Warp Scheduler），即通常执行 SIMD 算术的那组 32 个 CUDA Core。通道内的归约相当容易，但如果我们需要跨通道，则至少需要经过 VMEM/XLU/SMEM，这要慢得多。更多细节请参阅 [GPU 章节](../gpus)。

### 标量核心 {#标量核心}

标量核心（scalar core）是 TPU 的控制单元。它获取并分发所有指令，执行从 HBM 到 VMEM 的传输，并且可以被编程来完成标量元数据工作。由于标量核心是单线程的，由此带来的一个副作用是：TPU 的每个核心每周期只能创建一个 DMA 请求。

具体地说，单个标量核心控制着一个 VPU（由 4096 个 ALU 组成）、4 个 MXU、2 个 XLU 以及多个 DMA 引擎。每单位算力所对应的控制极度不均衡，这既是硬件效率的来源，也限制了以任何有趣的方式实现数据相关向量化的能力。

### 附录 B：脉动阵列是如何工作的？ {#附录-b-脉动阵列是如何工作的}

TPU MXU 的核心是一个 `128x128` 的脉动阵列（TPU v6e 上为 `256x256`）。当完全饱和时，脉动阵列每 8 个时钟周期可以执行一次 `bf16[8,128] @ bf16[128,128] -> f32[8,128]`<d-footnote>如果你不熟悉这种记号，它的含义是：将一个元素为 bfloat16 的 `8x128` 矩阵与一个元素为 bfloat16 的 `128x128` 矩阵相乘，并将结果存入一个元素为 float32 的 `8x128` 矩阵。</d-footnote> 乘法。

* 从根本上说，脉动阵列是一个 `128x128`（即 16,384）的二维 ALU 网格，其中每个 ALU 都能执行一次乘加运算。
* 权重（**W**，即 `128x128` 的输入）从上往下传入（称为 RHS），而输入（**X**，即 `8x128` 的输入）从左往右传入（称为 LHS）。

下面是一段将一组权重（蓝色）与一组激活值（绿色）相乘的简化动画。你会注意到权重（RHS）先以对角线方式被部分载入，然后激活值也以对角线方式被喂入。在下面的每一帧中，我们把所有重叠的绿色与蓝色单元相乘，将结果与从上方传入的任意残差求和，然后把结果依次向下传过一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

下面是这个动画的一个更通用的版本，展示了输出从计算中流式输出：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

下面是一张示意图，展示了这如何在多个 RHS 与 LHS 阵列之间流水线化：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

在权重（RHS）和激活值（LHS）载入时，会有一个初始的流水线气泡。在这个初始气泡之后，新的输入和权重可以载入而不会产生额外的气泡。

下面是一段不太理想的 bf16[2, 3] x bf16[3, 3] 矩阵乘法动画，你可以把它想象成用一个 2x3 权重矩阵与批大小为 1、大小为 3 的输入激活值做的 matmul。它相对于前面的幻灯片是旋转过的，输入向右流出而非向下，但你大致能看到其结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以高效地将其流水线化，以相乘大矩阵而不会产生过大的流水线气泡。话虽如此，重要的是我们的矩阵形状要大于 MXU 的边长，该边长通常为 128x128。一些 TPU（自 TPU v3 起）拥有多个 MXU，TPU v3 为 2 个，TPU v4/5 为 4 个，因此我们需要确保分块维度大于 128 × MXU 的数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA) 有一段很好的相关动画。

Trillium（TPU v6e）拥有 `256x256` 的脉动阵列，这意味着它每周期可以执行 4 倍多的 FLOPs。这也意味着你的张量维度需要大到两倍才能充分利用 MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu) 给出了另一个关于固定权重矩阵脉动阵列乘法的精彩动画。
