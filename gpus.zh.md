---
layout: distill
title: "How to Think About GPUs（如何理解 GPU）"
description: "We love TPUs at Google, but GPUs are great too. This chapter takes a deep dive into the world of GPUs – how each chip works, how they're networked together, and what that means for LLMs, especially compared to TPUs. While there are a multitude of GPU architectures from NVIDIA, AMD, Intel, and others, here we will focus on NVIDIA GPUs. This section builds on <a href='https://jax-ml.github.io/scaling-book/tpus/'>Chapter 2</a> and <a href='https://jax-ml.github.io/scaling-book/training'>Chapter 5</a>, so you are encouraged to read them first."
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "Part 11: Conclusion"

next_section_url:
next_section_name: "The End"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "什么是 GPU？"
  - subsections:
    - name: "内存"
    - name: "GPU 规格总结"
    - name: "芯片层面 GPU 与 TPU 的对比"
    - name: "测验 1：GPU 硬件"
  - name: "网络互联"
  - subsections:
    - name: "节点层面"
    - name: "测验 2：GPU 节点"
    - name: "超越节点层面"
    - name: "测验 3：超越节点层面"
  - name: "GPU 上的集合通信如何运作？"
  - subsections:
    - name: "节点内集合通信"
    - name: "跨节点集合通信"
    - name: "测验 4：集合通信"
  - name: "GPU 上 LLM 扩展的屋顶线"
  - subsections:
    - name: "数据并行"
    - name: "张量并行"
    - name: "专家并行"
    - name: "流水线并行"
    - name: "实例"
    - name: "GPU 上 LLM 扩展的要点总结（TLDR）"
    - name: "测验 5：LLM 屋顶线"
  - name: "致谢与延伸阅读"
  - name: "附录"
  - subsections:
    - name: "附录 A：这在 GB200 上如何变化？"
    - name: "附录 B：更多网络细节"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
permalink: /gpus-zh/
sitemap: false
---

## 什么是 GPU？ {#什么是-gpu}

现代的 ML GPU（如 H100、B200）本质上是一堆专门从事矩阵乘法的计算核心（称为**流式多处理器**，即 **SM**），连接在一根高速内存（称为 **HBM**）上。下面是示意图：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>图：</b>H100 或 B200 GPU 的抽象布局示意图。H100 有 132 个 SM，而 B200 有 148 个。我们较为宽泛地使用“线程束调度器（Warp Scheduler）”一词，用来描述一组 32 个 CUDA SIMD 核心<i>以及</i>向它们分派工作的调度器。注意它与 TPU 是多么相似！"%}

每个 SM 与 TPU 的 Tensor Core 类似，拥有一个专用的矩阵乘法核心（不幸地也被称为 **Tensor Core**<d-footnote>GPU 的 Tensor Core 是 SM 的矩阵乘法子单元，而 TPU 的 TensorCore 是包含矩阵乘法单元（MXU）、向量处理单元（VPU）及其他组件的总括单元。</d-footnote>）、一个向量算术单元（称为**线程束调度器（Warp Scheduler）**<d-footnote>NVIDIA 并没有给这个单元起一个好名字，所以我们只是把它当作几个糟糕选项中最好的一个来使用。线程束调度器主要是一个将工作分派给一组 CUDA 核心的单元，但我们在这里用它来描述控制单元以及它所控制的那组核心。</d-footnote>），以及一个快速的片上缓存（称为 **SMEM**）。与 TPU 不同，TPU 至多只有 2 个独立的"Tensor Core"，而现代 GPU 拥有 100 多个 SM（H100 上为 132 个）。这些 SM 每一个都比 TPU 的 Tensor Core 弱得多，但整个系统更加灵活。每个 SM 几乎完全独立，因此 GPU 可以同时进行数百个独立的任务。<d-footnote>尽管 SM 各自独立，但它们常常被迫为达到峰值性能而相互协调，因为它们都共享一个容量受限的二级缓存（L2 cache）。</d-footnote>

让我们更详细地看一下 H100 的 SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>图：</b>H100 SM 的示意图（<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>），展示了 4 个<i>子分区（subpartition）</i>，每个子分区包含一个 Tensor Core、线程束调度器、寄存器文件，以及不同精度的若干组 CUDA Core。底部的“L1 Data Cache”就是 256kB 的 SMEM 单元。B200 看起来类似，但额外增加了大量张量内存（TMEM，Tensor Memory）来喂给庞大的 Tensor Core。"%}

每个 SM 被划分为 4 个完全相同的象限，NVIDIA 称之为 **SM 子分区（subpartition）**，每个子分区包含一个 Tensor Core、16k 个 32 位寄存器，以及一个被称为线程束调度器的 SIMD/SIMT 向量算术单元，其通道（ALU）被 NVIDIA 称为 **CUDA Core**。每个分区最核心的组件可以说是 Tensor Core，它执行矩阵乘法，构成了其 FLOPs/s 的绝大部分，但它并非唯一值得注意的组件。

* **CUDA Core：** 每个子分区包含一组被称为 CUDA Core 的 ALU，用于执行 SIMD/SIMT 向量算术。每个 ALU 通常每个周期可以执行 1 次算术操作，例如 `f32.add`。<d-footnote>较新的 GPU 支持 FMA（Fused-Multiply Add，融合乘加）指令，技术上每个周期执行两次 FLOPs，NVIDIA 毫无顾忌地利用这一点将其公布的规格翻倍。</d-footnote> 每个子分区包含 32 个 fp32 核心（以及较少数量的 int32 和 fp64 核心），它们在每个周期都执行相同的指令。与 TPU 的 VPU 类似，CUDA 核心负责 ReLU、逐点向量运算以及归约（求和）。<d-footnote>历史上，在 Tensor Core 出现之前，CUDA 核心是 GPU 的主要组件，用于渲染，包括光线-三角形相交和着色。在当今的游戏 GPU 上，它们仍承担大部分渲染工作，而 TensorCore 用于上采样（DLSS），使 GPU 能以较低分辨率渲染（像素更少 = 工作量更少）并用 ML 进行上采样。</d-footnote>

* **Tensor Core（TC）：** 每个子分区都有自己的 Tensor Core，它是像 TPU MXU 一样的专用矩阵乘法单元。Tensor Core 占据了 GPU 的 FLOPs/s 的绝大部分（例如在 H100 上，我们有 990 bf16 TC TFLOP/s，而 CUDA 核心只有 66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/) 配合 132 个 SM 以 1.76GHz 运行，意味着每个 H100 TC 每周期可执行 `7.5e12 / 1.76e9 / 4 ~ 1024` 次 bf16 FLOPs，大致是一个 8x8x8 的 matmul。<d-footnote>NVIDIA 并未公开很多 TC 的硬件细节，所以这更像是一种猜测而非确定事实——当然，它并不能说明 TC 是如何实现的。我们知道 V100 每 TC 每周期可执行 256 次 FLOPs，A100 为 512，H100 为 1024，而 B200 的细节尚未公布，但似乎很可能约为每 TC 每周期 2048 次 FLOPs，因为 `2250e12 / (148 * 4 * 1.86e9)` 约为 2048。更多细节在此<a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>确认</a>。</d-footnote>
  * 与 TPU 类似，GPU 可以以更高的吞吐量执行更低精度的 matmul（例如 H100 的 fp8 FLOPs/s 是 fp16 的 2 倍）。低精度训练或服务可以显著更快。
  * 自 Volta 以来，每一代 GPU 的 TC 规模都比上一代更大（[关于此的好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。到了 B200，TC 已经变得如此之大，以至于其输入无法再放进 SMEM，因此 B200 引入了一个新的内存空间，称为 TMEM。<d-footnote>在 Ampere 中，Tensor Core 可以由单个线程束喂入，而在 Hopper 中需要一个完整的 SM（warpgroup），在 Blackwell 中则由 2 个 SM 喂入。在 Blackwell 中 matmul 也变得如此之大，以至于其参数（具体而言是累加器）不再能放进寄存器内存/SMEM，因此 Blackwell 增加了 TMEM 来解决这个问题。</d-footnote>

**CUDA 核心比 TPU 的 VPU 更灵活：** 自 V100 以来的 GPU CUDA 核心采用所谓的 SIMT（*Single Instruction Multiple Threads*，单指令多线程）编程模型，而 TPU 的是 SIMD（*Single Instruction Multiple Data*，单指令多数据）模型。与 TPU VPU 中的 ALU 一样，子分区内的 CUDA 核心在每个周期必须执行相同的操作（例如，如果一个核心在将两个浮点数相加，那么该子分区内的每个其他 CUDA 核心也必须这样做）。但与 VPU 不同的是，每个 CUDA 核心（或 CUDA 编程模型中的"线程"）都有自己的指令指针，可以独立地_编程_。当同一个线程束中的两个线程被指示执行不同的操作时，你实际上会执行_两个_操作，并将不需要执行该分支操作的那些核心屏蔽掉。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图：</b>一组线程中线程束发散（warp divergence）的示例（<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>）。白色区域表示至少部分物理 CUDA 核心的停顿"%}

这使得在线程层面可以进行灵活的编程，但代价是：如果线程束过于频繁地发散，性能会悄无声息地下降。线程在它们能访问的内存方面也可以更灵活；VPU 只能操作连续的内存块，而 CUDA 核心可以访问共享寄存器中的单个浮点数，并维护每线程的状态。

**CUDA 核心的调度也更加灵活：** SM 的运行有点像多线程 CPU，因为它们可以并发地"调度"许多程序（**线程束（warp）**）（每 SM 最多 64 个），但每个_线程束调度器（Warp Scheduler）_在每个时钟周期只执行单个程序。<d-footnote>在某个 SM 上调度的线程束被称为"驻留（resident）"。</d-footnote> 线程束调度器会在活跃线程束之间自动切换，以隐藏像内存加载这样的 I/O 操作。相比之下，TPU 通常是单线程的。

### 内存 {#内存}

除了计算单元，GPU 还有一套内存层次结构，最大的是 HBM（GPU 主内存），然后是一系列较小的缓存（二级缓存（L2）、L1/SMEM、张量内存（TMEM）、寄存器内存）。

* **寄存器：** 每个子分区拥有自己的寄存器文件，在 H100/B200 上包含 16,384 个 32 位字（`4 * 16384 * 4 = 256kiB` 每 SM），可由 CUDA 核心访问。
  * 每个 CUDA 核心一次最多只能访问 256 个寄存器，所以尽管我们每 SM 可以调度多达 64 个"驻留线程束"，但如果每个线程使用 256 个寄存器，你一次只能容纳 8 个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM（L1 缓存）：** 每个 SM 都有自己 256kB 的片上缓存，称为 SMEM，它既可以由程序员作为"共享内存"控制，也可以由硬件用作片上缓存。SMEM 用于存储激活值和 TC matmul 的输入。

* **二级缓存（L2 Cache）：** 所有 SM 共享<d-footnote>严格来说，L2 缓存被分成两半，因此在 H100 上，一半的 SM 各自能访问 25MB。连接这两半之间有一条链路，但带宽较低。</d-footnote> 一个相对较大的约 50MB 的 L2 缓存，用于减少对主内存的访问。
  * 这在大小上与 TPU 的向量内存（VMEM）相近，但它**慢得多**，并且不是由程序员控制的。这导致了一点"幽灵般的远处作用（spooky action at a distance）"，程序员需要修改内存访问模式以确保 L2 缓存被良好利用。<d-footnote>L2 缓存跨所有 SM 共享这一事实，实际上迫使程序员以相当协调的方式运行 SM，尽管原则上它们是独立的单元。</d-footnote>
  * NVIDIA 并未公布其芯片的 L2 带宽，但有人[测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)出约为 5.5TB/s。这大约是 HBM 带宽的 1.6 倍，但它是全双工的，因此有效的双向带宽接近 3 倍。相比之下，TPU 的向量内存（VMEM）大 2 倍*而且*带宽高得多（约为 40TB/s）。

* **HBM：** GPU 的主内存，用于存储模型权重、梯度、激活值等。
  * HBM 的大小从 Volta 的 32GB 大幅增长到 Blackwell（B200）的 192GB。
  * 从 HBM 到 CUDA Tensor Core 的带宽称为 HBM 带宽或内存带宽，在 H100 上约为 3.35TB/s，在 B200 上约为 9TB/s。

### GPU 规格总结 {#gpu-规格总结}

下面是近期各型号 GPU 的规格总结。给定 GPU 的不同变体之间，SM 数量、时钟频率和 FLOPs 略有不同。下面是内存容量数据：

|  GPU  | 代次 |   时钟频率   | SM 数/芯片 | SMEM 容量/SM | 二级缓存容量/芯片 | HBM 容量/芯片 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100  |   Volta    | 1.25GHz/1.38GHz |    80    |       96kB       |       6MB        |       32GB        |
| A100  |   Ampere   | 1.10GHz/1.41GHz |   108    |      192kB       |       40MB       |       80GB        |
| H100  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       80GB        |
| H200  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       141GB       |
| B200  | Blackwell  |        ?        |   148    |      256kB       |      126MB       |       192GB       |

所有代次每 SM 都有 256kB 的寄存器内存。Blackwell 还额外增加了每 SM 256kB 的 TMEM。下面是每颗芯片的 FLOPs 和带宽数据：

|  GPU  | 代次 | HBM 带宽/芯片 | FLOPs/s/芯片（bf16/fp16） | FLOPs/s/芯片（fp8/int8） | FLOPs/s/芯片（fp4） |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

我们排除了 B100，因为它并未量产。<d-footnote>虽然 NVIDIA 推出过 B100 这一代，但它们只短暂地销售和生产过，据称是因为设计缺陷导致它们无法接近其宣称的规格运行。由于散热和功耗问题，它们难以在不降频的情况下达到峰值 FLOPs。</d-footnote> 有些规格会略微取决于 GPU 的确切版本，因为 NVIDIA GPU 不像 TPU 那样标准化。

下面是一份很有用的对照表，比较 GPU 与 TPU 的组件：

|              GPU              |     TPU     |              它是什么？              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器（SM） | Tensor Core | 包含其他单元的核心"单元" |
|        线程束调度器         |     VPU     |      SIMD 向量算术单元      |
|           CUDA Core           |   VPU ALU   |               SIMD ALU                |
|        SMEM（L1 缓存）        |    向量内存（VMEM）     |       快速的片上缓存内存       |
|          Tensor Core          |     MXU     |      矩阵乘法单元       |
|        HBM（即 GMEM）         |     HBM     |  高带宽、高容量内存  |

### 芯片层面 GPU 与 TPU 的对比 {#芯片层面-gpu-与-tpu-的对比}

GPU 最初是为了渲染电子游戏，但自从深度学习在 2010 年代兴起以来，它们越来越像专用的矩阵乘法机器——换句话说，越来越像 TPU。<d-footnote>在深度学习热潮之前，GPU（"Graphics Processing Units"，图形处理单元）做的是图形——主要用于电子游戏。电子游戏用数百万个小三角形表示物体，游戏将这些三角形渲染（或"栅格化"）成一张 2D 图像，每秒在屏幕上显示 30-60 次（这个频率称为帧率）。栅格化涉及将这些三角形投影到相机的坐标系中，并计算哪些三角形与哪些像素重叠，每秒数十亿次。可以想象，这非常昂贵，而且这还只是开始。你接着必须结合可能多个半透明三角形的颜色来为每个像素着色，这些三角形与光线相交。GPU 被设计成极快地进行这些操作，同时着眼于通用性；你需要同时运行许多不同的 GPU 工作负载（称为"着色器（shader）"），而没有单一操作占主导。因此，面向消费者的图形 GPU 可以做矩阵乘法，但这不是它们的主要功能。</d-footnote> 在某种程度上，这段历史解释了现代 GPU 为何是现在的样子。它们并非纯粹为 LLM 或 ML 模型而设计，而是作为通用加速器，硬件追求一种"通用性"，这既可能是福也可能是祸。GPU 在应用于新任务时往往更能"即插即用"，并且比 TPU 更不依赖优秀的编译器。但这也使得它们更难推理，或更难发挥出屋顶线（roofline）性能，因为如此多的编译器特性都可能成为瓶颈。

**GPU 更加模块化。** TPU 有 1-2 个大的 Tensor Core，而 GPU 有数百个小的 SM。同样，每个 TC 有一个由 4 个独立可编程的 8x128 单元（共 4096 个 ALU）组成的单一大 VPU；相比之下，一个 H100 有 132 * 4 = 528 个独立 SIMD 单元，每个 32 宽（共 16k 个 ALU）。下面是一个突出这一点的 GPU 与 TPU 的一一对比：

|              GPU              |           TPU            | H100 数量 | TPU v5p 数量 |
| :---------------------------: | :----------------------: | :----: | :-------: |
| SM（流式多处理器） |       Tensor Core        |  132   |     2     |
|        线程束调度器         |        VPU 槽位         |  528   |     8     |
|        SMEM（L1 缓存）        |           向量内存（VMEM）           |  32MB  |   128MB   |
|          寄存器           | 向量寄存器（VReg） |  32MB  |   256kB   |
|          Tensor Core          |           MXU            |  528   |     8     |

这种模块性上的差异一方面使得 TPU 更便宜、更易理解，但也给编译器带来了更多做好正确优化的负担。因为 TPU 只有一个控制线程，且只支持向量化的、VPU 宽度的指令，编译器需要手动对所有内存加载和 MXU/VPU 工作进行流水线化以避免停顿。而 GPU 程序员可以直接启动几十个不同的 kernel，每个运行在一个完全独立的 SM 上。另一方面，这些 kernel 可能因为抖动 L2 缓存或未能合并内存加载而获得糟糕的性能；因为硬件控制了如此多的运行时行为，很难推理背后发生了什么。结果，TPU 往往能用更少的功夫更接近峰值屋顶线性能。

**从历史上看，单颗 GPU 比可类比的 TPU 更强大（也更贵）：** 单颗 H200 的 FLOPs/s 接近 TPU v5p 的 2 倍，HBM 的 1.5 倍。与此同时，Google Cloud 上的标价约为 H200 每小时 \\$10，而 TPU v5p 为每小时 \\$4。TPU 通常比 GPU 更依赖将多个芯片联网在一起。

**TPU 拥有多得多的快速缓存内存。** TPU 的向量内存（VMEM）也比 GPU 的 SMEM（+TMEM）大得多，而且这块内存可以用来存储权重和激活值，使它们能被极其快速地加载和使用。如果可以持续地将模型权重存储或预取到向量内存（VMEM）中，这可以让 TPU 在 LLM 推理上更快。

### 测验 1：GPU 硬件 {#测验-1-gpu-硬件}

下面是一些用来练习上面部分内容的问题。答案已提供，但最好先试着自己回答，手里拿着纸笔。

**问题 1 [CUDA 核心]：** 一颗 H100 有多少个 fp32 CUDA 核心（ALU）？B200 呢？这与 TPU v5p 中独立 ALU 的数量相比如何？

{% details 点击此处查看答案。 %}

**答案：** 一颗 H100 有 132 个 SM，每个 SM 有 4 个子分区，每个子分区包含 32 个 fp32 CUDA 核心，因此我们有 `132 * 4 * 32 = 16896` 个 CUDA 核心。一颗 B200 有 `148` 个 SM，因此总共 `18944` 个。一颗 TPU v5p 有 2 个 TensorCore（通常通过大核（Megacore）连接），每个有一个 VPU，包含 (8, 128) 个通道，每个通道有 4 个独立 ALU，因此 `2 * 4 * 8 * 128 = 8192` 个 ALU。这大约是 H100 向量通道数量的一半，运行在大致相同的频率下。

{% enddetails %}

**问题 2 [向量 FLOPs 计算]**：单颗 H100 有 132 个 SM，运行在 1.59GHz 的时钟频率（最高 1.98GHz 加速）。假设每个 ALU 每周期能执行一次向量操作。每秒能做多少次向量 fp32 FLOPs？加速后呢？这与 matmul FLOPs 相比如何？

{% details 点击此处查看答案。 %}

**答案：** `132 * 4 * 32 * 1.59e9 = 26.9TFLOPs/s`。加速后为 33.5 TFLOPs/s。这是[规格表](https://www.nvidia.com/en-us/data-center/h100/)中数字的一半，因为技术上我们可以在一个周期内完成一次 FMA（fused-multiply-add，融合乘加），算作两次 FLOPs，但这在大多数情况下并无用处。我们可以做到 990 bfloat16 matmul TFLOPs/s，因此忽略 FMA 的话，Tensor Core 约多做 30 倍的 FLOPs/s。

{% enddetails %}

**问题 3 [GPU matmul 强度]：** H100 上 fp16 matmul 的峰值强度是多少？B200 呢？fp8 呢？*我们所说的强度是指 matmul FLOPs/s 与内存带宽的比值。*

{% details 点击此处查看答案。 %}

**答案：** 对于 H100，我们有峰值 990e12 fp16 FLOPs 和 3.35e12 字节/秒的带宽。因此临界强度为 `990e12 / 3.35e12 = 295`，与 TPU 中的 240 相当接近。对于 B200，为 `2250e12 / 8e12 = 281`，非常接近。这意味着，与 TPU 类似，我们需要大约 280 的批大小才能在 matmul 中达到算力受限。

对于 H100 和 B200，fp8 FLOPs 都是恰好 2 倍，因此峰值强度也翻倍到 590 和 562，尽管从某种意义上说它保持不变，如果我们考虑到权重很可能也会以 fp8 加载的话。

{% enddetails %}

**问题 4 [Matmul 运行时间]：** 利用问题 3 的答案，在一个单独的 B200 上，`fp16[64, 4096] * fp16[4096, 8192]` 这个 matmul 预计需要多长时间？`fp16[512, 4096] * fp16[4096, 8192]` 呢？

{% details 点击此处查看答案。 %}

根据上面，我们知道在批大小低于 281 个词元时我们会受通信限制。因此第一个纯粹受带宽限制。我们读取或写入 $2BD + 2DF + 2BF$ 字节（`2*64*4096 + 2*4096*8192 + 2*64*8192=69e6`），带宽为 `8e12` 字节/秒，因此大约需要 `69e6 / 8e12 = 8.6us`。在实践中我们可能只能拿到总带宽的一部分，因此可能接近 10-12us。当我们增大批大小时，我们完全受算力限制，因此我们预计 `T=2*512*4096*8192/2.3e15=15us`。我们再次只预期拿到总 FLOPs 的一部分，因此可能看到接近 20us。

{% enddetails %}

**问题 5 [L1 缓存容量]：** 一颗 H100 的总 L1/SMEM 容量是多少？寄存器内存呢？与 TPU 向量内存（VMEM）容量相比如何？

{% details 点击此处查看答案。 %}

**答案：** 每 SM 有 256kB SMEM 和 256kB 寄存器内存，因此每种大约 33MB（`132 * 256kB`）。加起来总共约 66MB。这大约是现代 TPU 向量内存（VMEM）120MB 的一半，尽管 TPU 总共只有 256kB 寄存器内存！TPU 向量内存（VMEM）的延迟低于 SMEM 延迟，这也是为什么 TPU 上的寄存器内存没那么关键的原因之一（溢出和填回向量内存（VMEM）很便宜）。

{% enddetails %}

**问题 6 [计算 B200 时钟频率]：** NVIDIA 在[此处](https://resources.nvidia.com/en-us-blackwell-architecture)报告，一颗 B200 可以执行 80TFLOPs/s 的向量 fp32 计算。已知每个 CUDA 核心在 FMA（融合乘加）操作中每周期可执行 2 次 FLOPs，估计其峰值时钟周期。

{% details 点击此处查看答案。 %}

**答案：** 我们知道有 148 * 4 * 32 = 18944 个 CUDA 核心，因此我们可以做到 `18944 * 2 = 37888` 次 FLOPs/周期。因此 `80e12 / 37888 = 2.1GHz`，这是一个较高但合理的峰值时钟频率。B200 通常采用液冷，因此更高的时钟周期更合理。

{% enddetails %}

**问题 7 [估算 H100 加法运行时间]：** 利用上面的数字，计算在单颗 H100 上将两`fp32[N]`向量相加应该需要多长时间。分别计算 $T_\text{math}$ 和 $T_\text{comms}$。这个操作的算术强度是多少？如果你能访问，也尝试在 PyTorch 或 JAX 中对 `N = 1024` 和 `N=1024 * 1024 * 1024` 运行这个操作。结果如何比较？

{% details 点击此处查看答案。 %}

**答案：** 首先，将两个`fp32[N]`向量相加执行 N 次 FLOPs，需要加载 `4 * N * 2` 字节并写回 4 * N 字节，总共 `3 * 4 * N = 12N`。计算其比值，我们得到 `总 FLOPs / 总字节数 = N / 12N = 1 / 12`，相当糟糕。

正如我们上面计算的，忽略 FMA 的话，我们可以做到大约 33.5 TFLOPs/s 的加速值。但这只有在所有 CUDA 核心都被使用时才行。对于 `N = 1024`，我们最多只能使用 1024 个 CUDA 核心或 8 个 SM，这会更慢（假设我们受算力限制，大约慢 16 倍）。我们还有 3.35e12 字节/秒的内存带宽。因此我们的峰值硬件强度为 `33.5e12 / 3.35e12 = 10`。<d-footnote>值得注意的是，这个强度在近期 GPU 代次中保持不变。对于 H100 是 33.5 / 3.5，对于 B200 是 80 / 8。原因尚不清楚，但这是一个有趣的观察。</d-footnote> 所以我们将极其受通信限制。因此我们的运行时间就是

$$T = \max(T_\text{comms}, T_\text{math}) = \frac{12 \cdot N}{\text{3.35e12}} = \frac{N}{\text{2.8e11}}$$

对于 `N = 65,536`，这大约是 0.23us。在实践中我们看到在 JAX 中约为 1.5us，这没问题，因为我们预期这里会极度受延迟限制。对于 `N = 1024 * 1024 * 1024`，我们的屋顶线约为 3.84ms，我们看到 4.1ms，这很好！

{% enddetails %}

## 网络互联 {#网络互联}

网络互联是 GPU 和 TPU 差异最大的领域之一。正如我们所见，TPU 连接在 2D 或 3D 环面上，每个 TPU 只与它的邻居相连。这意味着两个 TPU 之间发送消息必须经过每个中间的 TPU，并迫使我们只在网格上使用统一的通信模式。虽然在某些方面不便，但这也意味着每个 TPU 的链路数量是恒定的，我们可以将 TPU "pod" 扩展到任意大而不损失带宽。

另一方面，GPU 使用更传统的分层树状交换网络。称为**节点（node）**的一组 8 个 GPU（GB200 最多 72 个<d-footnote>node 这个词被重载了，可以表示两件事：NVLink 域（即一组通过 NVLink 互联完全连接的 GPU），或连接到单个 CPU 主机的那组 GPU。在 B200 之前，这两者通常是相同的，但在 GB200 NVL72 中，我们有一个包含 72 个 GPU 的 NVLink 域，但连接到每个主机的仍然只有 8 个 GPU。我们在这里用 node 一词指代 NVLink 域，但这存在争议。</d-footnote>）通过称为 NVLink 的高带宽互联在 1 跳内互相连接，而这些节点又通过附着在每颗 GPU 上的网卡（NIC）以较低带宽的 InfiniBand（IB）或以太网网络连接成更大的单元（称为**可扩展单元（SU，Scalable Unit）**）。这些单元进而可以通过更高级别的交换机连接成任意大的单元。

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图：</b>典型 H100 网络的示意图。一组 8 个 GPU 通过 NVSwitches（也称为 NVLink 交换机）连接成一个节点或 NVLink 域，这些节点又通过交换式 InfiniBand 结构互相连接。在 NVLink 域中，H100 各有约 450GB/s 的出口带宽，每个节点有 400GB/s 进入 IB 网络的出口带宽。"%}

### 节点层面 {#节点层面}

一个 GPU 节点是一个小单元，通常是 8 个 GPU（GB200 最多 72 个），通过全连接、全带宽、低延迟的 NVLink 互联连接。<d-footnote>NVLink 被描述为有点像加强版的 PCIe 连接，具有低延迟和协议开销，但并非为可扩展性或容错而设计，而 InfiniBand 更像以太网，为较大的、有损的网络而设计。</d-footnote> 每个节点包含几个高带宽 NVSwitch，在本地所有 GPU 之间交换数据包。节点层面的实际拓扑随时间变化很大，包括每节点的交换机数量，但对于 H100，我们有每节点 4 个 NVSwitch，GPU 以 `5 + 4 + 4 + 5` 的链路模式连接到它们，如下图所示：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b>从 Pascal（P100）开始的节点（即 NVLink 域）示意图。自 Volta（V100）以来，我们在一个节点内使用一组交换机实现了全连接。H100 节点有 4 个 NVSwitch，以 25GB/s 链路连接到全部 8 个 GPU。"%}

对于 Hopper 代次（NVLink 4.0），每个 NVLink 链路有 25GB/s 的全双工<d-footnote>这里的全双工是指每个方向 25GB/s，两个方向相互独立。你可以在该链路上总共发送 50GB/s，但每个方向最多 25GB/s。</d-footnote> 带宽（B200 为 50GB/s），由此得到每颗 GPU 进入网络的 `18 * 25=450GB/s` 全双工带宽。巨大的 NVSwitch 最多有 64 个 NVLink 端口，意味着一个带 4 个交换机的 8xH100 节点可以处理高达 `64 * 25e9 * 4=6.4TB/s` 的带宽。下面是这些数字随 GPU 代次变化的概览：

| NVLink 代次 | NVSwitch 代次 | GPU 代次 | NVLink 带宽（GB/s，全双工） | 每 GPU 的 NVLink 端口数 | 节点 GPU 到 GPU 带宽（GB/s，全双工） | 节点大小（NVLink 域） | 每节点 NVSwitch 数 |
| :--------: | :----------: | :------------: | :----------------------------------: | :----------------: | :------------------------------------------: | :-----------------------: | :-----------------: |
|  **3.0**   |   **2.0**    |     Ampere     |                  25                  |         12         |                     300                      |             8             |          6          |
|  **4.0**   |   **3.0**    |     Hopper     |                  25                  |         18         |                     450                      |             8             |          4          |
|  **5.0**   |   **4.0**    |   Blackwell    |                  50                  |         18         |                     900                      |           8/72            |        2/18         |

Blackwell（B200）有 8 个 GPU 的节点。GB200 NVL72 支持更大的 72 个 GPU 的 NVLink 域。我们展示了 8 个和 72 个 GPU 系统两者的细节。

### 测验 2：GPU 节点 {#测验-2-gpu-节点}

下面是一些关于网络互联的问答练习。我发现把这些亲手算出来特别有用，因为它们迫使你梳理实际的通信模式。

**问题 1 [H100 节点的总带宽]：** 在一个带 4 个交换机的 8xH100 节点中，每节点总共有多少带宽？*提示：* 同时考虑 NVLink 和 NVSwitch 带宽。

{% details 点击此处查看答案。 %}

**答案：** 我们有 Gen4 的 4 个 NVSwitch，每个有 `64 * 25e9=1.6TB/s` 的单向带宽。这将给我们交换机层面 `4 * 1.6e12=6.4e12` 的带宽。然而，注意每颗 GPU 最多只能处理 450GB/s 的单向带宽，因此我们最多有 `450e9 * 8 = 3.6TB/s` 带宽。由于这个更小，峰值带宽是 3.6TB/s。

{% enddetails %}

**问题 2 [对分带宽]**：对分带宽（bisection bandwidth）定义为任何对网络进行均匀二分后可用的最小带宽。换句话说，如果我们把一个网络分成相等的两半，有多少带宽穿过这两半？你能计算一个 8x H100 节点的对分带宽吗？*提示：* 对分带宽通常包括两个方向的流量。

{% details 点击此处查看答案。 %}

**答案：** 任何均匀的二分都会有每半 4 个 GPU，每个 GPU 可以向另一半出口 `4 * 450GB/s`。计入两个方向，这给了我们 `8 * 450GB/s` 字节穿过该二分，即 3.6TB/s 的对分带宽。这是 NVIDIA 报告的，例如[此处](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

{% enddetails %}

**问题 3 [AllGather 代价]**：给定一个 B 字节的数组，在一个 8xH100 节点上一次（受吞吐量限制的）AllGather 需要多长时间？对 bf16[D<sub>X</sub>, F] 做一下计算，其中 `D=4096`，`F=65,536`。*建议在回答前阅读 TPU 集合通信的[章节](https://jax-ml.github.io/scaling-book/sharding/)。先在这里想清楚，但关于集合通信我们接下来会讲更多。*

{% details 点击此处查看答案。 %}

**答案：** 每颗 GPU 可以出口 450GB/s，每颗 GPU 有 $B / N$ 字节（其中 `N=8`，即节点大小）。我们可以想象每个节点将其字节一次一个地发送给其他 $N - 1$ 个节点，总共 (N - 1) 轮，每轮 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$，即 $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$。这近似为 $B / W_\text{uni}$ 或 $B / \text{450e9}$。

对于给定的数组，我们有 `B = 4096 * 65536 * 2 = 536e6` 字节，因此总时间为 `536e6 * (8 - 1) / (8 * 450e9) = 1.04ms`（或者用近似值为 `536e6 / 450e9 = 1.19ms`）。这可能受延迟限制，因此实践中可能比这更长（实践中约为 1.5ms）。

{% enddetails %}

## 超越节点层面 {#超越节点层面}

在节点层面之上，GPU 网络的拓扑不那么标准化。NVIDIA 发布了一个[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，使用 InfiniBand 连接比单个节点更多的 GPU，但客户和数据中心提供商可以自由地根据需求定制。<d-footnote>例如，Meta 在一个与这个描述显著不同的数据中心网络上训练了 LLaMA-3，使用了以太网、三层交换结构，以及顶层过度订阅的交换机。</d-footnote>

下面是一个参考的 1024 GPU H100 系统的示意图，底排每个方框是一个单独的 8xH100 节点，带 8 个 GPU、8 个 400Gbps CX7 网卡（每 GPU 一个）和 4 个 NVSwitch。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图：</b>参考的 1024 H100 DGX SuperPod 示意图，包含 128 个节点（有时 127 个），每个节点有 8 个 H100 GPU，连接到 InfiniBand 横向扩展（scale-out）网络。每组 32 个节点（256 个 GPU）称为“可扩展单元（Scalable Unit）”或 SU。叶子（leaf）和脊（spine）IB 交换机提供了足够的带宽，以实现节点间的全对分带宽。"%}

**可扩展单元（Scalable Units）：** 每组 32 个节点称为一个"可扩展单元"（SU），位于一组 8 个叶子 InfiniBand 交换机之下。这个 SU 有 256 个 GPU，每节点 4 个 NVSwitch 和 8 个 InfiniBand 叶子交换机。图中所示所有布线都是 InfiniBand NDR（50GB/s 全双工），使用 64 端口 NDR IB 交换机（也是每端口 50GB/s）。*注意 IB 交换机的带宽是 NVSwitch 的 2 倍（64 个端口，每个 400 Gbps 链路）。*

**SuperPod：** 整个 SuperPod 进而用 16 个顶层"脊（spine）"IB 交换机连接这 4 个 SU，得到 1024 个 GPU，带有 512 个节点级 NVSwitch、32 个叶子 IB 交换机和 16 个脊 IB 交换机，总共 512 + 32 + 16 = 560 个交换机。叶子交换机以每组 32 个节点连接到节点，因此每组 256 个 GPU 有 8 个叶子交换机。所有叶子交换机都连接到所有脊交换机。

**我们有多少带宽？** InfiniBand 网络（称为"横向扩展网络"）的整体拓扑是一个**胖树（fat tree）**，其布线和交换机保证了节点层面之上的全对分带宽（这里为 400GB/s）。这意味着如果我们把节点平分为两半，每个节点可以同时向另一分区的某个节点出口 400GB/s。更直白地说，这意味着我们在横向扩展网络中应该有一个大致恒定的 AllReduce 带宽！虽然它可能不是这样实现的，但你可以想象在横向扩展网络中对任意多个节点做一个环形归约，因为你总能构造一个包含所有节点的环。

| 层级 | GPU 数 | 每单元交换机数 | 交换机类型 | 每单元带宽（TB/s，全双工） | GPU 到 GPU 带宽（GB/s，全双工） | 胖树带宽（GB/s，全双工） |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点  |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| 叶子  |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| 脊  |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

相比之下，一个 TPU v5p 每链路约有 90GB/s 的出口带宽，或沿 3D 环面所有轴的 540GB/s 出口。这不是点对点的，因此只能用于受限的、统一的通信模式，但它仍然给了我们高得多的 TPU 到 TPU 带宽，可以扩展到任意大的拓扑（至少到 8960 个 TPU）。

GPU 交换结构在理论上可以通过添加额外的交换机或间接层扩展到任意大小，代价是额外的延迟和昂贵的网络交换机。

<p markdown=1 class="takeaway">**要点**：在一个 H100 节点内，我们有每颗 GPU 450GB/s 的全胖树带宽，而在节点之外，这下降到节点到节点的 400GB/s。这最终对通信原语至关重要。</p>

**GB200 NVL72：** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，将 72 个 GPU 组合在单个 NVLink 域中，具有全 900GB/s 的 GPU 到 GPU 带宽。这些域进而可以连接成更大的 SuperPod，其 IB 胖树带宽成比例地更高（9 倍）。下面是该拓扑的示意图：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>一个 576 GPU 的 GB200 DGX SuperPod 示意图。底层的每个机柜包含 72 个 GB200 GPU。"%}

计算从单个节点（上面的橙线）的出口带宽，我们有 `4 * 18 * 400 / 8 = 3.6TB/s` 到叶子层面的带宽，是 H100 的 9 倍（正如该节点包含 9 倍多的 GPU）。这意味着关键的节点出口带宽要高得_多_，我们的跨节点集合通信带宽实际上可能_低于_节点内。

见[附录 A](#附录-a-这在-gb200-上如何变化)了解更多讨论。

|  节点类型  | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**要点**：GB200 NVL72 SuperPod 大幅增加了节点大小以及给定节点的出口带宽，这显著改变了我们的屋顶线。</p>

### 测验 3：超越节点层面 {#测验-3-超越节点层面}

**问题 1 [胖树拓扑]：** 使用上面的 DGX H100 示意图，计算整个 1024 GPU pod 在节点层面的对分带宽。证明每条链路的带宽选择是为了确保全对分带宽。*提示：确保同时计算链路带宽和交换机带宽。*

{% details 点击此处查看答案。 %}

**答案：** 让我们逐个组件来算：

* 首先，每个节点有 8x400Gbps NDR IB 线缆连接到叶子交换机，给每个节点 `8 * 400 / 8 = 400 GB/s` 到叶子的带宽。我们有 8 个叶子交换机，每个 3.2TB/s（64 个 400 GBps 链路），但我们只能使用 64 个端口中的 32 个从 SU 进入，因此是 `32 * 400 / 8 = 12.8TB/s` 对应 32 个节点，再次恰好是 400GB/s。
* 然后在脊层面，我们有 `8 * 16 * 2` 个 400Gbps NDR IB 线缆将每个 SU 连接到脊，给每个 SU `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` 到叶子的带宽。同样，这是每节点 400GB/s。我们有 16 个脊交换机，每个 3.2TB/s，给出 `16 * 3.2 = 51.2 TB/s`，对应 128 个节点再次是 400GB/s。

因此，如果我们以任何方式对节点进行二分，它们之间每节点都会有 400GB/s 的带宽。每个组件都恰好具有确保胖树所需的带宽。

{% enddetails %}

**问题 2 [扩展到更大的 DGX pod]：** 假设我们想在 2048 个 GPU 而不是 1024 个上训练。修改上面的 DGX 拓扑来处理这个问题，最简单/最好的方式是什么？4096 呢？*提示：没有唯一正确答案，但尽量压低成本。注意链路容量。[这](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)份文档可能有帮助。*

{% details 点击此处查看答案。 %}

**答案：** 一种选择是保持 SU 结构不变（8 个交换机下的 32 个节点），只是添加更多带有更多顶层交换机的 SU。我们需要 2 倍的脊交换机，因此会有 8 个 SU，带 32 个脊交换机，给出足够的带宽。

这样做的一个问题是每叶子交换机只有 64 个端口，而在上面的示意图中我们已经全部用完了。但改为每脊用 1x 400 Gbps NDR 线缆而不是 2x 很容易，这给出相同的总带宽，但节省了一些端口。

对于 4096 个 GPU，我们实际上用光了端口，因此需要添加另一层间接，也就是说，在层次结构中再加一层。NVIDIA 称之为"核心交换机（core switches）"，并用 128 个脊交换机和 64 个核心交换机构建了一个 4096 GPU 集群。你可以算一下，证明这给出了足够的带宽。

{% enddetails %}

## GPU 上的集合通信如何运作？ {#gpu-上的集合通信如何运作}

GPU 可以执行与 TPU 完全相同的集合通信：ReduceScatter、AllGather、AllReduce 和 AllToAll。与 TPU 不同的是，它们的运作方式取决于是在节点层面（通过 NVLink）还是在节点之上（通过 InfiniBand）执行。这些集合通信由 NVIDIA 在 [NVSHMEM](https://developer.nvidia.com/nvshmem) 和 [NCCL](https://developer.nvidia.com/nccl)（读作"nickel"）库中实现。NCCL 在此处[开源](https://github.com/NVIDIA/nccl)。虽然 NCCL 根据延迟要求/拓扑使用多种实现（[细节](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），但从这里开始，我们将讨论一个在交换树结构上理论最优的模型。

### 节点内集合通信 {#节点内集合通信}

**AllGather 或 ReduceScatter：** 对于节点层面的 AllGather 或 ReduceScatter，你可以像 TPU 一样围绕一个环执行它们，在每个跳上使用完整的 GPU 到 GPU 带宽。任意排列 GPU 的顺序，并使用完整的 GPU 到 GPU 带宽将数组的一部分绕环发送。<d-footnote>你也可以想象每颗 GPU 将其大小为 $\text{bytes} / N$ 的块发送给其他 $N - 1$ 颗 GPU 中的每一个，总共通信 $(N - 1) * N * bytes / N$ 字节，这给出了相同的答案。</d-footnote> 每跳的代价是 $T_\text{hop} = \text{bytes} / (N * \text{GPU 出口带宽})$，因此总代价为

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU 出口带宽}} \rightarrow \frac{\text{bytes}}{\text{GPU 出口带宽}}$$

你会注意到这与 TPU 上完全相同。对于 AllReduce，你可以像往常一样将 RS + AG 组合，代价为两倍。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图：</b>带宽最优的 1D 环形 AllGather 算法。对于 B 字节，这在顶层交换机上发送 B / X 字节 X - 1 次。"%}

如果你担心延迟（例如你的数组非常小），你可以做一个树形归约，在成对的 2、然后 4、然后 8 之间 AllReduce，总共 $\log(N)$ 跳而不是 $N - 1$ 跳，尽管总代价仍然相同。

<p markdown=1 class="takeaway">**要点：** 在单个节点内对 B 字节数组进行 AllGather 或 ReduceScatter 的代价约为 $T_\text{comms} = B * (8 - 1) / (8 * W_\text{GPU 出口}) \approx B / W_\text{GPU 出口}$。在 H100 上理论上约为 $B  / \text{450e9}$，在 B200 上约为 $B / \text{900e9}$。除非启用网内归约（in-network reductions），否则 AllReduce 的代价是这个值的两倍。</p>

<b markdown=1 style="color: #57cf57;">小测验 1 [AllGather 耗时]：</b> 使用具有 450 GB/s 全双工带宽的 8xH100 节点，AllGather(bf16[B<sub>X</sub>, F]) 需要多长时间？令 $B=1024$，$F=16,384$。

{% details 点击此处查看答案。 %}

**答案：** 我们总共有 $2 \cdot B \cdot F$ 字节，带 450e9 单向带宽。这大约需要 $T_\text{comms} = (2 \cdot B \cdot F) / \text{450e9}$，或更精确地 $(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。使用给定的值，这给我们大约 $(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，或更精确地 $\text{65us}$。

{% enddetails %}

**AllToAll：** 节点内的 GPU 具有全连接，这使得 AllToAll 相当容易。每颗 GPU 直接发送到目标节点。在节点内，对于 B 字节，每颗 GPU 有 $B / N$ 字节，并向 $N - 1$ 个目标节点各发送 $(B / N^2)$ 字节，总共

$$T_\text{AllToAll comms} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

与 TPU 比较，TPU 的代价是 $B / (4W)$。因此，在单个节点内，我们在运行时间上得到理论 2 倍的加速（$B / 4W$ 对比 $B / 8W$）。

对于混合专家（Mixture of Expert，MoE）模型，我们经常想做一个*稀疏或 ragged 的 AllToAll*，即我们保证输出维度上最多 $k$ 个 $N$ 分片是非零的，也就是说 $T_\text{AllToAll} \rightarrow K[B, N]$，其中每个轴上最多 $k$ 个 $N$ 项是非零的。其代价被 $k/N$ 降低，总共约为 $\min(k/N, 1) \cdot B / (W \cdot N)$。对于 MoE，我们经常独立随机地选择非零值，因此有一定概率少于 $k$ 个非零，大致给出

$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。<d-footnote>真实的代价实际上是 $$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$ 即 $K$ 次掷骰中不同结果的期望数量，但它与给出的近似值非常接近。详见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">小测验 2 [AllToAll 耗时]：</b> 使用具有 450 GB/s 单向带宽的 8xH100 节点，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N]) 需要多长时间？如果我们只知道 8 个条目中有 4 个非零呢？

{% details 点击此处查看答案。 %}

**答案：** 注意这里 $B$ 是数组的批次维度，因此总数组大小为 $V = 2 \cdot B \cdot N$ 字节。由上面可知，在稠密情况下，代价为 $V \cdot (N-1) / (W \cdot N^2)$，或大约 $V / (W \cdot N)$。如果我们知道只有 $\frac{1}{2}$ 的条目是非填充的，我们可以发送 $V \cdot k/N / (W \cdot N) = V / (2 \cdot W \cdot N)$，大约占总代价的一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 在单个节点内，GPU 上 B 字节数组的 AllToAll 代价约为 $T_\text{comms} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU 出口}) \approx B / (8 \cdot W_\text{GPU 出口})$。对于 ragged（top-$k$）的 AllToAll，这进一步降低到 $(B \cdot k) / (64 \cdot W_\text{GPU 出口})$。</p>

**经验测量：** 下面是一个 8xH100 节点上 AllReduce 带宽的经验测量。Algo BW 是测量到的带宽（字节 / 运行时间），Bus BW 计算为 $2 \cdot W \cdot (8 - 1) / 8$，理论上是对实际链路带宽的度量。你会注意到我们的确达到了接近 370GB/s，低于 450GB/s 但相当接近，尽管每设备只有约 10GB。这意味着，尽管这些估计在理论上是成立的，但需要很大的消息才能实现它。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>禁用 SHARP 的 8xH100 节点的 AllReduce 吞吐量。蓝色曲线是经验链路带宽，根据经验测量计算为 $2 * \text{bytes} * (N - 1) / (N * \text{runtime})$。注意即使使用 10GB 的巨大数组，我们也未能特别接近宣称的 450GB/s 带宽。"%}

这是一个真实的问题，因为它切实地复杂化了我们能做出的任何理论声明，因为例如，即使是像 LLaMA-3 70B 的 MLP（大小为 `bf16[8192, 28672]`，或以 8 路模型分片时为 `bf16[8192, 3584] = 58MB`）这样大小合理的数组上的 AllReduce，相比峰值 450GB/s 也只能达到约 150GB/s。相比之下，TPU 在更小的消息大小下就能达到峰值带宽（见附录 B）。

<p markdown=1 class="takeaway">**要点：** 尽管 NVIDIA 宣称 H100 NVLink 的带宽约为 450GB/s，但在实践中很难超过 370 GB/s，因此请相应调整上面的估计。</p>

**网内归约（In-network reductions）：** 自 Hopper 代次以来，NVIDIA 交换机支持["SHARP"（Scalable Hierarchical Aggregation and Reduction Protocol，可扩展分层聚合与归约协议）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，它允许"网内归约"。这意味着*网络交换机本身*可以执行归约操作，并将结果多路复用或"多播（MultiCast）"到多个目标 GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图：</b>没有 SHARP 的 AllReduce 理论代价是 2 倍，因为它必须经过每个 GPU 两次。在实践中，加速只有约 30%（来自 NCCL 2.27.5）。"%}

理论上，这几乎将 AllReduce 的代价减半，因为它意味着每颗 GPU 可以将其数据发送到一个顶层交换机，该交换机自己执行归约并将结果广播给每颗 GPU，而无需让每颗 GPU 出口两次，同时也降低了网络延迟。

$$T_\text{SHARP AR comms} = \frac{\text{bytes}}{\text{GPU 出口带宽}}$$

注意这是精确的，不会差 $1/N$ 的因子，因为每颗 GPU 先出口 $B \cdot (N - 1) / N$，然后接收其本地分片部分归约后的版本（摄入 $B/N$），完成归约，然后再次出口 $B/N$，然后摄入完全归约的结果（摄入 $B \cdot (N - 1) / N$），精确摄入 $B$ 字节。

然而，在实践中，启用 SHARP 时我们只看到约 30% 的带宽提升，而预测是 75%。这使我们仅仅达到约 480GB/s 的有效集合带宽，远非 2 倍。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>图：</b>在节点内启用和未启用 NVIDIA SHARP 的 AllReduce 算法带宽的经验测量。即使在峰值处增益也只有约 30% 的吞吐量提升，尽管从算法上讲它本应能达到接近 75% 的增益。"%}

<p markdown=1 class="takeaway">**要点：** 理论上，NVIDIA SHARP（在大多数 NVIDIA 交换机上可用）应将 B 字节 AllReduce 的代价从约 $2 * B / W$ 降到 $B / W$。然而，在实践中我们只看到约 30% 的带宽改善。由于纯 AllReduce 在 LLM 中相当罕见，这并不是特别有用。</p>

### 跨节点集合通信 {#跨节点集合通信}

当我们超出节点层面时，代价就有点微妙了。在树上做归约时，你可以想象自底向上归约，先在节点内，然后在叶子层面，然后在脊层面，在每个层面使用常规算法。特别是对于 AllReduce，你可以看到这让我们总体上通信更少的数据，因为在节点层面 AllReduce 之后，我们只需向上向叶子出口 $B$ 字节，而不是 $B * N$。

**这有多昂贵？** 作为初步近似，因为我们有全对分带宽，AllGather 或 ReduceScatter 的代价大致是缓冲区大小（字节）除以节点出口带宽（H100 上为 400GB/s），*与树归约的任何细节无关。*

$$T_\text{AG or RS comms} = \frac{\text{bytes}}{W_\text{node 出口}} \underset{H100}{=} \frac{\text{bytes}}{\text{400e9}}$$

其中 $W_\text{node}$ 出口对于上面的 H100 网络（每节点出口 8x400Gbps IB 链路）通常为 400GB/s。最清晰的想象方式是想象在集群的*每个节点*上做环形归约。由于胖树拓扑，我们总能构造一个环，任意两个节点之间有 $W_\text{node}$ 出口，并进行常规归约。节点层面的归约（几乎）永远不会成为瓶颈，因为它有更高的总带宽和更好的延迟，尽管一般情况下代价是

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network}) = \max\left[\frac{\text{bytes}}{W_\text{GPU 出口}}, \frac{\text{bytes}}{W_\text{node 出口}}\right]$$

{% details 你可以查看更精确的推导。 %}

我们可以更精确一点，注意到我们实际上在网络每一层都做了一个环形归约，这些我们大多可以重叠，因此我们有：

$$T_\text{AG or RS comms} = \text{bytes} \cdot max_\text{depth i}\left[\frac{D_i - 1}{D_i \cdot W_\text{link i}}\right]$$

其中 $D_i$ 是深度 $i$ 的度（深度 $i$ 的子节点数），$W_\text{link i}$ 是将每个子节点连接到节点 $i$ 的链路带宽。

利用这一点，我们可以计算给定拓扑下可用的 AllGather/AllReduce 带宽为 $min_\text{depth i}(D_i * W_\text{link i} / (D_i - 1))$。在上面的情况中，我们有：

* **节点：** $D_\text{node}$ = 8，因为节点内有 8 个 GPU，$W_\text{link i}$ = 450GB/s。因此我们的 AG 带宽为 `450e9 * 8 / (8 - 1) = 514GB/s`。
* **叶子：** $D_\text{leaf}$ = 32，因为 SU 中有 32 个节点，$W_\text{link i}$ = 400GB/s（8x400Gbps IB 链路）。因此我们的带宽为 `400e9 * 32 / (32 - 1) = 413GB/s`。
* **脊：** $D_\text{spine}$ = 4，因为我们有 4 个 SU，$W_\text{link i}$ = 12.8TB/s（来自上面的 `8 * 16 * 2 * 400Gbps` 链路）。我们的带宽为 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

因此我们的整体 AG 或 RS 带宽在叶子层面为 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`，所以在实践中 $T_\text{AG or RS comms} = B / \text{413GB/s}$，也就是说即使在最高层面我们也有约 413GB/s 的 AllReduce 带宽。对于启用 SHARP 的 AllReduce，它会略低于此（约 400GB/s），因为我们没有 $(N - 1) / N$ 这个因子。不过，450GB/s 和 400GB/s 足够接近，可以当作近似值使用。

{% enddetails %}

**其他集合通信：** 除非启用 SHARP，否则 AllReduce 仍然是上述代价的 2 倍。NVIDIA 也销售支持 SHARP 的 IB 交换机，尽管并非所有提供商都有。AllToAll 跨节点时变化相当大，因为它们不像 AllReduce 那样是"分层的"。如果我们想将数据从每颗 GPU 发送到每颗其他 GPU，我们就无法利用节点层面的全对分带宽。这意味着如果我们有一个跨 $M = N / 8$ 个节点的 N 路 AllToAll，代价为

$$T_\text{AllToAll comms} = \frac{B \cdot (M - 1)}{M^2 \cdot W_\text{node 出口}} \approx \frac{B}{M \cdot W_\text{node 出口}}$$

这实际上只有 50GB/s 而不是 400GB/s 的带宽。我们从一个单独的 H100 节点内的 $B / (8 * \text{450e9})$ 变成跨 2 个节点时的 $B / (2 \cdot \text{400e9})$，退化超过 4 倍。

下面是 1024-GPU DGX H100 SuperPod 架构的总结：

|   层级   | GPU 数量 | 度（# 子节点） | 交换机带宽（全双工，TB/s） | 线缆带宽（全双工，TB/s） | 集合带宽（GB/s） |
| :-------: | :------------: | :-----------------: | :----------------------------------: | :---------------------------------: | :-------------------------: |
|   节点    |       8        |          8          |                 6.4                  |                 3.6                 |             450             |
| 叶子（SU） |      256       |         32          |                 25.6                 |                12.8                 |             400             |
|   脊    |      1024      |          4          |                 51.2                 |                51.2                 |             400             |

我们用"集合带宽（Collective Bandwidth）"一词来描述我们可以出口 GPU 或节点的有效带宽。它也是 $\text{对分带宽} * 2 / N$。

<p markdown=1 class="takeaway">**要点：** 在节点层面之上，对 B 字节进行 AllGather 或 ReduceScatter 的代价约为 $B / W_\text{node 出口}$，在 H100 DGX SuperPod 上为 $B / \text{400e9}$，而 AllReduce 的代价是其两倍，除非启用 SHARP。整体拓扑是一个胖树，旨在给出任意两对节点间的恒定带宽。</p>

**当数组沿另一个轴分片时的归约：** 考虑像这样的归约的代价

$$\text{AllReduce}_X(A[I_Y, J]\ \{ U_X \})$$

其中我们要对本身沿另一个轴 $Y$ 分片的数组做 AllReduce。在 TPU 上，由于每轴发送的数据量减少了 $1 / Y$，这个操作的整体代价相比未分片版本降低了 $1 / Y$ 的因子。在 GPU 上，代价取决于哪个轴是"内层"轴（节点内 vs. 节点间），以及每个分片是否跨越多个节点。假设 $Y$ 是内层轴，数组总共有 $\text{bytes}$ 字节，整体代价被 $Y$ 降低，但仅当 $Y$ 跨越多个节点时：

$$T_\text{comms at node} = \frac{\text{bytes}}{W_\text{GPU 出口}} \cdot \frac{1}{\min(Y, D_\text{node})}$$

$$T_\text{comms in scale-out network} = \frac{\text{bytes}}{W_\text{node 出口}} \cdot \frac{D_\text{node}}{\max(D_\text{node}, Y)}$$

$$T_\text{total} = \max(T_\text{comms at node}, T_\text{comms in scale-out network})$$

其中 N 是 GPU 的数量，同样 $D_\text{node}$ 是节点中的 GPU 数量（节点的度）。如你所见，如果 $Y < D_\text{node}$，我们在节点层面获得收益，但通常看不到整体运行时间的减少，而如果 $Y > D_\text{node}$，我们获得的加速与所跨节点数成正比。

如果我们想精确描述环形归约，对于树形 AllGather<sub>X</sub>(A<sub>Y</sub>)（假设 Y 是内层轴）的一般规则是

$$T_\text{AR or RS comms} = \text{bytes} \cdot \max_{\text{depth } i}\left[\frac{D_i - 1}{D_i \cdot \max(Y, S_{i-1}) \cdot W_{\text{link } i}}\right]$$

其中 $S_i$ 是 M * N * …，即树中层级 i 下方子节点的规模。这大致是说，我们跨越的 GPU 或节点越多，可用的带宽就越大，但仅限于该节点内。

**小测验 3 [沿两个轴分片]：** 假设我们要执行 $\text{AllGather}_X(\text{bf16}[D_X, F_Y])$，其中 $Y$ 是跨单个 SU（256 颗芯片）的内层轴。作为 $D$、$F$ 和 $Y$ 的函数，这需要多长时间？

{% details 点击此处查看答案。 %}

**答案：** 我们可以将其分为两种情况，即 Y <= 8 和 Y > 8。当 $Y <= 8$ 时，我们仍受叶子交换机限制，因此答案和往常一样，$T_\text{comms} = 2 * D * F * (32 - 1) / (32 * 400e9)$。当 Y > 8 时，由上面可得，大致为

$$T_\text{comms} = \frac{2 \cdot D \cdot F \cdot 256}{Y \cdot \text{12.8e12}} = \frac{2DF}{Y \cdot \text{50GB/s}}$$

对于 `D = 8192`，`F = 32,768`，我们有：

{% include figure.liquid path="assets/gpu/sharded-all-gather-cost.png" class="img-fluid" caption="<b>图：</b>当内层轴跨越更多节点时，分片 AllGather 的理论代价。"%}

注意，如果我们恰好做 8 路模型并行，我们确实将节点层面归约的代价降低了 8，但让整体代价保持不变，所以它是免费的，但对改善整体带宽没有帮助。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 当我们有多个分片轴时，外层归约的代价会降低一个由内层轴所跨节点数构成的因子。</p>

### 测验 4：集合通信 {#测验-4-集合通信}

**问题 1 [SU AllGather]：** 只考虑一个具有 M 个节点、每节点 N 个 GPU 的单个 SU。在 AllGather 期间，节点级交换机精确收入（ingress）和发出（egress）了多少字节？顶层交换机呢？

{% details 点击此处查看答案。 %}

**答案：** 让我们一步步来，梳理归约的组成部分：

1. 每颗 GPU 向交换机发送 $B / MN$ 字节，总共收入 $NB / MN = B / M$ 字节。
2. 我们将完整的 $B / M$ 字节向上发出到脊交换机。
3. 我们从脊交换机收入 $B * (M - 1) / M$ 字节。
4. 我们将 $B - B / MN$ 字节发出 $N$ 次，总共 $N * (B - B / MN) = NB - B / M$。

总共是 $B$ 字节收入和 $BN$ 字节发出，因此我们应该受发出限制，总时间为 $T_\text{AllGather} = BN / W_\text{node} = B / \text{450e9}$。

对于脊交换机，数学实际上更简单。我们必须收入 $B / M$ 字节共 M 次（总共 $B$ 字节），然后发出 $B (M - 1) / M$ 共 M 次，总共 $B * (M - 1)$ 字节发出。由于这个明显更大，代价为 $T_\text{AllGather} = B \cdot (M - 1) / (M \cdot W_\text{node}) = B \cdot (M - 1) / (M \cdot \text{400e9})$。

{% enddetails %}

**问题 2 [单节点 SHARP AR]：** 考虑一个每节点 N 个 GPU 的单节点。在使用 SHARP（网内归约）的 AllReduce 期间，交换机精确收入（ingress）和发出（egress）了多少字节？

{% details 点击此处查看答案。 %}

**答案：** 和之前一样，让我们一步步来。

1. 每颗 GPU 发送 $B * (N - 1) / N$ 字节，因此我们有 $N * B * (N - 1) / N = B * (N - 1)$ 收入。
2. 我们累加部分和，并向每颗 GPU 发回 $B / N$ 字节，因此 $N * B / N = B$ 字节发出。
3. 我们在本地对残差做部分和，然后将这发回给交换机。这总共是 $N * B / N = B$ 字节收入。
4. 我们捕获所有分片并多播它们，向 $N$ 个目的地各发送 $B * (N - 1) / N$，总共 $B * (N - 1) / N * N = B * (N - 1)$ 发出。

因此总共是 $B * (N - 1) + B = BN$ 字节收入和发出。这支持了整体吞吐量恰好为 $B / W_\text{出口}$。

{% enddetails %}

**问题 3 [跨节点 SHARP AR]：** 考虑一个在单个 N 个 GPU 节点上分片的数组 bf16[D<sub>X</sub>, F<sub>Y</sub>]。AllReduce(bf16[D, F<sub>Y</sub>]) 需要多长时间？你可以假设我们做了网内归约。解释如果我们有多于单个节点，这会如何不同？

{% details 点击此处查看答案。 %}

**答案：** 我们可以尝试修改上面前一个问题的答案。基本上，我们首先从每颗 GPU 出口 $B * (X - 1) / XY$ 字节，然后向每颗 GPU 发回 $B / XY$，然后将相同的量发回给交换机，然后向每颗 GPU 发回 $B * (X - 1) / XY$。总共是 $NB / Y$ 收入与发出，因此总时间为 $T_\text{comms} = NB / (Y * N * W_\text{link}) = N * 2DF / (Y * N * W_\text{link}) = 2 * D * F / (Y * W_\text{link})$，因此总时间确实随 $Y$ 减少。

如果我们超出单个节点，我们可以做大致相同的归约，但当我们出口节点级交换机时，我们需要发送所有 B 字节，而不仅仅是 $B / Y$。这是因为我们需要保持每个分片分开。

{% enddetails %}

**问题 4 [脊层面 AR 代价]：** 考虑与上面相同的设置，但 $Y = 256$（因此 AR 发生在脊层面）。AllReduce 需要多长时间？同样，可以假设网内归约。

{% details 点击此处查看答案。 %}

**答案：** 这让我们可以利用脊层面相当惊人的带宽量。我们在 4 个 SU 上有 51.2TB/s 的脊带宽，即每 SU 12.8TB/s。使用 SHARP，这可能只需 `2 * D * F / 12.8e12` 秒。

{% enddetails %}

**问题 5 [2 路 AllGather 代价]：** 计算在恰好 2 个节点上 B 字节 AllGather 的精确代价。*确保计算精确代价而非近似值，并同时考虑节点内和跨节点的代价。*

{% details 点击此处查看答案。 %}

**答案：** 在节点层面，我们有 $T_\text{comms} = B * 7 / (8 * \text{450e9}) = B / \text{514e9}$，而在节点之外我们实际上有 $T_\text{comms} = B * (2 - 1) / (2 * \text{400e9}) = B / \text{800e9}$。因此，我们实际上受节点层面归约限制，而不是叶子层面！这就促成了例如 DeepSeek v3 所做的 2 路数据并行。

{% enddetails %}

## GPU 上 LLM 扩展的屋顶线 {#gpu-上-llm-扩展的屋顶线}

现在让我们看看这一切所通向的目标：理解 GPU 上 LLM 扩展的屋顶线。这是对 TPU 训练章节[此处](../training)的补充。正如我们在那里所做的，这里的目标是考察不同并行策略的总 $T_\text{math}$ 和 $T_\text{comms}$，并理解在什么时候 $T_\text{comms} > T_\text{math}$。和之前一样，我们只考虑带有如下运算的 MLP 块

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是全局批大小**以词元计**（即 $B = \text{批大小} \cdot \text{序列长度}$）。

这里我们将重制上面的表格，展示 GPU 和节点层面的有效带宽：

|  节点类型  | 每节点 GPU 数 | GPU 出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意：** GPU 和节点的出口带宽都决定了我们 LLM 的屋顶线。我们将使用术语 $W_\text{collective}$ 来描述 GPU 或节点带宽，取决于我们是在节点层面内还是之上操作。

让我们像对 TPU 那样，看看**数据并行、张量并行、流水线并行、专家并行**及其组合的算力-通信屋顶线。在本节的其余部分，我们将聚焦于 H100 针对具体计算的屋顶线。GB200-NVL72 有相同的一般屋顶线，但因为我们有更大的节点出口带宽，我们有时会受节点层面而非横向扩展层面的瓶颈限制。

### 数据并行 {#数据并行}

如前所述，DP 和 ZeRO 分片涉及在反向传播中做一次权重 AllReduce 或一次 ReduceScatter + AllGather。由于这两者代价相同，为了对纯数据并行或使用全分片数据并行（FSDP）*且不使用网内归约*的情况达到算力受限，在每层、在反向传播中、沿大小为 X 的轴，我们有

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此，为了 $T_\text{math} > T_\text{comms}$，我们需要 $B / (XC) > 1 / W_\text{collective}$，即

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是 GPU 或节点层面的出口带宽，取决于我们是在节点内还是跨节点分片。因此：

* **在节点内**，我们只需要每 GPU 的**词元**批大小 > $\text{990e12} / \text{450e9} = 2200$。
* **在 SU 内或脊层面**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比 TPU 上要高不少，TPU 上三个轴都考虑时是 850。例如，在 16000 个 H100 上训练的 LLaMA-3 将需要至少 40M 词元的批大小（作为参考，它们用了 16M）。DeepSeek v3 在 2048 个 H800 GPU 上训练，带宽较低为 300GB/s（而不是 H100 上的 450GB/s），将需要 $\text{990e12} / \text{300e9} = 3300$ 词元每 GPU，即约 6.7M（实践中它们用了 4M）。

如果启用网内归约并使用纯数据并行，理论上我们有 2 倍的 AllReduce 带宽，这会将这两个数字都减半。然而，在实践中收益更接近 30%，这基本上仅仅弥补了我们通常难以达到宣称数字的事实。此外，因为纯数据并行很少有用，这在实践中基本无关紧要。

**MoE 模型：** 对于混合专家（MoE）模型，其中我们有 E 个专家、每词元 k 个专家，这增加到

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这将每 GPU 词元批大小放大了 $E/k$ 的因子，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，新的 OpenAI 开源模型，其 $k=4$、$E=128$，这跨节点增加到 `32 * 2475  = 79,200`，一个高得有点荒谬的数字。

**当 X 较小时会发生什么？** 当我们只做例如 2 节点数据并行时，我们从 $(X - 1) / X$ 的缩放中受益，这给出

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中 X 是节点数，$N = 8 \cdot X$。那么对于稠密模型我们有 $B / N > \alpha \cdot (X - 1) / X$，或例如 $B / N > \text{1237}$，是上述值的一半。你会注意到出于这个原因，2 路数据并行相当常见。

<p markdown=1 class="takeaway">**要点：** 数据并行和 ZeRO 分片需要每 GPU 约 2500 词元的批大小，才能在 H100 或 B200 上达到算力受限，假设完美的重叠和模型浮点利用率（MFU）。对于 MoE 模型，这按 $E / k$（总参数与激活参数的比值）的因子增加。当进行少量数据并行时，临界批大小会下降。</p>

### 张量并行 {#张量并行}

张量并行需要对激活值进行一次 AllGather 和一次 ReduceScatter，我们需要将它们与 MLP 的 FLOPs 重叠。换句话说，在前向传播中，我们有

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

要算力受限，这给出规则

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在节点内，这给我们约 $F / 2200$，节点外为 $F / 2475$。对于像 LLaMA-3 那样的 $F=\text{28000}$，这约为 11 路 TP（或向下取整，约为 8 路，即一个节点的大小）。和上面一样，当我们恰好跨 2 个节点时获得额外的 2 倍带宽，因此我们通常可以做 16 路张量并行（$F > 2475 \cdot (Y - 8)$），理论上最多给我们 19 路模型并行。

<p markdown=1 class="takeaway">**要点：** 沿大小为 Y、前馈维度为 F 的轴进行张量并行，当 $Y > F / 2475$ 时变得通信受限，这通常将我们限制在仅节点内 TP 或最多 2 节点 TP。</p>

### 专家并行 {#专家并行}

正如我们上面已经指出的，混合专家（MoE）模型带有 E 倍的模型权重，却只有 k 倍的 FLOPs，使得数据并行明显更难。我们可以通过沿专家维度对权重分片来在一定程度上缓解这一点，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]。为了做 MLP 块，我们需要引入 2 倍 AllToAll 来将我们的激活值发送到相应的专家。

如上所述，如果这个 AllToAll<sub>Z->k</sub>([B, D, k]) 跨多个节点，其代价约为 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，因此对于纯专家并行我们需要

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们需要 $K > Z/8$ 且 $F > \alpha \cdot (Z - 8)/k$，或者 $Z \gg K$ 且 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这给你专家并行可行的两个域：一个是少量专家并行（大约 2 节点）和较小的 $F$，另一个是较大的 $F$ 和可以任意大（最多 E 路专家并行）的 $Z$。

你会在实践中看到两种情况：要么是少量专家并行（像 DeepSeek v3，其 F 非常小、跨节点的专家并行相对较小且受限），要么是具有较大 F 的模型，在这种情况下我们可以在 TP 之外做显著的跨节点 EP。

<p markdown=1 class="takeaway">**要点：** 如果 $F < 8 * C / W_\text{node}$，专家并行可以跨 1-2 个节点，代价与 TP 相似（略低）；或者如果 $F > 8 * C / W_\text{node}$，我们可以做大量的专家并行（最多 $E$ 个节点），代价相对较低。</p>

### 流水线并行 {#流水线并行}

流水线并行将层跨节点切分，通信代价极低，因为我们只是每隔几层发送小的微批次（microbatch）激活值。历史上流水线并行一直受困于"流水线气泡（pipeline bubbles）"，但有了新型零气泡流水线（zero-bubble pipelining）方法，通常可以避免。

流水线的总体通信代价很小：有 $N_\text{MB}$ 个微批次和 $N_\text{stages}$ 个阶段，我们有每跳 $T_\text{comms} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$，以及 $N_\text{MB} + N_\text{stages} - 2$ 跳，因此大致

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于我们是除以 $N_\text{layers}$，这远小于任何其他代价。换句话说，从通信的角度看，流水线基本上是免费的。那么为什么我们不直接用流水线呢？有几个原因：

(1) **代码复杂性：** 流水线不像其他方法与自动并行框架（如 XLA 的 GSPMD）配合得那么好。因为它引入了微批次来隐藏流水线气泡，它改变了程序的结构，而定制的零气泡流水线调度通过要求前向和反向传播的复杂交错而加剧了这个问题。

(2) **流水线使数据并行和 FSDP 变难：** 可能不采用流水线的最大原因是它与 FSDP 和数据并行配合得不好。特别是 ZeRO-3 分片表现很差，因为它要求我们在每个微批次上 AllGather 权重，而当只有 $B / N_\text{microbatches}$ 个词元来分摊 AllGather 代价时，这行不通。此外，在反向传播中，*直到最后一个微批次通过一个给定阶段，我们才能 AllReduce 或 ReduceScatter 梯度，这意味着我们有显著的非重叠通信时间。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图：</b>一个 2 阶段、2 微批次流水线的示例。F 表示一个阶段的前向传播，B 表示一个阶段的后向传播（代价 2 倍）。G 表示数据并行的 AllReduce，它可能明显长于单个微批次的时间。"%}

(3) **流水线气泡与阶段不平衡：** 正如你在上面（糟糕的）流水线调度中看到的，在一个朴素的流水线调度中很容易出现显著的气泡（即浪费的计算）。在上面，第二阶段在步骤 0 空闲，第一阶段从步骤 2 到 3 空闲，第二阶段在最后一步再次空闲。虽然我们可以通过仔细调度在一定程度上避免这些，但我们仍然经常有一些气泡。我们还必须在关键路径上将激活值从一个阶段传递到下一个阶段，这会增加开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图：</b>一个展示传递代价（红色）的流水线示例。这会使各阶段相对彼此偏移，并增加流水线气泡开销。"%}

对于每个问题都有变通办法，但它们往往实现复杂且难以维护；相比其他方法，流水线仍然是一种通信代价较低的技术。

**关于延迟的注意事项：** 如前所述，GPU 即使使用相当大的消息也难以达到全部 AllReduce 带宽。这意味着即使我们理论上可以将例如专家并行的 AllToAll 跨多个节点扩展，我们也可能难以达到总带宽的 50%。这意味着我们确实试图将 TP 或 EP 保持在较少数量的节点内，以最小化延迟开销。

### 实例 {#实例}

**DeepSeek 做了什么？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 在 2048 个 H800 GPU 上训练，采用：

* 64 路专家并行（EP），跨 8 个节点
* 16 路流水线并行（PP）
* 2 路 ZeRO-1 数据并行（DP）

它们的稳态批大小为 `4096 * 15360 = 62,914,560` 词元，即每 GPU 30k 词元。你可以看到这已经相当大了，但它们的模型也非常稀疏（k=8，E=256），所以你需要相当大的批大小。你可以看到，通过 64 路 EP 和 16 路 PP，我们最终总共得到 1024 路模型并行，这意味着 AllReduce 在脊层面完成，并且因为它是仅 2 路的，我们最终在实践中得到 $2 / (2 - 1) = 2$ 倍的更多带宽。这也有助于降低最终的数据并行 AllReduce 与最终流水线阶段重叠的代价。

**LLaMA-3 做了什么？** LLaMA-3 在 16k GPU 上以 16M 词元的 BS 训练，即每 GPU 约 1k 词元。它们采用：

* 节点内 8 路张量并行（TP）
* 16 路流水线并行（PP）
* 128 路 ZeRO-1 数据并行

这也是一个稠密模型，所以一般来说这些都相当简单。16 路 PP 将数据并行 AllReduce 的代价降低了 16 倍，这有助于我们降低临界批大小。

### GPU 上 LLM 扩展的要点总结（TLDR） {#gpu-上-llm-扩展的要点总结-tldr}

让我们退一步，对我们目前学到的东西做一个总体总结：

* **数据并行或 FSDP（ZeRO-1/3）需要每 GPU 约 2500 词元的本地批大小**，尽管理论上网内归约 + 纯 DP 可以在一定程度上降低这一点。
* **张量并行在最多约 8 路时是算力受限的**，但在变得通信受限之前我们缺乏将其扩展太多的带宽。这主要将我们限制在单个 NVLink 域（即单节点，或需要使用最多 72 个 GPU 的 GB200 NVL72）。
* **任何跨多个节点的模型并行形式都可以进一步降低 FSDP 的代价**，因此我们经常想混合 PP + EP + TP 来跨许多节点并降低 FSDP 代价。
* **如果你能处理零气泡流水线的代码复杂性，并保持相当大的批大小以避免数据并行瓶颈，流水线并行效果很好。** 流水线通常使 ZeRO-3 不可能（因为你需要对每个流水线阶段做 AllGather），但你可以改用 ZeRO-1。

**在高层面上，这给了我们一个在 GPU 上对大模型分片的诀窍：**

* 对于相对较小的稠密模型，如果你有批大小，激进的 FSDP 非常有效，如果需要的话可以配合一些流水线并行或张量并行。
* 对于更大的稠密模型，1-2 节点 TP + 多节点 PP + 纯 DP 的某种组合效果很好。
* 对于 MoE，上述规则适用，但我们也可以做专家并行，我们通常总体上更偏好它而非 TP。如果 $F > 8 * C / W_\text{node}$，我们可以做大量的多节点专家并行，否则我们被限制在大约 2 节点 EP。

### 测验 5：LLM 屋顶线 {#测验-5-llm-屋顶线}

**问题 1 [B200 屋顶线]：** 一个 B200 DGX SuperPod（**不是** GB200 NVL72）在节点内有 2 倍的带宽（900GB/s 出口），但在横向扩展网络中有相同的带宽（400GB/s）（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)）。总的 FLOPs 已在上面给出。这如何改变模型和数据的并行屋顶线？

{% details 点击此处查看答案。 %}

**答案：** 我们的 bfloat16 FLOPs/s 从 990 增加到 2250 TFLOPs，增加了 2.25 倍。带宽翻倍后，在节点内，我们的屋顶线大致保持不变。例如对于 TP，临界强度上升到 `2250e12 / 900e9 = 2500`，因此我们限制为 $Y < F / 2500$，只稍微高一点（而且除非节点大小增加，否则这帮不上忙）。

然而在节点之外，缺乏额外带宽实际上让我们更难达到算力受限！例如，对于数据并行，我们的临界批大小增加到 `2250e12 / 400e9 = 5625`，因为我们的 GPU 可以用相同的带宽做明显更多的 FLOPs。

带有 72-GPU 节点的 GB200 SuperPod 通过增加更多出口带宽改变了这一点（[来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)）。

{% enddetails %}

**问题 2 [如何对 LLaMA-3 70B 分片]：** 考虑 LLaMA-3 70B，以 bfloat16 训练，使用 fp32 优化器状态和 Adam。

1. 至少需要多少颗 H100 才能仅仅存储权重和优化器？
2. 假设我们想在 4096 颗 H100 GPU 上训练 15T 词元。假设我们达到了 45% MFU（模型浮点利用率，Model FLOPs Utilization）。训练需要多长时间？
3. LLaMA-3 70B 有 `F = 28,672`，并以约 4M 词元的批大小训练。在不通信受限的前提下，我们最多能做多少模型并行？加上纯 DP，我们能在 4k 芯片上保持算力受限地训练 LLaMA-3 吗？ZeRO-3 呢？8 路流水线呢？*注意：同时考虑通信代价和 GPU 内存使用。*

{% details 点击此处查看答案。 %}

1. 我们需要 2 字节存储权重，8 字节存储优化器状态，因此至少 700GB。有了 80GB 的 DRAM，我们至少需要至少 9 颗 GPU，或（向上取整）至少 2 个 8xH100 节点。这将需要永远的训练时间，而且放不下梯度检查点，但这是一个下界。
2. 这将总共需要 `6 * 70e9 * 15e12 = 6.3e24` bf16 FLOPs。每颗 GPU 可以做 `990e12` FLOPs，因此在 45% MFU 下我们可以做 1.8e18 FLOPs/s。因此整个训练将需要 3.5e6 秒，即 40 天。
3. 在节点内，我们有 450GB/s 的带宽，因此限制大约是 `F / 1995 = 28672 / 1995 = 14.372`。由于这不跨 2 个节点，实际上意味着我们最多会到 8 路模型并行。
   1. 这然后将要求我们做 512 路 DP。首先，我们需要看是否有足够的内存。由于我们的模型只分片了 8 路，这意味着 `700GB / 8 = 87.5GB / GPU`，这放不下，所以不行！
   2. 使用 ZeRO-3 和 8 路 TP，我们将做 512 路 ZeRO-3。这不会有任何内存问题，因为我们激进地对一切进行分片。我们将有每 GPU 批大小 `4e6 / 4096 = 976`。这相当低，甚至低于我们的纯 DP 限制，而且由于我们必须移动权重，这还是该限制的两倍。所以不行。
   3. 使用 8 路流水线，每个模型并行分片现在跨 8 个节点。正如我们所见，这将叶子层面 AllGather 的代价降低了 8，因此那里的总体 AllReduce/AllGather 带宽从 400GB/s 变为 `8 * 400GB/s = 3200GB/s`。屋顶线于是为 `990e12 / 3200e9 = 309`，所以我们应该没问题！我们只需高效地实现流水线。

{% enddetails %}

**问题 3 [Megatron-LM 超参数]：** 考虑来自 [Megatron-LM 仓库](https://github.com/NVIDIA/Megatron-LM) 的这张图，展示了它们的高 MFU 数字。

{% include figure.liquid path="assets/gpu/megatron-hparams.png" class="img-fluid" %}

注意它们的序列长度处处为 4096。对于 16B、70B 和 314B 模型，每 GPU 词元批大小是多少？假设数据并行是最外层轴，并假设 bfloat16 归约，判断其中每个模型在理论上是否算力受限或通信受限，以及是否存在更优的配置？

{% details 点击此处查看答案。 %}

**答案：** 让我们从每 GPU 批大小开始。

* **16B**：`192 * 4096 / 192 = 4096` 词元每 GPU
* **70B**：`384 * 4096 / 768 = 2048` 词元每 GPU
* **314B**：`1536 * 4096 / 3072 = 2048` 词元每 GPU

这意味着除第一个外，它们都徘徊在每批约 2k 词元，这明显接近我们为 FSDP 计算出的临界阈值。我们基于脊层面归约计算出该界限为每 GPU 2,472 词元，这大致应该在这里起作用。不过对于 70B 和 314B，因为我们有 16 和 64 路模型（PP + TP）分片，我们在脊层面分别获得 2 倍和 8 倍更好的吞吐量，这意味着我们应该分别在约 1k 和 300 词元/步时算力受限。

{% enddetails %}

## 致谢与延伸阅读 {#致谢与延伸阅读}

本章大量依赖于许多知识渊博的 GPU 专家的帮助，包括：

* Adam Paszke，他帮助解释了 GPU 上内核编程的现实。
* Swapnil Patil，他首先解释了 GPU 网络互联是如何运作的。
* Stas Bekman，他指出 GPU 的经验现实常常与宣称的规格不同。
* Reiner Pope，他帮助澄清了 GPU 和 TPU 在硬件层面如何比较。
* Frédéric Bastien，他对芯片层面的叙述给出了详细反馈。
* Nouamane Tazi，他在 GPU 上训练 LLM 的经验帮助改进了屋顶线一节。
* Sanford Miller，他帮助我理解 GPU 是如何联网的，以及 NVIDIA 的规格与现场通常部署的情况相比如何。

关于 GPU 有很多很好的阅读资料，但我最喜欢的一些包括：

* [SemiAnalysis 的 NVIDIA Tensor Core 历史](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇精彩的文章，描述 GPU 如何从电子游戏引擎转变为 ML 加速器。
* [SemiAnalysis 对 Blackwell 性能的分析](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得一读，以理解下一代 NVIDIA GPU。
* [H100 DGX SuperPod 参考](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：关于更大 GPU 集群如何联网的枯燥但有用的阅读。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576) 是一份关于 GB200 系统的类似文档。
* [关于 NVLink 交换机的 Hot Chips 演讲](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：关于 NVLink 和 NCCL 集合通信的有趣阅读，特别包括网内归约。
* [DeepSeek-V3 技术报告](https://arxiv.org/pdf/2412.19437)：一个大型半开放 LLM 训练报告的好例子，描述了他们如何选择分片设置。
* [如何优化一个 CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)：一篇很棒的博客，描述如何使用 CUDA Core 实现一个高效的 matmul，着眼于 GPU 上的缓存一致性。
* [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：一份关于 GPU 上 LLM 并行的指南，部分启发了本章。
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：一个更偏向 GPU 和 PyTorch 的 LLM 屋顶线与性能工程教程。
* [Cornell 理解 GPU 架构网站](https://cvw.cac.cornell.edu/gpu-architecture)：与本书记似，更具体地比较 GPU 和 CPU 的内部结构。

## 附录 A：这在 GB200 上如何变化？ {#附录-a-这在-gb200-上如何变化}

Blackwell 引入了许多重大网络变化，包括 NVLink 5，其整体 NVLink 带宽翻倍（900GB/s）。B200 仍然像 H100 一样有 8-GPU 节点，但 GB200 系统（将 B200 GPU 与 Grace CPU 组合）引入了更大的 NVLink 域（NVL72 中为 72 个 GPU，理论上最多 576 个）。这个更大的 NVLink 域也有效地增加了节点出口带宽，从而降低了节点之上的集合通信代价。

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>图：</b>展示 GB200 NVL72 单元如何构建的示意图，包含 18 个交换机和 72 个 GPU。"%}

在节点内，增加的带宽（从 450GB/s 到 900GB/s）没有太大区别，因为我们也让每颗 GPU 的总 FLOPs/s 翻倍了。我们的屋顶线大多保持不变，不过因为 NVLink 带宽好得多，专家并行变得更容易。

在节点之外，情况变化更大。下面是来自[此处](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的 SuperPod 示意图。

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>一个 576 GPU 的 GB200 DGX SuperPod 示意图。"%}

如你所见，每节点出口带宽增加到 `4 * 18 * 400 / 8 = 3.6TB/s`，高于 H100 的 400GB/s。由于我们的每芯片 FLOPs 也翻倍，这将在约 4 倍程度上改善有效的跨节点屋顶线。现在我们可能开始担心我们是受节点层面而非横向扩展层面的瓶颈限制。

**Grace Hopper：** NVIDIA 还销售 GH200 和 GB200 系统，将一定数量的 GPU 与 Grace CPU 配对。例如，一个 GH200 有 1 个 H200 和 1 个 Grace CPU，而一个 GB200 系统有 2 个 B200 和 1 个 Grace CPU。这个系统的一个优势是，CPU 使用全带宽 NVLink 连接（称为 NVLink C2C）连接到 GPU，因此你有非常高的 CPU 到 GPU 带宽，对于将参数卸载到主机内存（host RAM）很有用。换句话说，对于任何给定的 GPU，到达主机内存的带宽与到达另一个 GPU 的 HBM 相同。

## 附录 B：更多网络细节 {#附录-b-更多网络细节}

下面是 NVLink 4 交换机的一个示意图。总共有 64 个 NVLink4 端口（每个使用 2 条物理通道），以及一个处理通道间交换的大型交叉开关。相比之下，TPU 使用带有可动态重新配置镜子的光交换机。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图：</b>单个 NVLink4 交换机的较低层视图。"%}

在每个层面，我们可能受可用链路带宽或总交换机带宽的瓶颈限制。

* **节点层面：** 在节点层面，我们有 4 * 1.6TB/s = 6.4TB/s 的 NVSwitch 带宽，但我们 8 个 GPU 每个只能出口 450GB/s 到交换机，这意味着我们在节点内实际上有 450e9 * 8 = 3.6TB/s（全双工）的峰值带宽。
* **SU/叶子层面：** 在 SU 层面，我们有 8 个交换机以全连接方式连接 32 个节点，使用 1x400 Gbps InfiniBand。这给我们从节点出来的 8 * 32 * 400 / 8 = 12.8TB/s 出口带宽，而我们在交换机层面有 8 * 1.6TB/s = 12.8TB/s，因此两者精确一致。
* **脊层面：** 在脊层面，我们有 16 个交换机以 2x400 Gbps 链路连接 32 个叶子交换机，因此我们有 32 * 16 * 400 * 2 / 8 = 51.2TB/s 出口带宽。与叶子交换机不同，脊交换机的所有 64 个端口都朝下，因此每个交换机可以移动 64 * 400 / 8 = 3.2TB/s 的流量，给出我们在交换机层面 16 * 3.2TB/s = 51.2TB/s，再次精确一致。

每 GPU，这在节点层面给我们 450GB/s 的 GPU 到 GPU 带宽，在 SU 和脊层面都是 50GB/s。

**GPU 经验 AR 带宽：**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>8xH100 集群上的 AllReduce 带宽（节点内，SHARP 禁用）。"%}

TPU v5p 带宽（1 轴）：

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5p 4x4x4 集群上的 AllReduce 带宽（沿一个轴）。"%}

下面是 AllGather 带宽：

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>8xH100 集群上的 AllGather 带宽（节点内）。"%}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5e 8x16 集群上的 AllGather 带宽（沿一个轴）。"%}

**关于 AllToAll 代价的更多内容：**

在这里我们可以将近似 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$ 进行比较。除了 Z 较小时，它们很相似。

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>图：</b>随着分片数量增加，ragged AllToAll 的近似代价与真实代价的比较。"%}
