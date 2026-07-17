---
layout: distill
title: "How to Scale Your Model（如何扩展你的模型）"
subtitle: "A Systems View of LLMs on TPUs"
# permalink: /main/
description: "Training LLMs often feels like alchemy, but understanding and optimizing the performance of your models doesn't have to. This book aims to demystify the science of scaling language models: how TPUs (and GPUs) work and how they communicate with each other, how LLMs run on real hardware, and how to parallelize your models during training and inference so they run efficiently at massive scale. If you've ever wondered \"how expensive should this LLM be to train\" or \"how much memory do I need to serve this model myself\" or \"what's an AllGather\", we hope this will be useful to you."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: roofline
next_section_name: "Part 1: Rooflines"

bibliography: main.bib

citation: true

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
  - name: 高层概览
  - name: 各节链接

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
permalink: /index-zh/
sitemap: false
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习在很大程度上仍然近乎一种黑魔法，但优化模型的性能却不必如此——即便在超大规模下也是如此！相对简单的原理处处适用——从处理单个加速器到数万个——而理解它们能让你做许多有用的事情：

- 大致估算模型各部分距离其理论最优值有多近。
- 在不同规模下就不同并行方案做出明智选择（即如何在多个设备间拆分计算）。
- 估算训练和运行大型 Transformer 模型所需的成本与时间。
- 设计能够利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特性](https://arxiv.org/abs/2007.00072)的算法。
- 基于对当前算法性能瓶颈的明确理解来设计硬件。

**预修背景：** 我们将假设你对 LLM 和 Transformer 架构有基本了解，但不一定了解它们如何在大规模下运行。你应该知道 LLM 训练的基础知识，最好对 JAX 也有一定熟悉。一些有用的背景阅读包括[这篇博客](https://jalammar.github.io/illustrated-transformer/)（关于 Transformer 架构）以及[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。也请查看[这份清单](conclusion#further-reading)以获取更多有用的同期及未来阅读。

**目标与反馈：** 读完本书后，你应该能够从容地为一个给定硬件平台上的 Transformer 模型估算出最佳并行方案，并大致知道训练和推理需要多长时间。如果做不到，欢迎给我们发邮件或留言！我们很想知道如何能把内容讲得更清楚。

<p markdown=1 class="announce">你可能也会喜欢阅读关于 NVIDIA GPU 的新[第 12 节](gpus)！</p>

### 你为什么应该在意？ {#你为什么应该在意}

三、四年前，我觉得大多数 ML 研究者未必需要理解本书中的任何内容。但今天，即便是"小"模型也运行得如此接近硬件极限，以至于开展新颖的研究都需要你思考大规模下的效率问题。<d-footnote>历史上，ML 研究在系统创新与软件改进之间遵循着某种"嘀嗒"循环。Alex Krizhevsky 当年不得不编写非常规的 CUDA 代码来让 CNN 跑得快，但短短几年内，Theano、TensorFlow 这类库就让这种努力变得不再必要。也许这里也会发生同样的事，本书中的一切在几年后都会被抽象掉。但扩展定律不断将我们的模型推到硬件的最前沿，而且在可预见的未来，开展前沿研究很可能与"如何高效地将模型扩展到大型硬件拓扑"这一理解密不可分。</d-footnote> **如果在基准上提升 20% 却要以屋顶线效率下降 20% 为代价，那这一提升就毫无意义。** 有前景的模型架构之所以经常失败，要么是因为它们 _无法_ 在大规模下高效运行，要么是因为没人投入精力去让它们做到这一点。

**"模型扩展"的目标，是能够在增加用于训练或推理的芯片数量的同时，实现吞吐量成比例、线性的增长。** 这被称为"*强扩展（strong scaling）*"。尽管增加芯片（"并行"）通常会缩短计算时间，但代价是芯片间通信的增加。当通信耗时超过计算时，我们就变得"通信受限"，从而无法强扩展。<d-footnote>随着你的计算时间缩短，通常还会在单个芯片层面遇到瓶颈。你崭新的 TPU 或 GPU 也许标称每秒能执行 500 万亿次运算，但如果你不小心，当它在内存中来回搬运参数而陷入停滞时，实际可能只发挥出十分之一的性能。单芯片计算、内存带宽与总内存之间的相互作用，对扩展这件事至关重要。</d-footnote> 如果我们对自己的硬件足够了解，能预见到这些瓶颈会在何处出现，就可以设计或重新配置模型来规避它们。<d-footnote>硬件设计者面临的是相反的问题：打造一种既为我们的算法提供刚好足够的算力、带宽和内存，又将成本最小化的硬件。你可以想象这个"协同设计"问题的压力有多大：你必须押注于当首批芯片真正可用时（通常要 2 到 3 年之后）算法会是什么样子。TPU 的故事是这场博弈中响亮的胜利。矩阵乘法是一种独特的算法，因为它每字节内存所使用的 FLOPs 远多于几乎任何其他算法（每字节 N 个 FLOPs），而早期 TPU 及其脉动阵列架构，相比同时代的 GPU 取得了远优的每美元性能。TPU 是为 ML 工作负载而设计的，而配备 Tensor Core 的 GPU 也正在迅速演变以填补这一空白。但你可以想象，如果神经网络没有兴起，或者发生了某种根本性的改变，那代价会有多大……[已截断]</d-footnote>

*本书的目标是解释 TPU（和 GPU）硬件如何工作，以及 Transformer 架构如何演进以在当前硬件上表现良好。我们希望这既能对设计新架构的研究者有用，也能对致力于让当代 LLM 跑得更快的工程师有用。*

## 高层概览 {#高层概览}

本书的整体结构如下：

[第 1 节](roofline)讲解屋顶线（roofline）分析，以及哪些因素会限制我们的扩展能力（通信、计算和内存）。[第 2 节](tpus)和[第 3 节](sharding)详细讨论 TPU 如何工作——既作为单独的芯片，也——这一点至关重要——作为一个互联系统，其芯片间链路在带宽和延迟上都很有限。我们将回答诸如以下问题：

* 特定大小的矩阵乘法应该耗时多久？它在什么情况下会受算力限制、受内存限制或受通信带宽限制？
* TPU 是如何连接在一起组成训练集群的？系统每个部分的带宽有多大？
* 在多个 TPU 之间收集、散布或重新分布数组需要多长时间？
* 我们如何高效地乘那些在不同设备上以不同方式分布的矩阵？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图：</b>来自<a href='tpus'>第 2 节</a>的一幅示意图，展示 TPU 如何执行逐元素乘积。根据数组大小和各种链路的带宽，我们可能处于算力受限（用满硬件算力）或内存受限（受内存加载瓶颈限制）的状态。"%}

五年前，ML 的架构图景还很丰富——ConvNets、LSTMs、MLPs、Transformers——但现在我们基本上就只剩 Transformer 了<d-cite key="transformers"></d-cite>。我们坚信，理解 Transformer 架构的每一处细节都是值得的：每个矩阵的确切大小、归一化发生在何处、每个部分包含多少参数和 FLOPs<d-footnote>浮点运算（FLoating point OPs），基本上就是所需加法和乘法的总数。尽管许多资料把 FLOPs 理解为"每秒运算次数"，但我们显式使用 FLOPs/s 来表示这一点。</d-footnote>。 [第 4 节](transformers)会仔细梳理这套"Transformer 数学"，展示如何统计训练和推理各自的参数量与 FLOPs。这能告诉我们模型将占用多少内存、我们会在计算或通信上花费多少时间，以及注意力相对于前馈块何时变得重要。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图：</b>一个标准 Transformer 层，其中每个矩阵乘法（matmul）表示为圆中的一个点。所有参数（不包括归一化层）以紫色显示。<a href='transformers'>第 4 节</a>会更详细地讲解这幅图。"%}

[第 5 节：训练](training)和[第 7 节：推理](inference)是本书的核心，我们将在此讨论一个根本性问题：给定一个大小和芯片数量已知的模型，我该如何并行化我的模型以保持在"强扩展"状态？这是一个简单的问题，答案却出奇地复杂。从高层来看，有四种主要的并行技术用于在多个芯片间拆分模型（**数据**、**张量**、**流水线**和**专家**），以及若干其他用于降低内存需求的技术（**重计算**、**优化器/模型分片（即 ZeRO）**、**主机卸载**、**梯度累积**）。我们在此会讨论其中许多技术。

我们希望读完这些章节后，你能够针对新架构或新场景自行在其中做出选择。[第 6 节](applied-training)和[第 8 节](applied-inference)是实用教程，将这些概念应用到流行的开源模型 LLaMA 3 上。

最后，[第 9 节](profiling)和[第 10 节](jax-stuff)探讨如何在 JAX 中实现其中一些想法，以及当代码出问题时如何进行性能剖析与调试。[第 12 节](gpus)是一节新的内容，同样深入讲解 GPU。

通篇我们都会尽量给你一些可以自己动手解决的问题。请不必有压力要去读完所有章节，也不必按顺序阅读。欢迎留下反馈。目前这是一份草稿，还会持续修订。谢谢！

*我们在此致谢 James Bradbury 和 Blake Hechtman，本书中的许多观点由他们推导得出。*

<h3 markdown=1 class="next-section">闲话少说，[这里是第 1 节](roofline)，讲的是 TPU 屋顶线。</h3>

## 各节链接 {#各节链接}

*这个系列可能比需要的更长，但我们希望这不会让你却步。前三章是预备知识，如果你已经熟悉相关内容可以跳过，不过它们引入了后文会用到的记号。最后三部分可能是最实用的，因为它们讲解了如何处理真实模型。*

**第一部分：预备知识**

* [**第 1 章：屋顶线分析简述**](roofline)。算法受三样东西约束：计算、通信和内存。我们可以用它们来估算算法运行会有多快。

* [**第 2 章：如何理解 TPU**](tpus)。TPU 如何工作？这又如何影响我们能训练和服务的模型？

* [**第 3 章：分片矩阵及其乘法**](sharding)。这里我们通过最喜欢的一种运算——（分片）矩阵乘法——来讲解模型分片与多 TPU 并行。

**第二部分：Transformer**

* [**第 4 章：你需要了解的全部 Transformer 数学**](transformers)。一个 Transformer 在前向和反向传播中会用到多少 FLOPs？你能算出参数量吗？它的 KV cache 大小？我们在此逐步推导这些数学。

* [**第 5 章：如何为训练并行化 Transformer**](training)。FSDP。Megatron 分片。流水线并行。给定一定数量的芯片，我该如何以尽可能高的效率，用给定的批大小训练一个给定大小的模型？

* [**第 6 章：在 TPU 上训练 LLaMA 3**](applied-training)。我们该如何在 TPU 上训练 LLaMA 3？需要多久？要花多少钱？

* [**第 7 章：Transformer 推理全解**](inference)。一旦训练好模型，我们就得去服务它。推理带来了一个新的考量——延迟——并改变了内存格局。我们将讨论分离式服务如何运作，以及如何思考 KV cache。

* [**第 8 章：在 TPU 上服务 LLaMA 3**](applied-inference)。在 TPU v5e 上服务 LLaMA 3 要花多少钱？延迟与吞吐量的权衡是什么？

**第三部分：实用教程**

* [**第 9 章：如何剖析 TPU 代码**](profiling)。真实的 LLM 从来不会像上面的理论那么简单。这里我们讲解 JAX + XLA 技术栈，以及如何使用 JAX/TensorBoard 性能剖析器来调试并修复实际问题。

* [**第 10 章：在 JAX 中编程 TPU**](jax-stuff)。JAX 提供了一堆用于并行化计算的神奇 API，但你需要知道如何使用它们。有趣的示例与已解习题。

**第四部分：结论与附加内容**

* [**第 11 章：结论与延伸阅读**](conclusion)。收尾思考，以及关于 TPU 和 LLM 的延伸阅读。

* [**第 12 章：如何理解 GPU**](gpus)。关于 GPU 的附加章节，讲解它们如何工作、如何组网，以及它们的屋顶线与 TPU 有何不同。
