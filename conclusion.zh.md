---
layout: distill
title: "Conclusions and Further Reading（总结与延伸阅读）"
permalink: /conclusion-zh/
sitemap: false
# permalink: /main/
description: "感谢你的阅读！在这里我们会再附上一些可供深入学习的参考资料。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff-zh"
previous_section_name: "第 10 部分. 用 JAX 编写 TPU 程序"

next_section_url: "../gpus-zh"
next_section_name: "第 12 部分. 如何理解 GPU"

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
  - name: "致谢"
  - name: "延伸阅读"
  - name: "反馈"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
  .algorithm

  .algorithm li
---

**感谢你读完整本书，也恭喜你一路坚持到了最后。** 在收尾之前，先致谢几位贡献者：

## 致谢 {#致谢}

本文凝聚了 Google DeepMind 许多同事大量的集体投入，在此我们想简要地表达感谢！

- James Bradbury、Reiner Pope、Noam Shazeer 和 Blake Hechtman 最初推导了本文中的诸多思想，并且很早就理解了 Transformer 的系统视角。
- Sholto Douglas 撰写了本文的初稿，并负责启动了整个项目。他对本文整体叙事的塑造，比其他任何人都更有贡献。
- Jacob Austin 主导了将初稿从粗糙的笔记打磨成更完善、更全面的成稿的工作。他承担了本文大量的编辑、排版与发布工作，并协调了其他作者的贡献。
- 大部分示意图与动画由 Anselm Levskaya 和 Charlie Chen 制作。
- Charlie Chen 撰写了推理章节，并绘制了许多推理相关的示意图。
- Roy Frostig 在出版、编辑以及这一路中的许多其他环节提供了帮助。

我们还想感谢在此过程中给予关键反馈的许多人，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 以及许多其他人。感谢 Ruiqi Gao 在 HTML 排版方面提供的帮助。

**谢谢大家！**

<p markdown=1 class="announce">在离开之前，你也许还会喜欢阅读关于 NVIDIA GPU 的新[第 12 部分](../gpus)！</p>

## 延伸阅读 {#延伸阅读}

这里还有不少相关的文章，包括下面这些：

- [**TPU 深度解析**](https://henryhmko.github.io/posts/tpu/tpu.html)：一篇精彩的、与本书风格一致的 TPU 架构深度剖析。
- [**面向 AI 推理的领域专用架构**](https://fleetwood.dev/posts/domain-specific-architectures)：一篇与本书风格一致的硬件与模型深度剖析。
- [**用于训练深度神经网络的领域专用超级计算机**](https://dl.acm.org/doi/pdf/10.1145/3360307)：这是 TPU 的元老级论文之一，其中包含大量关于 Google TPU 计划的精彩细节，本书并未涵盖。
- [**从第一性原理让深度学习跑得飞快**](https://horace.io/brrr_intro.html)：一篇更偏重 GPU 与 PyTorch 的教程，讲解大语言模型的屋顶线与性能工程。
- [**用 Pallas 编写 TPU 内核**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：如今，TPU 编程越来越需要借助 Pallas 编写自定义内核。本系列讨论了如何编写内核，以及许多本文未提及的底层 TPU 细节。
- [**如何优化一个达到 cuBLAS 级别性能的 CUDA 矩阵乘内核：工作日志**](https://siboehm.com/articles/22/CUDA-MMM)：尽管内容针对 GPU 与 CUDA，但这篇优秀的博客展示了如何在 CUDA 中优化一个 matmul 内核。它或许是深入了解 TPU 与 GPU 差异的好素材。
- [**分布式数组与自动并行化**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：这是一份非常棒的 JAX 并行化 API 指南，也是学习如何真正落地我们在此讨论过的某些思想的良好途径。
- [**Rafi Witten 的高性能大语言模型 2024 课程**](https://github.com/rwitten/HighPerfLLMs2024)：我们曾经的同事 Rafi 开设了一门精彩的 TPU 性能工程课程，所有幻灯片都在 GitHub 上。其中对许多内容的讲解比本书更深入。
- [**\[2211.05102\] 高效扩展 Transformer 推理**](https://arxiv.org/abs/2211.05102)：一篇关于 Transformer 推理数理的详尽论文。本文的诸多内容都受其启发。
- [**Huggingface 超大规模实战手册**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：某种程度上可视为本书的 GPU 对应读物，它更深入地讲述了 PyTorch 在训练过程中如何实现各种并行技术与省内存技术。
- [**Transformer 推理算术**](https://kipp.ly/transformer-inference-arithmetic/)：一篇博客，其中包含许多与本书相同的思想，以及一些精彩的图示。
- [**斯坦福 CS336 幻灯片与视频**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：一门精彩的斯坦福课程，涵盖了大语言模型训练与服务的诸多细节，并配有实用的练习。其中作业 1 和作业 2 尤其相关。
- [**Stas Bekman 的机器学习工程手册**](https://github.com/stas00/ml-engineering)：一份高度实用的机器学习基础设施指南，涵盖了本书未涉及的主题，例如如何与云服务商谈判、集群管理，以及对 GPU 吞吐量的实测。
- [**ezyang 的博客**](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)：一位 PyTorch 核心开发者的博客，内容涵盖分片与 PyTorch 的方方面面，包括一篇[PyTorch 内部机制指南](https://blog.ezyang.com/2019/05/pytorch-internals/)和一篇[分片矩阵乘法讲解](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)。这里还有许多其他好内容。

这一领域仍有很大的空间容纳系统性的写作，因此我们希望本文能激励更多此类作品的诞生！我们也相信，这是一个值得深入研究和探索的丰硕领域。在许多情况下，哪怕手头没有大量硬件加速器，也能开展相关研究。

## 反馈 {#反馈}

请留下你的评论或问题，以便我们进一步改进。你可以通过 jacobaustin123 [at] gmail [dot] com 联系我们的通讯作者 Jacob Austin，也可以在[GitHub](https://github.com/jax-ml/scaling-book)上提交 issue、pull request 或 discussion 来建议修改。
