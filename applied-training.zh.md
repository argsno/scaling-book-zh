---
layout: distill
title: "Training LLaMA 3 on TPUs（在 TPU 上训练 LLaMA 3）"
# permalink: /main/
description: "我们一起来仔细看看，要如何运用上一节学到的知识，在 TPU v5p 上训练 LLaMA 3 模型。这些模型有多大？在不同配置下训练要花多大代价？它们是如何分片的？我们这就动手，做一些粗略的估算，看看前面几节的内容如何映射到真实的模型上。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training-zh"
previous_section_name: "第 5 部分. 如何对 Transformer 进行训练并行化"

next_section_url: "../inference-zh"
next_section_name: "第 7 部分. Transformer 推理全解"

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
  - name: "What does LLaMA 3 look like?"
  - name: "Counting parameters and FLOPs"
  - name: "How to shard LLaMA 3-70B for training"
  - name: "Worked Problems"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
permalink: /applied-training-zh/
sitemap: false
---

_本节的目标是运用前一节的结论来解决一个非常实际的问题：训练 LLaMA 3 系列模型（herd）。与前几节不同的是，我们希望你亲自动手完成其中大量工作。因此，我们把每一节的解答都隐藏了起来，以便你先尝试自己作答。不妨拿起笔试着手动推导一下！_

### What does LLaMA 3 look like? {#what-does-llama-3-look-like}

LLaMA-3 模型家族<d-cite key="llama3"></d-cite> 包含 3 个主要模型：LLaMA 3 8B、70B 和 405B。我们主要聚焦于 70B，将 8B 和 405B 留给你在末尾的习题部分自行探索。以下是 LLaMA 3-70B 的架构，取自 LLaMA 的 [HuggingFace 页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数（hyperparam）**    | **数值（value）** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了说明这些信息有多容易获取，下面给出配置文件本身，以及一份对应关系：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_建一张大表，把这些数字填入许多不同的开源 LLM 中，会很有帮助，这样你就能快速比较它们所做的设计决策。_

### Counting parameters and FLOPs {#counting-parameters-and-flops}

**问题：** 从这张表中，我们能否算出 LLaMA 3-70B 的参数数量？🤫 让我们应用 [第 4 节](../transformers) 的内容，看看能否得到 70B！

| 参数（param）        | 公式（formula）                                                                                                                                   | 数量（count）                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 前馈参数（FFW params）       | d_model * d_ff * 3 (for SwiGLU gate, up, and down projections) * n_layers                                                                                         | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| 词表参数（Vocab params）     | 2 (input and output embeddings) * n_embeddings * d_model                                                                                          | 2 * 128,256 * 8,192 = **2.1e9**                              |
| 注意力参数（Attention params） | n_layers * [ 2 (for q embedding and concatenated output projection) * d_model * n_heads * d_qkv + 2 (for k and v) * d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

太好了！我们得到了预期的数字。你会注意到，正如所料，前馈（FFW）参数在整体参数数量中占据了绝对主导，不过注意力参数也不可忽略。

<p markdown=1 class="takeaway">**要点**：MLP 模块中的 3 个大型权重矩阵，其规模远大于 Transformer 中所有其他数组，以至于我们在估算模型内存或 FLOPs 时，几乎可以忽略所有其他参数。对于 LLaMA 3-70B，它们占了 70B 参数中的 56B。</p>

接下来我们看看 FLOPs！*请记住 [第 4 节](../transformers) 中关于训练的通用规则。*

**问题：** LLaMA-3 每个训练步、每个词元要执行多少 FLOPs？_这有助于我们判断整个训练过程会有多昂贵。_

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：如 [第 4 节](../transformers) 所示，每个词元我们大约执行 $$6 \cdot \text{param count}$$ 次 FLOPs，因此这里大约是 `6 * 70e9 = 4.2e11` FLOPs / 词元。也就是每个词元、每步大约半 TFLOP。如果我们处于算力受限状态，在单个 TPU v5p 芯片上、假设浮点运算利用率达到完美，这大约需要 `4.2e11 / 4.59E+14 = 1ms`。

{% enddetails %}

**问题：** LLaMA 3 大约以 15 万亿词元进行训练。那么总共是多少 FLOPs？

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：这很简单，就是 `4.2e11 * 15e12 = 6.3e24 FLOPs` 总量。即 6.3 yottaFLOPs。这是个庞大的数字！在单个 TPU 上，这将花费 `6.3e24 / 4.59E+14 = 435 年`。同样是个庞大的数字！

{% enddetails %}

**问题：** 假设我们想在一个完整的 TPU v5p Pod 上训练，包含 16x20x28 = 8960 个芯片。如果处于算力受限状态、在 bfloat16 下达到 40% 的模型浮点利用率（MFU），这大约需要训练多久？

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：我们知道每个 TPU v5p 可以执行 4.59e14 FLOPs / 秒。在 40% MFU 下，大约需要 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 秒`。**这大约是 44 天！** 假设我们确实能达到 40% MFU，这个数字还算合理。

{% enddetails %}

**问题：** LLaMA 3-70B 以约 4M 词元的批大小进行预训练。要在这种批大小下训练，我们至少需要多少个 TPU？_你可以假设参数为 bfloat16、优化器状态为 float32，并且每层对梯度做 4 次检查点。_

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：这个问题主要问的是内存用量，因为这是对可用算力的唯一硬性约束。在训练过程中，HBM 有三大主要用途：模型参数、优化器状态和梯度检查点。如果我们假设权重为 bfloat16、优化器状态为 float32，并采用一种_非常_保守的梯度检查点方案（每层 4 次），那么：

| **参数（Params）** | 2 * 70GB | ~140GB |
| **优化器状态（Optimizer State）** | 8 * 70GB | ~560GB |
| **梯度检查点（Gradient Checkpoints）** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计（Total）**                |                         | ~21.6TB |

这里的总量约为 21.6TB。你会注意到，即便采用非常保守的检查点方案，梯度检查点化仍然在内存占用中占据绝对主导。从技术上讲，我们可以降到每层 1 个检查点，或者采用微批次（microbatch），但这已经是一个合理的图景。基于这些假设，由于每个 TPU v5p 拥有 96GB 的 HBM，我们需要 `21.6e12 / 96e9 = 225` 个 TPU。其实这个数量并不多！

*那我们为什么不这样做呢？* 因为那样训练将花费我们 `44 天 * 8960 / 225 = 1752 天`。这接近四年。**这可是相当长的时间。** 不过，这清楚地表明：我们使用这些大型集群，并非因为内存受限，而是因为我们需要额外的 FLOPs。

{% enddetails %}

**问题：** 在与上一题相同的假设下，如果我们使用 8960 个 TPU v5p 芯片，每芯片将使用多少内存？

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：我们的总内存仍然约为 21.6TB，因此每芯片大约会使用 2.4GB，这基本可以忽略不计。即便我们采用激进得多的检查点方案，例如每层 12 个检查点，每芯片也才仅仅 8GB。在这种规模下，训练期间我们离内存受限还差得远。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：在技术上，即便在非常小的拓扑上，也可以训练甚至非常大的模型，前提是它们可能会花很长时间。能够计算出一次训练运行的总 FLOPs，使我们得以通过假设一个 modest 的 MFU 和一个已知的拓扑，对其训练时间做出大致估计。</p>

### How to shard LLaMA 3-70B for training {#how-to-shard-llama-3-70b-for-training}

我们沿用上面的设定，假设要在 8960 个芯片的 TPU v5p Pod 上，以 4M 词元的批大小（每批次 1024 条长度为 4096 的序列）训练 LLaMA 3-70B。我们来讨论一下这个模型的最佳分片策略。

**问题：** 在上述假设下，我们能否仅用全分片数据并行（FSDP）来训练我们的模型？首先，假设我们不能做任何序列/上下文并行。_这应该是你首先想到的思路，因为它很简单，而且如果可行，不会引入额外的通信。_

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：这个答案会有点较真。如前所述，LLaMA 3-70B 最初以长度为 4K 的序列训练，因此 4M 词元的批大小给出的*序列批大小*为 1024。这意味着我们最多只能做纯数据并行/FSDP 直到 1024 个芯片，_因为这正是我们要在其上进行数据并行的序列数量_。所以，在"完全数据并行、无额外通信"这个简单意义上，答案是不能。下一个问题会回答一个不那么较真的版本。

{% enddetails %}

**问题：** 我们放宽"不做任何序列分片"的要求。如果我们允许自己在批次轴_和_序列轴上都进行 FSDP，能否仅用 FSDP 在 8960 个芯片上训练 LLaMA 3-70B？

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：既然我们现在允许自己做序列/上下文并行，我们就可以扩展得多得多。首先计算每个设备的批大小。如果我们做 8960 路 FSDP，最终每 TPU 的批大小为 `4 * 1024 * 1024 / 8960 = 468 词元`。根据前一节我们知道，当 $$\text{每设备批大小} < 2550 / M_X$$ 时，我们会因 FSDP 而受 ICI 限制。由于我们在一个完整的三维 Pod 中可以投入 3 个轴，这给出的下界是 850，而我们的数值远低于它。**所以答案是否定的，即便有 3 个轴也不行。我们将稳稳地处于通信受限状态。**

{% enddetails %}

**问题：** 现在我们来考察张量并行与 FSDP 的混合。是否存在某种组合能让我们保持算力受限？如果存在，我们该做多少 FSDP 和张量并行？

{% details 点击此处查看答案（在你思考之后！） %}

**答案**：先检查一下这到底能不能装下。我们知道，当每芯片批大小小于 $2550^2 / 2F = 113$ 时，我们会受通信限制。正如我们上面看到的，我们略高于这个值。所以很好！现在要选出最优的 FSDP 数量，我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

取整到一个合理的 2 的倍数，这大约给出 2048 路 FSDP 和 4 路张量并行。这应该能很好地工作！

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们可以在完整的 TPU v5p Pod 上，以数据并行（1024 路）、序列并行（2 路）与张量并行（4 路）的混合方式，训练 4M 词元批大小的 LLaMA-3，而不会受通信限制。如果我们尝试纯 FSDP 或 FSDP + 序列并行，就会受通信限制。我们在前一节推导出的那些方程非常实用。</p>

## Worked Problems {#worked-problems}

**问题 1 [将 LLaMA 70B 扩展到更多芯片]：** 假设我们想以相同的批大小在 4 个 Pod 上训练 LLaMA 3-70B。我们会采用哪种并行方案？我们会处于算力受限还是通信受限状态？训练大约需要多久？*务必使用正确的屋顶线（roofline）界限。*

**问题 2 [LLaMA 405B]：**

(a) 使用 LLaMA 3-405B 的 [config](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)（一个受限模型，因此你可能需要登录并申请权限才能查看），仿照上文写一张包含所有关键超参数的表。该模型总共有多少参数？每步训练需要多少 FLOPs？如果我们以 15T 词元训练，总共要执行多少 FLOPs？

(b) 假设我们想在 8 个 TPU v5p Pod 上训练。我们会采用哪种并行方案？训练需要多久？我们会处于算力受限还是通信受限状态？

<h3 markdown=1 class="next-section">第 6 节到此结束。关于 Transformer 推理的第 7 节，请点击[此处](../inference)。</h3>
