---
layout: distill
title: "Serving LLaMA 3-70B on TPUs（在 TPU 上部署服务 LLaMA 3-70B）"
# permalink: /main/
permalink: /applied-inference-zh/
sitemap: false
description: "我们一起来仔细看看，要如何在 TPU v5e 上部署 LLaMA 3-70B 模型。在不同模型处于屋顶线（roofline）约束下时，部署它们要花多大代价？它们的 KV cache 有多大？我们应该用多大的批大小？推理过程中参数与激活值是如何分片的？我们这就动手，对生产环境中的延迟与吞吐量做一些粗略的估算。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling
next_section_name: "Part 9: Profiling"

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
  - name: "LLaMA 的服务故事是怎样的？"
  - subsections:
    - name: "关于吞吐量的思考"
    - name: "那预填充呢？"
  - name: "可视化延迟-吞吐量权衡"
  - name: "习题"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img
  .fake-img p
---

*本节将探讨部署服务 LLaMA-3 需要什么条件，以及能够做到多高的效率。与前一个"应用"章节一样，请尝试在查阅答案之前，自己用笔和纸把答案推导出来！*

## LLaMA 的服务故事是怎样的？ {#llama-的服务故事是怎样的}

我们先回顾一下 LLaMA 3-70B 的结构（参考 [Section 6](../applied-training)）：

| **超参数**                 | **取值**  |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

我们先从一个简单的问题开始：**我们应该在哪个硬件上部署服务？** 答案基本就是：哪个在 FLOPs/美元 上最便宜就用哪个。<d-footnote>这并不总是成立，有时更关键的是更大的 HBM 或更高的 ICI 带宽，而不是 FLOPs，但这不失为一个好的经验法则。</d-footnote> 因此，我们通常希望在 TPU v5e 上部署服务，它是我们当前专用的推理芯片（价格来自 [Google Cloud pricing](https://cloud.google.com/tpu/pricing)，截至 2025 年 2 月）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元/小时** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每块 TPU v5e 拥有 16GB 的 HBM，这就要求我们对模型进行相当激进的分片。我们先从一些可能对我们重要的基本量入手：

**问题：** LLaMA 3-70B 每个词元的 KV cache 有多大？ *你可以假设我们以 int8 存储它们。这决定了在给定拓扑上我们的批大小能有多大。*

{% details 想清楚之后再点击这里！ %}

LLaMA 3-70B 有 8 个 KV 头，因此每个词元的大小为 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**请注意这有多大！** 如果我们的序列长度为 32k 词元（这很常见），这会占用 `160e3 * 32,768 = 5.3GB / sequence`。对于批大小 BS=240，这就是 1.3TB！由于每块 TPU v5e 只有 16GB，要装下这么多内存，我们大约需要 `(70e9 + 1.3e12) / 16e9 = 86` 块 TPU v5e 芯片。另外请注意，与 70GB 的模型参数相比，这个数值是多么庞大。

{% enddetails %}

**问题：** 假设我们想在批大小 32、序列长度 8192 的条件下部署服务 L3 70B，且所有内容（参数和 KV）都用 int8。这会占用多少总内存？我们能在其上部署服务的最小切片是多大？

{% details 答案 %}

由于我们的 KV 在 int8 下为 `160e3` 字节，我们的 KV 总内存为 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数为 `70e9` 字节，因为每个参数占 1 字节。因此，我们的总内存占用为 `41.9e9 + 70e9 = 112GB`。

我们能使用的最小切片需要 `112e9 / 16e9 = 7` 块 TPU，或者（取整到偶数规模）TPU v5e `4x2`。这会非常紧凑，考虑到其他开销，我们可能还无法完全装下，因此我们至少需要一个 `4x4`（或者减小批大小）。

{% enddetails %}

**问题：** 在这种批大小和量化条件下，在 TPU v5e `4x2` 上，我们预期每个解码步的延迟大约是多少？吞吐量（tokens/sec/芯片）又是多少？`4x4` 呢？ *假设我们以 bfloat16 执行 FLOPs，且所有内容都完全分片。*

{% details 答案 %}

我们可以套用上一节的公式：

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

这里我们的临界批大小约为 120，因为我们的参数用 int8 存储，但 FLOPs 用 bfloat16 计算。我们也可以手动计算右边的最大值，但那基本上就是我们之前已经做过好几次的计算了。**因此，对于我们的 matmul 和 FLOPs 来说，我们都处于明显的内存受限区间。**

严格地从内存带宽来看，我们的步时间基本为 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.2e11) = 17ms`。**因此理论上我们的步时间约为 17ms。** 我们的吞吐量为 `32 / .017 = 1882 tokens / sec`，即 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个需要注意的地方，就是检查我们的 matmul 是否可能受 ICI 限制。在这里我们可以分配 2 个轴给它，因此理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时我们才会受 ICI 限制，所以我们毫无压力！

如果我们在 `4x4` 上运行，ICI 方面依然没问题，因此我们的延迟会降到 `17 / 2 = 8.5ms`，但每芯片的吞吐量保持不变。

{% enddetails %}

### 关于吞吐量的思考 {#关于吞吐量的思考}

我们花点时间单纯思考一下吞吐量。当我们为吞吐量做优化时，我们希望处于算力受限状态，也就是尽可能利用 TPU 矩阵乘法单元（MXU）的全部算力。通常这意味着我们希望批大小尽可能大，从而完成尽可能多的工作。

**问题：** 在 TPU v5e 上，使用 bfloat16 权重和激活值，要让 matmul 处于算力受限状态，我们的批大小需要多大？如果我们用 int8 权重但以 bfloat16 执行 FLOPs 呢？用 int8 权重加 int8 FLOPs 又如何？

{% details 答案 %}

如第 7 节所述，对于任意满足 $B \ll D, F$ 的 bfloat16 matmul，我们有

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

当我们的权重为 int8 时，分母会损失一个 2 倍的因子，因此我们有 $2BDF / DF = 2B > 240$，即 $B > 120$，为之前临界批大小的一半。这对我们非常有利！当我们使用 int8 权重和 int8 FLOPs 时，我们必须使用 int8 对应的 TPU FLOPs/s 值，它从 bfloat16 的 1.97e14 变为 3.94e14，几乎翻倍。这意味着我们又回到了起点，即大约 $B > 240$。

int8 权重加 bfloat16 FLOPs 的情况相当常见，因为对参数做无损量化通常比进行低精度算术运算更容易。

{% enddetails %}

**问题：** 使用 bfloat16、int8 和 int4（KV 和参数均如此）且上下文长度为 8k，我们能在多小的 TPU v5e 拓扑上部署服务 LLaMA 3-70B？ *这里你可以认为 KV cache 小到可忽略。*

{% details 答案 %}

这很简单！如果我们能接受很小的批大小，那么唯一的限制就是把参数内存装进 HBM，也就是说，它只是 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，或者 `ceil(70e9 * sizeof(dtype) / 16e9)`，再四舍五入到最接近的合理拓扑（2 的某个倍数）：

| dtype | 参数大小 | 每词元 KV 大小（字节） | 最少 TPU v5e 数 | 实际最小切片 | 剩余可用于 KV cache 的 HBM | 8k 下的 KV cache 数量 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             29              |         43         |

这相当酷！它告诉我们，如果有需要，可以把 LLaMA 70B 装进一个 TPU v5e 2x2。只是你会注意到 KV cache 的数量非常小。那就是我们的批大小！这意味着我们的 FLOPs 利用率会非常糟糕。我们会很乐意使用更大的拓扑，从而把批大小推到 240。

{% enddetails %}

**问题：** 假设我们使用能装进这些拓扑的最大批大小，我们预期每个生成步的延迟是多少？

{% details 答案 %}

这也很简单，因为我们挑选批大小就是为了填满所有 HBM！这只是一个把相当于一整块 TPU v5e 的字节数加载进 MXU 需要多长时间的问题。也就是 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`，因此这是 **19ms/步**。假设我们的生成中位长度为 512 词元，那每次解码大约为 9s。注意，使用更小的批大小可以获得略微更好的延迟，例如如果我们只看 int4 的模型参数，由于 HBM 不再被填满，我们的最小延迟约为 10ms/步。

{% enddetails %}

<p markdown=1 class="takeaway">**要点**：我们总可以通过"把所有模型参数从 HBM 加载进 MXU 需要多长时间"来为解码延迟给出下界。当我们的 KV cache 很小时，你可以把每一层想象成只是逐块加载权重，然后丢弃它们。除非我们使用很大的批大小或大量的设备间通信，否则这通常是一个合理的界限（误差在 1.5 倍以内）。当我们的批大小更大时，我们还需要对 KV cache 加载建模，因为它会主导参数所占的部分。</p>

同样地，在 FLOPs 受限区间（例如训练或大批次推理），我们可以使用 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ 这个下界，它假设没有通信。

**问题：** 对以上每种情况，这能给我们带来多少每芯片吞吐量（以 queries/芯片 计）？ *你可以假设我们的中位解码长度为 512 词元。*

{% details 答案 %}

这是一个重要的问题，因为它与每词元成本完全相关。

基于我们对中位解码长度的假设，我们的吞吐量就是 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approx 43 / (0.019 * 512 * N)$$。这大约给出 $$(4.42 / N)$$ QPS，因此代入 $$N$$ 我们得到：

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

请注意，这是相当乐观的，因为它完全忽略了前向传播的工作内存（分配给激活值和注意力的内存）。在 Flash Attention 下这并非毫无道理，但也不现实。真实的数值可能约为这个的一半。要获得绝对最大的吞吐量，我们可能希望把芯片数量增加一倍以上，同时显著增大批大小。

{% enddetails %}

**问题：** 如果对上述每个例子都把拓扑翻倍，我们的峰值吞吐量会如何变化？

{% details 答案 %}

如果我们在 bfloat16 下使用 4x8 切片，我们将有 372GB 留给 KV cache，这能让我们将批大小增加到 140。然后由于我们的步时间保持不变，我们将获得 `14.39 / num_chips` 的吞吐量，即

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.44    |
|   int8 (on 4x4)   |    0.90    |
|   int4 (on 2x4)   |    1.80    |

进一步的增大将带来更大的收益！最重要的结论是：**在受 KV cache 大小限制的情况下，最小拓扑并不总是性能最高的拓扑**。

{% enddetails %}

**问题：** 现在我们深入探讨分片的问题。假设我们想在 TPU v5e 4x8 上以 bfloat16 部署服务。在生成阶段，我们在 TPU v5e 4x8 上会对模型采用怎样的分片？我们能否避免受通信限制？

{% details 答案 %}

如上一节所述，在生成阶段，我们真正用于分片的选项只有一个：模型并行。在我们受通信限制之前，能做到什么程度？正如上一节所讨论的，我们的模型大致在以下条件下受通信限制：

$$Y > \frac{F \cdot M_Y}{2200}$$

对于 LLaMA 3-70B，我们有 `F = 28,672`，因此如果我们做 2 个轴的模型分片，这大约给出 $$Y = 28672 \cdot 2 / 2200 = 26$$，所以一般来说，我们最多可以把规模扩大到大约 16 块芯片而不受通信限制，这让我们能用 `4x4` 但不能用 `4x8`。通常，由于我们不能完美地让计算与通信重叠，即便是这个估计也过于乐观了。

**要点：我们实际上无法用纯模型并行在 4x8 上部署服务。** 我们在这里最多能做到 4x2，或者_也许_ 4x4。

然而，正如我们之前讨论过的，当我们的批大小较小时，我们往往可以做更多的模型并行而不会显著损害吞吐量，因为我们的模型是内存带宽受限而非 FLOPs 受限的。我们之前说过，这个值大约是 $Y=F / (8\cdot B)$，因此如果我们用批大小 64，理论上在受 ICI 限制之前，我们可以将模型并行度提高到 `Y = 28,672 / (8 * 64) = 56`。为了验证这一点，我们可以查看单个 matmul 的 $T_\text{ici comms}$、$T_\text{hbm comms}$ 和 $T_\text{math}$。显然我们有：

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

对于 `4x8`，这会给我们 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`、$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.2e11) = 18us`、$T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，因此理论上我们仍然受 HBM 带宽限制，这非常好！*请注意，从 `4x4` 扩大到 `4x8` 从吞吐量角度看可能并没有帮助，但它会降低我们的延迟！*

如果我们看 int8 和 int4 配置，我们_可以_用纯模型并行来做。因此我们到达了这样一个节点：量化实际上给了我们超越更快 FLOPs 的有意义优势——它让我们能在受通信限制之前使用更大的批大小。**所以这个故事的结局是：我们无法在 4x8 上达到峰值吞吐量，但对于 int8 和 int4 配置，我们可以用纯模型并行。**

{% enddetails %}

<p markdown=1 class="takeaway">**提示**：有用的模型并行的最大程度取决于 $$d_{ff}$$ 以及你对模型进行分片所跨的轴数。该最大值通常介于 8 到 32 之间，具体取决于模型大小。你可以超出这个限制来扩大规模，以一定的吞吐量代价换取更低的延迟。</p>

### 那预填充呢？ {#那预填充呢}

我们在这里基本忽略了预填充，因为它要简单得多。我们不妨把几个概念结合起来，思考一下端到端的图景。

**问题：** 假设我们在预填充阶段达到 40% 的模型浮点利用率（MFU）。在 16 块 TPU v5e 芯片上，长度为 8192 的预填充需要多长时间？

{% details 答案 %}

在 8k 词元下，我们稳稳地处于算力受限状态，因此我们只需要从 FLOPs 角度推理。我们知道模型有 `70e9` 个参数，因此每次前向传播使用 `2 * 70e9 * B` FLOPs。假设 40% 的模型浮点利用率（MFU），这给出大约 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s` 的运行时间。与之前我们看到的那些数字相比，这其实相当大！

{% enddetails %}

**问题：** 假设我们的中位预填充长度为 8192 词元，中位解码长度为 4096 词元。假设我们的生成批大小为 32。平均每一步有多少条序列完成解码？平均每一步从我们的 KV cache 中驱逐多少个词元？

{% details 答案 %}

这有点直截了当。由于我们的中位解码长度为 4096 词元，一条序列大约每 1 / 4096 个词元就会完成一次。给定批大小为 32，这意味着我们每步驱逐 `32 / 4096` 条序列。由于我们的 KV cache 长度大约为 `8192 + 4096`，这就是 `32 * (8192 + 4096) / 4096 = 96` 个词元每步被驱逐。通用公式为 $B * (P + G) / G$，其中 $P$ 和 $G$ 分别是预填充和生成的长度。

{% enddetails %}

**问题：** 假设我们进行分离式服务，中位预填充长度为 8192，中位解码长度为 512。假设采用上面在 bfloat16 下计算得到的预填充和生成延迟。为了让预填充与生成服务器都保持充分饱和，你需要的预填充:生成服务器比例是多少？

{% details 答案 %}

这是一个很有趣的问题。令 $P$ 为预填充服务器的数量，$G$ 为生成服务器的数量。大致来说，这是一个流水线问题：我们以 `P / prefill_latency` 的速率把序列喂入，并以 `B * G / (generate_latency * median_decode_length)` 的速率消费它们。我们之前计算过每步预填充 `910ms`、每步解码 `19ms`（批大小 43，我们把它当作 32）。因此我们需要 `P / 0.91 = 32 * G / (0.019 * 512)`，即 `P = 3G`，也就是说我们大约需要 3 倍于生成服务器的预填充服务器！

{% enddetails %}

## 可视化延迟-吞吐量权衡 {#可视化延迟-吞吐量权衡}

暂时仍以 LLaMA 70B 为例，我们实际来看一下生成阶段不同批大小下的延迟和吞吐量。正如我们在上一节针对 PaLM 模型所展示的，这为我们给出了一条吞吐量/延迟的帕累托前沿。我们假设采用 16 路张量并行，因为这是我们在 MLP 块中保持算力受限时所能使用的合理上限。我们这里将使用 TPU v5e 4x4 拓扑。**滑块控制序列长度，这样你可以看到更大 KV cache 的影响。**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **看看成本与延迟之间的权衡有多剧烈。** 以每词元延迟翻倍为代价，我们可以将每词元成本降低大约 100 倍。此外，我们的延迟范围很广，在小批大小下为 5.5ms，在极大批次下可达 20ms。
* 注意，在 2k 上下文下，当达到 BS 120 屋顶线（roofline）（这里为 120，是因为我们使用 int8 权重但 bf16 FLOPs）时，吞吐量实际上在约 1 token/ms/芯片 处趋于平稳。然而，随着序列长度增加，我们不再能把该批大小装进内存，因此我们永远无法达到完全饱和的点。
* 注意，在相同的吞吐量下，大批次的延迟要高得多，因为 KV 加载变得占主导（而非参数加载）。

我们可以通过把成本和延迟的来源拆解为参数加载时间、KV 加载时间和 FLOPs 时间，来更好地理解这一点。红色扇形区域是我们预期在 MLP 块中处于算力受限状态的区域。

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

这讲述了一个相当清晰的故事。你可以看到，最初参数加载占据了延迟的绝大部分，直到批大小变得足够大，FLOPs 和 KV 加载才变得更重要。值得注意的是，在所有大于 2048 的序列长度下，我们花在 KV cache 加载上的时间都多于花在 FLOPs 上的时间！**因此，虽然我们可以通过增大批大小来提高硬件利用率，但在长上下文长度下，KV 加载始终主导着总的步时间。**

<p markdown=1 class="takeaway">**要点：** 对于 LLaMA 3-70B，在几乎所有这些配置下，我们都强烈受 KV cache 内存带宽限制（以及 HBM 限制），这凸显了减小 KV cache 大小对生成吞吐量有多么重要。同时请注意，这里的延迟/吞吐量权衡依然如此剧烈。</p>

{% details 计算这些的代码相当简单。 %}

下面是计算这些屋顶线（roofline）的代码：

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
bytes_per_param = 1  # int8 means 1 byte per param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

注意我们是如何非常明确地把延迟拆成两个来源——KV 加载和参数加载——以及延迟是如何受 FLOPs 或通信限制（取决于哪个更大）的。

{% enddetails %}

## 习题 {#习题}

这里有几道习题。其中一些重复了上面已经算过的内容，但可能在教学上很有用。

**问题 1：** LLaMA 3-405B 每次前向传播每个词元使用多少 FLOPs？假设我们处于 FLOPs 受限状态，在 TPU v5e 上 N 块芯片上单次前向传播的下界是多少？如果我们处于通信受限状态呢？ *忽略模型无法装进单块芯片这一事实。*

**问题 2：** 假设我们想以 BS240 部署服务 LLaMA 3-8B，使用 int8 权重和 int8 KV cache。（a）模型参数、（b）KV cache、（c）峰值工作激活值（大概）各占用多少字节？我们能在多小的拓扑上运行它？

**问题 3：** 你该如何在 TPU v5e 上部署服务 LLaMA 3-405B？假设 int8 权重和 bfloat16 FLOPs。假设我们有一个 15ms/词元 的硬性限制，我们能达到的最高吞吐量配置是什么？理论最小步时间是多少？

<h3 markdown=1 class="next-section">第 8 部分到此结束！关于第 9 部分——深入 XLA 与 TPU 性能分析，请点击[这里](../profiling)。</h3>
