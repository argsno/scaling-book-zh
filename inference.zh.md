---
layout: distill
title: "All About Transformer Inference（Transformer 推理全解）"
permalink: /inference-zh/
description: "在 Transformer 上执行推理，可能与训练大相径庭。这部分是因为推理带来了一个需要考虑的新因素：延迟。在本节中，我们会从「自模型中采样出单个新词元」起步，一路讲到如何高效地把一个大型 Transformer 扩展到许多加速器切片之上，使其成为推理引擎的一部分。"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training-zh"
previous_section_name: "第 6 部分. 在 TPU 上训练 LLaMA 3"

next_section_url: "../applied-inference-zh"
next_section_name: "第 8 部分. 在 TPU 上部署服务 LLaMA 3-70B"

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
  - name: "Transformer 推理基础"
  - subsections:
    - name: "我们究竟要优化什么？"
    - name: "线性操作：什么在限制我们？"
    - name: "那注意力呢？"
    - name: "LLM 延迟与吞吐量的理论估计"
    - name: "那内存呢？"
    - name: "为 LLaMA 2-13B 建模吞吐量与延迟"
  - name: "改善生成吞吐量与延迟的技巧"
  - name: "在多个加速器上分布推理"
  - subsections:
    - name: "预填充"
    - name: "生成"
    - name: "对 KV cache 分片"
  - name: "设计一个高效的推理引擎"
  - subsections:
    - name: "连续批处理（Continuous batching）"
    - name: "前缀缓存（Prefix caching）"
    - name: "来看一个实现：JetStream"
  - name: "习题"
  - name: "附录"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Transformer 推理基础 {#transformer-推理基础}

假设你已经训练好了一个 Transformer，并想用它来生成一些新的序列。_说到底，基准分数上涨、损失曲线下降，都只是"真正上路之后会不会发生什么有趣的事"的某种代理指标！_<d-footnote>历史上，你可以在完全不碰推理的情况下做大量 Transformer 研究——基于打分的多种选择题基准，无需一个真正的 KV cache 或生成循环实现也能高效运行。这意味着，尤其是在研究代码库中，推理代码路径往往存在大量唾手可得的优化空间。</d-footnote>

从概念上讲，采样很简单。我们输入一个序列，我们最爱的 Transformer 就会吐出 $$\log p(\text{next token}_i \vert \text{previous tokens})$$，也就是所有可能下一个词元上的对数概率。我们可以从这个分布中采样，得到一个新的词元。把这个词元追加进去并重复这个过程，我们就得到了一个序列，它是提示词的延续。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>Figure:</b> 对 Transformer 的朴素采样。蓝色的 logits 给出了下一个词元上的分布，我们可以从中采样。注意每一步都会重新处理整个前缀，导致算法的运行时间为 $\Theta(n^2)$。"%}

我们刚才描述的是 Transformer 采样的朴素实现，虽然它能工作，但**在实践中我们绝不会这么做**，因为每生成一个词元我们都在重新处理整个序列。这个算法生成 $$n$$ 个词元，在 FFW（前馈）上的复杂度是 $$O(n^2)$$，在注意力机制上是 $$O(n^3)$$！

**我们如何避免这一点？** 与其每次都做完整的 forward pass，事实证明我们可以保存每次 forward pass 的一些中间激活值，从而避免重新处理之前的词元。具体来说，由于在点积注意力中，某个词元只 attend 到之前的词元，我们可以简单地把每个词元的 key 和 value 投影写入一个名为 **KV cache** 的新数据结构。一旦我们为过去的词元保存了这些 key/value 投影，未来的词元就能直接计算它们的 $$q_i \cdot k_j$$ 乘积，而无需对更早的词元做任何新的 FLOPs。太棒了！

有了这个认识，推理有两个关键部分：

* <b style="color: red;">预填充（Prefill）</b>：给定一个长提示词，我们同时处理提示词中的所有词元，并把得到的激活值（具体地说，即 key-value 投影）保存在一个 **"KV cache"** 中。我们也保存最后一个词元的 logits。
* <b style="color: blue;">生成（Generation）</b>：给定一个 KV cache 和之前的 logits，我们以一个词元为单位增量式地从中采样，把这个词元喂回 Transformer，并为下一步生成一组新的 logits。我们也把这个新词元的 KV 激活值追加到 KV cache 中。我们重复这一过程，直到遇到一个特殊的 `<EOS>` 词元，或达到某个最大长度限制。

下面是带 KV cache 的采样示意图：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>Figure:</b> 带 KV cache 的高效 Transformer 采样示意图。<b style='color: red;'>预填充（Prefill）</b>处理我们的提示词，并把每个词元的 key-value 激活值全部保存在一个 cache 中。<b style='color: blue;'>生成（Generation）</b>接收这个 cache（以及最后一个词元的 logits），采样一个新词元，并让它通过模型，attend 到 KV cache，同时把新词元的 key-value 投影存回 cache。在 MLP 块中，这是一个 $O(n)$ 的算法。"%}

通过使用带 KV cache 的采样，我们把生成 $$n$$ 个词元的时间复杂度降到了 FFW 上 $$O(n)$$、注意力上 $$O(n^2)$$，因为我们从不重新处理之前的词元。然而，生成一个序列仍然需要很多次 forward pass——这正是你查询 Gemini 或 ChatGPT 时结果流式返回给你时发生的事。每个词元（通常）都是一次独立的（但部分被缓存的）对巨型 Transformer 的调用。

我们很快就会看到，<b style="color: red;">预填充（prefill）</b> 和 <b style="color: blue;">生成（generation）</b> 是完全不同的两种"猛兽"——Transformer 推理其实是披着一张皮的两个任务！相比训练，KV cache 也是一个全新且重要的复杂度来源。

### 我们究竟要优化什么？ {#我们究竟要优化什么}

在继续之前，值得先强调推理中一个全新的方面：延迟（latency）。训练时我们只关心吞吐量（**每颗芯片**每秒处理的总词元数），而推理时我们不得不担心生成词元的速度（包括**首词元延迟（TTFT）**和**每词元延迟**）。例如：

* **离线批量推理**（用于评测和数据生成）只关心推理的总体成本，而对单个样本延迟毫无感知。
* **聊天界面/流式任务**需要在大规模下以低成本运行，同时具备较低的 TTFT，并且生成词元的速度要快到超过人类阅读速度。
* **边缘推理**（例如你笔记本上的 `llama.cpp`）只需要以尽可能低的延迟一次服务一个用户，可能还伴随着严苛的硬件限制。

最大化硬件利用率仍然至关重要，并且有助于降低成本与 TTFT，但和训练不同，它并不*必然*在所有情境下都转化为单个用户更好的体验。在加速器、系统以及模型架构等多个层面上的许多优化，都是在延迟、吞吐量、上下文长度乃至模型质量之间做权衡取舍。

### 更细粒度地看待 Transformer {#更细粒度地看待-transformer}

到目前为止，我们基本把 Transformer 当作一堆前馈块的堆叠。虽然从 FLOPs 和内存的角度看这通常合理，但它不足以恰当地建模推理。<d-footnote>你会在本节各处注意到，推理比训练要"不留情面"得多。我们通常拥有少得多的 FLOPs、少得多的批处理机会，以及对延迟高得多的敏感度。KV cache 也让推理大幅复杂化。</d-footnote> 正如我们在 [Part 4](../transformers) 中所见，一个 Transformer forward pass 的主要组成部分是：

1. **一堆线性操作**，包括 MLP（$W_{in}$、$W_{out}$）以及注意力的 QKV 投影和输出投影（$W_Q$、$W_K$、$W_V$ 和 $W_O$）。这些都涉及从 HBM 读取参数和一批激活值，做若干 FLOPs，再把结果写回 HBM。
2. **点积注意力**。我们需要从 HBM 读取一批 key-value 投影和一批 query 激活值，做几次内积和一些 softmax 操作，再把注意力结果写回 HBM。
3. **其他一切**，包括应用层归一化、激活函数、词元采样、更新 KV cache，以及位置编码。这些确实消耗一些 FLOPs，但都被上面二者所主导，或被融合进上面二者之中。

在接下来的几节里，我们将在预填充和生成的语境下分别考察这三者，并思考什么可能成为我们性能的瓶颈。在单颗加速器内部，我们是算力受限还是内存受限？我们想强调的是，对于预填充与生成，答案会有多么不同。

### 线性操作：什么在限制我们？ {#线性操作-什么在限制我们}

我们所有的线性操作在概念上都是相同的，无论它们位于 MLP 块还是注意力中。它们的算术强度取决于批大小。我们在 [Section 1](../roofline) 中做过这个计算，但值得重复一遍。来看一个 $\text{bf16[B, D]}$ 的批次乘以一个 $\text{bf16[D, F]}$ 矩阵的矩阵乘。它可以是大的 MLP 块（$W_\text{in}$ 或 $W_\text{out}$），也可以是较小的某个注意力投影（$W_Q$、$W_K$、$W_V$、$W_O$）。要做这个 matmul，我们需要把这两个数组从 HBM 载入矩阵乘法单元（MXU），做乘法，再把结果写回 HBM。和之前一样，我们有：

$$T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} = \frac{2BDF}{\text{Accelerator FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} = \frac{2BD + 2FD + 2BF}{\text{Bandwidth Bytes/s}}$$

TPU 或 GPU 可以在做计算的同时进行载入来让这两者重叠，因此要算力受限，我们需要 $$T_\text{math} \geq T_\text{comms}$$，即：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中 RHS 是我们硬件的算术强度。现在假设 $D$ 和 $F$ 都远大于 $B$（通常我们的批次最多 500，而 $D$ 和 $F > 10k$），我们可以利用 $\small{2BD + 2DF + 2BF \approx 2DF}$ 这一事实来简化分母，得到

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果我们对权重做量化，或在矩阵乘中使用更低精度的 FLOPs，这个临界批大小会变化。例如，如果我们把权重量化到 int8 或 fp8，$B_\text{crit}$ 会减半。如果我们在 int8 或 fp8 下做 FLOPs，$B_\text{crit}$ 会翻倍。因此，如果我们令 $\beta = \text{bits per param} / \text{bits per activation}$、$\alpha_\text{hbm} = C / W_\text{hbm}$，那么我们的临界批大小实际上是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点：** Transformer 的 matmul 是算力受限的*当且仅当*每副本的**词元**批大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{bits per param} / \text{bits per activation}) = \beta \cdot \alpha_\text{hbm}$。对于 TPU v5e 上的 bf16 激活值，这个值是 240 个词元。对于 H100，约为 280 个词元。</p>

在训练时，因为我们在一个非常大的批次上复用同一份权重，我们所有的矩阵乘都有很高算术强度。**这种高算术强度会延续到预填充，因为用户提示词通常长达数百乃至上千个词元。** 如前所述，TPUv5e 的硬件算术强度是 240，因此如果把一个长于 240 词元的序列喂入在这套硬件上以 bf16 运行的稠密模型，我们会预期它是算力受限的，一切安好。比这更短的提示词在技术上可以批处理到一起以达到更高的利用率，但通常没有必要。

<p markdown=1 class="takeaway">**要点：** 在预填充阶段，所有的矩阵乘基本总是算力受限的。因此，简单地最大化硬件利用率或 MFU（模型浮点利用率，Model FLOPs Utilization）就足以最大化每芯片吞吐量（成本）和延迟（以 TTFT 的形式）。除非提示词极短，否则在每提示词层级做批处理只会为预填充吞吐量带来微小的提升，却增加了延迟。</p>

然而，在生成阶段，对于每个请求，由于步与步之间存在顺序依赖，我们一次只能以一个词元为单位做 forward pass！因此我们只能（容易地）通过把多个请求批处理到一起、在批次维度上并行来取得好的利用率。我们稍后会更详细地讨论这一点，但实际上，在不影响延迟的前提下把许多并发请求批处理到一起是很难的。正因如此，**用生成来喂饱硬件的 FLOPs 要困难得多。**

<p markdown=1 class="takeaway">**要点：** 在生成阶段，总的词元批大小必须大于 $B_{\text{crit}}$ 才能在线性/前馈操作上算力受限（在 TPU v5e 上使用 bf16 参数时为 240）。由于生成是逐词元、串行发生的，这就要求我们把多个请求批处理到一起，而这很难！</p>

*值得注意这一数字有多么大！* 240 的生成批大小意味着同时有 240 个并发请求在生成，并且对于稠密模型有 240 个独立的 KV cache。这意味着它在实践中很难达到，除了某些批量推理场景。相比之下，在预填充阶段把超过 240 个词元推过去相当常规，尽管随着稀疏性增加也需要一些小心。

**注意这个确切的数字会因量化方式和硬件而异。** 加速器往往能在更低精度下提供更多的算术能力。例如，如果我们有 int8 参数但以 bf16 做计算，临界批大小会降到 120。对于 int8 激活值和 int8 参数，它会跳回 240，因为 TPUv5e 能提供 400 TOPs/s 的 int8 × int8 算力。

### 那注意力呢？ {#那注意力呢}

当我们考察点积注意力操作时，事情变得更复杂了，尤其是因为我们还要把 KV cache 考虑在内。来看一个纯多头注意力的单个注意力头。在一次 Flash Attention 融合中，我们<d-footnote>这里我们略去不少细节，忽略了应用 softmax、掩码等中的非 matmul FLOPs。它们本应与计算或 HBM 读取重叠，但在某些 TPU 代次上这并不容易做到。虽然这些细节不改变主要结论——即 KV cache 通常是内存受限的——但它们值得留意。</d-footnote>：

1. 从 HBM 读取形状为 $\text{bf16[B, T, D]}$ 的 $Q$ 激活值。
2. 从 HBM 读取 $KV$ cache，它是一对 $\text{bf16[B, S, D]}$ 张量。
3. 在 $$QK$$ matmul 中执行 $2BSTD$ 次 FLOPs。借助 Flash Attention，我们不需要把 $\text{bf16[B, S, T]}$ 注意力矩阵写回 HBM。
4. 在注意力 $$AV$$ matmul 中执行 $2BSTD$ 次 FLOPs。
5. 把得到的 $\text{bf16[B, T, D]}$ 张量写回 HBM。

把它们合起来，我们得到：

$$\text{Multiheaded Attention Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，$S=T$，因为我们在做自注意力，所以它简化为 $T^2 / 2T = T / 2$。这非常好，因为它意味着**预填充阶段注意力的算术强度是 $\Theta(T)$**。这意味着要算力受限是相当容易的。只要我们的序列长度相当大，就一切没问题！

但由于生成的序列维度微不足道，且 $B$ 和 $D$ 维度相互抵消，我们可以做如下近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为它意味着我们在生成阶段无法做任何事来提升注意力的算术强度。我们在做极小量的 FLOPs，同时却要加载一个巨大的 KV cache。**所以我们基本上在注意力上总是内存带宽受限的！**

<p markdown=1 class="takeaway">**要点：** 在预填充阶段，对于任意合理的序列长度（大约 $\gt 480$ 个词元），注意力通常是算力受限的；而在生成阶段，我们的算术强度低且恒定，因此我们总是内存带宽受限的。</p>

*从概念上讲，为什么会这样？* 主要原因在于，我们在模型的线性部分算力受限，是因为参数（这些内存带宽密集的组件）被许多批次项复用。然而，每个批次项都有自己的 KV cache，所以更大的批大小意味着更多的 KV cache。除非架构被激进地调整，否则我们在这里几乎*总是*内存受限的。

这也意味着，一旦参数内存与 KV cache 内存相当，增加批大小带来的吞吐量收益就会递减。这种递减收益对你的伤害程度，取决于单个序列的参数字节数与 KV cache 字节数之比，即大致为比值 $2DF / SHK$。由于 $HK\approx D$，这大致取决于 $F$ 与 $S$（序列长度）之比。这也取决于能让 KV cache 变小（我们稍后会详述）的架构修改。

### LLM 延迟与吞吐量的理论估计 {#llm-延迟与吞吐量的理论估计}

从这些计算中，我们可以对优化时应该瞄准的步时间得到相当好的上下界。**（注意：如果整章只让读者记住一件事，那就是下面这句。）** 对于生成时较小的批大小（这很常见），我们可以通过假设注意力和 MLP 块都内存带宽受限，来对我们的每步延迟给出下界：

$$\begin{equation*}
\text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}{\text{Total Memory Bandwidth}}
\end{equation*}$$

类似地，对于吞吐量：

$$\begin{equation*}
\text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total Memory Bandwidth}}{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}
\end{equation*}$$

最终，随着批大小增长，FLOPs 开始主导参数加载，所以在实践中我们有更一般的公式：

$$\begin{align}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align}$$

其中注意力部分（左侧）永远不会算力受限，因此不需要 FLOPs 屋顶线。这些对于粗略估算相当有用，例如：

<b markdown=1 style="color: #57cf57;">小测验：</b> 假设我们想在一个 4x4 的 TPU v5e 切片上，以 int8 权重、bf16 FLOPs、8192 上下文、每个词元 100 kB 的 KV cache，从一个 30B 参数的稠密模型取一个批大小为 4 词元的生成步。这个操作延迟的一个合理下界是多少？如果我们想采样一个 256 词元的批次呢？

{% details 点击此处查看答案。 %}

**答案：** 在 int8 下，我们的参数将占用 30e9 字节，而根据给定的规格，我们的 KV cache 每个将占用 `100e3 * 8192 = 819MB`。我们有 16 颗芯片，每颗有 `8.2e11` 字节/秒的带宽和 `1.97e14` bf16 FLOPs/s。由上面的公式，由于我们的批大小很小，我们预期步时间至少为 `(4 * 819e6 + 30e9) / (16 * 8.2e11) = 2.5 ms`。在 256 个词元时，我们的 MLP 块早已进入算力受限区间，所以我们的步时间大致为 `(256 * 819e6) / (16 * 8.2e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`。

{% enddetails %}

如你所见，这里吞吐量和延迟之间存在明显的权衡。小批次快，但不能很好地利用硬件。大批次慢，但高效。下面是针对一些较老的 PaLM 模型计算得到的延迟-吞吐量帕累托前沿（来自 [ESTI 论文](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>）：

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>Figure:</b> 若干 PaLM 模型在成本（即吞吐量）相对于延迟上的帕累托前沿。注意芯片数（C）和批大小（B）是如何让你沿着帕累托前沿移动的，唯一的例外是绿点（PaLM 540B 的 C:32 B:16），当时可用内存限制了该配置支持的批大小，导致吞吐量受损。注意吞吐量通常在批大小超过 240 后趋于平缓。int8 权重提供了更好的延迟-吞吐量帕累托最优点，但并没有更好的最大吞吐量。"%}

我们不仅用批大小作为旋钮来权衡延迟与吞吐量，也可能更倾向于一个更大的拓扑而非更小的，这样如果我们受限于 HBM，就能塞下更大的批次。 [下一节](../applied-inference) 会更详细地探讨这一点。

<p markdown=1 class="takeaway">**要点：** 如果你关心生成吞吐量，就使用尽可能大的每芯片批大小。任何超过 TPU 算术强度（$B_\text{crit}$，通常为 120 或 240）的每芯片批大小都会最大化吞吐量。你可能需要增大你的拓扑才能达到这一点。更小的批大小能让你以牺牲吞吐量为代价改善延迟。</p>

{% details 从硬件角度看这有一些需要注意的地方。点击此处查看一些细节。 %}

这都相当理论化。在实践中，由于几个原因，我们常常看不到一个陡峭的屋顶线：

* 我们假设 HBM 读取会与 FLOPs 完美重叠是不现实的，因为我们的编译器（XLA）并非完美无缺。
* 对于分片后的模型，XLA 还常常无法高效地让模型分片矩阵乘的 ICI 通信与 FLOPs 本身重叠，因此我们常常在 batch size 超过 $$\text{BS}=32$$ 的线性操作上开始承受延迟损失。
* 大于理论屋顶线的批大小仍会看到一些吞吐量的提升（因为重叠不完美），但这仍是一个很好的启发式规律。

{% enddetails %}

### 那内存呢？ {#那内存呢}

我们花了一些时间看带宽和 FLOPs，却没看内存。得益于我们的新数据结构 KV cache，推理时的内存图景看起来大不相同。在本节，我们挑一个真实模型（LLaMA 2-13B）来演示这其中有何不同：

| 超参数             | 值     |
| ------------------ | ------ |
| L（层数）          | 40     |
| D（d_model）       | 5,120  |
| F（ffw 维度）      | 13,824 |
| N（注意力头数）    | 40     |
| K（KV 头数）       | 40     |
| H（qkv 维度）      | 128    |
| V（词表大小）      | 32,000 |

推理时是什么在占用内存？显然，是我们的参数。数一下，我们有：

| 参数               | 公式                                                                                                             | 大小（字节）                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| FFW 参数           | d_model<sup>2</sup> x ffw_multiplier x 3（对应 SwiGLU 的 gate、up、down 投影）x n_layers                          | 5,120 x 5,120 x 2.7 x 3 x 40 = **8.5e9**                       |
| 词表参数           | 2（输入与输出嵌入）x n_embeddings x d_model                                                                       | 2 x 32,000 x 5,120 = **0.3e9**                                 |
| 注意力参数         | [2（*q 与输出*）x d_model x n_heads x d_qkv + 2（*k 与 v*）x d_model x n\_kv\_heads x d_qkv] x n_layers          | (2 x 5,120 x 40 x 128 + 2 x 5,120 x 40 x 128) x 40 = **4.2e9** |

把这些参数加起来，我们得到 8.5e9 + 4.2e9 + 0.3e9 = **13e9 总参数**，正如预期。正如我们在前面几节所见，训练时我们可能以 bfloat16 存储参数，并以 float32 存储优化器状态。这可能要用掉大约 100GB 内存。与之相比，我们的梯度检查点相形见绌——梯度检查点可以用掉好几个 TB。

**推理有何不同？** 在推理时，我们存储一份参数副本，假设为 bfloat16。这要用 26GB——而实际上我们常常能借助量化做得比这好得多。没有优化器状态或梯度需要跟踪。因为我们不做检查点（为反向传播保留激活值），我们的激活值占用对于预填充<d-footnote>尤其是多亏了 Flash Attention，它避免物化我们的注意力矩阵</d-footnote>和生成都可忽略不计。如果我们预填充 8k 词元，单个激活值只占用大约 `8,192 x 5,120 x 2 bytes = 80MB` 内存。更长的预填充可以被拆成许多更小的 forward pass，所以更长的上下文也不是问题。生成用的词元比这还少，所以激活值可忽略。

**主要的区别在于 KV cache**。这些是全部过去词元的 key 和 value 投影，其大小上限只由允许的最大序列长度决定。对于 $$T$$ 个词元，总大小为

$$\text{KV cache size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T$$

其中 $$H$$ 是每个头的维度，$$K$$ 是 KV 头的数量，$$L$$ 是层数，2 来自同时存储 key 和 value。

**这会非常快地变得很大**，即便在适中的批大小和上下文长度下也是如此。对于 LLaMA-13B，单个 8192 序列在 bf16 下的 KV cache 为

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}$$

**仅仅 4 个这样的 cache 就超过了我们参数的内存用量！** 说清楚一点，LLaMA 2 并未针对更长上下文下的 KV cache 大小做优化（它并不总是这么糟，因为通常 $K$ 要小得多，如 LLaMA-3），但这仍然很有说明性。在内存和延迟估计中，我们都不能忽略它们。

### 为 LLaMA 2-13B 建模吞吐量与延迟 {#为-llama-2-13b-建模吞吐量与延迟}

让我们看看，如果在 8xTPU v5e 上、在不同的批大小下以完美效率执行生成，一直到前面为最大理论吞吐量推导出的临界批大小（240），会发生什么。

| 批大小                            |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV Cache 内存（GiB）              |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| 总内存（GiB）                     |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| 理论步时间（ms）                  |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| 理论吞吐量（词元/秒）             | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5e 给了我们 128GiB 的 HBM、6.5TiB/s 的 HBM 带宽（每颗 0.82TiB/s）以及 1600TF/s 的算力。

对于这个模型，增大批大小确实带来了更好的吞吐量，但我们很快就面临急剧递减的回报。我们在批大小超过 16 时 OOM，并且需要大一个数量级的内存才能接近 240。更大的拓扑能改善延迟，但我们在每芯片吞吐量上撞到了一堵墙。

假设我们保持参数总数不变，但神奇地把 KV cache 缩小 5 倍（比如借助 1:5 的 [GMQA](#改善生成吞吐量与延迟的技巧)，即 8 个 KV 头在 40 个 Q 头上共享——详见下一节）。

| 批大小                            |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV Cache 内存（GiB）              |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| 总内存（GiB）                     |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| 理论步时间（ms）                  |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| 理论吞吐量（词元/秒）             | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

有了更小的 KV cache，我们仍然面临递减回报，但每芯片的理论吞吐量持续到批大小 240 都在增长。我们能塞进一个更大的批大小 64，并且在所有批大小下延迟都持续更优。延迟、最大吞吐量以及最大批大小全都大幅改善！事实上，后来的 LLaMA 代次就用了这个确切的优化——LLaMA-3 8B 有 32 个 query 头和 8 个 KV 头（[来源](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)）。

<p markdown=1 class="takeaway">**要点：** 除参数之外，KV cache 的大小对模型最终的推理性能有着很大影响。我们要通过架构决策与运行时优化的组合来把它控制在合理范围内。</p>

## 改善生成吞吐量与延迟的技巧 {#改善生成吞吐量与延迟的技巧}

自原始的 [Attention is All You Need 论文](https://arxiv.org/abs/1706.03762) 以来，人们开发了许多让模型更高效的技术，通常特别针对 KV cache。笼统地说，更小的 KV cache 让我们更容易在不损害延迟的前提下增大生成步的批大小和上下文长度，也让 Transformer 周围的系统（如请求缓存）更轻松。忽略对质量的影响，我们可能会看到：

**分组多查询注意力（又名 GMQA、GQA）：** 我们可以减少 KV 头的数量，并在注意力机制中让它们被许多 Q 头共享。在极端情况下，可以让单个 KV 头在全部 Q 头之间共享。这会把 KV cache 相较纯 MHA 缩小 Q:KV 比例的倍数，并且人们观察到模型性能对这一步改变相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也有效地提升了注意力计算的算术强度（见 [Section 4](../transformers) 的问题 4）。

**混入一些局部注意力层：** 局部注意力将上下文限制在较小到中等的最大长度内。在训练时和预填充时，这涉及把注意力矩阵掩码成一个对角条带，而不是一个三角形。这有效地把局部层的 KV cache 最大长度封顶。通过在模型中混入一些局部层与一些全局层，在超过局部窗口长度的上下文上，KV cache 的大小被大幅减小。

**跨层共享 KV：** 模型可以学会以某种模式在各层之间共享同一个 KV cache。虽然这确实减小了 KV cache 大小，并在增大批大小、缓存、离线存储等方面带来好处，但共享的 KV cache 可能需要从 HBM 多次读取，*所以它并不必然改善步时间。*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="<b>左：</b> 多层纯全局注意力。<b>右：</b> 一个与相邻层共享的全局/局部交错模式的示例。来源：<a href='https://research.character.ai/optimizing-inference/?ref=blog.character.ai'>Character.ai 博客</a>。"%}

**量化（Quantization）：** 推理通常对参数和 KV 的精度较不敏感。通过量化参数和 KV cache（例如到 int8、int4、`fp8` 等），我们可以在这两者上都节省内存带宽，减小达到算力屋顶线所需的批大小，并腾出内存以在更大批大小下运行。量化的额外好处是，即便模型并非以量化方式训练的，它也常常能在训练后应用。

**使用不规则 HBM 读取与分页注意力（Paged Attention）：** 在上面的计算中，我们为每个 KV cache 分配了 8k 的上下文，但通常没必要从内存读取整个 KV cache——请求的长度分布差异很大，并不会用满模型的最大上下文，因此我们通常可以实现一些 kernel（例如 Flash Attention 的变体），它们只读取 KV cache 中非填充的部分。

分页注意力<d-cite key="paged"></d-cite> 是对此的进一步完善，它以操作系统风格的页表存储 KV cache，并基本完全避免了 KV cache 的填充。这增加了很多复杂度，但意味着每个批次只使用它所需的那么多内存。这是一个运行时优化，因此同样与架构无关。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>Figure:</b> 在生成阶段，单个词元（“forth”）attend 到多个 KV cache 块/页。通过对 KV cache 分页，我们避免了加载或存储超出所需的内存。取自 <a href='https://arxiv.org/pdf/2309.06180'>PagedAttention 论文</a>。" %}

<p markdown=1 class="takeaway">**大局观：** 总而言之，这些 KV cache 优化能把 KV cache 大小相对于标准 MHA Transformer 减少一个数量级以上。这能带来 Transformer 总体成本一个数量级的改善。</p>

## 在多个加速器上分布推理 {#在多个加速器上分布推理}

到目前为止，我们一直粗略带过如何扩展到单颗芯片之外。遵循 [Section 5](../training)，让我们来探讨可用的不同策略及其权衡。和往常一样，我们会分别考察预填充和生成。

### 预填充 {#预填充}

从屋顶线角度看，**预填充几乎与训练完全相同**，几乎所有相同的技术和权衡都适用——模型（Megatron）并行、序列分片（对足够长的上下文）、流水线，甚至 FSDP（全分片数据并行）都是可行的！你只需让 KV 留着，以便稍后做生成。如同训练中一样，增加芯片数让我们得到更多 FLOPs/s（可能更低的 TTFT），但也增加了通信开销（可能降低每芯片吞吐量）。

**预填充分片的一般规则：** 这里有一套预填充的通用规则。我们假设只在单个序列上做预填充（无批次维度）：

1. *模型分片：* 我们通常先做一定量的模型并行，直到我们变得 ICI 受限。正如我们在 [Section 5](../training) 中看到的，对于 1 个轴这大约是 $F / 2200$（通常约为 4-8 路分片）。
2. *序列并行：* 在此之外，我们做序列并行（类似数据并行，但沿序列维度分片）。虽然序列并行在注意力中引入了一些额外通信，但在较长上下文下它通常相当小。如同训练中一样，我们可以让通信与计算重叠（分别借助集合矩阵乘用于 Megatron 和环形注意力）。

<p markdown=1 class="takeaway">**要点：** 在预填充阶段，几乎任何能在训练中工作的分片方式都能很好地工作。先做模型并行直到 ICI 边界，然后做序列并行。</p>

### 生成 {#生成}

生成是比预填充更复杂的存在。一方面，由于我们需要把许多请求批处理到一起，取得大批次更难。延迟目标也更低。综合起来，这意味着我们通常更内存受限，对通信开销更敏感，从而限制了我们的分片策略：

1. **FSDP 不可能：** 由于我们在从 HBM 向 MXU 加载参数和 KV cache 时内存受限，我们不想通过 ICI 来移动它们，因为 ICI 比 HBM 慢几个数量级。*我们想移动激活值而不是权重。* 这意味着类似 FSDP 的方法通常对生成完全不可行。<d-footnote>在训练后不小心把它开着，是一种常见的、能让性能退化一个数量级的错误</d-footnote>

2. **没有理由做数据并行：** 纯数据并行没有帮助，因为它复制了我们的参数，又不能帮我们更快地加载参数。你最好另起模型的多个副本来跑。<d-footnote>我们的意思是，以更小的批大小另起多个带有模型副本的服务器。模型层级的数据并行严格更差。</d-footnote>

3. **没有序列 = 没有序列分片。** 祝你好运做序列分片。

_这基本上给我们留下了针对稠密模型生成的、模型分片的若干变体_。如同预填充，我们能做的最简单的事就是简单的模型并行（激活值完全复制，权重沿 MLP 的隐藏维度完全分片），直到我们变得 ICI 受限，达到 4-8 路。然而，由于我们常常内存带宽受限，我们实际上可以超越这个限制来改善延迟！

**关于生成 ICI 边界的说明：** 在训练中我们想算力受限，所以我们的屋顶线看 ICI 通信时间何时超过 FLOPs 时间。然而在生成时，如果我们因加载参数而内存带宽受限，我们就可以把模型分片做到这个点之外，并以最小的吞吐量代价（以词元/秒/芯片计）改善延迟。更多的模型分片给了我们更多 HBM 来加载权重，而我们的 FLOPs 无关紧要。<d-footnote>在这个意义上，FLOPs 时间并不构成瓶颈，所以我们需要担心的是 ICI 时间超过参数加载时间。</d-footnote> 让我们看看在做多少模型并行之前它会成为瓶颈。

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。这个数字在 TPU v5e 和 TPU v6e 上通常约为 8。这意味着例如如果 $F$ 为 16,384、$B$ 为 32，理论上我们可以做最多 `16384 / (32 * 8) = 64` 路的模型并行而不对吞吐量造成明显影响。这假设我们能把 KV cache 完全 64 路分片，这很难：我们会在下面讨论。

对于注意力层，我们也以 Megatron 风格沿头维度对注意力 $$W_Q$$ 和 $$W_O$$ 做模型分片。KV 权重相当小，复制它们通常比超过 $K$ 路分片更便宜。

<p markdown=1 class="takeaway">**要点：** 在生成阶段，我们唯一的选择是模型并行的各种变体。我们的目标是在移动激活值，而不是更大的 KV cache 或参数。当我们的批大小较大时，我们做模型并行直到 FLOPs-ICI 边界（$F / \alpha$）。当我们的批大小较小时，我们可以通过做更多模型分片来改善延迟（以适度的吞吐量代价为代价）。当我们想做的模型分片路数多于我们拥有的 KV 头数时，我们也可以沿批次维度对 KV 做分片。</p>

### 对 KV cache 分片 {#对-kv-cache-分片}

**我们还有一个需要被分片的数据结构——KV cache。** 同样地，我们几乎总是倾向于避免复制这个 cache，因为它是注意力延迟的主要来源。为此，我们先沿头维度以 Megatron 方式对 KV 做分片。这被限制为 $K$ 路分片，所以对于那些头数很少的模型，我们尽可能沿头维度分片，然后沿批次维度分片，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这意味着 KV cache 被完全分布式存储。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>Figure:</b> 注意力机制对比：（a）纯模型分片的多头注意力，与（b）对 KV cache 做批次分片的多查询注意力。注意我们需要两个额外的 AllToAll 来把激活值从模型分片切换到批次分片，这样它们才能作用于 KV cache。"%}

这样做的代价是每层两个 AllToAll——一个把 Q 激活值切换到批次分片，这样我们就能以批次分片做注意力；另一个把批次分片后的注意力输出切回纯模型分片。

{% details 这是完整算法！ %}

这里我们将写出在 $Y$ 和 $Z$ 上都做模型并行的完整注意力算法。抱歉用 $K$ 同时表示 key 张量和 KV 头维度。令 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ...（已有的激活值，来自上一层、未分片）
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K<sub>Y</sub>, H] = ...（已有的 KV cache，批次分片）
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>, M])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub> })

这相当复杂，但你能大致看出它是怎么工作的。新的通信开销适中，因为它们作用在较小的激活值上；作为交换，我们节省了加载 KV（它们是静止的）的巨大内存带宽。

</div>

{% enddetails %}

* **序列分片：** 如果批大小太小，或上下文很长，我们可以对 KV cache 做序列分片。同样地，我们在此要为跨分片累积注意力付出集合通信的代价。首先我们需要 AllGather Q 激活值，然后以类似 Flash Attention 的方式累积 KV。

## 设计一个高效的推理引擎 {#设计一个高效的推理引擎}

到目前为止，我们看了如何孤立地、高效地优化和分片单独的预填充与生成操作。要真正有效地使用它们，我们需要设计一个推理引擎，它能在延迟/吞吐量帕累托前沿上我们选定的某一点，驱动这两个操作。

最简单的方法就是先跑一批预填充，再跑一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>Figure:</b> 在最简单的设置中，请求被聚合，服务器在运行一批预填充与调用生成函数之间交替，直到所有序列都完成。"%}

这容易实现，也是大多数代码库中的第一个推理设置，但它有多个缺点：

1. **延迟糟糕。** 我们把预填充和生成的批大小耦合在一起。首词元延迟（TTFT）在大预填充批大小下很糟糕——你需要完成所有预填充，用户才能看到词元。生成吞吐量在小批大小下很糟糕。
2. **较短的生成被较长的生成所阻塞。** 许多序列会在其他序列之前完成，在生成期间留下空的批槽位，进一步伤害生成吞吐量。随着批大小和生成长度增加，问题加剧。
3. **预填充存在填充浪费。** 预填充被补齐到最长序列，浪费大量计算。对此有解决方案，但历史上 XLA 让跳过这些 FLOPs 相当困难。同样，批大小和预填充序列长度越大，这越糟。
4. **我们被迫在预填充和生成之间共享分片。** 预填充和生成都运行在同一个切片上，这意味着我们对两者使用相同的拓扑和分片（除非你保留两份权重），这通常不利于性能，例如生成想要多得多的模型分片。

因此，这个方法只推荐用于边缘应用（通常只关心服务单个用户，并使用每字节 FLOPs 更少的硬件）以及 Transformer 代码库生命周期早期的快速迭代（因其简单性）。

一个略好的方法涉及在批大小为 1 时做预填充（此时它算力受限且延迟合理），但在生成阶段把多个请求批处理到一起：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这会避免来自批量预填充的浪费 TTFT，同时保持高生成吞吐量。我们称之为**交错（interleaved）**配置，因为我们把预填充和生成步"交错"开来。这对于像评测这样以吞吐量为主要目标的大批量生成应用非常强大。编排器可以配置为：一旦有任何生成槽位空出来，就优先做预填充，从而即使对于非常大的生成批大小也能确保高利用率。我们也可以避免把预填充补齐到最大长度，因为它并未与另一个请求批处理在一起。

主要的缺点是，当服务器在做预填充时，所有其他请求的生成都会暂停，因为所有计算资源都会被预填充消耗。用户 A 的响应正在解码，会被正在做预填充的用户 B 阻塞。这意味着即使 TTFT 改善了，词元生成平均而言也会是抖动且缓慢的，这对许多应用而言不是好的用户体验——其他用户的预填充处于一个请求总体延迟的关键路径上。

为了绕开这一点，我们把解码和预填充分离开来。虽然 Transformer 推理可以在一台服务器上完成，但从延迟角度看，把这两个不同任务放到两组 TPU/GPU 上执行通常更好。预填充服务器生成 KV cache，它们经网络发送给生成服务器，生成服务器把多个 cache 批处理到一起，并为它们各自生成词元。我们称之为 **"分离式（disaggregated）"**服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这提供了一些优点：

1. **大规模下的低延迟**：一个用户的请求永远不会被另一个用户的请求阻塞，除非预填充容量不足。请求应该被立即预填充，然后发送给生成服务器，再立即被排入生成缓冲区。如果我们预期会有许多并发请求涌入，我们可以独立于生成服务器数量来扩展预填充服务器数量，这样用户就不会在预填充队列中滞留过长时间。

2. **专精化（Specialization）：** 通常，预填充和生成在延迟最优的参数分片策略/硬件拓扑上相当不同（例如，更多模型并行对生成有用，但对预填充无用）。把两个操作限制为使用相同的分片会损害两者的性能，而保留两份权重又会占用内存。此外，把预填充移到它自己的服务器上，它除了当前正在处理的那个 KV cache 之外，不需要持有任何 KV cache。这意味着我们有更多空闲内存可用于历史缓存（见下一节）或优化预填充延迟。

一个缺点是，KV cache 现在需要经网络传输。这通常可以接受，但也再次提供了减小 KV cache 大小的动机。

<p markdown=1 class="takeaway">**要点：** 对于延迟敏感、高吞吐量的服务，我们通常必须把预填充和生成分离到独立的服务器上，其中预填充以批大小 1 运行，而生成把许多并发请求批处理到一起。</p>

### 连续批处理（Continuous batching） {#连续批处理-continuous-batching}

上面的问题（2）引出了**连续批处理**的概念。我们优化并编译：

* 一个预填充函数，它处理可变长度的上下文，并把结果插入到一个带有某个最大批大小和上下文长度/页数的 KV 缓冲区中。
* 一个生成函数，它接收 KV cache，并为所有当前活跃的请求执行生成步。

然后我们把这些函数与一个编排器组合在一起，编排器把到来的请求排队，根据可用的生成槽位调用预填充和生成，处理历史缓存（见下一节），并把词元流式输出。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存（Prefix caching） {#前缀缓存-prefix-caching}

由于预填充昂贵且算力受限（留给我们的余量更小），降低其成本的最佳方法之一就是少做一些预填充。因为 LLM 是自回归的，查询 ["I", "like", "dogs"] 和 ["I", "like", "cats"] 产生的 KV cache 在前两个词元上是相同的。这意味着，原则上，如果我们先算 "I like dogs" 的 cache，再算 "I like cats" 的 cache，我们只需做 1/3 的计算。我们可以通过复用 cache 省下大部分工作。这在几个特定场景下特别强大：

1. **聊天机器人**：大多数聊天机器人对话都涉及一来一回、严格向后追加的对话。这意味着如果我们能保存每一轮对话的 KV cache，我们就能跳过除最新词元之外的所有计算。
2. **少样本提示（Few-shot prompting）：** 如果我们有任何少样本提示，它可以免费被保存和复用。系统指令通常也具有这种形式。

这件事之所以难，唯一的原因是内存限制。正如我们所见，KV cache 很大（常常是好几个 GB），而且要让缓存有用，我们需要把它们保留到后续查询到来为止。通常，预填充服务器上任何未使用的 HBM 都可以用于一个本地缓存系统。此外，加速器在其 CPU 主机上通常有很大内存（例如一个 8xTPUv5e 服务器有 128GiB 的 HBM，但约有 450GiB 的主机 DRAM）。这块内存比 HBM 慢得多——通常慢到无法做生成步——但用于一次缓存读取则足够快。在实践中：

* 由于 KV cache 对于处理初始请求的那组 TPU 来说是本地的，我们需要某种形式的亲和性路由，以确保后续查询到达同一个副本。这可能在负载均衡上引发问题。
* 一个更小的 KV cache 会有帮助（再次）——它让我们能在同样的空间里保存更多 KV cache，并减少读取时间。
* KV cache 及其查找可以相当自然地存储为一棵树或 trie。淘汰可以基于 LRU 进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>Figure:</b> 以 LRU trie 实现的 KV 前缀缓存。我们可以通过共享前缀来避免重复 KV 内存。来源：<a href='https://research.character.ai/optimizing-inference/?ref=blog.character.ai'>Character.ai 博客</a>。"%}

### 来看一个实现：JetStream {#来看一个实现-jetstream}

Google 开源了一个实现这套逻辑的库，名为 [JetStream](https://github.com/google/JetStream)。该服务器有一组"预填充引擎"和"生成引擎"，通常在不同的 TPU 切片上，由单一控制器编排。预填充发生在 "[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)"，而生成发生在 "[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)"。我们还有一个 "[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)"，它负责编排 KV cache 从预填充切片到生成切片的复制。

引擎接口（实现于 [此处](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)）是任何 LLM 都必须提供的通用接口。关键方法有：

* **prefill（预填充）：** 接收一组输入词元并生成一个 KV cache。
* **insert（插入）：** 接收一个 KV cache，并将其插入到生成所基于的 KV cache 批次中。
* **generate（生成）：** 接收一组批处理的 KV cache，并为每个批次项生成一词元，为每个词元把一个词元的 KV cache 追加到解码状态（decode state）中。

我们也提供了一个 JetStream 的 PyTorch 版本，见 [此处](https://github.com/google/jetstream-pytorch)。

## 习题 {#习题}

我将基于 LLaMA-2 13B 为本节虚构一个新模型。细节如下：

| 超参数             | 值     |
| :----------------- | :----- |
| L（层数）          | 64     |
| D（d_model）       | 4,096  |
| F（ffw 维度）      | 16,384 |
| N（注意力头数）    | 32     |
| K（KV 头数）       | 8      |
| H（qkv 维度）      | 256    |
| V（词表大小）      | 32,128 |

**问题 1：** 上面的模型有多少个参数？在 int8 下，每个词元的 KV cache 有多大？*你可以假设我们共享输入和输出投影矩阵。*

{% details 点击此处查看答案。 %}

**参数量：**

* MLP 参数计数：$L * D * F * 3$
* 注意力参数计数：$L * 2 * D * H * (N + K)$
* 词表参数：$D * V$（因为我们共享这些矩阵）

因此我们的总参数量为 $L * D * (3F + 2H * (N + K)) + D * V$。代入上面的数字，我们有 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，这个模型约有 184 亿参数。

KV cache 每个词元在 int8 下为 $2 * L * K * H$，即 `2 * 64 * 8 * 256 = 262kB` 每词元。

{% enddetails %}

**问题 2：** 假设我们想在一个 TPUv5e 4x4 切片上服务这个模型，并且能在这个拓扑上完全分片我们的 KV cache。在假设一切使用 int8、并想支持 128k 序列长度的前提下，我们能塞下的最大批大小是多少？如果我们把 KV 头数降到 1 呢？

{% details 点击此处查看答案。 %}

我们的 KV cache 每个词元在 int8 下大小为 $2 \cdot L \cdot K \cdot H$，即 `2 * 64 * 8 * 256 = 262kB`。对于 128k 序列，这意味着 `262e3 * 128e3 = 33.5GB` 每批次项。由于每颗 TPU 有 16GB HBM（含我们的参数），我们能塞下的最大批大小为 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。如果我们有 $K=1$，我们会多出 8 倍，即约 56。

{% enddetails %}

**问题 3：** 假设参数在 TPU v5e 4x4 切片上完全分片，把所有参数从 HBM 加载进 MXU 需要多长时间？假设为 int8 参数。*这是每步延迟的一个很好的下界。*

{% details 点击此处查看答案。 %}

我们总共有 18.4B 参数，即 int8 下 18.4e9 字节。每颗芯片有 8.2e11 的 HBM 带宽，因此大致需要 `18e9 / (8.2e11 * 16) = 1.4ms`，假设我们能完全利用 HBM 带宽。

{% enddetails %}

**问题 4：** 假设我们想在一个 TPUv5e 4x4 切片上，使用 int8 FLOPs 以及参数/激活值来服务这个模型。对于预填充和解码，我们要如何分片它？*提示：也许先回答这些问题：*

1. 4x4 上的 ICI 长什么样？
2. 张量并行的屋顶线边界是什么？
3. 我们如何对 KV cache 分片？

对于这个分片方案，生成的粗略每步延迟是多少？

**问题 5：** 假设上面的模型其实是一个 MoE。一个 MoE 模型本质上是一个带有 E 份 FFW 块副本的稠密模型。每个词元通过 k 个 FFW 块，这 `k` 个被平均以产生输出。我们使用 `E=16`、`k=2` 以及上面的设置。

1. 它总共有多少参数、激活了多少参数？*激活（activated）指被任意给定词元使用到的。*
2. 在 TPU v5e 上需要多大的批大小才能算力受限？
3. 每个词元的 KV cache 有多大？
4. 一个含 T 个词元的 forward pass 涉及多少 FLOPs？

{% details 点击此处查看答案。 %}

(1) 作为一个 MoE，现在每个 MLP 块有 $3 * E * D * F$ 个参数，相比稠密变体增加了 $E$ 倍。因此它现在有 $L * D * (3EF + 2H * (N + K)) + D * V$，即 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` 总参数，增加了约 12 倍。对于激活参数，我们有 $k$ 而非 $E$ 个被激活的参数，总共 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，相比稠密变体增加不到 2 倍。

(2) 因为我们有 $E$ 倍的参数却只有 $k$ 倍的 FLOPs，我们的 HBM 屋顶线增加了 $E/k$ 倍。这意味着在 TPU v5e 上我们需要约 `240 * (16 / 2) = 1920` 个词元。

(3) KV cache 大小保持不变，因为 MoE 特性不改变注意力机制的任何东西。

(4) 这仍然是 $2 \cdot \text{activated params} \cdot T$。因此这是 $2 * \text{31.2e9} * T$。

{% enddetails %}

**问题 6：** 对于 MoE，我们可以做"专家分片（expert sharding）"，即把专家沿网格的一个轴切开。在我们的标准记法中，第一个 FFW 权重形状为 `[E, D, F]`，我们把它分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 仅在训练时作为我们的 FSDP 维度使用。假设我们想在 TPU v5e 上做推理：

1. 在上面 Y=8、Z=16 的 TPU v5e 8x16 切片上，该模型的 HBM 权重加载时间是多少？每颗 TPU 可用多少空闲 HBM？
2. 我们能把这个模型塞下的最小切片是多大？

**问题 7 [2D 模型分片]：** 这里我们将推导 [ESTI 论文](https://arxiv.org/pdf/2211.05102) 所称的 2D 权重静止（weight-stationary）分片的数学。我们在附录 B 中简要描述了它，但先试着做这道题，看看你能否推导出其中的数学。2D 权重静止分片的基本思想是沿 $D$ 和 $F$ 两个轴都对权重分片，使每个分块大致是正方形。这减少了通信负载，并让我们能稍微扩展得更远。

下面是 2D 权重静止的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>})
4.  Out[B, D<sub>X</sub>] {U<sub>YZ</sub>} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U<sub>YZ</sub>})
</div>

你的目标是推导出该算法的 $T_\text{math}$ 和 $T_\text{comms}$，并找出它何时会优于传统的 3D 模型分片？

{% details 点击此处查看答案！ %}

让我们推导 $T_\text{math}$ 和 $T_\text{comms}$。我们所有的 FLOPs 都是完全分片的，因此和之前一样我们有 $T_\text{math} = 4BDF / (N \cdot C)$，但我们的通信现在是

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

其中我们注意到 AllReduce 贵一倍，并且我们按每个操作所跨的轴数来缩放通信。假设我们可以自由选择拓扑，并假设 $F=4D$（如 LLaMA-2），我们断言（通过一些基础微积分）最优的 $X$、$Y$、$Z$ 取值为 $X = \sqrt{N / 8}$、$YZ = \sqrt{8N}$，于是总通信为

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，由上文可知，普通的 1D 模型并行会有 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$，那么新的通信何时更小？我们有

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 72
\end{align*}$$

对于一般的 $F$，我们断言这个条件为

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

所以这说明，如果我们有超过 72 颗芯片，用这个新方案就更划算。诚然，这是个有点奇怪的结果，因为我们历史上发现自己在约 20 路张量并行时就 ICI 受限了。但在这里，即便我们通信受限，我们的总通信量仍随芯片总数的增加而持续下降！这告诉我们，我们可以持续增加芯片、增大批大小、做更多参数扩展，并看到延迟下降。

{% enddetails %}

<h3 markdown=1 class="next-section">第 7 部分到此结束！关于第 8 部分——看看我们如何在 TPU 上服务 LLaMA 3，请点击[此处](../applied-inference)。</h3>

## 附录 {#附录}

### 附录 A：批大小 > 240 这条规则有多真实？ {#附录-a-批大小-240-这条规则有多真实}

我们上面给出的简单规则——批大小必须大于 240 个词元才能算力受限——大致成立，但它忽略了一些 TPU 在其他操作未用满所有可用 HBM（例如做设备间通信时）时预取权重的能力。

下面是一个小 Transformer（d<sub>model</sub> 8192、d<sub>ff</sub> 32768、每层仅 2 个 matmul）的层时间（微秒）经验图。它来自 [这个 Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。你会看到步时间非常缓慢地增长，直到约批大小 240，然后线性增长。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

下面是实际的吞吐量，单位为词元/微秒。这让论点相当清楚。由于我们的层在这里约 600M 参数、分片 4 路，我们预期最小延迟约为 365us。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

所以至少在这个模型中，我们确实看到吞吐量一直增长到每数据并行分片约 BS240。

### 附录 B：2D 权重静止分片 {#附录-b-2d-权重静止分片}

随着拓扑增长，如果我们可以使用更高维的网格（如 TPU 的网格），就可以通过引入第二个分片轴，以"**2D 权重分片**"进一步完善。我们称之为"**2D 权重静止**"，并在 [高效扩展 Transformer 推理论文](https://arxiv.org/abs/2211.05102) 中有更详细的描述。

因为在 Megatron 中我们只对隐藏的 $$F$$ 维度分片，一旦芯片数随 1D 分片增长得很大，它就会变得明显小于 $$E$$（$$d_\text{model}$$ 维度）。这意味着在更大的批大小下，在应用 MLP 第一层之后、沿隐藏维度做一部分集合通信会更经济。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

该图展示了：

1. 1D 权重静止分片，即纯 Megatron 分片，其中激活值在 AllGather 之后完全复制，权重沿隐藏 F 维度完全分片。
2. 2D 权重静止分片，其中权重沿隐藏 F 维度和归约 E 维度都做了分片，激活值沿 E 维度分片。我们在第一层之前沿 (yz) 轴做 AllGather，然后沿 (x) 轴做 ReduceScatter。

对于注意力层，在芯片数较少时，Megatron 风格分片也相对简单。然而，Megatron 是沿 $$n_\text{heads}$$ 维度进行的，这给可能的分片量设了上限。把 2D 分片改造用于注意力（不是对隐藏维度分片，而是对 $$n_\text{heads}$$ 维度分片），我们便获得了进一步扩展的能力。

### 附录 C：延迟受限的通信 {#附录-c-延迟受限的通信}

作为回顾，在 [Section 3](../sharding) 中我们推导了在一个 1D 环上、经 X 颗芯片、以全双工带宽 WICI 和延迟 Tmin，把大小为 B 的张量 AllGather 到每颗 TPU 上所花的时间。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于大的 B，挂钟时间保持相对恒定，因为随着你往系统中加入更多芯片，你同时放大了执行该操作所需的数据移动量和总可用带宽。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于在延迟优化的推理中移动的数据量相对较少，激活值上的集合通信常常受限于延迟项（尤其在小批大小下）。我们可以通过数完成前需要多少跳（hop）来相当容易地可视化这个延迟。

在 TPU 上，如果通信中与张量大小相关的部分小于每跳 1 微秒（一跳是相邻两台设备之间的通信），瓶颈就可能在于实际分派集合通信的固定开销。在 `4.5e10` 单向 ICI 带宽下，当 $(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$ 时，ICI 通信变成延迟受限。对于 8 路 Megatron 分片，这发生在 `buffer_size < 360kB` 时。**这在推理中其实不算那么小：** 在 int8 下，当 `BS=16`、`D=8192` 时，我们的激活值将占用 `16*8192=131kB`，所以我们已经延迟受限了。

<p markdown=1 class="takeaway">**要点：** 当 $$\text{total bytes} < W_{ICI} \times 1e-6$$ 时，我们的通信变成延迟受限。例如，沿 $$Y$$ 做模型并行时，在 int8 下当 $$Y > BD / 45,000$$ 时受限。</p>

这里可以与算力屋顶线做一个类比——我们都在承受某些小操作的固定成本（通信的延迟、matmul 的内存带宽）。

### 附录 D：推测采样（Speculative Sampling） {#附录-d-推测采样-speculative-sampling}

当我们*真的*关心端到端延迟时，还有一个额外的技巧可用，称为推测采样<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。作为回顾，我们通常从一个大 Transformer 中逐个生成词元：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

借助推测采样，我们用一个更小、更便宜的模型来生成词元，然后用大模型检验结果。这用*贪心解码（greedy decoding）*最容易理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从某个更小、更便宜的模型中贪心采样。理想情况下我们使用一个训练得能匹配大模型的小模型，例如通过蒸馏，但它也可以简单到只用 n-gram，或在小语料上做词元匹配。
2. 在我们生成了 K 个词元之后，我们用大模型计算我们迄今生成的所有词元的下一个词元 logits。
3. 由于我们做贪心解码，我们只需检查小模型生成的词元是否是所有可能词元中概率最高的。如果其中某个词元错了，我们取最长正确前缀，并把第一个错误词元替换为正确词元，然后回到 (1)。如果所有词元都正确，我们可以用最后一个正确的 logit 在回到 (1) 之前额外采样一个词元。

**为什么这是延迟上的胜利？** 这个方案仍然要求我们对每个词元做相当于一次大模型 forward pass 的 FLOPs，但因为我们可以把一堆词元批处理到一起，我们可以在一次 forward pass 中做完所有这些 FLOPs，并利用我们*并非*算力受限这一事实，免费为更多词元打分。

每个被接受的词元在平均意义上变得更贵（因为有些会被拒绝，而且我们得调用一个草稿模型），但我们从硬件中榨出了更多 FLOPs，而小模型很便宜，所以总体上仍占优。我们还跨多个步共享 KV cache 加载，因此**推测解码对于长上下文也可以是一个吞吐量上的胜利**。因为一切都被大模型检验过，我们完全不改变采样分布（不过对于非贪心的情况，确切的轨迹会不同）。

传统上，推测解码依赖于存在一个与目标模型采样分布相近的小模型，例如 LLaMA-2 2B 对应 LLaMA-2 70B，而这种模型常常不存在。即便有，如果接受率很低，较小的草稿模型仍可能太贵。相反，把一个草稿器嵌入主模型内部会很有帮助，例如通过给基模型的某后层加一个专用的草稿头<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。因为这个头与主模型共享大部分参数，它运行更快，并且更紧密地匹配采样分布。

对于普通的自回归采样，词元/秒与步时间相同。我们仍然受制于此处算术强度一节给出的理论最小步时间（事实上，推测采样的步时间通常比普通自回归采样慢不少，但因为平均每一步产出多于 1 个词元，我们能得到好得多的词元/秒）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>Figure:</b> 该图展示了 Chinchilla（DeepMind 的一个 70B 模型）配合一个 4B 参数的草稿模型（小模型）时的每步延迟与推测成功率。对于 XSum（一个自然语言数据集），理想的推测量约为提前 3-4 个词元，而 HumanEval（一个代码数据集）更具可预测性，能从更激进的推测中获益。"%}

**这对非贪心解码如何工作？** 这稍微复杂一些，但本质上归结为受 Metropolis-Hastings 启发的算法，其中有 $$P_{\text{draft model}}(\text{chosen token})$$ 和 $$P_{\text{target model}}(\text{chosen token})$$ 由 logits 导出，并且如果这两个概率之比小于某个阈值，就以一定概率拒绝所选词元。

这两篇 [论文](https://arxiv.org/abs/2211.17192) 和 [论文](https://arxiv.org/abs/2302.01318) 同时推导了这一点，并给出了它在实践中如何运作的好例子。

<p markdown=1 class="takeaway">**要点：** 推测采样是又一根强大的杠杆，用以将吞吐量换取为更好的每词元延迟。然而，在批大小受限的场景（例如较小的硬件占用或较大的 KV cache）中，它会变成双赢。</p>
