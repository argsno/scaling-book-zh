# 翻译术语表（Bilingual Translation Glossary）

本书（《How to Scale Your Model》中文版）的统一术语标准。所有章节译者**必须**以此文件为准，确保全书术语一致、无歧义、可机械化套用。

---

## 总体翻译准则（General translation guidance）

1. **保留原文，逐字直译**：数学公式 `$...$` 与 `$$...$$`、代码块（``` fenced code ```）、行内代码、URL、以及所有标准专有名词/缩写，**原样保留，不翻译、不转写**。
2. **图注与脚注要翻译**：`<d-footnote>...</d-footnote>` 内的说明文字、图片 `caption`、表格标题等需要翻译；但其中的代码、公式、专有名词仍按本表保留英文。
3. **每术语唯一译法**：同一英文术语在全书范围内只使用本表给出的中文译法，不切换同义词（例如不时而"分片"时而"切分"）。
4. **首现可加注英文**：翻译性术语在**首次**出现时，可在中文后括注英文，例如"全分片数据并行（FSDP）""重计算（rematerialization）"。后续出现不再重复括注。
5. **保留英文的术语**：凡列在本表"保留英文（KEEP ENGLISH）"标注下的术语，一律不翻译，全书统一大写/写法与原文一致（如 `AllReduce`、`BF16`、`ICI`）。
6. **大小写与连字符**：代码标识符（如 `shard_map`、`with_sharding_constraint`、`psum`）原样保留；自然语言中的复合词按本表译法处理。
7. **单位与数字**：`GB/s`、`TFLOPs`、`FLOPs/s` 等带单位写法原样保留；数字与单位间不留多余空格，保持原文风格。

---

## 1. 硬件（Hardware）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| TPU | TPU | 保留英文（Google 自研 AI 加速器） |
| GPU | GPU | 保留英文 |
| TPU v5e / v4 / v3 / v6e / v7 | TPU v5e / v4 / v3 … | 保留英文（代次型号） |
| Pod / SuperPod | Pod / SuperPod | 保留英文（TPU 集群单元） |
| Host | 主机 | 首现可注 host；指连接加速器的 CPU 机器 |
| HBM / HBM3 | HBM / HBM3 | 保留英文（高带宽内存） |
| SRAM | SRAM | 保留英文 |
| DRAM | DRAM | 保留英文 |
| PCIe | PCIe | 保留英文 |
| NVLink | NVLink | 保留英文 |
| NVSwitch | NVSwitch | 保留英文 |
| InfiniBand | InfiniBand | 保留英文 |
| ICI | ICI | 保留英文（TPU 片间互联） |
| DCN | DCN | 保留英文（TPU 数据中心网络） |
| NIC | 网卡 | 首现可注 NIC（network interface card） |
| MXU | 矩阵乘法单元 | 首现可注 MXU（matrix multiply unit） |
| VPU | 向量处理单元 | 首现可注 VPU（vector processing unit） |
| VMEM | 向量内存 | 首现可注 VMEM |
| VREG | 向量寄存器 | 首现可注 VREG |
| XLU | 转置单元 | 首现可注 XLU（transpose unit） |
| Tensor Core | Tensor Core | 保留英文（NVIDIA 张量核心） |
| CUDA Core | CUDA Core | 保留英文 |
| scalar core | 标量核心 | 标量运算单元 |
| SM (Streaming Multiprocessor) | 流处理器 | 首现可注 SM（GPU 流式多处理器） |
| Warp Scheduler | 线程束调度器 | 首现可注 Warp Scheduler |
| TMEM | 张量内存 | 首现可注 TMEM（Tensor Core 内存） |
| L2 Cache | 二级缓存 | 亦可读作"L2 缓存" |
| register(s) | 寄存器 | — |
| optical switch | 光交换机 | TPU 用可重构光交换 |
| Grace CPU / Grace Hopper | Grace CPU / Grace Hopper | 保留英文（NVIDIA Grace 方案） |
| NVLink C2C | NVLink C2C | 保留英文（CPU-GPU 互联） |
| node | 节点 | 指一台机器/一个计算节点 |
| SU (Scalable Unit) | 可扩展单元 | 首现可注 SU |
| leaf / spine switch | 叶子 / 脊交换机 | 胖树拓扑中的层级 |
| fat tree | 胖树 | 网络拓扑术语，保留"fat tree"括注亦可 |
| bisection bandwidth | 对分带宽 | — |
| egress bandwidth | 出口带宽 | — |
| systolic array | 脉动阵列 | — |
| megacore | 大核 | 首现可注 megacore |
| slice / tray | 切片 / 托盘 | TPU 硬件封装单位 |

---

## 2. 分布式通信与集合通信（Distributed communication / collectives）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| AllReduce | AllReduce | 保留英文 |
| AllGather | AllGather | 保留英文 |
| ReduceScatter | ReduceScatter | 保留英文 |
| AllToAll | AllToAll | 保留英文（含 dense / ragged / sparse 变体，保留英文修饰词） |
| psum | psum | 保留英文（partial sum，部分和） |
| pmean | pmean | 保留英文（部分和均值） |
| ppermute | ppermute | 保留英文（部分和置换） |
| pcast | pcast | 保留英文（部分广播） |
| ring reduction | 环形归约 | — |
| tree reduction | 树形归约 | — |
| broadcast | 广播 | 动词/名词均译"广播" |
| scatter | 散布 | — |
| gather | 收集 | — |
| reduce | 归约 | — |
| permute | 置换 | — |
| in-network reduction | 网内归约 | 亦作"网络内归约" |
| SHARP | SHARP | 保留英文（NVIDIA 网内归约技术） |
| collective | 集合通信（操作） | 指 AllReduce 等集体通信原语 |
| NCCL | NCCL | 保留英文（NVIDIA 集合通信库） |
| NVSHMEM | NVSHMEM | 保留英文 |
| collective matmul | 集合矩阵乘法 | — |
| partial sum | 部分和 | 亦作 psum |

---

## 3. 并行策略（Parallelism strategies）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| data parallelism | 数据并行 | — |
| tensor parallelism | 张量并行 | 缩写 TP 保留 |
| pipeline parallelism | 流水线并行 | 缩写 PP 保留 |
| expert parallelism | 专家并行 | 缩写 EP 保留 |
| model parallelism | 模型并行 | — |
| fully-sharded data parallelism / FSDP | 全分片数据并行 | 首现可注 FSDP |
| ZeRO | ZeRO | 保留英文（含 ZeRO-1/2/3） |
| Megatron | Megatron | 保留英文（Megatron-LM / Megatron 切分） |
| sequence parallelism | 序列并行 | — |
| context parallelism | 上下文并行 | — |
| strong scaling | 强扩展 | — |
| weak scaling | 弱扩展 | — |
| GSPMD | GSPMD | 保留英文 |
| SPMD | SPMD | 保留英文（单程序多数据） |
| sharding | 分片 | 名词 |
| shard | 分片 | 动词，如"将权重分片" |
| microbatch | 微批次 | 首现可注 microbatch |
| zero-bubble pipelining | 零气泡流水线 | — |
| 1F1B | 1F1B | 保留英文（一前一后调度，首现可注"1F1B"） |
| Auto / Explicit / Manual sharding | 自动 / 显式 / 手动 分片 | 指 JAX 三种分片模式 |

---

## 4. 数学与性能指标（Math / performance metrics）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| roofline | 屋顶线（分析） | 首现可注 roofline；"roofline 分析"亦可 |
| arithmetic intensity | 算术强度 | — |
| operational intensity | 运算强度 | 算术强度的同义表述，统一译"运算强度/算术强度" |
| FLOPs | FLOPs | 保留英文（浮点运算次数） |
| FLOPs/s | FLOPs/s | 保留英文（浮点运算速率） |
| TFLOPs | TFLOPs | 保留英文 |
| PFLOPs | PFLOPs | 保留英文 |
| FLOP | 浮点运算 | 单数，可作动词"做 FLOP"；通常与 FLOPs 区分 |
| MFU (Model FLOPs Utilization) | 模型浮点利用率 | 首现可注 MFU |
| compute-bound | 受算力限制 / 算力受限 | 二译皆可，全书统一用"算力受限" |
| memory-bound | 受内存限制 / 内存受限 | 全书统一用"内存受限" |
| communication-bound / comms-bound | 受通信限制 / 通信受限 | 全书统一用"通信受限" |
| bandwidth-bound | 受带宽限制 / 带宽受限 | — |
| latency | 延迟 | — |
| throughput | 吞吐量 | — |
| bottleneck | 瓶颈 | 动词"成为瓶颈" |
| utilization | 利用率 | — |
| parameter | 参数 | 缩写 params 译"参数量" |
| activation | 激活值 | 区别于"激活函数" |
| token | 词元 | 全书统一"词元"（不译"令牌"） |
| per-replica | 每副本 | — |
| critical intensity | 临界强度 | — |
| critical batch size | 临界批大小 | — |
| Pareto frontier | 帕累托前沿 | — |
| FLOP counting | 浮点运算计数 | 指估算 FLOPs 的方法 |

---

## 5. 内存与训练技术（Memory & training techniques）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| rematerialization | 重计算 | 亦称 remat；首现可注 rematerialization |
| rematerialize | 重计算 | 动词 |
| gradient checkpointing | 梯度检查点 | 亦作"梯度检查点技术" |
| host offload | 主机卸载 | — |
| gradient accumulation | 梯度累积 | — |
| optimizer state | 优化器状态 | — |
| quantization | 量化 | 含 int8 / int4 / fp8 / bf16 量化 |
| KV cache | KV cache | 保留英文（键值缓存） |
| Flash Attention | Flash Attention | 保留英文 |
| causal masking | 因果掩码 | — |
| prefill | 预填充 | 推理阶段 |
| generation / decode | 生成 / 解码 | 推理解码阶段；可作"生成（解码）" |
| disaggregated serving | 分离式服务 | 亦作"解耦式服务" |
| MoE | MoE | 保留英文（混合专家） |
| gating einsum | 门控 einsum | einsum 保留英文 |
| SwiGLU | SwiGLU | 保留英文（门控激活） |
| pre-norm / post-norm | 前置归一化 / 后置归一化 | — |
| MHA / MQA / GMQA | MHA / MQA / GMQA | 保留英文（多头 / 多查询 / 分组查询注意力） |
| sparsity | 稀疏性 | — |
| host RAM | 主机内存 | — |
| activation memory | 激活值内存 | 亦作"激活内存" |

---

## 6. JAX 与实现（JAX / implementation）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| JAX | JAX | 保留英文 |
| XLA | XLA | 保留英文（编译器） |
| Pallas | Pallas | 保留英文 |
| jit | jit | 保留英文（即时编译）；可括注"即时编译" |
| shard_map | shard_map | 保留英文（API 名） |
| with_sharding_constraint | with_sharding_constraint | 保留英文 |
| make_mesh | make_mesh | 保留英文 |
| set_mesh | set_mesh | 保留英文 |
| AxisType.Auto / Explicit | AxisType.Auto / Explicit | 保留英文 |
| NamedSharding | NamedSharding | 保留英文 |
| PartitionSpec | PartitionSpec | 保留英文 |
| device_put | device_put | 保留英文 |
| named_scope | named_scope | 保留英文 |
| einsum | einsum | 保留英文（爱因斯坦求和约定） |
| StableHLO / HLO / LLO | StableHLO / HLO / LLO | 保留英文（IR 层级） |
| IMEM | IMEM | 保留英文（指令内存） |
| fusion | 融合 | 指算子融合（kernel fusion） |
| tiling | 分块 | 亦作"平铺" |
| layout | 布局 | — |
| ragged_dot | ragged_dot | 保留英文 |
| Shardy | Shardy | 保留英文（分片 IR） |
| contract (einsum) | 收缩 | 张量缩并；指 einsum 的求和维度 |
| batch dimension | 批次维度 | — |

---

## 7. 通用动词与短语（General verbs & phrases）

| 英文 | 中文译法 | 用法说明 |
|---|---|---|
| gather | 收集 | — |
| scatter | 散布 | — |
| redistribute | 重新分布 | — |
| broadcast | 广播 | — |
| reduce | 归约 | — |
| all-reduce | 全归约 | — |
| pipeline | 流水线 | 动词"将…流水线化" |
| shard | 分片 | 动词 |
| concatenate | 拼接 | — |
| partition | 划分 / 分区 | 可作动词"划分" |
| contract | 收缩 | 见 JAX 节 |
| batch | 批处理 / 批次 | 动词"分批" |
| scale | 扩展 | 亦作"缩放"；结合 strong/weak scaling |
| offload | 卸载 | — |
| checkpoint | 检查点 | 动词"做检查点" |
| amortize | 分摊 | — |
| overlap | 重叠 | 如"计算与通信重叠" |
| synchronize / sync | 同步 | — |
| elide | 省略 | — |
| expose | 暴露 | "向…暴露" |
| in particular | 特别地 | — |
| as a function of | 作为…的函数 | — |
| on the order of | 量级为 / 大约为 | — |
| roughly | 大致 / 约 | — |
| naive / naive approach | 朴素（方法） | — |
| in turn | 进而 / 依次 | — |
| by contrast | 相比之下 | — |
| in practice | 在实践中 | — |
| at the cost of | 以…为代价 | — |
| trade-off | 权衡 | 名词 |
| threshold | 阈值 | — |
| overhead | 开销 | — |
| yield | 产生 / 得出 | — |
| roughly speaking | 粗略地说 | — |

---

## 附：保留英文术语速查（KEEP ENGLISH master list）

以下术语**全书一律保留英文**，不翻译：

TPU, GPU, JAX, XLA, Pallas, PyTorch, TensorFlow, CUDA, NVLink, InfiniBand, HBM, HBM3, SRAM, DRAM, FLOPs, FLOPs/s, TFLOPs, PFLOPs, BF16, FP32, FP16, INT8, ZeRO, AllReduce, AllGather, ReduceScatter, MoE, KV cache, matmul, GEMM, softmax, LayerNorm, RMSNorm, GSPMD, SPMD, AMP, TPU v5e/v4/v3/v6e, Pod, ICI, DCN, Tensor Core, CUDA Core, NVSwitch, NCCL, SHARP, NVSHMEM, Megatron, MHA/MQA/GMQA, Flash Attention, SwiGLU, StableHLO/HLO/LLO, IMEM, Shardy, ragged_dot, shard_map, einsum, NamedSharding, PartitionSpec。

（注：matmul / GEMM / softmax / LayerNorm / RMSNorm / AMP 等虽列于保留英文表，但在自然语言流畅处可酌情以"矩阵乘""softmax""层归一化"等表述，但**首次出现或作为标识符时必须保留英文**；为求全书一致，建议统一保留英文写法。）
