# vLLM进阶笔记：从快速部署到核心原理解析

* [返回上层目录](../model-deployment-practice.md)
* [前言：为什么选择vLLM](#前言：为什么选择vLLM)
* [第一次跑起来](#第一次跑起来)
* [工程化部署与日常运维](#工程化部署与日常运维)
* [KV-Cache与PagedAttention：核心原理](#KV-Cache与PagedAttention：核心原理)
* [Scheduler调度器：token级并发管理](#Scheduler调度器：token级并发管理)
* [一次请求的完整生命周期（串讲）](#一次请求的完整生命周期（串讲）)
* [多GPU/多机部署](#多GPU/多机部署)
* [常见问题排查](#常见问题排查)
* [附录](#附录)

很多人第一次接触 vLLM，只是照着教程把服务跑起来。但当遇到 OOM、并发上不去、参数不知道怎么调的时候，才发现自己其实不理解它在做什么。

这篇笔记的目标，是帮你建立一个**完整的 vLLM 心智模型**——不只是"能跑"，而是"知道它为什么这么跑"。从 KV Cache 的显存占用，到 PagedAttention 的分页机制，再到 Scheduler 的 token 级调度，每一个概念都会拆到你能自己推演的程度。

如果你正在用 vLLM 做推理服务，或者准备深入理解它的原理，这篇笔记应该能帮到你。

整体内容安排概览：

| 章节                       | 状态 | 主要内容                                                |
| :------------------------- | :--- | :------------------------------------------------------ |
| 前言                       | ✅    | 一句话定义、技术栈位置、两种用法、OpenAI API、vs Ollama |
| 第一次跑起来               | ✅    | 安装、模型选择、启动命令、参数详解、常见问题            |
| 工程化部署                 | ✅    | nohup、双模型、缓存迁移、检查清单                       |
| KV Cache 与 PagedAttention | ✅    | 核心原理、显存估算、Block 结构、误解澄清                |
| Scheduler 调度器           | ✅    | token 级调度、Prefill/Decode、多 worker 分析            |
| 完整生命周期               | ✅    | Server 启动、请求流程图                                 |
| 多 GPU 部署                | ✅    | TP vs PP 原理、单机多卡、多机部署、Ray 配置             |
| 常见问题排查               | ✅    | OOM、启动失败、性能问题                                 |
| 附录                       | ✅    | 命令速查、curl 示例、环境变量、术语表                   |

特点：

- 从"跑起来"到"理解原理"的完整路径

- 大量表格、代码块、流程图

- 结合 AutoDL 等真实场景

- 术语表帮助快速查阅

# 前言：为什么选择vLLM

## vLLM 的一句话定义与核心目标

> **vLLM 是一个面向"大规模并发推理"的大语言模型推理引擎，它解决的不是"模型能不能跑"，而是"模型在多人同时用时还能不能高效地跑"。**

vLLM 的核心目标：

| 目标 | 说明 |
|------|------|
| **并发执行** | 多个请求可以同时在 GPU 上计算 |
| **长上下文支持** | 通过 KV Cache 管理历史 token |
| **流式输出** | 边生成 token 边返回，不用等完整结果 |
| **高 GPU 利用率** | 批量 forward，减少空闲和重复计算 |

## vLLM 在 LLM 技术栈中的位置（认知地图）

先建立一个全局认知：

```
模型权重（Qwen / LLaMA / Mistral）
        ↓
Transformer 推理框架（PyTorch / CUDA Kernel）
        ↓
推理引擎（vLLM / TGI / LMDeploy）  ← vLLM 在这里
        ↓
服务层（OpenAI API / HTTP Server）
        ↓
应用（Chat / Agent / 后端服务）
```

如果从“单次请求”的视角再往里看一层，典型流程可以抽象成：

```text
用户 HTTP / OpenAI 请求
            │
            ▼
      HTTP / API 层
            │
            ▼
        Tokenizer
            │
            ▼
         Scheduler
            │
   ┌────────┴────────┐
   │                 │
   ▼                 ▼
 Prefill            Decode(多轮)
   │                 │
   └────────┬────────┘
            ▼
   Transformer 前向 + KV Cache
            │
   ┌────────┴────────┐
   ▼                 ▼
 流式返回给用户      Block 回收
```

后面的「一次请求的完整生命周期（串讲）」章节会在这个骨架之上，按时间顺序把每一格都展开。

**vLLM 的定位：**

- ❌ 不做模型
- ❌ 不做训练
- ✅ 专注在：推理、调度、显存利用、高并发

### 一个极其重要的认知

> **大模型推理的瓶颈，通常不是算力（FLOPs），而是内存和调度。**

直觉类比：

- 训练像：一次性大矩阵运算，GPU 吃满
- 推理像：不断有人插队问问题，每个人的对话长度还不一样

问题立刻出现：有人只问一句，有人已经聊了 3000 token，有人正在等下一个 token。

**GPU 很快，但显存很容易碎。**

这就是 vLLM 的"出场动机"。

## vLLM 的两种使用方式：Python 库 vs API 服务

### 方式一：作为 Python 推理引擎

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
outputs = llm.generate("你好")
```

适合：单机实验、算法研究、代码内嵌调用

**但注意**：这不是 vLLM 的"主战场"，它真正的价值没有完全发挥出来。

### 方式二：作为模型推理服务（核心用法）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct
```

这一条命令，本质上是在说：

> **"帮我起一个高并发的大模型 API 服务。"**

适合：多用户并发、生产环境、API 服务

## 为什么是 OpenAI API 格式？

这是一个非常工程、非常现实的设计选择。

### OpenAI API 已经是"事实标准"

- LangChain / LlamaIndex / Agent 框架
- 大量前端 / 后端代码
- 都已经写死了 OpenAI API 规范

如果 vLLM 另搞一套，几乎没人会用。

### vLLM 的策略

- **外部**：完全兼容 OpenAI API
- **内部**：自己玩 PagedAttention、Block 调度

这让你可以：

> **把 vLLM 当成"一个本地 OpenAI"**

只需要把 `base_url` 从 `https://api.openai.com` 改成 `http://localhost:8000`，其他代码一字不改。

## vLLM vs Ollama：不是替代，是不同定位

很多人会问："vLLM 比 Ollama 好吗？"

这个问题本身就是错的。它们从一开始就不是为同一类问题设计的。

| 维度 | Ollama | vLLM |
|------|--------|------|
| **面向对象** | 个人用户 | 服务/团队 |
| **典型场景** | 一台机器 + 一个用户 | 多 GPU + 多用户 |
| **并发能力** | 低（1-2 个请求） | 高（几十上百个请求） |
| **KV Cache 管理** | 连续分配 | 分页管理（PagedAttention） |
| **调度复杂度** | 简单 | 复杂（token 级调度） |
| **使用门槛** | 极低 | 偏工程 |
| **核心优势** | 开箱即用 | 高吞吐、高并发 |

### 本质差异

- **Ollama**：解决"我能不能用模型"
- **vLLM**：解决"很多人一起用，服务器会不会炸"

它们不是替代关系，而是：

> **个人工具 vs 服务基础设施**

## 谁应该用 vLLM？谁应该用 Ollama？

### 选择 Ollama 的场景

- ✅ 本地玩模型、写 Demo
- ✅ 个人研究、学习
- ✅ 不需要多人并发
- ✅ 追求"5 分钟跑起来"

### 选择 vLLM 的场景

- ✅ 需要对外提供 API 服务
- ✅ 多用户同时使用
- ✅ 需要高吞吐、低延迟
- ✅ 有 GPU 服务器资源
- ✅ 愿意花时间理解原理和调参

### 一个实用的判断标准

问自己一个问题：

> **"我的模型，会不会同时有多个人在用？"**

- 如果不会 → Ollama
- 如果会 → vLLM

# 第一次跑起来

vLLM 启动的本质，是在 GPU 显存中做两件事：

1. **加载模型权重**（固定占用，7B 模型约 14GB）
2. **预留 KV Cache 空间**（动态分配，用于存储对话历史）

启动参数的核心作用，就是控制这两部分显存的分配比例。理解这一点，后面所有参数的含义都会变得清晰。

## 环境准备：5 分钟快速上手

### 系统要求

| 要求 | 说明 |
|------|------|
| **操作系统** | Linux（Ubuntu 20.04+ 推荐） |
| **Python** | 3.8 – 3.11（推荐 3.10） |
| **CUDA** | 11.8 或 12.1+（需与 PyTorch 版本匹配） |
| **GPU** | NVIDIA GPU，计算能力 7.0+（V100、A100、RTX 30/40 系列） |

### 安装步骤

```bash
# 1. 创建独立的 Python 环境
conda create -n vllm python=3.10 -y
conda activate vllm

# 2. 安装 vLLM（会自动安装匹配的 PyTorch 和 CUDA 工具链）
pip install -U vllm
# 如果网络慢，使用国内镜像：
# pip install -U vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 验证安装
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"
# 应该输出：CUDA: True, GPU count: 1
```

**如果 CUDA 显示 `False`**：检查 GPU 驱动（`nvidia-smi`）和 PyTorch GPU 版本。

### 国内服务器配置（重要）

如果你在 AutoDL、阿里云、腾讯云等国内服务器上，**启动前必须先配置 HuggingFace 镜像**，否则模型下载会超时失败。

```bash
# 配置 HuggingFace 国内镜像（永久生效）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 如果使用数据盘存放模型（推荐，避免系统盘空间不足）
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface' >> ~/.bashrc

# 使配置生效
source ~/.bashrc
```

## 模型选择与显存估算

**核心原则：** 模型越大，效果越好，但显存占用也越大。选择模型时，要确保"模型权重 + KV Cache"不超过 GPU 显存。

| 显存档位 | 推荐模型 | 模型权重占用 | 说明 |
|---------|---------|------------|------|
| 8–12 GB | `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | 能跑但效果偏弱，适合测试 |
| 16–24 GB | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | **最通用，推荐**，效果和显存平衡 |
| 24–48 GB | `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | 效果更好，但需要大显存 |

**显存占用快速估算（7B 模型，FP16）：**
- 模型权重：~14 GB（固定）
- CUDA 开销：~1 GB（固定）
- KV Cache：`并发数 × 上下文长度(K) × 0.5` GB（动态）

**示例：** 24GB 显存，7B 模型，0.85 利用率
- 剩余显存：24 - 14 - 1 = 9 GB
- KV Cache 可用：9 × 0.85 ≈ 7.6 GB
- 并发能力：7.6 ÷ (2 × 0.5) ≈ 7 个 2K 上下文并发

> **公式来源**：7B 模型每 token 的 KV Cache 约 512 KB（详见"KV Cache 与 PagedAttention"章节）。2K 上下文 = 2048 tokens × 512 KB ≈ 1 GB。

## 最简启动命令

**一条"不追求极致性能，但几乎不可能踩雷"的启动命令：**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

**启动前检查清单：**
1. ✅ 已激活 conda 环境：`conda activate vllm`
2. ✅ 已配置 HuggingFace 镜像（国内服务器）：`echo $HF_ENDPOINT`
3. ✅ 端口未被占用：`lsof -i:8000` 应无输出
4. ✅ GPU 可用：`nvidia-smi` 应显示 GPU 信息

### 启动过程：4 个阶段

首次启动可能需要 **5-15 分钟**（取决于网络和 GPU），后续启动通常 1-2 分钟：

| 阶段 | 日志特征 | 耗时 | 如何判断正常 |
|------|---------|------|------------|
| 1. 下载模型 | `Downloading config.json...` 或 `Downloading model-*.safetensors...` | 首次较长（几 GB） | 看到下载进度条，磁盘写入活动（**不要 Ctrl+C**） |
| 2. 加载权重 | `Loading model weights...` | 1-3 分钟 | `nvidia-smi` 显存开始上涨（7B 约 14GB） |
| 3. 编译 Kernel | `Compiling CUDA kernels...` 或 CPU 占用高 | 首次 2-5 分钟 | CPU 占用高，无报错（**首次启动特有**，后续会快很多） |
| 4. 初始化 KV Cache | `GPU blocks: 956, CPU blocks: 512` | 几秒 | 看到 `GPU blocks` 数值，数值越大说明并发能力越强 |

**成功标志：**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Available routes: /v1/chat/completions
```

**启动过程中可以做什么：**
- 另开终端监控显存：`watch -n 1 nvidia-smi`
- 检查磁盘空间：`df -h` 确保有足够空间下载模型

## 验证启动成功

看到 `Uvicorn running on http://0.0.0.0:8000` 只说明 HTTP Server 启动成功，还需要确认三件事：

### 1. 模型是否在 GPU 上？

```bash
nvidia-smi
```

**预期结果**（7B 模型，FP16）：

- **Memory-Usage** 应显著上升（约 14-16 GB）
- **进程名**包含 `python` 或 `vllm`
- **GPU 利用率**在空闲时可能为 0%，有请求时会上升

### 2. KV Cache 是否初始化成功？

日志中应看到类似信息：

```
INFO: GPU blocks: 956, CPU blocks: 512
INFO: Maximum number of running requests: 256
```

**关键指标解读：**

| 指标 | 含义 | 正常值 | 如何判断 |
|------|------|--------|---------|
| `GPU blocks` | KV Block Pool 大小 | **至少几百个**（如 500+） | 几十个说明空间太小，需要调整参数 |
| `Maximum number of running requests` | 最大并发请求数（软上限） | 应该 ≥ 你预期的并发数 | 如果太小，提高 `--max-num-seqs` |

**快速估算并发能力：**
```python
# 假设看到 "GPU blocks: 956"
gpu_blocks = 956
block_size = 16  # tokens per block（vLLM 默认值）
max_context_len = 4096  # --max-model-len 的值

blocks_per_request = max_context_len // block_size  # = 256
max_concurrent = gpu_blocks // blocks_per_request  # = 3.7 ≈ 3
print(f"理论最大并发: {max_concurrent} 个 {max_context_len} token 的请求")
```

**如果 GPU blocks 太少（< 500）**：降低 `--max-model-len`（如 8192 → 4096）或提高 `--gpu-memory-utilization`（小心 OOM）

### 3. 验证没有"隐性 OOM"（启动成功但第一个请求 OOM）

**验证方法：**

```bash
# 发送一个简单的测试请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 10
  }'
```

**成功响应特征：**
- 返回 JSON 格式（包含 `choices` 和 `usage` 字段）
- 延迟合理：首次请求可能稍慢（3-10 秒），后续应在秒级（1-3 秒）
- GPU 在工作：另开终端运行 `nvidia-smi -l 1`，显存应有波动，GPU 利用率应有峰值（50-100%）

**如果返回 OOM 错误**：降低 `--gpu-memory-utilization`（如 0.85 → 0.7）或 `--max-model-len`（如 8192 → 4096）

## 核心启动参数详解

### `--model`：指定模型

```bash
--model Qwen/Qwen2.5-7B-Instruct
```

**支持两种方式：**

1. **HuggingFace 模型名**（推荐新手）：vLLM 会自动下载，首次启动需要下载，后续使用缓存
2. **本地路径**（推荐生产环境）：`--model /data/models/Qwen2.5-7B-Instruct`，路径必须包含 `config.json`

**这个参数决定：**
- 模型权重占多少显存：7B 模型约 14GB（FP16），14B 模型约 28GB
- 每个 token 的 KV Cache 大小：模型越大，每个 token 的 KV Cache 也越大

### `--gpu-memory-utilization`：为 KV Cache 预留显存

```bash
--gpu-memory-utilization 0.85
```

**含义**：GPU 总显存中，有多少比例可以被 vLLM 使用（用于模型权重 + KV Cache）。

**分配逻辑：**
```
1. 先加载模型权重（固定占用，如 7B 模型约 14GB）
2. 剩余显存 = 总显存 - 模型权重 - CUDA 开销（~1GB）
3. KV Block Pool 大小 = 剩余显存 × gpu-memory-utilization
```

**为什么不设 1.0？** CUDA context 和 kernel workspace 需要空间（约 0.5-1GB），显存占用会有波动，需要 buffer。

**经验值：**
- 新手/测试：0.8 – 0.85（保守配置）
- 单模型生产：0.85 – 0.9（追求更高并发）
- 双模型同机：先启先占，第二进程在「剩余显存」上乘自己的比例；两进程实际占用之和 ≤ 总显存并留余量（见「双模型同机部署」一节）

### `--max-model-len`：限制最大上下文长度

```bash
--max-model-len 4096
```

#### 精确定义：它到底在限制什么？

**严格定义**（默认无滑动窗口时）：

> 对于一条请求，在同一条上下文中允许存在的 token 总数上限：
>
> $$
> \text{上下文长度} = L_{\text{in}} + L_{\text{out}} \le \text{max-model-len}
> $$
> 其中：
>
> - ($L_{\text{in}}$)：这次请求的**输入 token 数**（完整对话 flatten 之后的总长度）
> - ($L_{\text{out}}$)：本次推理过程中，**已经生成的 token 数**（可以是多轮 streaming 累积）

**关键点：**
- 它**不是只限制输入**，也**不是只限制输出**；
- 而是**“输入 + 当前已生成”这条完整序列的总长度上限**。

#### 生成过程中会发生什么？

假设：

```bash
--max-model-len 4096
```

又假设这次请求的输入长度是：

```shell
L_in = 3000   # prompt flatten 后一共 3000 个 token
```

在生成阶段：

- 当生成到第 1000 个 token 时：
  - 当前总长度：\(3000 + 1000 = 4000 小于等于 4096\)，仍然合法。
- 当生成到第 1096 个 token 时：
  - 当前总长度：\(3000 + 1096 = 4096\)，刚好到上限。
- **如果还想生成下一个 token（第 1097 个），就会让总长度变成 4097，超过上限。**

在这一步之前，vLLM 会检查：

> “如果我再生成 1 个 token，这条序列的长度是否会超过 `max-model-len`？”

如果答案是“会”，典型行为是：

- **提前停止生成**，返回一个“因为到达长度上限而结束”的响应（类似 `finish_reason="length"` 的语义）；
- 而不是“已经生成了第 4097 个 token 之后才崩溃”。

特殊情况是：

- **如果 prompt 本身就超过 `max-model-len`（例如 `L_in = 5000`，`max-model-len = 4096`）**：
  - 这时还没开始生成，就已经不合法；
  - 通常会直接报错或拒绝这条请求。

> 可以这样记：  
> - **请求开始前**：检查 `L_in ≤ max-model-len`，否则直接报错/拒绝。  
> - **生成每个新 token 前**：检查 `L_in + L_out + 1 ≤ max-model-len`，不满足就停止生成。

#### “上下文长度”具体指什么？


“上下文长度”指的是：**这一条请求在模型内部被串联成的一整条 token 序列的长度**。

例如一次对话：

```text
系统：你是一个助手
用户：你好
助手：你好，我能帮你做什么？
用户：给我讲讲 vLLM 的 KV Cache？
```

在进入模型之前，会被 flatten 成一长串 token：

```text
[系统提示 token...] +
[第一轮用户 token...] +
[第一轮助手 token...] +
[当前用户 token...]
```

> 上面方括号里的“系统提示 / 第一轮用户 / 第一轮助手 / 当前用户”只是**说明性标签**，用来表示“这段 token 来自哪一类消息”。
>
> 实际送入模型的是你构造好的长字符串（例如带 `<|system|> / <|user|> / <|assistant|>` 等 special token 的 chat 模板，或者用“系统：/用户：/助手：”这类前缀的自然语言），
>
> “系统 / 用户 / 助手”这种角色信息，靠你怎么构造字符串来体现
>
> 常见有两类做法：
>
> - 有官方聊天格式的模型（有专用 special token）
>
> 比如一些 Chat 模型会要求你用固定模板：
>
> ```
> <|system|> ...系统内容... <|end|>
> <|user|> ...第一轮用户内容... <|end|>
> <|assistant|> ...第一轮助手内容... <|end|>
> <|user|> ...当前用户内容... <|end|>
> ```
>
> 这里的 <|system|>, <|user|>, <|assistant|> 会被 tokenizer 编成特殊 token，
>
> 模型是在训练时就学过这些标记的含义。
>
> - 没有专用格式时，你自己用自然语言标注
>
> 比如：
>
> ```
> 系统：你是一个乐于助人的助手。
> 用户：你好
> 助手：你好，我能帮你做什么？
> 用户：给我讲讲 vLLM 的 KV Cache？
> ```
>
> 这里的“系统：”“用户：”“助手：”也是普通文本，会被切成普通 token，
>
> 模型通过语义去理解，效果一般不如专用 chat 模型，但也能用。
>
> 然后再由 tokenizer 把整段字符串切成一串整数 token；模型本身只看到那串整数，并不知道“第几轮用户/助手”，这些语义都是通过你在字符串里放的标记体现出来的。

这条“长串”的 token 个数，就是本次请求的 ($L_{\text{in}}$)。  

在此基础上，模型每生成 1 个新 token，就在尾部再追加 1 个，逐步形成：

```text
上下文 = 输入 token（L_in） + 已生成 token（L_out）
```

> **上下文长度 = 这条“长串 + 已生成 token”的总长度**，
>
> 不关心“有几轮对话、几条消息”，只关心 flatten 后有多少个 token。

#### 为什么 KV Cache 不能随便超过这个长度？

从 vLLM 的实现角度看，`max-model-len` 同时扮演了两重角色：

- **逻辑上**：限制单条请求的“最长对话长度”（输入 + 输出），防止“无限拉长”；
- **物理上**：告诉 vLLM：“我要按这个最长长度，为每条请求预留最坏情况的 KV Cache 空间。”

如果运行时某条请求偷偷长到了远超 `max-model-len`，会带来几个严重问题：

- **显存规划失效**：
  - 启动时，vLLM 按 `max-model-len` 和 `gpu-memory-utilization` 规划了 KV Block 数量；
  - 某条序列突然翻倍变长，相当于这条序列要额外抢一倍的 KV Block；
  - 超出预留空间 → 直接 OOM / 分配失败。
- **调度估算失真**：
  - Scheduler 认为“每条最长不过 `max-model-len`，可以并发 N 条”；
  - 结果来了一个“超规格对话”，把 KV Cache 吃爆，影响其他请求。
- **模型本身的极限**：
  - 模型训练时就有自己的最大 context（如 4K / 8K / 32K）；
  - 硬拉得比训练时还长，位置编码超范围，质量严重不可预测。

> 因此，vLLM 选择用 `max-model-len` 把“单条序列的最大长度”**硬性钉死**：  
> - 超出这个长度：要么一开始就拒绝，要么在生成过程中提前停止（或配合滑动窗口丢弃最早的一段历史）。  
> - 不允许某些请求“偷偷超标”，否则显存规划和调度都会变得不可控。

#### 间接限定了最大并发数

`-max-model-len` 和 `KV Block Pool` 一起，间接地“限定了最大并发数”，但真正“限制并发”的是两者的组合，不是 `-max-model-len` 单独一项。

（1）它们是怎么一起决定最大并发的？

大致关系：

- 总 Block 数：Pool_blocks（启动日志里的 GPU blocks: xxx）

- 单请求最坏 Block 数：
  $$
  \text{blocks\_per\_request\_max}\approx \frac{\text{max-model-len}}{\text{block\_size}}
  $$

- 理论最大“拉满长度”的并发数：
  $$
  \text{max\_concurrency}\approx \frac{\text{Pool\_blocks}}{\text{blocks\_per\_request\_max}}
  $$

所以：

- `-max-model-len` 设 越大：

  - 单请求最坏要吃的 Block 越多

  - 能撑的“全长请求”的并发数就越少

- `-max-model-len` 设 越小：

  - 单请求吃的 Block 少

  - 理论最大并发上限就越高（但单请求能用的上下文也被你限制住了）

（2）超过“能安全承载的并发”会怎样？

当“当前在跑的请求 + 想新接入的请求”预计会让总 Block 使用量超出 Pool 容量时，Scheduler 会：

- 要么排队：
  - 不立即接入新请求，等已有请求结束、释放出 Block，再把排队请求放进来。

- 要么直接拒绝：
  - 如果已经明显超出可承受范围（比如 Pool 本来就很小、请求都很长），可能直接返回资源不足/OOM 类错误。

不会出现的是：

- 静默地把所有请求都接进来，然后在中途因为 KV 不够用“悄悄丢 KV 继续算”。

（3）可以怎么在脑子里记住？

你可以这样记：

- `-max-model-len`：约定“单条请求最多能吃多长上下文”，也就决定了“单条请求最坏会吃多少 Block”。

- `KV Block Pool`：给所有并发请求共享的一大块 KV 容量上限。

- 最大并发数：由这两者的比值决定。

- 当“预估 Block 用量”超过 Pool 时，新请求会被延迟接入或直接拒绝，而不是硬往里塞。

#### 选择建议（结合显存）

在理解了上面的语义之后，可以按显存给个经验值：

- 8–12 GB 显存：`--max-model-len 2048`
- 16–24 GB 显存：`--max-model-len 4096`（**最通用，推荐**）
- 24–48 GB 显存：`--max-model-len 8192`（前提是并发不要设太高）

### `--dtype`：数据精度与显存占用

```bash
--dtype float16
```

**dtype 决定两件事：** 模型权重的精度 + **KV Cache 的精度**（永远跟着 dtype 走）

| 选项 | 说明 | 显存占用 | 适用场景 |
|------|------|---------|---------|
| `auto` | 自动检测模型原始精度 | 取决于模型 | 不确定时的默认选择 |
| `float16` | 半精度浮点 | **最通用** | 几乎所有 GPU 都支持，推荐 |
| `bfloat16` | Brain Float 16 | 与 float16 相同 | A100/H100 等新卡，数值稳定性更好 |
| `float32` | 全精度 | **显存翻倍** | 一般不用，除非特殊需求 |

**选择建议：** 不确定用 `auto`，RTX 30/40 系列用 `float16`，A100/H100 用 `bfloat16`

### `--max-num-seqs`：最大并发请求数

```bash
--max-num-seqs 256
```

#### 精确定义：它到底在数什么？

在 vLLM 的调度器眼里，**一条“活着的 token 序列”就是一个 seq**：

- 对一次普通请求来说：
  - prefill 阶段：这条请求的 token 序列被送入模型，成为 1 个新的 seq；
  - decode 阶段：只要这条请求还在生成/还占着 KV Cache，它就始终算“1 个活跃 seq”；
- 当请求完成、KV Cache 释放，这个 seq 才会从调度器中移除。

在最常见的场景下，你可以安全地近似为：

> **一个未完成的请求 ≈ 一个活跃 seq。**

于是：

> **`--max-num-seqs` = 调度器允许“同时存在的活跃 seq 数量上限”。**  
> 也可以理解为：**“这个实例在任意时刻，最多能有多少条对话上下文挂在 GPU 上被服务”。**

#### 为什么叫“软上限”？它和显存的关系

真正能跑多少并发，还受 **KV Block Pool（显存里能放下多少 KV）** 限制：

- 每条活跃 seq 会消耗一定数量的 KV blocks（和 `max-model-len`、生成长度等有关）；
- KV Block Pool 总容量是固定的，只能容纳有限条 seq。

因此，**实际能撑住的最大并发**由两部分共同决定：

- 你配置的软上限：`--max-num-seqs`
- 显存层面的硬上限：KV Block Pool 能装下的最大 seq 数

可以用一句话概括：

> **实际并发能力 ≈ min（KV Block Pool 能容纳的活跃 seq 数，`--max-num-seqs`）。**  
> - 如果 `--max-num-seqs` 设得很大：最后是显存说了算。  
> - 如果 `--max-num-seqs` 设得很小：你人为给实例加了一个“闸刀”，即使显存还有富余也不会再接新并发。

#### 超过这个上限会怎样？

- 当**当前活跃 seq 数 < `--max-num-seqs`** 时：
  - 新请求可以被接入调度器，成为新的活跃 seq。
- 当**当前活跃 seq 数已经达到 `--max-num-seqs`** 时：
  - 新请求不会再被立刻送进模型，而是根据上层服务逻辑：
    - 要么进入排队/等待状态，等有旧的 seq 完成释放名额；
    - 要么被限流/拒绝（由 API 层决定）。

你可以把它想象成：

> GPU 前面有一个服务窗口：  
> **`--max-num-seqs` 控制“窗口前同时能站多少个顾客”，其余人只能在外面排队或被请下次再来。**

#### 和`--max-model-len`一起决定实际的最大并发数

但是既然`--max-model-len`限定了最大并发数，那不是还有个`--max-num-seqs`参数吗，这个不也是限定最大并发数的吗？那这不就冲突了吗

不冲突，它们是两个不同维度的约束，Scheduler 会同时考虑。

（1）`--max-num-seqs`：显式的“请求数量上限”

- 含义：Scheduler 允许同时处理的请求数上限（硬限制）。

- 作用：当“当前在跑的请求数”达到 --max-num-seqs 时，新请求会被直接拒绝或排队，不管 Block 是否还有余量。

（2）`--max-model-len`：间接影响并发，但不是“请求数上限”

- 含义：单条请求的最大上下文长度（prompt + 生成）。

- 作用：

  - 影响“单请求最坏会吃多少 Block”（max-model-len / block_size）

  - 进而影响“在给定 Pool 大小下，理论能撑多少并发”

  - 但它本身不是“最多接 N 个请求”的硬限制

（3）它们如何一起工作？

Scheduler 在决定是否接入新请求时，会同时检查两个约束：

- 约束 1：当前请求数 < --max-num-seqs？
  - 否 → 直接拒绝/排队（不管 Block 够不够）

- 约束 2：预估 Block 用量 ≤ Pool 容量？
  - 否 → 拒绝/排队（即使请求数还没到上限）

实际的最大并发数 = min(理论并发上限, --max-num-seqs)

（4）为什么需要两个参数？

- `--max-num-seqs`：控制调度负载和延迟上限，避免请求数过多导致延迟爆炸。

- `--max-model-len`：控制单请求长度上限，影响 Block 使用和资源预估。

它们互补：

- `--max-num-seqs` 控制“数量上限”（调度/延迟维度）

- `--max-model-len` 控制“长度上限”（资源/Block 维度）

总结

- `--max-num-seqs`：硬限制“最多同时处理多少个请求”

- `--max-model-len`：限制“单请求最多多长”，间接影响并发上限

- 最终并发上限 = min(由 Block 算出的理论并发, `--max-num-seqs`)

两者不冲突，而是共同约束。

**进一步：实际生效的并发上限是怎么算的？**

即便 `--max-num-seqs` 设得很大，如果按显存算出来的“能撑的并发数”更小，实际也会按显存这一侧来限制。更准确的关系是：

- **每请求最坏 Block 数**：`⌈ max-model-len / block_size ⌉`
- **实际最大并发** = min( `--max-num-seqs`，Pool 能撑的并发数 )

也就是说：**实际并发由「Pool 能撑的并发数」和 `--max-num-seqs` 里更小的那个决定。** 公式里“Pool 能撑的并发数”要除的是「每请求最坏 Block 数」，不是直接除 `max-model-len`。

**区分“总容量”和“实时还能接多少”**

- **理论最大并发（总容量）**：空 Pool 时，最多能同时跑多少条“拉满 max-model-len”的请求。  
  \[
  \approx \frac{\text{Pool 总 Block 数}}{\text{每请求最坏 Block 数}}
  \]

- **实时还能接多少（调度用）**：当前已经有一部分 Block 被在跑请求占用，Scheduler 决定“接不接这个新请求”时，看的是**剩余** Block 数。  
  \[
  \approx \frac{\text{Pool 剩余 Block 数}}{\text{每请求最坏 Block 数}}
  \]

调度器在决定是否接入新请求时，用的是“实时还能接多少”：  
当前已占用 Block 数 = 所有在跑请求的 Block 之和，剩余 Block 数 = Pool 总 Block 数 − 已占用 Block 数，再用剩余 Block 数除以每请求最坏 Block 数得到当前还能接的并发数。

#### 什么时候需要手动调这个参数？

大部分情况下：**不需要手动设置**，交给 vLLM 按 KV Block Pool 自动推算即可。

只有在下面这些情况，它非常有用：

- 你想**给单个实例设置一个明确的“最大并发能力”**，方便做容量规划；
- 你的请求都很长（大上下文+长生成），担心高并发时 OOM，希望**主动压低并发**换稳定性；
- 你有一层业务限流逻辑，希望“vLLM 这一层永远不要吃超过 N 条活跃对话”。

> 一个安全的心智模型是：  
> **“一个活跃请求 ≈ 一个 seq，`--max-num-seqs` 就是这个实例能同时挂在 GPU 上的活跃请求的软上限，最终实际并发还要再受显存可用的硬上限约束。”**

#### 它和 Scheduler / batch 的关系

容易误解的一点是：**`--max-num-seqs` 控制的是“活跃池子的容量”，而不是“每个 batch 的大小”。**

可以先分层理解整体结构：

- **活跃 seq 池子（由 `--max-num-seqs` + KV Cache 决定）**：
  - 里面装的是“所有还没结束、还占着 KV Cache 的请求”；
  - 这些请求都可能在接下来的某一轮被调度送进模型。
- **每一轮调度出的 batch**：
  - Scheduler 在这一刻从“活跃池子”里，挑出一批“当前需要计算的 seq”（prefill 或 decode）；
  - 把它们打包成一个 batch，送进模型做一次 forward；
  - 这一批的实际大小 ≤ 活跃池子大小，但**通常远小于 `--max-num-seqs`**。

> 直观类比：  
> - `--max-num-seqs` 决定的是“候车大厅最多能站多少人”；  
> - Scheduler 每一轮发车，只会从大厅里挑一批人上车（组成一个 batch），而不是所有人一次性全上。

更具体一点，Scheduler 每一轮循环大致是这样：

1. **收集“当前需要计算”的 seq：**
   - prefill 阶段的 seq（第一次送进模型）；
   - decode 阶段的 seq（需要下一个 token）；
   - 某些 seq 可能还在等待上一个 step 的结果或被暂时冻结，这轮就不参赛。
2. **根据策略从中挑出一批 seq，形成一个 batch：**
   - 考虑哪些 seq 等了比较久（公平性）；
   - 尽量让 batch 内各 seq 的长度/shape 接近（提高算子利用率）；
   - 确认这批 seq 的 KV Block 占用不会超过显存；
   - 这批 seq 的数量**一定 ≤ `--max-num-seqs`，但往往远小于它**。
3. **把这一批 seq 的“当前这一步所需的 token”拼成 batch，送进模型做一次 forward：**
   - 对 prefill：batch 里每个样本是整段 prompt；
   - 对 decode：batch 里每个样本通常是“1 个新 token + 对应的 KV 引用”。
4. **算完这一轮，再回到调度循环：**
   - 已经生成结束的 seq 从活跃池子中移除，释放 KV 与并发名额；
   - 新请求加入池子，成为新的活跃 seq；
   - 然后重复 1–3 步。

在高并发场景下：

- 活跃池子里往往有很多 seq，Scheduler 几乎每一轮都能挑出一个较大的 batch，提升吞吐；

在低并发 / 冷启动场景下：

- 活跃 seq 很少时，Scheduler 不会傻等“攒满到 `--max-num-seqs` 再发”，而是用**较小的 batch 优先保证延迟**；
- 这样即使只有 1–2 条 seq，也能及时被送进模型处理。

总结这层关系可以记一句话：

> **`--max-num-seqs` 决定“最多同时有多少条对话挂在系统里”；  
> Scheduler 在这些对话里按需分批组成 batch，送入模型执行，每一批的具体大小由当前活跃 seq 数量、显存与调度策略共同决定，而不是简单等量攒满。**

#### batch 大小到底是谁决定的？

很多人会下意识以为“batch size = 某个固定参数”，在 vLLM 里不是这样的。  
**batch 大小是 Scheduler 在每一步 forward 时，结合当前状态和约束“动态算出来”的结果。**

可以把影响因素分三类：

- **并发上限约束（池子容量）**：
  - `--max-num-seqs = C_soft`：活跃 seq 池子的软上限；
  - `C_kv`：在当前显存 / Block 使用情况下，从现在起最多还能容纳多少条活跃 seq（硬上限）；
  - 理论上，从空仓开始把系统压满，**最大活跃 seq 数 ≈ min(C_soft, C_kv)**。
- **单步计算约束（这一步最多算多少）**：
  - 一些内部/版本参数会限制“单次 forward 里总共能算多少 token / 多少条 seq”（比如 max batched tokens、max batch size），以防单步过大导致 latency 失控。
- **调度策略（在约束内如何取舍）**：
  - 在不违反上述约束的前提下，Scheduler 会综合考虑：
    - 等待时间（优先照顾等得久的 seq，保证公平性）；
    - 序列长度和 shape（尽量把长度接近的样本放一批，减少 padding 浪费）；
    - 当前 ready 的 seq 数量（有多少 Prefill/Decode 可以立刻算）。

综合起来，一步中的 batch 大小可以理解为：

> 在“当前 ready 的 seq”集合里，  
> Scheduler 找到**一批 seq**，  
> 使得：
> - 数量 / token 总数不超过并发和显存限制；
> - 同时在公平性、延迟和吞吐之间做出折中。

这意味着：

- **batch 大小不是常数**：不同时间步、不同负载下的 batch 大小会变化；
- 你能通过 `--max-num-seqs`、显存、相关上限参数“间接规定它的上界和倾向”，  
  但**具体每一步选多少条，是 Scheduler 的动态决策，而不是某个恒定的配置值。**

#### 为什么一个 batch 里可以同时有 Prefill 和 Decode？

直觉上，很多人会以为：**“能放进同一个 batch 的样本，必须走完全相同的计算逻辑”**，  
于是看到“一个 batch 里同时有 Prefill 和 Decode 的 seq”就会本能地觉得不对劲。

要打破这个直觉，可以从三个层次理解：

##### 数学层：Prefill / Decode 本质上在做同一件事

对任意一条序列来说，Attention 的本质都是：

> $$
> \text{Attention}(Q_{\text{new}}, K_{\text{all}}, V_{\text{all}})
> $$

- **Prefill 时：**
  - ($Q_{\text{new}}$)：这次 prompt 的**所有输入 token** 的 Q（长度可能是 100、2000…）；
  - ($K_{\text{all}}, V_{\text{all}}$)：同一批 token 自己的 K/V（之前还没有 KV Cache）；
  - 算完后，把这批 token 的 K/V **全部写入 KV Cache**。
- **Decode 时：**
  - ($Q_{\text{new}}$)：“这一步要生成的 **1 个新 token** 的 Q”；
  - ($K_{\text{all}}, V_{\text{all}}$)：**历史所有 token 的 KV**（已经在 Cache 里）；
  - 算完后，把这 1 个新 token 的 K/V **追加到 KV Cache 尾部**。

从公式角度看：

- Attention 算的都是同一套操作：矩阵乘、softmax、加权求和；
- 只是：
  - **新来的 Q 的“长度”不同**（Prefill 一次来很多，Decode 一次来 1 个）；
  - **已经存在的 KV 的“长度”不同**（Prompt 越长 / 生成越多，历史 KV 越长）。

> 换句话说：  
> **从“网络结构/算子”角度，Prefill 和 Decode 跑的是同一套 Transformer 前向，区别只是“本轮有多少新 token、历史 KV 有多长”。**

##### Kernel / 实现层：如何在一个 batch 里“混在一起”？

vLLM 的连续批处理（continuous batching）在内部是这样组织数据的（概念上）：

- batch 维度：有很多条 seq；
- 对 batch 里的每条样本，都带上一些 **元信息（meta）**：
  - 这条样本当前是不是已经 Prefill 完；
  - 这一轮它有多少个“新 token”要算（Prefill 可能是几十/几百个，Decode 通常是 1 个）；
  - 它在 KV Cache 里的起始 Block、当前已用长度等。

Attention kernel 启动时看到的是类似这样的结构：

- 一个大 batch，批大小 = \(B\)；
- 对第 \(i\) 条样本：
  - `num_new_tokens[i]`：本轮要计算多少个新 token；
  - `kv_len[i]`：历史 KV 的长度是多少；
  - `is_prefill[i] / is_decode[i]`：当前处于哪个阶段（也可以用 `kv_len==0` 之类间接区分）。

在 kernel 内部：

- **对于 Prefill 样本：**
  - 读入 `num_new_tokens[i]` 个新 token，计算它们的 Q/K/V；
  - 用它们的 Q 与自身这批 token 的 K/V 做 Attention；
  - 把这些 token 的 K/V **整体写入 KV Cache**。
- **对于 Decode 样本：**
  - 读入 1 个新 token，计算它的 Q；
  - 与“历史所有 KV”做 Attention（**不重算历史 token 的 K/V**）；
  - 把这个新 token 的 K/V **追加写入 KV Cache 尾部**。

> 关键在于：  
> - **同一轮 batch 的“逻辑分支”是在 kernel 里按样本（per-sample）处理的**，
>   
>   而不是“先启动一个 GPU kernel 专门算 Prefill，再启动一个 kernel 专门算 Decode”；  
>   
> - 对 GPU 来说，这就是“一个 batch，其中每一行样本有不同的 `num_new_tokens` / 不同的 KV 长度和写入方式”。
>
>   算子本身是同一类，只是参数和输入 shape 因样本而异。

当然，出于效率考虑，vLLM 的调度器会**尽量把形状/阶段相近的样本凑在同一批**里（例如一批主要是 Decode、长度差不多），以减少 padding 和分支开销，但从功能上是 **支持 Prefill 和 Decode 混合在同一 batch 里的**。

##### 再回到“能不能混”的直觉问题

现在可以更精确地回答你最初的疑问：

> “能组成 batch 的前提是，这一批可以按照相同的逻辑处理……  
>  prefill 和 decode 的处理方式是一样的吗？一个 batch 内可以同时存在吗？”

更准确的说法是：

- **从“网络结构/数学”角度**：两者是同一套 Transformer 前向；
- **从“执行路径”角度**：
  - Prefill：这条样本这一轮有很多新 Q，要写入很多 KV；
  - Decode：这一轮只有 1 个新 Q，只追加 1 个 KV；
  - 差异通过 **per-sample meta + kernel 内部分支**解决；
- 因此：
  - **可以在同一个 batch 里同时存在 Prefill 和 Decode 的 seq**，这是 vLLM 连续批处理的一个重要能力；
  - 为了更高效率，Scheduler 会尽量把“类型/长度接近的 seq”凑成批（例如一批主要是 Decode），但这属于优化策略，而不是功能限制。

> 总结一句话：
>
> **同一个 batch 可以同时包含 Prefill 和 Decode 的 seq，**
>
> **因为在底层 Attention / Transformer 算子里，它们是“同一个程序、不同样本配置”的两种情况，而不是两套完全不同的计算逻辑。**

### `--trust-remote-code`：信任远程代码

某些模型（如 Qwen 早期版本、ChatGLM、Baichuan 等）包含自定义代码，需要加这个参数才能加载。

| 情况 | 是否需要 |
|------|---------|
| 官方 HuggingFace 模型（如 LLaMA、Mistral） | 通常不需要 |
| 国产模型（Qwen、ChatGLM、Baichuan） | 很多需要 |
| 报错 `trust_remote_code` | 加上这个参数 |

**安全提示**：只对你信任的模型使用此参数，它允许执行模型仓库中的 Python 代码。

## 参数联动：理解并发能力的本质

很多新手会分开理解每个参数，但它们其实是**联动**的，共同决定了你的 vLLM 服务能承载多少并发。

### 核心公式

```
实际并发能力 = KV Block Pool 大小 ÷ 单请求平均 KV Cache 占用
```

**其中：**
- **KV Block Pool 大小**：由 `--gpu-memory-utilization` 决定（剩余显存的多少比例给 KV Cache）
- **单请求最大占用**：由 `--max-model-len` 决定（单请求最多用多少 token）
- **并发上限**：由 `--max-num-seqs` 决定（软上限，实际受 Block Pool 限制）

### 参数联动示例：24G 显存能跑什么？

以 RTX 4090 (24 GB) + Qwen 7B + FP16 为例：

```
GPU 显存 (24 GB)
│
├─ 模型权重: ~14 GB（固定）
├─ CUDA overhead: ~1 GB（固定）
│
└─ 剩余可用: 9 GB
      │
      └─ × gpu-memory-utilization (0.85)
         │
         └─ KV Block Pool: ~7.6 GB
               │
               └─ 每个 2K 请求需要 ~1 GB
                  → 理论并发: 7.6 ÷ 1 ≈ 7 个 2K 并发
```

**验证方法**：启动后看日志 `GPU blocks: 956`
- 每个 2K 请求需要 2048 ÷ 16 = 128 blocks
- 并发 = 956 ÷ 128 ≈ 7 个并发

**如果改成 4K 上下文：**
- 每个 4K 请求需要 4096 ÷ 16 = 256 blocks
- 并发 = 956 ÷ 256 ≈ 3 个并发（并发能力下降）

### 参数权衡

**为什么 `--gpu-memory-utilization` 不设 0.95？**
- CUDA context 和 kernel workspace 需要空间（约 0.5-1GB）
- 显存占用会有波动，需要 buffer
- 显存碎片会导致"账面够用但实际分配失败"

**为什么 `--max-model-len` 不设 8192 或更大？**
- 单请求占用更多 Block，并发能力下降（如上面的例子，4K 并发从 7 降到 3）
- 启动变慢（vLLM 会按最大长度预分配结构）
- 第一个请求可能直接 OOM（如果显存不够分配）

**结论：**
- `--gpu-memory-utilization`：0.85-0.9 是安全区间
- `--max-model-len`：4096 对大多数场景足够，显存充足可用 8192

### 如何判断你的配置是否合理？

启动后检查这些指标：

| 检查项 | 期望值 | 如果不符合 | 调整方法 |
|--------|-------|-----------|---------|
| `GPU blocks` | 至少几百（如 500+） | blocks 太少（< 500） | 降低 `--max-model-len` 或提高 `--gpu-memory-utilization`（小心 OOM） |
| `Maximum running requests` | 符合你的并发预期 | 太小 | 提高 `--max-num-seqs`（但实际受 blocks 限制） |
| 首次请求成功 | 不 OOM | OOM | 参数太激进，降低 `--gpu-memory-utilization` 或 `--max-model-len` |

## 按显存快速选择启动命令

根据你的显存，直接复制对应命令（保守配置，优先保证能跑起来）：

### 8-12 GB 显存

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

**并发能力**：约 3-5 个 2K 上下文并发

### 16-24 GB 显存（最推荐）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

**并发能力**：约 6-8 个 2K 上下文并发（**最通用，推荐**）

### 24-48 GB 显存

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

**并发能力**：约 10-15 个 4K 上下文并发

## 常见问题快速解决

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| **下载超时** | `Connection to huggingface.co timed out` | 设置 `export HF_ENDPOINT=https://hf-mirror.com`（见环境准备部分） |
| **启动 OOM** | `CUDA out of memory` 在启动阶段 | 按顺序调整：1) 降低 `--gpu-memory-utilization`（0.85 → 0.7）2) 降低 `--max-model-len`（8192 → 4096）3) 换更小的模型 |
| **首请求 OOM** | 启动成功，第一个请求 OOM | 同"启动 OOM"，降低 `--gpu-memory-utilization` 或 `--max-model-len` |
| **模型找不到** | `Model not found` 或 `No such file or directory` | 检查模型名拼写（注意大小写），或使用本地路径：`--model /data/models/Qwen2.5-7B-Instruct` |
| **端口被占用** | `Address already in use` | 换端口：`--port 8001` 或停止占用进程：`lsof -i:8000` 然后 `kill <PID>` |
| **CUDA 不可用** | `CUDA not available` | 检查 GPU 驱动：`nvidia-smi`，检查 PyTorch GPU 版本：`python -c "import torch; print(torch.cuda.is_available())"` |

**双模型同机时**：每个服务的 `--gpu-memory-utilization` 是作用在「自己启动时看到的剩余显存」上，不是对总显存的比例；先启的进程会占掉大部分显存，后启的只能在剩余里分配。规划时保证两进程实际占用之和 ≤ 总显存并留余量即可，详见「双模型同机部署」一节。

## 测试调用：验证服务可用

### curl 测试（最简单）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 10
  }'
```

### 流式输出测试

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "写一首关于春天的诗"}],
    "stream": true
  }'
```

**成功标志**：看到逐行输出的 `data: {...}` 格式，每行一个 token，最后以 `data: [DONE]` 结束。

### Python SDK 调用（推荐）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不需要 API key，但 OpenAI SDK 要求必须有
)

# 非流式调用
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**安装依赖：** `pip install openai`

# 工程化部署与日常运维

上一章的启动方式是"前台运行"，适合调试。生产环境需要：后台运行、日志持久化、多模型共存、缓存管理。

## 后台运行与日志管理（nohup 模式）

### 基本 nohup 启动

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  > vllm.log 2>&1 &
```

**命令拆解：**

| 部分 | 含义 |
|-----|------|
| `nohup` | 忽略挂断信号，终端关闭后进程继续运行 |
| `> vllm.log` | 标准输出重定向到日志文件 |
| `2>&1` | 标准错误也重定向到同一文件 |
| `&` | 后台运行 |

**启动后立即做两件事：**

```bash
# 1. 记录 PID（后面停止服务要用）
echo $!
# 或者查看
ps aux | grep vllm

# 2. 确认启动成功（等待 1-2 分钟后）
tail -f vllm.log
# 看到 "Uvicorn running on http://0.0.0.0:8000" 说明成功
```

### nohup 的局限性

| 局限 | 说明 | 影响 |
|------|------|------|
| 崩溃不会自动重启 | 如果 vLLM 因为 OOM 或其他原因退出，进程就没了 | 服务中断 |
| 日志无限增长 | 没有自动轮转，日志文件会越来越大 | 磁盘撑爆 |
| 不便于管理 | 多个服务时，需要手动记录 PID | 运维复杂 |

**更好的方案（按复杂度递增）：**

| 方案 | 适用场景 | 特点 |
|------|---------|------|
| **tmux / screen** | 开发测试 | 可以随时 attach 看日志，但仍需手动重启 |
| **systemd** | 生产环境 | 自动重启、开机启动、日志集成 journald |
| **Docker + restart policy** | 容器化部署 | 环境隔离、易于迁移、自动重启 |

> 本文使用 nohup 讲解，因为它最简单、最容易理解。生产环境建议升级到 systemd 或 Docker。

### 日志管理

```bash
# 实时查看日志（Ctrl+C 退出）
tail -f vllm.log

# 查看最后 100 行
tail -100 vllm.log

# 搜索错误（不区分大小写）
grep -i error vllm.log
grep -i oom vllm.log

# 查看日志文件大小
ls -lh vllm.log
du -h vllm.log
```

**日志轮转（防止日志撑爆磁盘）：**

```bash
# 方法 1：手动轮转（适合偶尔清理）
# 先停止服务，再清理
mv vllm.log vllm.log.$(date +%Y%m%d)
# 重新启动服务

# 方法 2：使用 logrotate（推荐，自动轮转）
# 创建配置文件 /etc/logrotate.d/vllm
cat << 'EOF' | sudo tee /etc/logrotate.d/vllm
/path/to/vllm.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
EOF
```

### 进程管理

```bash
# 查看 vLLM 进程
ps aux | grep vllm

# 查看 GPU 占用（实时刷新）
nvidia-smi -l 1

# 查看哪个进程占用 GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

**停止服务：优雅停止 vs 强制停止**

| 方式 | 命令 | 说明 | 使用场景 |
|------|------|------|---------|
| 优雅停止 | `kill <PID>` | 发送 SIGTERM，允许进程清理资源 | 正常停止 |
| 强制停止 | `kill -9 <PID>` | 发送 SIGKILL，立即终止 | 进程卡死时 |

```bash
# 获取 PID
PID=$(ps aux | grep 'vllm.entrypoints' | grep -v grep | awk '{print $2}')

# 优雅停止（推荐）
kill $PID

# 等待几秒，确认进程退出
sleep 5
ps aux | grep vllm

# 如果还在，强制停止
kill -9 $PID
```

**为什么优先用优雅停止？**
- vLLM 会等待当前请求处理完
- 释放 GPU 显存
- 关闭网络连接

强制 kill 可能导致：
- 正在处理的请求直接失败
- GPU 显存可能没有正确释放（需要等 CUDA 自动回收）

## 双模型同机部署：端口与显存分配

典型场景：一个 Instruct 模型（对话）+ 一个 Embedding 模型（向量化），共用一张 GPU。

### 为什么可以双模型共存？

两个 vLLM 进程共用一张 GPU 时，**每个进程的 `--gpu-memory-utilization` 都是作用在「自己启动时看到的剩余显存」上**，不是作用在整块 GPU 的总显存上。因此：

- **第一个进程**（例如 Instruct）：

  加载模型权重 + CUDA context 后，在**剩余显存**上乘以 0.65，得到自己的 KV Block Pool 大小。

  它会占掉 GPU 里很大一块（权重 + 约 65% 的“当时剩余”）。

- **第二个进程**（例如 Embedding）：
  
  启动时 GPU 已被第一个进程占去大部分，它**只能看到“当时还剩多少显存”**。
  
  在这个剩余上再加载自己的权重 + CUDA context，然后在**新的剩余**上乘以 0.25，得到自己的 KV Block Pool。
  
  所以第二个进程实际占用的，是“第一个剩下来的那一块”里的一小部分，**不是**整块 GPU 的 25%。

因此：**两个参数的“数值相加”没有直接意义**。

例如 0.65 + 0.25 = 0.9，并不表示“一共用掉 90% 总显存”；

0.65 是“第一个进程在自己看到的剩余上拿 65%”，0.25 是“第二个进程在自己看到的剩余上拿 25%”。

甚至可以把两个都设成 0.9，只要第二进程启动时剩余显存足够装下自己的权重和一点 KV Pool 即可；

反之若第一个把显存吃得太狠，第二个就会 OOM 或几乎拿不到 KV Pool。

**正确的心智模型：**

1. 先启动的进程按「总显存 − 模型 − context」的剩余，乘以自己的 `--gpu-memory-utilization`，得到 KV Pool。

2. 后启动的进程按「总显存 − 第一个进程已占 − 自己模型 − context」的剩余，乘以自己的 `--gpu-memory-utilization`，得到自己的 KV Pool。

3. 共存条件：**两进程实际占用之和 ≤ GPU 总显存**。

   规划时建议预留一点余量（例如合计不超过 0.9～0.95 的“经验安全值”），留给 CUDA context、碎片和波动，避免边界 OOM。

### 显存分配的计算逻辑

以 24 GB 显存、先启 Instruct 再启 Embedding 为例，**按“谁先启动谁先占”的顺序**理解：

| 阶段 | 显存占用 |
|------|----------|
| **第一个进程（Instruct）启动** | 权重 ~14 GB + CUDA context；剩余约 10 GB × 0.65 ≈ 6.5 GB 做 KV Pool；合计约 20.5 GB |
| **第二个进程（Embedding）启动** | 此时 GPU 只剩约 3.5 GB；再装 Embedding 权重 ~1.2 GB + 自己的 context；在**新剩余**上 × 0.25 做 KV Pool；若剩余不足则可能 OOM 或 KV 很小 |

**分配思路（说人话）：**

1. 先算第一个模型要占多少：权重 + 合理 KV（由 0.65 作用于“总显存 − 权重 − context”的剩余）。

2. 再算“还剩多少给第二个”：总显存 − 第一个进程全部占用。

3. 第二个模型必须在这块“剩余”里装下：自己的权重 + context + 自己的 KV Pool（由 0.25 作用于“剩余 − 自己权重 − context”）。

4. 若希望两模型都跑得舒服，就控制第一个不要占太满（例如 0.65），给第二个留足空间；

   **不是**简单要求“两个参数之和 ≤ 0.9”，而是保证“两进程实际占用之和 ≤ 总显存，并留一点余量”。

### 双模型启动命令

**重要：必须等第一个模型完全启动后，再启动第二个。**

```bash
# 步骤 1：启动 Instruct 模型（端口 8000）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 4096 \
  > instruct.log 2>&1 &

# 步骤 2：等待第一个模型启动完成（关键！）
echo "等待 Instruct 模型启动..."
sleep 60  # 首次启动可能需要更长时间
tail -5 instruct.log  # 确认看到 "Uvicorn running"

# 步骤 3：确认第一个模型占用显存后，再启动第二个
nvidia-smi  # 确认显存已被占用

# 步骤 4：启动 Embedding 模型（端口 8001）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/qwen3-embedding-0.6b \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 4096 \
  > embedding.log 2>&1 &
```

**为什么要等待？**

| 如果不等待 | 后果 |
|-----------|------|
| 两个模型同时初始化 | 显存分配冲突，可能都失败 |
| vLLM 会预分配显存 | 第一个模型还没占位，第二个可能"抢"走太多 |

### 调用方式

两个模型使用不同端口和 API 接口：

```bash
# 调用 Instruct 模型（对话）
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'

# 调用 Embedding 模型（向量化）
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/qwen3-embedding-0.6b",
    "input": "这段话需要向量化"
  }'
```

> **注意**：Embedding 模型使用 `/v1/embeddings` 接口，不是 `/v1/chat/completions`。

### 统一管理脚本（推荐）

保存为 `start_vllm_dual.sh`：

```bash
#!/bin/bash
# 双模型统一启动脚本，适合 24 GiB 显存
# 特点：顺序启动、等待确认、健康检查

set -e  # 遇到错误立即退出

# ============ 配置参数 ============
INSTRUCT_MODEL="Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL="Qwen/qwen3-embedding-0.6b"
INSTRUCT_PORT=8000
EMBED_PORT=8001
INSTRUCT_GPU_UTIL=0.65
EMBED_GPU_UTIL=0.25
MAX_LEN=4096
DTYPE="float16"
LOG_DIR="./logs"

# ============ 准备工作 ============
mkdir -p $LOG_DIR
echo "日志目录: $LOG_DIR"

# ============ 健康检查函数 ============
wait_for_service() {
    local port=$1
    local name=$2
    local max_wait=180  # 最多等待 3 分钟
    local waited=0
    
    echo "等待 $name 启动 (端口 $port)..."
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "$name 启动成功！"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  已等待 ${waited}s..."
    done
    
    echo "错误: $name 启动超时！"
    return 1
}

# ============ 启动 Instruct 模型 ============
echo ""
echo "========== 启动 Instruct 模型 =========="
nohup python -m vllm.entrypoints.openai.api_server \
  --model $INSTRUCT_MODEL \
  --host 0.0.0.0 \
  --port $INSTRUCT_PORT \
  --dtype $DTYPE \
  --gpu-memory-utilization $INSTRUCT_GPU_UTIL \
  --max-model-len $MAX_LEN \
  > $LOG_DIR/instruct.log 2>&1 &
INSTRUCT_PID=$!
echo "Instruct PID: $INSTRUCT_PID"

# 等待第一个模型启动完成
wait_for_service $INSTRUCT_PORT "Instruct 模型"

# ============ 启动 Embedding 模型 ============
echo ""
echo "========== 启动 Embedding 模型 =========="
nohup python -m vllm.entrypoints.openai.api_server \
  --model $EMBED_MODEL \
  --host 0.0.0.0 \
  --port $EMBED_PORT \
  --dtype $DTYPE \
  --gpu-memory-utilization $EMBED_GPU_UTIL \
  --max-model-len $MAX_LEN \
  > $LOG_DIR/embedding.log 2>&1 &
EMBED_PID=$!
echo "Embedding PID: $EMBED_PID"

# 等待第二个模型启动完成
wait_for_service $EMBED_PORT "Embedding 模型"

# ============ 输出汇总 ============
echo ""
echo "============================================"
echo "双模型启动完成！"
echo ""
echo "Instruct 模型:"
echo "  - PID: $INSTRUCT_PID"
echo "  - 端口: $INSTRUCT_PORT"
echo "  - 日志: $LOG_DIR/instruct.log"
echo ""
echo "Embedding 模型:"
echo "  - PID: $EMBED_PID"
echo "  - 端口: $EMBED_PORT"
echo "  - 日志: $LOG_DIR/embedding.log"
echo ""
echo "测试命令:"
echo "  curl http://localhost:$INSTRUCT_PORT/v1/models"
echo "  curl http://localhost:$EMBED_PORT/v1/models"
echo "============================================"
```

使用方法：

```bash
chmod +x start_vllm_dual.sh
./start_vllm_dual.sh
```

**脚本特点：**
- 顺序启动，避免显存冲突
- 健康检查，确认启动成功再继续
- 超时退出，避免无限等待
- 清晰的输出信息

### 停止双模型

```bash
# 方法 1：根据端口找进程并停止
kill $(lsof -t -i:8000)
kill $(lsof -t -i:8001)

# 方法 2：根据进程名停止所有 vLLM
pkill -f "vllm.entrypoints"
```

## 磁盘空间规划与模型缓存迁移

### 模型文件大小参考

| 模型规模 | 大约磁盘占用 |
|---------|-------------|
| 3B | 6 – 8 GB |
| 7B | 14 – 16 GB |
| 14B | 28 – 32 GB |
| 70B | 130 – 150 GB |

### 缓存目录配置

默认情况下，HuggingFace 模型缓存在 `~/.cache/huggingface/`。

在云服务器（如 AutoDL）上，系统盘空间有限，建议将缓存放到数据盘：

```bash
# 设置缓存目录（写入 bashrc 一劳永逸）
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface' >> ~/.bashrc
source ~/.bashrc
```

### 模型预下载到数据盘

如果不想依赖在线下载，可以提前下载模型：

```bash
# 创建模型目录
mkdir -p /data/models
cd /data/models

# 使用 git lfs 下载
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

然后启动时指定本地路径：

```bash
--model /data/models/Qwen2.5-7B-Instruct
```

### 软链接方式（系统盘程序 + 数据盘模型）

```bash
# 创建软链接
mkdir -p ~/.cache/huggingface/hub
ln -s /data/models/Qwen2.5-7B-Instruct ~/.cache/huggingface/hub/Qwen--Qwen2.5-7B-Instruct
```

这样即使用 HuggingFace 模型名启动，也会使用数据盘上的模型文件。

## 生产环境检查清单

### 启动前检查

| 检查项 | 命令 | 期望结果 |
|-------|------|---------|
| 显存足够 | `nvidia-smi` | 可用显存 > 模型需求（7B 需要 ~16GB） |
| 磁盘空间 | `df -h /path/to/cache` | 剩余空间 > 模型大小 × 2 |
| 端口未占用 | `lsof -i:8000` | 无输出（端口空闲） |
| 环境变量 | `echo $HF_ENDPOINT` | 显示镜像地址（国内服务器） |
| CUDA 正常 | `python -c "import torch; print(torch.cuda.is_available())"` | True |

**快速检查脚本：**

```bash
#!/bin/bash
echo "=== 启动前检查 ==="
echo "1. GPU 状态:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "2. 磁盘空间:"
df -h . | head -2
echo ""
echo "3. 端口 8000:"
lsof -i:8000 || echo "端口空闲 ✓"
echo ""
echo "4. 环境变量:"
echo "   HF_ENDPOINT: ${HF_ENDPOINT:-未设置}"
echo "   HF_HOME: ${HF_HOME:-未设置}"
```

### 启动后检查

| 检查项 | 命令 | 期望结果 |
|-------|------|---------|
| 进程存活 | `ps aux \| grep vllm` | 看到 python 进程 |
| 显存占用 | `nvidia-smi` | 显存被占用（7B 约 14-16GB） |
| 日志无报错 | `grep -i "error\|oom" vllm.log` | 无输出 |
| API 可访问 | `curl http://localhost:8000/v1/models` | 返回模型列表 JSON |
| 健康检查 | `curl http://localhost:8000/health` | 返回 200 OK |

**快速验证命令：**

```bash
# 一条命令验证服务是否正常
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":10}' \
  | python -m json.tool
```

### 日常运维

| 任务 | 频率 | 命令/操作 |
|------|------|---------|
| 检查进程 | 每天 | `ps aux \| grep vllm` |
| 检查显存 | 每天 | `nvidia-smi`（注意显存是否持续增长） |
| 检查日志大小 | 每周 | `du -h vllm.log`（超过 1GB 考虑轮转） |
| 日志轮转 | 每月 | 见上文 logrotate 配置 |
| 检查磁盘 | 每周 | `df -h`（确保不会满） |

### 常见运维问题速查

| 现象 | 可能原因 | 排查命令 |
|------|---------|---------|
| 服务无响应 | 进程挂了 | `ps aux \| grep vllm` |
| 响应变慢 | 并发太高 / 显存不够 | `nvidia-smi` 看显存和 GPU 利用率 |
| 日志报 OOM | 请求太长 / 并发太多 | 降低 `--max-model-len` 或限制并发 |
| 磁盘满 | 日志太大 | `du -sh vllm.log`，然后轮转 |
| GPU 利用率低 | batch 太小 | 正常现象（decode 阶段），不一定是问题 |

# KV-Cache与PagedAttention：核心原理

## 推理到底在算什么：token-by-token 生成

LLM 推理是**逐 token 生成**的过程：

```
输入：今 天 天 气 怎 么 样
生成第 1 个 token：很        （上下文 = 输入 7 个 token）
生成第 2 个 token：好        （上下文 = 8 个 token）
生成第 3 个 token：！        （上下文 = 9 个 token）
```

每生成一个 token，上下文就增加一个。如果每次都从头计算所有 token 的 Attention，计算量会随上下文长度**平方增长**——这在工程上不可接受。

于是，一个关键优化被引入：**KV Cache**。

## KV Cache 是什么

### 先回顾 Self-Attention 的计算

如果你了解 Transformer，应该见过这个公式：

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

其中：
- **Q (Query)**：当前 token 的查询向量，用于"提问"
- **K (Key)**：所有 token 的键向量，用于"被匹配"
- **V (Value)**：所有 token 的值向量，用于"提供信息"

**关键观察**：计算当前 token 的输出时，Q 只来自当前 token，但 K 和 V 需要包含**所有历史 token**。

### 为什么 K/V 可以缓存？

在 Decoder-only 模型（GPT 类）的推理阶段：

| 阶段 | Q | K | V |
|------|---|---|---|
| 生成第 1 个 token | Q(1) | K(1) | V(1) |
| 生成第 2 个 token | Q(2) | K(1), K(2) | V(1), V(2) |
| 生成第 3 个 token | Q(3) | K(1), K(2), K(3) | V(1), V(2), V(3) |

观察规律：
- **Q 每次都不同**（只需要当前 token 的）
- **K 和 V 是累加的**（历史 token 的 K/V 不变，只追加新 token 的）

因此：

> **KV Cache = 把历史 token 的 K/V 向量缓存起来，下次只计算新 token 的 K/V，然后拼接。**

### 用一个具体例子理解

假设生成 "I am" 两个 token，在第 0 层 Transformer：

**生成第 1 个 token "I"：**

```python
x1 = embedding("I")
Q0(1) = Wq · x1    # 查询向量
K0(1) = Wk · x1    # 键向量 → 存入 KV Cache
V0(1) = Wv · x1    # 值向量 → 存入 KV Cache

# Attention: Q0(1) 和 K0(1), V0(1) 计算
```

**生成第 2 个 token "am"：**

```python
x2 = embedding("am")
Q0(2) = Wq · x2    # 只算当前 token 的 Q
K0(2) = Wk · x2    # 只算当前 token 的 K → 追加到 KV Cache
V0(2) = Wv · x2    # 只算当前 token 的 V → 追加到 KV Cache

# Attention: Q0(2) 和 [K0(1), K0(2)], [V0(1), V0(2)] 计算
#            ↑ 历史的 K0(1), V0(1) 直接从 Cache 读取，不重算
```

**计算量对比：**

| 方式 | 生成第 N 个 token 时的计算量 |
|------|----------------------------|
| 无 Cache | 计算 N 个 token 的 Q/K/V + N² 的 Attention |
| 有 Cache | 计算 1 个 token 的 Q/K/V + N 的 Attention |

这就是为什么 **KV Cache 是所有大模型推理的基础优化**。

### KV Cache 的精确定义

> **KV Cache = 对于每一层 L、每一个已生成 token t，存储该 token 在该层产生的 K_L(t) 和 V_L(t)**

用结构图表示（假设 2 层、3 个 token）：

```
Token t1:
  Layer 0: K0(t1), V0(t1)
  Layer 1: K1(t1), V1(t1)

Token t2:
  Layer 0: K0(t2), V0(t2)
  Layer 1: K1(t2), V1(t2)

Token t3:
  Layer 0: K0(t3), V0(t3)
  Layer 1: K1(t3), V1(t3)
```

注意：**每一层都有独立的 KV Cache**，因为每一层的 Wk、Wv 不同，产生的 K/V 也不同。

### 为什么只缓存 K/V，不缓存 Q 或 hidden state

| 对象 | 是否适合缓存 | 原因 |
|-----|-------------|------|
| **Q (Query)** | ❌ | Q 只对当前 token 有意义。生成下一个 token 时，需要的是新 token 的 Q，历史 Q 完全无用 |
| **hidden state** | ❌ | hidden state 是每一层的中间结果，会参与 LayerNorm、MLP 等运算，每一层都不同，缓存成本高、复用率低 |
| **K / V** | ✅ | K/V 在推理阶段一旦算出就不变，且所有未来 token 的 Attention 都需要它们 |

**深入理解**：从第 2 层开始，K/V 已经编码了"上下文信息"

```python
# 第 0 层：K/V 只来自 token 自己的 embedding
K0(t) = Wk0 · embedding(t)

# 第 1 层：K/V 来自第 0 层的输出 h0(t)，而 h0(t) 已经融合了历史 token
h0(t) = Attention(Q0(t), K0(≤t), V0(≤t))  # 包含了历史 token 的信息
K1(t) = Wk1 · h0(t)  # K1 已经"知道"上下文了
```

这意味着：**即使两个对话中出现相同的 token（如 "am"），它们的 K/V 值也完全不同**（因为上下文不同）。这是后面理解"vLLM 不能跨对话复用 KV Cache"的关键。

### 显存估算：用 7B 模型算一笔账

选一个真实模型配置（Qwen / LLaMA 7B 级）：

```
Transformer 层数（L）       = 32
Hidden size               = 4096
Attention heads           = 32
Head dim                  = 128  (4096 / 32)
KV heads                  = 32   (标准 MHA)
数据类型                  = FP16 (2 bytes)
```

#### 单层、单 token 的 K/V 大小

```
K_l(t): shape = [num_heads, head_dim] = [32, 128]
V_l(t): shape = [num_heads, head_dim] = [32, 128]

K + V = 2 × 32 × 128 = 8192 个 FP16 数
     = 8192 × 2 bytes = 16 KB
```

这是"1 个 token，在 1 层"的 KV Cache。

#### 加上层数

```
16 KB × 32 层 = 512 KB / token
```

> **每生成 1 个 token，就永久新增 ~512 KB 的 KV Cache**

#### 加上上下文长度

假设 2048 tokens 的上下文：

```
512 KB × 2048 = 1,048,576 KB ≈ 1 GB
```

> **一个 7B 模型，一个 2K 上下文的请求，KV Cache 就要 ~1 GB 显存**

#### 多并发场景

| 场景 | KV Cache 显存 | 说明 |
|------|---------------|------|
| 1 请求 × 2K 上下文 | 1 GB | 单用户聊天 |
| 8 并发 × 2K 上下文 | 8 GB | 小团队使用 |
| 16 并发 × 4K 上下文 | 32 GB | 生产服务 |
| 32 并发 × 8K 上下文 | 128 GB | 需要多卡 |

一张 A100 40G，模型权重 14GB + CUDA 开销 ~2GB，剩余约 24GB 给 KV Cache，也就是 **最多 24 个 2K 并发**。

**快速估算公式（7B 模型，FP16，标准 MHA）：**

```
KV Cache (GB) ≈ 并发数 × 上下文长度(K) × 0.5
```

例如：8 并发 × 4K = 8 × 4 × 0.5 = 16 GB

### GQA/MQA：新模型如何压缩 KV Cache

标准的 Multi-Head Attention (MHA) 中，每个 Attention head 都有独立的 K 和 V：

```
Attention heads = 32
KV heads = 32（每个 head 一套 K/V）
```

但研究者发现：**K/V 不需要和 Q 一样多**。于是有了两种优化：

| 技术 | KV heads | 说明 |
|------|----------|------|
| **MQA (Multi-Query Attention)** | 1 | 所有 Q heads 共享同一套 K/V |
| **GQA (Grouped Query Attention)** | 介于 1 和 Attention heads 之间 | Q heads 分组，每组共享一套 K/V |

**“K/V 不需要和 Q 一样多”是什么意思？**

这里说的“多/少”指的是 **K、V 的“套数”（即有多少组独立的 K、V 投影）**，不是序列长度。

- 标准 MHA：有 32 个 head，每个 head 各自有一套 Q、K、V 的线性投影 → 32 套 Q、32 套 K、32 套 V。

- 所谓“K/V 不需要和 Q 一样多”：**K/V 的套数可以少于 Q 的 head 数**，让多组 Q 共用同一套（或同一组）K、V。

  这样显存里只需存“少数几套”K、V，KV Cache 就变小了。

**MQA（只有 1 套 K/V）时，32 个 Q 怎么算注意力？**

- 只有 **1 套 K、1 套 V**（整段序列经这一套投影后，得到一个 K 矩阵、一个 V 矩阵，形状均为 [序列长度, head_dim]）。

- **32 个 Q head** 仍然各自有一个 Q 向量（或 Q 矩阵）。

- 对每个 head i：  
  - 用该 head 的 Q 和**这一套共享的 K** 算注意力权重：`α_i = softmax(Q_i @ K^T)`；  
  
  - 再用 `α_i` 对**这一套共享的 V** 做加权和：`out_i = α_i @ V`。
  
    因为每个 head 的 Q 不同，所以得到的 `α_i` 不同，对同一套 V 的加权和就不同。
  
- 最终这一层的输出是 **32 个 head 的 out_i 拼起来**：`h = concat(out_1, ..., out_32)`。

  所以依然是 32 个“通道”的输出，只是这 32 个通道**共用同一套 K、V**，各自用不同的注意力权重去加权 V。

**GQA（例如 8 套 K/V、32 个 Q）时，怎么“分配”？**

- 把 **32 个 Q head 分成 8 组**，每组 4 个 Q head。

- **8 套 K/V**：每组对应一套 K、一套 V（即 K_1,V_1 … K_8,V_8）。

- 第 g 组（g=0..7）：  
  - 4 个 Q head 共用 **同一套 K_g、V_g**；  
  - 每个 head 仍用自己的 Q 和 K_g 算注意力权重，再对 V_g 做加权和，得到 4 个不同的 out；  
  - 这 4 个 out 拼成这一组的 4 个通道。
  
- 8 组共得到 32 个 out，再拼成最终的 `h`。

  所以“分配”方式就是：**按组划分 Q head，同一组内的多个 Q 共用同一套 K、V；组与组之间用不同的 K、V。**

总结一句：

**MQA/GQA 都是“多组 Q 共用少量 K、V”；每个 Q head 仍然用自己的 Q 和对应的 K、V 算注意力、得到自己的输出向量，只是 K、V 的套数变少了，所以 KV Cache 变小。**

**以 LLaMA 2 70B 为例：**

```
Attention heads = 64
KV heads = 8（GQA）
→ 每 8 个 Q heads 共享一套 K/V
→ KV Cache 缩小 64/8 = 8 倍
```

**对 KV Cache 的影响：**

| 配置 | 单 token KV Cache 大小 (7B 模型) |
|------|--------------------------------|
| MHA (KV heads = 32) | 512 KB |
| GQA (KV heads = 8) | 128 KB |
| MQA (KV heads = 1) | 16 KB |

这是为什么 LLaMA 2/3、Qwen 2.5 等新模型都采用 GQA 的原因——**在几乎不损失效果的情况下，大幅减少显存占用**。

**如何查看你的模型用的是什么？**

```bash
# 查看模型配置
cat ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/config.json | grep -E "(num_attention_heads|num_key_value_heads)"
```

如果 `num_key_value_heads` < `num_attention_heads`，就是 GQA。

## 多并发下的工程灾难

### 长度不一、释放回收、显存碎片

KV Cache 的真正难点不是"大"，而是：

> **KV Cache 是"会不断增长、长度各不相同、生命周期不可预测"的内存对象。**

这三个特性同时出现，才是工程噩梦：

| 特性 | 说明 |
|-----|------|
| **不断增长** | 每生成一个 token，KV Cache 就在尾部多加一段。一开始分配的空间几乎一定不够用 |
| **长度各不相同** | 请求 A 生成 50 token，请求 B 生成 4000 token，无法提前知道 |
| **生命周期不可预测** | 用户随时断连，streaming 过程中被 cancel，显存要频繁释放再重新分配 |

### 传统做法的局限

传统推理框架的思路：

> **"一个请求，占一整块连续的 KV Cache 空间。"**

这就像电影院排座位：

- 显存 = 一整排座位
- 每个请求 = 一群人，必须连续坐在一起

问题：

```
第一批来 5 人 → 坐 1–5
第二批来 8 人 → 坐 6–13
第三批来 3 人 → 坐 14–16

第二批走了 → 座位 6–13 空了

新来 10 人：
  总空位 ≥ 10
  但没有连续的 10 个座位！
```

这就是 **显存碎片化（fragmentation）**。

传统做法导致：

- 显存利用率很低
- 可用显存 ≠ 可分配显存
- 并发一上来就崩

于是很多系统只能：限制最大上下文长度、限制最大并发、或直接拒绝请求。

## PagedAttention：把 KV Cache 当显存页管理

### 直觉类比：操作系统的虚拟内存分页

> **PagedAttention = 不再要求 KV Cache 在显存中连续，而是拆成固定大小的"页"来存。**

如果你学过操作系统，会立刻产生共鸣：

| 操作系统 | vLLM |
|---------|------|
| Page | KV Block |
| 虚拟地址 | token 序号 |
| 页表 | Block Table |
| 物理内存 | GPU 显存 |

**vLLM 把操作系统的虚拟内存管理，搬进了 GPU 显存。**

### KV Block / Block Table / Block Pool

#### KV Block 的结构

一个 KV Block = **连续 N 个 token 的 KV（所有层）**

```
KV Block (block_size = 16 tokens)

Token t0:   [Layer0: K,V] [Layer1: K,V] ... [Layer31: K,V]
Token t1:   [Layer0: K,V] [Layer1: K,V] ... [Layer31: K,V]
...
Token t15:  [Layer0: K,V] [Layer1: K,V] ... [Layer31: K,V]
```

**关键设计决策：为什么 Block 包含所有层？**

Block 有两种可能的切法：

| 切法 | 结构 | 问题 |
|------|------|------|
| **按层切** | 一个 Block = 某一层的 N 个 token | Decode 时需要遍历所有层，同一 token 的 KV 分散在不同 Block，访存跳跃 |
| **按 token 切** ✅ | 一个 Block = N 个 token 的所有层 | 同一 token 的 KV 连续存放，访存友好 |

vLLM 选择**按 token 切**，因为 Decode 阶段的 Attention kernel 是逐层执行的：

```
for layer in range(32):
    # 每一层都需要访问所有历史 token 的 K/V
    attn_output = attention(Q[layer], K_cache[layer], V_cache[layer])
```

如果同一 token 的 32 层 KV 是连续的，GPU 的缓存利用率更高。

#### vLLM 内部维护的三样东西

| 结构 | 说明 |
|------|------|
| **Block Pool** | 所有可用 KV Block 的"池子" |
| **Block Table** | 每个请求 → 对应哪些 Block |
| **映射关系** | 逻辑 token 序列 → 实际显存位置 |

运行时的状态：

```
KV Block Pool (GPU memory):

[ A0 ][ B0 ][ A1 ][ C0 ][ A2 ][ free ][ free ]

A、B、C 是不同请求，每个 block = 16 tokens
Block 可以在显存中任意位置，只要逻辑顺序对即可
```

#### PagedAttention 的硬核之处

传统 Attention kernel 假设 K/V 在内存中连续。

vLLM **定制了 Attention kernel**：

- kernel 不再假设连续内存
- 按 Block Table 一块一块地读 K/V
- 在计算层面"拼接成连续逻辑序列"

**逻辑连续 ≠ 物理连续**

### block size 为什么是 16/32

这是一个**工程折中解**，不是魔法数字。

#### 为什么不能 block size = 1（每 token 一个 block）？

| 问题 | 说明 |
|------|------|
| metadata 开销 | 每个 block 需要 id、指针、ref count 等，block 太小时 metadata 比 KV 本身还大 |
| GPU 访存对齐 | GPU 喜欢连续内存、可预测 stride、向量化 load；1 token 太小，cache line 利用率极低 |
| kernel 效率 | FlashAttention / PagedAttention 假设 sequence length 是一个小批量连续区间，block 太小导致频繁切换 |

#### 为什么不能 block size = 128（很大）？

| 问题 | 说明 |
|------|------|
| 尾部浪费 | block_size = 128 时，最坏浪费 127 tokens 的显存 |
| 调度粒度 | 每次只能以 128 token 为单位分配，小请求被大 block 卡死 |

#### 16/32 的平衡点

| block_size | 最坏浪费 | 调度灵活性 | kernel 效率 |
|------------|---------|-----------|------------|
| 1 | 0 | 极高 | 极低 |
| 16 | 15 tokens | 高 | 高 |
| 32 | 31 tokens | 中 | 很高 |
| 128 | 127 tokens | 低 | 极高 |

16/32 是当前"吞吐、延迟、显存利用率"的最优解区间。

### 收益与代价：为什么适合 vLLM，不适合 Ollama

#### PagedAttention 的收益

| 收益 | 说明 |
|------|------|
| 显存利用率大幅提升 | 不再需要"预留最大长度"，用多少 token 占多少 Block |
| 并发能力指数级提升 | 新请求只要有 Block 就能进，不需要等待"大块连续显存" |
| 请求结束立即回收 | 就像 free 掉几个 page，不影响其他请求 |
| Streaming / 长对话友好 | KV Cache 按需增长，不需要提前猜长度 |

#### PagedAttention 的代价

| 代价 | 说明 |
|------|------|
| 更复杂的内存管理 | 需要维护 Block Table、映射关系 |
| 更复杂的 kernel | 定制 Attention kernel，不能直接用标准实现 |
| 更高的系统复杂度 | 调度逻辑复杂，debug 更难 |

#### 为什么 Ollama 不用 PagedAttention

对 Ollama 的目标场景（单用户、低并发）：

- 显存压力可控
- 复杂性反而是负担
- 引入 PagedAttention 会牺牲稳定性、提高门槛、增加维护成本，却几乎得不到收益

> **这是一个"工程上理性"的选择，而不是技术能力问题。**

## 显存分配的全局视角：模型权重 vs KV Block Pool

vLLM 启动后，GPU 显存的分配：

```
GPU 显存
├─ 模型权重（固定，启动时加载）
├─ CUDA context / workspace（固定）
└─ KV Block Pool（动态，由 --gpu-memory-utilization 控制）
      ├─ 请求 A 的 Block
      ├─ 请求 B 的 Block
      └─ free blocks
```

**理解要点：**

| 部分 | 特性 | 大小 |
|------|------|------|
| 模型权重 | 固定，不随请求变化 | 7B ≈ 14GB (FP16) |
| KV Block Pool | 动态，服务所有并发请求 | 由 `--gpu-memory-utilization` 决定 |

KV Block Pool 的大小直接决定：

- 最大并发数
- 最大上下文容量

### 如何在 vLLM 日志中验证这些概念

启动 vLLM 时，日志会显示关键的内存分配信息：

```bash
# 启动时观察日志
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 2>&1 | tee startup.log
```

**关键日志解读：**

```
INFO: GPU blocks: 1500, CPU blocks: 512
```

这行告诉你：

| 信息 | 含义 | 计算 |
|------|------|------|
| `GPU blocks: 1500` | KV Block Pool 有 1500 个 block | |
| 每个 block = 16 tokens | vLLM 默认值 | |
| 总容量 | 1500 × 16 = 24000 tokens | |
| 单请求 4K 上下文 | 需要 4096 ÷ 16 = 256 blocks | |
| 理论最大并发 | 1500 ÷ 256 ≈ 5 个 4K 请求 | |

**手动计算你的配置能支持多少并发：**

```python
# 假设你看到日志 "GPU blocks: 1500"
gpu_blocks = 1500
block_size = 16  # tokens per block (vLLM 默认)

max_context_len = 4096  # 你设置的 --max-model-len
blocks_per_request = max_context_len // block_size  # = 256

max_concurrent = gpu_blocks // blocks_per_request  # = 5
print(f"理论最大并发: {max_concurrent} 个请求")
```

## 常见误解澄清：vLLM 的"复用"到底复用什么

### 误解：vLLM 可以跨对话复用 KV Cache

❌ **错误**

从第 2 层 Transformer 开始，每个 token 的 K/V 就已经编码了"整个上下文"的信息：

```
K₁(t) = Wk₁ · h₀(t)    // h₀(t) 已经融合了历史 token
```

所以：

- 换一个上下文，K/V 完全不同
- 即使 token 相同（如 "am"），在不同对话中 K/V 值完全不同

> **KV Cache 是"请求私有的语义状态"，不是"跨请求共享的模型记忆"**

### vLLM 真正复用的是什么？

| ❌ 不复用 | ✅ 真正复用 |
|----------|-----------|
| KV 的语义内容 | 显存空间（Block） |
| A 对话的 KV 给 B 用 | 调度结构 |
| | Attention kernel |
| | 执行流水线 |

> **vLLM 复用的是"内存与算力"，不是"上下文语义"**

### 对话结束后发生什么

```
[ Req A blocks ] [ Req B blocks ] [ free blocks ]
        ↓
Req A 结束
        ↓
[ free blocks ] [ Req B blocks ] [ free blocks ]
```

- Block 被标记为 free，可以立即分配给其他请求
- 显存不会被 memset 清零（太慢），只是逻辑上失效、物理上复用

# Scheduler调度器：token级并发管理

前面我们搞清楚了"KV Cache 存什么"和"PagedAttention 怎么管理显存"。现在进入最后一个核心问题：

> **多个请求同时在跑，vLLM 如何决定每一步 GPU 该算谁的 token？**

## Continuous Batching：vLLM 调度的核心创新

传统推理框架的调度是 **Request-level**（请求级）：

```
时间线：
[Req A 全部算完] → [Req B 全部算完] → [Req C 全部算完]
```

问题：如果 Req A 要生成 500 token，Req B 和 C 必须等 A 完全结束才能开始。

vLLM 的调度是 **Iteration-level**（迭代级），也叫 **Continuous Batching**：

```
时间线：
Step 1: [A_token_1, B_token_1, C_token_1]  → 一起算
Step 2: [A_token_2, B_token_2, C_token_2]  → 一起算
...
Step 50: [A_token_50]  → B、C 已结束，只剩 A
...
```

**核心区别：**

| 维度 | 传统调度 | Continuous Batching |
|------|---------|---------------------|
| 调度粒度 | 整个请求 | 单个 token（每次迭代） |
| 请求加入时机 | 等当前 batch 全部结束 | 任意 iteration 可加入 |
| 请求退出时机 | 等当前 batch 全部结束 | 生成完立即退出 |
| GPU 利用率 | 被最长请求拖累 | 持续保持高利用率 |

这就是为什么 vLLM 在高并发场景下吞吐远超传统框架。

## Prefill vs Decode：两个阶段的本质差异

Scheduler 需要区别对待两个阶段，因为它们的**计算特性完全不同**。

### Prefill（输入处理阶段）

处理用户输入的 prompt，一次性计算所有 token。

```
输入 prompt: "请解释什么是机器学习" (10 tokens)
Prefill: 一次 forward 处理 10 个 token，填充 KV Cache
```

**计算特性：Compute-bound（计算密集）**

| 特征 | 说明 |
|------|------|
| token 数量 | 多（几百到几千） |
| 矩阵运算 | 大（Q × K 矩阵是 [N, N]） |
| GPU 利用率 | 高（大矩阵运算吃满算力） |
| 瓶颈 | GPU 计算能力 |

### Decode（生成阶段）

逐个生成输出 token，每次只处理 1 个新 token。

```
已有上下文: 10 tokens
Decode step 1: 生成第 11 个 token
Decode step 2: 生成第 12 个 token
...
```

**计算特性：Memory-bound（访存密集）**

| 特征 | 说明 |
|------|------|
| token 数量 | 少（每请求每步只有 1 个） |
| 主要操作 | 读取 KV Cache（上下文越长，读取越多） |
| GPU 利用率 | 低（大量时间在等显存读取） |
| 瓶颈 | 显存带宽 |

### 为什么这个区别重要？

Scheduler 针对两个阶段采用不同策略：

| 阶段 | Scheduler 策略 |
|------|---------------|
| Prefill | 尽量凑大 batch，充分利用 GPU 算力 |
| Decode | 把多个请求的 decode 合并，摊薄访存开销 |

这也解释了一个常见现象：**首 token 延迟（TTFT）往往比后续 token 延迟高得多**——因为 Prefill 要处理整个 prompt。

## Scheduler 的输入与决策

每一轮迭代，Scheduler 看到的状态：

```python
# 伪代码：Scheduler 的视角
scheduler_state = {
    "waiting_queue": [新到的请求，还没开始 Prefill],
    "running_queue": [正在 Decode 的请求],
    "free_blocks": 150,  # 剩余可用 KV Block 数
    "max_batch_tokens": 4096,  # 单次 forward 最大 token 数
}
```

**决策过程：**

```
每一轮迭代:
1. 从 running_queue 收集所有需要继续 decode 的请求
2. 检查 waiting_queue，是否有新请求可以开始 Prefill
3. 计算：如果加入新请求，需要多少 block？
4. 如果 free_blocks 足够 → 加入新请求
5. 如果不够 → 新请求继续等待（背压）
6. 组成本轮 batch，提交 GPU
```

### 一个具体的数值例子

**场景：**
- GPU blocks = 1000，已用 800，剩余 200
- block_size = 16 tokens
- 当前 3 个请求在 Decode

**状态：**
```
Running:
  Req A: 上下文 1000 tokens, 占用 63 blocks
  Req B: 上下文 500 tokens, 占用 32 blocks  
  Req C: 上下文 200 tokens, 占用 13 blocks

Waiting:
  Req D: prompt 800 tokens, 需要 50 blocks
```

**Scheduler 决策：**
```
本轮 Decode: A、B、C 各生成 1 token → 需要 0~3 个新 block（取决于是否跨 block 边界）
新请求 D: 需要 50 blocks

剩余 200 blocks > 50 blocks → Req D 可以加入
本轮 batch = [A_decode, B_decode, C_decode, D_prefill]
```

**如果 free_blocks = 30：**
```
30 < 50 → Req D 必须等待
本轮 batch = [A_decode, B_decode, C_decode]
```

## 什么时候请求会被暂停？

Scheduler 可能**暂停（preempt）**一个正在运行的请求：

| 触发条件 | 说明 |
|---------|------|
| KV Block 即将耗尽 | 新请求 Prefill 需要空间，但 block 不够 |
| 优先级调度 | 高优先级请求需要资源 |

**暂停后会发生什么？**

vLLM 支持两种策略（通过 `--preemption-mode` 配置）：

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| `recompute` | 丢弃被暂停请求的 KV Cache，恢复时重新 Prefill | 显存紧张 |
| `swap` | 把 KV Cache 换出到 CPU 内存，恢复时换回 | CPU 内存充足 |

大多数情况下用默认的 `recompute` 即可。

## 从日志观察 Scheduler 行为

启动 vLLM 时，可以增加日志级别观察调度细节：

```bash
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server ...
```

**关键日志信息：**

```
# 每轮迭代的 batch 信息
Scheduler: num_running=5, num_waiting=2, num_swapped=0

# KV Block 使用情况
Block manager: used=850/1000, free=150

# Prefill vs Decode 统计
Batch: prefill_tokens=0, decode_tokens=5
```

**监控指标（如果接入 Prometheus）：**

| 指标 | 含义 |
|------|------|
| `vllm:num_requests_running` | 当前正在处理的请求数 |
| `vllm:num_requests_waiting` | 等待队列中的请求数 |
| `vllm:gpu_cache_usage_perc` | KV Block 使用率 |

## 为什么多 worker 不一定更快

传统 Web 服务的直觉：**并发不够？加 worker！**

在 vLLM 中，这个直觉可能是错的。

### 根本原因：Scheduler 需要全局视角

Continuous Batching 的威力来自于：**把所有请求的 token 放在一起调度**。

```
单 worker（推荐）:
  Scheduler 看到全部 10 个请求
  → 组成 batch_size=10 的大 batch
  → GPU 高效执行

双 worker（可能更慢）:
  Worker 1 的 Scheduler 看到 5 个请求 → batch_size=5
  Worker 2 的 Scheduler 看到 5 个请求 → batch_size=5
  → 两个小 batch，GPU 利用率下降
  → 还要各自加载一份模型权重，KV Cache 空间减半
```

### 多 worker 的三个代价

| 代价 | 说明 |
|------|------|
| **显存翻倍** | 每个 worker 独立加载模型 + 独立的 KV Block Pool |
| **batch 变小** | 请求被分散到多个 worker，每个 worker 的 batch 更小 |
| **调度碎片化** | 每个 Scheduler 只能看到局部请求，无法全局优化 |

### 什么时候多 worker 才合理

| 场景 | 推荐 |
|------|------|
| 单 GPU | **1 worker**，把 batch 做大 |
| 多 GPU | **每 GPU 1 worker**，避免竞争 |
| CPU 瓶颈（tokenization、网络 IO） | 多 worker 分摊 CPU 压力 |
| 强隔离需求（不同优先级用户） | 多 worker 隔离故障域 |

**实用法则：**

```
单 GPU 吞吐不够？
  ↓
先调大 --max-num-batched-tokens（默认可能偏保守）
  ↓
还不够？增加 GPU，不是增加 worker
```

## 本章小结

| 概念 | 一句话解释 |
|------|-----------|
| **Continuous Batching** | 每次迭代重新组 batch，请求随时进出 |
| **Prefill** | 处理 prompt，compute-bound，吃算力 |
| **Decode** | 逐 token 生成，memory-bound，吃带宽 |
| **Scheduler 核心任务** | 在 KV Block 和 GPU 计算预算内，最大化吞吐 |
| **为什么单 worker 更好** | Scheduler 需要全局视角才能组大 batch |

# 一次请求的完整生命周期（串讲）

前面我们分别讲了 KV Cache、PagedAttention、Scheduler。现在把它们串起来，看一个完整的请求是如何在 vLLM 中流转的。

先看一张**总览图**，把关键组件串在一条线上：

```text
用户 HTTP / OpenAI 请求
            │
            ▼
      HTTP / API 层
   （解析 JSON、组装 messages）
            │
            ▼
        Tokenizer
   （文本 → token IDs）
            │
            ▼
         Scheduler
   （排队、选入本轮 batch）
            │
   ┌────────┴────────┐
   │                 │
   ▼                 ▼
 Prefill 样本        Decode 样本
（第一次进模型）   （继续生成下一个 token）
   │                 │
   └────────┬────────┘
            ▼
   Transformer 前向计算
   - 读 / 写 KV Cache（PagedAttention）
   - 同一套 Attention / MLP 逻辑
            │
   ┌────────┴────────┐
   ▼                 ▼
 流式返回给用户      Block 回收
（detokenize + SSE） （请求结束释放 KV，Block 重新标记为 free）
```

> 你可以用这张图来定位：  
> - `--max-model-len` 主要约束的是“单条 seq 在 KV Cache 中能活多长”（上下文长度）；  
> - `--max-num-seqs` 和 KV Block Pool 约束的是“同时能挂多少条 seq 在 Scheduler 的活跃池子里”；  
> - Prefill / Decode 则是这条 seq 在“Transformer 前向 + KV Cache”这一格里所处的不同阶段。

## 先建立全局视角：请求的两大阶段

在深入细节之前，先记住一个关键概念：**每个请求都会经历两个截然不同的阶段**。

| 阶段 | 做什么 | 计算特性 | 用户感知 |
|------|-------|---------|---------|
| **Prefill** | 处理用户输入的 prompt | 计算密集（Compute-bound） | 等待首个 token |
| **Decode** | 逐个生成输出 token | 访存密集（Memory-bound） | 看到逐字输出 |

这个区分非常重要：
- **Prefill 决定了"首 token 延迟"**（TTFT, Time To First Token）
- **Decode 决定了"生成速度"**（tokens/sec）

用户常说的"模型响应慢"，可能是 Prefill 慢（prompt 太长），也可能是 Decode 慢（并发太高），需要区分对待。

### Prefill vs Encoder：概念澄清

很多人第一次看到 Prefill 这个词，会下意识联想到 Transformer 的 **Encoder**，以为 Prefill = Encoder 阶段。  
这是一个常见的概念混淆，需要明确区分：

| 概念层级 | Encoder / Decoder | Prefill / Decode |
|---------|------------------|-----------------|
| **所属层面** | 模型架构层面 | 推理引擎层面（vLLM） |
| **定义** | Encoder：把输入编码成表示<br>Decoder：基于表示生成输出 | Prefill：第一次把 prompt 送进模型，计算并写入 KV Cache<br>Decode：在已有 KV Cache 基础上，每次生成 1 个新 token |
| **适用模型** | Encoder-Decoder 架构（如 T5、BART） | 所有生成式模型（包括 Decoder-only 如 GPT、LLaMA、Qwen） |

**关键点：**

- 你笔记里提到的模型（Qwen、LLaMA 等）通常是 **decoder-only** 架构：
  - 没有独立的 Encoder；
  - Prefill 和 Decode 都是**同一个 decoder-only Transformer 的前向过程**，只是：
    - **Prefill**：一次性处理所有输入 token，计算它们的 KV 并写入缓存；
    - **Decode**：每次只处理 1 个新 token，用已有 KV 生成下一个。

> 一句话总结：
>
> **Prefill / Decode 是 vLLM 对“推理过程”的划分，不是模型架构的 Encoder/Decoder。**
>
> 对 decoder-only 模型来说，Prefill 和 Decode 都是 decoder 的前向，只是“一次处理多少新 token、是否已有 KV Cache”不同。

#### 为什么 Decoder-only 需要 Prefill？从 Encoder-Decoder 对比理解

如果你熟悉 Encoder-Decoder 架构（如 T5、BART），可能会疑惑：**为什么 Decoder-only 会多出一个 Prefill 阶段？**

关键在于理解两种架构的**工作流程差异**：

**Encoder-Decoder 架构的推理流程（Transformer，如 T5、BART）：**

```text
输入："Hello, how are you?"  (6 个 token)
        │
        ▼
   ┌─────────┐
   │ Encoder │  一次性处理所有输入 token
   │         │  输出：6 个 hidden states（长度 = 输入长度）
   │         │  每个 token 对应一个编码后的表示
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │ Decoder │  基于 Encoder 的输出（cross-attention），自回归生成
   │         │  每次生成 1 个 token
   └─────────┘
```

- Encoder 只跑一次，输出**序列长度 = 输入长度**的 hidden states（每个输入 token 对应一个）；
- Decoder 在生成时，通过 **cross-attention** 访问 Encoder 的所有输出，然后自回归生成。

> 注意：Transformer Encoder 的输出**不是固定长度**，而是**长度 = 输入长度**的序列。

**Decoder-only 架构的推理流程（如 GPT、LLaMA、Qwen）：**

```text
输入："Hello, how are you?"
        │
        ▼
   ┌─────────────┐
   │   Decoder   │
   │             │
   │  Step 1:    │  一次性处理所有输入 token
   │  Prefill    │  计算并缓存它们的 KV
   │             │  输出：第一个生成 token 的概率分布
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   Decoder   │
   │             │
   │  Step 2-N:  │  每次只处理 1 个新 token
   │  Decode     │  用已有 KV Cache + 新 token 的 Q
   │             │  生成下一个 token
   └─────────────┘
```

- **没有独立的 Encoder**，Decoder 既要“理解输入”，也要“生成输出”；
- **Prefill 阶段**：Decoder 第一次看到所有输入 token，一次性计算它们的 KV 并写入缓存（相当于“理解输入”的过程）；
- **Decode 阶段**：Decoder 基于 Prefill 阶段建立的 KV Cache，每次只处理 1 个新 token，自回归生成后续输出。

**Prefill 的本质：**

> Prefill 就是 **“Decoder-only 模型第一次处理输入 prompt 的过程”**。
>
> 它相当于 Encoder-Decoder 架构里“Encoder 处理输入”的那一步，
>
> 但区别是：
>
> - Encoder 输出的是“每个输入 token 的 hidden state”（序列长度 = 输入长度），Decoder 通过 cross-attention 访问这些表示；
> - Prefill 输出的是“所有输入 token 的 KV Cache”（序列长度 = 输入长度），后续 Decode 阶段直接用这些 KV 做 self-attention，不需要 cross-attention。

**时间线对比：**

| 架构 | 阶段 1 | 阶段 2 | 阶段 3 | ... |
|------|--------|--------|--------|-----|
| **Encoder-Decoder** | Encoder 处理输入 | Decoder 生成 token 1 | Decoder 生成 token 2 | ... |
| **Decoder-only** | **Prefill**（Decoder 处理输入 + 缓存 KV） | **Decode**（生成 token 1） | **Decode**（生成 token 2） | ... |

**为什么需要 Prefill？**

- Decoder-only 模型是**自回归生成**的：每一步生成时，都需要“看到所有历史 token”；
- 如果每次 Decode 都重新计算所有历史 token 的 K/V，计算量会爆炸（O(n²)）；
- Prefill 的作用就是：**一次性把所有输入 token 的 KV 算好并缓存**，后续 Decode 阶段只需要：
  - 读已有的 KV Cache（历史 token）；
  - 计算新 token 的 Q；
  - 做 Attention，生成下一个 token。

> 可以这样记忆：  
> - **Encoder-Decoder（Transformer）**：Encoder 一次性处理输入，输出长度 = 输入长度的 hidden states → Decoder 通过 cross-attention 访问这些表示，然后自回归生成；  
>
> - **Decoder-only**：Prefill 一次性处理输入（并缓存 KV，长度 = 输入长度）→ Decode 基于 KV Cache 做 self-attention，自回归生成。
>
>   Prefill 就是 Decoder-only 架构里“准备输入表示”的那一步，它准备的是可以动态增长的 KV Cache（随着生成不断追加），而不是像 Encoder 那样输出后就固定不变。

#### Transformer Decoder 的 Cross-Attention 详解：为什么它和 Decoder-only 不一样？

你可能会觉得 Transformer Decoder 和 Decoder-only 很像，因为它们都在“生成”。关键区别在于：**Decoder 有两套 Attention，而 Decoder-only 只有一套**。

**Transformer Decoder 的每一层结构：**

```text
Decoder Layer (每一层都包含)：
┌─────────────────────────────────────┐
│ 1. Self-Attention                   │  ← 看自己已经生成的部分（masked）
│    Q, K, V 都来自 Decoder 自己      │
│    （只能看到已生成的 token，不能看未来）│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ 2. Cross-Attention                  │  ← 看 Encoder 的输出
│    Q 来自 Decoder（当前要生成的 token）│
│    K, V 来自 Encoder（所有输入 token）│
│    （可以看 Encoder 的完整输出）      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ 3. Feed-Forward Network             │
└─────────────────────────────────────┘
```

**具体例子：生成翻译任务**

假设输入是 `"Hello"`（英文），要生成 `"你好"`（中文）：

**阶段 1：Encoder 处理输入**
```text
输入："Hello" (1 个 token)
        │
        ▼
   Encoder
        │
        ▼
  输出：[h_encoder]  (1 个 hidden state，长度 = 输入长度)
```

**阶段 2：Decoder 生成第一个 token "你"**
```text
Decoder 当前状态：
- 已生成：<start> (起始 token)
- 要生成：下一个 token

Decoder Layer 内部：

1. Self-Attention：
   Q, K, V 都来自 Decoder：
   - Q: <start> 的 query
   - K, V: <start> 的 key/value
   → 只能看到自己已生成的部分（<start>）

2. Cross-Attention：
   Q 来自 Decoder，K/V 来自 Encoder：
   - Q: <start> 的 query（Decoder 当前状态）
   - K, V: "Hello" 的 key/value（Encoder 的输出）
   → 可以"看"到 Encoder 处理过的所有输入 token

3. 结合 Self-Attention 和 Cross-Attention 的输出
   → 生成 "你"
```

**阶段 3：Decoder 生成第二个 token "好"**

```text
Decoder 当前状态：
- 已生成：<start> "你"
- 当前处理：刚生成的 "你" 这个 token
- 要生成：下一个 token

Decoder Layer 内部：

1. Self-Attention：
   Q, K, V 都来自 Decoder：
   - Q: "你" 的 query（当前这个刚生成的 token）
   - K, V: <start> 和 "你" 的 key/value（已生成的部分）
   → 输出：Decoder 当前状态的表示（基于已生成的部分）

2. Cross-Attention：
   Q 来自 Decoder，K/V 来自 Encoder：
   - Q: Decoder 当前状态的表示（来自 Self-Attention 的输出，即"你"经过 Self-Attention 后的表示）
   - K, V: "Hello" 的 key/value（Encoder 的输出，和之前一样）
   → 输出：结合了 Decoder 状态和 Encoder 输入的表示

3. Feed-Forward + 输出层：
   → 基于 Cross-Attention 的输出，预测下一个 token 的概率分布
   → 采样/选择 → 生成 "好"
```

**关键区别总结：**

| 方面 | Transformer Decoder | Decoder-only |
|------|-------------------|--------------|
| **Self-Attention** | ✅ 有，但只看**已生成的部分**（masked） | ✅ 有，看**所有历史**（输入 + 已生成） |
| **Cross-Attention** | ✅ 有，Q 来自 Decoder，K/V 来自 Encoder | ❌ 没有（因为没有 Encoder） |
| **输入信息的访问方式** | 通过 Cross-Attention 访问 Encoder 的输出 | 通过 Self-Attention 直接访问输入 token 的 KV |
| **输入信息是否变化** | Encoder 输出在生成过程中**固定不变** | KV Cache 会**随着生成不断追加** |

**为什么 Decoder-only 不需要 Cross-Attention？**

因为 Decoder-only 模型（如 GPT、LLaMA）在 Prefill 阶段就已经把**输入 token 的 KV 算好并缓存**了：

```text
Prefill 阶段：
输入："Hello"
        │
        ▼
   Decoder (Self-Attention)
   - 计算 "Hello" 的 Q, K, V
   - 把 K, V 写入 KV Cache
        │
        ▼
   KV Cache: [K_Hello, V_Hello]

Decode 阶段（生成 "你"）：
Decoder (Self-Attention)
   - Q: 当前要生成的 token 的 query
   - K, V: 从 KV Cache 读取（包括 "Hello" 的 K/V）
   → 直接通过 Self-Attention 就能"看到"输入
```

所以 Decoder-only 不需要 Cross-Attention，因为：
- **输入和输出都在同一个序列里**（输入 token 的 KV 已经存在 Cache 里）；
- **Self-Attention 就能同时访问输入和已生成的部分**。

而 Transformer Decoder 需要 Cross-Attention，因为：
- **输入和输出是分离的**（Encoder 处理输入，Decoder 生成输出）；
- **Decoder 的 Self-Attention 只能看已生成的部分**（masked），所以需要 Cross-Attention 来访问 Encoder 的输出。

> 一句话总结：  
> - **Transformer Decoder**：Self-Attention（看已生成）+ Cross-Attention（看 Encoder 输出）→ 两套机制分离输入和输出；  
> - **Decoder-only**：只有 Self-Attention（看所有历史，包括输入和已生成）→ 输入和输出在同一个序列里，不需要 Cross-Attention。

## vLLM Server 启动后内部发生了什么

用"组件视角"来看启动过程：

### 模型权重加载

```
GPU 显存
├─ 模型权重加载 ← 启动时完成，占显存的大头，相对固定
└─ （剩余空间待分配）
```

这一步将模型的所有参数（Wq, Wk, Wv, MLP 等）加载到 GPU。7B 模型 FP16 约占 14GB。

### KV Block Pool 初始化

```
GPU 显存
├─ 模型权重（已加载）
└─ KV Block Pool ← 在剩余显存中划出，切成固定大小的 Block
      ├─ block 0
      ├─ block 1
      └─ ... (全部标记为 free)
```

**这一步非常关键**：
- Block 数量决定了最大并发数
- 每个 Block 存储 16/32 个 token 的 KV（所有层）
- 这就是前面讲的 PagedAttention 的物理载体

### Scheduler 启动

vLLM 内部启动一个调度循环（可以理解为一个"永不停歇的 while True"）：

```python
# 伪代码：Scheduler 核心循环
while True:
    # 1. 收集所有等待的请求
    # 2. 检查 free blocks 是否足够
    # 3. 选择哪些请求可以进入本轮 batch
    # 4. 组成 batch，提交 GPU
    # 5. 等待 GPU 完成，更新状态
```

### HTTP Server 就绪

```
Uvicorn running on http://0.0.0.0:8000
```

此时 vLLM 处于"等待请求"状态，Scheduler 循环在后台运行，Block Pool 全部 free。

## 从 HTTP 请求到流式响应的完整流程

假设一个请求输入 `"Hello, how are you?"`，模型要生成 `"I am fine."`。

### 阶段一：请求接收与预处理

```
前端请求: "Hello, how are you?"
             │
             ▼
        ┌─────────┐
        │ HTTP 层 │  接收请求，解析 JSON
        └────┬────┘
             │
             ▼
        ┌───────────────────────────────────────────┐
        │             Tokenization                  │
        │                                           │
        │  输入文本: "Hello, how are you?"            │
        │       ↓                                    │
        │  Tokenizer 查表 + BPE 分词                  │
        │       ↓                                    │
        │  Token IDs: [15496, 11, 703, 527, 499, 30] │
        │                                            │
        │  这些数字是模型词表中的索引                     │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
                ┌─────────────────────────────┐
                │  Scheduler                  │
                │                             │
                │  检查：                      │
                │  - 有多少 free blocks？      │
                │  - 这个请求需要多少 blocks？   │
                │  - 是否可以加入本轮 batch？    │
                └──────┬──────────────────────┘
```

**Tokenization 细节**：

- 不同模型的 tokenizer 不同，同样的文本会产生不同的 token 数量
- 中文通常比英文产生更多 token（一个汉字可能对应 2-3 个 token）
- token 数量直接决定 KV Cache 大小和 Block 消耗

### 阶段二：Prefill（输入处理）

```
                         │
                         ▼
        ┌────────────────────────────────────┐
        │          Prefill 阶段               │
        │                                    │
        │  输入：6 个 token                    │
        │  需要：ceil(6/16) = 1 个 Block       │
        │                                     │
        │  GPU 做什么：                        │
        │  1. Embedding: token IDs → 向量      │
        │  2. 32 层 Transformer 前向           │
        │  3. 每层产生 K, V → 存入 Block        │
        │  4. 最后一层输出 → 预测第一个 token     │
        │                                      │
        │  耗时：取决于 prompt 长度               │
        │  - 100 token → ~50ms                 │
        │  - 2000 token → ~500ms               │
        └────────────────┬─────────────────────┘
                         │
                         │ 第一个输出 token 产生！
                         ▼
```

**Prefill 阶段的关键特性**：

- 一次性处理所有输入 token
- 计算量大（Q × K 矩阵是 [N, N]），GPU 利用率高
- 这就是"首 token 延迟"的来源
- Prefill 完成后，KV Cache 已经填充了输入的所有 K/V

> 换成“单条请求的时间线”来看：
>
> 对于一条 seq，**推理并不是“一次前向就完事”**，而是：
>
> **1 次 Prefill 大前向 + N 次 Decode 小前向（N = 实际生成的 token 数）**。
>
> Prefill 只做一次，用来把整段上下文写入 KV Cache；后面的每一步 Decode 都只在这个 KV 基础上往后“长 1 个 token”。

### 阶段三：Decode（逐 token 生成）

> 注意：**第一个生成 token（例如 "I"）的概率分布，已经在 Prefill 阶段算好了。  
> Decode 阶段做的，是“拿到已经采样出的 token，补上它的 KV，并用它来预测下一个 token”。**

```
        ┌─────────────────────────────────────────────────┐
        │                Decode 阶段（循环）                │
        │                                                 │
        │  ┌─────────────────────────────────────┐        │
        │  │ Step 1: 处理第一个生成的 token "I"     │        │
        │  │                                     │        │
        │  │ 0. 在 Prefill 阶段，已经算出首 token 的 │        │
        │  │    概率分布，并从中采样得到 "I"          │        │
        │  │ 1. 对新 token "I" 计算各层的 Q/K/V，   │        │
        │  │    并把它的 K/V 追加到 KV Cache        │        │
        │  │ 2. 以当前序列 [Hello, ,, how, are,     │       │
        │  │    you, ?, I] 为历史，预测下一个 token  │       │
        │  │    （例如 " am"）                      │       │
        │  │ 3. 流式返回 "I" 给前端                  │       │
        │  └──────────────────────────────────────┘        │
        │                   ↓                              │
        │  ┌─────────────────────────────────────┐         │
        │  │ Step 2: 处理第二个生成的 token " am"   │         │
        │  │                                     │         │
        │  │ KV Cache 现在包含：                   │         │
        │  │ [Hello, ,, how, are, you, ?, I]     │         │
        │  │                                     │         │
        │  │ 1. 对新 token \" am\" 计算各层 Q/K/V， │        │
        │  │    追加到 KV Cache                   │         │
        │  │ 2. 以当前序列 [Hello, ,, how, are,    │         │
        │  │    you, ?, I,  am] 为历史，预测下      │         │
        │  │    一个 token（例如 \" fine\"）        │         │
        │  │ 3. 流式返回 \" am\"                   │         │
        │  └──────────────────────────────────────┘         │
        │                   ↓                               │
        │  Step 3: 同理处理 \" fine\" → 预测下一个 → 流式返回   │
        │                   ↓                               │
        │  Step 4: 同理处理 \".\" → 预测 EOS → 流式返回        │
        │                   ↓                               │
        │  Step 5: 收到 EOS → 结束生成                        │
        └────────────────────┬──────────────────────────────┘
```

**Decode 阶段的关键特性**：
- 每次只处理 1 个 token（当前请求）
- 主要时间花在读取 KV Cache（访存密集）
- GPU 利用率相对较低（单 token 矩阵运算太小）
- 这就是为什么 vLLM 要 batch 多个请求一起 Decode

### 阶段四：流式返回与资源回收

```
                         │
                         ▼
        ┌────────────────────────────────────┐
        │           流式返回机制              │
        │                                      │
        │  每生成一个 token：                  │
        │  1. token ID → 文本（detokenize）   │
        │  2. 立即通过 HTTP 响应流发送        │
        │  3. 前端收到后立即显示              │
        │                                      │
        │  技术实现：                          │
        │  - HTTP chunked transfer            │
        │  - Server-Sent Events (SSE)         │
        │  - 不等待整个响应完成               │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │           Block 回收               │
        │                                      │
        │  请求结束时：                        │
        │  1. Block Table 中该请求的条目删除   │
        │  2. 对应的 Blocks 标记为 free        │
        │  3. 立即可被其他请求使用             │
        │                                      │
        │  注意：Block 内容不会被清零          │
        │  （只是逻辑上释放，物理上复用）      │
        └────────────────────────────────────┘
```

**流式返回的用户价值**：
- 用户不需要等待整个响应生成完毕
- 首 token 一出来就能看到
- 体验上感觉"模型在思考并逐字回答"

### 完整流程总结图

```
HTTP 请求
   │
   ├─→ Tokenization（文本 → token IDs）
   │
   ├─→ Scheduler 检查资源、分配 Block
   │
   ├─→ Prefill：处理输入，填充 KV Cache
   │      └─→ 产生第一个输出 token
   │
   ├─→ Decode（循环）：
   │      ├─→ 生成 token
   │      ├─→ 更新 KV Cache
   │      ├─→ 流式返回
   │      └─→ 重复，直到 EOS
   │
   └─→ Block 回收，请求结束
```

## 与前面章节的关联

| 阶段 | 涉及的核心机制 | 对应章节 |
|------|--------------|---------|
| KV Cache 填充/使用 | 每个 token 的 K/V 存储与复用 | KV Cache 原理 |
| Block 分配/回收 | 分页管理，避免碎片 | PagedAttention |
| Batch 组织 | 多请求 token 合并计算 | Scheduler 调度 |
| Prefill vs Decode 调度 | 不同阶段不同策略 | Scheduler 调度 |

## 常见问题解答

### 为什么首 token 慢，后续 token 快？

- **首 token = Prefill 阶段**：要处理整个 prompt，计算量与 prompt 长度成正比
- **后续 token = Decode 阶段**：每次只计算 1 个 token，利用 KV Cache

如果 prompt 有 2000 token，Prefill 可能要 500ms；但后续每个 token 只要 20-30ms。

### 流式输出是怎么实现的？

vLLM 使用 HTTP chunked transfer 或 SSE（Server-Sent Events）：

```python
# 客户端收到的流式响应示例
data: {"text": "I", "finish_reason": null}
data: {"text": " am", "finish_reason": null}
data: {"text": " fine", "finish_reason": null}
data: {"text": ".", "finish_reason": "stop"}
```

每个 `data:` 行是一个 token，前端收到后立即追加显示。

### 多个请求同时在跑，会互相影响吗？

会，但 vLLM 的设计就是为了最小化这种影响：

- **Scheduler** 会把多个请求的 Decode token 合并成一个 batch
- **PagedAttention** 保证每个请求的 KV Cache 独立，不会混淆
- **Block Pool** 是共享的，但每个请求通过 Block Table 独立管理自己的 Blocks

影响主要体现在：
- 并发多时，每个请求分到的 GPU 时间片变少 → 单请求延迟上升
- Block 紧张时，新请求可能要等待 → 排队延迟

### 请求结束后，KV Cache 的内容去哪了？

- **逻辑上**：Block 标记为 free，Block Table 条目删除
- **物理上**：显存内容不清零（太慢），直接被下一个请求覆盖
- **安全上**：不同请求的 KV 内容不会泄露（因为通过 Block Table 隔离）

# 多GPU/多机部署

当模型太大单卡放不下，或者需要更高吞吐时，就需要多 GPU 部署。本章介绍两种核心并行策略及其使用方法。

## 什么时候需要多 GPU

| 场景 | 典型情况 | 解决方案 |
|------|---------|---------|
| **单卡放不下** | 70B 模型需要 ~140GB 显存，单卡最大 80GB | 多卡分摊模型权重 |
| **需要更大 KV Cache** | 长上下文、高并发场景，KV Cache 吃掉大量显存 | 多卡增加总显存池 |
| **需要更高吞吐** | 单卡 GPU 计算已饱和 | 多卡并行计算 |

**显存需求快速估算**：

| 模型规模 | 权重显存 (FP16) | 推荐最低配置 |
|---------|----------------|-------------|
| 7B | ~14 GB | 单卡 24G |
| 14B | ~28 GB | 双卡 24G 或 单卡 48G |
| 32B | ~64 GB | 4×24G 或 2×48G |
| 70B | ~140 GB | 4×48G 或 8×24G |

## 两种并行策略：TP vs PP

vLLM 支持两种并行方式，理解它们的本质区别是正确配置多卡部署的前提。

### 张量并行 (Tensor Parallel, TP)

**核心思想**：把同一层的权重矩阵按列/行切分到多个 GPU，每个 GPU 计算一部分，然后通过 AllReduce 汇总结果。

```
TP=2 时，每一层的 Attention 计算：

输入 X
   │
   ├──────────────┬──────────────┐
   ▼              ▼              │
  GPU0           GPU1            │
  ┌────────┐    ┌────────┐       │
  │ Wq 前半 │    │ Wq 后半 │      │ 权重矩阵按列切分
  │ Wk 前半 │    │ Wk 后半 │      │ 每个 GPU 存一半
  │ Wv 前半 │    │ Wv 后半 │      │
  └────────┘    └────────┘       │
       │              │          │
       ▼              ▼          │
  计算部分 Q/K/V  计算部分 Q/K/V    │
       │              │          │
       └──── AllReduce ────┘     │  ← 每层都要同步一次
              │                  │
              ▼                  │
         完整 Attention 输出      │
              │                  │
              ▼                  │
         下一层（重复以上过程）──────┘
```

**关键特性**：

| 特性 | 说明 |
|-----|------|
| 通信频率 | **高**——每层都要 AllReduce |
| 通信数据量 | 激活值（相对较小） |
| 对带宽要求 | **高**——需要 NVLink 或高速 PCIe |

**TP 下 KV Cache 是怎么存的？**

很多人会下意识以为：“既然多卡一起算，是不是每块 GPU 上都存一份完整的 KV Cache？”  
实际上，在张量并行下，**KV Cache 也会按张量切分方式分布到各自 GPU 上**。

以最常见的“按 head 维度切”的 TP 为例（例如 `tensor_parallel_size = 2`，一层有 32 个 Attention head）：

- 权重切分：
  - GPU0：head 0–15 的 Wq/Wk/Wv
  - GPU1：head 16–31 的 Wq/Wk/Wv
- KV Cache 也随之切分：
  - 对于同一个 token：
    - GPU0：只存 head 0–15 的 K/V
    - GPU1：只存 head 16–31 的 K/V
  - 随着上下文增长，每块 GPU 只为“自己负责的那些 head × 所有层 × 所有 token”追加 K/V。

从全局看：

- **完整的一份 KV Cache** = 多块 GPU 上这些“按 head 切分的碎片（shard）”拼起来；  
- **单块 GPU 上的 KV Cache** = 只包含自己负责那部分 head 的 K/V，而不是一整份副本。

这样设计的好处是：

- 当你用 TP 扩大模型（更多 head、更大的 hidden），**KV Cache 的显存压力也会被多卡平摊开去**；  
- 不会出现“每块卡都存一整份 KV，显存直接按 TP 数量翻倍”的情况。

| 显存分摊 | 权重 + KV Cache 都分摊 |

### 流水线并行 (Pipeline Parallel, PP)

**核心思想**：把模型的不同层分配到不同 GPU/机器。Layer 0-15 在机器 A，Layer 16-31 在机器 B，数据像流水线一样依次流过。

```
PP=2 时，32 层模型的切分：

     Machine 0                    Machine 1
┌─────────────────┐          ┌─────────────────┐
│ Layer 0         │          │ Layer 16        │
│ Layer 1         │          │ Layer 17        │
│ ...             │  hidden  │ ...             │
│ Layer 14        │  states  │ Layer 30        │
│ Layer 15        │ ───────► │ Layer 31        │
│                 │   传输    │                 │
│ KV Cache 前半    │          │ KV Cache 后半   │
└─────────────────┘          └─────────────────┘

每个 token 的数据流：
输入 → Machine0 处理 L0-15 → 传给 Machine1 → Machine1 处理 L16-31 → 输出
```

**关键特性**：

| 特性 | 说明 |
|-----|------|
| 通信频率 | **低**——每个 stage 边界传输一次 |
| 通信数据量 | hidden states（几 MB 量级） |
| 对带宽要求 | **中**——可容忍普通以太网 |
| 显存分摊 | 权重按层切分，KV Cache 也按层切分 |

**PP 下 KV Cache 是怎么存的？**

可以类比 TP，只不过切分的不是 head，而是 **层（layer）**：

- 例如 32 层模型，PP=2：
  - Machine 0 / GPU0：负责 Layer 0–15
  - Machine 1 / GPU1：负责 Layer 16–31
- 对于任意一个 token：
  - GPU0：只存自己负责的 Layer 0–15 的 K/V
  - GPU1：只存自己负责的 Layer 16–31 的 K/V
- 序列变长时，每块 GPU 都只为“自己那一段层 × 所有 token”追加 K/V。

从全局看：

- **完整的一份 KV Cache** = 各个 stage（各 GPU）上“按层切分”的 KV 片段拼起来；  
- **单块 GPU 上的 KV Cache** = 只包含自己负责那几层的 K/V，而不是整个模型所有层的 KV。

这样，PP 和 TP 一样，都实现了“**参数和 KV Cache 都随并行维度分摊到多卡上**”，  
区别只是：TP 沿着 head/hidden 维切，PP 沿着 layer 维切。

### TP vs PP 对比总结

| 维度 | TP (张量并行) | PP (流水线并行) |
|------|--------------|----------------|
| 切什么 | 切权重矩阵（同一层内） | 切层（不同层） |
| 通信频率 | 每层多次 AllReduce | 每 stage 一次 |
| 通信带宽要求 | 高（需 NVLink） | 中（以太网可接受） |
| 适用场景 | **单机多卡** | **多机部署** |
| 延迟影响 | 小（同步快） | 大（流水线填充延迟） |

**选择原则**：

- **单机多卡 → 用 TP**：NVLink/PCIe 带宽足够
- **多机 → TP + PP 组合**：机内 TP，机间 PP
- **能单机的不跨机**：跨机通信是性能瓶颈

## 单机多 GPU 部署

单机多卡是最常见的多 GPU 场景，配置相对简单，性能也最好（因为卡间通信走 NVLink/PCIe，带宽高、延迟低）。

### 启动命令

```bash
# 指定使用哪些 GPU（可选，默认使用所有可见 GPU）
export CUDA_VISIBLE_DEVICES=0,1

# 启动双卡张量并行
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

**参数说明**：

| 参数 | 含义 | 注意事项 |
|------|------|---------|
| `--tensor-parallel-size` | 张量并行的 GPU 数量 | 必须能被模型 attention heads 整除 |
| `CUDA_VISIBLE_DEVICES` | 指定使用的 GPU 编号 | 数量要 ≥ tensor-parallel-size |

### 验证启动成功

**1. 观察启动日志**

```
INFO:     Loading model weights...
INFO:     Model weights loaded on 2 GPUs
INFO:     GPU blocks: 3000, CPU blocks: 512  # blocks 数量应该比单卡多
```

**2. 确认显存分布**

```bash
nvidia-smi

# 预期结果：两张卡显存占用相近
# +-----------------------------------------------------------------------------+
# | GPU  Name        ...  Memory-Usage  |
# |-----------------------------------------------------------------------------+
# |   0  NVIDIA A100  ...  18432MiB / 40960MiB |  ← 约 18G
# |   1  NVIDIA A100  ...  18432MiB / 40960MiB |  ← 约 18G（相近）
# +-----------------------------------------------------------------------------+
```

**3. 发送测试请求**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Hello"}]}'
```

### TP 约束：attention heads 必须能整除

这是 TP 的硬性限制。**原因**：TP 是把 Attention 的 Q/K/V 矩阵按 head 维度切分，每个 GPU 处理一部分 head。

```
Qwen2.5-7B 有 28 个 attention heads
→ TP 必须是 28 的因数：1, 2, 4, 7, 14, 28
→ TP=3 或 TP=8 都会报错

LLaMA 2-70B 有 64 个 attention heads
→ TP 可选：1, 2, 4, 8, 16, 32, 64
```

**如何查看模型的 attention heads 数量**：

```bash
# 查看模型配置文件
cat /path/to/model/config.json | grep num_attention_heads
```

### NVLink vs PCIe 对性能的影响

TP 每层都要做 AllReduce 同步，对卡间带宽敏感：

| 连接方式 | 带宽 | TP 性能损失 | 查看方法 |
|---------|------|-----------|---------|
| NVLink 4.0 | 900 GB/s | 几乎无损 | `nvidia-smi topo -m` |
| NVLink 3.0 | 600 GB/s | ~5% | 看输出中的 NV* |
| PCIe 4.0 x16 | 32 GB/s | 10-20% | 看输出中的 PIX/PHB |
| PCIe 3.0 x16 | 16 GB/s | 20-40% | |

```bash
# 查看 GPU 间连接拓扑
nvidia-smi topo -m

# 输出示例：
#         GPU0    GPU1
# GPU0     X      NV12   ← NV12 表示 NVLink，带宽高
# GPU1    NV12     X
#
# 如果显示 PIX 或 PHB，表示走 PCIe，性能会有损失
```

### TP 数量选择建议

| 场景 | 推荐 TP | 理由 |
|------|--------|------|
| 14B 模型 + 2×24G | TP=2 | 刚好够用，通信开销最小 |
| 70B 模型 + 8×24G | TP=8 | 必须 8 卡才放得下 |
| 70B 模型 + 4×80G | TP=4 | 虽然 2 卡也能放下，但 TP=4 留更多 KV Cache 空间 |
| 7B 模型 + 2×24G | TP=2 | 不是为了放下，而是为了更大 KV Cache 和更高吞吐 |

**原则：能 TP=2 就不用 TP=4**——TP 越大，AllReduce 开销越高。

## 多机多 GPU 部署

这一章要回答的问题是：**当一台机器、一张或几张 GPU 撑不住大模型（如 70B+）时，如何用多机多卡把同一个 vLLM 服务“拼”起来，并让它对外看起来仍然是一个统一的 OpenAI 接口？**  
换句话说，本章不是讲“怎么多开几个服务”，而是讲：

- **如何组合 TP + PP**：机内用张量并行增加单机算力，机间用流水线并行把模型切到多台机器上；
- **如何用 Ray 把多台机器组织成一个 vLLM 集群**：谁是 head、谁是 worker、节点间如何互联；
- **如何一步步启动/排错**：从 Ray 集群启动，到在 head 上起 vLLM，再到多机共享同一份模型文件。

看完这一章，你应该能做到：给一组 IP 和 GPU 数量，算出需要的 TP/PP 配置，按步骤把一个 70B 级别的模型跑在 2 台或更多机器上，并且仍然通过一个统一的 `http://host:port/v1/...` 对外提供服务。

### 部署架构：机内 TP + 机间 PP

典型配置是**机内用张量并行（TP），机间用流水线并行（PP）**：

```
Machine 0 (2×GPU)                Machine 1 (2×GPU)
┌─────────────────┐              ┌─────────────────┐
│  GPU0 ─┬─ TP=2  │              │  GPU0 ─┬─ TP=2  │
│  GPU1 ─┘        │              │  GPU1 ─┘        │
│                 │              │                 │
│  Layer 0-15     │  hidden      │  Layer 16-31    │
│  (前半层)        │  states      │  (后半层)        │
│  KV Cache 前半   │ ────────────►│  KV Cache 后半  │
└─────────────────┘   PP=2       └─────────────────┘
```

**关键公式**：总 GPU 数 = TP × PP

示例：TP=2, PP=2 → 需要 4 张 GPU（2 台机器各 2 张）

### Ray 与 vLLM：多机多卡的“调度大脑”

在多机多 GPU 场景下，**vLLM 自己并不直接管理所有机器/进程，而是把“跨机调度、进程管理、节点间通信”等底层工作交给 Ray 来做，自己专注于“模型推理和 KV Cache 管理”**。

可以这样理解两者关系：

- **Ray 是“分布式运行时 / 调度框架”**：
  - 负责在多机上拉起 worker 进程、维护集群状态；
  - 提供远程对象（actor）、任务调度、节点间 RPC 等能力；
  - 对 vLLM 来说，更像一个“分布式版 Python 运行环境 + 管理员”。

- **vLLM 是“跑在 Ray 上的推理服务”**：
  - 把自己的 Engine（推理引擎）封装成 Ray 的 actor；
  - 在 head 节点上起 OpenAI 兼容的 HTTP 服务进程，这个进程通过 Ray RPC 去调用分布在各个 worker 上的 Engine actor；
  - TP/PP 的具体并行计算、KV Cache 的分布和读取，都发生在这些 Engine actor 里。

所以是：**“vLLM 用 Ray 当底层分布式调度和通信库”，而不是“Ray 用 vLLM 做底层”**。  
本章后面的步骤，其实就是在做三件事：

1. 用 Ray 把多台机器连成一个集群（`ray start --head` / `ray start --address=...`）；  
2. 在这个 Ray 集群之上，启动 vLLM 的 Engine actor（带上 TP/PP 配置）；  
3. 在 head 节点上起一个 OpenAI API Server，把外部的 HTTP 请求转成对 Engine actor 的远程调用。

当你在客户端发起一次 `chat.completions` 请求时，链路大致是：

`客户端 → HTTP Server（vLLM）→ Ray RPC → 多机多卡上的 Engine actor → 计算结果再经 HTTP 返回给客户端`。

如果把这个流程展开成“组件 + 步骤”，可以这样在脑子里画图：

1. **客户端**（浏览器、后端服务、脚本）：  
   - 只知道一个 HTTP 地址：`http://host:port/v1/...`。
2. **vLLM HTTP Server（OpenAI API）**：  
   - 跑在某台机器上（通常是 Ray head 节点）；  
   - 接收 HTTP 请求、解析 JSON、做基础校验；  
   - 作为 Ray 的“driver”，负责把请求转成对后端 Engine actor 的远程调用。
3. **Ray 集群**：  
   - 把“有 GPU 的进程”抽象成一堆 worker 节点；  
   - 负责把来自 HTTP Server 的调用路由到正确的 Engine actor 上，并把结果再路由回来。
4. **vLLM Engine actor（多机多卡推理引擎）**：  
   - 每个 actor 绑定一组 GPU，内部用 TP/PP 把模型算子拆到多卡上；  
   - 管理这些 GPU 上的 KV Cache；  
   - 完成 Prefill + Decode，返回本次请求的生成结果。

最终从外部看，**你只是在调一个“普通的 OpenAI 接口”**；

从内部看，则是：**HTTP Server 负责“接单 + 转单”，Ray 负责“派单 + 路由”，Engine actor 负责“干活 + 管 KV Cache”。**

当单机显存不够（如 70B+ 模型需要 140GB+），就需要跨机部署。vLLM 使用 **Ray** 框架进行分布式调度。

### 前置准备检查清单

| 检查项 | 要求 | 检查命令 |
|-------|------|---------|
| Python 版本 | 所有节点一致 | `python --version` |
| CUDA 版本 | 所有节点一致 | `nvcc --version` |
| vLLM 版本 | 所有节点一致 | `pip show vllm` |
| 模型文件 | 所有节点**相同路径**可访问 | NFS 共享或每节点下载一份 |
| 网络互通 | 节点间可互相访问 | `ping <对方IP>` |
| 防火墙 | Ray 端口开放 | 6379 + 随机端口 |

### 多机启动步骤（详细版）

假设场景：
- 主节点 IP：`192.168.1.100`（2 张 GPU）
- 工作节点 IP：`192.168.1.101`（2 张 GPU）
- 模型：70B，需要 4 张卡

**Step 1：在主节点启动 Ray Head**

```bash
# 在 192.168.1.100 上执行
ray start --head --port=6379

# 成功输出：
# Local node IP: 192.168.1.100
# Ray runtime started.
# Next steps:
#   To add another node, run: ray start --address='192.168.1.100:6379'
```

**Step 2：在工作节点加入集群**

```bash
# 在 192.168.1.101 上执行
ray start --address="192.168.1.100:6379"

# 成功输出：
# Local node IP: 192.168.1.101
# This node has been added to the cluster.
```

**Step 3：验证集群状态**

```bash
# 在任意节点执行
ray status

# 预期输出：
# ======== Cluster Resources ========
# Nodes
# -----
# 192.168.1.100: GPU 2.0, CPU 16.0  ← 主节点，2 张 GPU
# 192.168.1.101: GPU 2.0, CPU 16.0  ← 工作节点，2 张 GPU
# 
# Totals
# ------
# GPU: 4.0/4.0  ← 总共 4 张 GPU，符合预期
```

**关键确认点**：

- GPU 总数是否正确（应该是 4）
- 所有节点是否都出现在列表中

**Step 4：在主节点启动 vLLM**

```bash
# 只需在主节点执行，vLLM 会自动调度 Ray 集群中的所有 GPU
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

**参数解释**：

| 参数                         | 值        | 含义                           |
| ---------------------------- | --------- | ------------------------------ |
| `--tensor-parallel-size 2`   | 2         | 每台机器内 2 张 GPU 做张量并行 |
| `--pipeline-parallel-size 2` | 2         | 2 台机器做流水线并行           |
| 总 GPU 数                    | 2 × 2 = 4 | 必须与 Ray 集群中的 GPU 数匹配 |

拓扑：

在这里，“拓扑”说的是：**`tensor_parallel_size` / `pipeline_parallel_size` 这两个参数，最后会在 Ray 集群里的哪些机器、哪些 GPU 上，按什么结构“排布”模型的各个分片。**  
你可以把它想成一张“逻辑 rank → 物理 GPU” 的映射图。

结合上面的示例命令：

```bash
python -m vllm.entrypoints.openai.api_server \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  ...
```

可以这样在脑子里画拓扑：

- **总卡数**：TP=2, PP=2 → 总共需要 2×2 = 4 张 GPU；  
- **按 PP 切 stage（流水线 stage）**：
  - Stage 0（前半层，如 Layer 0–15）：占用 2 张 GPU，用 TP=2 做张量并行；
  - Stage 1（后半层，如 Layer 16–31）：也占用 2 张 GPU，用 TP=2 做张量并行；
- **按 TP 切每个 stage 内的层**：
  - 对于 Stage 0：这 2 张卡共同承担每一层的一部分 head/hidden（前半/后半）；  
  - 对于 Stage 1：另一组 2 张卡同理。

在 Ray 视角下，大致可以理解为：

- vLLM 会为这个模型创建 2 个“流水线 stage actor”（PP=2），每个 stage actor 绑定 2 张 GPU（TP=2）；  
- Ray 再根据当前集群的 GPU 分布，把这 4 个逻辑 GPU rank 分配到实际的机器和设备 ID 上（例如：Machine0 的 GPU0、GPU1；Machine1 的 GPU0、GPU1）。

所以，当你配置 TP 和 PP 时，可以用下面这条记忆规则来想象“拓扑”：

> **先按 PP 把模型在层的维度上切成若干段（stage），每段再按 TP 在 head/hidden 维度上切到多卡。  
> 最终的拓扑 = 各个 stage × 各个 TP rank 被 Ray 映射到集群里的具体 GPU 上。**

#### 为什么这条命令会用到 Ray 集群？

很多人看到这里会有同样的问题：**“我只是跑了一条 `python -m vllm...`，为什么模型就分布到整个 Ray 集群上了？”**

关键点有两个：

- **前提：Ray 集群已经在 Step 1–3 中启动完毕，并且当前这个 Python 进程就是在 Ray runtime 环境里运行的。**
  - 主节点：`ray start --head ...`
  - 工作节点：`ray start --address="主节点IP:6379"`
  - `ray status` 能看到所有节点和 GPU
- **vLLM 的分布式执行后端默认就是 Ray（多机场景），当你设置了 `--tensor-parallel-size` / `--pipeline-parallel-size` > 1 时，它会：**
  - 在内部创建一个 Ray 分布式执行器（`RayDistributedExecutor`）
  - 使用 `ray.init(address="auto")` 连接到当前环境里的 Ray 集群
  - 把模型切分（TP/PP）和 KV Cache / worker 任务都派发到 Ray 集群中的各个节点上

所以，这条命令本身**并不会“平地起一个 Ray 集群”**，它只是**利用了你在 Step 1–3 已经准备好的 Ray 集群**：

> 一句话：**先有 Ray 集群，再在集群中的某个节点上跑 vLLM，vLLM 就会自动把整个 Ray 集群当成一个大的“设备池”来用。**

#### 如果我没有提前启动 Ray，会发生什么？

- **没有启动任何 Ray，只在一台机器上跑这条命令：**
  - vLLM 只能看到本机的 GPU/NPU，不会跨机。
  - 通常会退化为“单机多卡”的分布式执行（使用默认的单机后端，如 `multiprocessing`），**不会自动去用别的机器**。
- **有多台机器，但都没按 Step 1–3 启 Ray：**
  - 在每台机器上各自跑这条命令，只会得到“几台互不相干的单机服务”，
  - **不会**自动合并成一个跨机 4 卡的整体模型。

> 记忆规则：  
> **不启 Ray：最多单机多卡；想多机多卡：必须先有 Ray 集群。**

如果你想显式指定“一定要用 Ray 后端”，可以加上：

```bash
  --distributed-executor-backend ray
```

#### GPU 数量 vs `tensor_parallel_size` / `pipeline_parallel_size`

这两个参数决定了 vLLM 需要用多少张卡：

- **总卡数 = `tensor_parallel_size × pipeline_parallel_size`**
  - 本例中：2 × 2 = 4，表示需要 4 张 GPU/NPU。

常见情况：

- **可见卡数 < 需要的总卡数（例如只有 3 张卡，却配置 TP=2、PP=2，需要 4 张）：**
  - vLLM 一般会直接启动失败，报类似“没有足够的 GPU 资源 / placement 失败”的错误，
  - 不会自动“降级”为 3 卡。
- **可见卡数 ≥ 需要的总卡数（例如有 5 张卡，配置 TP=2、PP=2，只需要 4 张）：**
  - vLLM 只会从可见设备里**挑出 4 张卡来用**，多出来的 1 张会闲置。
  - 你可以用 `CUDA_VISIBLE_DEVICES` 来精确控制用哪几张卡，例如：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

这样可以避免“乱选卡”，也方便你做更精细的资源规划。

**Step 5：验证部署成功**

```bash
# 发送测试请求
curl http://192.168.1.100:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/Qwen2.5-72B-Instruct",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# 同时在两台机器上查看 GPU 占用
ssh 192.168.1.100 nvidia-smi
ssh 192.168.1.101 nvidia-smi
# 两台机器的所有 GPU 都应该有显存占用
```

### 流水线并行的工作流程

PP 将模型按层切分，数据像流水线一样流过各个 stage：

```
一个请求的数据流：

用户请求 "Hello, how are you?"
         │
         ▼
    Tokenization → [15496, 11, 703, 527, 499, 30]
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ Machine 0: 处理 Layer 0-15                               │
│                                                         │
│   Embedding → Layer0 → Layer1 → ... → Layer15           │
│                                                         │
│   输出：hidden states (shape: [seq_len, hidden_size])    │
│         ≈ 6 tokens × 4096 dim × 2 bytes = 48 KB         │
└─────────────────────────────────────────────────────────┘
         │
         │ 网络传输 hidden states（~48 KB）
         ▼
┌─────────────────────────────────────────────────────────┐
│ Machine 1: 处理 Layer 16-31                              │
│                                                         │
│   Layer16 → Layer17 → ... → Layer31 → LM Head           │
│                                                         │
│   输出：下一个 token 的概率分布                             │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    生成 token "I" → 返回给用户
```

**PP 的吞吐优化**：通过流水线调度让多个请求交替处理

```
时间轴 →   t1      t2      t3      t4      t5      t6

Machine0  Req1    Req2    Req3    Req1    Req2    Req3
          L0-15   L0-15   L0-15   L0-15   L0-15   L0-15
          (tok1)  (tok1)  (tok1)  (tok2)  (tok2)  (tok2)
            │       │       │
            ▼       ▼       ▼
Machine1         Req1    Req2    Req3    Req1    Req2
                L16-31  L16-31  L16-31  L16-31  L16-31
                (tok1)  (tok1)  (tok1)  (tok2)  (tok2)
                   │       │       │
                   ▼       ▼       ▼
Output                  tok1    tok1    tok2    tok2
                        Req1    Req2    Req1    Req2
```

**理解要点**：
- 当 Machine1 处理 Req1 的第一个 token 时，Machine0 已经在处理 Req2
- 多个请求像工厂流水线一样交替流过
- 这避免了 GPU 空闲等待，最大化吞吐

### 多机多卡通信相关名词速查

**软件层**：NCCL/HCCL：软件层的“交通调度员”，负责组织“大家一起发消息”（AllReduce 等）

- NCCL：NVIDIA 提供的集合通信库，专门负责多 GPU 之间的 AllReduce / AllGather / Broadcast 等“集体通信”操作的软件层。

  - 跑在 CUDA 之上；

  - 底下实际走什么链路（PCIe / NVLink / InfiniBand / 以太网）由它自己和环境变量一起决定。

**硬件层**：PCIe / NVLink / IB / Ethernet：底下真正跑电信号/光信号的“马路”，NCCL 会根据拓扑和环境变量选择走哪条路。

- **单机硬件**：PCIe / NVLink / IB / Ethernet：底下真正跑电信号/光信号的“马路”，NCCL 会根据拓扑和环境变量选择走哪条路。
  - PCIe：主板上的通用高速总线，用来连接 CPU 与 GPU / 网卡 / SSD 等，是单机内最基础的传输通道。所有 GPU、网卡最终都挂在某条 PCIe 拓扑上。
  
  - NVLink：NVIDIA 自家的 GPU–GPU 高速互联总线（只存在于同一台机器内部）。
  
    - 带宽、延迟都优于“GPU 通过 PCIe 经 CPU/芯片组再绕一圈”的路径；
  
    - NCCL 可以优先用 NVLink 做 GPU–GPU 通信。
  
- **多机硬件**

  - InfiniBand（IB）：用于机与机之间的高性能网络互联技术（硬件 + 协议栈）。

    - 专为 HPC 设计，带宽高、延迟低；

    - 多机多卡训练/推理里，NCCL 如果能走 IB，跨机 AllReduce 会明显更快。

  - 以太网（Ethernet）：最常见的通用网络硬件+协议栈（比如 1G/10G/25G/100G 网卡），
    - 也可以承载 NCCL 的跨机通信，只是带宽/延迟通常不如 IB。

它们的关系大致是：

> NCCL = 软件，跑在 NVLink / PCIe / InfiniBand / 以太网上，把这些硬件当“路”来在多 GPU 之间搬数据。

#### NCCL：多 GPU 集体通信的软件库

是什么：

- NVIDIA 提供的 C/C++ 库，全名 NVIDIA Collective Communication Library；

- 是 NVIDIA 官方的多 GPU 通信库，专门帮你在多个 GPU 之间做 AllReduce、AllGather、Broadcast 等集体通信操作。

它自己不“带网卡”，也不“带线”：

- 它只是一个软件层，会根据当前环境去选择合适的传输通道：

  - 单机内：优先用 NVLink，不行就走 PCIe；

  - 多机间：如果有 InfiniBand，尝试用 IB；没有就走 以太网（TCP）。

在 PyTorch/DeepSpeed/vLLM 里扮演的角色：

- 在你用 TP/PP、数据并行、 vLLM 多机多卡训练/推理时，各张卡之间大量的“同步中间参数/结果”都是靠 NCCL 完成的，NCCL 再去决定走哪条物理链路（走 PCIe / NVLink / 网络网卡）。

可以把它想象成：

> NCCL = 一套“搬箱子工人”的调度系统，
>
> 具体走楼梯（PCIe）、走空中走廊（NVLink）、还是搭货车（InfiniBand/以太网），
>
> 取决于楼里/楼外实际有什么路可走。

#### HCCL ：多 GPU 集体通信的软件库

HCCL 是什么？干嘛用的？

- 全称 Huawei Collective Communication Library；

- 是 华为昇腾（Ascend）NPU 对应的 NCCL 替代品；

- 用途和 NCCL 类似，只是底层对接的是华为自家的 NPU 和网络栈，而不是 NVIDIA GPU。

可以用一句“人话”来记：

> NCCL/HCCL = 多卡之间“说话”的底层库，
>
> TP/PP、跨机流水线里各 GPU/NPU 之间要交换的激活/梯度/参数分片，
>
> 本质上都是通过它们在 PCIe/NVLink/网卡（以太网或 InfiniBand）上来回搬的。

#### NVLink：GPU 之间的高速“专用通道”（硬件）

是什么：

- NVIDIA 设计的一种 GPU‑GPU 高速互联总线；

- 带宽远高于传统 PCIe，延迟也更低。

典型场景：

- 同一台机器里有多块 GPU，主板/背板上有 NVLink Bridge 或 NVSwitch；

- 当 NCCL 检测到 GPU 之间有 NVLink 时，会优先通过 NVLink 通信，而不是通过 PCIe。

地位：

- 是一种硬件互联方式，类似于 CPU 上的“本地总线”，只对同一机器内部生效；

- 不负责跨机器——出机器还是要靠网卡（以太网或 InfiniBand）。

> 简记：NVLink = 机箱内部 GPU 之间的快车道。

#### PCIe：通用主机内部扩展总线

PCIe（Peripheral Component Interconnect Express）：外设部件高速互连总线

- 一种通用高速总线标准，用来在主机 CPU 和各类扩展设备之间传输数据；

- GPU、网卡、SSD 等，都是通过 PCIe 插槽连到主板上的；

- 带宽按“x4 / x8 / x16 + Gen3/4/5”来算，比如 PCIe 4.0 x16 理论带宽约 32GB/s；

- 对单机多卡来说，GPU 与 CPU 之间、很多情况下 GPU 与 GPU 之间的通信，底层其实都是走 PCIe（除非有 NVLink 这种专门的 GPU‑GPU 互联）。

#### InfiniBand：多机之间的高性能专用网（硬件 + 协议）

是什么：

- 一套为高性能计算（HPC）设计的网络互联系统，包括：

- 专用的网卡（HCA，Host Channel Adapter）；

- 专用的交换机；

- 一整套传输协议栈。

特点：

- 非常高的带宽（例如 100 Gbit/s、200 Gbit/s、400 Gbit/s 级别）；

- 非常低的延迟，支持 RDMA（远程内存直接访问）。

在 NCCL 里的角色：

- 如果机器之间有 IB，NCCL 可以通过 ibverbs 等接口直接用 IB 做底层传输；

- 这时多机 AllReduce/AllGather 基本就是走 IB 这条“高速公路”。

> 简记：InfiniBand = 机房里机与机之间的高性能专线，
>
> 相当于“给多机多卡集群准备了一套专用的高速网”。

#### 以太网（Ethernet）：通用网络（硬件 + 协议）

是什么：

- 最普及的网络技术（1GbE / 10GbE / 25GbE / 100GbE …），

- 包括普通的网卡、交换机，以及基于 IP/TCP 的协议栈。

在多机多卡里的角色：

- 如果没有 IB，或者没有配置 IB，那么 NCCL 在多机通信时就会退回到基于 TCP 的以太网传输。

性能：

- 带宽和延迟明显弱于 NVLink/IB，但部署成本低、兼容性好。

> 简记：以太网 = 平时连内网/互联网用的那个网，只是带宽高一点。

#### 它们之间的关系，总结成一张“层次表”

| 名称       | 性质         | 作用范围           | 在多机多卡里的角色                                       |
| :--------- | :----------- | :----------------- | :------------------------------------------------------- |
| NCCL       | 软件库       | 同机 + 跨机        | 负责调用下层传输通道，在 GPU 之间做 AllReduce 等集体通信 |
| NVLink     | 硬件互联总线 | 同一台机器内的 GPU | 为单机多卡提供高带宽/低延迟的 GPU‑GPU 直连通道           |
| PCIe       | 硬件总线     | 主机内部 CPU ↔ 设备 | 通用扩展总线，GPU/网卡/SSD 等都挂在 PCIe 上，单机多卡时很多流量实际走 PCIe |
| InfiniBand | 硬件+协议    | 机与机之间         | 为多机之间提供高带宽/低延迟的网络，NCCL 可以跑在其上     |
| 以太网     | 硬件+协议    | 机与机之间         | 通用网络，NCCL 在无 IB 时会退回到基于 TCP 的以太网传输   |

可以把整个栈想成这样：

- 最上层：PyTorch / vLLM / DeepSpeed 等框架调用 NCCL 做多卡同步；

- 中间层（软件）：NCCL 根据环境选择用什么“路”来传数据；

- 底层（硬件路）：

  - 单机：NVLink / PCIe；

  - 多机：InfiniBand 或 以太网。

### 网络配置（关键）

多机部署最容易出问题的就是网络配置。

在单机多卡场景下，你很少“感受到”通信层的存在——NCCL/HCCL 会在本机 PCIe / NVLink 上自动帮你搞定 GPU 之间的 AllReduce。

但一旦上了多机多卡，**这些通信库就需要通过“网卡 + IP”把不同机器上的 GPU/NPU 串起来**，如果网络接口没选好、环境变量没配对，很容易出问题。

于是就会出现几类典型坑：

- 多机训练/推理时卡死在 `Waiting for NCCL connection`；
- 有 InfiniBand 但 NCCL 没用上，始终在走慢速以太网，带宽打不满；
- 没有 InfiniBand，却让 NCCL 误以为有，结果一直在尝试用不存在的 IB 设备报错。

所以，这一小节的目的就是回答三个问题：

**（1）NCCL/HCCL 在多机场景里到底在干什么？**  

- 它们是 GPU/NPU 之间做 AllReduce / AllGather 等“集合通信”的底层库；  
- 在 TP/PP、跨机流水线里，大量的中间激活/梯度/参数分片，都是通过 NCCL/HCCL 在各卡之间同步的。

**（2）为什么要显式指定“用哪块网卡”？**  

- 一台机器上可能有多个网络接口（`lo`、`eth0`、`ens3`、InfiniBand 网卡等）；  
- NCCL/HCCL 需要知道“哪一块网卡用来做节点间通信”，并且**所有节点要保持一致**，否则就像有的机器在房间 A 说话、有的在房间 B 听，永远连不上。

**（3）为什么要区分有/没有 InfiniBand？**  

- 有 IB：希望 NCCL 优先走 IB，高带宽、低延迟；  
- 没 IB：要明确告诉 NCCL“不要去找 IB”，否则它会反复尝试一个不存在的设备，导致启动慢甚至报错。

下面的环境变量示例，就是在回答这三个问题：**告诉 NCCL/HCCL：多机通信应该走哪张网卡、是否启用 InfiniBand，用哪几块设备 ID 参与通信。**

**NCCL 环境变量**（适用于 NVIDIA GPU）：

```bash
# 指定 NCCL 使用的网络接口（必须所有节点一致）
export NCCL_SOCKET_IFNAME=eth0  # 替换为你的网卡名
# 告诉 NCCL：多机通信走哪块网卡（以太网/IB 对应的那个网口）

# 查看网卡名
ip addr | grep -E "^[0-9]+:" | awk -F: '{print $2}'
# 输出如：lo, eth0, ens3 等，选择用于节点间通信的那个

# 告诉 NCCL：有 IB 就用 / 没有 IB 就别去找：
# 有 IB：0，让 NCCL 走 InfiniBand 这条快路；
# 没 IB：1，避免 NCCL 一直尝试不存在的 IB 设备。
# （1）如果有 InfiniBand（推荐，性能更好）
export NCCL_IB_DISABLE=0
# （2）如果没有 InfiniBand（普通以太网）
export NCCL_IB_DISABLE=1  # 避免 NCCL 尝试使用 IB 而报错
```

有了上面的整体图景，你可以这样记住一句话：

> NCCL 是搬运工，NVLink/InfiniBand/以太网是路，
>
> 配环境变量就是在告诉它：哪条路可走、哪条路禁止、默认走哪条。

**HCCL 环境变量**（适用于华为 NPU）：

华为 NPU 使用 **HCCL（Huawei Collective Communication Library）** 而不是 NCCL。HCCL 是华为自研的集合通信库，专门用于昇腾（Ascend）NPU 之间的多机多卡通信。

```bash
# HCCL 相关环境变量（华为 NPU 专用）
export HCCL_WHITELIST_DEVICE=0,1  # 指定使用的 NPU 设备ID
export HCCL_IF_IP=192.168.1.100    # 指定本机IP地址（用于多机通信）

# 查看 NPU 设备信息（华为 NPU）
npu-smi info  # 类似 nvidia-smi，但用于华为 NPU

# 华为 NPU 多机通信配置示例
# 在每台机器上设置：
export HCCL_IF_IP=<本机IP>
export RANK_TABLE_FILE=/path/to/hccl_config.json  # HCCL 配置文件路径
```

**NCCL vs HCCL 对比**：

| 特性 | NCCL | HCCL |
|------|------|------|
| 适用硬件 | NVIDIA GPU | 华为昇腾 NPU |
| 通信方式 | GPU 间直接通信 | NPU 间通过 HCCL 通信 |
| 配置方式 | 环境变量 | 环境变量 + 配置文件 |
| 查看设备 | `nvidia-smi` | `npu-smi info` |
| 多机通信 | 支持（通过 NCCL） | 支持（通过 HCCL） |

**防火墙配置**：

```bash
# Ray 需要的端口
# - 6379: Ray head 端口
# - 10000-20000: Ray 内部通信（范围可能更大）

# 临时关闭防火墙（测试用）
sudo systemctl stop firewalld  # CentOS
sudo ufw disable               # Ubuntu

# 或只开放必要端口
sudo firewall-cmd --add-port=6379/tcp --permanent
sudo firewall-cmd --add-port=10000-20000/tcp --permanent
sudo firewall-cmd --reload
```

### 常见问题排查

| 问题现象 | 可能原因 | 排查方法 |
|---------|---------|---------|
| `NCCL timeout` | 网络不通 / 网卡配置错误 | 检查 `NCCL_SOCKET_IFNAME`，用 `ping` 测试 |
| `Ray cluster not found` | Ray 未正确启动 | `ray status` 检查，确认所有节点都加入 |
| 只有主节点 GPU 在用 | PP/TP 配置错误 | 检查 `--tensor-parallel-size × --pipeline-parallel-size = GPU 总数` |
| 启动后立即 OOM | 显存不够 | 降低 `--max-model-len` 或增加 TP/PP |
| `Connection refused` | 端口未开放 | 检查防火墙，开放 Ray 端口 |

**调试建议**：

```bash
# 1. 先确认 Ray 集群正常
ray status

# 2. 查看 Ray 日志
ls /tmp/ray/session_latest/logs/
tail -f /tmp/ray/session_latest/logs/raylet.out

# 3. 手动测试节点间 NCCL 通信（高级）
# 可以用 pytorch 的 dist 测试
```

### 停止多机部署

```bash
# 先停止 vLLM 进程
pkill -f "vllm.entrypoints"

# 在每个节点上停止 Ray
ray stop

# 确认 Ray 已停止
ray status  # 应该报错 "Ray cluster not found"
```

### 机器 GPU 数量不一致怎么办

**vLLM 的约束**：每个 PP stage 内的 GPU 数量必须相同。

| 实际情况 | 处理方法 |
|---------|---------|
| 机器A: 4卡, 机器B: 4卡 | 理想情况，TP=4, PP=2 |
| 机器A: 4卡, 机器B: 2卡 | 限制机器A只用2卡：`CUDA_VISIBLE_DEVICES=0,1` |
| 机器A: 2卡, 机器B: 1卡 | 不推荐。要么只用机器A，要么升级机器B |

```bash
# 限制 GPU 使用数量
# 在机器A上（4卡，只用前2张）
export CUDA_VISIBLE_DEVICES=0,1
ray start --address="192.168.1.100:6379"
```

**原则：宁可少用几张卡，也要保证 TP 配置对称。**

## 华为 NPU（昇腾）多机部署特殊说明

vllm-ascend多机分布式推理相关的手册：

* [https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/DeepSeek-R1.html](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/DeepSeek-R1.html)
* [https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/ray.html](https://docs.vllm.com.cn/projects/ascend/en/latest/tutorials/ray.html)

### 硬件环境理解

**Atlas 300I Duo 卡的特点**：
- 一个 PCIe 插槽，集成了**两个 NPU 卡**（都是 310p 型号）
- 一个主 NPU，一个从 NPU
- 单卡推理速度：4 token/秒
- 双卡推理速度：12 token/秒（说明双卡有加速效果）

**测试场景转换**：

- 原计划：用 Atlas 300I Duo（单机双卡）测试多机多卡
- 问题：只有一张 Atlas 300I Duo 卡，无法测试多机方案
- 解决方案：使用两个独立的 310p 子卡，分别插在两台机器上（双机各单卡）

### NCCL vs HCCL 的区别

| 概念 | NCCL | HCCL |
|------|------|------|
| **全称** | NVIDIA Collective Communication Library | Huawei Collective Communication Library |
| **适用硬件** | NVIDIA GPU | 华为昇腾 NPU（如 Atlas 300I、310p） |
| **使用范围** | 广泛使用（NVIDIA 生态） | 仅华为 NPU 使用 |
| **作用** | GPU 间多机多卡通信 | NPU 间多机多卡通信 |
| **配置方式** | 环境变量（如 `NCCL_SOCKET_IFNAME`） | 环境变量 + HCCL 配置文件 |

**关键理解**：
- **NCCL** 是 NVIDIA 的通信库，用于 GPU 之间的数据同步
- **HCCL** 是华为的通信库，用于 NPU 之间的数据同步
- 两者功能类似，但**不能混用**：NPU 必须用 HCCL，GPU 必须用 NCCL

### vLLM 容器"适配华为 NPU"的含义

当说"vLLM 容器已经适配好了华为的 NPU"时，通常指：

1. **NPU 驱动已安装**：容器内包含了华为昇腾 NPU 的驱动和运行时库
2. **PyTorch 支持 NPU**：使用了支持 NPU 的 PyTorch 版本（如 `torch_npu`）
3. **vLLM 支持 NPU backend**：vLLM 可以识别并使用 NPU 作为计算设备
4. **HCCL 库已配置**：多机通信所需的 HCCL 库已安装

**验证方法**：
```bash
# 在容器内检查 NPU 是否可用
npu-smi info  # 应该能看到 NPU 设备信息

# 检查 PyTorch 是否支持 NPU
python -c "import torch; print(torch.npu.is_available())"  # 应该输出 True

# 检查 HCCL 是否可用
python -c "from torch_npu.contrib import transfer_to_npu; print('HCCL available')"
```

### Ray 集群配置（华为 NPU 环境）

**两个 vLLM 容器的角色**：
- **容器1（主节点）**：配置为 `header + worker`
  - `header`：Ray 集群的主节点，负责调度
  - `worker`：同时作为工作节点，运行 vLLM 推理任务
- **容器2（工作节点）**：配置为 `worker`
  - 只作为工作节点，加入 Ray 集群

**Ray 集群启动步骤（华为 NPU 环境）**：

```bash
# ===== 在容器1（主节点）上执行 =====
# 启动 Ray Head（主节点）
ray start --head --port=6379 --node-ip-address=<容器1的IP>

# 输出会显示：
# Local node IP: <容器1的IP>
# Ray runtime started.
# To add another node, run: ray start --address='<容器1的IP>:6379'

# ===== 在容器2（工作节点）上执行 =====
# 加入 Ray 集群
ray start --address="<容器1的IP>:6379" --node-ip-address=<容器2的IP>

# 成功输出：
# Local node IP: <容器2的IP>
# This node has been added to the cluster.

# ===== 验证集群状态 =====
# 在任意容器内执行
ray status

# 预期输出应该显示两个节点，每个节点有 NPU 资源
# ======== Cluster Resources ========
# Nodes
# -----
# <容器1的IP>: NPU 1.0, CPU X.0
# <容器2的IP>: NPU 1.0, CPU X.0
# 
# Totals
# ------
# NPU: 2.0/2.0  ← 总共 2 个 NPU，符合预期
```

**注意事项**：
- 两个容器必须能互相访问（网络互通）
- Ray 端口（默认 6379）必须开放
- 容器内的 NPU 设备必须能被 Ray 识别

### 测试计划理解

根据这段话，测试计划分为三个阶段：

#### 阶段1：单卡 310p 用 vLLM 单卡推理
```bash
# 单卡推理，不需要并行参数
python -m vllm.entrypoints.openai.api_server \
  --model <模型路径> \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85
```

**目标**：验证单卡推理功能正常，基准性能（4 token/秒）

#### 阶段2：单机两卡
```bash
# 单机双卡张量并行
python -m vllm.entrypoints.openai.api_server \
  --model <模型路径> \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85
```

**目标**：验证单机内双卡通信（通过 HCCL），性能提升（12 token/秒）

#### 阶段3：双机单卡
这是最复杂的阶段，需要解决两个问题：

**3.1 两个 NPU 通信（HCCL）**

```bash
# 在每台机器上配置 HCCL
export HCCL_IF_IP=<本机IP>
export RANK_TABLE_FILE=/path/to/hccl_config.json

# HCCL 配置文件示例（hccl_config.json）
# {
#   "server_count": "2",
#   "server_list": [
#     {"server_id": "192.168.1.100", "device": [{"device_id": "0", "device_ip": "192.168.1.100"}]},
#     {"server_id": "192.168.1.101", "device": [{"device_id": "0", "device_ip": "192.168.1.101"}]}
#   ],
#   "status": "completed"
# }
```

**3.2 构建 Ray 集群**

按照上面的 Ray 集群启动步骤，在两个容器间建立 Ray 集群。

**3.3 在 Ray 集群里启动 vLLM 推理**

```bash
# 在主节点容器内执行
python -m vllm.entrypoints.openai.api_server \
  --model <模型路径> \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.85
```

**参数说明**：
- `--tensor-parallel-size 1`：每台机器内只有 1 张卡，不需要 TP
- `--pipeline-parallel-size 2`：2 台机器做流水线并行
- 总 NPU 数 = 1 × 2 = 2，符合双机各单卡的配置

### 并行策略说明

| 并行策略 | 含义 | 适用场景 | 需要的卡数 |
|---------|------|---------|-----------|
| **张量并行（TP）** | 将模型按层或参数切分到多张卡 | 单机多卡，模型太大放不下单卡 | 2+ 张卡（同一台机器） |
| **数据并行（DP）** | 不同卡处理不同的数据批次 | 提高吞吐量，处理更多并发请求 | 2+ 张卡 |
| **混合并行** | TP + PP 或 TP + DP 组合 | 大规模模型，需要多机多卡 | 4+ 张卡（跨机器） |

**这段话中提到的"混合并行，需要四张卡"**：

- 可能是指：TP=2（每机2卡）+ PP=2（2台机器）= 总共 4 张卡
- 或者：TP=2 + DP=2 = 总共 4 张卡

### 关键检查点总结

在实施双机单卡测试前，需要确认：

| 检查项 | 验证方法 | 预期结果 |
|-------|---------|---------|
| NPU 驱动 | `npu-smi info` | 能看到 NPU 设备 |
| PyTorch 支持 NPU | `torch.npu.is_available()` | 返回 `True` |
| 网络互通 | `ping <对方IP>` | 能 ping 通 |
| Ray 端口 | `telnet <对方IP> 6379` | 能连接 |
| HCCL 配置 | 检查 `RANK_TABLE_FILE` | 配置文件存在且正确 |
| Ray 集群 | `ray status` | 显示 2 个节点，2 个 NPU |

## 配置速查表

### 按模型规模选配置

| 模型 | 显存需求 | 推荐配置 | TP | PP | 启动命令关键参数 |
|------|---------|---------|----|----|----------------|
| 7B | ~14 GB | 1×24G | 1 | 1 | 默认，无需额外参数 |
| 14B | ~28 GB | 2×24G | 2 | 1 | `--tensor-parallel-size 2` |
| 32B | ~64 GB | 4×24G | 4 | 1 | `--tensor-parallel-size 4` |
| 70B | ~140 GB | 8×24G 或 4×48G | 8/4 | 1 | `--tensor-parallel-size 8` |
| 70B | ~140 GB | 2机×2卡 | 2 | 2 | `--tensor-parallel-size 2 --pipeline-parallel-size 2` |
| 405B | ~800 GB | 4机×4卡 | 4 | 4 | `--tensor-parallel-size 4 --pipeline-parallel-size 4` |

### 配置决策流程

```
模型放得下单卡吗？
       │
      是 → 单卡部署，不需要并行
       │
      否
       ▼
单机多卡放得下吗？
       │
      是 → 只用 TP，设置 --tensor-parallel-size = 所需最少卡数
       │
      否
       ▼
需要多机部署
       │
       ▼
机内 TP = 每台机器的 GPU 数
机间 PP = 机器台数
```

### 核心原则

| 原则 | 说明 |
|-----|------|
| **单机优先** | 跨机通信是性能瓶颈，能单机搞定就不跨机 |
| **TP 尽量小** | TP=2 能放下就不用 TP=4，减少 AllReduce 开销 |
| **PP 用于跨机** | 跨机时用 PP，通信频率比 TP 低很多 |
| **对称配置** | 每个 PP stage 的 GPU 数必须相同 |
| **预留显存** | `--gpu-memory-utilization` 不要设太高，留给 KV Cache |

# 常见问题排查

## OOM 相关问题

### 启动 OOM

**现象**：vLLM 启动时报 CUDA out of memory

**常见原因与解决方案**：

| 原因 | 解决方案 |
|------|---------|
| 模型太大，显存不够 | 换更小的模型（如 7B → 3B） |
| `--gpu-memory-utilization` 太高 | 降低到 0.8 或 0.7 |
| `--max-model-len` 太大 | 降低到 4096 或 2048 |
| 有其他进程占用显存 | `nvidia-smi` 检查并 kill |

**推荐调参顺序**：

```bash
# 第一步：降低显存利用率
--gpu-memory-utilization 0.7

# 第二步：降低最大上下文长度
--max-model-len 4096

# 第三步：如果还不行，换更小的模型
```

### 首请求 OOM

**现象**：启动成功，但第一个请求直接 OOM

**原因**：启动时只加载了模型权重，KV Cache 还没有真正分配。第一个请求触发 KV Block 分配时才发现显存不够。

**解决方案**：

```bash
# 降低显存利用率，给 KV Cache 留更多空间
--gpu-memory-utilization 0.8

# 降低最大上下文长度
--max-model-len 4096
```

### 运行中 OOM

**现象**：运行一段时间后 OOM

**常见原因**：

| 原因 | 说明 |
|------|------|
| 并发请求太多 | 超过 KV Block Pool 容量 |
| 单个请求上下文太长 | 接近 `--max-model-len` 上限 |
| 内存泄漏（少见） | vLLM 版本问题 |

**解决方案**：

- 限制最大并发：`--max-num-seqs 64`
- 降低单请求上下文：`--max-model-len 4096`
- 升级 vLLM 版本

## 启动失败问题

### 模型下载问题

**现象**：启动卡在下载，或报 `Connection to huggingface.co timed out`

**原因**：国内网络无法直连 HuggingFace

**解决方案**：

```bash
# 方案 1：临时设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案 2：写入 bashrc（一劳永逸）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

设置后重新启动 vLLM，无需修改其他参数。

### 环境依赖问题

**现象**：import 报错、版本不兼容

**常见问题与解决**：

| 问题 | 解决方案 |
|------|---------|
| CUDA 版本不匹配 | 确保 PyTorch CUDA 版本与驱动匹配 |
| vLLM 版本太旧 | `pip install -U vllm` |
| transformers 版本冲突 | `pip install -U transformers` |

**验证环境**：

```bash
# 检查 CUDA
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查 vLLM 版本
python -c "import vllm; print(vllm.__version__)"
```

## 性能问题定位

### 延迟高

**现象**：单个请求响应慢

**排查步骤**：

| 检查项 | 正常值 | 异常处理 |
|-------|-------|---------|
| 首 token 延迟（TTFT） | < 1s（7B 模型） | 检查是否在 Prefill 阶段卡住 |
| GPU 利用率 | > 80% | 如果很低，可能 batch 太小 |
| KV Cache 使用率 | < 90% | 如果满了，请求会排队 |

**常见原因**：

- Prefill 阶段输入太长 → 考虑截断输入
- 并发太高，请求排队 → 增加 GPU 或降低并发
- 模型太大 → 换更小的模型或增加 GPU

### 吞吐低

**现象**：请求处理速度慢，QPS 低

**排查步骤**：

```bash
# 1. 检查 GPU 利用率
nvidia-smi -l 1

# 2. 检查日志中的 batch size
tail -f vllm.log | grep batch
```

**优化方向**：

| 问题 | 解决方案 |
|------|---------|
| batch 太小 | 增加 `--max-num-batched-tokens` |
| KV Cache 不够 | 增加 `--gpu-memory-utilization` |
| 单 GPU 瓶颈 | 增加 GPU，使用 tensor parallel |
| CPU 瓶颈（Python GIL） | 在多 GPU 场景下考虑多 worker |

# 附录

## 启动命令速查表

### 单模型启动模板

```bash
# 基础版（测试用）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096

# 生产版（后台运行）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  > vllm.log 2>&1 &
```

### 双模型启动模板

```bash
# Instruct 模型（端口 8000，显存 65%）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 4096 \
  > instruct.log 2>&1 &

# Embedding 模型（端口 8001，显存 25%）
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/qwen3-embedding-0.6b \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 4096 \
  > embedding.log 2>&1 &
```

## curl 测试示例速查

### Chat Completions（对话）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "你好，简单介绍一下你自己"}
    ]
  }'
```

### Embeddings（向量化）

```bash
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/qwen3-embedding-0.6b",
    "input": "这是一段需要向量化的文本"
  }'
```

### 流式输出

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

## 环境变量清单

| 变量 | 作用 | 示例值 |
|------|------|--------|
| `HF_ENDPOINT` | HuggingFace 镜像地址 | `https://hf-mirror.com` |
| `HF_HOME` | HuggingFace 缓存目录 | `/data/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers 缓存目录 | `/data/huggingface` |
| `CUDA_VISIBLE_DEVICES` | 指定使用的 GPU | `0,1` |

**一键配置（写入 bashrc）**：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface' >> ~/.bashrc
source ~/.bashrc
```

## 术语表

| 术语 | 含义 |
|------|------|
| **KV Cache** | Key-Value Cache，存储历史 token 的 K/V 向量，避免重复计算 |
| **PagedAttention** | vLLM 的核心技术，将 KV Cache 分页管理，解决显存碎片问题 |
| **KV Block** | KV Cache 的最小分配单位，通常存储 16/32 个 token 的 KV |
| **Block Pool** | 所有 KV Block 的池子，由 Scheduler 统一管理 |
| **Block Table** | 记录每个请求使用了哪些 Block 的映射表 |
| **Prefill** | 输入阶段，一次性处理所有 prompt token |
| **Decode** | 生成阶段，每次生成一个 token |
| **Scheduler** | 调度器，决定每一步哪些 token 进入 GPU 计算 |
| **Tensor Parallel** | 张量并行，将模型权重切分到多个 GPU |
| **Pipeline Parallel** | 流水线并行，将模型层切分到多个 GPU/机器 |
| **TTFT** | Time To First Token，首 token 延迟 |
| **GQA/MQA** | Grouped/Multi-Query Attention，减少 KV Cache 大小的技术 |



# 参考资料

---

[图解大模型计算加速系列之：vLLM核心技术PagedAttention原理](https://zhuanlan.zhihu.com/p/691038809)

[破解LLM性能瓶颈：你必须了解的两项注意力优化技术](http://cloud.tencent.com/developer/article/2569255)

[vLLM中文站](https://vllm.hyper.ai/docs/)

[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
