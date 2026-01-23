# Ollama本地大模型部署与调用指南

* [返回上层目录](../model-deployment-practice.md)
* [使用前准备工作](#使用前准备工作)
* [Ollama的交互模式](#Ollama的交互模式)
* [Ollama在linux下的服务配置](#Ollama在linux下的服务配置)
* [Ollama推理服务机制与工程边界](#Ollama推理服务机制与工程边界)
* [Ollama本地调用服务和API远程调用服务的本质区别](#Ollama本地调用服务和API远程调用服务的本质区别)
* [OpenAI-API已成为大模型服务接口标准](#OpenAI-API已成为大模型服务接口标准)

Ollama 是一个本地大模型推理平台，支持在个人电脑或私有服务器上运行大规模语言模型（LLM），无需依赖远程云服务。它不仅提供模型管理、下载和运行功能，还兼容 OpenAI 风格的 API 调用，使开发者能够轻松在本地环境中实现文本生成、问答、指令执行和向量嵌入等功能。

相比在线 API：

- 避免了联网调用和数据传输；
- 适合对隐私敏感或需要低延迟的场景。

本指南旨在帮助开发者快速上手 Ollama，包括模型的安装、调用方法以及流式交互实践。文档涵盖从基础部署到 Python 接口调用的完整流程，并演示如何利用流式输出进行实时文本生成。无论你是希望在本地实验大模型，还是打算将其集成到项目中，本指南都提供了实用的操作示例和工程化建议。

#  使用前准备工作

## 官网下载安装Ollama

打开Ollama下载页面：[Download Ollama](https://ollama.com/download/linux)

![download-ollama](pic/download-ollama.png)

用指令下载+安装：

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

下载中

![download-ollama-2](pic/download-ollama-2.png)

 安装完成：

![download-ollama-3](pic/download-ollama-3.png)

**执行流程是：**

1. 使用 `curl` 从 `https://ollama.com/install.sh` **下载一个安装脚本**
2. 将下载到的脚本内容 **通过管道 `|` 直接交给 `sh` 执行**
3. 脚本中包含：
   - 系统检测（Linux / macOS / 架构）
   - 下载 Ollama 可执行文件
   - 安装到系统路径（通常是 `/usr/local/bin` 或类似位置）
   - 可能创建 systemd 服务（Linux）

👉 所以结论是：**下载 + 安装 + 配置**，一步完成。

等价于：

```shell
curl -O https://ollama.com/install.sh
chmod +x install.sh
./install.sh
```

只是被压缩成了一行。

## 安装Ollama完成检查

1️⃣ `ollama` 已正确安装

验证方式：

```shell
ollama --version
# ollama version is 0.14.3
```

如果能正常输出版本号，说明：

- 二进制已安装
- PATH 配置正确

2️⃣ Ollama 服务已运行

大多数 Linux / macOS 安装脚本会**自动启动后台服务**。

验证方式：

```shell
ollama ps
# NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL
```

能返回结果（哪怕是空） → 服务正常

报错（如 connection refused） → 服务未启动

如果未启动，手动启动方式（Linux）：

```shell
ollama serve
```

## 使用Ollama下载模型

在终端窗口输入命令，下载需要的模型。

例如本项目默认使用的大语言模型是`qwen3:qwen3:4b-instruct-2507-q4_K_M`，嵌入模型是`qwen3-embedding:0.6b-q8_0`，可通过以下命令下载：

```shell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-embedding:0.6b-q8_0
```

注：MacOS系统M芯片+16G以上内存，或Windows系统30/40系列N卡+12G以上显存，建议使用7b以上模型。24G以上显存可使用更大的量化模型。

下载`qwen3:4b-instruct-2507-q4_K_M`中：

![ollama-pull](pic/ollama-pull.png)

下载`qwen3-embedding:0.6b-q8_0`中：

![ollama-pull-2](pic/ollama-pull-2.png)

两个模型全部下载完成：

![ollama-pull-3](pic/ollama-pull-3.png)

`ollama pull` 在做什么（不是简单下载）

它会：

1. 从 Ollama 官方 registry 拉取模型 manifest

2. 下载对应的 **GGUF 分片文件**

3. 校验哈希

4. 存储到本地模型缓存目录，通常是：

   ```
   ~/.ollama/models/
   ```

5. 注册到本地 Ollama 模型索引

完成后即可 **本地离线使用**

## 模型介绍

你拉的这两个模型是否合理？

✅ `qwen3:4b-instruct-2507-q4_K_M`

- 4B 参数
- instruct 微调
- `q4_K_M`：中等质量 / 显存占用较低
- 非常适合：
  - 本地推理
  - CPU 或中端 GPU
  - 指令对话 / agent

✅ `qwen3-embedding:0.6b-q8_0`

- embedding 专用模型
- 高精度 `q8_0`
- 典型用途：
  - RAG
  - 向量检索
  - 语义相似度

这是一个**标准 LLM + embedding 的正确组合**

## 模型命名解释

### instruct模型名称解释

以`qwen3:4b-instruct-2507-q4_K_M`为例：

---

1. `instruct` 是什么含义？

核心结论

> **`instruct` 表示这是一个“指令微调（Instruction-tuned）模型”**

也可以理解为：**已经被训练成“听人话、会对话”的版本**

对比说明

| 类型            | 特点                      | 适合场景                  |
| --------------- | ------------------------- | ------------------------- |
| `base` / 无后缀 | 纯预训练模型              | 研究、继续微调            |
| **`instruct`**  | **指令对齐 + RLHF / DPO** | **对话、Agent、工具调用** |

如果没有 `instruct`，模型的行为通常是：

- 更像语言建模器
- 不一定遵守指令
- 不一定给“有用的回答”

---

2. `q4` 是什么？

核心结论

> **`q4` 表示 4-bit 量化（4-bit quantization）**

量化的目的只有一个：

> **用更小的显存 / 内存，换取可接受的精度损失**

常见量化等级对照

| 标记     | 含义      | 资源占用 | 精度               |
| -------- | --------- | -------- | ------------------ |
| `q8`     | 8-bit     | 高       | 接近原模型         |
| `q6`     | 6-bit     | 中       | 很好               |
| **`q4`** | **4-bit** | **低**   | **可接受（主流）** |
| `q2`     | 2-bit     | 极低     | 明显退化           |

4B 模型用 `q4`，是**本地推理的黄金平衡点**。

---

3. `K` 是什么意思？

这是 GGUF 量化体系里的关键点。

`K` 的含义

> **`K` = K-quant（块量化 / kernel-aware quantization）**

解释成工程语言：

- 参数不是逐元素量化
- 而是**按 block 分组**
- 每个 block 有独立的 scale / zero-point
- **显著降低精度损失**

也就是说：

```
q4_K  >  q4_0 / q4_1
```

在**相同 4-bit**下，`K` 的质量明显更好。

---

4. `M` 是什么意思？

核心结论

> **`M` = Medium（中等质量 / 中等显存）变体**

在 GGUF 中，常见组合有：

| 标记         | 含义       | 特点                |
| ------------ | ---------- | ------------------- |
| `q4_K_S`     | Small      | 更省内存，略降精度  |
| **`q4_K_M`** | **Medium** | **质量 / 性能平衡** |
| `q4_K_L`     | Large      | 更高质量，更吃资源  |

你用的：

```
q4_K_M
```

在社区里通常被认为是：

> **“默认推荐版本”**

5. 把整个名字翻译成一句“人话”

```
qwen3:4b-instruct-2507-q4_K_M
```

可以翻译为：

> Qwen3 系列
>  4B 参数规模
>  经过指令微调（适合对话）
>  2025 年 07 月版本
>  使用 4-bit K-quant 量化
>  Medium 配置，质量与资源占用平衡

6. 给你一个“选型速查表”（很实用）

| 你要做什么            | 推荐                  |
| --------------------- | --------------------- |
| 对话 / Agent / 工具   | `instruct`            |
| 本地 CPU / 小显存 GPU | `q4_K_M`              |
| RAG embedding         | 单独的 embedding 模型 |
| 研究 / 微调           | base + fp16 / q8      |

### embedding模型名称解释

> 那对应的，qwen3-embedding:0.6b-q8_0名字的含义是啥？加embedding代表啥？这个模型的输出是embedding吗？q8表示啥，q8_0最后的0表示啥？

下面我**严格按模型命名语义与工程实现**来拆解，不做泛泛而谈。
 模型名：

```
qwen3-embedding:0.6b-q8_0
```

1. `embedding` 是什么意思？

核心结论

> **`embedding` 表示这是一个“专门用于向量表示”的模型，而不是生成文本的模型。**

也就是说：

- ❌ 不能正常“对话”
- ❌ 不适合生成长文本
- ✅ 只用于把文本 → 向量

和 `instruct` 的根本区别

| 模型类型        | 输入     | 输出                  |
| --------------- | -------- | --------------------- |
| `instruct`      | 文本     | 文本                  |
| **`embedding`** | **文本** | **向量（embedding）** |

所以你问的这个问题：

> 这个模型的输出是 embedding 吗？

**答案是：是的，而且“只会”输出 embedding。**

2. embedding 模型在系统中的角色（工程视角）

`qwen3-embedding` 的典型用途：

- RAG（检索增强生成）
- 向量数据库（FAISS / Milvus / Qdrant）
- 语义相似度
- 聚类 / 检索 / rerank 前的召回

在一个标准 LLM 系统里：

```
文本 → embedding → 向量库 → top-k → LLM
```

3. `0.6b` 表示什么？

> **模型参数规模是 0.6 Billion（6 亿）**

这对 embedding 模型来说是**相当主流的尺寸**：

- 小于生成模型（4B / 7B / 14B）
- 但足够捕捉语义结构
- 推理速度快、吞吐高

4. `q8` 表示什么？

核心结论

> **`q8` = 8-bit 量化（int8 / 类 int8）**

和你前面见过的 `q4` 类似，但定位不同。

为什么 embedding 通常用 `q8`

embedding 模型对数值精度**非常敏感**：

- 向量距离
- cosine similarity
- dot product

4-bit 量化可能会：

- 拉平向量分布
- 影响相似度排序
- 降低 recall / precision

因此工程实践中：

> **embedding：优先 q8
>  生成模型：常用 q4**

这是行业共识。

5. `q8_0` 里的 `_0` 是什么意思？

这是 GGUF / llama.cpp 体系中的**量化变体标识**。

结论先给出

> **`_0` 表示“标准 / 默认的 q8 量化方案”**

更精确一点的解释

在 GGUF 中：

| 标记   | 含义                                   |
| ------ | -------------------------------------- |
| `q8_0` | 逐 block 的 8-bit 对称量化（baseline） |
| `q8_1` | 改进版 q8（更复杂，收益有限）          |

`q8_0` 的特点是：

- 稳定
- 广泛支持
- 几乎无精度问题
- 计算开销可控

所以你看到 **绝大多数 embedding 模型都选 `q8_0`**。

6. 把整个名字翻译成人话

```
qwen3-embedding:0.6b-q8_0
```

可以翻译为：

> Qwen3 系列
>  专用 embedding 模型（不生成文本）
>  0.6B 参数规模
>  使用 8-bit 标准量化
>  保证向量质量与数值稳定性

7. 和你前一个模型的“系统级对照”

| 模型                            | 角色                | 量化             |
| ------------------------------- | ------------------- | ---------------- |
| `qwen3:4b-instruct-q4_K_M`      | 生成 / 推理 / Agent | q4（省资源）     |
| **`qwen3-embedding:0.6b-q8_0`** | **向量化 / 检索**   | **q8（保精度）** |

这是一个**非常“工程正确”的组合**。

8. 最后一句工程总结

> - `embedding` = 输出向量，不是文本
> - `q8` = 为相似度计算保精度
> - `q8_0` = 稳定、默认、业界通用

## 验证pull下来的模型

拉完之后怎么验证

查看本地已有模型

```shell
ollama list
```

![ollama-list](pic/ollama-list.png)

你应该能看到类似：

```shell
qwen3:4b-instruct-2507-q4_K_M
qwen3-embedding:0.6b-q8_0
```

Ollama **对 tag 非常严格**，多一个字符都会失败。

如果不确定，可以搜索局部名字：

```
ollama search qwen3
```

## 快速测试模型

### 快速测试对话模型

```shell
ollama run qwen3:4b-instruct-2507-q4_K_M
```

![ollama-test-instruct](pic/ollama-test-instruct.png)

Use `Ctrl + d` or `/bye` to exit.

### 快速测试embedding模型

```shell
ollama run qwen3-embedding:0.6b-q8_0 "你好，世界"
```

![ollama-test-embedding](pic/ollama-test-embedding.png)

或通过 API（更常见）：

```shell
curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "qwen3-embedding:0.6b-q8_0",
    "prompt": "强化学习中的 PPO 是什么"
  }'
```

> 为什么输出了很长一串数字？这是多长啊？谁规定的长度？在哪里能看到长度？

这是 **embedding 模型的正常、且唯一合理的输出形式**。下面我按你的三个问题逐一说明：**为什么是一串数字、到底多长、由谁规定、以及如何确认长度**。

1. 为什么会输出一长串数字？

核心原因

> **embedding 模型的输出不是文本，而是一个定长向量（vector）。**

你运行的是：

```
ollama run qwen3-embedding:0.6b-q8_0 "你好，世界"
```

这类模型的目标函数不是“生成下一个 token”，而是：

> 将输入文本映射到一个 **高维语义空间中的点**

因此输出形式必然是：

```
[float, float, float, ..., float]
```

这些数字：

- 是浮点数（通常 float32 / 量化后再反解）
- 每一维代表一种**语义特征投影**
- 本身**不可人工解读**
- 只用于数值计算（cosine / dot product）

2. 这串数字“有多长”？是不是随便的？

结论先给出

> **长度是固定的，由模型结构决定，不随输入变化。**

也就是说：

- “你好” → 向量长度 = N
- “你好，世界，这是一个测试” → 向量长度 **仍然 = N**

embedding 向量长度由什么决定？

**由模型在训练时就已经固定的 embedding head 决定**：

```
Transformer hidden size
        ↓
Projection / pooling
        ↓
Embedding dimension = 固定值
```

这是模型设计的一部分，不是运行时参数。

3. 谁规定这个长度？

严格来说，有三层“规定者”

1. **模型架构设计者（Qwen 团队）**
   - 决定 embedding 维度（如 768 / 1024 / 1536 等）
2. **训练目标**
   - 要在向量空间中做相似度学习
   - 维度太低 → 表达力不足
   - 维度太高 → 计算和存储成本过高
3. **下游生态（向量数据库 / 检索系统）**
   - 实际工程中常用固定维度
   - 便于索引、ANN 搜索、GPU 加速

4. 那“具体到底是多少维”？在哪里能看到？

最严谨、工程上推荐的做法：**直接数**

方法一：用 Ollama embeddings API（推荐）

```
curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "qwen3-embedding:0.6b-q8_0",
    "prompt": "你好，世界"
  }'
```

返回结果类似：

```
{
  "embedding": [ ... 一长串数字 ... ]
}
```

你只需要数：

```
len(embedding)
```

这就是**官方、无歧义的真实维度**。

# Ollama的交互模式

**Ollama的交互模式是标准的 HTTP API 方式；Ollama 的交互模式本质上就是“本地模型服务 + REST API”**。

你现在看到的命令行 `ollama run` 只是**最外层的一个 CLI 封装**。

下面我从**架构 → 交互模式 → 和 OpenAI API 的对应关系 → Python 示例**，一次性讲清楚。

## Ollama的真实交互模式是什么

核心一句话

> **Ollama 是一个本地常驻模型服务（daemon），CLI 只是它的一个客户端。**

安装完成后，Ollama 会在本地启动一个服务：

```shell
http://localhost:11434
```

所有交互最终都是：

```
Client  →  HTTP API  →  Ollama Server  →  本地模型
```

你现在用的：

```sehll
ollama run qwen3:4b-instruct-...
```

等价于：

> “CLI 客户端调用了本地的 HTTP API”

那 Ollama 的“交互模式”到底是什么？

用一句工程语言总结：

> **Ollama 是一个本地模型推理服务，**
>
> **CLI / Python / LangChain / LlamaIndex 都只是不同的客户端。**

推荐你在脑子里把它理解成：

```
Ollama ≈ 本地版 OpenAI API Server
```

## Ollama提供哪些API交互方式

官方支持的三类

| 方式               | 用途     | 本质                                                         |
| ------------------ | -------- | ------------------------------------------------------------ |
| CLI (`ollama run`) | 手动测试 | **命令行工具**，背后其实调用了 Ollama 的内部接口，但对用户来说是封装好的命令，不用自己写 HTTP 请求或者 SDK 代码 |
| **HTTP REST API**  | 程序调用 | ⭐ **正式接口**，可以用任何语言通过 HTTP 请求调用模型，适合集成在程序里 |
| OpenAI 兼容 API    | 生态对接 | **OpenAI 风格接口**，兼容现有 OpenAI SDK 或工具，方便现有生态接入 |

你关心的是 **第 2 和第 3 类**。

其中，CLI 把 API 调用包装成命令行命令，用户只需要输入命令，不关心网络请求、参数格式等细节。即

* CLI = “人类友好的快捷方式”

* HTTP REST / OpenAI-style = “程序可调用的正式接口”

## Ollama原生HTTP-API（最底层、最稳定）

### Chat/Generate（对应 chat completion）

请求示例（curl）

```shell
curl http://localhost:11434/api/chat \
  -d '{
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"}
    ]
  }'
```

返回的是 **流式 JSON（默认）**，这是 Ollama 的一个设计特点。

![chat](pic/chat.png)

```json
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.139350244Z","message":{"role":"assistant","content":"当然"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.267803294Z","message":{"role":"assistant","content":"可以"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.300357724Z","message":{"role":"assistant","content":"！"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.308803575Z","message":{"role":"assistant","content":"下面"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.31432336Z","message":{"role":"assistant","content":"是一"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.320268998Z","message":{"role":"assistant","content":"段"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.326082192Z","message":{"role":"assistant","content":"简单的"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.331884798Z","message":{"role":"assistant","content":" Python"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.337606066Z","message":{"role":"assistant","content":" "},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.34375524Z","message":{"role":"assistant","content":"代码"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.349786681Z","message":{"role":"assistant","content":"，"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.355623272Z","message":{"role":"assistant","content":"用于"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.361422103Z","message":{"role":"assistant","content":"判断"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.368357912Z","message":{"role":"assistant","content":"一个"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.373856886Z","message":{"role":"assistant","content":"整"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.37951655Z","message":{"role":"assistant","content":"数"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.384878482Z","message":{"role":"assistant","content":"是不是"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.390348508Z","message":{"role":"assistant","content":"偶"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.395855814Z","message":{"role":"assistant","content":"数"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.40109197Z","message":{"role":"assistant","content":"：\n\n"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.406350076Z","message":{"role":"assistant","content":"```"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.411655142Z","message":{"role":"assistant","content":"python"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.41687238Z","message":{"role":"assistant","content":"\n"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.422294556Z","message":{"role":"assistant","content":"def"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.427823167Z","message":{"role":"assistant","content":" is"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.433885641Z","message":{"role":"assistant","content":"_even"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.440038984Z","message":{"role":"assistant","content":"(n"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.44572543Z","message":{"role":"assistant","content":"):\n"},"done":false}
```

> 请问什么是流式 JSON？这设计特点有啥用？

#### 什么是流式 JSON？

核心概念

**流式 JSON = 服务器不一次性返回完整 JSON，而是把结果**“分片 / 分事件 / 按 token / 按 chunk”**逐步发送”**。

你看到的：

```json
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.139350244Z","message":{"role":"assistant","content":"当然"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.267803294Z","message":{"role":"assistant","content":"可以"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:39.300357724Z","message":{"role":"assistant","content":"！"},"done":false}
...
...
...
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:40.32677329Z","message":{"role":"assistant","content":"😊"},"done":false}
{"model":"qwen3:4b-instruct-2507-q4_K_M","created_at":"2026-01-22T16:37:40.332322571Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":1423876756,"load_duration":98720916,"prompt_eval_count":35,"prompt_eval_duration":129005264,"eval_count":183,"eval_duration":1119074549}
```

每一行都是 **一个 JSON 对象（一个 chunk）**：

- `message.content` → 这一段文本（通常是一个或几个 token）
- `done` → 是否生成完成（`false` 表示还没结束）
- `created_at` → timestamp

等生成结束时，最后一条会是：

```json
{"done": true}
```

直观比喻

> 就像 **流媒体视频**
>
> - 不是等整部电影生成完再给你看
> - 而是一边生成，一边播放
> - 你可以边看边接收

流式 JSON 就是文本生成版的“流媒体”。

#### 为什么 Ollama 设计成流式？

**1、实时性 / 响应速度**

- **不必等模型生成完整答案**
- 用户或客户端可以**立即显示部分结果**
- 提高交互体验，尤其是长文本生成

例如：

```
"下面是一段Python代码..." → 用户可以边显示边运行
```

**2、内存 / 网络优化**

- 如果一次返回上千 token 的 JSON：
  - JSON 太大 → 网络传输慢
  - 客户端内存占用高
- 流式返回每次只传少量 token → **网络压力低、延迟小**

**3、支持流式消费**

客户端可以：

- **边解析边显示**
- **边写入文件 / 数据库**
- **边做 token 级的逻辑处理**（如实时语音合成、RAG 插入）

#### 这种设计在工程上的意义

| 优点               | 场景示例                        |
| ------------------ | ------------------------------- |
| **低延迟响应**     | UI 即时显示生成文字             |
| **可中途截断**     | 用户取消请求 → 模型立即停止     |
| **边生成边处理**   | 生成向量 / 检索 / 音频实时合成  |
| **适合长文本生成** | 上万 token 文本不会一次性爆内存 |

#### 使用技巧

- **完整输出**：把每条 `message.content` 拼接起来
- **Python 示例**：

```python
import requests
import json

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"}
        ]
    },
    stream=True
)

full_text = ""
for line in resp.iter_lines():
    if line:
        data = json.loads(line)
        full_text += data["message"]["content"]
print(full_text)
```

* 这样你就可以得到 **完整文本**，而不是一条条 JSON chunk。

#### 总结一句话

> **流式 JSON = 模型生成按 token / chunk 分片返回 JSON，**
>
> **可以边生成边消费，提高实时性、降低内存和网络压力，是大模型本地服务（Ollama）交互的标准设计特点。**

如果你愿意，我可以帮你画一张**Ollama 流式 JSON 生成流程图**，让你一眼就看懂 **token → chunk → client → 拼接** 的整个过程。

好的，我用 **字符+文字** 给你画一张 **Ollama 流式 JSON 生成流程图**，完全用文本表示，方便在文档里直接用。

```
┌───────────────┐
│   Client      │
│ (你的程序/CLI)│
└───────┬───────┘
        │ 发送请求 (POST /api/chat)
        ▼
┌───────────────┐
│ Ollama Server │
│ (本地模型服务)│
└───────┬───────┘
        │ 生成文本（token by token）
        ▼
  ┌───────────────┐   每生成一小段 token
  │ JSON Chunk 1  │ ──────────┐
  │ {"message":{" │           │
  │ "content":"你"}}│          │
  │ "done":false} │           │
  └───────────────┘           │
  ┌───────────────┐   下一段 token
  │ JSON Chunk 2  │ ──────────┤
  │ {"message":{" │           │
  │ "content":"好"}}│          │
  │ "done":false} │           │
  └───────────────┘           │
         ...                   │
  ┌───────────────┐            │
  │ JSON Chunk N  │ ───────────┘
  │ {"message":{" │
  │ "content":"!"}}│
  │ "done":true}  │  ← 生成完成标志
  └───────────────┘
        │
        ▼
┌───────────────┐
│ Client 拼接   │
│ full_text =   │
│ chunk1+chunk2+│
│ ...+chunkN    │
└───────────────┘
```

解释：

1. **Client** → 发送请求（POST /api/chat）
2. **Ollama Server** → 模型开始生成，按 token 或小片段产生文本
3. **流式返回 JSON chunk**：
   - 每条 JSON 包含 `content`（当前生成的文本）和 `done`（是否完成）
4. **Client** → 边接收边拼接，得到完整结果
5. **优点**：
   - 实时显示
   - 低内存
   - 可中途取消或处理

#### resp.iter_lines是怎么持续接受断续的字符的

问题：

```python
resp = requests.post(...
# ...
for line in resp.iter_lines():
	# ...
```

是怎么持续接受断续的字符的，同时，是怎么知道结束了的？

的？

好的，我们分两部分仔细讲：**1）如何持续接受断续字符**，**2）如何知道流结束**。我尽量从底层原理解释，但用你能直接理解的 Python/HTTP 角度。

##### 1️⃣ 持续接受断续字符：`resp.iter_lines()` 怎么做到的？

**背景**

- 你在请求 `curl` 或 `requests.post(..., stream=True)` 时
- HTTP 连接是 **长连接 / 持续传输**，服务端可以一边生成一边发数据
- 数据不是一次性返回，而是**按块（chunked transfer）发送**

**1.1 HTTP Chunked Transfer**

HTTP/1.1 支持 **Transfer-Encoding: chunked**：

```shell
HTTP/1.1 200 OK
Transfer-Encoding: chunked

4\r\n
Wiki\r\n
5\r\n
pedia\r\n
0\r\n
\r\n
```

- 服务端可以分多块发送数据
- 每块大小可变
- 客户端读到最后一个长度为 0 的 chunk 就知道结束

Ollama 就是用类似机制把每个 token 的 JSON 逐个发送。

**1.2 Python `requests` 的处理**

```python
resp = requests.post(..., stream=True)
```

- `stream=True` → **不一次性读取整个响应**
- `resp.iter_lines()` → 迭代器，每次返回 **一行**（以 `\n` 或 `\r\n` 分割）
- 底层原理：
  1. `requests` 从 socket 持续读取数据块
  2. 缓冲区拼接数据，遇到换行符 `\n` 就返回一行
  3. 如果还没换行，就继续等待新数据到来

> 因此你看到的是“断续接收”，每个 JSON chunk 一行一行地出来。

**1.3 可类比示意（伪代码）**

```python
socket_buffer = ""
while True:
    new_data = socket.recv(1024)  # 等待服务端新数据
    if not new_data:
        break  # 服务端关闭连接
    socket_buffer += new_data
    while "\n" in socket_buffer:
        line, socket_buffer = socket_buffer.split("\n", 1)
        yield line  # 交给 iter_lines
```

这就是 Python `requests` 在后台做的事情。

> 我没太理解这里你说的类比代码。
>
> 因为我觉得resp = requests.post(完成之后，才开始for line in resp.iter_lines():的，不可能返回去又来一次request啊，只能一次性啊

啊，我明白你的疑惑了，你的理解里有一个 **常见误区**：你以为

```python
resp = requests.post(...)
for line in resp.iter_lines():
```

会等 `requests.post` 完全结束，才开始 `for line`，所以觉得怎么可能“持续接收断续字符”。其实不是这样。让我详细解释。

1️⃣ `requests.post(..., stream=True)` 真正的行为

- 当你写 `stream=True` 时：

```python
resp = requests.post("http://localhost:11434/api/chat", stream=True, json=...)
```

1. `requests.post` **发起 HTTP 请求**
2. **得到响应头**（HTTP status code, headers）
3. **并不会读取完整响应体**（因为 `stream=True`）
4. `resp` 对象已经拿到，但 **响应体仍在服务端生成**
5. 后续的 `for line in resp.iter_lines()` 才开始**按行读取 socket 流**

> 换句话说，`requests.post` 完成的是“请求发出 + 响应头收到”，不是“响应体收完”。

2️⃣ 类比真实过程（一次 TCP 连接）

可以把整个过程想成：

```python
Client: 发送 HTTP POST 数据
Client: 收到 HTTP 响应头 → resp 对象生成
while True:
    Client: 从 socket 读取一小块数据（不阻塞）
    if 有换行:
        yield line
    if 服务端 done 或关闭连接:
        break
```

- **重点**：这是一个 **单个 HTTP 连接**
- `for line in resp.iter_lines()` 并不是重新发请求
- 它只是“在这个已建立的 socket 上，边等待边读取数据”

3️⃣ 为什么可以“边生成边返回”

服务端（Ollama）是**边生成 token 边发数据**：

```python
{"message":{"content":"你"},"done":false}\n
{"message":{"content":"好"},"done":false}\n
...
{"done":true}\n
```

- 你的 HTTP 连接一直保持 open
- 每次生成一小段 JSON 就 flush 到 socket
- Python `iter_lines()` 检测到 `\n` 就返回一行
- **无需等待整个生成结束**

4️⃣ 用一个小动画文字比喻

```
时间 → 

服务端生成 token: "你"   → socket flush → client iter_lines() yield "你"
服务端生成 token: "好"   → socket flush → client iter_lines() yield "好"
服务端生成 token: "！"   → socket flush → client iter_lines() yield "！"
...
服务端生成完成          → done=True → client break
```

- 你看到的是**逐行输出**
- **整个过程只有一次请求**
- 并不是“请求返回后再发请求”，是**同一连接上流式接收**

5️⃣ Python 角度总结

```python
resp = requests.post(..., stream=True)  # 发起一次请求，拿到响应头
for line in resp.iter_lines():          # 持续从 socket 读取，每次一行
    process(line)                       # 可以边处理边拼接
```

- `requests.post` ≠ 等待全部数据
- `iter_lines` ≈ “阻塞迭代器 + socket buffer”
- 真实 HTTP 连接是一条线，数据逐步流过来

##### 2️⃣ 怎么知道流结束？

在流式 JSON 的设计里，有两种方式告诉客户端结束：

**2.1 `done` 字段**

你看到的每条 JSON 都有：

```
{"message":{"content":"!"}, "done":false}
```

- 当模型生成完毕，最后一条会：

```
{"done": true}
```

- 客户端就可以停止迭代，知道 **所有文本生成完成**

**2.2 HTTP 连接关闭**

- 在 chunked transfer 的底层，服务端最后会发送 **长度为 0 的 chunk**
- Python `requests.iter_lines()` 会自动检测到 EOF
- 如果服务端连接关闭，但最后一条 `done` 也没出现，迭代器也会结束

##### 3️⃣ 总结流程（结合你 Python 代码）

```python
full_text = ""
for line in resp.iter_lines():
    if line:  # 每次 line 对应一个 JSON chunk
        data = json.loads(line)
        full_text += data["message"]["content"]
        if data.get("done"):
            break  # 流结束
```

- **持续接收**：`requests` + socket + iter_lines()
- **断续**：每条 JSON chunk 对应几个 token / 一小段文本
- **结束条件**：1）`done=True` 或 2）HTTP 连接关闭

------

✅ 关键理解：

- **iter_lines() = 持续监听 socket，按行返回数据**
- **done = true** = 模型生成完成
- **流式 JSON = 边生成边返回，不必等完整文本**

### Python版本（官方推荐风格）

```python
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"}
    ],
    "stream": False
}

resp = requests.post(url, json=payload, timeout=100)
print(resp.json()["message"]["content"])
```

**这已经是完整可用的“API 交互方式”**。

返回：

![chat-2](pic/chat-2.png)

````markdown
当然可以！下面是一段简单的 Python 代码，用于判断一个整数是否是偶数：

```python
def is_even(n):
    return n % 2 == 0

# 示例使用
number = int(input("请输入一个整数："))
if is_even(number):
    print(f"{number} 是偶数。")
else:
    print(f"{number} 是奇数。")
```

### 说明：
- `n % 2 == 0` 表示整数 `n` 除以 2 的余数是否为 0。
- 如果余数为 0，说明是偶数；否则是奇数。
- 代码会提示用户输入一个整数，并输出判断结果。

你可以直接运行这段代码来测试！😊
````

## OpenAI风格API：你最关心的部分

好消息

> **Ollama 原生支持 OpenAI-compatible API。**

只需要把 `base_url` 指向本地即可。

你之前的代码，几乎不用改

你给的代码：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key="sk-xxxxxxxx"
)
```

改成 **Ollama 本地**：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 随便写，Ollama 不校验
)

response = client.chat.completions.create(
    model="qwen3:4b-instruct-2507-q4_K_M",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"}
    ],
    timeout=100,
)

print(response.choices[0].message.content)
```

**这在工程上是完全合法、完全可用的。**

返回：

![chat-3](pic/chat-3.png)

````markdown
当然可以！以下是一段简单的 Python 代码，用于判断一个整数是不是偶数：

```python
def is_even(n):
    return n % 2 == 0

# 示例使用
number = int(input("请输入一个整数："))
if is_even(number):
    print(f"{number} 是偶数。")
else:
    print(f"{number} 不是偶数。")
```

### 说明：
- `n % 2 == 0` 表示整数 `n` 除以 2 的余数是否为 0。
- 如果余数为 0，说明它是偶数；否则是奇数。

### 示例运行：
```
请输入一个整数：4
4 是偶数。
```

你可以直接运行这段代码，输入任意整数，它就会告诉你是不是偶数。😊
````

## 对应关系一览表（帮你建立心智模型）

| OpenAI 概念        | Ollama 对应                 |
| ------------------ | --------------------------- |
| `chat.completions` | `/v1/chat/completions`      |
| `embeddings`       | `/v1/embeddings`            |
| `model`            | 本地已 pull 的模型名        |
| `api_key`          | 不需要（占位即可）          |
| `base_url`         | `http://localhost:11434/v1` |

## embedding 的 OpenAI-style 用法（你前面正好用到了）

```python
response = client.embeddings.create(
    model="qwen3-embedding:0.6b-q8_0",
    input="你好，世界"
)

embedding = response.data[0].embedding
print(len(embedding))
```

这和你用 OpenAI embedding 的方式 **在接口层完全一致**。

## 什么时候用哪种方式？

| 场景              | 推荐                  |
| ----------------- | --------------------- |
| 手动测试          | `ollama run`          |
| Python / 系统集成 | OpenAI-compatible API |
| 流式 / 高控制     | 原生 `/api/chat`      |
| RAG / Agent 框架  | OpenAI-compatible     |

> Python版本（官方推荐风格） 和 OpenAI风格API：你最关心的部分 到底现在一般用哪个？

Python 版本（官方推荐风格） vs OpenAI 风格 API

### 1️⃣ Python 版本（官方推荐风格）

```
import requests

url = "http://localhost:11434/api/chat"
payload = {
    "model": "qwen3:4b-instruct-2507-q4_K_M",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"}
    ],
    "stream": False
}

resp = requests.post(url, json=payload, timeout=100)
print(resp.json()["message"]["content"])
```

- **本质**：直接调用 HTTP 接口
- **依赖**：只有 `requests` 库
- **优点**：
  - 透明、轻量
  - 官方推荐、适合本地或低依赖环境
- **缺点**：
  - 参数和流式处理需要自己实现
  - 不方便在多厂商/多模型环境下统一调用

### 2️⃣ OpenAI 风格 API

```
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434", api_key="")
response = client.chat.completions.create(
    model="qwen3:4b-instruct-2507-q4_K_M",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"}
    ]
)
print(response.choices[0].message.content)
```

- **本质**：在 Python 层封装了 HTTP 请求和响应解析
- **优点**：
  - 与 OpenAI API 完全兼容 → 多厂商可迁移
  - 内置流式和 JSON 解析
  - 生态友好，支持 LangChain、AutoGen 等框架
- **缺点**：
  - 依赖 `openai` 库
  - 对底层 HTTP 透明性略低

### 3️⃣ 工程实践趋势

- **现在大部分实际项目和主流框架都用 OpenAI 风格 API**
  - 原因：生态兼容性高、支持多厂商、流式处理方便
  - LangChain、LlamaIndex、AutoGen 等框架都默认使用 OpenAI 风格 API
- **Python 官方风格（requests）**：
  - 适合学习、低依赖本地调用
  - 或只想简单调用单个模型的场景

### 4️⃣ 总结建议

| 场景                               | 推荐方式                    |
| ---------------------------------- | --------------------------- |
| 学习 / 本地快速调用 / 低依赖       | Python 官方风格（requests） |
| 工程化 / 多厂商兼容 / 生态框架支持 | OpenAI 风格 API             |

> 简单一句话：**如果追求可迁移性和工程实践，大部分团队现在直接用 OpenAI 风格 API；如果只是本地或轻量调用，官方 Python 风格也够用。**

## 最后一句总结（非常关键）

> **你现在已经不需要“命令行交互”了，**
>
> **Ollama 完全可以作为你系统中的一个“模型服务节点”。**

如果你愿意，下一步我可以直接帮你：

- 把 Ollama 接进 **你已有的 Agent / RL / Planner 架构**
- 或给你一个 **“Ollama + embedding + 向量库”的完整最小工程结构**

# Ollama在linux下的服务配置

看使用场景和运维需求，我给你整理一下 **Linux 下 Ollama 用户的两种主要方式**，并分析利弊：

## 1️⃣ 临时 `export` 环境变量 + 手动启动

**流程：**

```shell
# 监听地址，默认 127.0.0.1:11434，可改 0.0.0.0:11434 允许外网访问
export OLLAMA_HOST="0.0.0.0"
# 跨域 / API来源
export OLLAMA_ORIGINS="*"
# 并发能力，runner的数量
export OLLAMA_NUM_PARALLEL=2
# 同时加载模型数量
export OLLAMA_MAX_LOADED_MODELS=2
# 模型驻留时间，按需加载到内存，超时会卸载（OLLAMA_KEEP_ALIVE）
export OLLAMA_KEEP_ALIVE="2h"
# 自定义模型下载和寻找路径
# export OLLAMA_MODELS="/data/ollama_models"
# 启动服务
ollama serve
```

- **特点：**
  - 只在当前 shell 会话生效
  - 关闭终端或重启服务器后，需要重新设置
  - 启动服务时可以随时修改参数，非常灵活
  - 适合 **开发、测试、临时调试**，尤其是尝试不同模型、并发数时
- **优点：**
  - 灵活、快速，不改系统文件
  - 易于在不同用户/项目间切换设置
  - 不需要 sudo 权限
- **缺点：**
  - 重启机器或服务后失效，需要重新 export
  - 对于长期开机提供 API 服务不稳定（易忘记启动）

## 2️⃣ 修改 systemd 服务配置（override 或直接修改 service）

1、前提

- Ollama 官方 FAQ / 中文社区推荐在 Linux 下使用 systemd 来管理长期运行的服务。
- 核心目标：
  1. 服务开机自启
  2. 设置环境变量（并发数、多模型、驻留时间等）
  3. 保持稳定运行
- systemd 有两种配置方式：
  - **直接修改主 service 文件** `/etc/systemd/system/ollama.service`
  - **使用 override.conf** 在 `/etc/systemd/system/ollama.service.d/override.conf` 添加增量配置（官方推荐，因为不改原文件，升级不会被覆盖）

2、正确的统一做法（推荐 override.conf）

步骤

* 创建或编辑 override 配置：

```
sudo systemctl edit ollama.service
```

然后在打开的编辑器里写：

```
[Service]
# 设置环境变量
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=4"
Environment="OLLAMA_KEEP_ALIVE=2h"
# Environment="OLLAMA_MODELS=/data/ollama_models"
# 如果需要覆盖 ExecStart
# ExecStart=
# ExecStart=/usr/local/bin/ollama serve
```

> **注意**：
>
> - 如果你只是加环境变量，**不需要覆盖 ExecStart**。只加环境变量即可。
> - 只有当想修改启动命令本身（比如路径或参数）时才需要 `ExecStart=` 覆盖原有值。

修改完后，按`Ctrl+O`报错，按`Enter`键确认，按`Ctrl+X`退出。

* 重新加载 systemd 配置：

```
sudo systemctl daemon-reload
```

* 重启 Ollama 服务：

```
sudo systemctl restart ollama.service
```

* 开机自启（可选）：

```
sudo systemctl enable ollama.service
```

3、为什么你看到两种写法不一样

| 写法                                         | 特点                                   | 是否推荐                     |
| -------------------------------------------- | -------------------------------------- | ---------------------------- |
| 写入 `[Service]` 环境变量 + 不覆盖 ExecStart | 简单，保留原来的启动命令，只加环境变量 | ✅ 官方/社区推荐              |
| 写入 `[Service]` 环境变量 + 覆盖 ExecStart   | 可以改启动命令，例如改路径或参数       | ⚠️ 仅在需要修改启动命令时使用 |

> 所以两种写法都是“正确”的，只是用途不同：
>
> - **加环境变量 → 推荐方式**
> - **改启动命令 → 覆盖 ExecStart**

4、统一总结

- **Linux 下推荐做法**：
  - 使用 `override.conf` 或 `systemctl edit` 添加环境变量
  - 不改原 `ExecStart` 就够了
  - `daemon-reload + restart` 生效
  - `enable` 保证开机自启
- **不推荐**：直接修改主 service 文件（升级后可能被覆盖）
- **ExecStart 覆盖** 仅在确实需要修改启动命令时使用

**特点：**

- 开机自启动
- 配置固定，不会随 shell 会话消失
- 适合 **生产、长期提供 API 的服务器**

**优点：**

- 长期稳定运行，无需手动启动
- 对团队或服务部署标准化
- 可以配合 `systemctl restart/reload` 管理

**缺点：**

- 配置不灵活，每次修改需要编辑 override 并 reload systemd
- 临时测试新的模型或参数需要额外修改服务或改回 export

## 总结

| 使用方式                     | 场景                   | 优缺点                                  |
| ---------------------------- | ---------------------- | --------------------------------------- |
| 临时 `export + ollama serve` | 开发、调试、单用户测试 | 灵活、快速，但不稳定、每次需重新 export |
| systemd 服务配置             | 生产、长期 API 服务    | 稳定、开机自启、团队共享，但修改不灵活  |

**一般用户选择：**

- **开发者/个人实验**：临时 export + serve
- **生产/团队 API 服务**：systemd service + override.conf

# Ollama推理服务机制与工程边界

## Ollama 推理服务概览

Ollama 是一个面向开发者的**本地大模型推理工具**，其核心目标是降低使用门槛，而非提供生产级、高并发的推理能力。它通过统一的模型管理和简化的 API，让用户可以在本机快速运行和测试大语言模型。

从系统形态上看，Ollama = **本地推理服务进程 + 按需加载的模型实例**。

## 服务生命周期：启动、常驻与关闭

### 服务是否开机自启

- **默认不开机自启**。
- Ollama 的后台服务通常在以下情况被启动：
  - 执行 `ollama run <model>`
  - 通过 HTTP API（如 `POST /api/chat`）发起请求
- 如果服务尚未运行，**第一次调用会自动触发服务启动**。

结论：是否提前启动服务并不影响使用逻辑，只会影响**第一次请求的延迟**。

### 服务启动后是否会自动关闭

- **不会因为空闲而自动退出**。
- 只要：
  - 系统不关机
  - 用户未手动停止
  - 未被系统强制回收

服务进程会一直常驻。

> 服务进程是常驻的，但模型不是。

### 检测模型服务是否在启动

这个不常用，因为没必要，因为就算没有启动，你一旦调用服务，服务就会启动。

## 模型加载与卸载机制

### 服务启动是否加载模型

- **不会**。
- 启动 Ollama 服务仅表示：
  - 监听端口（默认 11434）
  - 等待推理请求

模型权重只有在**首次请求某个具体模型**时才会被加载。

### 模型是否会一直驻留在内存/显存

- 模型加载后会驻留一段时间
- 在以下情况下，模型可能被卸载：
  - 长时间无请求
  - 系统内存 / 显存压力增大

- 卸载是**内部策略**，并非精确定时或可配置行为

结论：

- **服务进程：常驻**
- **模型实例：按需加载，空闲可回收**

## CPU与GPU推理支持

### 是否支持 GPU 推理

- **支持**。
- 常见支持场景：
  - NVIDIA GPU（CUDA）
  - Apple Silicon（Metal）

### 设备选择策略

- Ollama 会**自动检测可用 GPU**
- 若 GPU 可用且兼容：
  - **默认使用 GPU 推理**
- 否则：
  - 回退到 CPU 推理

> 用户无需、也无法在 API 或 CLI 中显式指定使用 CPU 或 GPU。

### 如何确认是否在使用 GPU

- 推理过程中查看 `nvidia-smi`
- 观察显存占用和进程信息

## 并发能力分析

### Ollama 是否支持并发

结论：

> **Ollama 能接收并发请求，但不是为并发推理设计的。**

### 网络层并发（支持）

- Ollama 是 HTTP 服务
- 可以同时接收多个请求
- 不会在连接层直接拒绝并发请求

也就是说：

```
Client A ──┐
           ├──> Ollama HTTP Server
Client B ──┘
```

### 推理层并发（非常有限）

- 实际推理时：
  - GPU / CPU 基本只处理一个推理任务
- 其他请求：
  - 排队等待
  - 串行执行

可以理解为：

```
[ Request 1 ] ---> 推理中
[ Request 2 ] ---> 等待
[ Request 3 ] ---> 等待
```

不存在：

- 请求调度器
- 动态 batch
- KV cache 共享
- token 级并发调度

### 并发下的典型表现

- 并发请求数增加：
  - 总吞吐不线性增长
  - 单请求延迟显著上升
- GPU 利用率提升有限

> 能并发接收请求 ≠ 能并发推理。

## Serve与Runner的关系

####  Ollama的服务/进程机制

你可以在 Linux 上用以下命令确认 Ollama 服务是怎么启动的：

```shell
# 查看哪个进程在监听 11434 端口
sudo lsof -i :11434

# 或者用 netstat / ss
ss -ltnp | grep 11434

# 查看进程路径
ps -ef | grep ollama
```

- 如果路径是 Desktop App 的安装目录 → 是 Desktop 自动启动的。
- 如果路径是 CLI 的 `ollama serve` → 说明有人配置了 systemd 或你手动启动过。

**Linux CLI 默认不开机启动**

**你的开机就能运行** → 很可能是 Desktop 版后台启动或 systemd 自动启动。

Python SDK / HTTP 请求 **不会自动启动服务**，它只是向已存在的服务发请求。

```shell
ps -ef | grep ollama
# ollama       192       1  0 08:56 ?        00:00:00 /usr/local/bin/ollama serve
# ollama      1240     192 12 09:30 ?        00:00:06 /usr/local/bin/ollama runner --ollama-engine --model ... --port 41513
```

分析

1. **PID 192: `/usr/local/bin/ollama serve`**

- 父进程 PID 是 1，说明它是 **init/systemd 启动的守护进程**。
- 说明这个 Ollama 服务确实是在开机时被启动的，而不是你手动执行的。

1. **PID 1240: `/usr/local/bin/ollama runner ...`**

- 父进程是 PID 192，也就是上面的 `ollama serve`。
- 这是实际加载模型并处理请求的 **推理进程**。
- `--port 41513` 是内部端口，用于服务与 runner 通信。

1. **说明**

- Python SDK 发 HTTP 请求时，访问的是 `localhost:11434`，也就是 `ollama serve` 的端口。
- 它不会自动启动 `ollama serve`；服务是开机就启动的，所以请求可以直接命中服务。

> 可是我的推理服务刚才就调用了一次，代码运行了一次就结束了，为啥还能看到PID 1240呢？

哦，这就清楚了，你看到的 `1240` 其实就是 **runner 进程并不是一次性执行 Python 请求就退出的**，而是 `ollama serve` 启动的 **常驻推理进程**。解释如下：

Ollama 的服务/进程机制

1、**`ollama serve`（PID 192）**

- 这是守护进程，常驻系统，负责监听 HTTP 请求（默认端口 11434）。
- 它本身不会做模型推理，只管理 **runner**。

2、**ollama runner`（PID 1240）**

- 当 `ollama serve` 启动时，它会 **启动一个或多个 runner 进程**，用来真正加载模型并处理推理请求。
- runner **是长期存在的**，不会随着单次 Python 请求结束而退出。
- runner 会一直保留模型在内存里，以便 **快速响应下一次请求**。

3、**Python 调用行为**

- 你 Python 发送请求时，只是把请求发给 `ollama serve`。
- `ollama serve` 将请求转给已经存在的 runner 处理。
- 请求处理完毕后，runner **不会退出**，因此你仍然能看到 PID 1240。

总结

- **单次请求不会杀掉 runner** → 推理服务保持常驻。
- **多次请求都用同一 runner** → 避免重复加载模型，提高响应速度。
- 如果你想关闭服务，需要手动 `kill` 或停止 `ollama serve`，否则 runner 会一直存在。

Ollama 推理服务由 **Serve** 和 **Runner** 两个主要进程组成，它们的职责和关系如下：

```
Ollama Serve（守护进程）
─────────────────────────────
角色：
1. 管理 runner 的生命周期（启动、监控、重启等）
2. 负责 HTTP/REST 接口的请求收发（接收请求 → 转发给 runner → 返回结果）
3. 并不是做模型推理的“干活者”，只是协调者

Ollama Runner（推理进程）
─────────────────────────────
角色：
1. 被 serve 管理，是否启动由 serve 决定
2. 执行实际模型推理（加载模型、保持模型在内存、处理请求）
3. 常驻进程，保持模型在内存以加速后续推理请求
```

可以把它理解为 **Serve 是管理层 + 通信层**，**Runner 是执行层**。

#### Serve的角色

- **管理 Runner**：负责启动、调度、回收 Runner，保证有足够的计算资源处理请求。
- **收发信息**：监听客户端请求（HTTP 或 OpenAI 兼容 API），将请求转发给空闲的 Runner，返回结果给客户端。
- **并发控制**：通过配置控制同时运行的 Runner 数量，从而支持一定程度的并发推理。

#### Runner的角色

- **执行模型推理**：加载指定模型，在 Serve 的调度下完成具体计算任务。
- **被 Serve 管理**：是否启动、何时回收、处理哪个请求都由 Serve 控制，用户通常不直接操作 Runner。

#### Serve与Runner的关系概述

- Serve 是“大脑”，Runner 是“工人”。
- Serve 根据请求负载管理 Runner 的数量和状态。
- Runner 只做具体推理，不处理请求分发或并发管理。
- 当 Serve 配置多个 Runner 时，可以同时处理多个请求，实现并发能力；单个 Runner 时并发受限。

Ollama 的 **服务与进程关系 + 请求流程**，尽量直观展示常驻服务、runner、Python 调用之间的关系：

```
┌─────────────────────────────────────────────┐
│                 Linux 系统                  │
│                                             │
│  ┌───────────────┐       ┌───────────────┐  │
│  │  ollama serve │──────▶│  HTTP 监听端口 │  │
│  │   (守护进程)    │       │   11434      │  │
│  └───────┬───────┘       └────────┬──────┘  │
│          │                        │         │
│          │ 启动 runner             │         │
│          ▼                        │         │
│  ┌───────────────┐                │         │
│  │ ollama runner │◀───────────────┘         │
│  │ (模型加载 &   │ 处理请求并返回结果            │
│  │  推理处理)   │                             │
│  └───────┬───────┘                           │
│          │                                   │
│          │ 保持常驻                           │
│          ▼                                   │
│  ┌───────────────┐                           │
│  │ Python/CLI/API│                           │
│  │ 调用请求模型    │──────────────────────────▶│
│  └───────────────┘                           │
│          │                                   │
│          │ 等待响应                           │
│          ▼                                   │
│      输出结果                                 │
└─────────────────────────────────────────────┘
```

**serve 守护进程**

- 常驻系统，负责 HTTP 请求路由。
- 只管理 runner，不直接做推理。

**runner**

- 常驻进程，负责加载模型、执行推理。
- 不会随单次请求退出，保持模型在内存中以加速后续请求。

**请求流程**

- Python/CLI/API 发送请求到 `serve` → `serve` 转发给 runner → runner 推理并返回结果 → serve 返回给调用方。

**优势**

- 多次请求复用同一 runner → 避免重复加载模型。
- 可以并发处理多条请求（每个请求可分配不同线程/runner 池）。

#### 如何启动多个runner

在 Ollama 里，**多个 Runner 并不是手动每次去启动一个新进程**那么简单，它的并发管理是 **Serve 控制 Runner 的启动和分配**，你手动很少直接启动 Runner。具体来说：

#### Serve启动Runner的机制

- 当你执行 `ollama serve` 后，Serve 会：
  1. 根据配置或默认设置决定要启动多少 Runner（通常是 1 个）。
  2. 当有请求到达时，如果空闲 Runner 不够，它可以按需启动新的 Runner（取决于 Ollama 版本和内部策略）。
  3. Serve 会管理这些 Runner 的生命周期：启动、分配请求、回收空闲 Runner。

#### 手动控制Runner数量

Ollama CLI 本身不提供“直接命令启动 N 个 Runner”的公开接口。想要多 Runner 通常有两种方法：

1. **调整 Serve 配置（推荐）**

   - 在 Ollama 的配置文件里（通常在 `~/.ollama/config.json` 或类似路径），设置：

     ```
     {
       "runner_pool_size": 3
     }
     ```

     这样 Serve 启动时就会预先启动 3 个 Runner。

   - 优点：Serve 管理自动分配请求，模型内存复用更高效。

2. **手动启动多个 Runner（不常用）**

   - 可以直接用 `ollama runner --ollama-engine --model <model_path> --port <port>` 启动，但：
     - Serve 默认只会和自己管理的 Runner 通信，其他手动启动的 Runner 可能不能被 Serve 调度。
     - 实际用途不大，一般用于调试或特殊测试。

#### 总结

- **正常并发使用**：只需通过 Serve 接收请求，Serve 根据请求自动启动/调度 Runner。
- **配置并发上限**：修改 Serve 配置里的 `runner_pool_size` 或类似参数。
- **手动启动 Runner**：可行但不推荐，可能无法完全接入 Serve 调度。

## Ollama 的工程定位

Ollama 的设计取舍可以概括为：

- **优先易用性**
- **牺牲性能可控性与并发能力**

适合场景：

- 本地开发
- 单人使用
- Agent 调试
- 低并发、无 SLA 要求的应用

不适合场景：

- 多用户在线服务
- 高并发推理
- 对 P95 / P99 延迟有要求
- 需要精细控制 GPU / 显存

## 与专业推理框架的边界

当出现以下需求时，应考虑迁移到专业推理框架（如 vLLM、TensorRT-LLM）：

- 明确的并发需求
- 稳定的延迟指标
- 高 GPU 利用率
- 多卡或集群部署

> Ollama 是本地推理的“入口工具”，而不是生产级推理系统的终点。

## 小结

- Ollama 服务：按需启动，常驻运行
- 模型加载：首次请求加载，空闲可卸载
- 设备选择：自动优先 GPU，无需显式配置
- 并发能力：HTTP 层支持，推理层弱并发

理解这些边界，有助于在正确的阶段使用 Ollama，并在需要时平滑升级到更专业的推理架构。

# Ollama本地调用服务和API远程调用服务的本质区别

> openai api远程调用和ollama这种本地的模型服务的本质核心区别是不是：
> openai api是远程调用大模型厂商（OpenAI，阿里，DeepSeek）服务器的模型推理服务，而ollama是调用本机的模型推理服务？
> 实际是这样吗？

你的理解**是正确的，而且已经抓住了本质**。
 我会在肯定你结论的基础上，把“哪里完全一样、哪里不一样、工程上该怎么理解”讲清楚，避免你后面形成模糊概念。

## 先直接回答你的判断

> **openai API 是远程调用大模型厂商服务器上的模型推理服务，
>  ollama 是调用本机（或内网服务器）上的模型推理服务。**

**是的，这句话在“本质层面 100% 成立”。**

## 用一句话建立正确的“心智模型”

> **两者的本质都是：
>  「Client → HTTP API → Model Inference Service」**

区别只在于：
 👉 **Model Inference Service 在哪里运行**

## 把两者画成“同一张架构图”

### OpenAI / 云厂商 API

```
你的程序
   |
   |  HTTPS
   v
OpenAI / 阿里 / DeepSeek 云端 API
   |
   |  高性能 GPU 集群
   v
大模型推理
```

特征：

- 模型在 **厂商服务器**
- 你只看到 API
- 不接触模型权重
- 按 token / 请求计费

### Ollama（本地模型服务）

```
你的程序
   |
   |  HTTP (localhost / 内网)
   v
Ollama 本地服务
   |
   |  CPU / GPU / Metal
   v
本地模型推理
```

特征：

- 模型在 **你自己的机器**
- 权重在磁盘上
- 不走公网
- 不按 token 计费

## 为什么你会“感觉它们很像”？

这是一个**非常关键的设计点**：

> **Ollama 故意把自己设计成 OpenAI API 的“等价替身”**

原因是：

- 生态已经围绕 OpenAI API 建好了
- LangChain / LlamaIndex / Agent 框架
- 开发者已经熟悉 `chat.completions`

所以你才会发现：

```
client = OpenAI(base_url=..., api_key=...)
```

**这一行几乎不变**。

## 那它们“真正不一样”的地方在哪？

下面是工程层面的关键差异。

### 资源责任

| 维度        | OpenAI API | Ollama       |
| ----------- | ---------- | ------------ |
| 计算资源    | 厂商负责   | **你负责**   |
| 显存 / 内存 | 不关心     | **必须关心** |
| 并发能力    | 极高       | 受限于本机   |

### 模型控制权

| 维度       | OpenAI API | Ollama       |
| ---------- | ---------- | ------------ |
| 模型权重   | 看不到     | **完全可见** |
| 版本固定性 | 厂商决定   | **你决定**   |
| 可复现实验 | 较弱       | **极强**     |

### 成本模型

| 维度     | OpenAI API | Ollama     |
| -------- | ---------- | ---------- |
| 单次调用 | 付费       | 免费       |
| 长期成本 | 线性增长   | 硬件一次性 |
| 离线可用 | 不可       | **可**     |

## 你应该怎么“正确理解”？

你刚接触，这里给你一个**非常重要的定位建议**：

> **不要把 Ollama 当成“另一种 API”，**
>
> **而是把它当成“你自己搭建的一个小型模型服务器”。**

就像：

- 你可以用云数据库（RDS）
- 也可以自己装 MySQL

**概念完全一致**。

## 一个你未来一定会遇到的“进阶问题”

当你继续深入时，你会发现还有第三类：

**自建推理服务（vLLM / TensorRT-LLM / Triton）**

就是说，Ollama和vLLM / TensorRT-LLM / Triton一样，都是推理服务的一种吗？

```
你的程序
   |
   v
你自己部署的推理服务（多卡 / 并发 / 批处理）
```

Ollama 实际上是：

> **“面向个人和小团队的、极低门槛的推理服务封装”**

## 最终一句总结（非常重要）

> ✔ 你的理解是**完全正确的**
>  ✔ 两者的区别不在 API，而在 **模型在哪里跑**
>  ✔ Ollama = 本地模型推理服务
>  ✔ OpenAI API = 远程模型推理服务

你现在已经完成了**“从 API 使用者 → 系统理解者”**这一步。

# OpenAI-API已成为大模型服务接口标准

> 各种大模型服务的api接口现在都是OpenAI API这种接口吗？就是说，都用的是from openai import OpenAI吗？

**简短结论先给出：不是“都”，但「OpenAI API 风格」已经成为事实上的行业标准。**

下面我从**历史、现状、例外、以及你在工程上该如何应对**四个层面讲清楚。

## 现在是不是“大家都用 OpenAI API”？

准确说法应该是：

> **大多数通用大模型服务，都在“兼容 OpenAI API”，**
>
> **但不等于它们的原生 API 就是 OpenAI API。**

也就是说：

- OpenAI API ≈ **通用接口标准**
- 但并非唯一接口

## 为什么 OpenAI API 成了事实标准？

三个关键原因

**1、先发优势 + 开发者惯性**

- OpenAI 最早提供：

  ```
  from openai import OpenAI
  ```

- ChatCompletion / Embeddings 定义得非常早

- 大量开源框架围绕它构建

---

**2、生态“锁定效应”**

以下生态基本**默认 OpenAI API**：

- LangChain
- LlamaIndex
- AutoGen
- CrewAI
- 各种 Agent / Tool 框架

厂商如果不兼容：

> 基本无法进入主流生态

---

3、兼容成本极低

对模型服务提供方来说：

- OpenAI API 是一层 **HTTP 协议**
- 不涉及模型内部结构
- 做一层 adapter 即可

## 现实中的分类（非常重要）

### 第一类：原生OpenAI-API

| 厂商             | 说明     |
| ---------------- | -------- |
| OpenAI           | 原生     |
| Azure OpenAI     | 轻微差异 |
| Yunwu / 中转服务 | 代理     |

你的使用OpenAI API的代码就是这一类。

### 第二类：**OpenAI-compatible API（主流）**

| 服务                  | 情况     |
| --------------------- | -------- |
| Ollama                | 原生支持 |
| vLLM                  | 官方支持 |
| LM Studio             | 支持     |
| Text Generation WebUI | 支持     |
| DeepSeek              | 支持     |
| 阿里云（通义）        | 支持     |
| 智谱 / 月之暗面       | 支持     |

👉 **你可以继续用：**

```
from openai import OpenAI
```

只改：

```
base_url
api_key
model
```

### 第三类：**非 OpenAI 风格（少数）**

这类一般是：

- 历史包袱重
- 或做垂直封装

例如：

- Hugging Face Inference API
- Google Gemini（早期）
- 一些私有企业接口

但即便如此：

> **它们现在也在补 OpenAI-compatible 接口**

## 那 `from openai import OpenAI` 是不是“唯一正确姿势”？

工程视角的正确答案

> **它是“最低阻力路径”，但不是架构上必须绑定的。**

推荐你的工程结构是：

```
你的业务逻辑
   |
   v
LLM Client 抽象层
   |
   +-- OpenAI API
   +-- Ollama (OpenAI-compatible)
   +-- DeepSeek
```

这样：

- 切换 provider ≈ 改配置
- 不改业务代码

## 一个你现在就该有的认知（非常关键）

> **OpenAI API ≠ OpenAI 公司**
>
> *它已经演变成一种“协议 / 接口规范”**

就像：

- SQL ≠ MySQL
- POSIX ≠ Linux

## 那未来会不会“统一到一个标准”？

非常现实的判断

- **短期：OpenAI API 继续主导**
- **中期：功能分叉（tool / vision / agent）**
- **长期：可能出现正式标准，但短期不会**

## 你现在该怎么做（实用建议）

既然你刚接触，我给你一个**极简、但正确的建议**：

> **优先使用 OpenAI-compatible API**
>
> *避免厂商私有 SDK 绑定**

你已经本能地走在**正确路径上**了。

## 一句话总结

- OpenAI API 已成为事实上的大模型服务接口标准
- 多数厂商选择“兼容”而不是“发明新接口”
- `from openai import OpenAI` 更像是“HTTP 客户端”，而不是“只属于 OpenAI”

# 参考资料



## 大模型

* [Ollma pull 使用指导](https://chatgpt.com/c/69722b54-8524-8322-93ea-757066c63fb1)

如果你下一步是：

- 接 Python / LangChain / LlamaIndex
- 用作 RAG
- 或和你现有的 **RL / agent 系统**对接

我可以直接给你**最小可运行示例**，不走弯路。