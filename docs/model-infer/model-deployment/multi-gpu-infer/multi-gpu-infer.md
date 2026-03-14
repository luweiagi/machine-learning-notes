# 单机多卡与多机多卡：从 PCIe 到 RDMA 的大模型训练与推理全景解析

* [返回上层目录](../model-deployment.md)
* [0. 写在前面：这篇文档是给谁看的？](#0. 写在前面：这篇文档是给谁看的？)
* [1. 整体大图：从一张“系统地图”开始](#1. 整体大图：从一张“系统地图”开始)
* [2. 硬件角色总览：谁负责算？谁负责搬？](#2. 硬件角色总览：谁负责算？谁负责搬？)
* [3. 单机内部的高速互联：PCIe / NVLink / NVSwitch / HCCS / 超节点](#3. 单机内部的高速互联：PCIe / NVLink / NVSwitch / HCCS / 超节点)
* [4. 跨机互联与 RDMA：IB / RoCE / 以太网到底是啥关系？](#4. 跨机互联与 RDMA：IB / RoCE / 以太网到底是啥关系？)
* [5. AllReduce：只讲到你真正需要的程度](#5. AllReduce：只讲到你真正需要的程度)
* [6. 单机多卡：数据在机内是怎么跑的？](#6. 单机多卡：数据在机内是怎么跑的？)
* [7. 多机多卡：从单机环到跨机大环（训练与推理）](#7. 多机多卡：从单机环到跨机大环（训练与推理）)
* [8. 通信库：NCCL / HCCL 在整条链路中的真实位置](#8. 通信库：NCCL / HCCL 在整条链路中的真实位置)
* [9. 310P、Atlas A2/A3 与实际工程选型](#9. 310P、Atlas A2/A3 与实际工程选型)
* [10. 单机 / 多机训推流程总览：从一行脚本到底层线缆](#10. 单机 / 多机训推流程总览：从一行脚本到底层线缆)
* [11. 给后来者的阅读建议与延伸方向](#11. 给后来者的阅读建议与延伸方向)
* [12. 附录：术语表与常见误解纠正](#12. 附录：术语表与常见误解纠正)



# 0. 写在前面：这篇文档是给谁看的？

在你开始正式看第 1 章之前，先用一小节把这篇文档“讲清楚”：

- 它是写给谁的？
- 它想解决什么样的困惑？
- 你大概可以用什么方式来阅读它？

## 0.1 这篇文档是写给谁的？

先说结论：  

> **它不是写给已经在搞 GPU 通信十几年的人看的。** 
>
> 而是写给“刚入门，却不想只背名词而是想理解本质”的工程师 / 学生 /爱折腾的人看的。  

更具体一点，如果你符合下面几条中的任意几条，这篇文档就是给你写的：

- 你开始接触：
  - 单机多卡训练 / 推理。
  - 多机多卡训练 / 推理。
  - 或者在看 MindIE / vLLM-Ascend / HCCL / NCCL 的文档。
- 你脑子里已经积了一堆名词：
  - PCIe / NVLink / NVSwitch / HCCS / 超节点。
  - RDMA / IB / RoCE / GPUDirect RDMA。
  - NCCL / HCCL / AllReduce / Ring / Hierarchical。
  - 310P / Atlas 800T/800I A2/A3 / MindIE / vLLM-Ascend ……
- 但这些名词之间的关系是“散的”：
  - 你知道大概是干嘛的，但放不到同一张图上。
  - 一遇到实际问题（比如：为什么 310P 做不了多机？为什么要超节点？）就会瞬间迷茫。

如果你有这样的感觉：

> “我不想只是会照着教程敲命令，
>
> 我想知道 **数据到底在卡和卡、机和机之间怎么走**，
>
> 想知道 **每次看到的这些名词在整条链路中的位置**。”  

那这篇文档就是为你准备的。

## 0.2 这篇文档想解决的核心困惑

这篇文档的出发点，其实就一句话：

> 帮你把“多机多卡训推”这件事，从一堆碎片化的名词，变成一张可以在脑子里重放的**完整动态图**。  

它希望帮你搞清楚的，不是某一个单独的点，而是整个链路：

- 从你写下 `loss.backward()` / `model(input)` 这一行开始：
  - 框架（PyTorch / MindSpore / MindIE / vLLM-Ascend）做了什么？
  - 通信库（NCCL / HCCL）在什么时候被调用？
  - 谁在决定“现在要 AllReduce 一块梯度”？
  - 单机内时是谁在发 DMA？走的是 PCIe 还是 NVLink/HCCS？
  - 多机场景下是哪个 NIC 在发起 RDMA？走的是 IB 还是 RoCE？
  - CQ 完成通知是怎么回到 NCCL / 框架 / 你的代码那一层的？

换句话说，你读完之后应该能做到：

- 看到一句“某模型在 8×A100 上用 tensor parallel + data parallel 训练”时：
  - 能在脑子里画出大致的数据流图，不至于完全靠想象。
- 遇到“310P 多机不支持”这种结论时：
  - 能从互联架构 + HCCL 支持矩阵的角度解释“为什么不是扯淡，而是架构设计如此”。
- 在做工程选型 / 排查训练慢的问题时：
  - 不再只能猜“是不是 batch size 太小”，而是能顺着 DMA / RDMA / 拓扑 / AllReduce 这条线往下查。

## 0.3 你可以怎样阅读这篇文档？

这不是一篇“必须线性一口气读完”的论文，而更像一本“可以反复翻”的小册子。一个建议的阅读方式是：

- **第一遍**：从第 1 章顺读到第 4 章，再跳到第 6 / 7 / 8 / 10 章。
  - 目的：
    - 先有一个粗略的全局认知：谁是硬件、谁是路、谁是搬运工、谁是调度者。
  - 不要纠结每一个细节，只要能跟上故事线就行。

- **第二遍**：把你最感兴趣 / 最迷茫的章节多看几次。
  - 比如：
    - 想搞清楚单机多卡内部：多看第 2 / 3 / 6 章。
    - 想搞清楚多机 RDMA：多看第 4 / 7 / 10 章。
    - 想搞清楚 NCCL/HCCL：看第 8 / 10 章。

- **第三遍及之后**：把它当成“参考书”和“速查表”。
  - 忘了某个名词，翻第 12 章。
  - 遇到 AllReduce 性能问题，翻第 5 / 6 / 7 / 10 章。
  - 做设备选型 / 和别人争论 310P 能不能多机时，翻第 9 章。

你可以把这篇文档当成是**将来你自己给别人讲这套东西时的讲稿/底稿**。  

## 0.4 一点点阅读上的小建议

- **不要怕来回跳章节**：
  - 比如在看第 7 章多机数据流时，回去翻第 2 / 3 / 4 章补一下名词，是非常正常的。
- **可以自己在纸上画图 / 写小例子**：
  - 比如：画出“单机 4 卡 P2P 拓扑图”、“两机 8 卡 AllReduce 环路图”。
  - 甚至写一点小伪代码，把 `loss.backward()` → `ncclAllReduce` → DMA / RDMA 的调用链条自己过一遍。
- **把不懂的地方当成“锚点”而不是挫折**：
  - 任何一个你现在觉得绕的地方，其实都是整个系统中的一个“关键转折点”。
  - 很多大厂工程师也会在这些点上反复踩坑，只要你愿意多看几遍，就已经比大多数“只记命令”的人走得更远了。

最后一句话：  

- 这篇文档不追求“术语最全”，而是追求“你能真的在脑子里把多机多卡训推的过程从头走一遍”。
- 只要你愿意耐心从第 1 章往下看，你现在的“强烈求知欲 + 好奇心”就足以把这套东西吃透。  

# 1. 整体大图：从一张“系统地图”开始

这一章的目标，是先给你一张“脑内总地图”：

- 你写的那一行训练 / 推理脚本，**往下到底踩到了什么东西**。
- 为什么会有“单机多卡”“多机多卡”这些概念。
- 后面所有名词（RDMA、NVLink、NCCL、HCCL、超节点……）大概分别处在这张图的哪一层。

你可以先不用记任何细节，只要把这张“大致轮廓”装进脑子，后面的章节就是在一点点把这张图从“轮廓”变成“高清”。

## 1.1 为什么会有单机多卡和多机多卡？

从本质上讲，就两句话：

- **模型和数据太大，一张卡吃不下 / 算不完。**
- **一张卡算得太慢，必须把计算摊到多张卡 / 多台机器上。**

更细一点拆：

- **单机多卡**：
  - 一台服务器里插了多张 GPU / NPU（比如 4 卡、8 卡）。
  - 适用场景：
    - 模型能塞进单机（显存总量够），但想加速训练 / 推理。
    - 做 Tensor Parallel、Pipeline Parallel 但规模还不算离谱。
  - 关键问题：**机内卡与卡之间怎么高效“互相抄作业”（传梯度 / 激活 / 参数）？**

- **多机多卡**：
  - 多台服务器，每台各有多张卡，比如：
    - 8 卡 × 16 机 = 128 卡集群。
  - 适用场景：
    - 模型太大，单机显存不可能装下（上百亿、上千亿参数）。
    - 训练 / 推理吞吐要求极高，需要几十 / 上百张卡协同。
  - 关键问题：
    - 机与机之间如何高效传数据？
    - **不能再靠 CPU + 普通网卡 + TCP 这一套，必须用 RDMA / IB / RoCE。**

换个更形象的类比：

- 单机多卡：在**同一个实验室里**的 8 个人，互相传纸条，快得多（NVLink / PCIe P2P）。
- 多机多卡：**不同楼层 / 不同大楼之间**传纸条，一定要依赖电梯 / 快递系统（RDMA + IB/RoCE）。

你看到的各种硬件名词（NVLink、NVSwitch、HCCS、RDMA、IB、RoCE）说白了都在回答一个问题：

> “当我们不止用一张卡时，**这些卡之间怎么才能尽量快地互相“同步信息”**？”

## 1.2 从模型代码到底层硬件的一条完整链路

假设你在某台机器上跑了这样一行训练代码（伪代码）：

```python
loss = model(input)
loss.backward()
optimizer.step()
```

在“单机多卡 / 多机多卡”的世界里，其实底下依次发生了这些事（大致分层）：

1. **框架层（PyTorch / MindSpore / 其它）**
   - 你写的 `model(input)` 和 `loss.backward()` 会被框架分解成一堆算子。
   - 框架会根据你使用的并行方式（`DistributedDataParallel` / `ModelParallel` 等）决定：
     - 哪些算子在哪张卡上跑。
     - 梯度、激活、权重需要在什么时候在卡与卡之间同步。

2. **通信库层（NCCL / HCCL）**
   - 当框架需要“多卡一起干一件事”（例如把每张卡上的梯度求和）时，会调用通信库：
     - 典型调用：`allreduce(grad_buffer)`、`broadcast(weights)` 等。
   - 通信库负责：
     - 选定算法（Ring AllReduce / Tree AllReduce / 分层 AllReduce 等）。
     - 规划拓扑（谁给谁发、顺序怎样）。
     - 调用底层接口触发 **单机 DMA** 或 **跨机 RDMA**。

3. **DMA / RDMA 层（数据搬运硬件）**
   - **单机内部**：
     - 由 **GPU 内部的 DMA Engine** 发起搬运，通过 **PCIe P2P / NVLink / HCCS** 把数据从 GPU0 显存搬到 GPU1 显存。
     - CPU 不搬数据，只在一开始下了一个“搬运任务”。
   - **跨机之间**：
     - 由 **NIC 上的 RDMA Engine** 发起搬运：
       - 通过 PCIe 直接 DMA 读取本机 GPU 显存。
       - 把数据打包通过 **IB / RoCE 网络**丢给远端机器。
       - 远端 NIC 再 DMA 写入目标 GPU 显存。

4. **网络 / 高速互联层**
   - 单机内部：
     - 数据走的是 **PCIe 总线** 或 **NVLink / NVSwitch / HCCS**。
   - 多机之间：
     - 数据走的是 **RDMA 网络**：
       - 要么是 **InfiniBand（IB）**。
       - 要么是 **RoCE（在以太网上跑 RDMA）**。

5. **物理硬件层**
   - 真正“在搬比特”的实体：
     - GPU 内部的 DMA Engine。
     - NIC 上的 RDMA Engine。
     - PCIe / NVLink / HCCS 总线。
     - IB / RoCE 交换机和光纤 / 铜缆。

整条链路可以用一行话概括：

> 你写的 `loss.backward()`，最后会变成一堆对 **NCCL/HCCL** 的调用，
>
> NCCL/HCCL 把工作再拆成 **GPU DMA** 和 **NIC RDMA** 的硬件指令，
>
> 通过 **PCIe / NVLink / HCCS / IB / RoCE** 把各种张量在卡和卡、机和机之间搬来搬去。

后面几章，就是把这条链路从“大致几行文字”拆成“你可以手动画出每一步时序图”的程度。

## 1.3 训练 vs 推理：相同点与不同点

很多新手一开始会有个直觉：

> “多机多卡好像是用来**训练**大模型的，推理是不是就没那么复杂？”  

实际上，从**通信和数据流**角度看：

- 训练和推理有不少**共通的基础设施**：
  - 都需要把 **权重** 分片 / 复制到不同卡上（张量并行 / 数据并行 / 张量切分）。
  - 都可能需要在 **卡与卡之间传中间结果**（激活、KV Cache、专家路由结果等）。
  - 都受到底层 **单机 DMA + 跨机 RDMA 架构** 的约束。

- 但在“何时通信、通信什么”的细节上差异很大：
  - **训练**：
    - 每一个 step、每一张卡都算出一份梯度。
    - 需要大量 **AllReduce** 来把各卡梯度求和 / 平均。
    - 通信模式：**频繁、大量、双向**（梯度、参数、激活）。
  - **推理**：
    - 没有反向传播，一般没有梯度同步。
    - 但会有：
      - **权重广播**（初始化时从某处把权重发给所有参与卡）。
      - **KV Cache / 中间结果** 在卡与卡之间的交换（特别是张量并行 / 长序列 / 分层并行的场景）。
    - 通信模式更偏向：
      - 初始阶段的大量权重分发。
      - token-by-token 的中间结果交换。

你可以这么记：

- 训练：**“算完要对齐认知”** → 强依赖 AllReduce。
- 推理：**“一起分工算一条/很多条样本”** → 强依赖权重分发 + 中间结果路由。

从本篇文档的角度，我们关心的是：**无论是训练还是推理，它们都踩在同一套“单机 DMA + 多机 RDMA + 通信库”的地基上。**

理解了这个地基，你看任何训练 / 推理框架，都可以沿着相同思路往下拆。

## 1.4 名词导航：本篇会彻底解释的关键术语一览

为了让你读后面的章节时少“来回翻”，这里先列一遍后文会重点讲清楚的名词，你可以当作导航图：

- **硬件 / 互联相关**
  - `PCIe`：所有卡插在上面的主干总线，支持设备之间 DMA。
  - `GPU DMA Engine`：GPU 内部的硬件搬运单元，负责单机内显存搬运。
  - `NVLink / NVSwitch`：NVIDIA 的 GPU-GPU 高速互联及其交换结构。
  - `HCCS`：昇腾体系的 GPU-GPU / NPU-NPU 高速互联。
  - `超节点（SuperNode）`：把多个板 / 多张卡组成一个更大的高速互联域。
  - `RDMA NIC`：支持 RDMA 的网卡，芯片里有自己的 RDMA Engine。
  - `IB（InfiniBand）`：一整套为 RDMA 设计的独立网络体系。
  - `RoCE`：在以太网上实现 RDMA 的协议族。

- **通信 / 软件栈相关**
  - `RDMA`：Remote Direct Memory Access，远程直接内存访问。
  - `GPUDirect RDMA`：让 RDMA NIC 直接 DMA GPU 显存的机制。
  - `NCCL`：NVIDIA 的多 GPU 通信库。
  - `HCCL`：华为昇腾生态的多 NPU 通信库。
  - `AllReduce / Ring AllReduce / 分层 AllReduce`：多卡同步梯度 / 中间结果的关键算法。
  - `RDMA verbs / ibv_reg_mr / ibv_post_send / CQ`：RDMA 编程中最核心的一组接口和概念。

- **设备 / 平台相关**
  - `Ascend 910B / A3`、`Atlas 800T/800I A2/A3`、`超节点服务器`：昇腾训练 / 推理平台。
  - `310P`、`Atlas 300I Duo`：偏推理侧的 PCIe 插卡方案。
  - `MindIE`、`vLLM-Ascend`：围绕昇腾做推理 / 服务化的上层软件。

后面的章节，会一边沿着“单机多卡 → 多机多卡 → 训推流程”的顺序讲，一边在合适的地方把这些名词一个个“拆开讲透”，而不是让你背一堆定义。

# 2. 硬件角色总览：谁负责算？谁负责搬？

这一章的核心，是帮你把几个关键硬件的“分工”想清楚：

- **谁负责算（Compute）？**
- **谁负责搬数据（DMA / RDMA）？**
- **谁只是“高速通道”（PCIe / NVLink / HCCS / 网络）？**

一旦这张“角色分工表”清楚了，你在脑子里走数据流的时候就不会乱。

## 2.1 GPU / NPU 是什么：算力芯片的基本结构

不管是 NVIDIA 的 GPU，还是昇腾的 NPU，本质结构都可以粗略抽象成三块：

- **计算核心（Compute Cores）**
  - NVIDIA：SM（Streaming Multiprocessor）、Tensor Cores。
  - 昇腾：AI Core、Vector Core 等。
  - 干的事情：做矩阵乘加、卷积、非线性激活等核心算子。
- **片上 / 片外显存子系统（Memory Subsystem）**
  - 典型是 HBM（高带宽内存），或者 GDDR。
  - 决定了“**一次能喂给核心多少数据**”。
- **外部接口（I/O Subsystem）**
  - 和主机 / 其它卡交互的接口：
    - PCIe 控制器。
    - 高速互联控制器（NVLink / HCCS 等）。
  - 决定了“**能和外部世界多快地交换数据**”。

一个很重要的点：

> GPU/NPU 自己并不会“发网络包”，也不会用 TCP/IP。
>
> 它对外的“语言”是：PCIe、NVLink/HCCS 这样的总线协议 + 自己内部的 DMA 指令。

所以当我们说“多机多卡”，真正把 GPU/NPU 连在一起的是：**主板 + 总线 + 网卡 + 交换机**，而不是 GPU 本身直接在说网络协议。

## 2.2 显存、HBM、Device Buffer：数据到底放在哪？

你可以先记一个简单划分：

- **CPU 内存（Host Memory）**：
  - 普通的系统内存（DDR），由 CPU 直接访问。
  - 适合：数据加载、预处理、和磁盘 / 网络打交道。
- **设备显存（Device Memory / GPU Memory / HBM）**：
  - 挂在 GPU/NPU 上的那块高带宽内存。
  - 存的是：权重、激活、中间张量、KV Cache 等。
  - 只有 GPU/NPU 和通过总线 DMA 的设备（比如 NIC）可以直接访问。
- **Device Buffer（设备缓冲区）**：
  - 实际上就是“在设备显存里分出来的一块区域”，用于：
    - 存一段要 AllReduce 的梯度。
    - 存一段要发送给远端卡的激活。
    - 存一组从远端卡收到的中间结果。

从数据流的角度看，最关键的路径是：

- **单机内部**：
  - GPU0 显存 ↔ GPU1 显存（PCIe P2P / NVLink / HCCS）。
- **多机之间**：
  - 本机 GPU 显存 ↔ NIC ↔ 远端 NIC ↔ 远端 GPU 显存（GPUDirect RDMA）。

后面所有的 DMA / RDMA 讨论，其实就是在讲：**如何高效地在这些 Device Buffer 之间挪动数据**。

## 2.3 PCIe 总线：所有设备的“主干公路”

在一台服务器内部，你可以把 **PCIe** 理解成所有扩展卡共享的一张“主干公路”：

- GPU、NIC、NVMe SSD……都插在这条“公路”上。
- 每个设备都有自己的 **PCIe 控制器**，可以发起或响应数据传输。
- 只要主板 / BIOS / IOMMU 允许，**一个 PCIe 设备可以直接访问另一个 PCIe 设备的内存**，这就叫：
  - **PCIe Peer-to-Peer（P2P）访问 / DMA**。

对单机多卡来说，PCIe 至少有三个关键作用：

1. **GPU 与 CPU 之间的数据传输**
   - 比如：从 CPU 内存把 batch 数据拷到 GPU 显存。
2. **GPU 与 GPU 之间的 P2P 访问**
   - 比如：GPU0 直接 DMA 把一块显存写到 GPU1 显存。
3. **GPU 与 NIC 之间的访问**
   - 比如：NIC 通过 PCIe DMA 直接读取 GPU 显存（GPUDirect RDMA）。

因此，你可以记一句：

> 在单机内部，**PCIe 是所有“设备间 DMA”的基础设施**。

NVLink / HCCS 只是“在 PCIe 之外，再拉一条更粗的专用通道”，但 PCIe 这条主干公路是永远存在的。

## 2.4 GPU 内部的 DMA Engine：硬件级搬运工

之前的对话里你已经敏感地抓到了这个词：**GPU DMA Engine**。它非常关键，也很容易被文档一笔带过。

你可以把它理解成：

- GPU 里内建的一个（或多个）**Copy Engine / DMA Engine**。
- 它的职责是：**在不同地址之间搬数据，而不占用计算核心（SM/AI Core）**。

典型用途包括：

- 在同一张 GPU 上：
  - 不同 buffer 之间的 copy（比如 `cudaMemcpyAsync`）。
- 在不同 GPU 之间（单机内）：
  - 通过 PCIe P2P / NVLink 直接把 GPU0 的一块显存写到 GPU1。

注意两点：

1. **DMA 是硬件行为，不是 CPU 在 for 循环 memcpy。**
   - 一旦 DMA Engine 被配置好，它就会自己在总线上发起读 / 写事务，CPU 只需要等“完成通知”。
2. **在单机多卡 P2P 场景下，通常是“源 GPU 的 DMA Engine 发起读 / 写”，目标 GPU 被动接收。**
   - 也就是说：GPU0 负责发起 DMA，把数据推送到 GPU1 的显存里。

后面讲单机多卡 Ring AllReduce 时，你会多次看到这句话：

> “GPU0 DMA Engine → PCIe/NVLink → GPU1 显存 → 完成通知 NCCL”  

真正干体力活的，就是这个 DMA Engine。

## 2.5 NIC / 网卡：普通网卡 vs RDMA 网卡

在多机多卡的世界里，**NIC（网卡）**是 GPU 能“接触到远端 GPU 显存”的唯一桥梁。

要分清两类：

- **普通以太网网卡**
  - 典型用途：上网、REST API、SSH、RPC 等。
  - 模式：基于 TCP/IP，收到包之后：
    - 写到主机内存（CPU 参与）。
    - 由内核 / 用户态进程再处理、拷贝。
  - 对大规模训练来说：**延迟高、CPU 开销大，几乎不能胜任多机同步梯度的需求。**

- **RDMA 网卡（IB NIC / RoCE NIC）**
  - 网卡内部集成了 **RDMA Engine**：
    - 可以直接通过 PCIe DMA 访问本机 GPU 显存。
    - 可以根据 rkey 等信息，DMA 写入远端机器的 GPU 显存。
  - 支持 **RDMA verbs** API（`ibv_reg_mr`、`ibv_post_send` 等）：
    - 应用层 / NCCL 通过 verbs 告诉网卡：
      - 应该从哪段显存读。
      - 发给哪台机器 / 哪个 QP。
      - 写入远端的哪段显存（通过 rkey + 地址）。
  - 对于多机多卡：
    - 它是**真正跨机搬 GPU 显存的执行者**。

你可以这么记：

- 单机内部：**GPU DMA Engine 搬**（走 PCIe/NVLink）。
- 多机之间：**NIC RDMA Engine 搬**（走 IB/RoCE 网络）。

CPU 在两种情况下都不直接搬运数据，只是：

- 通过 NCCL/HCCL / RDMA verbs 下发命令。
- 在 CQ（Completion Queue）上等“搬完了”的通知。

## 2.6 一张“谁可以直接访问谁的内存/显存”的关系矩阵

为了把上面的角色关系彻底固定在脑子里，我们用一张“谁能直接 DMA 谁”的文字矩阵来总结（这里只讨论**硬件上直接 DMA 的可能性**，不讨论权限 / 安全等细节）：

- **GPU → 自己的显存**
  - ✅ 完全可以，DMA Engine 最基本的功能。
- **GPU → 同机内另一张 GPU 的显存**
  - ✅ 可以，前提是：
    - 主板 / BIOS / IOMMU 支持 PCIe P2P。
    - 或者有 NVLink / HCCS 这样的专用互联。
  - 典型用法：单机多卡 P2P 复制、单机 Ring AllReduce。
- **GPU → 远端机器 GPU 的显存**
  - ❌ 不能直接访问。
  - 需要通过：
    - GPU 显存 ←→ 本机 NIC（PCIe DMA）。
    - 本机 NIC ←→ 远端 NIC（IB/RoCE）。
    - 远端 NIC ←→ 远端 GPU 显存（PCIe DMA）。

- **NIC（RDMA）→ 本机 GPU 显存**
  - ✅ 可以，通过 `ibv_reg_mr` 注册后，NIC 的 RDMA Engine 可以直接 DMA 访问这段显存。
  - 典型用法：GPUDirect RDMA。
- **NIC（RDMA）→ 远端 GPU 显存**
  - ✅ 可以，前提是：
    - 远端也完成了 memory region 注册，拥有 rkey。
    - 网络连通（IB / RoCE）。
  - 典型用法：多机多卡 AllReduce、参数广播、KV Cache 交换等。

- **CPU → GPU 显存**
  - ✅ 可以，但速度相对“GPU DMA / RDMA NIC DMA”来说慢。
  - 典型用法：从主机内存把 batch 送到 GPU；从 GPU 拉日志 / 中间结果。
  - 在大规模训练的关键路径里，通常会尽量避免让 CPU 参与大规模数据搬运。

如果把这一节压缩成一句话，就是：

> **单机内**：GPU 自己（DMA Engine）可以直接和其它 GPU 的显存打交道。
>
> **多机间**：只有 RDMA 网卡可以跨机器直接打到远端 GPU 显存，中间不会绕 CPU。  

后面所有关于“单机多卡内部 DMA 细节”“机间 RDMA 流程和细节”“NCCL 调 NIC 读写显存”的内容，都是在这张“谁能 DMA 谁”的关系图上，沿着时间轴展开时序。接下来，我们就先从“单机内的高速互联”开始，把 PCIe / NVLink / HCCS / 超节点讲清楚。

# 3. 单机内部的高速互联：PCIe / NVLink / NVSwitch / HCCS / 超节点

第 2 章讲的是“谁负责算 / 谁负责搬”，现在我们换个视角看：  

> 在一台机器内部，多张 GPU/NPU 之间的**“路”**到底长什么样？  

同样是“多卡”，有的机器只是“普通 8 卡服务器”，有的机器却被称为“超节点”。

本章就是要把这些术语（PCIe / NVLink / NVSwitch / HCCS / 超节点）讲清楚，让你能**从拓扑图和硬件结构的角度**理解单机多卡的性能差异。

## 3.1 PCIe：基础但不够快的“公共道路”

先从所有机器都有的那条“老路”说起：**PCIe（Peripheral Component Interconnect Express）**。

你可以把 PCIe 想象成：

- 一条连接 CPU、GPU、NIC、NVMe SSD 等所有外设的“主干公路”。
- 每个设备在这条公路上都有“车道”（通道，x4 / x8 / x16 等）。
- 车流可以从某个设备流向另一个设备，前提是：
  - 主板 / BIOS / IOMMU 允许。
  - 设备具备 DMA 能力（GPU DMA Engine / NIC DMA Engine）。

在单机多卡场景中，PCIe 有几个关键作用：

1. **CPU ↔ GPU 数据通路**
   - 把 batch 数据从 CPU 内存送到 GPU 显存。
   - 把日志 / 结果从 GPU 拷回 CPU。
2. **GPU ↔ GPU P2P 数据通路**
   - 在支持 P2P 的主板上，GPU0 可以通过 PCIe 直接 DMA 到 GPU1。
3. **GPU ↔ NIC 通路**
   - GPUDirect RDMA 中，NIC 需要通过 PCIe DMA 读写 GPU 显存。

你之前问过一个非常关键的问题：

> “单机内部互传，需要 NIC 的 RDMA 硬件吗？还是 GPU 自己就能通过 PCIe 把数据发给别的 GPU？”  

答案是：

- **单机内部**：
  - 搬运主体是 **GPU 自己的 DMA Engine**（或 runtime 触发的 GPU copy engine）。
  - 通过 PCIe P2P 或 NVLink/HCCS 把数据发给其他 GPU。
  - NIC 的 RDMA Engine 不参与。

从带宽角度看：

- 每条 PCIe x16（Gen4）理论带宽大约是 32 GB/s（双向），Gen5 大约是 64 GB/s。
- 对 AllReduce 这种几十 GB 级别的数据同步来说：
  - PCIe 是能用的，但如果只有 PCIe，**很快就会变成瓶颈**。
  - 这也是为什么后来要加 NVLink / HCCS 等更粗的专用通道。

小结：

- PCIe 是所有单机设备的“共同母语”和最基础的互联方式。
- 无论有没有 NVLink/HCCS，PCIe 一定存在。
- 单机 P2P、GPUDirect RDMA 这些能力，本质上都是在“PCIe 能做设备间 DMA”这个前提上叠出来的。

## 3.2 NVLink：在 GPU 之间拉的“专用高速直连线”

仅靠 PCIe，带宽和拓扑都有限：

- 带宽有限：单链路带宽不够大。

- 拓扑限制：
  - GPU 之间要么共用上层 PCIe Switch，要么通过 CPU Root 复杂回路。
  - 无法轻易做到“任意两卡单跳高带宽互联”。

于是 NVIDIA 设计了 **NVLink**：

- 一种专门为 GPU ↔ GPU / GPU ↔ CPU 设计的**高速点对点链路**。
- 每条 NVLink 的带宽远高于单条 PCIe 通道。
- 一张 GPU 上可以有多条 NVLink 端口，连接到其它 GPU 或 NVSwitch。

你可以把 NVLink 想象成：

- 在两张 GPU 之间，拉了一根（或者多根）粗得多的“专用高速网线”。
- 它不是用来替代 PCIe 的，而是：
  - 在 GPU 之间建立一个比 PCIe 更快的“旁路专线”。

数据流（单机内）：

```text
GPU0 显存
  └─(GPU0 DMA Engine 发起)→ 通过 NVLink → GPU1 显存
```

对上层的 NCCL 来说：

- 它只知道“GPU0 可以高效地给 GPU1 发 P2P 数据”。
- 底层 runtime / 驱动会自动选择：
  - 如果两卡之间有 NVLink，就优先走 NVLink。
  - 否则退回 PCIe P2P。

NVLink 解决的问题是：

- **单机内邻近 GPU 之间的点对点带宽和延迟**。

还没解决的问题是：

- 如果有 8 / 16 张 GPU，要做到“任意两卡之间都能单跳高带宽连接”，光靠点对点拉线会非常复杂。

这就引出了 NVSwitch 和“交换结构”的概念。

## 3.3 NVSwitch：给 NVLink 用的“交换机”，让多卡全互联

**NVSwitch** 可以理解为：

> “专门服务于 NVLink 的高速交换芯片”。  

它的工作方式类似于：

- 每张 GPU 把自己的多条 NVLink 接到 NVSwitch 上。
- NVSwitch 负责：
  - 把来自一张 GPU 的数据转发到目标 GPU 所在的 NVLink 端口。
  - 提供一个近似“全互联”的拓扑。

拓扑示意（简化文字版）：

```text
       [ NVSwitch Fabric ]
       /   /   |   |   \   \
    GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 ...
```

在这样的结构里：

- 任意两张 GPU 之间的通信可以在 1~2 跳内完成。
- 带宽对称，延迟接近常数。
- 对上层算法来说，整个 NVSwitch 域就像一个“完全互联的多 GPU 高速域”。

这对大模型训练为什么关键？

- AllReduce / 张量并行中的通信模式：
  - 频繁。
  - 数据量大。
  - 需要任意两卡之间都能高效交换数据。
- 有了 NVSwitch：
  - AllReduce 中的每一轮 “GPU_i → GPU_j” 都可以在一个高带宽、低延迟、拓扑对称的环境中执行。

对比：

- **只有 PCIe / 点对点 NVLink 时**：
  - 有些卡之间的通信可能需要多跳（经过中间卡或上层 Switch）。
  - 带宽和延迟会不均匀，AllReduce 算法要为此做更多适配。
- **有 NVSwitch 时**：
  - 可以近似认为所有卡之间的通信条件相同。
  - 算法实现更简单，性能也更接近理论上限。
  - 后面在讲“单机 8 卡 Ring AllReduce”时，你可以直接把整个 NVSwitch 域当成一个“完全互联的黑盒”来思考：任意两卡之间都能以近似恒定的成本互相发送 chunk。

简单说：

- NVLink 是“粗的线”。
- NVSwitch 是“把这些粗线织成一个大网的交换芯片”。

## 3.4 昇腾 HCCS：Ascend 体系下的 NVLink 类比

在华为昇腾体系里，没有 NVLink 这个名字，但有非常类似的东西：**HCCS（High-speed Cache Coherent System）**。

从功能上看，HCCS 承担的角色与 NVLink 类似：

- 提供 NPU / GPU 之间的高速互联。
- 组成单机或板级的多卡高速域。
- 为 AllReduce、张量并行、流水并行提供高带宽通路。

典型结构：

- 一台 Atlas 800T A2 / A3 内部：
  - 多颗 Ascend 910B/A3 通过 HCCS 互联。
  - 形成一个类似 “单机 8 卡高速域” 的结构。
- 在 A3 超节点中：
  - 多个计算板通过更大规模的 HCCS / 板级互联结构连接在一起。
  - 把高速域从“机箱级”扩展到了“多板级”。

对于 HCCL 来说：

- HCCS 域内的卡可以被看作一个“大一体化高速域”。
- 在这个高速域内做 AllReduce / Broadcast 时：
  - 数据走的是 HCCS，而不是普通 PCIe 或以太网。

和 NVLink/NVSwitch 对比：

- NVLink + NVSwitch：
  - 在 NVIDIA GPU 之间提供高速互联。
- HCCS + 超节点互联：
  - 在 Ascend NPU/GPU 之间提供高速互联。

概念完全可以一一对应，只是名字和实现细节不同。

## 3.5 什么是“超节点（SuperNode）”：高速互联域的边界从 8 卡扩到 16 / 32 卡

你有一个反复纠结的问题：  

> “普通 8 卡服务器 vs 超节点，到底区别在哪里？是不是只是卡多了？”  

关键不是“卡多”，而是：

> **高速互联域（High-speed domain）的物理边界被扩展到了更大的范围。**

用更具体的话来说：

- 普通 8 卡服务器：
  - 高速互联主要局限于：
    - 机内 8 卡之间（通过 HCCS / NVLink / NVSwitch / PCIe）。
  - 一旦出机（到另一台服务器）：
    - 就要走 RoCE / IB 等网络，延迟和带宽都明显下降。
  - → 高速域的“边界” ≈ 机箱。

- 超节点（以 Atlas 800T A3 为例）：
  - 不止机内 8 卡通过 HCCS 互联。
  - 多个板 / 多个节点之间也通过高速板级互联连成一个更大的统一高速域。
  - 在这个域内：
    - 16 / 32 甚至更多张卡之间的通信都可以走 HCCS 等高带宽通路。
  - 只有超过这个规模（再往外扩到其它机柜 / 集群）时，才需要走 RoCE/IB。
  - → 高速域的“边界” > 单机，可以是多板 / 多机箱级别。

从 AllReduce / 并行训练的角度看：

- 高速域内：
  - 可以放心做大规模张量并行、AllReduce，通信成本可控。
- 高速域外：
  - 必须分层（Hierarchical AllReduce / 分层并行）。
  - 需要更多“跨机网络优化”的技巧。

这就是“超节点”的真正含义：  
不是“8 卡服务器的升级版”，而是“**把高速互联域这条红线画得比机箱更大**”的系统级设计。

## 3.6 普通 8 卡服务器 vs 超节点：本质差异不在卡数，而在高速域大小

最后，用一张文字对比图，把你一开始困惑的点彻底收束一下。

```text
【普通 8 卡训练服务器】（以 Atlas 800T A2 为例）

机内：
  - 8 × 910B
  - 通过 HCCS / PCIe / 板级连接形成一个 8 卡高速域

机间：
  - 服务器之间通过 RoCE / IB 相连
  - Any: [8卡节点] ──(RoCE/IB)── [8卡节点]

高速域边界：
  - 基本等于“机箱”
```

```text
【超节点】（以 Atlas 800T A3 超节点为例）

机内 + 板级：
  - 多个计算板（每板多卡）通过 HCCS / 光模块高速互联
  - 形成一个 16 / 32 卡级别的统一高速域

机间：
  - 超节点 ↔ 超节点 之间通过 RoCE / IB 相连
  - Any: [16/32卡高速域] ──(RoCE/IB)── [16/32卡高速域]

高速域边界：
  - > 单机，可以覆盖多个板 / 多个机箱
```

对通信 / 并行策略的直接影响是：

- 在普通 8 卡服务器上：
  - 你可以把“单机 8 卡”当成一个高速域，在其内做 AllReduce / 张量并行。
  - 再往外扩（多机），就必须分层。
- 在超节点上：
  - 你可以把“16/32 卡高速域”当成一个更大的单节点。
  - 很多原本跨机的 TP/AllReduce，现在都可以在一个统一高速域内解决，减少跨 RoCE/IB 的开销。

用一句话概括这一节：

> 普通 8 卡服务器和超节点的根本区别，不是“卡多了”，而是“**单个高速互联域的规模被拉大了**”。  
>
> 这个规模直接决定了：你能在多大范围内“假装自己只有一台超大 GPU”，不需要被跨机网络的带宽和延迟拖后腿。 

# 4. 跨机互联与 RDMA：IB / RoCE / 以太网到底是啥关系？

这一章是“多机多卡”的底层基础，如果这一块模糊，后面看到 IB / RoCE / RDMA / GPUDirect RDMA 时就会很容易乱。

我们的目标是：

- 搞清楚：**以太网 vs IB vs RoCE** 各自是什么、有什么区别。
- 搞清楚：**RDMA 是什么，它和 GPU/NPU 有什么关系**。
- 搞清楚：**GPUDirect RDMA 到底比“普通 RDMA + CPU 内存中转”多了哪一步**。

## 4.1 以太网是什么：日常“上网”的那张网

你每天用的：

- 公司内网。
- 家里的光猫 + 路由器。
- 数据中心里普通业务服务器之间的网络。

本质上几乎全是：**以太网（Ethernet）**。

以太网的几个特点：

- 有统一的帧格式（以太网帧）。
- 上层跑的是 IP / TCP / UDP 等协议。
- 使用“**尽力而为（Best Effort）**”的传输模型：
  - 允许丢包。
  - 丢了再靠上层（TCP）重传。
- 面向的是：
  - Web 服务、数据库访问、RPC 调用、SSH 远程登录等普通业务流量。

对于普通应用，这很好：

- 实现简单。
- 可以用交换机 / 路由器轻松扩展到很大规模。

但对大模型训练 / 推理中的“**每秒几十 GB、几百 GB 的梯度或激活**”来说，普通以太网有两个致命问题：

1. **延迟高且抖动大**：
   - 中间要经过内核协议栈、TCP 拥塞控制等。
   - 路上丢包后需要重传，延迟飙升。
2. **CPU 要参与每次收发**：
   - 数据先到 NIC，然后被拷贝到主机内存，再由用户态进程处理。
   - CPU 需要处理协议栈、做 memcpy，对大规模 AllReduce 来说，很快会成为瓶颈。

因此，**只靠“普通以太网 + TCP”是不足以支撑高效的多机多卡训练的**。

## 4.2 RDMA 是什么：远程直接内存访问，为什么深度学习离不开它

**RDMA（Remote Direct Memory Access）** 这四个词可以拆开看：

- Remote：远程机器（另一台服务器）。
- Direct：直接（跳过某些中间环节）。
- Memory Access：访问内存（或显存）。

组合起来的意思是：

> 一台服务器可以 **直接读写另一台服务器的内存**，  
> 并且 **不需要对方 CPU 参与收/发处理**。

和“普通 TCP 收发”的对比：

- 没有 RDMA 时（传统 TCP）：

  ```text
  A 进程 → A 内核 → A NIC → 网络 → B NIC → B 内核 → B 进程
  ```

  - CPU 需要参与：
    - 协议栈处理（TCP/IP）。
    - 内核态 / 用户态的数据拷贝。

- 有了 RDMA 之后：

  ```text
  A NIC → 网络 → B NIC → B 内存（直接 DMA 写入）
  ```

  - 对方 CPU：
    - 可以完全不知道这次写入的存在（当然应用层会约好哪些内存可以被写）。
  - 对方内核：
    - 不参与收发，只是在初始化时设置好 RDMA 权限和映射。

**对大模型训练/推理有什么影响？**

- 训练时：
  - 梯度 AllReduce 可能每步要同步几十 GB 的数据。
  - 如果这些数据每次都要经过对方 CPU 和内核栈，CPU 会直接被压垮。
  - RDMA 让这些数据交换走“**纯硬件路径**”，CPU 只在初始配置和收“完成通知”时参与。

- 推理时：
  - 初始化时需要把大模型权重分发到所有节点上。
  - 如果用 RDMA，可以大幅减少 CPU 开销并提高带宽利用率。

你可以把 RDMA 想象成：

> 在两台服务器之间，挖了一条“绕过 CPU / 内核栈的专用数据通道”，  
> NIC 之间直接商量好“我往你内存的哪一段写多少字节”。
>
> 所以当你看到“**AllReduce 用 RDMA**”这句话时，脑子里应该自动浮现的是：**GPU 显存里的梯度不是一跳一跳通过 TCP+CPU 转发，而是由两端的 RDMA 网卡直接搬进搬出远端内存/显存**。

## 4.3 IB（InfiniBand）：为 RDMA 专门造的“高铁系统”

**InfiniBand（IB）** 是一种专门为高性能计算设计的网络体系，它不是以太网的一个变体，而是一个**平行宇宙**：

- 有自己的：
  - 物理层 / 链路层协议。
  - 交换机（IB Switch）。
  - 网卡（IB HCA）。
  - 地址 / 路由机制（LID / GID 等）。
- 原生支持 RDMA：
  - 每个连接（Queue Pair）都有 RDMA 语义。
  - 网络默认是 **无丢包** 的（通过信用流控 / 链路级机制保障）。

你可以这么理解：

- 以太网：
  - 像一个为**各种车**（摩托、轿车、卡车）设计的通用公路系统。
  - 你可以往上跑 Web、数据库、文件服务、IPC……
- InfiniBand：
  - 像是为**高速列车**单独修的一套高铁轨道系统。
  - 它的目标就是：极低延迟、极高带宽、稳定可控。

在大模型训练领域：

- IBM BlueGene、很多超算中心、NVIDIA DGX SuperPOD 等大规模训练集群，常常使用 IB 作为集群内部主网络。

**优点**：

- 延迟极低（亚微秒级）。
- 原生 RDMA、无丢包、拥塞控制成熟。
- 对大规模 AllReduce 极其友好。

**代价**：

- 需要部署一套独立的网络（专用交换机 / 网卡）。
- 成本和运维门槛相对更高。

## 4.4 RoCE：在以太网上实现 RDMA 的技术路线

**RoCE（RDMA over Converged Ethernet）** 的含义是：

> 在现有的以太网体系上，实现 RDMA 能力。

也就是说：

- 物理上仍然是以太网：
  - 以太网交换机。
  - 以太网线缆 / 光纤。
- 协议栈加入 RDMA 支持：
  - RoCE v1：跑在以太网帧之上。
  - RoCE v2：跑在 UDP/IP 之上。
- 网卡是支持 RoCE 的 RDMA NIC。

**为什么要搞 RoCE？**

- 很多数据中心已经有非常成熟的大规模以太网基础设施。
- 重新铺一套 IB 网络成本高、改造大。
- 如果能在以太网上“开辟出一条 RDMA 专用车道”，就可以复用大量已有设备和经验。

和 IB 相比：

- 出身不同：
  - IB：为 RDMA 从零设计的专用网络。
  - RoCE：在“原本允许丢包”的以太网上，通过额外机制构造“尽可能无丢包”的 RDMA 通道。

- 实现无丢包的方式不同：
  - IB：协议层内建信用流控，链路本身是无丢包的。
  - RoCE：需要在以太网上启用：
    - PFC（Priority Flow Control）。
    - ECN（Explicit Congestion Notification）。
    - DCQCN 等拥塞控制机制。
  - 配不好可能出现：死锁 / 暴涨的延迟 / 丢包导致性能抖动。

在昇腾 / Atlas 生态里：

- 机器之间常见的是：**RoCE 网络**。
  - Atlas 800T A2 / A3 等训练服务器之间。
  - 超节点集群的跨机互联。

## 4.5 IB vs RoCE：出身、协议、延迟与稳定性的差异

用一个表把它们的差异压缩一下：

```text
维度          IB（InfiniBand）                RoCE（以太网上的 RDMA）
---------------------------------------------------------------
网络体系      完整独立体系                    依托以太网体系
物理设备      IB 交换机 / IB NIC             以太网交换机 / RoCE NIC
是否原生 RDMA 是                              是（协议扩展实现）
默认是否无丢包 是（信用流控）                 否（需 PFC/ECN 等配置）
延迟稳定性    更稳定、抖动小                  依赖网络配置，易抖动
部署成本      高，需专门网络                  较低，可复用以太网
典型场景      超算 / DGX SuperPOD            通用数据中心 / AI 集群
```

对多机多卡训练来说：

- 在 **几十 / 上百节点规模** 时：
  - 两者都能工作。
  - IB 通常延迟更小、更稳定。
  - RoCE 如果调得好，性能也可以很接近 IB。

- 在 **大规模 AllReduce** 中：
  - 最怕的是 **延迟抖动** 和 **丢包重传**。
  - IB 在这两点上天然更有优势。
  - RoCE 则依赖非常好的网络调优。

换句话说：

- 从“理论链路形态”的角度，两者都能实现：`GPU 显存 ↔ NIC ↔ 网络 ↔ NIC ↔ GPU 显存` 的 RDMA。
- 从“工程实践”的角度，IB 更偏向极致性能场景，RoCE 更偏向综合成本/灵活性。

## 4.6 RoCE 是否和普通上网流量混用？独立组网是怎么回事？

你之前提过一个很好的直觉问题：

> “RoCE 既然跑在以太网上，那是不是就和普通上网流量混在一起了？是不是一根网线既在上 Web，又在跑 RDMA？”  

从“协议允许”的角度：

- 一个以太网交换机 / 链路上，确实可以同时承载：
  - 普通 TCP/IP 业务流量。
  - RoCE 的 RDMA 流量。

但从“工程实践”的角度：

- 高性能训练 / 推理集群通常会：
  - 给 RoCE 流量单独划 VLAN 或优先级。
  - 使用专门的 ToR（Top-of-Rack）交换机、独立的物理端口。
  - 甚至物理上独立一套“AI 训练网络”，和业务网络分离。

这是因为：

- RoCE 对网络质量的要求非常苛刻：
  - **严禁拥塞造成的大规模丢包**。
  - 需要对 PFC/ECN/DCQCN 等进行精细配置。
  - 业务流量的突发可能会破坏 RDMA 流量的稳定性。

所以更准确的说法是：

- **技术上**：RoCE 是以太网上的一种 RDMA 协议。
- **部署上**：在 AI 训练集群中，RoCE 网络通常会被视为“逻辑乃至物理上相对独立的一张网”。

换成类比：

- 公路上可以划出“公交专用道”。
- 但为了不和其它车混抢车道，有些城市干脆在物理上独立出 BRT 专用通道。

在很多实际 AI 集群里，也会类似地**显式区分“业务以太网”和“RoCE 训练网”**：即便物理上复用同一机架交换机，也会通过 VLAN / 优先级等方式把训练流量和普通业务流量隔离开。

## 4.7 GPUDirect RDMA：网卡如何直接读写 GPU 显存（对比普通 RDMA 与 PCIe P2P）

前面我们讲了：

- **普通 RDMA** 的语义是：
  - NIC 可以直接读写远端的“主机内存”（CPU DRAM）。

在没有 GPUDirect RDMA 之前，如果你想跨机同步 GPU 显存数据，路径通常是：

```text
源机器：
  GPU 显存 →(PCIe DMA)→ CPU 内存
  CPU 内存 →(RDMA)→ 远端 CPU 内存

目标机器：
  远端 CPU 内存 →(PCIe DMA)→ 目标 GPU 显存
```

问题：

- 每次都要：显存→CPU内存→网络→CPU内存→显存。
- 多了一次上 / 下 GPU 的开销。
- CPU / 内存带宽成为瓶颈。

**GPUDirect RDMA** 做的事情就是：

> 让 RDMA NIC 像访问主机内存一样，**直接 DMA 访问 GPU 显存**。

也就是说：

- 源机器：

  ```text
  GPU 显存 →(PCIe DMA by NIC)→ NIC →(RDMA)→ 远端 NIC
  ```

- 目标机器：

  ```text
  远端 NIC →(PCIe DMA by NIC)→ 目标 GPU 显存
  ```

中间的 CPU 完全不参与数据搬运。

对你之前问的那条问题链“**NCCL 调用 NIC 驱动来让 NIC 的 DMA 读取显存数据**”来说，这里就是核心：

1. 应用 / NCCL 通过 `ibv_reg_mr` 把 GPU 显存注册为 RDMA memory region。
2. RDMA 驱动 / IOMMU 建立映射：允许 NIC 对这段 GPU 显存发起 DMA。
3. NCCL 通过 `ibv_post_send` 等 verbs 请求 NIC 发起一次 RDMA Write/Read：
   - NIC **作为 PCIe Master**，对 GPU 显存执行 DMA 读 / 写。
4. NIC 把读到的数据通过 IB/RoCE 网络发给远端 NIC，远端 NIC 再 DMA 写入远端 GPU 显存。

如果再和 **PCIe P2P（单机内 GPU→GPU 直接 DMA）** 对比：

- 单机内：

  ```text
  源 GPU DMA Engine
    ──(PCIe / NVLink / HCCS)→ 目标 GPU 显存
  ```

- 跨机 + GPUDirect RDMA：

  ```text
  源 GPU 显存
    ──(PCIe DMA by NIC A)→ NIC(A)
    ──(IB/RoCE)→ NIC(B)
    ──(PCIe DMA by NIC B)→ 目标 GPU 显存
  ```

两者使用的“直接访问显存”的底层机制是统一的：

- 都依赖：PCIe 总线 + IOMMU 映射。
- 区别只是：发起 DMA 的主体：
  - 单机：源 GPU 的 DMA Engine。
  - 多机：NIC 的 RDMA Engine。

---

到这里，关于“跨机互联与 RDMA：IB / RoCE / 以太网”的底层概念已经铺好了：

你现在可以清楚地说出：

- 以太网是日常用的“公路体系”，RoCE 是在这条公路上划的 RDMA 高速车道。
- IB 是为 RDMA 专门建的“高铁系统”，和以太网是平行世界。
- RDMA 是让 NIC 直接读写远端内存/显存的能力，避免 CPU 参与搬运。
- GPUDirect RDMA 则是让这套能力**直接作用到 GPU 显存**上，跳过 CPU 内存中转。

接下来的第 5/6/7/8/10 章，其实都是这几个核心概念在不同层（算法、单机、多机、通信库、完整流程）上的展开。你随时可以回到这一章，把任何一个名词捞回它真正所属的层和作用范围。 

# 5. AllReduce：只讲到你真正需要的程度

前面几章你已经知道：

- 单机 / 多机之间有各种“路”（PCIe / NVLink / HCCS / IB / RoCE）。
- GPU / NIC 上有各种“搬运工”（GPU DMA Engine / NIC RDMA Engine）。

这一章只补上一个核心“调度动作”——**AllReduce**——的本质含义和你真正需要理解到的程度。我们不会把所有数学细节和所有变体展开成教材，而是聚焦在：

- 它到底在算什么？

- 为什么数据并行训练离不开它？

- Ring AllReduce 在 8 卡场景下大致是怎么“绕圈”的？

- 为什么一旦链路差，AllReduce 就会成为性能瓶颈？

## 5.1 AllReduce 的数学定义与直观例子

先看抽象定义：

> AllReduce(op) = 所有参与者各自提供一段数据，
> 先对这些数据做一次全局归约（reduce），
> 然后把归约结果再分发给所有参与者（all）。

以求和为例（AllReduce(sum)）：

- 假设有 4 张卡，各自有一个标量：
  - GPU0: 2
  - GPU1: 3
  - GPU2: 5
  - GPU3: 10

- 做一次 AllReduce(sum) 后：
  - 每张卡都会得到同样的结果 `2+3+5+10 = 20`：
    - GPU0: 20
    - GPU1: 20
    - GPU2: 20
    - GPU3: 20

如果把每张卡上的数据从“标量”换成“向量 / 张量”，含义完全一样：

- 每张卡有一个同形状的 tensor `x_i`。
- AllReduce(sum) 的结果是：
  - 每张卡拿到 `x_sum = sum_i x_i`。

如果想要平均（average），只需要再除以参与卡数：

$$
x_{\text{avg}} = \frac{1}{N} \sum_{i=1}^N x_i
$$

## 5.2 它在数据并行训练中的角色：同步梯度，让所有卡模型一致

在**数据并行（Data Parallel）**训练中：

- 每张卡有一份完整模型参数 W。
- 每张卡处理不同的子 batch：
  - GPU0 看 `input_0`，GPU1 看 `input_1` ……
- backward 之后，每张卡会算出一份本地梯度：
  - GPU0: `grad_W0`
  - GPU1: `grad_W1`
  - ……

如果此时直接 `optimizer.step()`：

- 每张卡的参数会根据自己的本地梯度更新：
  - `W0 ← W0 - lr * grad_W0`
  - `W1 ← W1 - lr * grad_W1`
  - ……
- **模型很快就会在不同卡上“分裂”成不一致的几份**。

为了让所有卡上的模型保持一致，数据并行训练会在更新参数前做两件事：

1. 把所有卡上的梯度做一次全局求和：

   $$
   \text{grad\_W\_sum} = \sum_i \text{grad\_W\_i}
   $$

2. 用全局平均梯度来更新参数：

   $$
   \text{grad\_W\_avg} = \frac{1}{N} \text{grad\_W\_sum}
   $$
   这两步，正是通过 **AllReduce(sum)** 来完成的：

- AllReduce(sum) 把所有卡上的 `grad_W_i` 聚合成 `grad_W_sum`，并把它分发回所有卡。
- 再除以 N（通常由框架 / 优化器内部处理），得到 `grad_W_avg`。

因此可以说：

> 在数据并行训练中，AllReduce 的角色就是：
>
> **“让所有人对梯度达成一致，再一起更新参数”**。

没有 AllReduce，训练会发散；

AllReduce 越慢，训练越慢。

## 5.3 Ring AllReduce：8 卡环形通信的文字推演

在实际实现中，AllReduce 有多种算法：Ring、Tree、分层（Hierarchical）等等。

这里我们只用一个直觉上比较容易画图的 **Ring AllReduce** 举例，帮助你理解“为什么它如此依赖底层互联带宽/延迟”。

假设：

- 有 8 张卡：GPU0~GPU7。
- 每张卡有一大块梯度 buffer，已经按 8 份（chunk0~chunk7）切分好。
- 我们要对这些 buffer 做一次 AllReduce(sum)。

Ring AllReduce 通常分两阶段：

- **Reduce-Scatter**：边传边加，最后每张卡只留下自己“负责”的那个 chunk 的全局和。
- **All-Gather**：再把这些已经求和好的 chunk 传一圈，让每张卡都拿到全量结果。

我们用“8 卡成环”的文字拓扑来描述：

```text
GPU0 → GPU1 → GPU2 → GPU3 → GPU4 → GPU5 → GPU6 → GPU7 → GPU0
```

**Reduce-Scatter 阶段（直观版本）**

1. **第 1 轮：每张 GPU 把 chunk0 发给右边的 GPU**
   - GPU0 把自己的 chunk0 发给 GPU1。
   - GPU1 把自己的 chunk0 发给 GPU2。
   - ……
   - GPU7 把自己的 chunk0 发给 GPU0。
   - 每张卡在收到 chunk0 后，把它和自己当前持有的 chunk0 相加。

2. **第 2 轮：再次传递累加后的 chunk0**
   - GPU0 把 `(来自 GPU7 的 chunk0 + 自己的)` 发给 GPU1。
   - GPU1 把 `(来自 GPU0 的 chunk0 + 自己的)` 发给 GPU2`……`。
   - 如此往复，总共进行 7 轮。

3. **第 7 轮结束后**
   - 每个 chunk0 会在环上“绕一圈”，在每一跳被累加一次。
   - 最终会全部累加到某一张卡上（具体哪一张由算法安排）。

对 chunk1~chunk7 也是类似，只是它们在环上的起点和归宿不同。

通过巧妙安排，每一轮传输中，每张卡都在同时发送 / 接收不同的 chunk，使得链路始终保持高利用率。

**All-Gather 阶段（直观版本）**

- 当每个 chunk 的全局和已经被放在某一张卡上之后，再做一次“环形广播”：
  - 把这些结果在环上传递一圈。
  - 使每张卡都拿回所有 chunk 的结果。

**和你关心的 DMA / RDMA 有什么关系？**

- 在单机 8 卡场景下：
  - 环上的每一跳 `GPU_i → GPU_{i+1}` 都是一次 **P2P DMA**：
    - 源 GPU 的 DMA Engine 通过 PCIe/NVLink/HCCS 把 chunk 写到目标 GPU 的显存。
    
  - 所以 Ring AllReduce 的性能几乎完全由：
    - GPU DMA Engine 性能。
    
    - 单机互联带宽（PCIe / NVLink / HCCS）。
    
      决定。
  
- 在多机多卡场景下：
  - 环上的某些跳会变成：`(Machine A, GPU_i) →(Machine B, GPU_j)`：
    - 这时搬运主体变成 **NIC 的 RDMA Engine**，路径变成：

      ```text
      源 GPU 显存 →(PCIe DMA by NIC A)→ NIC(A)
                     →(IB/RoCE)→ NIC(B)
                     →(PCIe DMA by NIC B)→ 目标 GPU 显存
      ```

  - 所以 Ring AllReduce 的性能就高度依赖：
    - RDMA NIC 性能。
    - IB/RoCE 网络的带宽和延迟。

你不需要记住所有数学公式，但要牢牢记住这个“绕圈 + 每一跳都是一次 DMA”的直觉。

## 5.4 Tree / Hierarchical AllReduce：为什么要“分层”？

Ring AllReduce 的优点是：

- 实现简单。
- 带宽利用率高（每一条链路几乎都在干活）。

但它有一个明显的缺点：

- **延迟随着参与卡数 N 线性增长（O(N)）**：
  - 因为每个 chunk 要绕环一圈，需要 N-1 跳。

在卡数很大、特别是跨很多机器时：

- 这条“大环”上有很多跨机跳。
- 每一次跨机跳的延迟都叠加上去。
- AllReduce 的总时间变得不可接受。

这时就会用到 **Tree / Hierarchical AllReduce**：

- 思路是：
  - **先在局部（比如机内或机架内）做一次 AllReduce**。
  - 再在更高层（比如机间）的代表节点之间做 AllReduce。
  - 最后再把结果向下广播给局部。

类比：

- Ring：所有人排成一个大圈，钱一个一个往下传，相当于“大家围着地球跑一圈传话”。
- 分层 Tree：
  - 先在每个班级内部传完话（机内）。
  - 选班长去年级开会汇总（机间）。
  - 再让班长回来告诉各自班级。

在你已经理解第 6/7/10 章的前提下，只要记住一句：

> 分层 AllReduce 做的本质是：“尽量在高速域内把事情做完，把跨机 / 跨机柜的那部分压缩到最少”。  

它背后的直觉完全基于你前面理解的“高速域 vs 跨机网络”的差异。

## 5.5 AllReduce 性能为何完全被互联架构“卡住”

把这一章的重点再压缩一下：

1. **AllReduce 本质上是“在所有卡之间来回搬梯度/张量”**
   - 数学上是在做求和 / 求平均。
   - 实现上是在大量做 P2P DMA / RDMA 操作。

2. **每一步 AllReduce 都是同步障碍（barrier）**
   - 所有参与卡都要等这一次 AllReduce 完成，才能继续下一步计算。
   - 只要其中某条链路慢，**整个 AllReduce 都会被拖慢**。

3. **互联架构（单机 + 跨机）就是 AllReduce 的“地基”**
   - 单机：PCIe / NVLink / HCCS / NVSwitch / 超节点。
   - 跨机：IB / RoCE + RDMA NIC + 交换机拓扑。
   - 这些决定了：
     - 每一跳 P2P/ RDMA 的带宽 / 延迟。
     - 能否在更大范围内保持“单高速域”。

4. **AllReduce 算法会围绕互联架构做适配**
   - 小规模 / 单机：Ring 就够好。
   - 大规模 / 多机：需要 Tree / Hierarchical / 分层拓扑，以减少跨慢链路的次数。

所以当你以后看到：

- 某篇论文 / 某个框架在吹“我们优化了 AllReduce 性能”，你可以直接问自己：
  - 它是：
    - 换了算法（Ring → Tree / 分层）？
    - 利用更好的单机高速域（NVSwitch / 超节点）？
    - 或者利用了更好的跨机网络（IB / 调优过的 RoCE）？

而当你看到“某个 GPU 集群训练得很慢”，也可以本能地沿着这条线回溯：

- 是单机内 PCIe/NVLink/HCCS 拓扑限制了 AllReduce？
- 还是跨机的 IB/RoCE 网络让 AllReduce 延迟 / 抖动过大？

理解这一章，不是为了你去自己写 AllReduce 算法，而是：

- 让你在看任何多机多卡训练 / 推理方案时，都能立刻意识到：
  - **互联架构 + AllReduce + 通信库 + DMA/RDMA** 是一整套连在一起的系统。  
  - 任何一个环节设计不好，结果都会在 AllReduce 上“暴露出来”。

# 6. 单机多卡：数据在机内是怎么跑的？

这一章，我们不讲任何“抽象概念”，只回答两个非常具体的问题：

- 在一台机器里，**GPU0 把一块数据给 GPU1**，底下到底发生了什么？
- 做一次 **单机 8 卡的 Ring AllReduce**，数据是怎么一跳一跳绕一圈的？

你可以把这章当成是对前面“GPU DMA Engine / PCIe / NVLink”那张角色图的**时序展开版**。

## 6.1 单机 GPU 拓扑：PCIe 插槽、NVLink 拓扑的直觉图

现实中的 4 卡 / 8 卡服务器内部，大致会长成下面这种结构（文字示意）：

```text
CPU / 内存
   │
PCIe Root Complex
   ├── GPU0
   ├── GPU1
   ├── GPU2
   └── GPU3
```

如果是带 NVLink / HCCS 的机器，还会有一层“GPU 之间的直接连线”，例如：

```text
GPU0 ── NVLink ── GPU1
  │                  │
 NVLink           NVLink
  │                  │
GPU2 ── NVLink ── GPU3
```

你可以这么理解：

- **PCIe**：所有卡共享的“主干公路”，无论有没有 NVLink，都一定有 PCIe。
- **NVLink/HCCS**：在部分 GPU 之间额外拉的“专用高速直连线”。

后面我们说“单机 P2P 复制”，本质就是：

- 源 GPU 的 DMA Engine 发起一个“把我这块显存搬到对面那块显存”的 DMA 操作。
- 真实走的通路可能是：
  - 只有 PCIe：`GPU0 → PCIe → GPU1`。
  - 同时有 NVLink：`GPU0 → NVLink → GPU1`（更快）。

## 6.2 GPU DMA Engine 的角色：单机内部谁在发起 DMA？

先把一个核心结论钉死：

> 在单机内部，**源 GPU 的 DMA Engine 是主动发起者**，
>
> 目标 GPU 只是被动接收写入的那一方。

更具体一点：

- 当 NCCL 决定“现在该把 GPU0 的某个 chunk 发给 GPU1”时，它会：
  - 通过 CUDA/HIP/CANN 等接口，触发 **GPU0 的 DMA Engine**：
    - 从 `GPU0: buffer_src` 读数据。
    - 写到 `GPU1: buffer_dst`。
- 支持 P2P 时，这个 DMA 请求会被翻译成：
  - 对 GPU0 自己显存的读事务。
  - 对 GPU1 显存的写事务。
  - 中间通过 PCIe / NVLink 物理链路搬运。

所以，这一小节的重点是：

- **不是 GPU1 主动去拉 GPU0 的数据。**
- 而是 **GPU0 自己的 DMA Engine 把数据“推”给 GPU1。**

GPU1 需要做的事只有：

- 确保目标 buffer 已经分配并准备好接收数据。
- 在 DMA 完成后，通知上层（例如 NCCL）“这个 chunk 已经写完，可以用来做下一步计算 / 累加”。

## 6.3 PCIe P2P：GPU0 通过 PCIe 直接写入 GPU1 显存的完整路径

我们现在只看最基础的场景：**没有 NVLink / HCCS，只有 PCIe P2P**。

目标：GPU0 把一块 16MB 的梯度 chunk 发给 GPU1。

完整路径可以拆成下面几步：

1. **NCCL 决策 & 调用**
   - NCCL 的 Ring AllReduce 调度器判断：
     - 当前轮到 GPU0 把 chunk0 发给 GPU1。
   - NCCL 调用底层接口（例如基于 CUDA 的 P2P copy）：
     - “从 GPU0:addr\_src 拷贝 16MB 到 GPU1:addr\_dst”。

2. **GPU0 DMA Engine 配置**
   - GPU 驱动接到指令后，在 GPU0 上配置 DMA Engine：
     - 源地址：GPU0 显存中的 `addr_src`。
     - 目标地址：GPU1 显存对应的 PCIe 可寻址地址（通过 P2P 映射得到）。
     - 长度：16MB。

3. **发起 PCIe 读 / 写事务**
   - GPU0 DMA Engine 开始工作：
     - 对 GPU0 自己的显存发起读请求。
     - 把读出的数据打包成 PCIe TLP（事务层包）。
     - 把这些包通过 PCIe Root Complex / Switch，路由到 GPU1 的 PCIe 端口。

4. **GPU1 接收并写入显存**
   - GPU1 的 PCIe 控制器收到这些写事务：
     - 确认目标地址是合法映射到自己显存的区域。
     - 把数据写入对应的显存行。

5. **完成通知**
   - 当 DMA Engine 检测到所有数据都已经成功写入：
     - 在 GPU0 一侧产生一个“DMA 完成”事件。
     - 驱动 / 运行时（CUDA / CANN）会把这个完成事件反馈给 NCCL。
   - NCCL 知道这一个 chunk 已经在 GPU1 那边就位，接下来可以：
     - 在 GPU1 上对这个 chunk 做累加。
     - 或者继续把它发给下一个 GPU（在 Ring 里）。

用极简文字图总结就是：

```text
NCCL 调度 → GPU0 DMA Engine 配置好一次 P2P 复制
GPU0 显存 ──(PCIe 读/写 TLP)──> GPU1 显存
DMA 完成 → 通知 NCCL / 上层
```

整个过程中：

- **CPU 不搬数据，只发起 /等待完成。**
- **PCIe 是纯通道，不做任何“智能”决策。**

## 6.4 旧式“显存→CPU内存→显存”路径 vs 现代 P2P：为什么后者更关键？

在没有 P2P / NVLink 的老系统里，GPU0 → GPU1 可能只能这样搬：

```text
GPU0 显存 → CPU 内存 → GPU1 显存
```

分解一下：

1. GPU0 显存 → CPU 内存（一次 PCIe 传输）。
2. CPU 内存 → GPU1 显存（再一次 PCIe 传输）。
3. CPU 必须参与两次 DMA 发起和同步。

问题在于：

- **每次都要经过 CPU 内存，浪费 PCIe 带宽。**
- 延迟更高，CPU 也会因此更忙。
- 在大模型场景里，一步同步可能就是几十 GB，如果还走“显存→CPU 内存→显存”，几乎不可接受。

而有了 P2P 之后：

```text
GPU0 显存 ──(PCIe P2P)──> GPU1 显存
```

只有一次 PCIe 传输，没有 CPU 内存中转。 

再加上 NVLink / HCCS 的话，甚至可以绕开 PCIe，用更宽更快的链路。

所以现代单机多卡设计，都会尽量确保：

- 主板 / BIOS 支持 PCIe P2P。
- GPU 之间有良好的 P2P 拓扑。
- 对关键路径上的数据同步，**优先走 P2P/NVLink，而不是 Host 内存中转**。

## 6.5 NVLink / NVSwitch 场景下的单机多卡通信

如果机器里有 NVLink / HCCS，那么 GPU0 → GPU1 的路径会变成：

```text
GPU0 显存 ──(NVLink)──> GPU1 显存
```

区别在于：

- 物理通道从 PCIe 换成了 NVLink/HCCS：
  - 带宽更高（多倍于 PCIe）。
  - 延迟更低。
  - 拓扑可以是：
    - 简单点对点。
    - 通过 NVSwitch / 超节点结构组成一个更大的“全互联域”。

对上层 NCCL/HCCL 来说：

- 它看到的仍然是“可以做 P2P 的两张卡”。
- 区别在于底层 runtime / 驱动会优先选择：
  - 如果有 NVLink，就走 NVLink。
  - 如果没有，就退回 PCIe P2P。

在 A3 超节点 / NVSwitch 这类结构里：

- 单机内部可以有 16/32 张卡，其间全部通过高速交换结构互联。
- 对通信库来说，这一整块就像一个“超大的单机高速域”。

这就是为什么在前面的讨论中，我们一直强调：

- A2：高速域 ≈ 单机 8 卡。
- A3 超节点：高速域 ≈ 16/32 卡。

因为在这个高速域内，所有 GPU 之间的 P2P / AllReduce 都可以走 NVLink/HCCS+交换结构，而不需要上 RoCE/IB。

## 6.6 单机 8 卡 Ring AllReduce：逐步文字时序图

现在我们把视角放大到：**单机 8 卡，做一次 Ring AllReduce**。

假设 GPU0~GPU7，每个 GPU 都有一个梯度 buffer，被平均分成 8 个 chunk：`chunk0 ~ chunk7`。

Ring AllReduce 分两大阶段：

- Reduce-Scatter（边传边加，最后每张卡只留一块“自己负责的 chunk”）。
- All-Gather（把各自负责的 chunk 再广播回所有卡）。

这里我们只看其中一轮的“环上传一圈”的通信模式，重点是：**每一跳就是一次 P2P DMA**。

以 Reduce-Scatter 的前几轮为例：

1. **初始状态**
   - GPU0~GPU7 各自有自己的 buffer，拆成 8 份：
     - GPU0: `[0a 0b 0c 0d 0e 0f 0g 0h]`
     - GPU1: `[1a 1b 1c 1d 1e 1f 1g 1h]`
     - ……
   - 目标：通过多轮“传+加”，让每个 chunk 的和最终落到一张 GPU 上。

2. **第 1 轮：每张 GPU 把 chunk0 发送给右边的 GPU**
   - 拓扑：`GPU0 → GPU1 → GPU2 → … → GPU7 → GPU0`。
   - 对于每一跳，比如 GPU0 → GPU1：
     - NCCL：
       - 决定 GPU0 应该把 `0a` 发给 GPU1。
       - 调用 P2P copy → 触发 GPU0 DMA Engine。
     - GPU0：
       - DMA Engine 通过 PCIe/NVLink 把 `0a` 写到 GPU1 的某个接收 buffer。
     - GPU1：
       - DMA 完成后，在本地把 `0a` 和 `1a` 相加，得到 `(0a+1a)`。
   - 整张机子内部，此时在同时发生 8 个这样的 GPU→GPU P2P DMA。

3. **第 2 轮：每张 GPU 把当前持有的 chunk0 发给右边 GPU**
   - GPU1 把 `(0a+1a)` 发给 GPU2。
   - GPU2 把 `(2a)` 发给 GPU3。
   - ……
   - 每一跳仍然是：
     - 源 GPU DMA Engine 发起 P2P DMA。
     - 数据走 PCIe/NVLink。
     - 目标 GPU 累加。

4. **多轮之后**
   - `chunk0` 会在环上转 N-1 轮。
   - 最终会全部累加到某一张 GPU（比如 GPU7）。
   - 其它 chunk 在并行的环路中也做类似的传播 / 累加，只是起点和终点不同。

在这个过程中，每一跳的本质动作都是：

```text
NCCL：决定谁发给谁 → 调用 P2P
源 GPU：DMA Engine 发起 P2P DMA
通路：PCIe / NVLink / HCCS
目标 GPU：写入显存 + 累加 + 完成通知
```

你可以看到：

- 整个 Ring AllReduce 在单机 8 卡里的核心，就是一串精心调度好的 **P2P DMA 链**。

## 6.7 单机内部是否需要 RDMA / NIC？和多机场景的对比

现在可以回答一个经常被问到的问题：

> “单机 8 卡 AllReduce / P2P 需不需要 RDMA 网卡参与？”  

答案是：**不需要。**

- 单机内部：
  - 完全可以由 **GPU DMA Engine + PCIe/NVLink/HCCS** 搞定。
  - NIC 在这里并不参与数据搬运（除非你把多机拓扑“压缩”到单机里模拟）。
- 多机场景：
  - GPU 无法直接访问远端 GPU 显存。
  - 必须通过 **本机 NIC RDMA Engine → 网络 → 远端 NIC RDMA Engine** 完成一次显存到显存的 end-to-end 搬运。

放在一条对比线上看，就是：

```text
【单机多卡】
  源 GPU DMA Engine
    ──(PCIe / NVLink / HCCS)──> 目标 GPU 显存

【多机多卡】
  源 GPU 显存
    ──(PCIe DMA by NIC)──> 本机 NIC
    ──(IB / RoCE 网络)──> 远端 NIC
    ──(PCIe DMA by NIC)──> 远端 GPU 显存
```

这一章的所有内容，都是为了后面第 7 章做铺垫：

你已经完全理解了“在一台机器里，GPU 之间是怎么靠 DMA + PCIe/NVLink/HCCS 来传来传去的”，

接下来我们只需要把这套思路扩展到“多台机器 + NIC + RDMA 网络”，就能自然看懂多机多卡下的数据流和时序。

# 7. 多机多卡：从单机环到跨机大环（训练与推理）

现在我们把视角从“单机 8 卡”扩展到“多机多卡”。  

这一章的目标，是让你能非常具体地回答这些问题：

- 多机多卡时，**GPU0 把数据发到远端机器 GPU4**，到底走了哪几步？
- `ibv_reg_mr` / `ibv_post_send` / RDMA verbs / CQ 分别在干什么？
- NCCL 是怎么把“单机环” + “跨机跳”拼成一个逻辑上的大 Ring AllReduce 的？

## 7.1 多机多卡的典型拓扑：2 机 × 4 卡、N 机 × 8 卡

先用一个最小但已经有“多机味道”的例子：**2 台机器，每台 4 卡**。

```text
Machine A: GPU0, GPU1, GPU2, GPU3
Machine B: GPU4, GPU5, GPU6, GPU7

每台机：
  GPU0~3 通过 PCIe / NVLink / HCCS 互联
两台机：
  通过 RDMA 网络互联（IB / RoCE）

拓扑上会大致长这样：

GPU0  GPU1  GPU2  GPU3      GPU4  GPU5  GPU6  GPU7
  \    |    |    /            \    |    |    /
   \   |    |   /              \   |    |   /
     [ PCIe / NVLink / HCCS ]     [ PCIe / NVLink / HCCS ]
              |                               |
             NIC(A)  ─────── RDMA ───────   NIC(B)
```

在更大的集群里，只是把“每机 4/8 卡”这个单元复制 N 份而已，本质拓扑是：

- 机内：
  - GPU ↔ GPU：PCIe P2P / NVLink / HCCS。
- 机间：
  - NIC ↔ NIC：IB / RoCE + RDMA。

此时 Ring AllReduce 就会变成：

- 机内：小环（充分利用高速域）。
- 机间：这些小环之间通过 NIC + RDMA 连接成逻辑上的“大环”。

## 7.2 初始化阶段：ibv_reg_mr、rkey、memory region 与 IOMMU 的作用

在任何一次跨机 GPUDirect RDMA 传输之前，都有一个**关键的初始化阶段**，这一步在你和 ChatGPT 的对话里已经出现了很多次：

> `ibv_reg_mr()`：注册内存为 RDMA 可访问区域（memory region）。  

我们用“GPU 显存”当例子来讲清楚它到底干了什么。

假设：

- 在 Machine A 的 GPU0 上，有一个梯度 buffer：`buf_A0`，长度 16MB。
- 在 Machine B 的 GPU4 上，有一个接收 buffer：`buf_B4`，长度 16MB。

**步骤 1：显存分配**

- 在每张 GPU 上，框架 / 通信库通过 CUDA/CANN 等接口：
  - 分配显存，拿到设备指针 `buf_A0` / `buf_B4`。

**步骤 2：注册为 RDMA memory region**

- NCCL/HCCL 调用 RDMA verbs：

  ```c
  ibv_mr* mr_A0 = ibv_reg_mr(pd, buf_A0, size,
                             IBV_ACCESS_LOCAL_WRITE |
                             IBV_ACCESS_REMOTE_READ |
                             IBV_ACCESS_REMOTE_WRITE);
  ```

  - 这里的 `ibv_reg_mr` 做了几件事：
    1. **锁定这段显存（pin memory）**，防止其被迁移。
    2. **通过 IOMMU 建立映射关系**：
       - 告诉系统：“这段 GPU 显存允许被某个 NIC 通过 PCIe DMA 访问”。
    3. **创建一个 memory region 对象**，并分配一个 `lkey` / `rkey`：
       - `lkey`：本机访问这段内存时用的“钥匙”。
       - `rkey`：远端 NIC 要访问这段内存时使用的“钥匙”。

**步骤 3：交换 rkey 和地址**

- 初始化阶段，所有参与节点会通过某种 bootstrap 通道（比如 TCP）交换：
  - 每个 buffer 的：
    - 所在机 / 所在 GPU。
    - 起始地址（或 offset）。
    - `rkey`。
  - 这样一来：
    - Machine A 上的 NIC 就知道：
      - 如果要写 Machine B 上 GPU4 的 `buf_B4`，应该用哪个 rkey + 地址。

你可以把 `ibv_reg_mr` 理解为：

> “把一段 GPU 显存在整个 RDMA 网络的视角下**登记建档**，并发一把钥匙（rkey）给远端 NIC，允许它对这段显存发起 DMA 读写。”  

这一切做完以后，NIC 才有权利说：“好，我知道这段显存在哪儿，可以在需要的时候直接 DMA。”

## 7.3 一次 GPU0 → GPU4（跨机）的完整 GPUDirect RDMA 流程

现在开始走你最关心的那条链路：  

> Machine A 的 GPU0，要把一块梯度 chunk 发送到 Machine B 的 GPU4。  

假设：

- 源：Machine A, GPU0, `buf_A0`。
- 目标：Machine B, GPU4, `buf_B4`。

完整流程可以拆成以下几步：

**（1）NCCL 决策：现在轮到谁发给谁？**

- 在 Ring AllReduce 中，NCCL 维护一个“谁是我的左邻右舍”的环形表。
- 某一轮中，NCCL 得出结论：
  - “现在轮到 Machine A 的 GPU0，把 `buf_A0` 的某个 chunk 发给 Machine B 的 GPU4。”

**（2）NCCL 调用 RDMA verbs：ibv_post_send**

- 在 GPU0 所在的进程中，NCCL 会调用：

  ```c
  ibv_post_send(qp, &wr, &bad_wr);
  ```

  - `qp`：Queue Pair，代表一条 RDMA 连接（或通道）。
  - `wr`：Work Request，描述了：
    - 从哪段本地内存（GPU 显存）读。
    - 要写到远端哪个 memory region（用 rkey + addr 标识）。
    - 是 RDMA Write / RDMA Read / Send / 其它操作。

此时还没有数据真正上路，**只是把“我要发一条写操作”的任务丢进 NIC 的发送队列里**。

**（3）本机 NIC(A) 开始干活：PCIe DMA 读 GPU0 显存**

- NIC(A) 内部的 RDMA Engine 轮询 / 处理发送队列中的 WR：
  - 看见一条“从 `buf_A0` 读 16MB，写到远端 Machine B:GPU4:buf_B4”的 RDMA Write。
  - 于是它做两件事：
    1. 通过 IOMMU 映射，确认 `buf_A0` 对应的物理地址。
    2. 作为 **PCIe Bus Master**，发起对 GPU0 显存的 DMA Read：
       - 这一步和单机 P2P 中“GPU 自己发 DMA”类似，只不过现在是 NIC 发。

**（4）本机 NIC(A)：打包成 IB/RoCE 数据包，发往 NIC(B)**

- NIC(A) 读完一段数据后：
  - 把数据切成若干个网络包，按照 IB / RoCE 协议封装。
  - 根据 QP 中记录的对端地址信息（IB LID / GID、RoCE IP/UDP 等），丢到交换机上。

**（5）远端 NIC(B)：拆包 + PCIe DMA 写 GPU4 显存**

- NIC(B) 收到这些 RDMA 包：
  - 解析出：
    - 这是一个 RDMA Write。
    - 目标 memory region 的 rkey / 地址 / 长度。
  - 进行权限校验（rkey 是否匹配）。
  - 作为 PCIe Bus Master，对 GPU4 显存发起 DMA Write：
    - 把数据写入 `buf_B4` 的对应区域。

**（6）完成通知：CQ → NCCL**

- 当 NIC(B) 确认写入完成后：
  - 在本地写入一个 **Completion Queue (CQ) 条目**。
  - 本机的 RDMA 驱动 / NCCL 在轮询 CQ 时看到：
    - “啊，这一次从 GPU0 发来的写操作完成了。”
- NCCL 这时可以认为：
  - GPU4 这部分 buffer 已经就绪。
  - 可以进行下一步计算（比如累加），或继续沿着环传给下一个节点。

把上面的过程压缩成一条直观的“物理路径”就是：

```text
GPU0 显存
  └─(PCIe DMA by NIC A)→ NIC(A)
       └─(IB / RoCE 网络)→ NIC(B)
             └─(PCIe DMA by NIC B)→ GPU4 显存

NCCL 通过 CQ 得知：GPU4 端的写入已经完成
```

关键点：

- **GPU0 本身没有发 DMA，也没有发网络包**。
- 它只是：
  - 提供了一个 buffer 地址。
  - 通过 `ibv_reg_mr` 等操作，让 NIC 知道“这段显存可以 DMA”。
- 实际搬运人：**NIC 上的 RDMA Engine**。

## 7.4 RDMA verbs 是什么：应用 / verbs 库 / 内核驱动 / NIC 的分层关系

你之前已经注意到：我们反复提到 `ibv_reg_mr`、`ibv_post_send`，它们都属于 **RDMA verbs**。

可以把分层关系画成这样：

```text
应用 / 通信库（NCCL / HCCL / 你的程序）
    │ 调用
    ▼
RDMA verbs 用户态库（libibverbs 等）
    │ 通过系统调用 / 内核接口
    ▼
RDMA 内核驱动（ib_*, mlx5_core, roce 等）
    │ 控制
    ▼
NIC 硬件（RDMA Engine + DMA Engine + 队列 + CQ）
```

作用可以概括为：

- **verbs 库**：
  - 把“我要注册一块内存 / 我要发一次 RDMA Write”这种意图，用统一 API 表达出来。
  - 管理 `QP`（Queue Pair）、`CQ`（Completion Queue）、`MR`（Memory Region）等抽象对象。
- **内核驱动**：
  - 把这些对象映射到底层硬件：
    - 在 NIC 上建立真实的 QP / CQ / MR。
    - 设置 DMA 映射。
    - 配置 NIC 的寄存器、队列、路由信息等。
- **NIC 硬件**：
  - 真正执行 DMA + 发包 + 收包 + 写 CQ。

所以，当你看到：

```c
ibv_reg_mr(...);
ibv_post_send(...);
```

要在脑子里自动翻译成：

> “我在告诉 NIC 驱动：那块显存你可以 DMA 了；现在请你根据这个描述，发起一次从这块显存到远端那块显存的 RDMA 操作。”  

## 7.5 ibv_post_send 到底在 NIC 里触发了什么？

`ibv_post_send` 本身并不会“立刻发一个网络包”，它做的事情是：

1. 在用户态构造一个 Work Request（WR）：
   - 含义：我要发一次 RDMA Write / Read / Send。
   - 携带信息：
     - 本地 buffer 的地址 + 长度 + lkey。
     - 远端 buffer 的地址 + rkey（如果是 RDMA Write / Read）。
     - 操作类型（RDMA Write / RDMA Read / Send / …）。
2. 把 WR 投递到 NIC 对应 QP 的 **发送队列（SQ）**。
3. NIC 硬件上的发送单元会轮询 / 触发处理这些 WR：
   - 读本地 buffer（DMA Read）。
   - 封包成 RDMA 包。
   - 通过网络发送。
   - 在完成后，向本地 CQ 写一个完成条目。

从这个角度看：

> `ibv_post_send` = “在 NIC 的待办队列里扔一条任务单”。  

真正干活的是：

- NIC 的 RDMA Engine（执行 DMA + 网络传输）。

## 7.6 Completion Queue：数据搬完后 NCCL 是怎么知道的？

当 NIC 完成了一条 Work Request（比如一条 RDMA Write），它会：

- 在对应的 **Completion Queue (CQ)** 里写入一个条目（Completion Queue Entry, CQE）。
  - 里面包括：
    - 这条 WR 的 ID。
    - 操作是否成功。
    - 传输的字节数等信息。

NCCL / HCCL 的做法通常是：

- 轮询 / 阻塞在 CQ 上，直到看到对应的 CQE：
  - 一旦看到：说明这一条 DMA + 网络 + 远端写入 **已经完成**。
  - 接下来：
    - 可以把这段 buffer 标记为“可用”。
    - 可以开始下一步计算（比如在 GPU 上做累加）。
    - 或者可以发起下一跳的 RDMA 操作。

你之前抓到的那个细节“**已写入 GPU1 buffer，NCCL 被通知**”其实就是：

- **NIC 写完 GPU1 显存 → 写 CQ → NCCL 读 CQ → 才知道 buffer 可用。**

## 7.7 机内小环 + 跨机大环：实战中的 Ring AllReduce 优化策略

理论上的 Ring AllReduce 可以把所有 GPU 按照全局顺序串成一个大环：

```text
GPU0 → GPU1 → GPU2 → GPU3 → GPU4 → GPU5 → GPU6 → GPU7 → GPU0
```

但在真实多机环境中，为了性能，通信库（NCCL/HCCL）往往会做两件优化：

1. **机内小环**
   - 在每台机器内部，先用单机 P2P / NVLink 把局部数据先尽量聚合。
   - 减少跨机需要传输的数据量。
2. **跨机大环**
   - 把每台机器视作一个“超级节点”，在机器之间再做一层 Ring。
   - 这一层的每一跳使用 RDMA + NIC 搬数据。

所以实际情况常常是：

```text
机内：
  GPU0 → GPU1 → GPU2 → GPU3 → GPU0   （PCIe/NVLink 小环）

机间：
  Machine A (GPU0~3) → Machine B (GPU4~7) → Machine C … → Machine A
  （RDMA 大环）
```

从你的角度看，有两个关键结论：

- “单机环”和“多机环”**本质一样**：都是某种模式下的 AllReduce / 传递。
  - 差别只是搬运的执行者：
    - 单机：GPU DMA Engine。
    - 多机：NIC RDMA Engine。
- 通信库（NCCL/HCCL）负责在这两层之间做协调。

## 7.8 单机 DMA vs 跨机 RDMA 的对照总图

到目前为止，我们已经有足够信息把“单机 DMA”和“跨机 RDMA”并排画出来了：

```text
【单机多卡，一跳 P2P】

NCCL：决定 GPU0 → GPU1
  │
  └─ 调用 P2P copy（CUDA/CANN）
        │
        └─ GPU0 DMA Engine
              │  读 GPU0 显存
              ▼
        (PCIe / NVLink / HCCS)
              ▼
           GPU1 显存写入
              │
              └─ DMA 完成 → 通知 NCCL

--------------------------------------------------------

【多机多卡，一跳 RDMA】

NCCL：决定 (Machine A, GPU0) → (Machine B, GPU4)
  │
  └─ 调用 ibv_post_send（RDMA Write）
        │
        └─ NIC(A) RDMA Engine
              │  通过 PCIe DMA 读 GPU0 显存
              ▼
        (IB / RoCE 网络)
              ▼
           NIC(B) RDMA Engine
              │  通过 PCIe DMA 写 GPU4 显存
              ▼
       写入完成 → NIC(B) 写 CQ → NCCL 轮询到完成
```

可以看到：

- **控制流**（谁决定发、谁等结果）非常相似：都是 NCCL / HCCL 在调度。
- **数据流**的区别只在于：
  - 单机：数据只在本机总线（PCIe/NVLink/HCCS）上跑。
  - 多机：数据要先经过本机总线到 NIC，再通过网络到远端 NIC，再通过远端总线到远端 GPU。

## 7.9 多机推理 vs 多机训练：在上述流程里只改了哪几步？

最后，把这些内容跟“训练 vs 推理”的差异捆起来看一眼：

- 在 **训练** 中，跨机 RDMA 主要用来：
  - 做 **梯度 AllReduce**（大量的 RDMA Write / Read / Send）。
  - 有时也用于激活 / 中间结果的交换（例如模型并行）。
- 在 **推理** 中，跨机 RDMA 主要用来：
  - 初始化时的 **权重广播 / 加载**（一次性/低频但数据量巨大）。
  - 推理过程中不同并行策略下的 **中间结果交换**：
    - 张量并行：每一层的输出 / 部分激活需要在卡之间交换。
    - MoE / 分层并行：专家选择结果、中间路由结果等的传递。

从“链路形态”来看：

- 无论是训练还是推理：
  - 跨机数据都是：`GPU 显存 ↔ NIC ↔ 网络 ↔ NIC ↔ GPU 显存`。
  - 控制都是：NCCL/HCCL → RDMA verbs → NIC RdmaEngine → CQ → NCCL。
- 区别只在于：
  - **何时发？发哪些 buffer？发多频繁？**（算法与并行策略层面）。

所以，当你搞清楚这一章的所有细节之后：

- 不管对方说的是“多机训练”还是“多机推理”，你都可以立刻在脑子里把对方描述的过程投影到：

```text
单机：GPU DMA Engine + PCIe/NVLink/HCCS
多机：NIC RDMA Engine + IB/RoCE + CQ + NCCL/HCCL
```

这就是你在一开始想要的那张“完整图景”：

从 GPU 内部 DMA，到单机 P2P，到 NIC 通过 RDMA 直接读写远端 GPU 显存，再到上层 NCCL/HCCL 的调度逻辑——都已经串成了一条可以在脑子里重放的时间线。下一章，我们会专门把 NCCL/HCCL 的“位置”和“职责”再单独拿出来讲清楚。

# 8. 通信库：NCCL / HCCL 在整条链路中的真实位置

前面两章，你已经看清了：

- 单机：谁发 DMA、数据走 PCIe/NVLink/HCCS，在 GPU 与 GPU 之间绕圈。
- 多机：谁发 RDMA、数据走 NIC + IB/RoCE，在机器与机器之间跳来跳去。

这一章我们专门回答一个问题：

> “在这一切当中，**NCCL / HCCL 到底是谁，它到底干了什么？**”  

## 8.1 为什么说“GPU 不会自己发网络包”？

从硬件上看：

- GPU / NPU：
  - 有计算核心（SM / AI Core）。
  - 有自己的显存控制器（访问 HBM 等）。
  - 有连接主机 / 其它卡的总线接口（PCIe / NVLink / HCCS）。
  - **没有网卡功能**，不会讲 TCP/IP / IB / RoCE。
- NIC：
  - 有 MAC / PHY / RDMA Engine。
  - 负责把 PCIe 上来的数据变成网络包发出去，反之亦然。

因此：

- GPU 能做的是：**通过自己的 DMA Engine 或被 NIC 的 DMA Engine “读写显存”**。
- 不能做的是：**直接在数据中心网络上发报文**。

那“多机多卡通信逻辑”放在哪儿呢？  
既不能放在 GPU 本身（它不会发包），也不能只放在 NIC（它只会搬数据，不懂模型）。  

这就轮到 **通信库（NCCL / HCCL）** 登场了。

## 8.2 NCCL / HCCL 的职责边界：调度者而非搬运工

你可以把 NCCL / HCCL 想象成：

> “负责全局通信的调度大脑 + 通信算法实现者”，
>
> 而不是“真正扛包的工人”。

更具体一点，它们主要负责这些事情：

1. **通信算法与拓扑**
   - 决定在当前环境下，该用：
     - Ring AllReduce？
     - Tree / 分层 AllReduce？
     - 哪种 Broadcast / Reduce / AllGather 变体？
   - 决定全局拓扑：
     - 哪些 GPU 在同一机内小环？
     - 哪些机器之间构成大环？
     - 哪些路径走单机 P2P，哪些路径走 RDMA？

2. **调度与时序**
   - 决定在每一轮、每一步：
     - 谁是发送方（源 GPU / 源机器）？
     - 谁是接收方（目标 GPU / 目标机器）？
     - 发送哪个 chunk？发多少字节？
   - 保证不会出现“大家都在等彼此”的死锁。

3. **调用底层搬运能力**
   - 在单机场景：
     - 调用 CUDA / CANN P2P 接口，让源 GPU 的 DMA Engine 发起 P2P DMA。
   - 在多机场景：
     - 调用 RDMA verbs（`ibv_reg_mr` / `ibv_post_send` 等），让 NIC 的 RDMA Engine 发起跨机 DMA。

4. **错误处理与重试**
   - 某条链路出问题时，负责：
     - 返回错误码给上层。
     - 在可能的情况下，在算法层面做重试 / 拓扑调整。

因此：

- **NCCL / HCCL 不直接搬数据**，它只是：
  - 根据算法和拓扑决定“搬运计划”。
  - 把“如何搬”的任务单，交给 GPU DMA Engine 或 NIC RDMA Engine 去执行。

你可以记一句：

> GPU / NIC 是搬运工，NCCL / HCCL 是调度室。

## 8.3 NCCL 如何同时调度单机 DMA 和多机 RDMA？

在真实系统中，**NCCL 同时面对两种截然不同的搬运机制**：

- 单机：
  - 使用 GPU DMA Engine + PCIe / NVLink / HCCS。
- 多机：
  - 使用 NIC RDMA Engine + IB / RoCE。

但对上层（比如 PyTorch / MindSpore）来说，它只看到一个统一的接口：

```c
ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
```

内部大致会经历这样几步：

1. **根据 comm（通信上下文）得出全局拓扑**
   - 哪些 rank 在同一节点？
   - 节点之间的连接是怎样的？
   - 是否支持 GPUDirect RDMA？

2. **把 AllReduce 分拆成多轮 send/recv 操作**
   - 在单机内部分：
     - 规划一系列 `GPU_i → GPU_j` 的 P2P 复制。
   - 在跨机部分：
     - 规划一系列 `(node_p, GPU_a) → (node_q, GPU_b)` 的 RDMA 传输。

3. **按位置选择不同后端**
   - 对于处于同一机内的小环：
     - 调用 CUDA / CANN 的 P2P 接口，触发 GPU DMA Engine。
   - 对于跨机的大环：
     - 调用 RDMA verbs，触发 NIC RDMA Engine。

对你来说，有两点是最关键的：

- **你不需要手写“GPU0 调用 P2P 复制到 GPU1，GPU1 再 RDMA 到别的机器 ……”这些细节**，NCCL 会统一帮你做。
- 但你要清楚：
  - 单机部分的“执行者”是 GPU DMA Engine。
  - 多机部分的“执行者”是 NIC RDMA Engine。

## 8.4 控制平面 vs 数据平面：CPU / NCCL 与 DMA / RDMA 的分工

在整个多机多卡系统中，常常会把工作分成两条“平面”：

- **控制平面（Control Plane）**
  - CPU / 通信库 / 框架 层：
    - 决定“要做什么”：
      - 执行哪种并行策略？
      - 这一步需要哪些 AllReduce / Broadcast？
    - 生成“搬运任务”：
      - 调用 NCCL/HCCL `AllReduce` / `Broadcast`。
      - 调用 RDMA verbs 去投递 WR（Work Request）。
  - 特征：
    - 指令少、逻辑复杂。
    - 适合 CPU / 高层库来处理。

- **数据平面（Data Plane）**
  - GPU DMA Engine / NIC RDMA Engine / PCIe / NVLink / HCCS / 网络：
    - 真正“搬字节”的部分。
    - 坐实了所有你在拓扑图上画的那条条“线”。
  - 特征：
    - 数据量巨大、模式相对固定。
    - 最适合让专门的硬件 DMA 来干。

从你现在已经理解的几个例子来看：

- `ncclAllReduce(...)` 调用 → 控制平面行为。
- `GPU0 DMA Engine 发起 P2P` / `NIC(A) DMA 读 GPU0 显存` → 数据平面行为。

控制平面不搬数据，数据平面不做高层决策。

NCCL / HCCL 正是在这两者之间承担“翻译 /调度”角色的组件。

## 8.5 PyTorch / MindSpore 怎么调用到 NCCL / HCCL？

作为一个“用这套系统的人”，你在代码里通常只会写类似这样的东西：

```python
model = DDP(model, device_ids=[local_rank])
...
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

或者在 MindSpore / Ascend 环境里使用它自己的并行接口。

这些高层框架会在内部：

1. **初始化通信环境**
   - 调用 `ncclCommInitRank` / `hcclCommInitRootInfo` 等接口：
     - 建立进程与 GPU 的映射（哪个 rank 对应哪张卡）。
     - 建立 NCCL/HCCL 通信上下文（comm）。

2. **在 backward() 中插入 AllReduce 等操作**
   - 例如 PyTorch DDP：
     - 在每个参数的梯度 ready 时，自动触发一次 `ncclAllReduce`。
   - MindSpore / 其它框架也会有类似逻辑。

3. **把这些 AllReduce / Broadcast 转交给 NCCL / HCCL**
   - 对你来说，`loss.backward()` 只是一个 Python 调用。
   - 对框架来说，它会：
     - 把这次 backward 展开成很多 “单 GPU 算子 + 多 GPU 通信”。
     - 把所有“多 GPU 通信”那部分丢给 NCCL/HCCL 负责。

从上往下的调用链，大致是：

```text
你的 Python 训练脚本
    │
    ▼
PyTorch / MindSpore 等框架
    │
    ▼
NCCL / HCCL（通信库）
    │
    ├─ 单机：调用 CUDA / CANN P2P 接口 → GPU DMA Engine
    └─ 多机：调用 RDMA verbs → 内核驱动 → NIC RDMA Engine
```

你可以直接把这条链条塞回到第 1 章的“整体大图”里去对照，会发现它们完全对得上。

---

到这里，关于“**是谁在算 / 谁在搬 / 谁在调度 / 谁在发指令**”这条线，已经闭环了：

- 第 2 章：硬件角色（GPU / DMA Engine / NIC / RDMA Engine）。
- 第 6 章：单机内部数据流（GPU DMA Engine + PCIe/NVLink/HCCS）。
- 第 7 章：多机数据流（NIC RDMA Engine + IB/RoCE）。
- 第 8 章：通信库层（NCCL/HCCL）在整个栈里的“真实位置”。

接下来，你可以带着这套完整心智模型，去看第 9 章（310P vs A2/A3 选型）和第 10 章（单机 / 多机训推全流程追踪），那时候你再遇到任何名词，基本上都能立刻知道它在这张图的哪一层。 

# 9. 310P、Atlas A2/A3 与实际工程选型

前面几章，你已经站在“原理”的高度理解了单机 / 多机、DMA / RDMA、NCCL / HCCL 的全图。

这一章我们回到现实一点的问题：

> “我手上如果只有 310P、或者想用 Atlas A2/A3 + vLLM/MindIE 做多机多卡推理/训练，**哪些方案是现实可行的，哪些从架构上就不行？**”  

## 9.1 310P 的硬件定位：单机推理解码卡，而不是集群节点

先给结论：**310P 更像是一张“单机/边缘推理解码卡”，而不是为大规模集群训练/推理设计的节点。**

从硬件上看，310P 的典型特征是：

- 以 **PCIe 插卡** 形态存在。
- 只有 PCIe 接口，没有 HCCS / 类 NVLink 的板级高速互联。
- 没有为“多机多卡集群内互联”预留专门的高速互联模块（没有 HCCS 口、没有板级超节点拓扑）。

这意味着：

- **单机内**：
  - 310P 可以作为 NPU 加速卡，通过 PCIe 接入主机。
  - 可以在一台机器里插多张 310P 做一定程度的多卡推理（受限于框架支持和带宽）。
- **多机之间**：
  - 310P 自身**没有**像 910B/A2/A3 那样的板级互联 + 集群级互联设计。
  - 多机训练/推理要依赖 NIC + 网络，但软件栈（尤其是 HCCL）对 310P 多机场景 **没有提供支持**。

简单说：310P 的“形态”和“互联能力”决定了它更适合作为**单机推理节点**，而不是大规模多机多卡集群的基础单元。

## 9.2 为什么 310P 做不了多机分布式训练 / 推理（硬件 + HCCL + 官方口径）

你在和华为技术支持的问答里，其实已经拿到了非常明确的三方面佐证：

1. **硬件层面：缺少集群级高速互联模块**
   - 技术支持给出的原话大意是：
     - “310P 做的标卡硬件上没有高速互联扩展模块 (HCCS 升腾内部互联类似 NVLink / RDMA 多机接口)，只有 PCIe 用于单机内互联。”
   - 对照前文：
     - Atlas A2/A3 之所以能做多机分布式，是因为：
       - 机内有 HCCS / NVLink 类似结构。
       - 机间有面向集群的网络拓扑支持。
     - 310P 缺少这一整套“为集群设计的互联层级”。

2. **软件层面：HCCL 不支持 310P 多机**
   - 技术支持明确提到：
     - HCCL 的多机互联场景在代码层面**没有针对 310P 的实现**。
     - 官方 HCCL 仓库和文档中，多机场景的支持对象主要是 910B / A2 / A3 等。
   - 这点很关键：
     - 即使你在网络上“勉强打通了”多机 TCP/RDMA 通路，
     - 如果 HCCL 在上层没有 310P 多机场景的逻辑，**就不会有那一整套 AllReduce / Broadcast 等分布式原语可用**。

3. **官方明确口径：310P 多机不支持**
   - 技术支持在你们的对话里已经直接给出：
     - “310P 多机不支持。”
   - 这不是“我没测过所以说不支持”，而是：
     - 从硬件互联能力、软件栈支持、产品定位这三方面综合给出的结论。

再叠加一个你们自己遇到的案例：

- 即便在 **单机 300I Duo**（也是偏推理解码卡）场景下：
  - 8 卡推理性能反而低于 4 卡。
  - 咨询华为后被告知这不在支持范围内。
  - 文档中也明确限制单机最大支持 ≤ 4 张卡。

这说明：

- 对于“推理解码类 PCIe 卡”，**华为官方既不鼓励也不支持你把它们当成大规模多机集群节点来用**。

归纳一句话：

> 310P 做不了多机分布式训练 / 推理，**不是“还没测”，而是架构和软件栈本身就没有朝这个方向设计和认证**。

## 9.3 Atlas 800T/800I A2 / A3 系列：基于 910B / A3 的集群节点设计

对比之下，Atlas A2/A3 系列的定位就清晰得多：

- **芯片代际**：
  - A2 系列：基于 Ascend 910B。
  - A3 系列：基于 910B 演进版 / A3 内核（如 910B3 等）。
- **产品形态**：
  - Atlas 800T A2 / A3：训练服务器。
  - Atlas 800I A2 / A3：推理服务器（但同样基于 910B/A3 体系）。
  - Atlas 900 A2 PoD / Atlas 900 A3 SuperPoD / 9000 A3 SuperPoD：由多台 800T/800I 组成的大规模集群。
- **互联结构**：
  - 机内：多张 910B/A3 通过 HCCS / 板级互联、甚至超节点结构组成一个大高速域。
  - 机间：通过 RoCE/IB 等网络互联，支持多机多卡 AllReduce。
- **软件栈支持**：
  - HCCL 的多机多卡支持，重点就是为 910B / A2 / A3 服务器设计的。
  - MindSpore / MindIE / vLLM-Ascend 等框架，在多机多卡场景下的测试 / 认证也都围绕这些设备做。

可以简单理解为：

- 310P：
  - 只解决“单机某些推理场景的算力补充”。
- 800T/800I A2/A3：
  - 从一开始就按“集群节点”设计，有：
    - 机内高速互联（HCCS / 超节点）。
    - 机间多机多卡互联（RoCE/IB）。
    - 与 HCCL / MindIE / vLLM-Ascend 等软件栈的一致支持。

这就是为什么在 vLLM-Ascend 的官方文档里，**明确列出的支持设备是 A2/A3 系列，而不是 310P**。

## 9.4 vLLM-Ascend 与 MindIE 的官方支持硬件列表解读

你在一开始贴过 vLLM-Ascend 文档中的支持列表，大致是这样的（简化）：

- Atlas A2 训练系列：
  - Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2。
- Atlas 800I A2 推理系列。
- Atlas A3 训练系列：
  - Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD。
- Atlas 800I A3 推理系列。
- \[实验性\] Atlas 300I 推理系列（Atlas 300I Duo）。

**注意缺失的名字：**

- 没有：310P。

MindIE 文档和你和技术支持的对话也印证了这一点：

- 多机多卡推理支持的硬件列表主要集中在：
  - Atlas 800/900/A2/A3 系列服务器 / 集群。
  - Atlas 300I Duo 在单机场景下有明确的卡数上限。

这说明两件事：

1. **官方的测试与认证重心**：
   - 选定了 A2/A3 系列作为“多机多卡训练/推理”的主力平台。
   - 310P、300I Duo 等更多定位在“小规模推理节点”上。

2. **工程上应该优先遵循官方支持矩阵**：
   - 如果你想在生产环境里跑多机多卡推理/训练：
     - 优先选 **A2/A3 系列**（例如 Atlas 800T A2 / A3）。
   - 在 310P / 300I Duo 上做多机场景，即便“勉强能跑出点东西”，也会处在“非官方支持”的灰色地带：
     - 遇到问题很难获得正式支持。
     - 可能踩到像“8 卡比 4 卡还慢”这种官方明确不支持的情况。

## 9.5 一个现实工程建议：用什么设备验证多机多卡推理最稳妥

综合你之前的诉求（想验证多机多卡推理 / vLLM-Ascend / MindIE 多机能力），以及你手上的资源（310P、可能借到的 Atlas 800T A2 等），一个相对稳妥的路线可以是：

1. **不要在 310P 上投入精力尝试“多机分布式”**：
   - 可以把 310P 当作：
     - 单机场景下的小规模推理节点。
     - 边缘部署 / 一机多路模型服务节点。
   - 但不要指望它在 HCCL / MindIE / vLLM-Ascend 上获得完整的多机场景支持。

2. **优先使用官方明确支持的 A2/A3 设备做多机多卡验证**
   - 例如：
     - 借用 /申请一台 Atlas 800T A2 / A3。
   - 在这些设备上：
     - vLLM-Ascend / MindIE 的多机多卡文档、示例和内部实现都是围绕它们写的。
     - HCCL 在机内 / 机间互联、AllReduce、Broadcast 等路径上都经过了大量验证。

3. **验证路径建议**
   - 从简单到复杂的大致步骤可以是：
     1. **单机多卡推理 / 训练**（在 800T A2/A3 上）：
        - 确认框架 + HCCL 在机内能正常跑满。
     2. **两机多卡 AllReduce / 推理**：
        - 用最小集群（2 机 × 4/8 卡）验证跨机 RDMA / HCCL 配置正确。
     3. **再扩展到更多机 / 更多卡**：
        - 调整并行策略（数据并行 / 张量并行 / pipeline 并行等）。

4. **把 310P 留在它擅长的领域**
   - 可以考虑：
     - 在单机 310P 上部署一些轻量化推理服务，验证模型压缩 / 量化后的效果。
     - 充分利用它在“单机推理功耗 / 性价比”上的优势，而不是强行拉上多机战场。

一句话总结这一章：

- **310P**：单机/边缘推理解码卡，从硬件、HCCL 软件栈、官方口径三个层面都不适合作为多机多卡集群的基础节点。
- **Atlas 800T/800I A2/A3 + Atlas 900/9000 系列**：为多机多卡训练/推理专门设计的节点和集群平台，是 vLLM-Ascend / MindIE / HCCL 等生态重点支持的对象。
- 做工程选型时，应该尊重这背后“互联架构 + 软件栈支持矩阵”的事实，而不是只看“芯片都是昇腾，就随便凑”。

## 9.6 个人疑问：310P为什么支持单机内多卡而不支持多机多卡

但是我还是有一个问题：就是关于310P为什么支持单机内多卡而不支持多机多卡，下面是我的疑问的具体表述：

> 单机内部GPU之间互传，需要NCCL的参与，而且需要的是GPU内部的DMA Engine来通过PCIE或者NVLink直接把数据发到对方的GPU内存里
>
> 多机GPU之间互传，也需要NCCL的参与，但是不需要GPU内部的DMA Engine，实际DMA硬件是NIC上的DMA，就是NCCL调用NIC的驱动来读取自身的显存数据，然后发送给远程机器，另一台机器上NIC网卡的DMA执行写入到其GPU1上的显存里。
>
> 这样看的话，只要GPU有NCCL，能单机多卡内部互传，想要实现多机多卡之间互传，那只要配好带有RDMA功能的网卡和网线，即IB或者RoCE，就一定能实现多机多卡之间的互传。
>
> 也就是说，多机多卡功能的实现，并不比单机多卡内部互传需要额外的GPU本身的硬件，只需要NCCL支持就行了。其实反而是单机多卡互传还额外需要GPU内部的DMA Engine硬件呢。
>
> 如此看的话，硬件是满足要求的，唯一可能多机比单机额外的要求就是需要NCCL/HCCL支持多机。我的理解对吗？

这段问题本身非常典型：**单机能跑 NCCL/HCCL，为什么多机就一定不能“顺便”支持 310P 呢？**

你的抽象思路其实大方向是对的，只是**少了几层“工程前提”**。下面一口气讲清楚。

**一、一句话的答案**

- 在**抽象模型**上，你的推理成立： 

  单机互传靠 GPU 内部 DMA，跨机互传靠 NIC 上的 RDMA DMA Engine，NCCL/HCCL 在上面做调度，**多机理论上只多出“RDMA NIC + 网络 + 通信库多机代码路径”**。

- 但对 **310P** 这张卡来说，**“作为多机集群节点”的整条链路从产品定义开始就没被设计、实现、验证过**：  
  - 没有针对 310P 的 GPU↔NIC GPUDirect RDMA 路径设计与适配。  
  
  - HCCL 没有为 310P 实现多机 backend。  
  
  - 官方明确口径是“310P 多机不支持”，遇到问题也**不在支持范围内**。 
  
    所以现实世界里，它只能被当成“单机/边缘推理解码卡”，而不是 A2/A3 那种“集群节点”。

**二、从抽象模型看：你的推理哪里是对的？**

先肯定一下你的抽象：

1. **单机多卡互传：**
   - 需要 NCCL/HCCL 决定谁发给谁（调度）。  
   - 真正搬数据的是 **GPU 内部的 DMA Engine**，通过 PCIe P2P / NVLink / HCCS 把数据从 GPU0 显存写到 GPU1 显存。

2. **多机多卡互传：**
   - 一样需要 NCCL/HCCL 决定谁发给谁。  
   - 真正搬数据的是 **NIC 上的 RDMA DMA Engine**：  
     - 源机 NIC 通过 PCIe DMA 读本机 GPU 显存。  
     - 通过 IB / RoCE 网络发给远端 NIC。  
     - 远端 NIC 再通过 PCIe DMA 写远端 GPU 显存。

3. **因此在“公式层面”的直觉：**
   - 单机：`GPU0(DMA) → PCIe/NVLink/HCCS → GPU1 显存`  
   - 多机：`GPU 显存 ↔(PCIe DMA by NIC)↔ NIC ↔(IB/RoCE)↔ 远端 NIC ↔(PCIe DMA by NIC)↔ 远端 GPU 显存`  
   - 所以你会很自然地得出：“**只要单机能跑 NCCL/HCCL，再配 RDMA 网卡 + 网络 + 多机开关，多机就应该能跑**。”

到这个层面为止，你的“脑内模型”是对的——**问题在于：现实产品要满足多机，还得额外满足一整套“平台前提条件”**。

**三、现实世界的三大前提：多机不只是“多一个 NCCL/HCCL 开关”**

要让“多机 RDMA ≈ 单机 P2P + NIC 这一层”在真实硬件上跑起来，通常要满足三件事。

**1）有 RDMA NIC，且 NIC 能够直接 DMA 访问加速卡显存（GPUDirect RDMA 能力）**

- 这就是前面多次提到的 **GPUDirect RDMA 模式**：  
  - NIC 和 GPU/NPU 在同一个 PCIe Root Complex 下。  
  - IOMMU / ATS 等支持它们之间的 **peer DMA 映射**。  
  - 驱动栈支持把 GPU 显存注册成 RDMA memory region（`ibv_reg_mr` 能作用在 GPU 显存上，而不仅仅是 CPU 内存）。
- 对 NVIDIA：是 CUDA + Mellanox NIC + 对应驱动一起配合出来的能力。  
- 对 Ascend 910B / A2 / A3：是 CANN/HCCL + RoCE NIC + HCCS 拓扑一起配合出来的能力。
- 如果这条“NIC ↔ 加速卡显存”的链路在**硬件/驱动上没打通**，那 HCCL 就算“原则上会 AllReduce”，也没法在那块卡上真正走 GPUDirect RDMA，只能退回到“显存→CPU 内存→RDMA→CPU 内存→显存”的慢路径，甚至直接报错。

**2）通信库（NCCL/HCCL）真的为“这类卡 + 这类 NIC + 这类拓扑”实现并验证了多机后端**

- “支持多机”不是一个抽象布尔值，而是很多很具体的工程工作：  
  - 针对某个设备类型，知道如何在它的显存上做 `ibv_reg_mr` / 注册 memory region。  
  - 清楚 NIC 应不应该、能不能对这块显存发起 DMA。  
  - 理解在这个平台上应该用哪块 NIC、哪组端口、怎样走机内/机间路由。  
  - 在这个组合上做过功能和性能测试，有明确的发布口径（写进“支持矩阵”）。
- 对 **Atlas 800T/800I A2/A3** 这类机器：  
  - HCCL 的多机场景是“有代码、有适配、有验证、有官方文档”的。  
- 对 **310P**：  
  - HCCL 没有为它提供多机场景的实现，也没有在 MindIE / vLLM-Ascend / HCCL 支持矩阵里把 310P 列为“多机支持对象”。

**3）从“公式”角度再看一遍：那一级 NIC 必须真实存在并被打通**

- 单机 P2P：  

  ```text
  GPU0(DMA) → PCIe/NVLink/HCCS → GPU1 显存
  ```

- 多机 RDMA：  

  ```text
  GPU 显存
    ↔(PCIe DMA by NIC)↔ NIC
    ↔(IB/RoCE)↔ 远端 NIC
    ↔(PCIe DMA by NIC)↔ 远端 GPU 显存
  ```

- 一旦在某个平台上，这个 “↔(PCIe DMA by NIC)↔” 环节在硬件 / 驱动 / IOMMU / 通信库任意一层没打通，**整条多机链路就断了**。

所以，更精确的话术是：

> “多机多卡比单机多卡额外要求的不只是‘多机版本的 NCCL/HCCL’， 
>
> 而是要有一整套 **NIC 与加速卡之间的 peer DMA 能力 + 驱动栈支持 + 官方验证**。”

**四、把这三条前提代入：310P vs Atlas A2/A3**

现在用同一套前提，对比两类典型产品。

- **310P：**
  - 形态：  
    - 一张“**NPU 作为 PCIe Endpoint 插到 CPU 主板上**”的推理解码卡。  
    - 自己不对外暴露 HCCS / 板级集群互联。  
    
  - 多机场景下，所谓“配好 RDMA 网卡”其实是：  
    - 主机（CPU）上插 RoCE / IB NIC，NIC 能 DMA 主机内存。  
    - 但**官方并没有承诺** NIC 能直接 DMA 310P 的 NPU 显存：  
      - IOMMU / 驱动 / 硬件路径是否允许 peer DMA？  
      - 是否做过这条链路的兼容性 / 性能验证？  
    - 从技术支持和文档来看，这条“NIC ↔ 310P 显存”的链路 **不在官方支持范围内**。  
    
  - 软件栈：  
    - HCCL 并没有为 310P 提供多机场景 backend。  
    
    - 官方技术支持给出的明确口径是：“**310P 多机不支持**”。  
    
    - 再结合 Atlas 300I Duo“>4 卡不在支持范围”的案例，可以看出： 
    
      这类 **PCIe 推理解码卡，并没有被纳入大规模多机多卡训推的正式产品线**。
  
- **Atlas 800T/800I A2/A3 等（基于 910B / A3）：**
  - 形态：  
    - 为集群设计的训练 / 推理服务器，**机内有 HCCS/超节点互联，机间有 RoCE/IB 网络**。  
  - 多机场景下：  
    - 硬件 + BIOS + 驱动一起打通了 NIC ↔ NPU 显存的 RDMA 能力（类似 GPUDirect RDMA）。  
    - HCCL 为这些设备实现了多机 AllReduce / Broadcast 等 collective 后端。  
  - 软件栈：  
    - vLLM-Ascend / MindIE / MindSpore 等多机多卡方案，都是**围绕这类服务器和 AI 集群（Atlas 900/9000 系列）设计和验证的**。

因此，对两类设备的“正确理解”应该是：

- 对 **A2/A3 这类“为集群设计的服务器”**：  
  - 单机多卡：GPU/NPU DMA Engine + PCIe/NVLink/HCCS。  
  - 多机多卡：在“有 RDMA NIC 且支持 peer DMA 的前提下”，**确实主要取决于 NCCL/HCCL 的多机实现**。
- 对 **310P 这类 PCIe 推理解码卡**：  
  - 单机多卡：只要本地 PCIe P2P / 驱动支持，确实可以做一定程度的多卡互传（前提是 MindIE/HCCL 等上层愿意支持）。  
  - 多机多卡：即便主机上有 RoCE/IB NIC，“NPU 显存 ↔ NIC RDMA Engine”这条通路在硬件 / 驱动 / 软件栈层面 **没有被官方打通 + 验证 + 背书**，HCCL 也没有对应实现，自然就“不支持多机”。

**五、回头再看你的那句总结：该怎么“改写”才准确？**

你原来的话是：

> “如此看的话，硬件是满足要求的，唯一可能多机比单机额外的要求就是需要NCCL/HCCL支持多机。我的理解对吗？”

更准确的版本可以改成：

> **对 A2/A3 等集群服务器来说**，硬件 + 驱动栈已经提供了 “NIC ↔ 加速卡显存” 的 RDMA 能力， 
>
> 在这个前提下，多机确实主要取决于 NCCL/HCCL 的多机实现。
>
> **但对 310P 这种 PCIe 插卡来说**，这个前提本身就不成立： 
>
> 不是“只差一个 NCCL 多机开关”，而是整条 “GPU 显存 ↔ NIC ↔ RDMA ↔ 远端 GPU 显存” 的通路
>
> 在硬件 / 驱动 / 通信库上都没有被官方设计、验证和支持。

用更短的一句话“定型”这件事，就是：

- **抽象层面**：你的思路是对的——单机靠 GPU DMA，跨机靠 NIC RDMA，NCCL/HCCL 负责调度。  

- **具体产品层（310P）**：它只被设计和验证成一张“单机/边缘推理解码卡”，

  单机内多卡 P2P 可以用，但那条 “GPU↔NIC↔网络↔NIC↔GPU” 的多机链路 + HCCL 多机后端**从一开始就不在它的产品定义里**， 

  所以官方说“多机不支持”并不是在“卡你一个开关”，而是整套前提就不存在。

这样，对于你当初的推理和疑问，分别从“抽象原理 + 具体产品”两个层面把 310P 为什么不能多机讲清楚了。

# 10. 单机 / 多机训推流程总览：从一行脚本到底层线缆

前面几章是“分模块讲清楚”：硬件是谁、DMA/RDMA 怎么跑、NCCL/HCCL 在哪一层。  
这一章我们把所有东西串成两个完整故事：

- 单机多卡：从 `loss.backward()` 到 “每张卡梯度都对齐”。
- 多机多卡：从 `loss.backward()` 到 “跨机所有卡梯度都对齐”。

然后再用同一套视角，看一眼推理时数据在单机 / 多机之间是怎么流的。

## 10.1 “单机多卡训练一次 step” 的全链路追踪（代码 → NCCL/HCCL → DMA → 硬件）

我们从一个最典型的场景开始：

- 单机 4 卡（GPU0~GPU3）。
- 使用数据并行（DDP）：
  - 每张卡上有一份完整的模型副本。
  - 每张卡处理 batch 的不同子集。
  - 每个 step 结束时，需要**把所有卡上的梯度平均一下**，保证每张卡的参数更新一致。

假设你写的训练代码是这样的（伪代码）：

```python
model = DDP(model, device_ids=[local_rank])
...
for input, target in dataloader:
    output = model(input)      # 前向
    loss = criterion(output, target)
    loss.backward()            # 反向，产生梯度
    optimizer.step()           # 用平均梯度更新参数
    optimizer.zero_grad()
```

我们沿着时间线，把一次 step 从上到下完整展开。

**Step 0：环境初始化（在第一个 step 之前）**

1. **框架初始化 DDP / 数据并行**
   - PyTorch DDP / MindSpore 并行接口：
     - 把你的 `model` 包装成一个并行模型对象。
     - 为每张卡（GPU0~GPU3）创建对应的 rank / 进程或线程。

2. **建立 NCCL/HCCL 通信上下文**
   - 调用诸如 `ncclCommInitRank` / `hcclCommInitRootInfo` 等：
     - 告诉通信库：“我们有 4 个 rank，分别在 GPU0~GPU3 上。”
     - 通信库内部：
       - 探测单机 GPU 拓扑（可用的 P2P、NVLink 等）。
       - 规划单机 Ring / Tree 拓扑。

3. **为参数分配梯度 buffer，并建立 P2P 映射**
   - 每个参数都会在每张 GPU 上对应一个梯度 buffer，比如：
     - GPU0: `grad_W0`, GPU1: `grad_W1`, …
   - 驱动 / runtime 尝试为 GPU 之间建立 P2P 访问：
     - GPU0 能通过 PCIe P2P / NVLink 直接访问 GPU1 显存。
     - 为这些映射在 IOMMU 里建表。

这些动作只在初始化做一次，之后每个 step 都可以依赖同一套通信环境。

---

**Step 1：前向计算（forward）**

在某个 step 内，每个 GPU 执行：

```python
output = model(input)
loss = criterion(output, target)
```

- GPU0~GPU3 各自用自己的那份参数，对各自的输入子 batch 做前向：
  - `model_0(input_0)`、`model_1(input_1)` ……
- 这一阶段：
  - 只涉及 **本地 GPU 上的矩阵乘 / 激活**。
  - 数据完全在各自 GPU 的显存内流动。
  - 不发生跨 GPU 通信（如果没有模型并行的话）。

此时每张卡上的状态大致是：

- 有一份参数副本（weights）。
- 为本 step 产生了一批激活（activations），为反向做准备。

---

**Step 2：反向传播（backward）——本地算梯度**

执行：

```python
loss.backward()
```

每张 GPU 上：

- 从 loss 对输出的梯度开始，一层层往前反传。
- 使用链式法则对每个参数算出本地梯度：
  - 对于某个参数张量 W：
    - GPU0 得到 `grad_W0`。
    - GPU1 得到 `grad_W1`。
    - ……

此时，假设共有 N 张卡：

- 每张卡上有一份本地梯度。
- 但还 **没有做 AllReduce**，每张卡的梯度只反映了本卡看到的子 batch 的信息。

---

**Step 3：触发梯度同步（调用 NCCL/HCCL AllReduce）**

DDP / 并行框架会在合适的时机（比如每个 bucket 的梯度 ready 时）调用：

```c
ncclAllReduce(grad_local, grad_local, count, datatype, ncclSum, comm, stream);
```

这一步从“你的 Python 程序视角”看不到，但从系统视角非常关键：

1. **框架层**
   - 检测到某个梯度 bucket 已经在所有参数上准备就绪。
   - 决定对这个 bucket 发起一次 AllReduce(sum)。
   - 把这次 AllReduce 的请求交给 NCCL/HCCL。

2. **NCCL/HCCL 层**
   - 使用第 5 章讲过的某种 AllReduce 算法（比如 Ring）。
   - 确定单机 4 卡的环：`GPU0 → GPU1 → GPU2 → GPU3 → GPU0`。
   - 把这个 bucket 的梯度 buffer 按 chunk 切分，排好每一轮谁给谁发哪个 chunk。

3. **准备调用 P2P / DMA**
   - 对单机 4 卡来说，所有发送 / 接收都是机内：
     - 不需要 NIC / RDMA。
     - 只需要 GPU DMA Engine + PCIe P2P / NVLink。

---

**Step 4：单机 4 卡 AllReduce 的一次“环跑”**

以某个 chunk 为例，单机内的一轮 AllReduce 通常会长这样：

1. NCCL 在每张 GPU 上发出本轮发送 / 接收计划：
   - GPU0：把 chunk0 发给 GPU1。
   - GPU1：把 chunk1 发给 GPU2。
   - GPU2：把 chunk2 发给 GPU3。
   - GPU3：把 chunk3 发给 GPU0。

2. 每一跳（例如 GPU0 → GPU1）内部发生的事情：

   - **控制平面**：
     - NCCL 在 GPU0 上调用 P2P copy 接口：

       ```c
       cudaMemcpyPeerAsync(dst=buf_1, dstDevice=1,
                           src=buf_0, srcDevice=0,
                           size=chunk_size, stream);
       ```

   - **数据平面**：
     - GPU0 的 DMA Engine 被配置好：
       - 源地址：GPU0 显存中的 `buf_0` 某一段。
       - 目标地址：GPU1 显存中对应的 `buf_1` 段（通过 P2P 映射）。
     - DMA Engine 发起 PCIe/NVLink 传输：
       - 读 GPU0 显存。
       - 通过 PCIe/NVLink 发送事务。
       - 写入 GPU1 显存。
     - 完成后，GPU0 / runtime 向 NCCL 报告 DMA 完成。

3. GPU1 收到 chunk0 后：
   - 在本地做一次累加：`grad_chunk0 = grad_chunk0 + recv_chunk0`。
   - 为下一轮可能的发送做好准备。

多轮之后：

每个 chunk 的和会分布在 4 张 GPU 上（Reduce-Scatter），再通过 All-Gather 环节广播回每个 GPU，使所有人持有同一份平均梯度。

---

**Step 5：平均梯度 & 参数更新**

AllReduce 完成后：

- 每张 GPU 上对应参数 W 的梯度，都变成了：

  $$
  \text{grad\_W\_avg} = \frac{1}{N} \sum_{i=0}^{N-1} \text{grad\_W\_i}
  $$

- 然后执行：

  ```python
  optimizer.step()
  ```

  - 实际上就是用 `grad_W_avg` 来更新本地的 W。
  - 由于每台卡上的 W 都用的是同一份平均梯度，所以更新后模型依然保持一致。

至此，“单机 4 卡训练一次 step”的关键路径就走完了：

```text
你的代码：
  loss.backward() → optimizer.step()

框架：
  backward 展开为很多算子 + grad buffer
  在合适时机调用 ncclAllReduce

NCCL：
  选择 AllReduce 算法（Ring / Tree）
  在单机内排好 P2P 拓扑和时序
  调用 CUDA/CANN P2P 接口

GPU 硬件：
  每一跳由源 GPU 的 DMA Engine 发起 P2P DMA
  数据走 PCIe/NVLink/HCCS
  目标 GPU 写入显存并累加
```

如果你能完整在脑子里“重放”上面这条时间线，就已经真正理解了“单机多卡训练一次 step 的全流程”。

## 10.2 “多机多卡训练一次 step” 的全链路追踪（代码 → NCCL/HCCL → RDMA → 硬件）

现在我们把场景升级为：

- 两台机器，每台 4 卡：

  ```text
  Machine A: GPU0, GPU1, GPU2, GPU3
  Machine B: GPU4, GPU5, GPU6, GPU7
  ```

- 使用数据并行：
  - 8 份模型副本。
  - 每张卡处理不同子 batch。
  - 每个 step 结束时，需要在所有 8 张卡之间做 AllReduce。

你的训练代码表面上仍然是同一行：

```python
loss.backward()
optimizer.step()
```

但 AllReduce 现在需要跨机器进行。

---

**Step 0：多机环境初始化**

与单机场景类似，但多了跨机部分：

1. **进程 / rank 与 GPU、机器的映射**
   - 例如：
     - rank 0~3 在 Machine A，对应 GPU0~3。
     - rank 4~7 在 Machine B，对应 GPU4~7。

2. **初始化 NCCL/HCCL 通信域**
   - 所有 rank 通过 TCP/SSH/bootstrap 通道互相交换：
     - 自己的 IP / IB LID / RDMA 地址信息。
   - 通信库建立包含 8 个 rank 的 `comm`。

3. **为 AllReduce 准备 RDMA 环境**
   - 在每个机器上：
     - 为本机参与 AllReduce 的梯度 buffer 调用 `ibv_reg_mr`。
     - 得到 memory region 和 rkey。
   - 各机器之间交换：
     - 每个 buffer 的地址 + rkey + 所在 rank / GPU 信息。

4. **拓扑规划**
   - 通信库分析：
     - 单机内：什么 P2P/NVLink 可用？
     - 机间：通过哪块 NIC / 哪个 IB/RoCE 交换机连接？
   - 决定最终的 AllReduce 计划：
     - 例如机内 4 卡做局部环，机间 2 节点做一层外部环。

这一切完成后，多机多卡的 AllReduce 才真正“准备好可以执行”。

---

**Step 1：前向 + 本地反向（和单机完全一样）**

每张卡仍然是：

```python
output = model(input)
loss = criterion(output, target)
loss.backward()   # 只算出本地的梯度
```

- 每个 rank / GPU 上：
  - 计算出各自的 `grad_W_i`。
  - 此时还没有跨机通信。

---

**Step 2：多机 AllReduce 触发（框架 → NCCL/HCCL）**

和单机一样，框架决定对某个梯度 bucket 做 AllReduce：

```c
ncclAllReduce(grad_local, grad_local, count, datatype, ncclSum, comm, stream);
```

区别在于：

- 现在这个 `comm` 包含 8 个 rank，跨越两台机器。

NCCL/HCCL 在内部会大致做两步拆分：

1. **机内阶段（Intra-node）**
   - 在每台机器内部，先在 4 卡之间做一次局部 Reduce-Scatter：
     - 只走 P2P / NVLink / HCCS。
   - 得到“本机聚合后”的部分结果。

2. **机间阶段（Inter-node）**
   - 在 Machine A 和 Machine B 之间做 AllReduce / AllGather：
     - 所有跨机通信通过 RDMA 网卡执行。

我们着重看第二阶段，因为这是多机多卡和单机场景的关键差异。

---

**Step 3：机内局部聚合（只用单机 DMA）**

这一步几乎等价于前面 10.1 里描述的单机 4 卡 AllReduce，只不过：

- 算的是“本机的部分和”，而不是全局和。
- 数据仍然完全留在本机。

对每个参数 / chunk 来说，经过机内 Reduce-Scatter 后，每个机器上的某一张 GPU（或多张）持有该 chunk 的“本机累加结果”。

---

**Step 4：跨机 AllReduce（引入 RDMA）**

现在，假设针对某个 chunk，NCCL 决定由：

- Machine A 的 GPU0 和 Machine B 的 GPU4 互相做 AllReduce。

所走的链路，与第 7 章 7.3 描述的一模一样：

1. **NCCL 在 Machine A 上调用 `ibv_post_send`**
   - 描述一次 RDMA Write / Read / Send：
     - 本地 buffer：GPU0 上的某段显存（通过 `ibv_reg_mr` 已注册）。
     - 远端 buffer：GPU4 上的某段显存（通过 rkey + 地址标识）。

2. **NIC(A) 通过 PCIe DMA 读 GPU0 显存**
   - 使用 RDMA Engine：
     - 读取缓冲区数据。
     - 切分为 RDMA 包。

3. **RDMA 网络传输**
   - 通过 IB / RoCE 交换机，发给 NIC(B)。 

4. **NIC(B) 通过 PCIe DMA 写 GPU4 显存**
   - 验证 rkey。
   - DMA 写入 GPU4 buffer。

5. **完成通知**
   - NIC(B) 写 CQ。
   - NCCL 在 Machine B 上读 CQ，确认这条写操作已经成功。

在 AllReduce 算法层面，可能是：

- 一部分数据从 A 到 B：B 做累加。
- 另一部分数据从 B 到 A：A 做累加。

最终：

- 两台机器都拿到了同一个 chunk 的全局和。

机内再做一次 AllGather，把全局和广播给每张卡，完成全局 AllReduce。

---

**Step 5：全局平均梯度 & 参数更新**

在跨机 AllReduce 完成之后：

- 所有 8 张卡上的每个参数梯度，都变成了全局平均梯度。
- 接下来的 `optimizer.step()`：
  - 在所有卡上同时对本地参数进行同样的更新。

至此，“多机多卡训练一次 step”的核心路径与数据流也完整走完。

把单机 4 卡 和 多机 8 卡 放在一起对比，你可以看到：

```text
【单机】
  backward：每卡本地算梯度
  AllReduce：
    - 全部走 GPU P2P / NVLink / HCCS
    - 源 GPU DMA Engine → PCIe/NVLink/HCCS → 目标 GPU 显存
  step：用平均梯度更新参数

【多机】
  backward：每卡本地算梯度（完全相同）
  AllReduce：拆成机内 + 机间两阶段
    - 机内：同单机，GPU DMA Engine P2P
    - 机间：NIC RDMA Engine + IB/RoCE + CQ
  step：用全局平均梯度更新参数（跨机一致）
```

关键差异只在于：**机间部分引入了 RDMA + NIC，而单机部分完全可以重用你对第 6 章内容的理解。**

## 10.3 推理场景下的区别：权重广播 / KV Cache / 张量切分 vs 梯度同步

到这里，你已经对“训练一次 step”从上到下非常熟了。

推理（尤其是多机多卡大模型推理）的流程可以用相同的框架来理解，只是：

- 没有反向传播。
- 没有梯度 AllReduce。
- 但有其它几种典型的通信模式。

我们仍然分“单机 / 多机”来说。

---

**单机多卡推理：**

典型场景：

- 模型较大（几十 B），一张卡显存不够，使用张量并行 / 切层等方式切在多张卡上。
- 或者模型能放下一张卡，但为了吞吐量，需要多卡并行跑不同请求。

关键数据流：

1. **初始化阶段的权重加载 / 广播**
   - 把模型权重从磁盘 / CPU 内存加载到各 GPU。
   - 可能：
     - 由一张卡加载后通过 P2P 广播到其它卡（NCCL Broadcast）。
     - 或者每张卡各自从 CPU 内存加载（多数框架支持这种方式）。
   - 这一步多数是一次性的，但数据量巨大。

2. **一次 forward 的数据路径**
   - 如果是纯数据并行推理（每卡一份完整模型）：
     - 各卡几乎独立运行，只在少数场景同步统计信息。
   - 如果是张量并行 / pipeline 并行：
     - 某一层的输出在卡之间拆分、交换：
       - 比如：
         - GPU0 负责权重矩阵的左半部分。
         - GPU1 负责右半部分。
       - 一次矩阵乘后，需要在 GPU0/1 之间交换 / 重排结果。
     - 这时，会有大量单机 P2P 通信（走 GPU DMA Engine + PCIe/NVLink）。

3. **KV Cache / 中间状态的传播**
   - 对于自回归大模型：
     - 每处理一个 token，会在各卡上产生新的 KV Cache。
     - 在张量并行 / 分层并行下，可能需要把某些 KV Cache 在卡之间同步或重排。
   - 这同样是通过 P2P 或集体通信接口实现。

从链路角度看：

- 单机推理的通信主要是：
  - 初始化时的大流量权重加载 / 广播。
  - 推理时的张量并行 / pipeline 并行产生的中间结果交换。

与训练的主要差别是：

- 训练：反复进行**大规模 AllReduce**（梯度同步）。
- 推理：更多是**张量切分 / 中间结果 / KV Cache 的点对点或小规模集体通信**。

---

**多机多卡推理：**

多机场景下，推理的链路可以看成：

- 在第 7、10.2 章中描述的训练链路中，**去掉反向链路和梯度 AllReduce**，换上：
  - 初始化时的权重分发。
  - 推理时的中间结果 / KV Cache 交换。

典型模式：

1. **初始化：权重分发 / 广播**
   - 可能模式：
     - 从一台“权重服务器”向所有节点广播。
     - 每台机器从共享存储 / CPU 内存直接加载。
   - 如果采用“广播”模式：
     - 链路和训练中的 AllReduce 很像：
       - 源：某台机器。
       - 目标：其余所有机器。
     - 实现：NCCL/HCCL 的 `Broadcast` / `AllGather` 接口 + RDMA / P2P。

2. **请求路由与结果聚合**
   - 在服务化框架（如 MindIE、vLLM-Ascend）中：
     - 前端收到请求后，可能按一定规则把请求拆分 / 路由到不同节点。
     - 推理结果可能需要跨机收集 / 合并。
   - 这部分通信，一部分在框架级别通过普通 RPC/TCP 完成，另一部分底层仍然可能用到 RDMA 加速某些内部通路。

3. **跨机张量并行 / KV Cache 同步**
   - 如果模型切分到多台机器上的多张卡：
     - 某些层的输出需要跨机交换：
       - 数据路径：`GPU 显存 ↔ NIC ↔ RDMA 网络 ↔ NIC ↔ GPU 显存`。
     - 和训练时的 AllReduce 不同的是：
       - 这里可能只是 AllGather / ReduceScatter / 点对点发送，而不是全量梯度求和。

总结来看：

- **链路形态相同**：
  - 单机：GPU DMA + PCIe/NVLink/HCCS。
  - 多机：NIC RDMA + IB/RoCE + CQ。
- **用途不同**：
  - 训练：反复做梯度 AllReduce，是“控制训练收敛”的中枢。
  - 推理：更多是一次性权重加载 + 每个 token 的中间结果/状态交换，是“控制吞吐和延迟”的关键。 

## 10.4 把所有名词放回大图：一张文字版“系统总览图”

最后，我们用一张“文字版总览图”把前面所有章节里出现的关键名词一一放回各自的位置。  

你可以把整套系统想成从上到下的 5 层：

```text
第 1 层：训练 / 推理代码（你的脚本）
第 2 层：深度学习框架（PyTorch / MindSpore / MindIE / vLLM 等）
第 3 层：通信库（NCCL / HCCL）
第 4 层：数据搬运层（GPU DMA Engine / NIC RDMA Engine）
第 5 层：互联与物理硬件（PCIe / NVLink / HCCS / IB / RoCE / 交换机 / 线缆）
```

把名词塞回去：

- **第 1 层：你的代码**
  - `loss.backward()`、`optimizer.step()`、`model(input)`。

- **第 2 层：框架**
  - PyTorch DDP、MindSpore 并行接口、MindIE 服务化、vLLM-Ascend 推理框架。
  - 把你的高层操作拆成：
    - 本地算子（矩阵乘、激活等）。
    - 多卡通信操作（AllReduce / Broadcast / AllGather 等）。

- **第 3 层：通信库（NCCL / HCCL）**
  - 决定：
    - 用 Ring 还是 Tree 还是分层 AllReduce。
    - 单机内的 P2P 拓扑、跨机的 RDMA 拓扑。
    - 每一步谁给谁发哪个 chunk。
  - 把“算法决策”翻译成：
    - 单机：CUDA/CANN P2P 调用。
    - 多机：RDMA verbs 调用（`ibv_reg_mr` / `ibv_post_send` / CQ 等）。

- **第 4 层：数据搬运层**
  - `GPU DMA Engine`：
    - 单机内：
      - `GPU0 显存 →(PCIe/NVLink/HCCS)→ GPU1 显存`。
  - `NIC RDMA Engine`：
    - 多机间：
      - `GPU0 显存 →(PCIe DMA)→ NIC(A) →(IB/RoCE)→ NIC(B) →(PCIe DMA)→ GPU4 显存`。
  - 这一层不做高层“谁先谁后”的决策，**只负责搬字节**。

- **第 5 层：互联与物理硬件**
  - `PCIe`：单机所有卡和 NIC 所在的主干总线，支持 P2P DMA。
  - `NVLink / NVSwitch`（NVIDIA） / `HCCS / 超节点`（昇腾）：
    - 在一台或多台机器内部构成更大的高速域。
  - `RDMA NIC`：IB NIC / RoCE NIC，内含 RDMA Engine。
  - `IB` / `RoCE`：跨机网络协议与交换体系。
  - 物理介质：铜缆 / 光纤 / AOC / 交换机机框等。

再叠加上 **设备型号与生态**：

- `Ascend 910B / A3`、`Atlas 800T/800I A2/A3`、`超节点服务器`：
  - 决定你能有多大的单机高速域、多好的 HCCS/NVLink 拓扑。
  - 决定 HCCL 在机内 / 机间可以怎么用。
- `310P`、`Atlas 300I Duo`：
  - 更多定位在单机推理/边缘推理，不提供全面的多机高速互联 / HCCL 多机场景。
- `MindIE`、`vLLM-Ascend`：
  - 在上述通信和硬件之上，提供更上层的推理服务框架和 API。

这样一来，你可以做到：

- 任意看到一个名词（IB / RoCE / RDMA / NCCL / HCCL / NVLink / HCCS / 超节点 / GPUDirect RDMA / 310P / Atlas 800T A2 ……），
- 都能在脑子里立刻把它放到：
  - “第几层？”
  - “参与算？参与搬？还是高速通道 / 硬件平台？”
  - “它对 AllReduce 或推理中的数据路径有什么影响？”

这就是本篇文档一开始你说的那个目标：“**不靠死记名词，而是靠一个完整的图景来理解单机多卡 / 多机多卡训推的底层逻辑**”。

# 11. 给后来者的阅读建议与延伸方向

这一章不是讲新概念，而是站在“你已经把这篇文档看完一遍”的角度，帮你规划：

- 回头再看时，**哪些章节值得多刷几次**。
- 如果想继续深入，**可以朝哪些方向查资料 / 看论文 / 看官方文档**。
- 如果你想从“懂原理”走到“能自己设计并行方案 / 做工程选型”，大概可以走哪些路径。

## 11.1 如果你刚入门，这几章建议重点多刷几遍

如果你是刚入门、第一次系统看单机多卡 / 多机多卡相关内容，建议这样使用这篇文档：

- **第一轮：顺读一遍，建立整体印象**
  - 不必纠结细节，先把“整张地图”装进脑子：
    - 第 1 章：整体大图，知道这篇文档要讲什么。
    - 第 2 章：谁负责算 / 谁负责搬。
    - 第 3 / 4 章：单机高速域 vs 跨机 RDMA 网络。
    - 第 6 / 7 / 8 / 10 章：单机、多机、通信库和完整训推流程。

- **第二轮：重点多刷几遍的章节**
  - **第 2 章 硬件角色总览**：
    - 谁是 GPU DMA Engine，谁是 NIC RDMA Engine，谁是 PCIe/NVLink/HCCS。
    - 建议能在纸上画出“谁可以 DMA 谁”的矩阵。
  - **第 3 章 单机内部高速互联**：
    - 把“普通 8 卡服务器 vs 超节点”的区别吃透。
    - 提醒自己：「本质在高速域边界，不在卡的数量」。
  - **第 4 章 RDMA / IB / RoCE / GPUDirect RDMA**：
    - 建议至少看两遍，从“网络 + RDMA”视角再串一次全局。
  - **第 6 章 单机多卡数据流**：
    - 直到你能闭眼重放一次 “GPU0 DMA Engine → PCIe/NVLink → GPU1 显存 → 完成通知”的全链路。
  - **第 7 章 多机多卡数据流**：
    - 建议和第 6 章一起对照看：
      - 看清楚“发起 DMA 的从 GPU 变成 NIC”这件事。
  - **第 8 章 通信库位置**：
    - 对理解 NCCL/HCCL 在整个栈中的作用非常关键。
  - **第 10 章 单机 / 多机训推总览**：
    - 建议当成“综合检查点”：
      - 每隔一段时间回头看一次，看看能不能在脑子里流畅地复述单机 / 多机一次 step 的全过程。

- **第三轮：当“参考手册”用**
  - **第 12 章 术语表 & 常见误解**：
    - 以后忘了某个名词，直接跳这里查。
  - **第 9 章 310P / A2/A3 选型**：
    - 真要在项目里做设备/架构选型时回来看。

## 11.2 推荐进一步深入的关键词与资料方向

如果你对这篇文档里的内容已经比较熟了，想进一步“把这块玩明白”，可以按下面几个方向延伸。

> 下面不直接给具体链接，而给出关键词，你可以根据自己的平台和时间去搜官方文档 / 论文 / 教程。

**1. 通信库与集体通信算法**

- 关键词：
  - “NCCL internals”“NCCL Ring AllReduce”“NCCL Tree AllReduce”“Hierarchical AllReduce”“HCCL AllReduce 实现原理”。
  - “MPI collectives”“MPI Allreduce algorithm”。
- 目标：
  - 了解不同 AllReduce 算法在不同拓扑下的复杂度和适用性。
  - 理解分层 AllReduce 如何优化跨机网络使用。

**2. RDMA 与 verbs 编程**

- 关键词：
  - “RDMA verbs 编程入门”“ibv_reg_mr ibv_post_send 教程”。
  - “RoCE PFC ECN DCQCN 调优”“InfiniBand credit-based flow control”。
- 目标：
  - 亲自用 verbs 写一个简单的 RDMA Write / Read 小程序（哪怕只在 CPU 内存上）。
  - 深化“Work Request → NIC → CQ”的直觉。

**3. GPU 互联与拓扑分析**

- 关键词：
  - “nvidia-smi topo -m 解释”“NVLink 拓扑结构”“NVSwitch topology”。
  - “Ascend HCCS 拓扑”“Atlas 超节点 互联结构”。
- 目标：
  - 学会看一台服务器的 GPU 拓扑图（谁和谁有 NVLink / HCCS、有没有跨 NUMA）。
  - 能从拓扑图上大致判断：
    - 哪些卡之间 P2P 更快。
    - 哪种并行配置更适合这台机器。

**4. 大模型并行策略**

- 关键词：
  - “Megatron-LM tensor parallelism”“pipeline parallelism”“DeepSpeed ZeRO”“3D parallelism”。
  - “MoE 并行”“expert parallelism”。
- 目标：
  - 把“数据并行 / 张量并行 / pipeline 并行 / ZeRO / 混合并行”这些词和“底下的 AllReduce / AllGather / P2P 消息模式”对上号。
  - 理解不同并行策略在网络 / 显存 / 吞吐之间的 trade-off。

**5. 昇腾 / Atlas 生态与官方文档**

- 关键词：
  - “昇腾 HCCL 多机部署指南”“Atlas 800T A2 用户指南”“Atlas 超节点 白皮书”。
  - “MindIE 多机推理实践”“vLLM Ascend 文档”。
- 目标：
  - 看清楚官方支持矩阵：
    - 哪些设备支持多机多卡训练 / 推理。
    - 哪些只支持单机 / 小规模场景。
  - 结合本篇文档的理解，带着问题去看官方文档，效果会更好。

## 11.3 如何从“理解原理”走向“真正能自己设计并行方案”

最后，如果你不满足于“只是搞懂原理”，而是想：

- 能自己参与到实际系统的设计 / 性能调优中；
- 或者在做项目选型时给出有依据的判断；

可以参考这样一条“进阶路线”。

**第 1 步：能自己画出完整的数据流**

- 目标：
  - 对一个给定的训练 / 推理任务（例如：多机多卡训练一个 70B 模型，或多机多卡推理一个 34B 模型），你能：
    - 画出单机内的数据流（梯度 / 激活 / KV Cache 的流向）。
    - 画出跨机的数据流（哪些 AllReduce / Broadcast / AllGather / P2P 发生在何时）。
    - 标出每一跳是 GPU DMA 还是 NIC RDMA，走 PCIe / NVLink / HCCS / IB / RoCE 的哪条线。

**第 2 步：能从拓扑推导出“合理”的并行策略**

- 目标：
  - 给你一台/一组实际机器（比如 8 卡 NVLink 服务器 × N 台、或 Atlas 超节点 × N）：
    - 能根据 GPU 拓扑和网络结构，判断：
      - 哪些卡适合做张量并行（在同一高速域内）。
      - 哪些适合做数据并行（跨高速域 / 跨机）。
      - 是否需要分层 AllReduce / 分层并行。
  - 能说出：
    - “在这套硬件上，TP=8 + DP=4 比 TP=16 + DP=2 更合理”，并能用网络和显存约束解释原因。

**第 3 步：能看懂并部分修改已有框架的并行配置**

- 目标：
  - 对于像 Megatron-LM / DeepSpeed / MindSpore 并行 / vLLM 这类框架：
    - 能看懂它们现有的并行策略配置（tensor\_model\_parallel\_size / pipeline\_model\_parallel\_size / data\_parallel\_size 等）。
    - 能根据自己机器的拓扑，调整这些数字，让系统不至于被某一条最慢链路卡死。
  - 能看懂框架日志里打印出的：
    - “用的是 Ring 还是 Tree / Hierarchical AllReduce”。
    - “各通信 group 的成员是哪些 rank / GPU”。

**第 4 步：能在遇到性能问题时做“有方向感”的排查**

- 目标：
  - 当训练 / 推理很慢时，你不会只是“瞎调 batch size / 学习率”，而是：
    - 先看 GPU 利用率和通信时间占比（profiling）。
    - 如果通信占比很高：
      - 查单机内是否有 P2P/NVLink 没启用的问题。
      - 查跨机是否因为 RoCE 配置不当导致丢包 / 抖动。
    - 如果只是个别卡 / 个别节点慢：
      - 查 GPU / NIC 是否在不利的 PCIe / NUMA 拓扑上。
  - 这时，本篇文档里的“单机 / 多机数据流图”和“互联架构解释”，就会成为你定位问题的“地图”。

**第 5 步：参与到选型与架构设计决策中**

- 目标：
  - 面对类似“310P vs Atlas 800T A2 / A3”“新集群用 IB 还是 RoCE”这类问题时：
    - 能从互联架构、HCCL 支持矩阵、框架兼容性、预算和目标规模等多角度给出合理建议。
  - 在设计一套新系统时：
    - 能够在“被硬件限制”和“利用硬件优势”之间做出清晰权衡。

达到这一步，你基本就已经从“被概念支配的新人”变成了“能看懂整个系统的工程师”。  
本篇文档没有也不可能覆盖所有实践细节，但希望它已经为你搭好了一套足够坚固的“认知骨架”，让你在今后继续学习 / 实践的路上不再那么迷茫。 

# 12. 附录：术语表与常见误解纠正

这一章是整篇文档的“速查表”和“纠偏区”。  

- 12.1：用一两句话重新解释本文中出现的关键术语。
- 12.2：列出你一开始最容易搞混的几组概念，对照纠正。
- 12.3：把几条最关键的“心智模型”再压缩一下，方便你以后回忆。

## 12.1 常见术语速查表（按模块归类）

**计算芯片 / 内存相关**

- **GPU / NPU**：负责跑矩阵乘、激活等算子本身的“算力芯片”，不直接讲网络协议，只通过 PCIe / NVLink / HCCS 等总线读写数据。
- **显存 / HBM**：挂在 GPU/NPU 旁边的高带宽内存，用来存模型权重、激活、中间张量、KV Cache 等，数据并行 / 张量并行本质都是在不同显存之间搬数据。
- **Device Buffer（设备缓冲区）**：在显存里划出来的一段连续区域，用来存一块要 AllReduce 的梯度、要发送的激活或从别处收到的结果。

**单机互联 / DMA 相关**

- **PCIe**：一台机器内部所有扩展卡（GPU、NIC、NVMe）插在上的“主干总线”，支持设备之间的 DMA（Peer-to-Peer），是单机 P2P 和 GPUDirect RDMA 的基础。
- **GPU DMA Engine / Copy Engine**：GPU 内部的硬件搬运单元，可以在显存与显存之间、显存与主机内存之间发起 DMA，不占用计算核心；单机多卡 P2P 的“执行者”。
- **PCIe P2P（Peer-to-Peer）**：一个 PCIe 设备（如 GPU0）直接通过 PCIe 总线 DMA 读写另一个设备（如 GPU1）的内存，而不经过 CPU 内存中转。
- **NVLink**：NVIDIA 的 GPU↔GPU 高速点对点互联，比 PCIe 更高带宽、更低延迟，底层还是基于 DMA 访问显存。
- **NVSwitch**：NVIDIA 为 NVLink 设计的“交换机芯片”，把多条 NVLink 接在一起，构成一个多 GPU 全互联的高速域。
- **HCCS**：华为昇腾体系里的高速互联，功能上接近 NVLink，用于在多颗 NPU/GPU 之间构建机内 / 板级高速域。

**跨机互联 / 网络相关**

- **以太网（Ethernet）**：最常见的网络体系，用在上网、RPC、数据库访问等场景，“尽力而为”（允许丢包，依赖上层重传）。
- **RDMA（Remote Direct Memory Access）**：允许一台机器的 NIC 直接 DMA 访问另一台机器的内存/显存，绕过对方 CPU / 内核协议栈，是大规模 AllReduce / 权重广播的底层能力。
- **InfiniBand（IB）**：一套为 RDMA 专门设计的独立网络体系，有自己的交换机、网卡、地址与流控，原生无丢包，常用于超算和高端训练集群。
- **RoCE（RDMA over Converged Ethernet）**：在以太网上实现 RDMA 的协议族，实质上是“在公路体系里划出 RDMA 高速车道”，但需要 PFC/ECN/DCQCN 等机制精心调教网络。 
- **RDMA NIC（HCA / RoCE NIC）**：支持 RDMA 的网卡，内部有 RDMA Engine，可以通过 PCIe DMA 直接读写本机内存/显存，并通过 IB/RoCE 网络把数据发给远端 NIC。

**通信库 / 软件栈相关**

- **NCCL**：NVIDIA 的多 GPU 通信库，实现 AllReduce / Broadcast / AllGather 等集体通信算法，是 PyTorch DDP 等的底层依赖。
- **HCCL**：华为昇腾生态的多 NPU 通信库，对应 NCCL，在 Ascend / Atlas 体系中负责 AllReduce 等多机多卡通信。
- **RDMA verbs（libibverbs）**：一组用户态 API，如 `ibv_reg_mr`、`ibv_post_send`，用来请求 RDMA NIC 注册内存 / 发起 RDMA 读写，本质是“和 NIC 驱动说话的接口”。
- **ibv_reg_mr**：将一段内存/显存注册为 RDMA memory region，锁定物理地址，通过 IOMMU 建立映射，并生成本地/远端访问钥匙（lkey/rkey）。
- **ibv_post_send**：把一条“我要做 RDMA Write/Read/Send”的请求投递到 NIC 的发送队列，真正的 DMA + 网络发送由 NIC 硬件异步执行。
- **CQ（Completion Queue）**：完成队列，NIC 在 RDMA 操作完成后会往里写一个条目，NCCL/HCCL 通过轮询 CQ 知道“这次发送/接收已经完成”。

**GPU 直连 / GPUDirect 相关**

- **GPUDirect RDMA**：让 RDMA NIC 能像访问主机内存一样，直接通过 PCIe DMA 访问 GPU 显存，避免“显存→CPU内存→网络→CPU内存→显存”的中转。
- **GPUDirect P2P（单机）**：让一张 GPU 的 DMA Engine 可以通过 PCIe P2P 直接读写另一张 GPU 的显存，是单机 P2P copy 的基础。

**并行与集体通信相关**

- **数据并行（Data Parallel）**：每张卡一份完整模型，处理不同子 batch，然后用 AllReduce 同步梯度。
- **张量并行（Tensor Parallel）**：把一个大矩阵或一部分网络权重按维度切到多张卡上，需要在每层前后交换 / 聚合张量。
- **Pipeline 并行**：把网络的不同层段切到不同卡/节点上，以流水线方式处理 batch。
- **AllReduce**：所有参与设备先本地有一个值，做归约（如求和），然后把归约结果分发给所有设备（人人都有同一份结果）。
- **Ring AllReduce**：常见 AllReduce 算法，把 N 个参与者排成一个环，数据按 chunk 在环上多轮传递 / 累加，带宽利用率高。
- **Reduce-Scatter / All-Gather**：常与 AllReduce 结合使用的两个阶段：前者做分布式归约并“散开”到各卡，后者把结果再“聚合”回每卡。

**设备与平台相关（昇腾 / Atlas / 推理框架）**

- **Ascend 910B / A3**：华为昇腾高性能训练/推理芯片系列，A2/A3 服务器的底层算力核心。
- **Atlas 800T/800I A2/A3**：基于 910B / A3 的训练/推理服务器，设计用于多机多卡集群场景，拥有 HCCS / 超节点互联结构。
- **超节点（SuperNode）**：在 A3 等平台中，通过板级/光模块高速互联把多卡合成一个更大的统一高速域，从“单机 8 卡”扩展到 16/32 卡级别的“单高速域”。
- **Atlas 900/9000 A2/A3 SuperPoD**：由多台 Atlas 800T/800I 组成的大规模训练集群，类似 NVIDIA DGX SuperPOD。
- **310P**：面向单机/边缘推理解码的 PCIe 昇腾卡，没有 HCCS 板级互联，HCCL 多机场景不支持，架构上不适合作为多机多卡训练/推理节点。
- **Atlas 300I Duo**：PCIe 推理卡，官方文档对单服务器最大卡数有明确上限（如 ≤ 4），不被定位为集群训练主力。
- **MindIE**：华为围绕昇腾做的推理服务框架，用于部署推理服务、多机推理协同等。
- **vLLM-Ascend**：vLLM 在昇腾平台上的版本，支持在 Atlas A2/A3 系列上进行高性能大模型推理。

## 12.2 容易混淆的一些点：对照纠正

这里列几组你之前反复问、也最容易搞混的概念，用一行“要点”纠正。

**IB vs RoCE**

- 容易误解：
  - “RoCE 是 IB 的一种？” 或 “IB=光纤，RoCE=网线？”
- 正确理解：
  - IB 和 RoCE 是 **两套不同的网络体系**，唯一共同点是都支持 RDMA：
    - IB：从头为 RDMA 设计的一整套专用“高铁系统”，有自己的交换机 / 网卡 / 协议。
    - RoCE：在以太网上实现 RDMA 的协议，在“公路体系”里划出 RDMA 专用车道。
  - 是否用光纤还是铜缆，是物理层实现问题，**与 IB/RoCE 的本质区别无关**。

**单机 DMA vs 多机 RDMA**

- 容易误解：
  - “多机跨机是不是也用 GPU 的 DMA Engine 搬数据？”
- 正确理解：
  - **单机内部**：
    - 搬运主体：源 GPU 的 DMA Engine。
    - 通路：PCIe P2P / NVLink / HCCS。
  - **多机之间**：
    - 搬运主体：NIC 上的 RDMA Engine，通过 PCIe DMA 读/写 GPU 显存。
    - 通路：`GPU 显存 ↔ NIC ↔ IB/RoCE ↔ NIC ↔ GPU 显存`。
  - GPU 在多机场景中只是“被 DMA 的一端”，**不直接发网络流量**。

**NVLink vs NVSwitch**

- 容易误解：
  - “NVLink=点对点线，NVSwitch=多机互联？”
- 正确理解：
  - NVLink：GPU 与 GPU 之间的高速点对点链路，就像一条非常粗的“专用网线”。
  - NVSwitch：专门为 NVLink 设计的交换芯片，把多条 NVLink 接在一起，形成一个多 GPU 全互联的“交换结构/背板”。
  - 二者一起构成单机或机内的大高速域，但**不直接负责跨机互联**（跨机仍然依赖 IB/RoCE 等）。

**超节点 vs 普通 8 卡服务器**

- 容易误解：
  - “超节点 = 比 8 卡更多卡的服务器，只是名字不同？”
- 正确理解：
  - 普通 8 卡服务器：高速域的边界往往就在“单机 8 卡”这一层，8 卡之间用 NVLink/HCCS，跨机走 RoCE/IB。
  - 超节点：通过板级/光模块互联，把高速域扩展到跨板 / 跨机箱的 16 / 32 卡，**在更大范围内保持“单高速域”**。
  - 关键不是卡数，而是：
    - **高速互联域的物理边界被拓展到了多板 / 多机箱级别**。

**RDMA vs GPUDirect RDMA**

- 容易误解：
  - “RDMA 本来就可以直接访问 GPU 显存？”
- 正确理解：
  - 普通 RDMA：
    - 最初设计主要面向 CPU 内存（主机内存）。
    - 跨机的数据流通常是：显存 → CPU 内存 → RDMA → 远端 CPU 内存 → 显存。
  - GPUDirect RDMA：
    - 在驱动 / IOMMU 支持下，让 NIC 能像访问主机内存一样，直接 DMA 访问 GPU 显存。
    - 流程变成：`GPU 显存 →(PCIe DMA by NIC)→ NIC → 网络 → NIC →(PCIe DMA by NIC)→ 远端 GPU 显存`。
  - 本质差异是：**是否需要 GPU→CPU 内存中转**。

**NCCL/HCCL vs GPU/NIC**

- 容易误解：
  - “NCCL 在搬数据” 或 “GPU 在发网络包”。
- 正确理解：
  - NCCL/HCCL：
    - 决定用哪种 AllReduce / Broadcast 算法。
    - 规划拓扑和时序：谁给谁发哪个 chunk。
    - 调用底层接口（P2P 或 RDMA verbs），但**不亲自搬数据**。
  - GPU DMA Engine / NIC RDMA Engine：
    - 才是真正读写显存 / 内存的“搬运工”。
  - GPU 自己不会发 IB/RoCE 包，只会通过总线（PCIe/NVLink/HCCS）参与 DMA。 

**310P / 300I Duo vs Atlas 800T/800I A2/A3**

- 容易误解：
  - “都是昇腾芯片，只要有 HCCL，任何卡都能凑在一起做多机多卡训练/推理”。
- 正确理解：
  - 310P / 300I Duo：
    - PCIe 推理解码卡，缺少集群级 HCCS/板级互联和多机场景的软件栈支持。
    - 官方文档和技术支持都明确多机/大卡数场景“不在支持范围内”。
  - Atlas 800T/800I A2/A3 + Atlas 900/9000：
    - 从硬件（高速互联）到软件（HCCL / MindIE / vLLM-Ascend 支持矩阵）都为多机多卡训练/推理设计。
  - → 工程选型时要尊重这一点，而不是只看“芯片型号”。

## 12.3 本文中用到的几个关键“心智模型”小结

最后再把几条最重要的心智模型压缩成几行，方便你以后快速回忆。

- **层级模型（5 层）**

  ```text
  第 1 层：你的训练 / 推理脚本
  第 2 层：框架（PyTorch / MindSpore / MindIE / vLLM-Ascend）
  第 3 层：通信库（NCCL / HCCL）
  第 4 层：数据搬运（GPU DMA Engine / NIC RDMA Engine）
  第 5 层：互联与硬件（PCIe / NVLink / HCCS / IB / RoCE / 交换机 / 线缆）
  ```

  任何一个名词，都能放进这五层里的某一层或两层之间的接口。

- **单机 vs 多机的“执行者”差异**

  ```text
  单机：源 GPU 的 DMA Engine
        →(PCIe / NVLink / HCCS)→ 目标 GPU 显存

  多机：源 GPU 显存
        →(PCIe DMA by NIC A)→ NIC(A)
        →(IB / RoCE)→ NIC(B)
        →(PCIe DMA by NIC B)→ 目标 GPU 显存
  ```

  单机和多机的 AllReduce / Broadcast 在算法层看起来类似，但执行者和通路不同。

- **IB vs RoCE 的直觉类比**

  - IB：为高性能计算专门修的“高铁网络”。
  - RoCE：在现有“高速公路”上划出的 RDMA 专用车道。
  - 两者都可以跑 RDMA，只是一个是专用铁轨，一个是改造过的公路。

- **“做什么”和“怎么搬”的分工**

  - “做什么”（AllReduce 还是 Broadcast、Ring 还是 Tree）：
    - 由通信库（NCCL/HCCL） + 框架决定（控制平面）。
  - “怎么搬”（DMA 谁发起、走哪条总线/网络）：
    - 由 GPU DMA Engine / NIC RDMA Engine + PCIe/NVLink/HCCS/IB/RoCE 决定（数据平面）。

只要记住这几条心智模型，你以后遇到任何新的“并行框架 / 通信库 / 集群拓扑”文档时，都可以用这套“坐标系”去快速定位：它到底在哪一层引入了什么东西，底下又是踩着哪些熟悉的 DMA / RDMA 机制在跑。  

