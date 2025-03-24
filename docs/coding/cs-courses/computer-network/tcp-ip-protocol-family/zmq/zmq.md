# ZeroMQ通信

* [返回上层目录](../tcp-ip-protocol-family.md)
* [ZMQ介绍](#ZMQ介绍)
* [入门示例](#入门示例)
* [ZMQ通信模式](#ZMQ通信模式)
  * [请求响应模式](#请求响应模式)
  * [发布订阅模式](#发布订阅模式)
  * [任务管道模式](#任务管道模式)
  * [一对一通信](#一对一通信)
  * [ROUTER/DEALER模式](#ROUTER/DEALER模式)
* [在C/C++中集成ZMQ](#在C/C++中集成ZMQ)
* [问题解答](#问题解答)



# ZMQ介绍

ZeroMQ（简称ZMQ）是一个**基于消息队列的多线程网络库**，其对套接字类型、连接处理、帧、甚至路由的底层细节进行抽象，提供跨越多种传输协议的套接字。

ZMQ是网络通信中新的一层，**介于应用层和传输层之间**（按照TCP/IP划分），其是一个可伸缩层，可并行运行，分散在分布式系统间。

ZMQ不是单独的服务，而是一个嵌入式库，它封装了网络通信、消息队列、线程调度等功能，向上层提供简洁的API，应用程序通过加载库文件，调用API函数来实现高性能网络通信。

# 入门示例

废话不多说，先看个入门示例找找感觉：

服务端：

```python
#server.py
import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
# socket.bind("tcp://*:5555")
socket.bind("tcp://127.0.0.1:5555")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)

    #  Do some 'work'
    time.sleep(2)

    #  Send reply back to client
    socket.send(b"World")
```

客户端（还支持多个客户端连接服务端）：

```python
# client.py

import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
while True:
    socket.send(b"Hello")
    #  Get the reply.
    message = socket.recv()
    print(f"Received reply [ {message} ]")
```

上面的代码属于请求响应模式

# ZMQ通信模式

这里简要介绍ZMQ常用的通信模式

更详细的看这个，如果要看详细介绍，一定看[ZeroMQ通信模式详解](https://www.jianshu.com/p/d5730baa14b4)

## 请求响应模式

```
ZMQ_REQ <--> ZMQ_REP
```

- 常规搭配：`ZMQ_REQ` + `ZMQ_REP`
- 带输入负载(Round Robin)均衡搭配：`ZMQ_REQ` + `ZMQ_ROUTER`
- 消息分发搭配：`ZMQ_ROUTER` + `ZMQ_DEALER`
- 带输出负载(load-balance)均衡搭配：`ZMQ_DEALER` + `ZMQ_REP`

`ZMQ_REQ`模式在发送消息时，`ZMQ`底层会在消息内容头部插入一个空帧，在接收消息时，会去掉空帧，将内容返回给应用层。

`ZMQ_REP`模式在接收消息时，会将消息空帧之前的信封帧保存起来，将空帧之后的内容传给上层应用。上层应用在响应消息时，底层会在响应消息前加上空帧以及对应请求的信封帧。

`ZMQ_ROUTER`模式在接收消息时，`ZMQ`底层会在消息头部添加上一个信封帧，用于标记消息来源。该信封帧可由发送端指定（调用`zmq_setsockopt(ZMQ_IDENTITY)`），也可由接收端自动生成唯一标识作为信封帧。在发送消息时，将信封帧之后的内容发送到以信封帧为标识的地址。

`ZMQ_DEALER`模式，对接收到的消息公平排队fair-queue，以Round-Robin方式发送消息。

**ZMQ_REQ 和 ZMQ_REP 的通信规则：**

1. **客户端（REQ）**：
   - 必须先发送一条消息（`send`），然后才能接收一条消息（`recv`）。
   - 不能连续发送多条消息。
2. **服务端（REP）**：
   - 必须先接收一条消息（`recv`），然后才能发送一条消息（`send`）。
   - 不能连续发送多条消息。



## 发布订阅模式

```
ZMQ_SUB <-- ZMQ_PUB
```

那PUB-SUB模式是不是有点类似UDP的网络传输的特性？就是需要即时接受，发送过来不接收就会被丢弃掉，所以是实时的

> 是的，**PUB-SUB 模式** 在一些方面确实与 **UDP**（无连接的网络协议）有相似之处，特别是在 **即时性** 和 **消息丢失** 的处理上。两者都强调实时通信，并且在某些情况下，如果消息没有被及时接收，它们会丢失。我们可以从以下几个方面来理解这种相似性：
>
> ### **1. 实时通信**
>
> - UDP 特性：
>   - UDP 是一种 **无连接的协议**，它不保证消息的传递可靠性或顺序。如果接收方没有准备好接收消息，或者接收方丢失了数据包，那么这些数据会丢失。UDP 的设计目标是 **低延迟** 和 **高效传输**，而不关注消息是否可靠送达。
> - PUB-SUB 模式特性：
>   - 在 **PUB-SUB** 模式中，**发布者（PUB）** 广播消息，**订阅者（SUB）** 接收消息。类似于 UDP，如果订阅者在消息发布时没有准备好接收，消息会丢失，且 **没有确认机制**。因此，消息的传递类似于 UDP 那样是即时的，并且没有保证消息会被接收或持久化。
>
> ### **2. 实时性和消息丢失**
>
> - UDP：
>   - 由于 UDP 不进行数据确认和重传，它非常适合实时应用场景，如视频流、语音通信等，虽然这些应用可能会丢失一些数据包，但对于实时性要求较高的应用来说，丢失一些数据是可以接受的。
> - PUB-SUB：
>   - **PUB-SUB** 模式也强调实时性，消息一旦发布，订阅者就必须在短时间内接收消息。如果订阅者没有及时接收消息，或者没有连接到发布者，消息就会被丢弃。这类似于 UDP 的特性：没有重传、没有存储、没有确认机制，数据就是丢失了。
>
> ### **3. 消息丢失和无持久化**
>
> - UDP：
>   - UDP 没有重发机制，也不保证消息的到达。如果接收方丢失了数据包，它无法请求重传。应用层需要处理数据丢失的情况。
> - PUB-SUB：
>   - 同样，**PUB-SUB** 模式中的发布者（`PUB`）不会保存消息。如果订阅者没有及时接收消息，消息将会丢失，并且 ZeroMQ 本身不提供持久化机制（除非使用其他系统，如数据库、消息队列等）。
>
> ### **4. 无连接性**
>
> - UDP：
>   - UDP 是无连接的协议，它发送数据包时不需要建立和保持连接。每个数据包都是独立发送的。
> - PUB-SUB：
>   - **PUB-SUB** 模式也不需要持续的连接，**订阅者** 可以在任何时刻连接到 **发布者**，并开始接收消息。发布者发送的消息是广播形式的，订阅者只要连接上并且订阅了相应的主题，就能接收到从该时刻起发布的消息。
>
> ### **5. 适用场景**
>
> - **UDP**：适用于实时性要求高、对丢失数据包宽容的场景，比如 **视频流**、**实时语音**、**在线游戏**。
> - **PUB-SUB**：适用于 **广播消息** 的场景，其中一些订阅者可能错过消息，但这对应用的影响不大。例如，**新闻广播**、**系统日志广播**，或者需要向多个接收方发送实时数据流（如股票价格、天气预报等）。
>
> ### **总结：PUB-SUB 与 UDP 的相似性**
>
> - **消息丢失**：与 UDP 类似，**PUB-SUB** 模式中如果消息没有及时接收，它会被丢弃。
> - **实时性要求**：两者都强调低延迟和实时传输，不保证消息的可靠性。
> - **无连接和无确认**：两者都不进行连接管理和确认机制，消息传递依赖于接收方的准备情况。
>
> 尽管 `PUB-SUB` 和 UDP 在某些方面有相似性，它们的应用场景和实现机制仍然存在差异。ZeroMQ 作为一个高性能的消息传递库，可以用于很多不同类型的场景，包括需要消息广播的场景，而 UDP 更多的是网络层的传输协议。
>
> 希望这些解释能够帮助你更好地理解 **PUB-SUB** 和 **UDP** 之间的相似性！

## 任务管道模式

```
ZMQ_PUSH -> [ ZMQ_PULL , ZMQ_PUSH] --> ZMQ_PULL
```

## 一对一通信

```
ZMQ_PAIR <--> ZMQ_PAIR
```

## ROUTER/DEALER模式

**ROUTER/DEALER 模式**是一种用于多对多通信的消息传递模式。可以把它想象成一个“客服中心”：

- **ROUTER（路由器）** → 就像总机，负责接收来自多个“客户”（DEALER）的请求，记录客户身份，并将回复准确地发回给对应的客户。
- **DEALER（经销商）** → 就像客户，负责主动向总机发送请求，接收来自总机的回复。

在强化学习中，Python 端作为 ROUTER，管理多个 C++ 仿真环境（DEALER）的请求，分发指令并收集反馈，方便实现并行训练。

**下面根据强化学习的需求来写出的代码**：

- **ROUTER 是接收多个 DEALER 请求的角色，处理这些请求后返回响应。**
  - **Python端**确实是接收来自多个 **C++仿真环境**（DEALER）的请求（即 state 数据），并根据这些请求计算出 **action**，然后将结果返回给对应的 **C++仿真环境**（DEALER）。因此，**Python端**的确应该是 **ROUTER**，负责处理并汇总来自多个 **C++端** 的请求，然后返回响应。
- **DEALER 是发起请求并等待响应的角色。**
  - **C++端**（仿真环境）每一轮会发送 **state** 数据给 Python端（ROUTER），然后等待 **Python端** 返回对应的 **action** 数据，再继续执行下一轮仿真。**C++端**作为 **DEALER**，是发起请求并等待响应。

结论：

- 在你的场景中，**Python端应该是ROUTER**，它接收来自多个 **C++端（DEALER）** 的请求，并返回 **action**。
- **C++端应该是DEALER**，它发送 **state** 数据给 **Python端（ROUTER）**，并等待 **action** 的返回。

以下是 Python 端（ROUTER）和 C++ 端（DEALER）的示例代码：

**Python 端（ROUTER）**

在 Python 端，我们将使用 `zmq.ROUTER` 来监听来自 C++ 端的请求，并发送相应的响应。因为 Python 端会处理来自多个 C++ 客户端的请求，并且需要返回不同的 `action`。

```python
import zmq
import json

# 创建一个ROUTER socket
context = zmq.Context()
router_socket = context.socket(zmq.ROUTER)
router_socket.bind("tcp://*:5555")  # 绑定到某个端口，监听来自 C++ 的请求

print("Python端（ROUTER）已启动，等待来自C++端的请求...")

while True:
    # 等待接收来自 C++ 端的消息
    identity, msg = router_socket.recv_multipart()
    
    # 解析接收到的消息（假设是JSON格式的state）
    state = json.loads(msg)
    print(f"接收到来自 C++ 端的state: {state}")
    
    # 在这里，进行你自己的强化学习模型推理
    action = {"action": "some_action"}  # 示例的action，这里可以是你强化学习的决策
    
    # 将action作为响应发送回C++端
    router_socket.send_multipart([identity, json.dumps(action).encode()])
    print(f"发送action给 C++ 端: {action}")
```

**C++ 端（DEALER）**

在 C++ 端，我们将使用 `zmq.DEALER` 来发送请求（即state数据）给 Python 端（ROUTER），并接收 Python 端返回的 `action` 数据。

```c++
#include <iostream>
#include <zmq.hpp>
#include <nlohmann/json.hpp>  // 用于处理JSON数据

using json = nlohmann::json;

int main() {
    // 初始化 ZeroMQ 上下文
    zmq::context_t context(1);
    
    // 创建 DEALER socket
    zmq::socket_t dealer_socket(context, zmq::socket_type::dealer);
    dealer_socket.connect("tcp://localhost:5555");  // 连接到 Python 端（ROUTER）
    
    std::cout << "C++端（DEALER）已连接到Python端（ROUTER）..." << std::endl;
    
    // 模拟一个状态数据
    json state = {
        {"position", {1.0, 2.0}},
        {"velocity", {3.0, 4.0}},
        {"altitude", 1000}
    };
    
    while (true) {
        // 发送状态数据给 Python 端
        dealer_socket.send(zmq::buffer(state.dump()), zmq::send_flags::none);
        std::cout << "发送state给 Python 端: " << state.dump() << std::endl;
        
        // 接收来自 Python 端的 action 数据
        zmq::message_t response;
        dealer_socket.recv(response, zmq::recv_flags::none);
        
        // 解析 Python 端返回的 action
        json action = json::parse(response.to_string());
        std::cout << "接收到来自 Python 端的 action: " << action.dump() << std::endl;
        
        // 在这里，你可以根据action进行下一步仿真
    }

    return 0;
}
```

解释：

1. **Python端（ROUTER）**：
   - 监听来自 C++ 端（DEALER）的请求。
   - 解析接收到的 `state` 数据。
   - 基于强化学习模型生成一个 `action`，并将其发送回 C++ 端。
   - `ROUTER` 会根据请求的身份将响应发回给对应的 `DEALER`（C++端），从而保证每个客户端都有唯一的通信通道。
2. **C++端（DEALER）**：
   - 向 Python 端（ROUTER）发送请求（即当前的状态 `state`）。
   - 等待并接收来自 Python 端的 `action`。
   - 根据接收到的 `action` 进行下一步仿真操作。

总结：

- **Python端** 是一个 `ROUTER`，它监听来自多个 C++ 客户端（DEALER）的请求，并返回相应的响应。
- **C++端** 是一个 `DEALER`，它发送请求（即状态信息）到 Python 端，并等待从 Python 端收到的响应（即动作信息）。

这样，你就可以按照这种结构来进行多 C++ 环境的并行仿真，并且在 Python 端进行强化学习训练了。

### 疑问

#### 如何启动多个 C++ 端

你可以启动多个 C++ 进程，每个进程都包含一个 `DEALER` socket，与 Python 端的 `ROUTER` 进行通信。这些 C++ 进程可以并行运行，并通过 ZeroMQ 与 Python 端进行消息交换。

**示例说明：**

1. **多个 C++ 进程**：你可以通过在多个 C++ 进程中运行相同的代码，每个进程创建一个 `DEALER` socket。每个 `DEALER` 会连接到 Python 端的 `ROUTER`。
2. **Python 端**：Python 端的 `ROUTER` 会管理与多个 `DEALER` 的连接，它会根据每个 `DEALER` 的身份区分并处理不同的请求。

**如何管理多个 C++ 进程（仿真环境）？**

每个 C++ 进程都会有自己的 `DEALER` socket 和身份标识，它们向 Python 端的 `ROUTER` 发送消息并接收响应。Python 端不需要关心 C++ 端有多少个，它会处理所有来自 C++ 端的请求。

启动多个 C++ 实例：

你可以启动多个 C++ 进程，每个进程都会创建一个 `DEALER` socket，连接到 Python 端的 `ROUTER`。例如，通过以下命令启动多个 C++ 进程：

```bash
./cpp_dealer_program &  # 启动多个仿真环境
./cpp_dealer_program &
```

每个 `cpp_dealer_program` 进程都会在其独立的线程中与 Python 端进行通信。

总结：

- Python 端的 `ROUTER` socket 可以管理多个 C++ 端（`DEALER`）的请求。
- 每个 C++ 进程都可以创建自己的 `DEALER` socket，连接到 Python 端，进行通信。
- Python 端会根据 C++ 端的身份区分不同的请求，并发送相应的响应。

这种模式能够实现分布式并行仿真训练，适应你后续的升级需求。

**如果同时启动了多个c++线程，那么python端的identity, state_msg = router_socket.recv_multipart()接收到的消息是什么格式呢？**

当 Python 端的 `ROUTER` 接收到来自多个 C++ `DEALER` 的消息时，`recv_multipart()` 会返回一个包含多个部分的列表，每个部分代表不同的消息段。在这个场景下，`recv_multipart()` 将会返回两个部分：

1. **身份标识符（Identity）**：这部分是 C++ `DEALER` 发送的标识符，用于区分不同的 C++ 实例（线程）。每个 `DEALER` 在连接时都会发送一个唯一的标识符，`ROUTER` 使用这个标识符来区分不同的 C++ 端。
2. **状态消息（State Message）**：这是 C++ 端发送的状态数据。它可以是你从仿真环境中提取的状态数据（如飞机的位置、速度等信息）。

**格式举例：**

假设有两个 C++ 进程（`DEALER`），它们同时向 Python 端（`ROUTER`）发送消息：

- **第一个 C++ 进程** 发送的消息：
  - Identity: `"dealer_1"`
  - State Message: `"state_data_1"`
- **第二个 C++ 进程** 发送的消息：
  - Identity: `"dealer_2"`
  - State Message: `"state_data_2"`

在这种情况下，Python 端的 `recv_multipart()` 接收到的消息会是这样的格式：

```python
identity, state_msg = router_socket.recv_multipart()

# identity: b'dealer_1'  (C++ 端发送的标识符)
# state_msg: b'state_data_1'  (C++ 端发送的状态数据)
```

**解释：**

- **identity** 是一个字节串（`bytes`），表示 C++ 端的标识符。你可以将其转换为字符串。
- **state_msg** 是 C++ 端发送的状态数据，同样是字节串。

**总结：**

每次接收到的消息由两个部分组成：

1. **身份标识符（Identity）**，标识哪个 C++ 端发送的消息
2. **状态消息（State Message）**，C++ 端发送的状态数据

Python 端使用 `identity` 来区分不同的 C++ 进程，并对每个 `DEALER` 的请求做出相应的响应。

> 明白了，我还以为是identity=[identity_1, identity_2]， state_msg = [state_msg_1, state_msg_2]，原来是不论有几个c++线程在给python发，python每次收到了数据都是其中一个，都是按照c++发送的数据来接收的，c++发送的数据会被存储在一个缓存中，python轮询按照先后顺序读取缓存中的c++信息？

对！你完全理解对了！👍

具体来说，`ROUTER` socket 维护了一个**内部消息队列（缓存）**，用于存储来自所有 `DEALER` 端的消息。这个缓存是按照**先进先出（FIFO）**的顺序来组织的。

详细过程是这样的：

1. **多个 C++ DEALER 端同时发送消息：**
    每个 `DEALER` 端在发送消息时，都会带上一个隐含的身份标识符（由 ZeroMQ 自动生成并在底层封装）。
2. **Python 端（ROUTER）接收消息时：**
   - Python 端通过 `recv_multipart()` 方法从缓存中取出一条完整的消息。
   - 这条消息会包含两个部分： 
     - 第一个部分是**身份标识符**（`identity`），ZeroMQ 会自动将其附在消息前面，用于区分是哪个 `DEALER` 端发来的。
     - 第二个部分是**实际消息内容**（`state_msg`），即 C++ 端发送的数据。
3. **Python 端轮询处理：**
   - `ROUTER` socket 通过轮询机制，按照 FIFO 的顺序从消息队列中读取消息。
   - 每次调用 `recv_multipart()`，都会取出缓存中最早的一条完整消息（包括身份和数据）。
4. **Python 端根据身份发送回复：**
   - Python 端根据接收到的身份标识符，发送特定的响应回到相应的 `DEALER` 端。
   - `send_multipart()` 通过附带身份标识符，ZeroMQ 会确保消息被发回给正确的 `DEALER` 端。

**示例流程（以两个 C++ 端为例）：**

| 时间 | 操作                                        | 发送方        | 接收方        | Python 队列状态                        |
| ---- | ------------------------------------------- | ------------- | ------------- | -------------------------------------- |
| `T1` | C++ 端 1 发送 state_1                       | C++ DEALER 1  | Python ROUTER | `[ (id_1, state_1) ]`                  |
| `T2` | C++ 端 2 发送 state_2                       | C++ DEALER 2  | Python ROUTER | `[ (id_1, state_1), (id_2, state_2) ]` |
| `T3` | Python 调用 `recv_multipart()` 取出 state_1 | Python ROUTER | ——            | `[ (id_2, state_2) ]`                  |
| `T4` | Python 生成 action_1，回复给 C++ 端 1       | Python ROUTER | C++ DEALER 1  | `[ (id_2, state_2) ]`                  |
| `T5` | Python 调用 `recv_multipart()` 取出 state_2 | Python ROUTER | ——            | `[]`                                   |
| `T6` | Python 生成 action_2，回复给 C++ 端 2       | Python ROUTER | C++ DEALER 2  | `[]`                                   |

**重点总结：**

* Python 端是通过 `identity` 来识别不同的 C++ 端。
* Python 端的 `recv_multipart()` 每次只取出一条完整的消息（而不是批量取出）。
* 多个 C++ 端的消息会被缓存，Python 端通过轮询（FIFO）方式逐个读取和处理。

 这就解释了为什么 ROUTER/DEALER 方式特别适合多线程或多进程的强化学习环境！ 

#### python端如何识别不同的C++请求端的身份

> python端识别到的不同的c++端的请求的身份，这个身份是怎么生成的？我在c++端也没看到发送的时候附带上c++段的身份啊？

在 ZeroMQ 的 `ROUTER/DEALER` 模式中，`ROUTER` socket 会自动为每个连接的 `DEALER` 客户端分配一个唯一的标识符（身份），这个身份会在 `ROUTER` 接收消息时被自动附加到每个消息的前面。

**身份（Identity）** 是 ZeroMQ 自动处理的，它是一个内建的机制，用于标识不同的客户端。每当 `DEALER` 发送消息时，ZeroMQ 会将这个唯一身份附加到消息的前面。Python 端的 `ROUTER` socket 会自动解析这个身份，并在发送响应时将相同的身份附带在响应消息前面。

因此，你不需要在 C++ 端手动附加身份，ZeroMQ 会处理这一部分。C++ 端只需要正常发送消息，而 Python 端会从收到的消息中自动提取身份并将其返回给 C++ 端。

**如何在 Python 端接收和使用身份？**

在 Python 端，你可以通过 `recv_multipart()` 方法接收消息时自动提取身份。`recv_multipart()` 会返回一个消息列表，列表的第一个元素就是 `DEALER` 发送的身份。然后，你可以将身份和数据一起使用。

示例代码解析：

**Python 端（ROUTER）**

```python
identity, msg = router_socket.recv_multipart()  # identity 是 C++ 端的身份
```

- 这里 `recv_multipart()` 会接收到一个多部分的消息。第一部分就是 C++ 端（DEALER）发送的身份，第二部分是 C++ 端发送的数据（如 `state`）。
- `identity` 变量就是 C++ 端的身份，它是由 ZeroMQ 自动分配和管理的。

**C++ 端（DEALER）**

C++ 端并不需要显式地指定身份，因为 ZeroMQ 会自动分配并使用身份。

```cpp
dealer_socket.send(zmq::buffer(state.dump()), zmq::send_flags::none);  // 不需要附加身份
```

总结：

- **身份是由 ZeroMQ 自动处理的**，不需要在 C++ 端显式地发送身份。
- 在 Python 端，`ROUTER` 会接收包含身份的消息，你可以通过 `recv_multipart()` 自动提取身份。
- Python 端使用身份来确定响应应该发送回哪个 C++ 端（`DEALER`）。

#### 接收阻塞机制是怎样的

> c++段的代码的这里：
>
> ```c++
> // 接收来自 Python 的 action
> zmq::message_t action_msg;
> dealer_socket.recv(action_msg, zmq::recv_flags::none);
> ```
>
> 我想问，当c++给python发完state数据，python正在处理时，此时c++运行到这里的代码，会阻塞住等待接收来自python将来发送action数据吗

是的，**会阻塞住**，直到 Python 端（ROUTER）发送回 `action` 数据。

`dealer_socket.recv()` 是一个**同步阻塞调用**。

在 Python 端（ROUTER）发送 action 之前，C++ 端会一直在这里阻塞，**不会继续执行后续的代码**。

Python 端通过 `send_multipart()` 将生成的 action 发送回 C++ 端后，`dealer_socket.recv()` 才会解除阻塞，接收 action 数据，并继续往下执行。

- 如果你想让 `recv()` 变成**非阻塞模式**，可以这样设置：

```cpp
dealer_socket.recv(action_msg, zmq::recv_flags::dontwait);
```

但这样做的结果是——如果 Python 端还没有发回 action，`recv()` 会立刻返回一个错误（抛出 `zmq::error_t` 异常）。

- 如果采用非阻塞模式，你需要在 C++ 端主动轮询（polling）或加一个超时机制来处理。

**强化学习中的常见做法：**

在强化学习训练中，由于交互是严格的“状态-动作”交替流程，通常会采用**阻塞式接收**，因为这样可以保持仿真环境和训练过程的严格同步。

所以，**保持阻塞等待是合理的做法**，你现在的代码逻辑是正确的！



> ok，我还想问，在你写的python端的这里：
>
> ```python
> # 接收来自 C++ 的消息
> identity, state_msg = router_socket.recv_multipart()
> ```
>
> 当缓存中没有来自c++的数据时，会阻塞在这里吗？

**是的，会阻塞在这里**，直到缓存中有新的来自 C++ 端的数据。

`recv_multipart()` 是一个**同步阻塞调用**。

如果缓存中还没有来自 C++ 端的数据，Python 端会在这里阻塞，**不会继续执行后续代码**。

当有新的 C++ 端（DEALER）发送的消息到达后，`recv_multipart()` 会立刻接收并返回消息内容。

如果你希望在没有消息时不阻塞，可以使用非阻塞模式：

```python
import zmq

context = zmq.Context()
router_socket = context.socket(zmq.ROUTER)
router_socket.bind("tcp://*:5555")

try:
    identity, state_msg = router_socket.recv_multipart(zmq.NOBLOCK)
except zmq.Again:
    # 没有新的消息时，会抛出 zmq.Again 异常
    print("No message received yet.")
```

- `zmq.NOBLOCK` 使 `recv_multipart()` 进入**非阻塞模式**。
- 如果没有数据，`recv_multipart()` 会立即抛出 `zmq.Again` 异常。
- 在强化学习中，**通常不推荐用非阻塞模式**，因为状态-动作交互是严格同步的，阻塞可以保证仿真环境和模型训练的一致性。

**所以目前的阻塞式调用是合理的！**

因为你要做强化学习中的状态-动作交互，每一步训练都是严格配对的，所以使用**阻塞模式**是正确的。



> 在你写的python端的这里：
>
> ```python
> # 接收来自 C++ 的消息
> identity, state_msg = router_socket.recv_multipart()
> ```
>
> 是不是只要缓存里有一条数据，它就执行？你能详细讲下这里的执行逻辑吗？

是的，**只要缓存里有一条数据，它就会立即执行**。

让我详细解释一下在 `router_socket.recv_multipart()` 中发生的具体逻辑：

**背景知识**

1. Python 端使用 `ROUTER` 套接字，C++ 端使用 `DEALER` 套接字。
2. `ROUTER` 套接字会为每个 `DEALER` 客户端（即每个 C++ 端）分配一个唯一的 `identity`（身份标识）。
3. ZMQ 维护一个内部的**消息队列（缓存）**，用来存储来自 `DEALER` 的消息。

**执行流程**

在 Python 端的这段代码中：

```python
# 接收来自 C++ 的消息
identity, state_msg = router_socket.recv_multipart()
```

 `recv_multipart()` 的执行逻辑如下：

1. **检查缓存中是否有消息**：
   - `router_socket` 会首先查看其内部消息缓存。
   - 如果有数据，直接取出并处理。
   - 如果没有数据，阻塞在这里，直到新的消息到来。
2. **获取消息格式**：
   - `router_socket` 期望接收一个多部分的消息（multipart）。
   - 在 ROUTER/DEALER 模式下，ZMQ 自动为每个 DEALER 连接分配一个身份标识 (`identity`)。
   - 所以，接收的消息由两部分组成： 
     - `identity` → 来自哪个 DEALER（即 C++ 端）
     - `state_msg` → 这个 DEALER 发送的数据
3. **缓存的处理策略**：
   - 如果缓存中有多条消息，按照**FIFO（先进先出）**顺序取出。
   - 如果缓存中有两条来自不同 C++ 端的消息，`recv_multipart()` 会按发送顺序逐个取出。
4. **不处理粘包问题**：
   - ZMQ 的每个消息是一个完整的单元，ZMQ 会在传输层面保证完整性。
   - 不会出现粘包或拆包的问题。

**示例场景**

假设有两个 C++ 仿真环境（DEALER）：

- `C++ 端 1`（identity = `b'client-1'`）发送了 state 数据 `b'state-1'`
- `C++ 端 2`（identity = `b'client-2'`）发送了 state 数据 `b'state-2'`

ZMQ 的缓存队列可能会是这样的状态：

```css
[b'client-1', b'state-1']
[b'client-2', b'state-2']
```

调用 `recv_multipart()` 时的行为：

1. 第一次调用 → 取出 `[b'client-1', b'state-1']`
2. 第二次调用 → 取出 `[b'client-2', b'state-2']`
3. 如果此时没有新消息，阻塞在 `recv_multipart()` 直到有新的消息进入。

**总结**

- 如果缓存中有消息，`recv_multipart()` 立即返回，按 FIFO 顺序处理。
- 如果缓存为空，阻塞，直到有新消息进入。
- 通过 `identity`，Python 端可以识别来自哪个 C++ 端的消息。
- 这也是为什么 ROUTER/DEALER 结构能够灵活地处理多客户端的原因。



#### 为什么ROUTER/DEALER模式要比rep-req模式好

>  在单线程的强化学习中（c++和python通信），为什么你觉得ROUTER/DEALER模式要比rep-req模式好？

在单线程的强化学习通信中，**ROUTER/DEALER 模式**相对于 **REQ/REP 模式** 仍然更好，主要是因为它提供了更大的**灵活性**和**容错性**，即使在单线程下也有优势，原因如下：

1. **非严格的一问一答模式，支持异步处理**

- **REQ/REP 模式**是严格的“请求-应答”模型，必须是 **请求 → 回复 → 请求 → 回复** 的固定顺序，如果某个请求或响应丢失，通信就会中断或阻塞。

- ROUTER/DEALER 模式

  允许更灵活的请求和响应顺序： 

  - ROUTER 可以接收多个 DEALER 的请求，按需要的顺序处理和响应。
  - DEALER 可以在未收到响应之前发送多个请求（即支持流水线式通信）。

2. **更健壮的通信机制**

- 在 **REQ/REP 模式** 下，如果某个请求或响应在网络中丢失，通信可能会陷入死锁或阻塞。
- **ROUTER/DEALER** 通过内部消息队列自动管理未完成的请求，能在丢包或时序问题出现时恢复通信。

3. **为未来的并行扩展提供便利**

即使目前是单线程通信，将来要扩展为多线程或多环境并行时，ROUTER/DEALER 直接就能适配：

- ROUTER 可以轻松管理多个 DEALER（即多个仿真环境）。
- 不需要修改通信逻辑，只需要在 Python 端管理更多的 identity。

4. **消息管理更灵活**

- ROUTER 能直接知道哪个 DEALER 发送了请求，并能有选择地回复某个 DEALER。
- REQ/REP 则没有这种灵活性，必须是严格的 1 对 1 通信。

**总结**

即使现在是单线程，采用 ROUTER/DEALER 可以：

- 让通信更灵活（不再严格一问一答）
- 更容易恢复丢包或超时引起的中断
- 为未来的多线程并行扩展打下基础

所以，哪怕你现在是单线程通信，**ROUTER/DEALER 依然是更优的架构选择**。



>  DEALER 可以在未收到响应之前发送多个请求（即支持流水线式通信）。 这句话是啥意思？

“**DEALER 可以在未收到响应之前发送多个请求（即支持流水线式通信）**”的意思是：

- 在 **REQ/REP 模式** 中，通信是严格的“请求-应答”顺序：
  - 必须是：**请求 → 回复 → 请求 → 回复**
  - 在发送了一个请求后，必须等待收到回复，才能发送下一个请求。
  - 如果没有收到回复，就会被阻塞，无法继续发送请求。
- 在 **DEALER/ROUTER 模式** 中，通信顺序更灵活：
  - **DEALER** 不需要等到收到上一个请求的回复才能发送下一个请求。
  - DEALER 可以在“未收到响应”时继续发送新的请求，形成一个“请求队列”（即流水线）。
  - ROUTER 端会按顺序或其他策略处理这些请求，并按需要的顺序返回响应。

**举个例子：**

在 REQ/REP 中：

1. DEALER 发送 `state_1` → 必须等 ROUTER 回复 `action_1` → 才能发送 `state_2`
2. 如果 ROUTER 处理慢或超时，DEALER 会被阻塞，无法继续发送 `state_2`

在 DEALER/ROUTER 中：

1. DEALER 发送 `state_1`
2. DEALER 立刻再发送 `state_2`，而不需要等 ROUTER 回复 `state_1`
3. ROUTER 可能先回复 `state_2` 的结果，再回复 `state_1` 的结果
4. DEALER 通过消息的 `identity` 识别出是哪个请求的结果

**流水线式通信的优势**

- DEALER 可以连续不断地发送请求，不会因为等待响应而被阻塞
- ROUTER 端可以并行处理多个请求，提高整体吞吐量
- 即使某个响应处理慢，DEALER 也可以继续发送新请求，不会被卡住

**总结**：REQ/REP 是“同步”的，DEALER/ROUTER 是“异步”的，流水线式通信让 DEALER 端始终保持高效通信，不会因为网络延迟或处理延迟而被卡住。



> DEALER 立刻再发送 state_2，而不需要等 ROUTER 回复 state_1
> 这句话的意思是，DEALER可以持续发送，只不过是发送到缓存中，至于缓存中的数据，是否被读取，它是不知道也不用管的是吗？就是对方是否读取了DEALER发送的消息，是不影响DEALER发送的？

完全正确！👍

**DEALER 的发送行为和对方是否读取消息是解耦的**，即：

1. **DEALER 负责发送**
   - DEALER 只管把消息发送出去。
   - 发送的消息会被 ZeroMQ 自动存储在发送缓存（send buffer）中。
2. **ROUTER 负责接收**
   - ROUTER 什么时候从缓存中读取这些消息，DEALER 完全不关心。
   - 如果 ROUTER 处理慢或者暂时未读取，DEALER 仍然可以继续发送新消息。
3. **缓存的存在让通信是“非阻塞”的**
   - 只要发送缓存未满，DEALER 就可以继续发送，ROUTER 是否读取不影响 DEALER 的发送操作。
   - 如果发送缓存满了，DEALER 才会因为缓存溢出而被阻塞。

**形象点的比喻**

- REQ/REP 就像在饭馆里点菜：
  - 顾客（REQ）点菜 → 厨房（REP）做菜 → 厨房把菜端上来 → 顾客才能继续点下一道菜
  - 如果厨房忙不过来，顾客只能干等着
- DEALER/ROUTER 就像快餐厅的订单系统：
  - 顾客（DEALER）可以连着下好几道订单（消息存入缓存）
  - 厨房（ROUTER）根据自己的节奏按顺序出餐（读取并处理消息）
  - 只要订单不超出系统的承载能力（缓存没满），顾客就可以继续下单，不用等厨房处理完

这也是为什么 **DEALER/ROUTER 模式的吞吐量更高**，因为通信是“解耦”的，DEALER 不会被 ROUTER 的响应速度拖慢。



> 那假设缓存中堆积了很多DEALER发送的消息，那ROUTER循环读取的时候，还是按照发送消息的顺序读取吗

是的，**ROUTER 按照消息在缓存中的存储顺序（FIFO，先进先出）来读取消息**。

**详细解释：**

1. **DEALER 发送消息**

   - DEALER 持续将消息发送到发送缓存（send buffer）。
   - 消息会按照发送的顺序，依次被存储在 ROUTER 的接收缓存（receive buffer）中。

2. **ROUTER 读取消息**

   - ROUTER 每次调用 `recv_multipart()` 时，会按**消息的存储顺序**从缓存中取出一条消息。
   - 如果缓存里有很多条消息，ROUTER 会严格按照 FIFO（First In, First Out）的原则依次读取。
   - 如果 ROUTER 读取得慢，缓存中的消息可能会堆积，但读取顺序仍然是按照 DEALER 发送的先后顺序来读取。

3. **示例（假设两个 DEALER 同时向一个 ROUTER 发送消息）：**

   - DEALER_A 发送：`state_A1`, `state_A2`

   - DEALER_B 发送：`state_B1`, `state_B2`

   - 如果 ROUTER 读取消息的顺序可能是： 

     ```
     nginx
     
     
     复制编辑
     state_A1 → state_B1 → state_A2 → state_B2
     ```

   - **不同 DEALER 之间的读取顺序可能是交错的**，但对于**同一个 DEALER**，消息读取顺序一定是发送顺序。

**总结**

* ROUTER 读取时是**严格按发送顺序**读取的。
* 多个 DEALER 发送的消息会被交错读取，但**同一 DEALER 的消息顺序会被保持**。
* ROUTER 读取速度慢的话，缓存会累积，但顺序不会被打乱。 😎



> > 多个 DEALER 发送的消息会被交错读取，但同一 DEALER 的消息顺序会被保持。
> > ROUTER 读取速度慢的话，缓存会累积，但顺序不会被打乱。
>
> 你的意思是，多个DEALER发送可能会互相交织纠缠，但是ROUTER读取的时候，是全部顺序读取的，所以对于指定的DEALER，其读取顺序就是其发送顺序？

完全正确！你总结得非常到位！

1. **多个 DEALER 发送的消息** → 在 ROUTER 的接收缓存中，这些消息可能会交织在一起（因为 DEALER 可能是并行发送的）。
2. ROUTER 读取消息 → 按照接收缓存中的存储顺序（即 DEALER 发送的先后顺序）逐条读取。 
   - 同一个 DEALER 发送的消息，顺序会被严格保持（因为 ZMQ 在同一连接中是有序的）。
   - 不同 DEALER 的消息在读取时可能会交错（取决于它们进入缓存的时间）。

**例子**

假设有两个 DEALER（A 和 B）：

1. DEALER_A 发送：`state_A1`, `state_A2`
2. DEALER_B 发送：`state_B1`, `state_B2`

**可能的发送顺序**（交织）：

```nginx
state_A1 → state_B1 → state_A2 → state_B2
```

**ROUTER 读取时：**

* 从缓存中读取顺序是**state_A1 → state_B1 → state_A2 → state_B2**（严格FIFO）。
* DEALER_A 的消息读取顺序是 `state_A1 → state_A2`（保持发送顺序）。
* DEALER_B 的消息读取顺序是 `state_B1 → state_B2`（保持发送顺序）。
* DEALER_A 和 DEALER_B 之间的读取顺序可能会交错，但不会打乱单个 DEALER 的顺序。

所以，**在并行场景下，ROUTER 读取时是“跨 DEALER 的顺序可能交错，但每个 DEALER 内部的顺序是严格保持的”**。





# 在C/C++中集成ZMQ

C语言的zmq实现之一为libzmq：https://github.com/zeromq/libzmq，更多其他实现查看：https://zeromq.org/languages/c/。

C++语言的zmq的实现之一为cppzmq：https://github.com/zeromq/cppzmq，更多其他实现查看：https://zeromq.org/languages/cplusplus/。

> cppzmq是一个轻量级的、头文件唯一的C++绑定库，它简化了与libzmq（ZeroMQ）的交互，让你仅需包含zmq.hpp(可能还有zmq_addon.hpp)即可开始开发。此库采用C++11及以上标准，着重于类型安全、异常处理和资源自动管理，是访问底层libzmq API的现代C++方式。

注：C++语言实现zmq其实都只是在基于C语言的libzmq上用几个头文件封装了接口，其并没有cpp源码。所以，编译C++语言的zmq链接库时，是需要先编译C语言的zmq库libzmq的，然后加个头文件就行。

这里给一个实现在C++中集成ZMQ的CmakeLists.txt吧：

注：需要提前把cppzmq下载到`/external/cppzmq`文件夹中，把libzmq下载到`/external/libzmq`文件夹中。

```cmake
cmake_minimum_required(VERSION 3.21)
project(your_proj_name)
set(CMAKE_CXX_STANDARD 14)

set(prj_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB_RECURSE root_src_files "${prj_src_dir}/*")
message(STATUS "root_src_files = ${root_src_files}")

set(PRJ_SRC_LIST)
list(APPEND PRJ_SRC_LIST ${root_src_files})
message(STATUS "PRJ_SRC_LIST = ${PRJ_SRC_LIST}")

# header path
include_directories(./source/;./source/lib/;)

# 注：cppzmq只是基于libzmq用头文件封装了下接口而已，所以需要先编译或者安装libzmq
# 添加libzmq并需要其内部的cmakelists.txt编译
add_subdirectory(external/libzmq)
# 输出libzmq库的生成路径
# 创建一个文件来保存生成表达式的输出
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/libzmq_path.txt" CONTENT "$<TARGET_FILE:libzmq>")
# 配置阶段提示
message(STATUS "libzmq path will be generated to ${CMAKE_BINARY_DIR}/libzmq_path.txt")

# 添加cppzmq的头文件路径
set(CPPZMQ_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/cppzmq")
include_directories(${CPPZMQ_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${PRJ_SRC_LIST})

# 链接libzmq
target_link_libraries(${PROJECT_NAME} PRIVATE libzmq)
```

[libzmq在windows下的编译](https://blog.csdn.net/xiexingshishu/article/details/131275090)

[windows下cppzmq简易使用指南](https://niaoge.blog.csdn.net/article/details/105543506)



# 问题解答

## ZMQ的异步消息队列模型机制

基于zmq的req-rep模式下，我注意到一个现象，在服务端rep没有启动时，客户端req竟然可以通过socket.send把消息发出去，而等到服务端rep启动后，居然能收到客户端在服务端rep启动前发的消息，请问这怎么理解？按照我的理解，服务端rep没有启动前，客户端req应该是发不出消息，会卡在socket.send或者发出了消息但是会丢失，但是事实不是这样的，那我就无法理解了。

> 这个现象在 **ZeroMQ 的设计中是完全正常的**，并且是 ZeroMQ 的**异步消息队列模型**的一个核心特性。以下是对这一行为的深入分析和解读。
>
> ------
>
> ### **1. ZeroMQ 的异步消息模型**
>
> - ZeroMQ 的设计核心是消息队列。即使在服务端未启动或未连接时，ZeroMQ 的 `REQ` 套接字也会将消息存储在一个内部的发送队列中，而不是立即丢弃或阻塞。
> - 当服务端（`REP`）启动并建立连接后，ZeroMQ 会将队列中的消息发送到服务端。
> - 这种机制提高了系统的弹性和容错能力，避免了客户端因为服务端未准备好而阻塞或丢失消息。
>
> ------
>
> ### **2. 消息的发送流程**
>
> 在 ZeroMQ 的 `REQ-REP` 模式中，消息的发送大致经历以下几个阶段：
>
> 1. **消息存储到客户端的发送队列**：
>    - 当你调用 `socket.send` 时，消息会被写入到 `REQ` 套接字的发送队列。
>    - 这一步是**非阻塞的**，消息会在本地被排队，等待网络连接和接收方的准备。
> 2. **等待网络连接建立**：
>    - 如果服务端（`REP`）尚未启动，客户端的 `REQ` 套接字会尝试不断重连。
>    - 一旦连接建立，客户端会自动将队列中的消息发送到服务端。
> 3. **服务端接收消息**：
>    - 服务端（`REP`）在启动后，会建立与客户端的连接，并接收客户端队列中的消息。
>
> ------
>
> ### **3. 为什么不会卡在 socket.send 或丢失消息？**
>
> #### **不会卡住的原因**
>
> - ZeroMQ 的 `socket.send` 在默认情况下是**非阻塞的**，它只负责将消息放入内部队列中，而不会等待服务端是否已连接。
> - 这也是 ZeroMQ 高性能设计的关键：消息的发送与网络的状态解耦。
>
> #### **不会丢失消息的原因**
>
> - 只要发送队列未达到容量限制（由高水位标记 `ZMQ_SNDHWM` 控制），消息会被保存在客户端的内部队列中。
> - 当队列未满且网络连接建立后，消息会自动传递给服务端。
>
> ------
>
> ### **4. 现象的详细解释**
>
> - **现象**：服务端未启动时，客户端调用 `socket.send` 成功，服务端启动后还能接收到消息。
> - 解释：
>   1. 客户端的 `REQ` 套接字将消息存储在内部队列中。
>   2. 服务端的 `REP` 套接字启动并连接后，客户端自动将队列中的消息发送给服务端。
>   3. 服务端接收并处理了这些延迟发送的消息。
>
> ------
>
> ### **5. 可能的限制与风险**
>
> 尽管 ZeroMQ 提供了这种便利的异步消息模型，但仍需注意以下几点：
>
> 1. **队列大小限制**：
>    - 每个套接字都有一个发送队列和接收队列，默认大小由高水位标记控制（`ZMQ_SNDHWM` 和 `ZMQ_RCVHWM`）。
>    - 如果队列满了，`socket.send` 会阻塞或失败（取决于是否启用了非阻塞模式）。
> 2. **消息生命周期**：
>    - 如果客户端的队列中消息等待时间过长，可能会超出程序的逻辑预期。
>    - 需要显式设置超时时间（例如 `ZMQ_RCVTIMEO` 或 `ZMQ_LINGER`）来避免死锁或无限等待。
> 3. **消息顺序问题**：
>    - 在某些场景中，队列中的旧消息可能在网络连接恢复后影响系统逻辑。
>    - 可以通过清空队列或标记消息（如加时间戳）来确保处理最新消息。
>
> ------
>
> ### **6. 控制消息行为的建议**
>
> 如果你希望更严格地控制这种行为，可以考虑以下方法：
>
> #### **方法 1：设置发送队列的高水位标记**
>
> 限制发送队列的大小，防止积压过多消息。
>
> ```c++
> socket.setsockopt(ZMQ_SNDHWM, 10); // 设置发送队列的最大长度为 10
> ```
>
> #### **方法 2：设置连接超时**
>
> 为套接字设置超时时间，避免无限等待连接。
>
> ```c++
> socket.setsockopt(ZMQ_RCVTIMEO, 5000); // 接收超时设置为 5 秒
> socket.setsockopt(ZMQ_SNDTIMEO, 5000); // 发送超时设置为 5 秒
> ```
>
> #### **方法 3：清空队列**
>
> 在某些场景中，你可能希望在连接前清空队列，避免发送过期的消息。
>
> ```c++
> socket.setsockopt(ZMQ_LINGER, 0); // 关闭时直接清空队列
> ```
>
> #### **方法 4：显式检查连接状态**
>
> 通过 `zmq_socket_monitor` 检测套接字的连接状态。
>
> ------
>
> ### **7. 总结**
>
> - **ZeroMQ 的 REQ-REP 模式是异步的**，消息在客户端的发送队列中缓存，等待网络连接建立后自动发送给服务端。
> - **不会卡在 socket.send 或丢失消息**，除非发送队列满了。
> - 这种设计提高了弹性和性能，但可能需要显式管理队列大小和超时时间以避免潜在问题。
>
> 通过合理配置 ZeroMQ 的选项，可以更好地控制消息的行为！



# 参考资料

* [Python ZeroMQ编程 网络通信协议详细说明和教程](https://blog.csdn.net/weixin_39589455/article/details/134898826)

“入门示例”参考此博客。

* [ZMQ通信模式 ](https://www.cnblogs.com/cherishui/p/10198375.html)

“ZMQ通信模式”参考此博客。

