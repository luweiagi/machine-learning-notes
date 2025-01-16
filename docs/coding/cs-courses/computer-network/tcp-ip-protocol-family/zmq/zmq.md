# ZeroMQ通信

* [返回上层目录](../tcp-ip-protocol-family.md)
* [ZMQ介绍](#ZMQ介绍)
* [入门示例](#入门示例)
* [ZMQ通信模式](#ZMQ通信模式)
  * [请求响应模式](#请求响应模式)
  * [发布订阅模式](#发布订阅模式)
  * [任务管道模式](#任务管道模式)
  * [一对一通信](#一对一通信)



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
ZMQ_SUB` <-- `ZMQ_PUB
```

## 任务管道模式

```
ZMQ_PUSH` -> [ `ZMQ_PULL` , `ZMQ_PUSH`] --> `ZMQ_PULL
```

## 一对一通信

```
ZMQ_PAIR` <--> `ZMQ_PAIR
```

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

