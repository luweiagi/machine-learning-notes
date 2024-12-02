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



# 参考资料

* [Python ZeroMQ编程 网络通信协议详细说明和教程](https://blog.csdn.net/weixin_39589455/article/details/134898826)

“入门示例”参考此博客。

* [ZMQ通信模式 ](https://www.cnblogs.com/cherishui/p/10198375.html)

“ZMQ通信模式”参考此博客。

