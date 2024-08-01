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

# 参考资料

* [Python ZeroMQ编程 网络通信协议详细说明和教程](https://blog.csdn.net/weixin_39589455/article/details/134898826)

“入门示例”参考此博客。

* [ZMQ通信模式 ](https://www.cnblogs.com/cherishui/p/10198375.html)

“ZMQ通信模式”参考此博客。

