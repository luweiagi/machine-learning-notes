## Learning Model Predictive Controllers with Real-Time Attention for Real-World Navigation

* [返回上层目录](../llm-based-control.md)



blog: [Performer MPC](https://performermpc.github.io/)



在语言模型和机器人学习方法(如 RT-1)的背后，都是 Transformer模型基于互联网规模的数据训练得到的；但与 LLM 不同的是，机器人学面临着不断变化的环境和有限计算的多模态表示的挑战。

2020年，谷歌提出了一种能够提高Transformer计算效率的方法Performers，影响到包括机器人技术在内的多个应用场景。

最近研究人员扩展该方法，引入一类新的隐式控制策略，结合了模拟学习的优势和对系统约束的鲁棒处理(模型预估计控制的约束)。

与标准的 MPC 策略相比，实验结果显示机器人在实现目标方面有40% 以上的进步，在人类周围导航时，social指标上有65% 以上的提升；Performance-MPC 为8.3 M 参数模型的延迟仅为8毫秒，使得在机器人上部署Transformer变得可行。

===

[Google AI年终总结第六弹：没有波士顿动力的谷歌机器人，发展得怎么样了？](https://mp.weixin.qq.com/s/JRCQP2S3CbLtUaq8MkP4pQ)