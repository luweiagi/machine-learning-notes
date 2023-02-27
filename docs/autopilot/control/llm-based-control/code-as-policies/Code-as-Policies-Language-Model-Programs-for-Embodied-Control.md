# Code as Policies: Language Model Programs for Embodied Control

* [返回上层目录](../llm-based-control.md)



paper: [Code as Policies](https://arxiv.org/abs/2209.07753)



当LLM遇上机器人

大型语言模型(LLM)的一个特性是能够将描述和上下文编码成「人和机器都能理解」的格式。

当把LLM应用到机器人技术中时，可以让用户仅通过自然语言指令就能给机器人分配任务；当与视觉模型和机器人学习方法相结合时，LLM 为机器人提供了一种理解用户请求的上下文的方法，并能够对完成请求所采取的行动进行规划。

研究人员选择使用 LLM 来预测完成长期任务的步骤顺序，以及一个表示机器人在给定情况下实际能够完成的技能的affordance 模型。

同时拥有 LLM 和affordance 模型并不意味着机器人能够成功地完成任务，通过内心独白（ Inner Monologue），可以结束基于 LLM 的任务规划中的循环；利用其他信息来源，如人工反馈或场景理解，可以检测机器人何时无法正确完成任务。



并且，用LLM 编写代码来控制机器人动作也是一个有前景的研究方向。

研究人员开发的代码编写方法展示了增加任务复杂性的潜力，机器人可以通过自主生成新代码来重新组合 API 调用，合成新函数，并表达反馈循环来在运行时合成为新行为。

[Google AI年终总结第六弹：没有波士顿动力的谷歌机器人，发展得怎么样了？](https://mp.weixin.qq.com/s/JRCQP2S3CbLtUaq8MkP4pQ)