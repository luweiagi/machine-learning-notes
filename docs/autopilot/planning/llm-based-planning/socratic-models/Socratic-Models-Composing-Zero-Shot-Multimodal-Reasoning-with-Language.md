# Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language

* [返回上层目录](../llm-based-planning.md)



paper: [Socratic Models](https://arxiv.org/abs/2204.00598)



当LLM遇上机器人

大型语言模型(LLM)的一个特性是能够将描述和上下文编码成「人和机器都能理解」的格式。

当把LLM应用到机器人技术中时，可以让用户仅通过自然语言指令就能给机器人分配任务；当与视觉模型和机器人学习方法相结合时，LLM 为机器人提供了一种理解用户请求的上下文的方法，并能够对完成请求所采取的行动进行规划。

其中一个基本方法是使用 LLM 来提示其他预先训练的模型获取信息，以构建场景中正在发生的事情的上下文，并对多模态任务进行预测。整个过程类似于苏格拉底式的教学方法，教师问学生问题，引导他们通过一个理性的思维过程来解答。

在「苏格拉底模型」中，研究人员证明了这种方法可以在zero-shot图像描述和视频文本检索任务中实现最先进的性能，并且还能支持新的功能，比如回答关于视频的free-form问题和预测未来的活动，多模态辅助对话，以及机器人感知和规划。

[Google AI年终总结第六弹：没有波士顿动力的谷歌机器人，发展得怎么样了？](https://mp.weixin.qq.com/s/JRCQP2S3CbLtUaq8MkP4pQ)