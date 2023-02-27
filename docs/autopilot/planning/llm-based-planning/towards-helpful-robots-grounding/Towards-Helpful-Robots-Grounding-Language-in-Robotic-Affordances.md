# Towards Helpful Robots: Grounding Language in Robotic Affordances

- [返回上层目录](../llm-based-planning.md)



blog: [Towards Helpful Robots: Grounding Language in Robotic Affordances](https://ai.googleblog.com/2022/08/towards-helpful-robots-grounding.html)



当LLM遇上机器人

大型语言模型(LLM)的一个特性是能够将描述和上下文编码成「人和机器都能理解」的格式。

当把LLM应用到机器人技术中时，可以让用户仅通过自然语言指令就能给机器人分配任务；当与视觉模型和机器人学习方法相结合时，LLM 为机器人提供了一种理解用户请求的上下文的方法，并能够对完成请求所采取的行动进行规划。

在「迈向有益的机器人: 机器人可用性的基础语言」一文中，研究人员与Everyday Robots合作，在机器人可用性模型中基于PaLM语言模型规划长期任务。

在之前的机器学习方法中，机器人只能接受诸如「捡起海绵」等简短的硬编码命令，并且难以推理完成任务所需的步骤，如果任务是一个抽象的目标，比如「你能帮忙清理这些洒出来的东西吗?」，就更难处理了。

研究人员选择使用 LLM 来预测完成长期任务的步骤顺序，以及一个表示机器人在给定情况下实际能够完成的技能的affordance 模型。

强化学习模型中的价值函数可以用来建立affordance 模型，即一个机器人在不同状态下可以执行的动作的抽象表示，从而将现实世界中的长期任务，如「整理卧室」与完成任务所需的短期技能，如正确挑选、放置和安排物品等联系起来。

[Google AI年终总结第六弹：没有波士顿动力的谷歌机器人，发展得怎么样了？](https://mp.weixin.qq.com/s/JRCQP2S3CbLtUaq8MkP4pQ)