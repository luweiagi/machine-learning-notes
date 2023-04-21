# RT-1: Robotics Transformer for Real-World Control at Scale

* [返回上层目录](../llm-based-control.md)



paper: [RT-1](https://robotics-transformer.github.io/assets/rt1.pdf)



**摘要：**谷歌机器人团队等提出了 Robotics Transformer 1 (RT-1)。这是一种多任务模型，可以 tokenize 机器人的输入和输出动作，从而在运行时实现高效推理，使实时控制成为可能。

RT-1 模型在包含 130k 个 episode 的大型真实机器人数据集上进行训练，该数据集涵盖了 700 多项任务，使用 Everyday Robots (EDR) 的 13 台机器人在 17 个月内收集而成。数据集中展示的一组高级技能包括拾取和放置物品、打开和关闭抽屉、将物品放入和取出抽屉、将细长的物品直立放置、敲倒物体、拉出餐巾纸和打开罐子。

**推荐：**轻松完成 700 多条指令、成功率达 97%！谷歌开源机器人领域 transformer。





谷歌开源用于实际大规模控制的机器人Transformer，以97%的成功率执行700多条指令

RT-1: Robotics Transformer for Real-World Control at Scale

谷歌机器人团队等提出了 Robotics Transformer 1 (RT-1)。这是一种多任务模型，可以 tokenize 机器人的输入和输出动作，从而在运行时实现高效推理，使实时控制成为可能。RT-1 模型在包含 130k 个 episode 的大型真实机器人数据集上进行训练，该数据集涵盖了 700 多项任务，使用 Everyday Robots (EDR) 的 13 台机器人在 17 个月内收集而成。数据集中展示的一组高级技能包括拾取和放置物品、打开和关闭抽屉、将物品放入和取出抽屉、将细长的物品直立放置、敲倒物体、拉出餐巾纸和打开罐子。与现有技术相比，RT-1 可以显著改进对新任务、环境和对象的零试（zero-shot）泛化。RT-1 可以 97% 的成功率执行 700 多个训练指令，并且可以泛化到新的任务、干扰因素和背景。 



**摘要**：

通过从与任务无关的大型、多样的数据集迁移知识，现代机器学习模型可以解决特定的下游任务，无论是 zero-shot 中还是使用小的特定于任务的数据集，以达到高水平的性能。虽然这种能力已经在计算机视觉、自然语言处理或语音识别等其他领域得到了展示，但它仍有待于机器人领域的展示，在机器人领域，由于收集真实世界的机器人数据的难度，模型的泛化能力尤其关键。本文认为，这种**通用机器人模型成功的关键之一是开放式任务的训练**，结合可以吸收所有不同机器人数据的高容量架构。在本文中，作者提出了一个模型类，称为 Robotics Transformer，它具有良好的可扩展模型特性。作者在对不同模型类的研究中验证了结论，并基于对执行真实世界任务的真实机器人的大规模数据收集，验证了它们作为数据大小、模型大小和数据多样性的函数的泛化能力。

===

[Google AI年终总结第六弹：没有波士顿动力的谷歌机器人，发展得怎么样了？](https://mp.weixin.qq.com/s/JRCQP2S3CbLtUaq8MkP4pQ)

[GoogleBrain一位女机器人工程师的博客，分享了对RT-1的看法](https://twitter.com/keerthanpg/status/1602751890021761026)

> Very excited to introduce the Robotics transformer 1, a large transformer-based imitation-learned robot manipulation policy, that can execute over 700 skills at a 97% success rate. It can also learn from other robots, execute long horizon SayCan and show massive scaling potential

