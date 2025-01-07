# Recurrent Proximal Policy Optimization using Truncated BPTT

- [返回上层目录](../proximal-policy-optimization.md)



GitHub代码：

* PPO+GRU [MarcoMeter/recurrent-ppo-truncated-bptt](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)
* PPO+Transformer [MarcoMeter/episodic-transformer-memory-ppo](https://github.com/MarcoMeter/episodic-transformer-memory-ppo)

测试环境：[MarcoMeter/endless-memory-gym](https://github.com/MarcoMeter/endless-memory-gym/) 环境库面有三个环境

论文：[Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents Arxiv202409](https://arxiv.org/pdf/2309.17207)

这篇论文测试了GRU和Transformer在需要记忆的环境中的表现。论文总结：

> 本文介绍了Memory Gym，这是一个用于测试智能体（agents）记忆能力的2D部分可观测环境套件。该套件包含三个环境：Mortar Mayhem、Mystery Path和Searing Spotlights，它们被设计用来评估决策制定中的记忆力。这些环境最初是有限的任务，但现在被扩展为无尽的格式，反映了累积记忆游戏（如“I packed my bag”）的挑战。这种任务设计的变化将重点从仅仅评估样本效率转移到也探测动态、持久场景中的记忆力效果。
>
> 为了填补基于记忆的深度强化学习（DRL）基线的空白，作者引入了一个集成了Transformer-XL（TrXL）与近端策略优化（PPO）的实现。这种方法使用TrXL作为情景记忆的形式，并采用滑动窗口技术。通过比较门控循环单元（GRU）和TrXL的性能，研究发现在不同设置中表现各异。TrXL在有限环境中的Mystery Path上显示出更好的样本效率，并在Mortar Mayhem中表现更优。然而，在Searing Spotlights中，GRU更有效。最值得注意的是，在所有无尽的任务中，GRU显著复苏，一致性地大幅度超越TrXL。
>
> ### 关键点
>
> 1. **Memory Gym环境套件**：包含三个2D部分可观测环境，用于测试智能体的记忆能力。
> 2. **无尽的任务设计**：将有限的任务扩展为无尽的格式，以模拟累积记忆游戏的挑战。
> 3. **基线实现**：提供了一个基于Transformer-XL和PPO的实现，使用TrXL作为情景记忆，并采用滑动窗口技术。
> 4. **性能比较**：GRU和TrXL在不同环境中的表现进行了比较，发现GRU在无尽的任务中表现更优。
> 5. **实验分析**：通过实验分析，揭示了TrXL在无尽任务中的低效率，并探讨了可能的原因。
> 6. **开源贡献**：作者提供了易于跟随的开源基线实现，以促进研究社区的进一步研究。
> 7. **关键发现**：在无尽的环境中，GRU在样本效率和效果上都超过了TrXL，而在有限环境中，TrXL在某些任务上表现更好。
> 8. **未来研究方向**：提出了对TrXL局限性的进一步调查，以及对其他记忆机制和DRL算法性能阈值的探索。
>
> 这篇论文为理解和改进基于记忆的DRL算法提供了新的视角，并为该领域的研究提供了新的工具和基准。





# 参考资料



===

* [探索记忆与策略：Recurrent Proximal Policy Optimization 项目推荐](https://blog.csdn.net/gitblog_00041/article/details/142194352)
* [开源项目指南：循环策略梯度 - PPO结合截断式反向传播](https://blog.csdn.net/gitblog_00822/article/details/142120090)

对该项目的介绍

* [在PPO中使用RNN](https://zhuanlan.zhihu.com/p/592700653)

类似的话题



* [Explain My Surprise: Learning Efficient Long-Term Memory by Predicting Uncertain Outcome Arxiv202211](https://arxiv.org/pdf/2207.13649)

含有关键字：We could not successfully train a single PPO-LSTM or AMRL agent even with TBPTT rollout of

