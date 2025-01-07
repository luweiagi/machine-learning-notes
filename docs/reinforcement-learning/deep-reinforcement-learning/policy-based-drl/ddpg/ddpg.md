# DDPG

- [返回上层目录](../paper.md)







# 参考资料



===

* [一文七问 | 经典论文：利用深度强化学习进行连续动作控制（DDPG）](https://mp.weixin.qq.com/s/OgBhIoWe4HptIEMYJnBOlw)

本篇推文将为大家介绍 DeepMind 团队于 2016 年在人工智能领域顶级会议 ICLR 上发表的一篇论文: Continuous Control with Deep Reinforcement Learning。该论文介绍了一种用于解决连续动作空间的深度强化学习方法。具体为：基于 DQN 与 DPG 的思想，利用深度网络对高维连续动作策略进行逼近，构成一种无模型的 Actor-Critic 结构的 off-policy 算法。本文同时加入了软更新、经验回放和批标准化的技巧，用于提高训练稳定性。