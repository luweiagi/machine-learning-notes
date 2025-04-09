# 混合动作空间

* [返回上层目录](../advanced-theme.md)



[混合动作空间｜揭秘创造人工智能的黑魔法（1）](https://zhuanlan.zhihu.com/p/462037789)

# 混合动作空间

[【RL】混合动作空间](https://zhuanlan.zhihu.com/p/683570631)





[欢迎来到 DI-ENGINE 中文文档](https://di-engine-docs.readthedocs.io/zh-cn/latest/index_zh.html)



[PPO × Family PyTorch 注解文档](https://github.com/opendilab/PPOxFamily)

混合动作空间的 mask，用于表达不同 action 部分之间的关系，例如某些动作类型对应特定的动作参数，可以参考这里的讲解例子，尤其是最后的 mask 使用部分 https://opendilab.github.io/PPOxFamily/hybrid_zh.html

视频讲解：[【PPO × Family】第二课：解构复杂动作空间](https://www.bilibili.com/video/BV1wv4y167w2/?vd_source=147fb813418c7610c21b6a5618c85cb7)，对应的代码：[Chapter2 Application Demo](https://github.com/opendilab/PPOxFamily/issues/4)。其中，有个场景hybrid_moving()，它能拐弯，可用这个作为飞机平面游戏场景。

[混合动作空间｜揭秘创造人工智能的黑魔法](https://www.zhihu.com/column/c_1505587066188111873)，里面介绍有环境[GYM-HYBRID](https://di-engine-docs.readthedocs.io/zh-cn/latest/13_envs/gym_hybrid_zh.html)和可能更全面的英文版[gym-hybrid readme](https://github.com/opendilab/DI-engine/tree/d919fa5f5da1ceb3efb187dc1d1f28f0be5b616d/dizoo/gym_hybrid/envs/gym-hybrid)，看这里的[DI-zoo 的用法](https://di-engine-docs.readthedocs.io/zh-cn/latest/11_dizoo/index_zh.html)来训练其[代码](https://github.com/opendilab/DI-engine/blob/d919fa5f5da1ceb3efb187dc1d1f28f0be5b616d/dizoo/gym_hybrid/config/gym_hybrid_hppo_config.py)。

而在近些年来，深度强化学习的研究者们将目光投向了**更通用**的混合动作空间建模方法，开始尝试设计额外的表征学习模块来获得更紧凑（compact）、更高效的动作表征，从而拓展强化学习在复杂动作空间上的应用。在本博客中，我们将会介绍相关工作之一：HyAR [1]。







PDF: [Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space IJCAI2019](https://www.ijcai.org/proceedings/2019/0316.pdf)

博客：[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space IJCAI2019](https://blog.csdn.net/quintus0505/article/details/111400717)

摘要：提出了一种 actor-critic 的混合模型算法 for reinforcement learning in parameterized action space，并且在PPO算法上面做出了改进，提出了 hybrid proximal policy optimization (H-PPO) 算法，并通过了实验验证了该算法的可靠性。

核心思想：传统的RL大多只针对于连续的或者离散的空间提出优化的方案，但是实际情况下更多的是混合的空间，如在足球场上踢球，在离散的空间中，agent只能选择跑动或者踢球的方向但是不能选择连续的跑动速度/距离或者踢球的力度，但是在混合空间下，使得agent 有可能做出离散 + 连续的选择。传统的RL无法有效的处理混合空间中的联合优化，因此文章提出了一个新的框架来解决这种方法，这种框架基于actor-critic 形式，policy gradient 和 PPO 都可以有效的同时处理离散的和连续的空间，文章选择了在PPO基础上提出H-PPO算法。

[【论文阅读IJCAI-19】Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space](https://zhuanlan.zhihu.com/p/649082917)

这个博客参考了上面的博客，但是带了代码：[HPPO混合动作PPO算法](https://blog.csdn.net/qq_45889056/article/details/137694740)
$$
loss_{actor1}=\frac{p_{new\_actor1}}{p_{old\_actor1}}adv\\
loss_{actor2}=\frac{p_{new\_actor2}}{p_{old\_actor2}}adv\\
loss_{actor}=loss_{actor1}+loss_{actor2}\\
loss_{actor}=\left(ratio_{actor1}\times ratio_{actor2}\right)\times adv=\frac{p_{new\_actor1}}{p_{old\_actor1}}\times \frac{p_{new\_actor2}}{p_{old\_actor2}}\times adv
$$




- 博客：[HyAR：通过混合动作表示解决离散-连续动作的强化学习问题](https://zhuanlan.zhihu.com/p/497347569)

离散-连续的混合动作空间在许多强化学习应用的实际应用场景中存在，例如机器人控制和游戏人工智能。比如在实际应用场景机器人足球世界杯中，一个足球机器人可以选择带球（离散）到某个位置（连续坐标）或者用力（连续）将球踢（离散）到某个位置等混合动作；在一些大型游戏中，玩家控制的每个角色在选择释放哪个技能（离散）后还需选择技能释放的位置（连续坐标）。然而，大多数已有的强化学习（RL）工作只支持离散动作空间或连续动作空间，很少考虑混合动作空间。

PDF和代码：[HyAR: Addressing Discrete-Continuous Action Reinforcement Learning via Hybrid Action Representation](https://openreview.net/forum?id=64trBbOhdGU)

代码里涉及到的环境[强化学习: 参数化动作空间环境gym-platform（1）](https://blog.csdn.net/markchalse/article/details/114507032)，还有[gym-goal](https://github.com/cycraig/gym-goal)





硕士论文：[基于混合动作表征的离散-连续动作强化学习算法研究 天津大学](https://cdmd.cnki.com.cn/Article/CDMD-10056-1023845015.htm)





[使用强化学习时，如果动作中既有连续动作，又有离散动作，应该如何处理？](https://www.zhihu.com/question/274633965/answer/2999234319)

建议看看腾讯AI lab的论文“Parametrized deep q-networks learning: Reinforcement learning with discrete-continuous hybrid action space”，里面详细介绍了用参数化Q网络的方法来处理连续离散混合动作的问题，我的直观理解是Q函数中action由离散和连续两个动作组成，其中连续的动作先用确定性策略网络来得到（类似DDPG），离散的动作还是和DQN一样取argmax（这里指当状态和连续动作确定时取Q值argmax)）。确定性网络损失函数是负Q值来得到最好的连续动作，与DDPG的actor类似；critic网络损失函数是与传统DQN一样，都是最小化td-error以得到精准Q值。





仿真环境

- [thomashirtz/gym-hybrid](https://github.com/thomashirtz/gym-hybrid)

创建了离散连续混合动作的gym

- [jason19990305/HPPO-Mujoco](https://github.com/jason19990305/HPPO-Mujoco)

有离散空间的mujoco和hppo代码



deepmind的一篇论文：[DeepMind: Continuous-Discrete Reinforcement Learning for Hybrid Control in Robotics CoRL2019](https://proceedings.mlr.press/v100/neunert20a/neunert20a.pdf)，感觉没啥用



基于参数化动作基元的机器人强化学习加速：[Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/b6846b0186a035fcc76b1b1d26fd42fa-Paper.pdf)，感觉可能也没啥用。其github：https://mihdalal.github.io/raps/。



多头动作PPO：[henrycharlesworth/multi_action_head_PPO](https://github.com/henrycharlesworth/multi_action_head_PPO)



