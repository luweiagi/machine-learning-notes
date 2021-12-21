# 强化学习

* [返回上层目录](../README.md)
* [强化学习概论](reinforcement-learning-introduction/reinforcement-learning-introduction.md)
* [多臂赌博机及其解法](multi-armed-bandit-and-solutions/multi-armed-bandit-and-solutions.md)
* [马尔科夫决策过程](markov-decision-processes/markov-decision-processes.md)
* [动态规划](dynamic-programming/dynamic-programming.md)
* [无模型方法一：蒙特卡洛](model-free-methods-1-monte-carlo/model-free-methods-1-monte-carlo.md)
* [无模型方法二：时间差分](model-free-methods-2-time-difference/model-free-methods-2-time-difference.md)
* [无模型方法三：多步自举](model-free-methods-3-multi-step-bootstrap/model-free-methods-3-multi-step-bootstrap.md)
* [函数近似和深度网络](function-approximation-and-deep-network/function-approximation-and-deep-network.md)
* [策略梯度算法](policy-gradient-algorithm/policy-gradient-algorithm.md)
* [深度强化学习](deep-reinforcement-learning/deep-reinforcement-learning.md)
* [基于模型的强化学习](model-based-reinforcement-learning/model-based-reinforcement-learning.md)
* [强化学习前景](reinforcement-learning-prospect/reinforcement-learning-prospect.md)
* [蒙特卡洛树搜索](monte-carlo-tree-search/monte-carlo-tree-search.md)
* [强化学习论文](reinforcement-learning-paper/reinforcement-learning-paper.md)
* [多智能体强化学习论文](multi-agent-reinforcement-learning-paper/multi-agent-reinforcement-learning-paper.md)



===

[【莫烦Python】强化学习 Reinforcement Learning](https://www.bilibili.com/video/BV13W411Y75P?p=5)

短小精悍

[李宏毅】2020 最新课程 (完整版) 强化学习 ](https://www.bilibili.com/video/BV1UE411G78S?p=2)

看这个，讲的很好很清楚

[David Silver 增强学习——Lecture 6 值函数逼近](https://zhuanlan.zhihu.com/p/54189036)

有空看这个，那个陈达贵的视频ppt其实就是这个。

[白话强化学习](https://www.zhihu.com/column/c_1215667894253830144)

这个知乎专栏讲的对各种知识点的直觉理解和分析都特别好。

[强化学习路线推荐及资料整理](https://zhuanlan.zhihu.com/p/344196096)

第一个是李宏毅老师21年最新的深度学习课程，将最新的内容都纳入了教学大纲

第二个是多智能体强化学习领域的：UCL的汪军老师新开的课程

[有哪些常用的多智能体强化学习仿真环境？](https://www.zhihu.com/question/332942236/answer/1295507780)

**Link：**[https://github.com/geek-ai/MAgent](https://github.com/geek-ai/MAgent)

这个是UCL汪军老师团队Mean Field 论文里用到的环境，主要研究的是当环境由**大量智能体**组成的时候的竞争和协作问题。也可以看成是复杂的Grid World环境。Render如下：





不重要：

[【RL系列】强化学习基础知识汇总](http://blog.sciencenet.cn/blog-3189881-1129931.html)

[强化学习无人机交互环境汇总](https://zhuanlan.zhihu.com/p/157867488)

作者在无人机姿态控制上使用PPO训练取得了比PID更好的效果，并成功从虚拟环境迁移到了现实世界。



[【重磅推荐: 强化学习课程】清华大学李升波老师《强化学习与控制》](https://mp.weixin.qq.com/s/bDra-n8stqJ3gcS9zr3IVA)

**《强化学习与控制》**这一门课程包括11节。

**第1讲**介绍RL概况，包括发展历史、知名学者、典型应用以及主要挑战等。

**第2讲**介绍RL的基础知识，包括定义概念、自洽条件、最优性原理问题架构等。

**第3讲**介绍免模型学习的蒙特卡洛法，包括Monte Carlo估计，On-policy/off-policy，重要性采样等。

**第4讲**介绍免模型学习的时序差分法，包括它衍生的Sarsa，Q-learning，Expected Sarsa等算法。

**第5讲**介绍带模型学习的动态规划法，包括策略迭代、值迭代、收敛性原理等。

**第6讲**介绍间接型RL的函数近似方法，包括常用近似函数，值函数近似，策略函数近似以及所衍生的Actor-critic架构等。

**第7讲**介绍直接型RL的策略梯度法，包括各类Policy Gradient, 以及如何从优化的观点看待RL等。

**第8讲**介绍深度强化学习，即以神经网络为载体的RL，包括深度化典型挑战、经验性处理技巧等。

**第9讲**介绍带模型的强化学习，即近似动态规划，包括离散时间系统的ADP，ADP与MPC的关联分析等。

**第10讲**介绍有限时域的近似动态规划，同时介绍了状态约束的处理手段以及它与可行性之间的关系

**第11讲**介绍RL的各类拾遗，包括POMDP、鲁棒性、多智能体、元学习、逆强化学习以及训练平台等。

