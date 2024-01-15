# 强化学习

* [返回上层目录](../README.md)
* [强化学习](reinforcement-learning/reinforcement-learning.md)
* [仿真环境](simulation-platform/simulation-platform.md)
* [MCTS蒙特卡洛树搜索](monte-carlo-tree-search/monte-carlo-tree-search.md)
* [模仿学习](imatation-learning/imatation-learning.md)
* [多智能体强化学习](multi-agent-reinforcement-learning/multi-agent-reinforcement-learning.md)
* [Transformer+RL](transformer-rl/transformer-rl.md)
* [决策大模型](decision-making-big-model/decision-making-big-model.md)
* [Offline RL离线强化学习](offline-reinforcement-learning/offline-reinforcement-learning.md)
* [MMRL多模态强化学习](multi-modal-reinforcement-learning/multi-modal-reinforcement-learning.md)
* [LLM+RL](llm-rl/llm-rl.md)
* [DiffusionModel+RL](diffusion-model-rl/diffusion-model-rl.md)
* [业界应用](industry-application/industry-application.md)



===

# 深度强化学习入门

[强化学习怎么入门好？](https://www.zhihu.com/question/277325426/answer/2786792954)

必须推荐王树森、黎彧君、张志华的新书《深度强化学习》，已经正式出版。这是一本正式出版前就注定成为经典的入门书籍——其在线公开课视频播放量超过一百万次，助力数万“云学生”——更加高效、方便、系统地学习相关知识。课程主页这里：https://github.com/wangshusen/DRL 还有对应的在线公开课视频和代码，B站、Github都有。下文内容来自作者王树森写的前言。

[强化学习怎么入门好？](https://www.zhihu.com/question/277325426/answer/1544863580)

1.看李宏毅的强化学习视频-b站随便找一个最新最全的；

2.看郭宪大佬的《深入浅出强化学习》-知乎有他的专栏文章；

3.代码刷openai的spinningup。

目前我认为最简洁最不走弯路的方法。至少节省大家半年的随机探索时间

其他的教材对于强化的公式推导不够透彻，

其他几门视频课难度高，不适合入门；

其他的代码库，新手根本看不懂。

最后贴上我基于spinup封装好的一个强化学习库：

https://github.com/kaixindelele/DRL-tensorflow

https://github.com/kaixindelele/DRLib

# 地图：

[全网首发|| 最全深度强化学习资料(永久更新)](https://mp.weixin.qq.com/s?__biz=MzU0MTgxNDkxOA%3D%3D&idx=1&mid=2247484575&scene=21&sn=42fe3fc7d5978ca9da467fde38a13245#wechat_redirect)

[NeuronDance/DeepRL](https://github.com/NeuronDance/DeepRL)

[NeuronDance/DeepRL/A-Guide-Resource-For-DeepRL/](https://github.com/NeuronDance/DeepRL/tree/master/A-Guide-Resource-For-DeepRL)



# 视频课程

[【莫烦Python】强化学习 Reinforcement Learning](https://www.bilibili.com/video/BV13W411Y75P?p=5)

短小精悍

[李宏毅】2020 最新课程 (完整版) 强化学习 ](https://www.bilibili.com/video/BV1UE411G78S?p=2)

看这个，讲的很好很清楚，比如其中强化学习[策略梯度](https://www.bilibili.com/video/BV1UE411G78S/?p=2&vd_source=147fb813418c7610c21b6a5618c85cb7)的部分。

[李宏毅深度强化学习(国语)课程(2018) ppo](https://www.bilibili.com/video/BV1MW411w79n?p=2&vd_source=147fb813418c7610c21b6a5618c85cb7)

[David Silver 增强学习——Lecture 6 值函数逼近](https://zhuanlan.zhihu.com/p/54189036)

有空看这个，那个陈达贵的视频ppt其实就是这个。

B站上deepmind的大佬David alived的强化学习的视频，点击率甚低。看来很多国人不知道阿发狗李的研发团队的首席科学家啊。

[CS294]

初学者非常不推荐看CS294，因为真的很难，可以看David Silver的课程

[CS234]是什么？

[白话强化学习](https://www.zhihu.com/column/c_1215667894253830144)

这个知乎专栏讲的对各种知识点的直觉理解和分析都特别好。

[强化学习路线推荐及资料整理](https://zhuanlan.zhihu.com/p/344196096)

第一个是李宏毅老师21年最新的深度学习课程，将最新的内容都纳入了教学大纲

第二个是多智能体强化学习领域的：UCL的汪军老师新开的课程

# 仿真环境

[有哪些常用的多智能体强化学习仿真环境？](https://www.zhihu.com/question/332942236/answer/1295507780)

**Link：**[https://github.com/geek-ai/MAgent](https://github.com/geek-ai/MAgent)

这个是UCL汪军老师团队Mean Field 论文里用到的环境，主要研究的是当环境由**大量智能体**组成的时候的竞争和协作问题。也可以看成是复杂的Grid World环境。Render如下：

# 强化学习与控制

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



[强化学习和最优控制的《十个关键点》81页PPT汇总](https://mp.weixin.qq.com/s?__biz=MzU0MTgxNDkxOA%3D%3D&chksm=fb257f19cc52f60f9c5a70260fd20cc0b7974bcd300041eb7a75e83eeaaba416d2d965679ff2&idx=1&mid=2247485168&scene=21&sn=59039fc39903a4ee721712b1a2c53b77#wechat_redirect)



# 多智能体强化学习

[【DeepMind】多智能体学习231页PPT总结](https://mp.weixin.qq.com/s?__biz=MzU0MTgxNDkxOA%3D%3D&chksm=fb25711ccc52f80aa10666bec175cafb2a349673f39a558811b8945392df45808a053d0f00c4&idx=1&mid=2247485685&scene=21&sn=54dcbcaf022795d05d1f8a7bf6a17c12#wechat_redirect)



[最近在写多智能体强化学习工作绪论，请问除了 MADDPG 以及 MAPPO 还有哪些算法？](https://www.zhihu.com/question/517905386/answer/2359101768)



# 专题

## Transformer+RL

[Transformer + RL专题 | 究竟是强化学习魔高一尺，还是Transformer道高一丈 （第1期）](https://mp.weixin.qq.com/s?__biz=Mzk0MTI1MzI0OQ==&mid=2247490100&idx=1&sn=56d484dd1cb6062b2783554b88816688&chksm=c2d46ddaf5a3e4cc96329fcbd38876a5936344df6b113e8b7f8bbe8de391a51261e76c6faf46&cur_album_id=2628430986300801025&scene=189#wechat_redirect)

[Transformer + RL专题｜强化学习中时序建模的千层套路（第2期）](https://mp.weixin.qq.com/s?__biz=Mzk0MTI1MzI0OQ==&mid=2247490498&idx=1&sn=4b3e2174d25e530b9a388aba3331295c&chksm=c2d46c2cf5a3e53a3154792c03f4b8be1bc20700ae5a48b70e042272da46e95dcc61a425f079&cur_album_id=2628430986300801025&scene=189#wechat_redirect)

[Transformer + RL 专题｜大力出奇迹，看 Transformer 如何征服超参数搜索中的决策问题 （第3期）](https://mp.weixin.qq.com/s/S0PgD3SEMbrbA4OE--ZCQg)



# 论坛

中科院自动化所2020智能决策论坛报告ppt：
论坛报告回放：https://space.bilibili.com/551888585/channel/detail?cid=167587
【柯良军】链接：https://pan.baidu.com/s/18uM3GU8HpZ2OAUIoN0timQ 提取码：rb4o 
【章宗长】链接：https://pan.baidu.com/s/1hg-YPfcjCaMnUIogZXMmTQ 提取码：dhdf 
【余超】链接：https://pan.baidu.com/s/1ZnU7oe8xB6YJgyVC1frY6Q 提取码：h42p 
【温颖】链接：https://pan.baidu.com/s/1AhV2v_JLtiYU3gekH0d4ow 提取码：p2h7



# 知识点

[强化学习中on-policy 与off-policy有什么区别？](https://www.zhihu.com/question/57159315/answer/1855647973)

[[原创] 强化学习里的 on-policy 和 off-policy 的区别](https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E9%87%8C%E7%9A%84-on-policy-%E5%92%8C-off-policy-%E7%9A%84%E5%8C%BA%E5%88%AB/)



# 框架，库

[[tensorlayer](https://github.com/tensorlayer)/**TensorLayer**](https://github.com/tensorlayer/TensorLayer/blob/cb4eb896dd063e650ef22533ed6fa6056a71cad5/examples/reinforcement_learning/README.md)

[Tensorflow2.0实现29种深度强化学习算法大汇总](https://mp.weixin.qq.com/s?__biz=MzU0MTgxNDkxOA%3D%3D&chksm=fb25706bcc52f97dd9a496508d570dde830a1d9181bf232136993eebcad3ade1936adec6d94a&idx=1&mid=2247485826&scene=21&sn=7faa04e1a7b922d3d42059246dcadc8a#wechat_redirect)

一定要看，非常好

欢迎Star：https://github.com/StepNeverStop/RLs

本文作者使用gym,Unity3D ml-agents等环境，利用tensorflow2.0版本对29种算法进行了实现的深度强化学习训练框架，该框架具有如下特性：

- 实现单智能体强化学习、分层强化学习、多智能体强化学习算法等约29种
-  适配gym、MuJoCo、PyBullet、Unity ML-Agents等多种训练环境

[mengwanglalala/**RL-algorithms**](https://github.com/mengwanglalala/RL-algorithms)

RL-algorithms，更新一些基础的RL代码，附带了各个算法的介绍

[Awesome Reinforcement Learning Library](https://github.com/wwxFromTju/awesome-reinforcement-learning-lib)

集合了各种强化学习库



**tensorlayer**

[对话TensorLayer项目发起者董豪](https://zhuanlan.zhihu.com/p/72304092)

[TensorLayer进阶资源](https://www.jianshu.com/p/d206fb7a190d)



# 新概念

## 重生强化

我记得我刚开始学强化的时候，好奇的一个问题，对于强化的网络，如果一个开始就全给的专家数据，和从零开始学习，从试错，到自己学成专家，哪个会更好一些？
看了Reset-RL和demo-RL之后，好像答案比较明确了，还是得有高质量的数据，然后少许交互就能快速获得一个高质量策略。而从头开始试错，不断摸索的策略，很可能会因为早期的垃圾数据，导致陷入局部最优（首因偏差)，上不去~

发布于 2022-12-09・IP 属地安徽

这两个是什么算法，求指路

后者是DDPGfD，前者是primacy bias  in rl [ResetNet-The Primacy Bias in Deep Reinforcement Learning](https://www.bilibili.com/video/BV1wG4y157we/?vd_source=147fb813418c7610c21b6a5618c85cb7)

[重生强化【Reincarnating RL】论文梳理](https://zhuanlan.zhihu.com/p/591880627)



# 论文

[Danzero+：用掼蛋为例讲解如何处理大规模动作空间的强化学习问题](https://zhuanlan.zhihu.com/p/673715817)



# 不重要

[【RL系列】强化学习基础知识汇总](http://blog.sciencenet.cn/blog-3189881-1129931.html)







