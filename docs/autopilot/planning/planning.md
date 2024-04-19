# 规划

* [返回上层目录](../autopilot.md)
* [基于LLM大语言模型的规划](llm-based-planning/llm-based-planning.md)







===

[刚研一做无人机路径规划到底怎么入门呀？](https://www.zhihu.com/question/622917913/answer/3228049082)

无人机路径规划按照浙大FAST LAB高飞团队的提法即为前端规划（Front End Planning）和后端规划（Back End Planning），其实对应的是自动驾驶中的全局路径规划（Global Path Planning）和局部路径规划（Local Path Planning）。前者负责初始粗路径的生成，一般表现为众多离散路径点的集合；后者在前者的基础上，在考虑众多约束的情况下，对初始路径进行优化，使之成为一条连续光滑并满足无人机动力学特性的轨迹。这些在深蓝学院开设的机器人运动规划课程中都有详细的讲解，大家感兴趣可以看一下

全局路径规划和局部路径规划分别存在着众多的算法以供学习和使用，**全局路径规划算法**包括Astar、Dstar和RRT等；**局部路径规划**包括贝塞尔曲线、B样条曲线、杜宾斯曲线（两个矢量点间的最短路径）以及五次多项式（考虑能耗最优和时间最优，根据庞特里亚金最小值原理推导而出）。在这两大类算法中，个人认为局部路径规划是整个路径规划的重点，因为它作为规划模块最终输出可用轨迹的一种方法，考虑了轨迹的连续性约束、平滑约束、碰撞约束和动力学约束等因素，需利用复杂的数值分析理论，即非线性优化，通过多次迭代得到可行解，从而生成可行轨迹。





先把一个无人机的仿真搞好，然后攒出一个小一点的三寸桨的无人机

一些相关的项目，第一个是捷克的一个实验室的多无人机框架

[ctu-mrs](https://github.com/orgs/ctu-mrs/repositories?type=all)

有人基于这个还做了路径规划：[poludmik/Path-planning-for-multiple-UAVs](https://github.com/poludmik/Path-planning-for-multiple-UAVs)



[对于机器人轨迹规划来说，优化的方式是当今主流吗?](https://www.zhihu.com/question/576631261/answer/3106771050)

这种非凸的问题没有万灵药，只能根据具体系统来看。比如你要是搞个小车的避障，如果障碍物很稀疏的，可能a*就够了。要是很多障碍物还要考虑运动学，rrt可能是不错的选择。如果赛道竞速，如何过弯，那得考虑动力学、胎地模型，基本就是最优化。(还有邪派武功RL)

最近比较关注Russ Tedrake他们搞的graph of convex set的方法，感觉理论很漂亮。这个是他们最近一篇文章的abstract，可以看看大佬对这个问题的看法:

Computing optimal, collision-free trajectories for high-dimensional systems is a challenging problem. Sampling-based planners struggle with the dimensionality, whereas trajectory optimizers may get stuck in local minima due to inherent nonconvexities in the optimization landscape. The use of mixed-integer programming to encapsulate these nonconvexities and find globally optimal trajectories has recently shown great promise, thanks in part to tight convex relaxations and efficient approximation strategies that greatly reduce runtime.



[为啥国内做轨迹优化的比较少呢？](https://www.zhihu.com/question/319353555/answer/3102886996)

Btraj是一个在线的四旋翼无人机的运动规划框架，主要用于未知环境中的自主导航。基于机载状态估计和环境感知，采用基于快速行进（FM *）的路径搜索方法，从基于构建的欧氏距离场（ESDF）中寻找路径，以实现更好的时间分配。通过对环境进行膨胀生成四旋翼无人机的飞行走廊。使用伯恩斯坦多项式将轨迹表示为分段Bezier曲线，并将整个轨迹生成问题转换成凸问题。通过使用Bezier曲线，可以确保轨迹的安全性和高阶动力学可行性，可以将轨迹完全限制在安全区域内。



[贝塞尔曲线之美](https://www.zhihu.com/zvideo/1599031183647617024)



[离线强化学习之后，强化的下一个爆点是什么？](https://www.zhihu.com/question/557673420/answer/2777046020)

这里分享两点观察：1）强化学习在整个机器学习圈子里发展步调要慢一个档位，其他领域诸如CV、NLP的发展领先于强化学习；2）从历史上来看，其他领域的先进算法最终都会被人用在强化学习上，并成为一个个的爆点，比如历史上的VAE+RL、GAN+RL、Neural Turing Machine+RL……。基于这个观察，我们姑且可以预言一波近期的爆点了。

首先是热门的技术。Transformer在其他领域方兴未艾，而在RL领域中还正在兴起，爆发只是时间问题。已经开始的爆点就是Diffusion Model+RL（分享一个可视化效果不错的项目，[diffusion-planning.github.io](http://diffusion-planning.github.io)）。



[Motion Plan软硬约束下的轨迹生成](https://zhuanlan.zhihu.com/p/672855397)

