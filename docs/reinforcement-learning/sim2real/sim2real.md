# Sim2Real

* [返回上层目录](../reinforcement-learning.md)



===

[最新综述 | 强化学习中从仿真器到现实环境的迁移 ](https://www.sohu.com/a/472645232_121123911)

其中，提到了

**方法的分类**

sim2real 中的典型工作大致可以分为以下五类：

- **Domain Adaption** 主要是通过学习一个模拟环境以及现实环境共同的状态到隐变量空间的映射，在模拟环境中，使用映射后的状态空间进行算法的训练；因而在迁移到现实环境中时，同样将状态映射到隐含空间后，就可以直接应用在模拟环境训练好的模型了。

- **Progressive Network**利用一类特殊的 Progressive Neural Network 来进行 sim2real。其主要思想类似于 cumulative learning，从简单任务逐步过渡到复杂任务（这里可以认为模拟器中的任务总是要比现实任务简单的）。

- **Inverse Dynamic Model**通过在现实环境中学习一个逆转移概率矩阵来直接在现实环境中应用模拟环境中训练好的模型。

  [Adversarial Active Exploration for Inverse Dynamics Model Learning](https://williamd4112.github.io/pubs/corl19_self-adv.pdf)

  [CS 294：Deep Reinforcement Learning（4）](https://paper.yanxishe.com/questionDetail/9991)

  注：还有一节课是请Open AI的专家更深入地讲解collocation method，是用inverse dynamic model（反动态模型），即![u_t=f^{-1}(x_t,x_{t+1})](https://www.zhihu.com/equation?tex=u_t%3Df%5E%7B-1%7D%28x_t%2Cx_%7Bt%2B1%7D%29)。但已经是最前沿model-based RL，所以我将不会写那节的笔记，有兴趣的同学可自行了解，我们最好还是快点开始model-free RL的学习。

  [**一种基于强化学习和迁移学习的无人机自主飞行训练方法**](http://www.lninfo.com.cn:8088/ShowDetail.aspx?d=1022&id=ZLW20210924000000391510)

- Domain Randomization 对模拟环境中的视觉信息或者物理参数进行随机化，例如对于避障任务，智能体在一个墙壁颜色、地板颜色等等或者摩擦力、大气压强会随机变化的模拟环境中进行学习。

在这一部分我将介绍一下 OpenAI 在 sim2real 领域做出的一个工作，其地位类似于多智能体强化学习领域的 OpenAI Five。

4.1 Learning Dexterous In-Hand Manipulation



[Sim2Real学习总结](https://zhuanlan.zhihu.com/p/510951914)

最近学习了Sim2Real领域的一些相关工作，以此文做一次学习总结，文章主要参照2020的一篇Survey：[《Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey》](https://arxiv.org/pdf/2009.13303.pdf)

