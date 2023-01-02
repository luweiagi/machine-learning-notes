# PPO近端策略优化

- [返回上层目录](../paper.md)



![paper](pic/paper.png)

PDF: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

OpenAI Blog: [Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/)和[Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id3)

GitHub: [openai/baselines/ppo2](https://github.com/openai/baselines/tree/master/baselines/ppo2)





正态分布：
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}\text{exp}\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$
![normal-dist](pic/normal-dist.png)

代码(参考资料：[Python绘制高斯分布（正态分布）图像](https://blog.csdn.net/qq_44444503/article/details/124377863))：

```python
import numpy as np
import math
import matplotlib.pyplot as plt


def gd(x, mu=0., sigma=1.):
    # sigma是标准差
    left = 1 / (np.sqrt(2 * math.pi) * sigma)
    right = np.exp(-(x - mu) ** 2 / (2 * np.square(sigma)))
    return left * right


if __name__ == '__main__':
    # 自变量
    x = np.arange(-5, 5, 0.05)
    # 因变量（不同均值或方差）
    y_1 = gd(x, 0, 0.01)
    y_2 = gd(x, 0, 0.1)
    y_3 = gd(x, 0, 0.5)
    y_4 = gd(x, 0, 1.0)
    y_5 = gd(x, 0, 2.0)

    # 绘图
    plt.plot(x, y_1, color='green')
    plt.plot(x, y_2, color='blue')
    plt.plot(x, y_3, color='thistle')
    plt.plot(x, y_4, color='red')
    plt.plot(x, y_5, color='fuchsia')
    # 设置坐标系
    plt.xlim(-5.0, 5.0)
    plt.ylim(-0.2, 2)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.legend(labels=[
        '$\mu = 0, \sigma=0.01$',
        '$\mu = 0, \sigma=0.1$',
        '$\mu = 0, \sigma=0.5$',
        '$\mu = 0, \sigma=1.0$',
        '$\mu = 0, \sigma=2.0$'])
    plt.show()
```





[游戏AI比赛番外篇2：为啥强化学习做游戏AI都喜欢用PPO算法？](https://zhuanlan.zhihu.com/p/550312933)

必看，因为这篇文章直观介绍了损失函数。



（Generalized Advantage Estimation）训练最稳定，调参最简单，适合高维状态 High-dimensional state，但是环境不能有太多随机因数。GAE会根据经验轨迹 trajectory 生成优势函数估计值，然后让Critic去拟合这个值。在这样的调整下，在随机因素小的环境中，不需要太

[mengwanglalala/**RL-algorithms**](https://github.com/mengwanglalala/RL-algorithms)



[进阶篇---PPO代码逐行分析](https://blog.csdn.net/qq_37395293/article/details/114254505)

A3C在每一次迭代中，都要采集T个samples(T远远小于回合长度)，计算T个n-step return，然后使用这T个数据更新评价网络：

PPO也使用了相同的思路，但是使用GAE实现



# 参考资料







===



* [PPO算法经典论文阅读](https://blog.csdn.net/shengzimao/article/details/126493407)

* [深度增强学习PPO（Proximal Policy Optimization）算法源码走读](https://blog.csdn.net/jinzhuojun/article/details/80417179)



* [PPO强化学习如何实现多维度的动作呢？](https://www.zhihu.com/question/417161289/answer/2207316616)

个人觉得用得比较多的是输出5个均值和方差，再从这5个分布中，采样5个值。

这里给你附上一个Pytorch实现PPO的代码，对多维连续空间进行了详细的实现，代码也比较简洁稳定：

[https://github.com/XinJingHao/PPO-Continuous-Pytorch](https://github.com/XinJingHao/PPO-Continuous-Pytorch)

请问如果action是5维，并且每一维是0,1,2,3,4这样的离散的数，这种情况应该怎么处理呢？

也可以作为independent action （见Section V subsection A [arxiv.org/pdf/2105.1380](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2105.13807.pdf)），代码可见 [github.com/vwxyzjn/PPO-](http://link.zhihu.com/?target=https%3A//github.com/vwxyzjn/PPO-Implementation-Deep-Dive/blob/lstm-and-multidiscrete/ppo_multidiscrete.py)。一般如果你的离散动作空间有很多维度，可能会有无效动作，这时候invalid action masking 很重要，见[arxiv.org/abs/2006.1417](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/2006.14171), 代码可见 [github.com/vwxyzjn/PPO-Implementation-Deep-Dive](http://link.zhihu.com/?target=https%3A//github.com/vwxyzjn/PPO-Implementation-Deep-Dive/blob/lstm-and-multidiscrete/ppo_multidiscrete_mask.py)。









* [如何理解看待 OpenAI 公布PPO算法？ 莫凡](https://www.zhihu.com/question/63067895/answer/214180615)

我也实践了一下 Python 的简单 PPO 算法. 毕竟 OpenAI 开源的那个 [baselines](https://link.zhihu.com/?target=https%3A//github.com/openai/baselines) 太复杂了, 看半天源码也看不懂. 所以下定决心自己写一个比他们的简单好多倍的代码. 自己写的教程在这里: [结合了 OpenAI 和 DeepMind 的 PPO](https://link.zhihu.com/?target=https%3A//mofanpy.com/tutorials/machine-learning/reinforcement-learning/6-4-DPPO/)。额，前面这个地址打不开了，直接看这个：[Distributed Proximal Policy Optimization (DPPO)](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/DPPO)，既可以看代码，也可以看环境源码。特别推荐。

简述一下自己写代码的感想. OpenAI 的 PPO 感觉是个串行的（要等所有并行的 Actor 搞完才更新模型）, DeepMind 的 DPPO 是并行的（不用等全部 worker）, 但是代码实践起来比较困难, 需要推送不同 worker 的 [gradient](https://www.zhihu.com/search?q=gradient&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A214180615}). 我取了 OpenAI 和 DeepMind 两者的精华. 用 OpenAI 中性能最好的 Policy 更新策略 (clipped surrogate) + DeepMind [parallel training](https://www.zhihu.com/search?q=parallel+training&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A214180615}) (但不是推送 gradient, 只是推送 collected data). 让原本[单线程](https://www.zhihu.com/search?q=单线程&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A214180615})的 PPO 速度飞起来了.

你确定你看过OpenAI PPO的原文，原文中算法就是有N个Actor并行训练的！



* [影响PPO算法性能的10个关键技巧（附PPO算法简洁Pytorch实现）](https://zhuanlan.zhihu.com/p/512327050)

具体代码见：[https://github.com/Lizhi-sjtu/DRL-code-pytorch](https://github.com/Lizhi-sjtu/DRL-code-pytorch)



* [强化学习笔记（五）--PPO](https://zhuanlan.zhihu.com/p/48293363)

2017年7月20日，OpenAI 刚刚通过自己的研究博客介绍了一种新的优化算法 Proximal Policy Optimization（近端策略优化，PPO）。据介绍，这种算法用在强化学习中时表现能达到甚至超过现有算法的顶尖水平，同时还更易于实现和调试。所以 OpenAI 已经把PPO作为自己强化学习研究中首选的算法。

之前 OpenAI 就详细介绍过 PPO的一个变种 （[NIPS 2016 论文视频：通过策略优化进行深度强化学习](https://link.zhihu.com/?target=https%3A//channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Deep-Reinforcement-Learning-Through-Policy-Optimization)），其中用一个自适应 Kullback–Leibler 惩罚项控制每次迭代中的策略变化程度。现在介绍的这种新变种则使用了一种其它算法中罕见的全新的目标函数：

PPO2：不用计算KL，同样可以控制θ与θ'之间差距。

说明：当A>0，也就是a是好的，我们希望增加Pθ的概率，但是，Pθ不能弄得太大，太大也就会使与Pθ'差距大，导致效果不好。反之亦然。



* [强化学习之PPO算法](https://zhuanlan.zhihu.com/p/468828804)

写的不错，参考了下面两个文章。

* [李宏毅深度强化学习(国语)课程(2018) ppo](https://www.bilibili.com/video/BV1MW411w79n?p=2&vd_source=147fb813418c7610c21b6a5618c85cb7)

  还有对应的课件：[李宏毅深度强化学习(国语)课程(2018) ppo 课件](https://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/PPO%20(v3).pdf)。

* [Proximal Policy Optimization(PPO)算法原理及实现！ 美团文哥的笔记](https://www.jianshu.com/p/9f113adc0c50)



[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

对应的中文翻译：[优化PPO](https://blog.csdn.net/CharilePuth/article/details/125555567)

