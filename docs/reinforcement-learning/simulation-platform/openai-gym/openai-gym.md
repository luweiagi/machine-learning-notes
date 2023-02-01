# OpenAI的Gym

* [返回上层目录](../simulation-platform.md)

[OpenAI Gym 官网](https://www.gymlibrary.dev/)

大家可以到官网上看一下GYM包含的环境。包含了从简单的Classic control、Toy text，到更复杂的MuJoCo、Robotics，当然包括Atari游戏。 环境还在不断扩展，现在也包括一些第三方的环境，例如围棋等。 所以我们经常用它来检验算法的有效性。



Gym的安装方式：

```shell
pip --default-timeout=100 install gym -i https://pypi.tuna.tsinghua.edu.cn/simple
```





[Gym小记（二） 浅析Gym中的env](https://blog.csdn.net/u013745804/article/details/78397106)

对倒立摆环境[Pendulum-v0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)做了分析。

[gym例子及二次开发](https://zhuanlan.zhihu.com/p/462248870)

给出一个最简单的例子，剖析了gym环境构建

[如何创建gym环境](https://blog.csdn.net/stanleyrain/article/details/127880978)

这就比较复杂了