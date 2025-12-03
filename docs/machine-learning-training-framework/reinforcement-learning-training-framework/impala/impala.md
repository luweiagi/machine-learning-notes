# IMPALA

* [返回上层目录](../reinforcement-learning-training-framework.md)

Ray是一个基于内存共享的分布式计算框架，适用于细粒度的并行计算和异构计算。RLlib则是基于Ray开发的强化学习框架，它封装了一系列的强化学习算法和训练器，只需指定算法名称和环境就能开始训练，而算法中使用的深度神经网络模型以及策略中使用的损失函数和梯度优化器等组件都是封装好的。

**正因为RLlib过好的封装，导致其个性化或者说自定义的难度大大提升**。对于初学者而言，只看文档中提供的自定义方法难以完成对模型和策略的自定义。



RLlib的Github：https://github.com/ray-project/ray/tree/master/rllib

# 安装RLLib

**PyTorch:**

```shell
conda create -n rllib python=3.8
conda activate rllib
pip install "ray[rllib]" torch "gymnasium[atari]" "gymnasium[accept-rom-license]" atari_py
# Run a test job:
rllib train --run APPO --env CartPole-v0 --torch
```

运行最后一行代码报错：

```shell
 No such option: --torch (Possible options: --restore, --stop, --trace)
```

# 简单的RLLib例子

参考：[RLlib一：RLlib入门](https://blog.csdn.net/Kiek17/article/details/134358062)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    .build()
)

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
```

其他参考：

[强化学习rllib简明教程 ray](https://blog.csdn.net/weixin_42056422/article/details/113995709)

[机器学习框架Ray -- 1.4 Ray RLlib的基本使用](https://blog.csdn.net/wenquantongxin/article/details/129969925)



下面这个很不错，值得好好看：

[Ray客2代](https://blog.csdn.net/wenquantongxin/category_12276185.html)，这个目录里有：

[机器学习框架Ray -- 3.1 RayRLlib训练Pendulum-v1](https://blog.csdn.net/wenquantongxin/article/details/130280000)





[Ray[RLlib] Customization：自定义模型和策略](https://zhuanlan.zhihu.com/p/460637302)

[RLlib：用户自定义模型代码示例](https://blog.csdn.net/Kiek17/article/details/135237146)

# 参考资料

RLlib：工业级强化学习

[RLlib：一个分布式强化学习系统的凝练](https://zhuanlan.zhihu.com/p/144842398)

[分布式框架Ray及RLlib简易理解](https://zhuanlan.zhihu.com/p/61818897)

[Scaling Deep Reinforcement Learning to a Private Cluster](https://stefanbschneider.github.io/blog/posts/rllib-private-cluster/index.html)

[Welcome to Ray: https://docs.ray.io/en/latest/index.html](https://docs.ray.io/en/latest/index.html)

