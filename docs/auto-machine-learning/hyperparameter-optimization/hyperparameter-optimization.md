# 超参数优化HPO

* [返回上层目录](../auto-machine-learning.md)
* [超参数优化HPO简介](#超参数优化HPO简介)
* [超参数优化HPO算法简介](#超参数优化HPO算法简介)
* [超参数优化HPO算法开源工具](#超参数优化HPO算法开源工具)
  * [Katib超参数训练系统](#Katib超参数训练系统)



# 超参数优化HPO简介

超参数优化 (HPO) 是 Hyper-parameter optimization的缩写，是指不是依赖人工调参，而是通过一定算法找出机器学习/深度学习中最优/次优超参数的一类方法。HPO的本质是是生成多组超参数，一次次地去训练，根据获取到的评价指标等调节再生成超参数组再训练，依此类推。从使用过的情况来看，使用HPO的代价较高，尤其是对复杂深度模型或中小企业来说。

# 超参数优化HPO算法简介

 从开源项目来看，超参数优化（HPO）算法一般都包括三类:

1. 暴力搜索类： 随机搜索(random search)、网格搜索(grid search)、自定义(custom)
2. 启发式搜索:   进化算法(Naïve Evolution, NE)、模拟退火算法(Simulate Anneal, SA)、遗传算法(Genetic Algorithm, GA)、粒子群算法(Particle Swarm Optimization, PSO)等自然计算算法、Hyperband
3. 贝叶斯优化:   BOHB、TPE、SMAC、Metis、GP

# 超参数优化HPO算法开源工具

 超参数优化（HPO）算法伴随着机器学习的发展而发展而来，至今已经发展得比较成熟了。

github项目等也比较多，一类是专注于HPO算法的。如hyperopt, advisor，scikit-optimize等；另外一类是大而全的AutoML，

超参数优化HPO只是其中一部分，如nni,  katib,  autokeras,  auto-sklearn等。

可以发现，python版的超参优化HPO颇受sklearn影响，毕竟是是最受欢迎的机器学习算法工具了。

## Katib超参数训练系统

Katib: Kubernetes native 的超参数训练系统





# 参考资料

* [超参数优化与NNI(HPO，Hyper-parameter optimization)](https://blog.csdn.net/rensihui/article/details/104591292)

本文参考了此博客。

* [Katib: Kubernetes native 的超参数训练系统](http://gaocegege.com/Blog/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/katib)

“Katib超参数训练系统”一节参考了此博客。

===

[AutoML——让算法解放算法工程师](https://mp.weixin.qq.com/s/cfFcMyabJjj4qPoBTvvj6A)

[HPO](http://www.noahlab.com.hk/opensource/vega/docs/algorithms/hpo.html)

[【知出乎争】超参自动优化方法总结](https://mp.weixin.qq.com/s/waPWzo6iIEXYaH_MQdLfYg)

本文旨在介绍当前被大家广为所知的超参自动优化方法，像网格搜索、随机搜索、贝叶斯优化和Hyperband，并附有相关的样例代码供大家学习。

写这篇文章的过程中，我主要学到了2点：
一是随机搜索在某些时候会比格子搜索好，因为某些参数对模型影响较小时，使用随机搜索能让我们有更多的探索空间。
二是了解贝叶斯优化的机理（准备采样点->GP拟合代理模型->采集函数选点->更新数据集->循环此过程，直到调优结束）。

但由于贝叶斯开销较大，所以As a general rule of thumb, any time you want to optimize tuning hyperparameters, think Grid Search and Randomized Search!