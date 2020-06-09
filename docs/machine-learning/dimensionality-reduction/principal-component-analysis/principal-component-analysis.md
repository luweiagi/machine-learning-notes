# PCA主成分分析

* [返回上层目录](../dimensionality-reduction.md)





pdf: *[A Tutorial on Principal Component Analysis](https://www.cc.gatech.edu/~lsong/teaching/CX4240spring16/pca_schlens.pdf)*



在实际生产生活中，我们所获得的数据集在特征上往往具有很高的维度，对高维度的数据进行处理时消耗的时间很大，并且过多的特征变量也会妨碍查找规律的建立。**如何在最大程度上保留数据集的信息量的前提下进行数据维度的降低**，是我们需要解决的问题。

对数据进行降维有以下**优点**：

* 使得数据集更易使用
* 降低很多算法的计算开销
* 去除噪声
* 使得结果易懂

降维技术作为数据预处理的一部分，即可使用在监督学习中也能够使用在非监督学习中。其中**主成分分析PCA**应用最为广泛，本文也将详细介绍PCA。

# PCA理解

## 什么是降维

比如说有如下的房价数据：



## 非理想情况如何降维



## 如何进行主成分分析



## 协方差矩阵





## 实例

对于512维度进行不同维度压缩后保留的信息：

```
256:0.9998875105696291
128:0.8964985919468982
64:0.6758438520261455
```





# 总结

作为一个非监督学习的降维方法，它只需要特征值分解，就可以对数据进行压缩，去噪。因此在实际场景应用很广泛。

PCA算法的主要优点有：

* 仅仅需要以方差衡量信息量，不受数据集以外的因素影响。　

* 各主成分之间正交，可消除原始数据成分间的相互影响的因素。

* 计算方法简单，主要运算是特征值分解，易于实现。

PCA算法的主要缺点有：

* 主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强。

* 方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响。



# 参考资料

* [如何通俗易懂地讲解什么是 PCA 主成分分析？](https://www.zhihu.com/question/41120789/answer/481966094)



===

[【直观详解】什么是PCA、SVD](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247486147&idx=1&sn=ae6c99fbd00c9ea7de31d691472dd21c&chksm=ebb43217dcc3bb01fe10d278b51fd44777a6c00ce95107698d6ab4df021856b3885154c07b0b&mpshare=1&scene=1&srcid=12267viOFiJ7XY3qIl9gSt89#rd)

主成分分析（Principal Component Analysis）：一种统计方法，它对多变量表示数据点集合寻找尽可能少的正交矢量表征数据信息特征。

PCA既是一种压缩数据的方式，也是一种学习数据表示的无监督学习方法。《深度学习》5.8.1 P92

有两种解释

1，深度学习 p30 2.12节

2，深度学习 p92 5.8.1节



Dimensionality Reduction——PCA原理篇

https://zhuanlan.zhihu.com/p/28317712

Dimensionality Reduction——PCA实现篇

https://zhuanlan.zhihu.com/p/28327257



PCA主成分分析学习总结

https://zhuanlan.zhihu.com/p/32412043



[理解主成分分析 (PCA)](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484754&idx=1&sn=b2c0d6798f44e13956bb42373e51d18c&chksm=fdb698c5cac111d3e3dca24c50aafbfb61e5b05c5df5b603067bb7edec8db049370b73046b24&mpshare=1&scene=1&srcid=06081PBJlyXnPpa3Clj5AOCM#rd)



[hinton对pca的理解](https://www.zhihu.com/question/30094611/answer/275172932)



[理解主成分分析 (PCA)](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484754&idx=1&sn=b2c0d6798f44e13956bb42373e51d18c&chksm=fdb698c5cac111d3e3dca24c50aafbfb61e5b05c5df5b603067bb7edec8db049370b73046b24&scene=21#wechat_redirect)



# PCA的损失函数

通过指定如下损失函数就可以得到PCA的第一个主向量
$$
J(w)=\mathbb{E}_{X\sim \hat{p}_{data}}\left \|x-r(x;w) \right \|^2_2
$$
模型定义为重构函数
$$
r(x)=w^Txw
$$
，并且w有范数为1的限制。

《深度学习》5.10 p96
