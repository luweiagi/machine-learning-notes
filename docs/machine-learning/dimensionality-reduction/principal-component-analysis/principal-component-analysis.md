# PCA主成分分析

* [返回上层目录](../dimensionality-reduction.md)



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
