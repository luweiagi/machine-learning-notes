# 机器学习面试

* [返回上层目录](../perface.md)
* [互联网AI岗位所需技能](#互联网AI岗位所需技能)
  * [系统工程师](#系统工程师)
  * [算法工程师](#算法工程师)
  * [数据挖掘工程师](#数据挖掘工程师)
* [常见面试问题](#常见面试问题)
  * [机器学习](#机器学习)
  * [深度学习](#深度学习)
  * [代码能力](#代码能力)
  * [工程能力](#工程能力)



# 互联网AI岗位所需技能

## 系统工程师

- **Linux 基本命令及 Bash Shell**

推荐阅读：《鸟哥的Linux私房菜》

- C/C++
  - 代码规范
  - C++11新特性

推荐阅读：《踏潮C++代码规范》、《Effective C++/STL》

- RPC框架
  - Thrift
  - Protobuf
- Web框架
  - Nginx with FastCGI
  - Apache
  - Django
- 数据存储
  - MySQL
  - MongoDB
  - Redis
  - Hadoop
  - HBase
  - Kafka
- 网络编程
  - 多线程同步
  - 进程通信
  - 流处理
- 分布式
  - 数据同步
  - Master-Slave
  - 竞选机制

## 算法工程师

- **Linux 基本命令及 Bash Shell**
- C/C++
  - 代码规范
  - C++11新特性

推荐阅读：《踏潮C++代码规范》、《Effective C++/STL》

- 回归计算
  - 最大似然估计
  - 随机梯度下降
- 分布式计算
  - MapReduce
- 并行计算
  - 加速比评测
  - 可扩放性标准
  - PRAM模型
  - POSIX Threads
  - CUDA基础

## 数据挖掘工程师

- 数据转换
  - 无量纲化
  - 归一化
  - 哑编码
- 数据清洗
  - 判断异常值
  - 缺失值计算
- 特征工程
  - 可用性评估
  - 采样
  - PCA/LDA
  - 衍生变量
  - L1/L2正则
  - SVD分解
- 提升
  - Adaboost
  - 加法模型
  - xgboost
- SVM
  - 软间隔
  - 损失函数
  - 核函数
  - SMO算法
  - libSVM
- 聚类
  - K-Means
  - 并查集
  - K-Medoids
  - KNN
  - 聚谱类SC
- EM算法
  - Jensen不等式
  - 混合高斯分布
  - pLSA
- 主题模型
  - 共轭先验分布
  - 贝叶斯
  - 停止词和高频词
  - TF-IDF
- 词向量
  - word2vec
  - n-gram
- HMM
  - 前向/后向算法
  - Baum-Welch
  - Viterbi
  - 中文分词
- 数据计算平台
  - Spark
  - Caffe
  - Tensorflow
- 推荐阅读：周志华——《机器学习》

# 常见面试问题

## 机器学习

* SVM和LR的区别和联系

* LR可不可以做非线性分类

* bagging和boosting的区别

* 决策树，GBDT

* 介绍一下xgboost，xgboost和GBDT的区别

* l1正则化和l2正则化

* 如何处理类别不均衡问题

* 如何判断分类器的好坏（分类器的评价指标）

* 介绍Kmeans算法

* 特征工程的方法

## 深度学习

* 防止过拟合的方法，dropout的原理

* 常用的激活函数，sigmoid的缺点，relu的缺点

* 平时用什么优化方法，adam的缺点及解决方法

* 介绍平时调参的经验

* 介绍BatchNorm

* RNN的缺点，LSTM的作用，LSTM和GRU

* 除了LSTM和GRU是否了解其它循环单元

* 如何处理OOV问题

* 介绍word2vec的作用，缺点

* 常用的文本分类方法，优缺点

* 残差网络resnet介绍，解决什么问题

* CNN在图像中1*1卷积的作用

* 一维卷积的作用

* 介绍transformer，为什么比LSTM好，怎么获取顺序信息

* 介绍BERT，ELMO等

机器学习/深度学习这里，《**百面机器学习**》看两遍就可以了，如果有多余时间可以看一遍西瓜书或者统计学习方法。

## 代码能力

除了机器学习/深度学习水平之外，还要考察工程能力，以及代码编写。代码编写准备的方法比较统一，就是刷题。至少要刷完《剑指offer》的所有以及leetcode的"top interview questions"。需要注意的是写题并不是写对了就可以了，还要尽量保证代码的工整性，以及复杂度、异常处理、边界值等等。

## 工程能力

工程能力就是考察“机器学习工程师”岗位除了“机器学习”的相关能力了。考察的点很多，比如操作系统，线程/进程，编程语言特点等等，这块通常面试算法工程师的同学都比较薄弱，需要多准备。分享几个常考到的：

* 进程/线程的区别，python中怎么实现，谈谈GIL

* python2和python3的区别

* python中浅拷贝和深拷贝

* python的生成器是什么



# 参考资料

* [踏潮 BI 学习大纲](https://zhuanlan.zhihu.com/p/22543073)

"互联网AI岗位所需技能"这一节内容来源于此文章。

* [如何准备机器学习工程师的面试 ？](https://www.zhihu.com/question/23259302/answer/1136153589)

“常见面试问题”主要参考此知乎回答。

===

* [七月在线：一站式刷遍各大互联网公司人工智能笔试面试题](https://www.julyedu.com/question/index?utm_source=zh&utm_medium=tiku&utm_campaign=t&utm_content=ti&utm_term=ti)

涵盖了所有知识点和课程。


