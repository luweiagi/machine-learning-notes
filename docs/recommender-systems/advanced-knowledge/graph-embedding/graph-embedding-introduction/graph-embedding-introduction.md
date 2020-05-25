# 图表征概述

* [返回上层目录](../../advanced-knowledge.md)
* [Network-Embedding概述](#Network-Embedding概述)
* [Embedding方法的学习路径](#Embedding方法的学习路径)
  * [word2vec基础](#word2vec基础)
  * [word2vec的衍生及应用](#word2vec的衍生及应用)
  * [Graph-Embedding](#Graph-Embedding)



# Network-Embedding概述

什么时候更新graph embedding的文章啊，最近在做这一块，很期待
还是找几篇论文看看吧 deepwalk note2vec line 都不难实现

[深度学习中不得不学的Graph Embedding方法](https://zhuanlan.zhihu.com/p/64200072)

[网络表示学习综述：一文理解Network Embedding](https://www.jiqizhixin.com/articles/2018-08-14-10)

[关于Network embedding的一些笔记(内含数据集）](https://blog.csdn.net/ZhichaoDuan/article/details/79570051)

# Embedding方法的学习路径

这篇文章来自[王喆的机器学习笔记](https://zhuanlan.zhihu.com/wangzhenotes)中的[《Embedding从入门到专家必读的十篇论文》](https://zhuanlan.zhihu.com/p/58805184)。

这里是**「王喆的机器学习笔记」**的第十篇文章，今天我们不分析论文，而是总结一下**Embedding方法的学习路径**，这也是我三四年前从接触word2vec，到在推荐系统中应用Embedding，再到现在逐渐从传统的sequence embedding过渡到graph embedding的过程，因此该论文列表在应用方面会对推荐系统、计算广告方面有所偏向。

## word2vec基础

**1.** [[Word2Vec\] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BWord2Vec%255D%2520Efficient%2520Estimation%2520of%2520Word%2520Representations%2520in%2520Vector%2520Space%2520%2528Google%25202013%2529.pdf)

Google的Tomas Mikolov提出word2vec的两篇文章之一，这篇文章更具有综述性质，列举了NNLM、RNNLM等诸多词向量模型，但最重要的还是提出了CBOW和Skip-gram两种word2vec的模型结构。虽然词向量的研究早已有之，但不得不说还是Google的word2vec的提出让词向量重归主流，拉开了整个embedding技术发展的序幕。

**2**. [[Word2Vec\] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BWord2Vec%255D%2520Distributed%2520Representations%2520of%2520Words%2520and%2520Phrases%2520and%2520their%2520Compositionality%2520%2528Google%25202013%2529.pdf)

Tomas Mikolov的另一篇word2vec奠基性的文章。相比上一篇的综述，本文更详细的阐述了Skip-gram模型的细节，包括模型的具体形式和 Hierarchical Softmax和 Negative Sampling两种可行的训练方法。

**3**. [[Word2Vec\] Word2vec Parameter Learning Explained (UMich 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BWord2Vec%255D%2520Word2vec%2520Parameter%2520Learning%2520Explained%2520%2528UMich%25202016%2529.pdf)

虽然Mikolov的两篇代表作标志的word2vec的诞生，但其中忽略了大量技术细节，如果希望完全读懂word2vec的原理和实现方法，比如词向量具体如何抽取，具体的训练过程等，强烈建议大家阅读UMich Xin Rong博士的这篇针对word2vec的解释性文章。惋惜的是Xin Rong博士在完成这篇文章后的第二年就由于飞机事故逝世，在此也致敬并缅怀一下Xin Rong博士。

## word2vec的衍生及应用

**4**. [[Item2Vec\] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BItem2Vec%255D%2520Item2Vec-Neural%2520Item%2520Embedding%2520for%2520Collaborative%2520Filtering%2520%2528Microsoft%25202016%2529.pdf)

这篇论文是微软将word2vec应用于推荐领域的一篇实用性很强的文章。该文的方法简单易用，可以说极大拓展了word2vec的应用范围，使其从NLP领域直接扩展到推荐、广告、搜索等任何可以生成sequence的领域。

**5**. [[Airbnb Embedding\] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BAirbnb%2520Embedding%255D%2520Real-time%2520Personalization%2520using%2520Embeddings%2520for%2520Search%2520Ranking%2520at%2520Airbnb%2520%2528Airbnb%25202018%2529.pdf)

Airbnb的这篇论文是KDD 2018的best paper，在工程领域的影响力很大，也已经有很多人对其进行了解读。简单来说，Airbnb对其用户和房源进行embedding之后，将其应用于搜索推荐系统，获得了实效性和准确度的较大提升。文中的重点在于embedding方法与业务模式的结合，可以说是一篇应用word2vec思想于公司业务的典范。

## Graph-Embedding

基于word2vec的一系列embedding方法主要是基于序列进行embedding，在当前商品、行为、用户等实体之间的关系越来越复杂化、网络化的趋势下，原有sequence embedding方法的表达能力受限，因此Graph Embedding方法的研究和应用成为了当前的趋势。

**6**. [[DeepWalk\] DeepWalk- Online Learning of Social Representations (SBU 2014)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BGraph%2520Embedding%255D%2520DeepWalk-%2520Online%2520Learning%2520of%2520Social%2520Representations%2520%2528SBU%25202014%2529.pdf)

以随机游走的方式从网络中生成序列，进而转换成传统word2vec的方法生成Embedding。这篇论文可以视为Graph Embedding的baseline方法，用极小的代价完成从word2vec到graph embedding的转换和工程尝试。

[Graph embedding: 从Word2vec到DeepWalk](https://zhuanlan.zhihu.com/p/59887204)

**7**. [[LINE\] LINE - Large-scale Information Network Embedding (MSRA 2015)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BLINE%255D%2520LINE%2520-%2520Large-scale%2520Information%2520Network%2520Embedding%2520%2528MSRA%25202015%2529.pdf)

相比DeepWalk纯粹随机游走的序列生成方式，LINE可以应用于有向图、无向图以及边有权重的网络，并通过将一阶、二阶的邻近关系引入目标函数，能够使最终学出的node embedding的分布更为均衡平滑，避免DeepWalk容易使node embedding聚集的情况发生。

**8**. [[Node2vec\] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BNode2vec%255D%2520Node2vec%2520-%2520Scalable%2520Feature%2520Learning%2520for%2520Networks%2520%2528Stanford%25202016%2529.pdf)

node2vec这篇文章还是对DeepWalk随机游走方式的改进。为了使最终的embedding结果能够表达网络局部周边结构和整体结构，其游走方式结合了深度优先搜索和广度优先搜索。

**9**. [[SDNE\] Structural Deep Network Embedding (THU 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BSDNE%255D%2520Structural%2520Deep%2520Network%2520Embedding%2520%2528THU%25202016%2529.pdf)

相比于node2vec对游走方式的改进，SDNE模型主要从目标函数的设计上解决embedding网络的局部结构和全局结构的问题。而相比LINE分开学习局部结构和全局结构的做法，SDNE一次性的进行了整体的优化，更有利于获取整体最优的embedding。

**10**. [[Alibaba Embedding\] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (Alibaba 2018)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%255BAlibaba%2520Embedding%255D%2520Billion-scale%2520Commodity%2520Embedding%2520for%2520E-commerce%2520Recommendation%2520in%2520Alibaba%2520%2528Alibaba%25202018%2529.pdf)

阿里巴巴在KDD 2018上发表的这篇论文是对Graph Embedding非常成功的应用。从中可以非常明显的看出从一个原型模型出发，在实践中逐渐改造，最终实现其工程目标的过程。这个原型模型就是上面提到的DeepWalk，阿里通过引入side information解决embedding问题非常棘手的冷启动问题，并针对不同side information进行了进一步的改造形成了最终的解决方案EGES（Enhanced Graph Embedding with Side Information）。

注：由于上面十篇论文都是我之前整理的paper list里面的内容，所以没有再引用原文链接，希望大家见谅。想偷懒的同学也可以star或者fork我的github paper list：[wzhe06/Reco-papers](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/tree/master/Embedding)



这里是**「王喆的机器学习笔记」 ，**关于Embedding的这十篇论文包括了从基础理论、模型改造与进阶、模型应用等几个方面的内容，还是比较全面的，希望能帮助你成为相关方向的专家。但一个人的视野毕竟有局限性，希望大家能够反馈给我其他embedding相关的著名文章，我可以进行补充和替换。



# 参考资料

* [Embedding从入门到专家必读的十篇论文-知乎王喆](https://zhuanlan.zhihu.com/p/58805184)

"Embedding方法的学习路径"一节参考了此知乎专栏文章。
