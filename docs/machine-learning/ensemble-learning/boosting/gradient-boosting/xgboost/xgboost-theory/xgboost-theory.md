# XGBoost理论

* [返回上层目录](../xgboost.md)
* [XGBoost概述](#XGBoost概述)
* [模型](#模型)
* [损失函数](#损失函数)
  * [正则项](#正则项)
  * [牛顿法](#牛顿法)
  * [损失函数的二阶泰勒展开](#损失函数的二阶泰勒展开)
  * [损失函数求导得最优值](#损失函数求导得最优值)
* [优化算法](#优化算法)
  * [XGBoost的增益函数](#XGBoost的增益函数)
  * [树结点分裂方法（split finding）](#树结点分裂方法（split finding）)
    * [暴力枚举（Basic Exact Greedy Algorithm）](#暴力枚举（Basic Exact Greedy Algorithm）)
    * [近似算法（Approximate Algo for Split Finding）](#近似算法（Approximate Algo for Split Finding）)
    * [加权分位点（Weighted Quantile Sketch）](#加权分位点（Weighted Quantile Sketch）)
    * [结点分裂时多机并行](#结点分裂时多机并行)
  * [稀疏感知分割（缺失值处理）](#稀疏感知分割（缺失值处理）)
* [XGBoost的系统设计](#XGBoost的系统设计)
  * [分块并行（Column Block for Parallel Learning）](#分块并行（Column Block for Parallel Learning）)
  * [缓存优化（Cache Aware Access）](#缓存优化（Cache Aware Access）)
  * [核外块计算（Blocks for Out-of-core Computation）](#核外块计算（Blocks for Out-of-core Computation）)
  * [XGBoost的其他特性](#XGBoost的其他特性)
* [XGBoost总结](#XGBoost总结)
  * [XGBoost优点](#XGBoost优点)
  * [XGBoost缺点](#XGBoost缺点)
  * [XGBoost和GradientBoost的比较](#XGBoost和GradientBoost的比较)
* [Xgboost使用经验总结](#Xgboost使用经验总结)



XGBoost本质只过不就是函数空间上的牛顿法（也可理解为自适应变步长的梯度下降法），使用了损失函数的二阶导数信息，所以收敛更快。

![xgboost-paper](pic/xgboost-paper.png)

paper： [*XGBoost: A Scalable Tree Boosting System*](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)

陈天奇演讲PPT：[《Introduction to Boosted Trees 》](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)

陈天奇论文演讲PPT翻译版：[GBM之GBRT总结（陈天奇论文演讲PPT翻译版）](http://nanjunxiao.github.io/2015/08/05/GBM%E4%B9%8BGBRT%E6%80%BB%E7%BB%93/)

[XGBoost官网](https://XGBoost.readthedocs.io/en/latest/)

# XGBoost概述

最近引起关注的一个Gradient Boosting算法：XGBoost，在计算速度和准确率上，较GBDT有明显的提升。XGBoost 的全称是eXtreme Gradient Boosting，它是Gradient Boosting Machine的一个c++实现，作者为正在华盛顿大学研究机器学习的大牛陈天奇 。XGBoost最大的特点在于，它能够自动利用CPU的多线程进行并行，同时在算法上加以改进提高了精度。它的处女秀是Kaggle的希格斯子信号识别竞赛，因为出众的效率与较高的预测准确度在比赛论坛中引起了参赛选手的广泛关注。值得我们在GBDT的基础上对其进一步探索学习。

XGBoost是从决策树一步步发展而来的：

- 决策树 ⟶ 对样本重抽样，然后多个树平均 ⟶ Tree bagging
- Tree bagging ⟶ 再同时对特征进行随机挑选 ⟶ 随机森林
- 随机森林 ⟶ 对随机森林中的树进行加权平均，而非简单平均⟶ Boosing (Adaboost, GradientBoost)
- boosting ⟶ 对boosting中的树进行正则化 ⟶ XGBoosting

从这条线一路发展，就能看出XGBoost的优势了。

# 模型

给定数据集$D = \{ (x_i, y_i) \}$，XGBoost进行additive training，学习$K$棵树，采用以下函数对样本进行预测：
$$
\hat{y}_i=\phi(x_i)=\sum_{k=1}^Kf_k(x_i),\quad f_k\in F
$$
，这里$F$是假设空间，$f(x)$是回归树（CART）：
$$
F=\{ f(x)=w_{q(x)} \}\ (q: \mathbb{R}^m\rightarrow T, w\in\mathbb{R}^T)
$$
$q(x)$表示将样本$x$分到了某个叶子节点上，$w$是叶子节点的分数（leaf score），所以，$w_{q(x)}$表示回归树对样本的预测值。

**例子**：预测一个人是否喜欢玩电脑游戏

![predict-play-computer-game](pic/predict-play-computer-game.png)

回归树的预测输出是实数分数，可以用于回归、分类、排序等任务中。对于回归问题，可以直接作为目标值，对于分类问题，需要映射成概率
$$
\sigma(z)=\frac{1}{1+\text{exp}(-z)}
$$

# 损失函数

XGBoost的损失函数（在函数空间中，即把函数当做自变量）
$$
L(\phi)=\sum_i l(\hat{y}_i,y_i)+\sum_k\Omega(f_k)
$$
其中，
$$
\Omega(f)=\gamma T+\frac{1}{2}\lambda||w||^2
$$
上式中，$\Omega(f)$为正则项，对每棵回归树的复杂度进行了惩罚。

相比原始的GBDT，XGBoost的目标函数多了正则项，使得学习出来的模型更加不容易过拟合。

## 正则项

有哪些指标可以衡量树的复杂度？

树的深度，内部节点个数，**叶子节点个数$T$**，**叶节点分数$w$** ...

而XGBoost采用的：
$$
\Omega(f)=\gamma T+\frac{1}{2}\lambda||w||^2
$$
对叶子节点个数和叶节点分数进行惩罚，相当于在训练过程中做了剪枝。

## 牛顿法

在看下一节前，有必要讲下[牛顿法](https://blog.csdn.net/qq_41577045/article/details/80343252?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)。

将$L(\theta^t)$在$\theta^{t-1}$处进行二阶泰勒展开：
$$
L(\theta^t)\approx L(\theta^{t-1})+L'(\theta^{t-1})\bigtriangleup\theta+L''(\theta^{t-1})\frac{\bigtriangleup\theta^2}{2}
$$
为了简化分析，假设参数是标量（即$\theta$只有一维），则可将一阶和二阶导数分别记为$g$和$h$：
$$
L(\theta^t)\approx L(\theta^{t-1})+g\bigtriangleup\theta+h\frac{\bigtriangleup\theta^2}{2}
$$
要使$L(\theta^t)$极小，即让$g\bigtriangleup\theta+h\frac{\bigtriangleup\theta^2}{2}$极小，可令：
$$
\frac{\partial \left( g\bigtriangleup \theta + h\frac{\bigtriangleup\theta^2}{2} \right)}{\partial \bigtriangleup \theta}=0
$$
求得$\bigtriangleup\theta=-\frac{g}{h}$，故
$$
\theta^t=\theta^{t-1}+\bigtriangleup\theta=\theta^{t-1}-\frac{g}{h}
$$
将参数$\theta$推广到向量形式，迭代公式：
$$
\theta^t=\theta^{t-1}-H^{-1}g
$$
这里$H$是海森矩阵。

怎么理解上面的迭代公式呢？其实很简单，可以理解为**自适应变步长的梯度下降法**。

我们回顾一下梯度下降法：
$$
\theta^t=\theta^{t-1}-\alpha L'(\theta^{t-1})=\theta^{t-1}-\alpha g
$$
看出来了没？牛顿法的$-(1/h)g$就相当于梯度下降法的$-\alpha g$，也就是**牛顿法中的梯度下降的学习率不再是固定的$\alpha$了，而是自适应的$1/h$了**，这个$1/h$是怎么自适应的呢？$h$是二阶导，$h$较大的时候，说明函数变化剧烈，所以学习率$1/h$就会很小；而$h$较小的时候，说明函数变化不剧烈，几乎就是一条直线，那么学习率$1/h$就会变大。所以**牛顿法要比梯度下降法收敛迅速**，因为它还获知了函数的二阶导数这一信息。

## 损失函数的二阶泰勒展开

第$t$次迭代后，模型的预测等于前$t-1$次的模型预测加上第$t$棵树的预测：
$$
\hat{y}^{(t)}_i=\hat{y}_i^{(t-1)}+f_t(x_i)
$$
此时损失函数可写作：
$$
L^{(t)}=\sum_{i=1}^nl(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)
$$
公式中，$y_i,\hat{y}_i^{(t-1)}$都已知，模型要学习的只有第$t$棵树$f_t$。

将损失函数在$\hat{y}_i^{(t-1)}$处进行二阶泰勒展开：
$$
L^{(t)}\approx\sum_{i=1}^n\left[ l(y_i, \hat{y}^{(t-1)}_i)+g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i) \right]+\Omega(f_t)
$$
其中，
$$
g_i=\frac{\partial l(y_i, \hat{y}^{(t-1)}_i)}{\partial \hat{y}^{(t-1)}_i},\quad
h_i=\frac{\partial^2 l(y_i, \hat{y}^{(t-1)}_i)}{\partial^2 \hat{y}^{(t-1)}_i}
$$
来，答一个小问题，在优化第$t$棵树时，有多少个$g_i$和$h_i$要计算？嗯，没错就是各有$N$个，$N$是训练样本的数量。如果有10万样本，在优化第$t$棵树时，就需要计算出个10万个$g_i$和$h_i$。感觉好像很麻烦是不是？但是你再想一想，**这10万个$g_i$之间是不是没有啥关系？是不是可以并行计算呢？**聪明的你想必再一次感受到了，为什么XGBoost会辣么快！因为$g_i$和$h_i$可以并行求出来。

而且，$g_i$和$h_i$是不依赖于损失函数的形式的，只要这个损失函数二次可微就可以了。这有什么好处呢？好处就是XGBoost可以**支持自定义损失函数**，只需满足二次可微即可。强大了我的哥是不是？

将公式中的常数项去掉（不影响求极值），得到：
$$
\tilde{L}^{(t)}=\sum_{i=1}^n\left[ g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i) \right]+\Omega(f_t)
$$
把$f_t, \Omega(f_t)$写成**树结构的形式**，即把下式带入损失函数中
$$
f(x)=w_{q(x)},\quad \Omega(f)=\gamma T+\frac{1}{2}\lambda||w||^2
$$
**注意：**这里出现了$\gamma$和$\lambda$，这是XGBoost自己定义的，在使用XGBoost时，你可以设定它们的值，显然，$\gamma$越大，表示越希望获得结构简单的树，因为此时对较多叶子节点的树的惩罚越大。$\lambda$越大也是越希望获得结构简单的树。为什么XGBoost要选择这样的正则化项？很简单，好使！效果好才是真的好。

得到：
$$
\begin{aligned}
\tilde{L}^{(t)}&=\sum_{i=1}^n\left[ g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i) \right]+\Omega(f_t)\\
&=\sum_{i=1}^n\left[ g_iw_{q(x_i)}+\frac{1}{2}h_iw_{q(x_i)}^2 \right]+\gamma T +\lambda\frac{1}{2}\sum_{j=1}^Tw_j^2\\
\end{aligned}
$$
注意上式最后一行的**左边是对样本的累加**，**右边是对叶节点的累加**，这该怎么**统一**起来？

定义每个叶节点$j$上的样本集合为（这里需要停一停，这里很重要，但是也不难理解，小学知识，认真体会下。$I_j$代表什么？它代表一个集合，即被分到第$j$个叶子结点的所有样本集合，集合中每个值代表一个训练样本的序号，整个集合就是被第$t$棵CART树分到了第$j$个叶子节点上的训练样本。）
$$
I_j=\{ i|q(x_i)=j \}
$$
需要解释下这个$w_{q(x)}$的定义，首先，一棵树有$T$个叶子节点，这$T$个叶子节点的值组成了一个$T$维向量$w$，$q(x)$是一个映射，用来将样本映射成1到$T$的某个值，也就是把它分到某个叶子节点，**$q(x)$其实就代表了CART树的结构。$w_{q(x)}$自然就是这棵树对样本$x$的预测值了。**

则损失函数可以写成按**叶节点**累加的形式：
$$
\begin{aligned}
\tilde{L}^{(t)}&=\sum_{i=1}^n\left[ g_iw_{q(x_i)}+\frac{1}{2}h_iw_{q(x_i)}^2 \right]+\gamma T +\frac{1}{2}\lambda\sum_{j=1}^Tw_j^2\\
&=\sum_{j=1}^T\left[ \left(\sum_{i\in I_j}g_i\right)w_j+\frac{1}{2}\left(\sum_{i\in I_j}h_i+\lambda\right)w^2_j \right]+\gamma T\\
&=\sum_{j=1}^T\left[ G_jw_j+\frac{1}{2}\left(H_j+\lambda\right)w^2_j \right]+\gamma T\\
\end{aligned}
$$
**这里是XGBoost最精髓的部分，它将基于样本的loss转化为了基于叶子节点的loss，即完成了参数的转变，这样才能将loss部分和正则部分都转为叶子节点$T$的目标方程**。

## 损失函数求导得最优值

如果确定了树的结构（即$q(x)$确定，但注意目前还没讲怎么确定树的结构），上式中叶子节点权重$w_j$有闭式解。

为了使目标函数最小，可以令上式导数为0：
$$
\begin{aligned}
\frac{\partial \tilde{L}^{(t)}}{\partial w_j}&=\frac{\partial \sum_{j=1}^T\left[ G_jw_j+\frac{1}{2}\left(H_j+\lambda\right)w^2_j \right]+\gamma T}{\partial w_j}\\
&=G_j+(H_j+\lambda)w_j\\
&=0\\
\end{aligned}
$$
解得每个叶节点的最优预测分数为：
$$
w^*_j=-\frac{G_j}{H_j+\lambda}
$$
然后将让损失函数最小的$w^{*}_j$（即上式）带入损失函数，得到最小损失为：
$$
\tilde{L}^*=-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T
$$
$\tilde{L}^*$代表了什么呢？它表示了**这棵树的结构有多好**，值越小，代表这样结构越好！也就是说，它是衡量第$t$棵CART树的结构好坏的标准。注意~注意~注意~，这个值仅仅是用来衡量结构的好坏的，与叶子节点的值可是无关的。为什么？请再仔细看一下$\tilde{L}^*$的推导过程。$\tilde{L}^*$只和$G_j$、$H_j$和$T$有关，而它们又只和树的结构$q(x)$有关，与叶子节点的值可是半毛关系没有。

上式能视作衡量函数来测量树结构$q$的质量，类似不纯度（基尼系数）的**衡量指标**，来衡量一棵树的优劣程度。下图展示了如何计算一棵树的分值：

![Gi-Hi](pic/Gi-Hi.png)

这里，我们对$w^{*}_j$给出一个直觉的解释，以便能获得感性的认识。我们假设分到$j$这个叶子节点上的样本只有一个。那么，$w^{*}_j$就变成如下这个样子：
$$
w_j^*=\left(\frac{1}{h_j+\lambda}\right)\cdot(-g_j)
$$
这个式子告诉我们，$w^{*}_j$的最佳值就是负的梯度乘以一个权重系数，该系数类似于随机梯度下降中的学习率。观察这个权重系数，我们发现，$h_j$越大，这个系数越小，也就是学习率越小。$h_j$越大代表什么意思呢？代表在该点附近梯度变化非常剧烈，可能只要一点点的改变，梯度就从10000变到了1，所以，此时，我们在使用反向梯度更新时步子就要小而又小，也就是权重系数要更小。

![newton-methed-to-newton-boosting](pic/newton-methed-to-newton-boosting.png)

**补充个理解上很重要的点，之前的GBM模型（GBDT、GBRank、LambdaMART等）都把Loss加在的树间而未改动单棵CART内部逻辑（或者说无伤大雅懒得改），XGBoost因为正则化要考虑优化树复杂度的原因，把Loss带入到CART分裂目标和节点权重上去了（或者说把树内和树间的优化目标统一了），即节点权重已经给出来了：**
$$
w^*_j=-\frac{G_j}{H_j+\lambda}
$$
**并不是像GBDT那样为了特意去拟合$-g/h$，所以，XGBoost新的树的输入其实无所谓，但为了计算$h_i$和$g_i$，则输入就成了**
$$
(y_i,\hat{y}_i^{(t-1)})
$$
**，而GBDT下一棵树的输入是$(x_i,-G_i)$。但是XGBoost已经把这种梯度带入到CART分裂目标和节点权重上去了，表现在其叶子节点的值是$-G_j/(H_j+\lambda)$，而非对$y_i$的拟合。**

**也就是说，XGBoost不刻意拟合任何数值，它在第$t$步只是寻找一种能使当前损失最小的树（GBDT也是寻找使损失最小的树，只不过它靠拟合残差）。因此它不像adaboost（拟合带权值样本集）和gbdt（拟合负梯度）一样以拟合为核心，而是以使损失函数最低为核心。它的方法就是通过分裂节点，使得新树的gain大于原来树的gain，从而降低损失函数，而不是数据拟合。**

在目标函数是二分类log loss损失函数下，这里给出一阶导$g_i$和二阶导$h_i$的推导：
$$
\begin{aligned}
l(y_i,\hat{y}_i^{(t-1)})&=-\sum_{i=1}^N\left( y_i\text{log}\hat{p}_i+(1-y_i)\text{log}(1-\hat{p}_i) \right)\\
&=-\sum_{i=1}^N\left( y_i\text{log}\left( \frac{1}{1+\text{exp}(-\hat{y}_i^{(t-1)})} \right)+(1-y_i)\text{log}\left(\frac{\text{exp}(-\hat{y}_i^{(t-1)})}{1+\text{exp}(-\hat{y}_i^{(t-1)})}\right) \right)\\
\end{aligned}
$$
则$g_i$为：
$$
\begin{aligned}
g_i&=\frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}\\
&=-y_i\left( 1-\frac{1}{1+\text{exp}(-\hat{y}_i^{(t-1)})} \right)+
(1-y_i)\left( \frac{1}{1+\text{exp}(-\hat{y}_i^{(t-1)})} \right)\\
&=\frac{1}{1+\text{exp}(-\hat{y}_i^{(t-1)})}-y_i\\
&=\text{Pred}-\text{Label}
\end{aligned}
$$
$h_i$为：
$$
\begin{aligned}
h_i&=\frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial^2 \hat{y}_i^{(t-1)}}\\
&=\frac{\text{exp}(-\hat{y}_i^{(t-1)})}{\left( 1+\text{exp}(-\hat{y}_i^{(t-1)}) \right)^2}\\
&=\text{Pred}\cdot (1-\text{Pred})\\
\end{aligned}
$$

# 优化算法

当回归树的结构确定时，我们前面已经推导出其最优的叶节点分数以及对应的最小损失值，问题是**怎么确定树的结构**？

* 暴力枚举所有的树结构，选择损失值最小的—NP难问题
* **贪心法，每次尝试分裂一个叶节点，计算分裂前后的增益，选择增益最大的**

暴力枚举显然不现实，这里我们像决策树划分那样选择贪心法。

## XGBoost的增益函数

有了评判树的结构好坏的标准，我们就可以先求最佳的树结构，这个定出来后，最佳的叶子结点的值实际上在上面已经求出来了。

分裂前后的增益怎么计算？

* ID3算法采用信息增益
* C4.5算法采用信息增益比
* CART采用Gini系数
* **XGBoost呢？**

其实前面我们已经XGBoost的最小损失为
$$
\tilde{L}^*=-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T
$$
上式中的
$$
\frac{G_j^2}{H_j+\lambda}
$$
衡量了每个叶子结点对总体损失的贡献，我们希望损失越小越好，则标红部分的值越大越好。

一棵树在该衡量指标下分值越低，说明这个树的结构越好（表示的是损失）。训练数据可能有很多特征，构建一棵树可能有许多种不同的构建形式，我们不可能枚举所有可能的树结构$q$来一一计算它的分值。所以主要采用贪心算法来解决这个问题，贪心算法从一个单独树叶开始，迭代地增加分支，直到最后停止。（如何更快地生成树是关键）

因此，对一个叶子结点进行分裂，分裂前后的增益定义为（**树间和树内的loss是统一的**）：
$$
\begin{aligned}
&\text{Gain}_{\text{split}}\\
=&\tilde{L}^*_{pre\_split}-\tilde{L}^*_{aft\_split}\\
=&\left(-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma\right)-\left(-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+2\gamma\right)\\
=&\frac{1}{2}\left[ \frac{(\sum_{i\in I_L} g_i)^2}{\sum_{i\in I_L} h_i+\lambda} + \frac{(\sum_{i\in I_R} g_i)^2}{\sum_{i\in I_R} h_i+\lambda} - \frac{(\sum_{i\in I} g_i)^2}{\sum_{i\in I} h_i+\lambda} \right]-\gamma\\
=&\frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right]-\gamma
\end{aligned}
$$
这个公式的计算结果，通常用于在实践中评估候选分裂节点是不是应该分裂的划分依据，我们尽量找到使之最大的特征值划分点。

$\text{Gain}$的值越大，分裂后损失函数减小越多。所以当对一个叶节点分割时，计算所有候选（feature, value）对应的$\text{Gain}$，选取$\text{Gain}$最大的进行分割。

这个$\text{Gain}$实际上就是单节点的$\tilde{L}^*$减去切分后的两个节点的树$\tilde{L}^*$，$\text{Gain}$如果是正的，并且值越大，表示切分后$\tilde{L}^*$越小于单节点的$\tilde{L}^*$，就越值得切分。**同时**，我们还可以观察到，$\text{Gain}$的左半部分如果小于右侧的$\gamma$，则$\text{Gain}$就是负的，表明切分后$\tilde{L}^*$反而变大了。$\gamma$在这里实际上是一个临界值，它的值越大，表示我们对切分后$\tilde{L}^*$下降幅度要求越严。这个值也是可以在xgboost中设定的。

扫描结束后，我们就可以确定是否切分，如果切分，对切分出来的两个节点，递归地调用这个切分过程，我们就能获得一个相对较好的树结构。

注意：xgboost的切分操作和普通的决策树切分过程是不一样的。普通的决策树在切分的时候并不考虑树的复杂度，而依赖后续的剪枝操作来控制。**xgboost在切分的时候就已经考虑了树的复杂度，就是那个$\gamma$参数**。所以，它不需要进行单独的剪枝操作。

为了限制树的生长，我们可以加入阈值，当增益大于阈值时才让节点分裂，上式中的$\gamma$即阈值，它是正则项里叶子节点数$T$的系数，所以xgboost在优化目标函数的同时相当于做了**预剪枝**。另外，上式中还有一个系数$\lambda$，是正则项里leaf score的$L_2$模平方的系数，对leaf score做了平滑，也起到了**防止过拟合**的作用，这个是传统GBDT里不具备的特性。

最优的树结构找到后，确定最优的叶子节点就很容易了。我们成功地找出了第$t$棵树！

## 树结点分裂方法（split finding）

**注意：**xgboost的切分操作和普通的决策树切分过程是不一样的。普通的决策树在切分的时候并不考虑树的复杂度，而依赖后续的剪枝操作来控制。xgboost在切分的时候就已经考虑了树的复杂度，就是那个$\text{Gain}_{split}$中的$\gamma$参数。所以，它不需要进行单独的剪枝操作。

### 暴力枚举（Basic Exact Greedy Algorithm）

在树学习中，一个关键问题是**如何找到每一个特征上的分裂点**。为了找到最佳分裂节点，分裂算法枚举特征上所有可能的分裂点，然后计算得分，这种算法称为Exact Greedy Algorithm，单机版本的XGBoost支持这种Exact Greedy Algorithm，算法如下所示：

遍历所有特征的所有可能的分割点，计算$\text{Gain}$值，选取值最大的(feature, value)去分割

![exact-greedy-algorithm-for-split-finding](pic/exact-greedy-algorithm-for-split-finding.png)

为了有效率的找到最佳分裂节点，算法必须先将该特征的所有取值进行排序，之后按顺序取分裂节点计算$L_{s p l i t}$。时间复杂度是$O(N_u)$，$N_u$是这个特征不同取值的个数。

### 近似算法（Approximate Algo for Split Finding）

Exact Greedy Algorithm使用贪婪算法非常有效地找到分裂节点，但是当数据量很大时，数据不可能一次性的全部读入到内存中，或者在分布式计算中，这样不可能事先对所有值进行排序，且无法使用所有数据来计算分裂节点之后的树结构得分。为解决这个问题，近似算法被设计出来。近似算法首先按照特征取值中统计分布的一些百分位点确定一些候选分裂点，然后算法将连续的值映射到buckets中，接着汇总统计数据，并根据聚合统计数据在候选节点中找到最佳节点。

XGBoost采用的近似算法对于每个特征，只考察分位点，减少复杂度，主要有两个变体：

- Global variant：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割
- Local variant：每次分裂前将重新提出候选切分点

全局划分建议比局部划分建议需要更少的步骤，但需要更多的划分候选点才能达到局部划分建议的准确率。如下图所示：

![global-local-variant](pic/global-local-variant.jpg)

$1 / \epsilon$表示buckets数，在同等准确率的情况global比local需要更多的候选点。

作者的系统实现了贪婪算法，也可以选择全局划分和局部划分来实现近似算法。

![approximate-algorithm-for-split-finding](pic/approximate-algorithm-for-split-finding.png)

具体算法如上，这里按照三分位点举例：

![approximate-algorithm-examplee](pic/approximate-algorithm-examplee.png)

找到其中最大的信息增量的划分方法：
$$
\begin{aligned}
\text{Gain}=\text{max}\{ 
&\text{Gain}, \\
&\frac{G_1^2}{H_1+\lambda}+\frac{G_{23}^2}{H_{23}+\lambda} - \frac{G_{123}^2}{H_{123}+\lambda}-\gamma,\\
&\frac{G_{12}^2}{H_{12}+\lambda}+\frac{G_3^2}{H_3+\lambda} - \frac{G_{123}^2}{H_{123}+\lambda}-\gamma
\}
\end{aligned}
$$

然而，这种划分分位点的方法在实际中可能效果不是很好，所以XGBoost实际采用的是加权分位数的方法做近似划分算法。

### 加权分位点（Weighted Quantile Sketch）

**带权重直方图算法**

由于用暴力枚举来划分分位点的方法在实际中可能效果不是很好，为了优化该问题，XGBoost实际采用的是一种新颖的分布式加权分位点算法，该算法的优点是解决了带权重的直方图算法问题，以及有理论保证。主要用于近似算法中分位点的计算。

实际上，XGBoost不是简单地按照样本个数进行分类，而是以二阶导数值作为权重。

假设分位点为$\{s_{k1},s_{k2},...,s_{kl}\}$，假设$D_k=\{(x_{1k},h_1),(x_{2k},h_2),...,(x_{nk},h_n)\}$表示所有样本的第$k$个特征值及二阶导数。

![approximate-algorithm-second-order-weights-select-split](pic/approximate-algorithm-second-order-weights-select-split.png)

则可以定义一个排序函数如下，该Rank函数的输入为某个特征值$z$，计算的是该特征所有可取值中小于$z$的特征值的总权重占总的所有可取值的总权重和的比例，输出为一个比例值，类似于概率密度函数$f(x)$的积分$F(x)$，变化范围由0到1。
$$
\begin{aligned}
&r_k:\mathbb{R}\rightarrow [0,+\infty)\ \text{as}\\
&r_k(z)=\frac{1}{\sum_{(x,h)\in D_k}h}\sum_{(x,h)\in D_k, x<z}h
\end{aligned}
$$
该函数表示特征值$k$小于$z$的实例比例。目标就是寻找候选分裂点集$\{s_{k1}, s_{k2}, ..., s_{kl}\}$。希望得到的分位点满足如下条件：
$$
|r_k(s_{k,j})-r_k(s_{k,j+1})|<\epsilon,\ s_{k1}=\mathop{\text{min}}_{i}\ x_{ik},\ s_{kl}=\mathop{\text{max}}_{i}\ x_{ik}
$$
$s_{k1}$是特征$k$的取值中最小的值$x_{ik}$，$s_{kl}$是特征$k$的取值中最大的值$x_{ik}$，这是分位数缩略图要求**需要保留原序列中的最小值和最大值**。这里$\epsilon$是近似因子或者说是扫描步幅，按照步幅$\epsilon$挑选出特征$k$的取值候选点，组成候选点集。这意味着大概有$1/\epsilon$个分位点。

**二阶导数$h$为权重的解释**：

这里每个数据点的权重$h_i$，从图上看可能更明显一些。**为什么每个数据点都用二阶代数$h_i$作为权重进行加权分位呢？**

因为损失函数还可以写成带权重的形式：
$$
\begin{aligned}
\tilde{L}^{(t)}&=\sum_{i=1}^n\left[ g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i) \right]+\Omega(f_t)\\
&=\sum_{i=1}^n\frac{1}{2}h_i(f_t(x_i)-g_i/h_i)^2+\Omega(f_t)+\text{Constant}
\end{aligned}
$$
上式就是一个加权平方误差，权重为$h_i$，label 为$-g_i/h_i$。可以看出$h_i$有对loss加权的作用，所以可以将特征$k$的取值权重看成对应的$h_i$。

如果损失函数是square loss，即$Loss(y, \hat{y})=(y-\hat{y})^2$，则$h=2$，那么实际上是不带权。 如果损失函数是log loss，则$h=pred\cdot (1-pred)$，这是个开口朝下的一元二次函数，所以最大值在0.5。当pred在0.5附近，这个值是非常不稳定的，很容易误判，$h$作为权重则因此变大，那么直方图划分，这部分就会被切分的更细：

![approximate-algorithm-second-order-weights-select-split-2](pic/approximate-algorithm-second-order-weights-select-split-2.png)

当数据量非常大时，也需要使用quantile summary的方式来近似计算分位点。

在xgboost中，需要根据特征值以及样本的权重分布，近似计算特征值的分位点，实现近似分割算法。近似计算特征值分位点的算法称为：weighted quantile sketch，该算法满足quantile summary通常的两个操作：merge和prune。

### 结点分裂时多机并行

节点分裂的时候是按照哪个顺序来的，比如第一次分裂后有两个叶子节点，先裂哪一个？ 
答案：呃，**同一层级**的（多机）并行，确立如何分裂或者不分裂成为叶子节点。

## 稀疏感知分割（缺失值处理）

在很多现实业务数据中，训练数据$x$可能很稀疏。造成这个问题得原因可能是：

1. 存在大量缺失值
2. 太多0值
3. one-hot encoding所致

算法能够处理稀疏模式数据非常重要，XGBoost在树节点中**添加默认划分方向**的方法来解决这个问题，如下图所示：

![missing-value](pic/missing-value.jpg)

XGBoost还特别设计了针对稀疏数据的算法，假设样本的第$i$个特征缺失时，无法利用该特征对样本进行划分，系统将实例分到默认方向的叶子节点。每个分支都有两个默认方向，最佳的默认方向可以从训练数据中学习到。算法如下：

![sparsity-aware-spilit-finding](pic/sparsity-aware-spilit-finding.png)

该算法的主要思想是：**分别假设特征缺失的样本属于右子树和左子树，而且只在不缺失的样本上迭代，分别计算整体缺失样本属于右子树和左子树的增益，选择增益最大的方向为缺失数据的默认方向**。

# XGBoost的系统设计

XGBoost 的快还体现在良好的系统设计，体现在几个方面：

## 分块并行（Column Block for Parallel Learning）

在建树的过程中，最耗时是找最优的切分点，为了节省排序的时间，XGBoost将数据存在内存单元block中，同时在block采用CSC格式存放（Compressed Column format），每一列（一个属性列）均升序存放，这样，一次读入数据并排好序后，后面节点分裂时直接根据索引得到梯度信息，大大减少计算量。在精确贪心算法中，将所有数据均导入内存，算法只要在数据中线性扫描已经预排序过的特征就可以。对于近似算法，可以用多个block（Multiple blocks）分别存储不同的样本集，多个block可以并行计算。

重要的是，由于这种块结构将数据按列存储，可以同时访问所有列，那么可以对所有属性同时执行split finding算法，从而并行化split finding.

* 特征预排序，以column block的结构存于内存中
* 每个特征会存储指向样本梯度统计值的索引，方便计算一阶导和二阶导数值（instance indices）
* block的每个块结构中都采用了稀疏矩阵存储格式（Compressed Sparse Columns Format，CSC）进行存储，一个block存储一个或多个特征值
* 缺失特征值将不进行排序
* 对于列的blocks，并行的split finding算法很容实现

这种块结构存储的特征之间相互独立，方便计算机进行并行计算。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是XGBoost能够实现分布式或者多线程计算的原因。

这个结构加速split finding的过程，只需要在建树前排序一次，后面节点分裂时直接根据索引得到梯度信息。

![column-block](pic/column-block.png)

![column-block-2](pic/column-block-2.png)

## 缓存优化（Cache Aware Access）

虽然Column Block的设计可以减少节点分裂时的计算量，但其按特征大小顺序存储，相应的样本的梯度信息（一阶和二阶梯度）是分散的，造成内存的不连续访问，降低CPU cache命中率。解决办法是：为每个线程分配一个连续的缓存区，将需要的梯度信息存放在缓冲区中，这样就是实现了非连续空间到连续空间的转换，提高了算法效率。

- 缓存优化方法
  - 预取数据到buffer中（非连续->连续），再统计梯度信息
  - 适当调整块大小，也可以有助于缓存优化。

## 核外块计算（Blocks for Out-of-core Computation）

当数据量过大时无法将数据全部加载到内存中，只能先将无法加载到内存中的数据暂存到硬盘中，并分成多个block存在磁盘上，直到需要时再进行加载计算，而这种操作必然涉及到因内存与硬盘速度不同而造成的资源浪费和性能瓶颈。

为了解决这个问题，XGBoost独立一个线程专门用于从硬盘读入数据，以实现处理数据和读入数据同时进行。但是由于磁盘IO速度太慢，通常跟不上计算的速度。所以，XGBoost还采用了下面两种方法优化速度和存储：

- **块压缩（Block compression）：**对Block进行按列压缩，并在读取时进行解压。具体是将block按列压缩，对于行索引，只保存第一个索引值，然后只保存该数据与第一个索引值之差（offset），一共用16个bits来保存offset,因此，一个block一般有$2^{16}$个样本。
- **块拆分（Block sharding）：**将每个块存储到不同的磁盘中，从多个磁盘读取可以增加吞吐量。

## XGBoost的其他特性

- 行抽样（row sample）

  子采样：每轮计算可以不使用全部样本，使算法更加保守

- 列抽样（column sample）

  XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。训练的时候只用一部分特征（不考虑剩余的block块即可）

- Shrinkage（缩减），即学习速率

  XGBoost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间；将学习速率调小，迭代次数增多，有正则化作用

- 支持自定义损失函数（需二阶可导）

  ![user-define-loss-function-example](pic/user-define-loss-function-example.png)

# XGBoost总结

## XGBoost优点

1. 利用了二阶梯度来对节点进行划分，相对其他GBM来说，精度更加高。
2. 利用局部近似算法对分裂节点的贪心算法优化，取适当的$\epsilon $时，可以保持算法的性能且提高算法的运算速度。
3. 在损失函数中加入了$L_1$和$L_2$项，控制模型的复杂度，提高模型的鲁棒性。
4. 提供并行计算能力，主要是在树节点求不同的候选的分裂点的Gain Infomation（分裂后，损失函数的差值）。
5. 可以找到精确的划分条件
6. 精度更高：GBDT只用到一阶泰勒展开，而XGBoost对损失函数进行了二阶泰勒展开。XGBoost引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
7. 灵活性更强：GBDT以CART作为基分类器，XGBoost不仅支持CART还支持线性分类器，（使用线性分类器的XGBoost相当于带$L_1$和$L_2$正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题））。此外，XGBoost工具支持自定义损失函数，只需函数支持一阶和二阶求导；
8. 正则化：XGBoost在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的$L_2$范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合；
9. Shrinkage（缩减）：相当于学习速率。XGBoost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间；
10. 列抽样：XGBoost借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算；
11. 缺失值处理：XGBoost采用的稀疏感知算法极大的加快了节点分裂的速度；
12. 可以并行化操作：块结构可以很好的支持并行计算。

## XGBoost缺点

1. 每次迭代训练时需要读取整个数据集，耗时耗内存；每轮迭代时，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。
2. 使用Basic Exact Greedy Algorithm计算最佳分裂节点时需要预先将特征的取值进行排序，排序之后为了保存排序的结果，费时又费内存；需要pre-sorted，这个会耗掉很多的内存空间（2 \* data \* features）。
3. 计算分裂节点时需要遍历每一个候选节点，然后计算分裂之后的信息增益，费时；数据分割点上，由于XGBoost对不同的数据特征使用pre-sorted算法而不同特征其排序顺序是不同的，所以分裂时需要对每个特征单独做依次分割，遍历次数为 (data \* features) 来将数据分裂到左右子节点上。
4. 由于pre-sorted处理数据，在寻找特征分裂点时（level-wise），会产生大量的cache随机访问。生成决策树是level-wise级别的，也就是预先设置好树的深度之后，每一颗树都需要生长到设置的那个深度，这样有些树在某一次分裂之后效果甚至没有提升但仍然会继续划分树枝，然后再次划分….之后就是无用功了，耗时。
5. 尽管使用了局部近似计算，但是处理粒度还是太细了。
6. 计算量巨大
7. 内存占用巨大
8. 易产生过拟合
9. 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
10. 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。

## XGBoost和GradientBoost的比较

首先**XGBoost**是Gradient Boosting的一种高效系统实现，只是陈天奇写的一个工具包，本身并不是一种单一算法。XGBoost可以看作是对GradientBoost的优化。其原理还是基于GradientBoost，它的创新之处是用了**二阶导数**和**正则项**。

GBDT将树$f$类比于参数，通过$f$对负梯度进行回归，通过负梯度逐渐最小化Object目标；XGBoost版本通过使得当前Object目标最小化，构造出回归树$f$，更直接。两者都是求得$f$对历史累积$F$进行修正。

下面具体地进行比较：

* 模型

  * XGBoost支持线性分类器

    传统GBDT以CART作为基分类器，XGBoost还支持线性分类器，这个时候XGBoost相当于带$L_1$和$L_2$正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）


* 模型
  * 学习目标和方法不同

    GBDT直接拿一阶导数来作为下一棵决策树的预测值，进行学习（具体树怎么学习不负责）；XGBoost则是拿一阶和二阶导数直接作为下一棵决策树的增益score指导树的学习。

    GBDT主要对loss $L(y, F)$关于$F$求梯度，利用回归树拟合该负梯度；XGBOOST主要对loss  $L(y, F)$二阶泰勒展开，然后求解析解，**以解析解obj作为标准，贪心搜索split树是否obj更优**。

    之前的GBM模型（GBDT、GBRank、LambdaMART等）都把Loss加在的树间而未改动单棵CART内部逻辑（或者说无伤大雅懒得改），XGBoost因为正则化要考虑优化树复杂度的原因，把Loss带入到CART分裂目标和节点权重上去了（或者说把树内和树间的优化目标统一了）

- 优化算法

  - GBDT使用了square loss的一阶求导，并且没有树本身因素的正则化部分。XGBoost使用了square loss的一阶和二阶求导，使用了树叶子节点（$T$）和叶子权重（$||w||^2$）作为正则化。**这里是XGBoost最精髓的部分，它将基于样本的loss转化为了基于叶子节点的loss，即完成了参数的转变，这样才能将loss部分和正则部分都转为叶子节点T的目标方程**。

    传统GBDT在优化时只用到一阶导数信息，**XGBoost则对损失函数进行了二阶泰勒展开，同时用到了一阶和二阶导数**。顺便提一下，XGBoost工具支持自定义代价函数，只要函数可一阶和二阶求导

  - XGBoost对树的结构进行了正则化约束(regularization)，防止模型过度复杂，降低了过拟合的可能性。

    XGBoost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的$L_2$模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，防止过拟合，这也是XGBoost优于传统GBDT的一个特性

  - 节点分裂的方式不同，GBDT是用的gini系数，XGBoost是经过优化推导后的

  - 类似于学习率，学习到一棵树后，对其权重进行缩减，从而降低该棵树的作用，提升可学习空间

- 其他细节

  - XGBoost使用了二阶导数。这个类似于gradient descent和Newton method的区别。所以，XGBoost应该优化目标函数更快，也就是**需要更少的树**

  - XGBoost支持**列抽样**（colsumn subsampling）

    XGBoost借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算，这也是XGBoost异于传统GBDT的一个特性

  - XGBoost对于特征值有**缺失**的样本，XGBoost可以自动学习出它的分裂方向

  - XGBoost工具**支持并行**

    boosting不是一种串行的结构吗？怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第$t$次迭代的代价函数里包含了前面$t-1$次迭代的预测值）。XGBoost的并行是在特征粒度上的（树的粒度上是串行）。

    我们知道，**决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）**，XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能。在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么**各个特征的增益计算就可以开多线程进行**。

  - 可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以XGBoost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点

  - NULL值的特殊处理，作为一个特殊值来处理（实践中很有用）

# Xgboost使用经验总结

1. 多类别分类时，类别需要从0开始编码
2. Watchlist不会影响模型训练
3. **类别特征必须编码**，因为xgboost把特征默认都当成数值型的
4. 训练的时候，为了结果可复现，记得设置随机数种子
5. **XGBoost的特征重要性**是如何得到的？某个特征的重要性（feature score），等于它被选中为树节点分裂特征的次数的和，比如特征A在第一次迭代中（即第一棵树）被选中了1次去分裂树节点，在第二次迭代被选中2次…..那么最终特征A的feature score就是 1+2+….

# 参考资料

* [GBDT算法原理与系统设计简介](http://wepon.me/files/gbdt.pdf)

本文主要参考这篇文档。

* [xgboost的原理没你想像的那么难](https://www.jianshu.com/p/7467e616f227)
* [gitlinux上的博客：XGBoost](http://gitlinux.net/2018-10-29-xgboost/)

本文参考了这篇文档。

* [xgboost导读和实战](https://wenku.baidu.com/view/44778c9c312b3169a551a460.html)

“损失函数求导得最优值”参考了这篇文档。

* [通俗、有逻辑的写一篇说下Xgboost的原理](https://blog.csdn.net/github_38414650/article/details/76061893)

"优化算法"参考了这篇文档。

* [为啥Xgboost比GradientBoost好那么多？](http://sofasofa.io/forum_main_post.php?postid=1000331)

"XGBoost和GradientBoost的比较"参考这篇文档。

* [XGBoost浅入浅出](http://wepon.me/2016/05/07/XGBoost%E6%B5%85%E5%85%A5%E6%B5%85%E5%87%BA/)

“Xgboost使用经验总结”参考此博客。

* [XGboost: A Scalable Tree Boosting System论文及源码导读](http://mlnote.com/2016/10/05/a-guide-to-xgboost-A-Scalable-Tree-Boosting-System/)

这篇论文适合看懂原理后直接照着这个文章推公式。

* [XGBoost解读(2)--近似分割算法](https://yxzf.github.io/2017/04/xgboost-v2/)

“加权分位点（Weighted Quantile Sketch）”参考了此博客。

* [机器学习算法中 GBDT 和 XGBOOST 的区别有哪些？](https://www.zhihu.com/question/41354392/answer/124274741)

介绍了xgboost的单机多线程和分布式的代码架构。

===

[30分钟看懂xgboost的基本原理](https://mp.weixin.qq.com/s/PSs0tdwWCm_6ajD9kQOHMw)

[xgboost 实战以及源代码分析](https://blog.csdn.net/u010159842/article/details/77503930?from=singlemessage)

[XGBoost缺失值引发的问题及其深度分析](https://mp.weixin.qq.com/s/hYuBHHfAGLO3Y0l5t6y94Q)

[灵魂拷问，你看过Xgboost原文吗？](https://zhuanlan.zhihu.com/p/86816771)

[GBDT、XGBoost、LightGBM 的使用及参数调优](https://zhuanlan.zhihu.com/p/33700459)

[详解《XGBoost: A Scalable Tree Boosting System》](https://zhuanlan.zhihu.com/p/89546007)

[XGBoost超详细推导](https://zhuanlan.zhihu.com/p/92837676)

[史上最详细的XGBoost实战](https://zhuanlan.zhihu.com/p/31182879)

[XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

[左手论文 右手代码 深入理解网红算法XGBoost](https://zhuanlan.zhihu.com/p/91817667)

[从Boosting到BDT再到GBDT](https://zhuanlan.zhihu.com/p/105990013)

[再从GBDT到XGBoost!](https://zhuanlan.zhihu.com/p/106129630)

[XGBoost从原理到调参](https://zhuanlan.zhihu.com/p/106432182)

