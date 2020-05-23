# 支持向量机

- [返回顶层目录](../../README.md)

- [线性可分支持向量机与硬间隔最大化](linear-separable-svm/linear-separable-svm.md)
- [线性支持向量机与软间隔最大化](linear-svm/linear-svm.md)
- [非线性支持向量机与核函数](nonlinear-svm-and-kernel-function/nonlinear-svm-and-kernel-function.md)
- [序列最小最优化算法SMO](smo/smo.md)
- [SVM总结](svm-summary/svm-README.md)



支持向量机（support vector machines，SVM）是一种二类分类模型。它的基本模型是定义**在特征空间上的间隔最大的线性分类器**，间隔最大使它有别于感知机；支持向量机还包括核技巧，这使它成为实质上的非线性分类器。支持向量机的**学习策略就是间隔最大化**，可形式化为一个求解凸二次规划（convex quadratic programming）的问题，也等价于正则化的合页损失函数的最小化问题。支持向量机的学习算法是求解凸二次规划的最优化算法。

支持向量机学习方法包含构建由简至繁的模型：线性可分支持向量机（ linear support vector machine in linearly separable case )、线性支持向量机（ linear support vector machine)及非线性支持向量机（non-linear support vector machine)。简单模型是复杂模型的基础，也是复杂模型的特殊情况。当训练数据线性可分时，通过硬间隔最大化（ hard margin maximization)，学习一个线性的分类器，即线性可分支持向量机，又称为硬间隔支持向量机；当训练数据近似线性可分时，通过软间隔最大化（ soft margin maximization)，也学习一个线性的分类器，即线性支持向量机，又称为软间隔支持向量机；当训练数据线性不可分时，通过使用核技巧（kemel trick）及软间隔最大化，学习非线性支持向量机。

当输入空间为欧氏空间或离散集合、特征空间为希尔伯特空间时，核函数（kernel function）表示将输入从输入空间映射到特征空间得到的特征向量之间的内积。**通过使用核函数可以学习非线性支持向量机，等价于隐式地在高维的特征空间中学习线性支持向量机**。这样的方法称为核技巧。核方法（ kernel method)是比支持向量机更为一般的机器学习方法。

Cortes与Vapnik提出线性支持向童机，Boser、Guyon与Vapnik又引入核技巧，提出非线性支持向量机。

本章按照上述思路介绍3类支持向量机、核函数及一种快速学习算法——序列最小最优化算法（SMO)。

# 参考资料

- 《统计学习方法》李航

本章的结构和大部分内容均参考此书对应章节。

------

以下待仔细研究：

- [支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)

待看

- [支持向量机SVM（一）](https://www.cnblogs.com/jerrylead/archive/2011/03/13/1982639.html)

这份材料从前几节讲的logistic回归出发，引出了SVM，既揭示了模型间的联系，也让人觉得过渡更自然。

- [攀登传统机器学习的珠峰-SVM (上)](https://zhuanlan.zhihu.com/p/36332083)

111

- [理解SVM的核函数和参数](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484495&idx=1&sn=4f3a6ce21cdd1a048e402ed05c9ead91&chksm=fdb699d8cac110ce53f4fc5e417e107f839059cb76d3cbf640c6f56620f90f8fb4e7f6ee02f9&mpshare=1&scene=1&srcid=0522xo5euTGK36CZeLB03YGi#rd)

111

- [规则化和不可分情况处理（Regularization and the non-separable case）](https://www.cnblogs.com/jerrylead/archive/2011/03/18/1988415.html)

正则化可能会用到

- [【分类战车SVM】附录：用Python做SVM模型](https://mp.weixin.qq.com/s?__biz=MjM5MDEzNDAyNQ==&mid=207384849&idx=7&sn=eda3ef452c5b07cf741e8e01e813a516#rd)

这属于项目实践部分，以后有时间了再写吧
