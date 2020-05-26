# 线性模型


* [返回上层目录](../machine-learning.md)
* [最大熵模型](maximum-entropy-model/maximum-entropy-model.md)
* [指数族分布与广义线性模型](exponential-family-distribution-and-generalized-linear-model/exponential-family-distribution-and-generalized-linear-model.md)


* [线性回归](linear-regression/linear-regression.md)
* [逻辑回归](logistic-regression/logistic-regression.md)

# 为什么线性模型有用

时至今日，深度学习早已成为数据科学的新宠。即便往前推10年，SVM、boosting等算法也能在准确率上完爆线性回归。 

那么，为什么我们还需要线性回归呢？

一方面，线性回归所能够模拟的关系其实远不止线性关系。线性回归中的“线性”指的是系数的线性，而通过对特征的非线性变换，以及广义线性模型的推广，输出和特征之间的函数关系可以是高度非线性的。另一方面，也是更为重要的一点，线性模型的易解释性使得它在物理学、经济学、商学等领域中占据了难以取代的地位。

# 线性模型的学习思路

本章的学习思路和背后的联系：要想明白softmax回归，需先搞清楚广义线性模型GLM，要想明白GLM又要先知道指数族分布，指数族分布又是从最大熵原理推导出来的。

也就是说，学习顺序是：最大熵模型->指数族分布->广义线性模型->softmax回归





