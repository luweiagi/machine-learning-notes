# 样本方差的无偏估计

* [返回上层目录](../variance-and-covariance.md)

总体方差（ population variance）：总体中变量离其平均值距离的平均。一组数据
$$
\begin{aligned}
x_1,x_2,x_3,...,x_N\\
\sigma^2=\frac{\sum_{i=1}^N(x_i-\mu)^2}{N}
\end{aligned}
$$
样本方差（sample variance）：样本中变量离其平均值距离的平均。一组数据
$$
\begin{aligned}
x_1,x_2,x_3,...,x_n\\
S^2=\frac{\sum_{i=1}^N(x_i-\bar{x})^2}{n-1}
\end{aligned}
$$
为什么样本方差的分母是$n-1$？两种不同角度的直观解释：

1、**样本的均值$\bar{x}$和样本服从分布的理论均值$\mu$有一定的偏差，所以，当以样本的均值$\bar{x}$来计算方差时，计算出的方差会偏小，而实际的方差本应该更大**，所以就需要把方差计算公式的分母$n$减小，到底减到多小呢？就需要严格的数学计算了。

2、是因为因为均值已经用了$n$个数的平均来做估计在求方差时，只有$n-1$个数和均值信息是不相关的。而你的第$n$个数已经可以由前$n-1$个数和均值来唯一确定，实际上没有信息量。所以在计算方差时，只除以$n-1$。

那么更严格的证明呢？请耐心的看下去。

为什么样本方差中分母是$n-1$而不是$n$？我们假设是$n$看看：
$$
\begin{aligned}
&\frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n}\\
=&\frac{1}{n}\sum_{i=1}^n\left[(x_i-\mu)+(\mu-\bar{x})\right]^2\\
=&\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2+\frac{2}{n}\sum_{i=1}^n(x_i-\mu)(\mu-\bar{x})+\frac{1}{n}\sum_{i=1}^n(\mu-\bar{x})^2\\
=&\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2+2(\bar{x}-\mu)(\mu-\bar{x})+(\mu-\bar{x})^2\\
=&\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2-2(\bar{x}-\mu)^2+(\bar{x}-\mu)^2\\
=&\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2-(\bar{x}-\mu)^2\\
\end{aligned}
$$
从上式来看，样本方差采用$n$做分母，会比总体方差更小，小的值为$(\bar{x}-\mu)^2$，这其实就是样本均值$\bar{x}$和总体分布的均值$\mu$的方差了。对应了上述直观解释的第一点。

除非$\mu=\bar{x}$，否则一定有：
$$
\frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n}
<\sigma^2
=\frac{\sum_{i=1}^n(x_i-\mu)^2}{n}
$$
样本方差计算公式里分母为$n-1$的目的是为了让方差的估计是无偏的。

无偏的估计(unbiased estimator)比有偏估计(biased estimator)更好是符合直觉的，尽管有的统计学家认为让mean square error即MSE最小才更有意义，这个问题我们不在这里探讨；**不符合直觉的是，为什么分母必须得是$n-1$而不是$n$才能使得该估计无偏**。

首先，我们假定随机变量的数学期望是已知的，然而方差未知。在这个条件下，根据方差的定义我们有
$$
\mathbb{E}\left[(x_i-\mu)^2\right]=\sigma^2, \forall i=1,...,n
$$
由此可得
$$
\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2\right]=\sigma^2
$$

因此$\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2$是方差的一个无偏估计，注意式中的分母不偏不倚正好是$n$！这个结果符合直觉，并且在数学上也是显而易见的。

现在，我们考虑随机变量$X$的数学期望是未知$\mu$的情形。这时，我们会倾向于无脑直接用样本均值$\bar{x}$替换掉上面式子中的$\mu$。

这样做有什么后果呢？后果就是，如果直接使用$\frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^2$作为估计，那么你会倾向于低估方差。

那么，在不知道随机变量真实数学期望的前提下，如何“正确”的估计方差呢？答案是把上式中的分母$n$换成$n-1$，通过这种方法把原来的偏小的估计“放大”一点点，我们就能获得对方差的正确估计了：
$$
\mathbb{E}\left[\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2\right]=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2\right]=\sigma^2
$$
至于为什么分母是$n-1$而不是$n-2$或者别的什么数，原因如下：
$$
\begin{aligned}
&\mathbb{E}(\bar{x})=\mathbb{E}\left(\frac{1}{n}\sum_{i=1}^nx_i\right)=\frac{1}{n}\sum_{i=1}^n\mathbb{E}(x_i)=\mu\\
&\mathbb{E}(\bar{x}-\mu)^2\\
=&\mathbb{E}(\bar{x}-\mathbb{E}(\bar{x}))^2=\text{Var}(\bar{x})\\
=&\text{Var}\left(\frac{1}{n}\sum_{i=1}^nx_i\right)=\frac{1}{n^2}\text{Var}\left(\sum_{i=1}^nx_i\right)\\
&\text{by}\ D(X\pm Y)=D(X)+D(Y)\pm \text{Cov}(X,Y)\\
=&\frac{1}{n^2}\sum_{i=1}^n\text{Var}(x_i)=\frac{1}{n^2}n\text{Var}(x)=\frac{1}{n}\text{Var}(x)\\
=&\frac{1}{n}\sigma^2\\
\end{aligned}
$$
所以可得样本方差$n$做分母和真实方差的关系为：
$$
\begin{aligned}
&\mathbb{E}\left(\frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n}\right)\\
=&\mathbb{E}\left(\frac{1}{n}\sum_{i=1}^n\left[(x_i-\mu)-(\bar{x}-\mu)\right]^2\right)\\
=&\mathbb{E}\left(\frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2\right)-\mathbb{E}\left((\bar{x}-\mu)^2\right)\\
=&\mathbb{E}(\sigma^2)-\frac{1}{n}\sigma^2\\
=&\sigma^2-\frac{1}{n}\sigma^2\\
=&\frac{n-1}{n}\sigma^2
\end{aligned}
$$
所以有
$$
\begin{aligned}
\sigma^2=&\frac{n}{n-1}\mathbb{E}\left(\frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n}\right)\\
=&\mathbb{E}\left(\frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n-1}\right)\\
=&\sigma^2
\end{aligned}
$$
即样本方差$S^2$为：
$$
S^2=\frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n-1}
$$
我们可以直观的看到随着样本总量$n$的增加，样本方差$s$会越来越接近总体方差。样本方差等于总体方差减样本均值的方差。如果用样本均值去估计总体均值，对总体方差的估计是有偏差的，偏差是样本均值的方差。需要做Bessel's correction去修正偏差，让偏差的期望等于0。

当$n$很大的时候，其实除以$n$和除以$n-1$的区别并不大。随着样本的增多，两者都会收敛到真实的总体方差。

# 参考资料

* [为什么样本方差的分母是n-1？为什么它又叫做无偏估计？](https://blog.csdn.net/qq_39521554/article/details/79633207)

本文参考此资料

* [贝塞尔校正（BesselsCorrection）](https://wenku.baidu.com/view/8b3162e15bf5f61fb7360b4c2e3f5727a4e92453.html)

本文也看了点这个。

===

[为什么样本方差（sample variance）的分母是 n-1？](https://www.zhihu.com/question/20099757/answer/658048814)

据说这个解释很好，但还没看。