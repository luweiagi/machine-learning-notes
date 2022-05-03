# Welfor算法：迭代更新方差

* [返回上层目录](../variance-and-covariance.md)
* [问题引入](#问题引入)
* [更新均值](#更新均值)
* [更新方差](#更新方差)

# 问题引入

这是一种用于迭代更新方差的简洁算法。

问题设定非常简单：想要计算长度为$N$的数列的均值和方差。即
$$
\begin{aligned}
\mu_N=&\frac{1}{N}\sum_{i=1}^Nx_i,\\
\sigma_N^2=&\frac{1}{N}\sum_{i=1}^N(x_i-\mu_N)^2
\end{aligned}
$$
这里我们把方差$\sigma_N^2$看做是总体方差，即其分母为$N$，而不是样本方差的分母$N-1$。上式中每个等式的计算复杂度均为$O(N)$，这是因为我们必须使用数列中的每一个元素。

需要解决的问题：现在将新元素$x_{N+1}$加入该数列，并想要更新长度为$N+1$的新数列的均值和方差。

本能的想法是，重新使用上述计算复杂度为$O(N)$的公式计算。但是，如果我们已知$N,\mu_N,\sigma_N^2$，是否有更好的方法呢？因为对于很大的$N$，为了加进一个新的元素值，就又要花费很大力气，代入所有的元素重新算一遍，这好像有点不值得啊。

答案是：有更好的方法。

# 更新均值

我们来看加入新元素$x_{N+1}$后的新的均值$\mu_{N+1}$，非常直接地：
$$
\begin{aligned}
\mu_{N+1}=&\frac{1}{N+1}\sum_{i=1}^{N+1}x_i\\
(N+1)\mu_{N+1}=&\sum_{i=1}^{N+1}x_i\\
=&\sum_{i=1}^{N}x_i+x_{N+1}\\
=&N\mu_{N}+x_{N+1}\\
u_{N+1}=&\frac{N}{N+1}\mu_N+\frac{1}{N+1}x_{N+1}\\
=&\frac{N+1}{N+1}\mu_N+\frac{1}{N+1}x_{N+1}-\frac{1}{N+1}\mu_N\\
=&\mu_N+\frac{1}{N+1}(x_{N+1}-\mu_N)\\
\end{aligned}
$$
这样做的话，就无需遍历一遍整个数列了，而是仅需做一点简单的计算就能更新均值了，其计算复杂度为$O(1)$。

# 更新方差

计算$\sigma_{N+1}^2$会稍微更复杂一点。首先回顾下计算方差的公式之一：
$$
\begin{aligned}
\sigma^2_N=\mathbb{E}(x^2)-\left[\mathbb{E}(x)\right]^2=\frac{1}{N}\sum_{i=1}^Nx_i^2-\mu_N^2
\end{aligned}
$$
由此可得两个通用形式：
$$
\begin{aligned}
N\sigma^2_N&=\sum_{i=1}^Nx_i^2-N\mu_N^2,\\
(N+1)\sigma^2_{N+1}&=\sum_{i=1}^{N+1}x_i^2-(N+1)\mu_{N+1}^2\\
&=\sum_{i=1}^Nx_i^2+x_{N+1}^2-(N+1)\mu_{N+1}^2
\end{aligned}
$$
注意，在上式中，没必要计算求和项，而是可以从已知的$N$，$\mu_N$，$\sigma_N^2$，即$N(\sigma_N^2,\mu_N)$中推出求和项，因此，上式等号右边的三项的时间复杂度为$O(1)$，从而，$\sigma_{N+1}^2$也能以$O(1)$的时间复杂度得到。

上述计算$\mu_{N+1}$和$\sigma_{N+1}^2$的公式，就是Welford算法的精髓了。但是，在实际使用时，不要直接使用上式（会有数值稳定性问题），而是要使用更加紧凑，类似循环的形式来表示上式，如下式所示：
$$
\begin{aligned}
&(N+1)\sigma^2_{N+1}\\
=&\sum_{i=1}^Nx_i^2+x_{N+1}^2-(N+1)\mu^2_{N+1}\\
&\text{by}\ \sigma^2_N=\frac{1}{N}\sum_{i=1}^Nx_i^2-\mu_N^2\Rightarrow \sum_{i=1}^Nx_i^2=N\sigma^2_N+N\mu_N^2\\
=&\left(N\sigma^2_N+N\mu_N^2\right)+x_{N+1}^2-(N+1)\mu_{N+1}^2\\
&\text{by}\ (N+1)\mu_{N+1}=N\mu_N+x_{N+1}
\Rightarrow\mu_{N+1}^2=\frac{(N\mu_N+x_{N+1})^2}{(N+1)^2}\\
=&N\sigma^2_N+N\mu_N^2+x_{N+1}^2-\frac{1}{(N+1)}(N\mu_N+x_{N+1})^2\\
=&N\sigma^2_N+N\mu_N^2+x_{N+1}^2-\frac{1}{(N+1)}(N^2\mu_N^2+2N\mu_Nx_{N+1}+x_{N+1}^2)\\
\end{aligned}
$$
给上式两边同乘以$N+1$：
$$
\begin{aligned}
&(N+1)^2\sigma^2_{N+1}\\
=&N(N+1)\sigma^2_N+N(N+1)\mu_N^2+(N+1)x_{N+1}^2-(N^2\mu_N^2+2N\mu_Nx_{N+1}+x_{N+1}^2)\\
=&N(N+1)\sigma^2_N+N^2\mu_N^2+N\mu_N^2+Nx_{N+1}^2+x_{N+1}^2-N^2\mu_N^2-2N\mu_Nx_{N+1}-x_{N+1}^2\\
=&N(N+1)\sigma^2_N+N\mu_N^2+Nx_{N+1}^2-2N\mu_Nx_{N+1}\\
=&N(N+1)\sigma^2_N+N(\mu_N^2-2\mu_Nx_{N+1}+x_{N+1}^2)\\
=&N(N+1)\sigma^2_N+N(\mu_N-x_{N+1})^2\\
\end{aligned}
$$
终于，我们可以用一个干净，循环的关系来表示$\sigma_{N+1}^2$了：
$$
\begin{aligned}
&\sigma_{N+1}^2\\
=&\frac{N(N+1)\sigma^2_N+N(\mu_N-x_{N+1})^2}{(N+1)^2}\\
=&\frac{N(N+1)\sigma^2_N+(N+1)\sigma^2_N+N(\mu_N-x_{N+1})^2-(N+1)\sigma^2_N}{(N+1)^2}\\
=&\frac{(N+1)(N+1)\sigma^2_N+N(\mu_N-x_{N+1})^2-(N+1)\sigma^2_N}{(N+1)^2}\\
=&\sigma^2_N+\frac{N(\mu_N-x_{N+1})^2-(N+1)\sigma^2_N}{(N+1)^2}\quad \text{(A)}
\end{aligned}
$$
即：
$$
\sigma_{N+1}^2=\sigma^2_N+\frac{N(\mu_N-x_{N+1})^2-(N+1)\sigma^2_N}{(N+1)^2}\tag{A}
$$
上式已经算是相当紧凑了，但是，还可以有另外的等效表达，我们可以对上式右边的那项作进一步的处理：
$$
\begin{aligned}
&\frac{N(\mu_N-x_{N+1})^2-(N+1)\sigma^2_N}{(N+1)^2}\\
=&\frac{(x_{N+1}-\mu_N)N(x_{N+1}-\mu_N)-(N+1)\sigma^2_N}{(N+1)^2}\\
=&\frac{(x_{N+1}-\mu_N)N(N+1)(\mu_{N+1}-\mu_N)-(N+1)\sigma^2_N}{(N+1)^2}\quad \text{details as belows}\\
=&\frac{(x_{N+1}-\mu_N)N(\mu_{N+1}-\mu_N)-\sigma^2_N}{N+1}\\
=&\frac{(x_{N+1}-\mu_N)(x_{N+1}-\mu_{N+1})-\sigma^2_N}{N+1}\quad \text{details as belows}\\
\end{aligned}
$$
上式中的一些细节推导如下：
$$
\begin{aligned}
x_{N+1}=&\sum_{i=1}^{N+1}x_i-\sum_{i=1}^Nx_i\\
=&(N+1)\mu_{N+1}-N\mu_N\\
x_{N+1}-\mu_N=&(N+1)\mu_{N+1}-N\mu_N-\mu_N\\
=&(N+1)\mu_{N+1}-(N+1)\mu_N\\
=&(N+1)(\mu_{N+1}-\mu_N)\\
N(\mu_{N+1}-\mu_N)=&(N+1)\mu_{N+1}-N\mu_N-\mu_{N+1}\\
=&\sum_{i=1}^{N+1}x_i-\sum_{i=1}^Nx_i-\mu_{N+1}\\
=&x_{N+1}-\mu_{N+1}
\end{aligned}
$$
所以有
$$
\sigma_{N+1}^2=\sigma^2_N+\frac{(x_{N+1}-\mu_N)(x_{N+1}-\mu_{N+1})-\sigma^2_N}{N+1} \tag{B}
$$
上面的式(A)和式(B)是等效的。

式(A)和式(B)写在一起方便查阅：
$$
\sigma_{N+1}^2=\sigma^2_N+\frac{N(\mu_N-x_{N+1})^2-(N+1)\sigma^2_N}{(N+1)^2}\tag{A}
$$

$$
\sigma_{N+1}^2=\sigma^2_N+\frac{(x_{N+1}-\mu_N)(x_{N+1}-\mu_{N+1})-\sigma^2_N}{N+1} \tag{B}
$$



# 参考资料

* [Welford algorithm for updating variance](https://changyaochen.github.io/welford/)

本文参考此github博客。

