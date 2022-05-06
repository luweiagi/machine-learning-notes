# Gamma伽马分布

* [返回上层目录](../probability-distribution.md)



伽马分布用于预测未来事件发生前的等待时间。当某物的自然最小值为0时，它很有用。



伽玛分布一般和指数分布一起理解：

*1、从意义来看：*指数分布解决的问题是“要等到一个随机事件发生，需要经历多久时间”，伽玛分布解决的问题是“要等到n个随机事件都发生，需要经历多久时间”。

所以，伽玛分布可以看作是n个指数分布的独立随机变量的加总，即，n个Exponential(λ)random variables--->Gamma(n,λ）

作者：CC思SS

链接：https://www.zhihu.com/question/34866983/answer/191286772

来源：知乎

著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



# 参考资料

* [怎么来理解伽玛（gamma）分布？](https://www.zhihu.com/question/34866983/answer/191286772)

必看！

===

* [Dirichlet Distribution 狄利克雷分布](https://zhuanlan.zhihu.com/p/425388698)

Gamma函数：Gamma函数是阶乘的在复数域上的推广。其定义域为除非正整数外的全体复数。



**Gamma分布的特殊形式**

当形状参数α=1时，伽马分布就是参数为γ的指数分布，X~Exp（γ）

当α=n/2，β=1/2时，伽马分布就是自由度为n的卡方分布，X^2(n)



[伽玛分布matlab_神奇的伽玛函数 | 伽玛函数和伽玛分布……](https://blog.csdn.net/weixin_39821605/article/details/112784184)



[狄利克雷分布可视化](https://tadaoyamaoka.hatenablog.com/entry/2017/12/09/224900)