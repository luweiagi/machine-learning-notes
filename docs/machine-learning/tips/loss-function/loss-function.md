# 损失函数

* [返回上层目录](../tips.md)
* [回归问题的损失函数](#回归问题的损失函数)
* [分类问题的损失函数](#分类问题的损失函数)
  * [交叉熵损失](#交叉熵损失)
    * [分类问题中交叉熵优于MSE的原因](#分类问题中交叉熵优于MSE的原因)
    * [多个二分类同时计算交叉熵损失](#多个二分类同时计算交叉熵损失)
  * [Logistic loss](#Logistic loss)
    * [Logistic loss和交叉熵损失损失的等价性](#Logistic loss和交叉熵损失损失的等价性)



# 回归问题的损失函数



# 分类问题的损失函数



## 交叉熵损失



### 分类问题中交叉熵优于MSE的原因

<https://zhuanlan.zhihu.com/p/84431551>



### 多个二分类同时计算交叉熵损失

通常的负采样中，只有一个正样本和多个负样本，但在某些时候，会有多个正样本和多个负样本同时存在，比如：
$$
[1, 1, 0, 0, 0, 0]
$$
那该怎么计算交叉熵受损失呢？

假设有m个正样本和n个负样本同时存在，则其交叉熵损失计算为
$$
\begin{aligned}
\text{entropy_loss}&=-\sum_{i=1}^{m}\frac{1}{m}\text{log}\frac{\text{exp}(s_i)}{\sum_{j=1}^{m+n}s_j}\\
&=-\sum_{i=1}^{m}\left(s_i-\text{log}\sum_{j=1}^{m+n}s_j\right)\ \text{去掉常数项1/m}\\
&=-\sum_{i=1}^{m}s_i+m\ \text{log}\sum_{j=1}^{m+n}s_j
\end{aligned}
$$
tensorflow代码为

```python
def cmp_cross_entropy_loss(logits, labels, pos_num):
    logits_exp_sum = tf.reduce_sum(tf.exp(logits), axis=1)
    logits_sum = tf.reduce_sum(tf.multiply(logits, labels), axis=1) 
    cross_entropy_loss_ = -1.0 * (logits_sum - tf.dtypes.cast(pos_num, tf.float32) * tf.math.log(logits_exp_sum))
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_) 
    return cross_entropy_loss, cross_entropy_loss_
```

### tf.nn.sparse_softmax_cross_entropy_with_logits函数

[tf.nn.sparse_softmax_cross_entropy_with_logits 函数简介](https://blog.csdn.net/wdh315172/article/details/106140608/)

测试交叉熵损失：

```python
import math
logit = [3, -3, -3]
softmax = math.exp(logit[0]) / sum([math.exp(logit[i]) for i in range(3)])
cross_entropy = - math.log(softmax)
print(cross_entropy)
```



## Logistic loss

有时也叫负二项对数似然损失函数（negative binomial log-likelihood）

### Logistic loss和交叉熵损失损失的等价性

对于解决分类问题的FM模型，

当标签为[1, 0]时，其损失函数为交叉熵损失：
$$
Loss=y\ \text{log} \hat{y}+(1-y)\text{log}(1-\hat{y})
$$
当标签为[1, -1]时，其损失函数为
$$
Loss=\text{log}\left(1+\text{exp}(-yf(x))\right)
$$
其中，f(x)是$w\cdot x$，不是$\hat{y}$。

这两种损失函数其实是完全等价的。

（1）当为正样本时，损失为

- 标签为[1, 0]
  $
  Loss=-y\text{log}(\hat{y})=-\text{log}\frac{1}{1+\text{exp}(-wx)}=\text{log}(1+\text{exp}(-wx)
  $

- 标签为[1, -1]
  $
  Loss=\text{log}\left(1+\text{exp}(-yf(x))\right)=\text{log}\left(1+\text{exp}(-wx)\right)
  $


（2）当为负样本时，损失为

- 标签为[1, 0]
  $
  \begin{aligned}
  Loss&=-(1-y)\text{log}(1-\hat{y})=-\text{log}(1-\frac{1}{1+\text{exp}(-wx)})\\
  &=\text{log}(1+\text{exp}(wx))
  \end{aligned}
  $

- 标签为[1, -1]
  $
  Loss=\text{log}\left(1+\text{exp}(-yf(x))\right)=\text{log}\left(1+\text{exp}(wx)\right)
  $


可见，两种损失函数的值完全一样。























# 参考文献

* [常见回归和分类损失函数比较](https://zhuanlan.zhihu.com/p/36431289)

本文参考了此博客。

* [MSE vs 交叉熵](https://zhuanlan.zhihu.com/p/84431551)

"分类问题中交叉熵优于MSE的原因"参考了此博客。

* [Notes on Logistic Loss Function](http://www.hongliangjie.com/wp-content/uploads/2011/10/logistic.pdf)
* [Logistic Loss函数、Logistics回归与极大似然估计](https://www.zybuluo.com/frank-shaw/note/143260)
* [Logistic loss函数](https://buracagyang.github.io/2019/05/29/logistic-loss-function/)

"Logistic loss和交叉熵损失损失的等价性"参考了此博客。
