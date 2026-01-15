# 损失函数

* [返回上层目录](../tips.md)
* [回归问题的损失函数](#回归问题的损失函数)
  * [HuberLoss](#HuberLoss)

* [分类问题的损失函数](#分类问题的损失函数)
  * [交叉熵损失](#交叉熵损失)
    * [分类问题中交叉熵优于MSE的原因](#分类问题中交叉熵优于MSE的原因)
    * [多个二分类同时计算交叉熵损失](#多个二分类同时计算交叉熵损失)
  * [Logistic loss](#Logistic loss)
    * [Logistic loss和交叉熵损失损失的等价性](#Logistic loss和交叉熵损失损失的等价性)



# 回归问题的损失函数

## HuberLoss

从第一性原理理解一种“受控梯度”的损失函数

在机器学习中，损失函数不仅决定了“误差如何被衡量”，更决定了**梯度如何作用于模型参数**。  
Huber Loss 的设计目标，正是对梯度行为进行约束，而不是单纯追求数值误差最小。

本文将从直觉出发，逐步推导 Huber Loss 的数学形式，并解释它解决了什么问题、为什么有效。

### 从一个根本问题开始

假设我们在做一个回归任务，预测值为 $\hat{y}$，真实值为 $y$，误差为：

$$
e = y - \hat{y}
$$
一个自然的问题是：

> **当误差变大时，我们真的希望梯度无限变大吗？**

这个问题，直接决定了损失函数的选择。

### 两种极端：MSE 和 MAE 的本质缺陷

#### 均方误差（MSE）

$$
L_{\text{MSE}}(e) = e^2
$$

**直觉：**

- 小误差：惩罚很小，训练细腻
- 大误差：惩罚呈平方增长

**梯度：**
$$
\frac{dL}{d\hat{y}} = -2e
$$
这意味着：
- 误差是 10，梯度是 20
- 误差是 100，梯度是 200

**问题在于：**
> 少数异常样本可以产生巨大的梯度，主导一次参数更新。

#### 绝对误差（MAE / L1）

$$
L_{\text{MAE}}(e) = |e|
$$

**梯度：**
$$
\frac{dL}{d\hat{y}} = -\text{sign}(e)
$$
**优点：**

- 对异常值非常鲁棒
- 梯度大小恒定，不会爆炸

**致命问题：**
> 即使误差已经非常小，梯度仍然是常数，模型无法“精细收敛”。

### Huber Loss 的设计思想

Huber Loss 的核心思想可以用一句话概括：

> **小误差时像 MSE，大误差时像 MAE。**

也就是说：
- 在“正常误差范围”内，使用平方惩罚，保证收敛精度
- 在“异常误差范围”内，切换为线性惩罚，限制梯度幅度

这不是经验拼凑，而是一个明确的工程目标：  **控制梯度的最大影响力。**

### Huber Loss 的数学定义

设定一个阈值 ( $\delta > 0$ )，Huber Loss 定义为：

$$
L_\delta(e) =
\begin{cases}
\frac{1}{2} e^2 & \text{if } |e| \le \delta \\
\delta \left(|e| - \frac{1}{2}\delta\right) & \text{if } |e| > \delta
\end{cases}
$$

#### 为什么要减去$\frac{1}{2}\delta$？

这是为了保证：
- 函数在 $|e| = \delta$ 处连续
- 一阶导数在该点也连续

换句话说：  

**Huber Loss 在连接点没有“折角”或梯度突变。**

### 梯度行为：Huber Loss 的真正核心

对预测值 $\hat{y}$ 求导，可得梯度：

$$
\frac{dL_\delta}{d\hat{y}} =
\begin{cases}
-e & |e| \le \delta \\
-\delta \cdot \text{sign}(e) & |e| > \delta
\end{cases}
$$

#### 梯度直觉

- **小误差区（$|e| \le \delta$）**
  - 梯度 ∝ 误差
  - 越接近正确值，更新越温和
  - 允许模型精细调整

- **大误差区（$|e| > \delta$）**
  - 梯度大小被限制为 $\delta$
  - 不再随误差增长
  - 单个样本无法“劫持”参数更新

> **Huber Loss 的本质不是换了一种误差度量，而是人为设定了梯度上限。**

### $\delta$ 参数的真实含义

$\delta$ 并不是一个神秘的超参数，它的含义非常明确：

> **你允许“平方惩罚”作用的最大误差范围。**

- $|e| \le \delta$：这是“可信误差区”
- $|e| > \delta$：这是“异常误差区”，只允许线性惩罚

因此：
- $\delta$ 小 → 更保守，梯度更早被限制
- $\delta$ 大 → 更接近 MSE，只有极端误差才被保护

### 直觉总结（非常重要）

从第一性原理看，Huber Loss 在做三件事：

1. **承认世界并不完美**

   数据、目标、标签中不可避免存在异常值

2. **拒绝让异常值主导学习过程**
   
   通过梯度上限，限制单点样本的影响力
   
3. **在收敛阶段依然保持精细更新能力**

   不牺牲小误差下的二阶收敛特性

### PyTorch 中的 Huber Loss

```python
import torch
import torch.nn.functional as F

pred = torch.tensor([2.5, 0.0, 8.0])
target = torch.tensor([3.0, -0.5, 7.0])

loss = F.huber_loss(
    pred,
    target,
    reduction="mean",
    delta=1.0
)
```

* `delta`：误差阈值

* `reduction="mean"`：对 batch 取平均

* `smooth_l1_loss` 是 `huber_loss` 的等价实现

### 一句话结论

> **Huber Loss 的价值不在于“拟合得更准”，而在于“不会因为极端误差而失控”。**

如果你关心的是训练过程的稳定性，而不仅是最终误差，

那么 Huber Loss 不是一个技巧，而是一种理性的设计选择。

# 分类问题的损失函数

## 交叉熵损失

[八百字讲清楚——BCEWithLogitsLoss二分类损失函数]()



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



### torch.nn.BCEWithLogitsLoss

[八百字讲清楚——BCEWithLogitsLoss二分类损失函数](https://blog.csdn.net/AdamCY888/article/details/130167567)

[五分钟理解：BCELoss 和 BCEWithLogitsLoss的区别](https://cloud.tencent.com/developer/article/1660961)



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
