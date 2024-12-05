# Self Attention机制

* [返回上层目录](../self-attention.md)
* [Self-Attention机制的原理讲解](#Self-Attention机制的原理讲解)
  * [把单词embedding化](#把单词embedding化)
  * [Self-Attention模块结构](#Self-Attention模块结构)
    * [计算QKV](#计算QKV)
    * [计算注意力分数（Attention-Scores）](#计算注意力分数（Attention-Scores）)
    * [Softmax计算注意力权重](#Softmax计算注意力权重)
    * [应用注意力权重](#应用注意力权重)
  * [自注意力机制的示例代码](#自注意力机制的示例代码)



# Self-Attention机制的原理讲解

## 把单词embedding化

这是什么意思呢？就是把一个单词用一串向量来表征，即把自然语言数字化，变为embedding表征的数组

比如，下面两句话：

```
我爱中国
太阳变红
```

对其进行分词，变为

```
[[[我],  [爱], [中国]],
 [[太阳],[变], [红]]]
```

然后假设已经有一个词表了，即每个单词都有一个对应的向量来表示。假设这个词表为：

| 单词 | 向量                             |
| ---- | -------------------------------- |
| 我   | [0.9535, 0.0033, 0.7889, 0.8760] |
| 爱   | [0.1234, 0.1995, 0.0506, 0.4779] |
| 中国 | [0.6134, 0.7662, 0.2646, 0.5671] |
| 太阳 | [0.8491, 0.1763, 0.7975, 0.6957] |
| 变   | [0.3699, 0.2550, 0.1919, 0.4196] |
| 红   | [0.6227, 0.5930, 0.1368, 0.7236] |

那么，就可以把上述的两句话从文字变为向量组成的矩阵了，即，

从

```
[[[我],  [爱], [中国]],
 [[太阳], [变], [红]]]
```

变为

```
[[[0.9535, 0.0033, 0.7889, 0.8760], [0.1234, 0.1995, 0.0506, 0.4779], [0.6134, 0.7662, 0.2646, 0.5671],
[[0.8491, 0.1763, 0.7975, 0.6957], [0.3699, 0.2550, 0.1919, 0.4196], [0.6227, 0.5930, 0.1368, 0.7236]]]
```

这样看起来不舒服，那就变成

```
[[[0.9535, 0.0033, 0.7889, 0.8760], 
  [0.1234, 0.1995, 0.0506, 0.4779], 
  [0.6134, 0.7662, 0.2646, 0.5671]],

 [[0.8491, 0.1763, 0.7975, 0.6957], 
  [0.3699, 0.2550, 0.1919, 0.4196], 
  [0.6227, 0.5930, 0.1368, 0.7236]]]
```

放一起看就是

```
[[[0.9535, 0.0033, 0.7889, 0.8760],  # 我
  [0.1234, 0.1995, 0.0506, 0.4779],  # 爱
  [0.6134, 0.7662, 0.2646, 0.5671]], # 中国

 [[0.8491, 0.1763, 0.7975, 0.6957],  # 太阳
  [0.3699, 0.2550, 0.1919, 0.4196],  # 变
  [0.6227, 0.5930, 0.1368, 0.7236]]] # 红
```

我相信到此为止没有看不懂的地方吧。。。

上述2x3x4维的矩阵，就是我们要输入自注意力机制的值，即矩阵的维度为(batch_size, seq_len, embed_size) = (2, 3, 4)，也即

```
batch_size = 2  # 句子的数量，每次输入两句话
seq_len    = 3  # 每句话的单词长度，每个句子包含有3个单词
embed_size = 4  # 每个单词的embedding的维度，这里是4
```

变为矩阵就是
$$
x = 
\begin{bmatrix}
\begin{bmatrix}
0.9535 & 0.0033 & 0.7889 & 0.8760 \\
0.1234 & 0.1995 & 0.0506 & 0.4779  \\
0.6134 & 0.7662 & 0.2646 & 0.5671  \\
\end{bmatrix}\\
\begin{bmatrix}
0.8491 & 0.1763 & 0.7975 & 0.6957 \\
0.3699 & 0.2550 & 0.1919 & 0.4196  \\
0.6227 & 0.5930 & 0.1368 & 0.7236  \\
\end{bmatrix}
\end{bmatrix}
 = 
\begin{bmatrix}
\begin{bmatrix}
\text{我} \\
\text{爱} \\
\text{中国} \\
\end{bmatrix}\\
\begin{bmatrix}
\text{太阳} \\
\text{变} \\
\text{红} \\
\end{bmatrix}\\
\end{bmatrix}
$$
注意这是个三维矩阵，最里面的2维分别是两个3x4的矩阵。

很简单是吧。

## Self-Attention模块结构

下面来讨论自注意力机制的原理

### 计算QKV

**核心组件：**

- **Query (Q)**: 用于表示当前序列中每个元素的“问题”向量。
- **Key (K)**: 表示序列中每个元素的“键”向量。
- **Value (V)**: 提供用于更新序列中每个元素的信息。

通过线性变换生成 Q, K, V：

```python
Q = self.query(x)
K = self.key(x)
V = self.value(x)
```

这三个向量的维度均为 (batch_size, seq_len, embed_size)。

这里的query，key，value分别是个神经网络，如果只有一4层的话，那也就只是$y = W\cdot x + b$。

此时我们为了方便调试和计算，假设query，key，value的$W$和$b$都是一样的，即
$$
\begin{aligned}
W &= 
\begin{bmatrix}
-0.2665 & -0.3861 & -0.4229 & -0.1167 \\
0.0900 & 0.0633 & 0.0439 & -0.3031  \\
0.4027 & -0.3294 & 0.2227 & -0.4405  \\
0.2106 & 0.1568 & -0.2439 & -0.0705
\end{bmatrix}\\
b &=
\begin{bmatrix}
0.4796 & 0.0029 & -0.4205 & -0.1166
\end{bmatrix}
\end{aligned}
$$
然后输入的x是
$$
x = 
\begin{bmatrix}
\begin{bmatrix}
0.9535 & 0.0033 & 0.7889 & 0.8760 \\
0.1234 & 0.1995 & 0.0506 & 0.4779  \\
0.6134 & 0.7662 & 0.2646 & 0.5671  \\
\end{bmatrix}\\
\begin{bmatrix}
0.8491 & 0.1763 & 0.7975 & 0.6957 \\
0.3699 & 0.2550 & 0.1919 & 0.4196  \\
0.6227 & 0.5930 & 0.1368 & 0.7236  \\
\end{bmatrix}
\end{bmatrix}
 = 
\begin{bmatrix}
\begin{bmatrix}
\text{我} \\
\text{爱} \\
\text{中国} \\
\end{bmatrix}\\
\begin{bmatrix}
\text{太阳} \\
\text{变} \\
\text{红} \\
\end{bmatrix}\\
\end{bmatrix}
$$
则`Q = self.query(x)`为
$$
Q = x\cdot W^T+b
$$
这其实是一个不带激活函数的神经网络。

具体的，我们来挨个计算
$$
\begin{aligned}
Q &= 
\begin{bmatrix}
\begin{bmatrix}
0.9535 & 0.0033 & 0.7889 & 0.8760 \\
0.1234 & 0.1995 & 0.0506 & 0.4779  \\
0.6134 & 0.7662 & 0.2646 & 0.5671  \\
\end{bmatrix}
\cdot
\begin{bmatrix}
-0.2665 & -0.3861 & -0.4229 & -0.1167 \\
0.0900 & 0.0633 & 0.0439 & -0.3031  \\
0.4027 & -0.3294 & 0.2227 & -0.4405  \\
0.2106 & 0.1568 & -0.2439 & -0.0705
\end{bmatrix}^T
+
\begin{bmatrix}
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
\end{bmatrix}
\\
\begin{bmatrix}
0.8491 & 0.1763 & 0.7975 & 0.6957 \\
0.3699 & 0.2550 & 0.1919 & 0.4196  \\
0.6227 & 0.5930 & 0.1368 & 0.7236  \\
\end{bmatrix}
\cdot
\begin{bmatrix}
-0.2665 & -0.3861 & -0.4229 & -0.1167 \\
0.0900 & 0.0633 & 0.0439 & -0.3031  \\
0.4027 & -0.3294 & 0.2227 & -0.4405  \\
0.2106 & 0.1568 & -0.2439 & -0.0705
\end{bmatrix}^T
+
\begin{bmatrix}
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
0.4796 & 0.0029 & -0.4205 & -0.1166 \\
\end{bmatrix}
\end{bmatrix}\\
&=
\begin{bmatrix}
\begin{bmatrix}
-0.2116 & -0.1420 & -0.2478 & -0.1694 \\
0.2925 & -0.1160 & -0.6358 & -0.1054 \\
-0.1578 & -0.0537 & -0.6168 &  0.0282 \\
\end{bmatrix}\\
\begin{bmatrix}
-0.2332 & -0.0854 & -0.2655 & -0.1537 \\
0.1524 & -0.0664 & -0.4976 & -0.0751 \\
-0.0576 & -0.1168 & -0.6534 &  0.0231 \\
\end{bmatrix}
\end{bmatrix}\\
\end{aligned}
$$
不信的话，你可以用python代码验证一下：

```python
import numpy as np

x = np.array([[[0.9535, 0.0033, 0.7889, 0.8760],
         [0.1234, 0.1995, 0.0506, 0.4779],
         [0.6134, 0.7662, 0.2646, 0.5671]],

        [[0.8491, 0.1763, 0.7975, 0.6957],
         [0.3699, 0.2550, 0.1919, 0.4196],
         [0.6227, 0.5930, 0.1368, 0.7236]]])
w = np.array([[-0.2665, -0.3861, -0.4229, -0.1167],
        [ 0.0900,  0.0633,  0.0439, -0.3031],
        [ 0.4027, -0.3294,  0.2227, -0.4405],
        [ 0.2106,  0.1568, -0.2439, -0.0705]])
b = np.array([ 0.4796,  0.0029, -0.4205, -0.1166])

# Q = self.query(x)
Q = np.matmul(x, w.T) + b
# Q = x@w.T + b
print(y)

'''
[[[-0.21163689 -0.141959   -0.24780254 -0.16944617]
  [ 0.29251728 -0.1159958  -0.63576845 -0.10536365]
  [-0.15778083 -0.05366561 -0.61675123  0.02820571]]

 [[-0.23320552 -0.08537763 -0.26549325 -0.1536928 ]
  [ 0.15244432 -0.06642385 -0.49763594 -0.07510127]
  [-0.05760369 -0.11683774 -0.65335335  0.0231437 ]]]
'''
```

同理可得Q和V，因为已经假设query，key，value的$W$和$b$都是一样的了，所以，`K = self.key(x)`和`V = self.value(x)`的结果和这里算的`Q = self.query(x)`是一样的。

### 计算注意力分数（Attention-Scores）

```python
# 计算注意力分数
scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)  # (batch_size, seq_len, seq_len)
# K.transpose(-2, -1): 互换最后两个维度，从(batch_size, seq_len, embed_size)变成(batch_size, embed_size, seq_len)。
# 假设Q的形状为 (batch_size, seq_len, embed_size)，我们希望计算Q与K的相似度。
# 通过转置K为 (batch_size, embed_size, seq_len)，我们可以进行矩阵乘法，使得输出的注意力分数矩阵形状为 (batch_size, seq_len, seq_len)。
attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
```

公式：
$$
\text{Scores} = \frac{Q \cdot K^T}{\sqrt{\text{embed_size}}}
$$

- $Q$和$K$的矩阵乘法：
  - $Q$: (batch_size,seq_len,embed_size)
  - $K^T$: (batch_size,embed_size,seq_len)
  - 结果分数矩阵形状：(batch_size,seq_len,seq_len)

* 归一化分数：
  * 对 `embed_size` 的平方根进行缩放，防止分数过大导致梯度消失或爆炸。

现在是不是对数学计算已经头大了，但是我们还是要坚持下来，毕竟一开始我们总是担心不实际算一遍感觉就没有彻底掌握。
$$
\begin{aligned}
Q\cdot K^T &=
\begin{bmatrix}
\begin{bmatrix}
-0.2116 & -0.1420 & -0.2478 & -0.1694 \\
0.2925 & -0.1160 & -0.6358 & -0.1054 \\
-0.1578 & -0.0537 & -0.6168 &  0.0282 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
-0.2116 & -0.1420 & -0.2478 & -0.1694 \\
0.2925 & -0.1160 & -0.6358 & -0.1054 \\
-0.1578 & -0.0537 & -0.6168 &  0.0282 \\
\end{bmatrix}^T
\\
\begin{bmatrix}
-0.2332 & -0.0854 & -0.2655 & -0.1537 \\
0.1524 & -0.0664 & -0.4976 & -0.0751 \\
-0.0576 & -0.1168 & -0.6534 &  0.0231 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
-0.2332 & -0.0854 & -0.2655 & -0.1537 \\
0.1524 & -0.0664 & -0.4976 & -0.0751 \\
-0.0576 & -0.1168 & -0.6534 &  0.0231 \\
\end{bmatrix}^T
\end{bmatrix}\\
&=
\begin{bmatrix}
\begin{bmatrix}
0.1551 & 0.1300 & 0.1891 \\
0.1300 & 0.5143 & 0.3492 \\
0.1891 & 0.3492 & 0.4090 \\
\end{bmatrix}\\
\begin{bmatrix}
0.1558 & 0.1138 & 0.1933 \\
0.1138 & 0.2809 & 0.3224 \\
0.1933 & 0.3224 & 0.4444 \\
\end{bmatrix}
\end{bmatrix}
\end{aligned}
$$
则
$$
\frac{Q\cdot K^T}{\sqrt{4}} =
\begin{bmatrix}
\begin{bmatrix}
0.1551 & 0.1300 & 0.1891 \\
0.1300 & 0.5143 & 0.3492 \\
0.1891 & 0.3492 & 0.4090 \\
\end{bmatrix} / 2.0\\
\begin{bmatrix}
0.1558 & 0.1138 & 0.1933 \\
0.1138 & 0.2809 & 0.3224 \\
0.1933 & 0.3224 & 0.4444 \\
\end{bmatrix} / 2.0
\end{bmatrix}
=
\begin{bmatrix}
\begin{bmatrix}
0.0775 & 0.0650 & 0.0945 \\
0.0650 & 0.2572 & 0.1746 \\
0.0945 & 0.1746 & 0.2045 \\
\end{bmatrix}\\
\begin{bmatrix}
0.0779 & 0.0569 & 0.0967 \\
0.0569 & 0.1405 & 0.1612 \\
0.0967 & 0.1612 & 0.2222 \\
\end{bmatrix}
\end{bmatrix}
$$
这是注意力得分，在变成注意力的权重之前，需要先通过softmax进行归一化。

不信的话你可以用python验证一下：

```python
K = Q
V = Q

score = np.matmul(Q, np.transpose(K, axes=(0, 2, 1)))
print(score)
'''
[[[0.15506063 0.1299577  0.18906373]
  [0.1299577  0.51432441 0.34921048]
  [0.18906373 0.34921048 0.40895243]]

 [[0.1557823  0.11378176 0.19331271]
  [0.11378176 0.28093313 0.32237344]
  [0.19331271 0.32237344 0.44437547]]]
'''

score = score / np.sqrt(4)
print(score)
'''
[[[0.07753032 0.06497885 0.09453187]
  [0.06497885 0.2571622  0.17460524]
  [0.09453187 0.17460524 0.20447621]]

 [[0.07789115 0.05689088 0.09665636]
  [0.05689088 0.14046656 0.16118672]
  [0.09665636 0.16118672 0.22218774]]]
'''
```

### Softmax计算注意力权重

```python
attention_weights = F.softmax(scores, dim=-1)
```

- 对分数矩阵的最后一维（seq_len）进行Softmax操作，确保权重和为1。

$$
\text{sotfmax}(\frac{Q\cdot K^T}{\sqrt{4}}) =
\text{sotfmax}(
\begin{bmatrix}
\begin{bmatrix}
0.0775 & 0.0650 & 0.0945 \\
0.0650 & 0.2572 & 0.1746 \\
0.0945 & 0.1746 & 0.2045 \\
\end{bmatrix}\\
\begin{bmatrix}
0.0779 & 0.0569 & 0.0967 \\
0.0569 & 0.1405 & 0.1612 \\
0.0967 & 0.1612 & 0.2222 \\
\end{bmatrix}
\end{bmatrix})
=
\begin{bmatrix}
\begin{bmatrix}
0.3328 & 0.3287 & 0.3385 \\
0.3005 & 0.3642 & 0.3353 \\
0.3125 & 0.3386 & 0.3489 \\
\end{bmatrix}\\
\begin{bmatrix}
0.3335 & 0.3266 & 0.3399 \\
0.3128 & 0.3400 & 0.3472 \\
0.3125 & 0.3333 & 0.3543 \\
\end{bmatrix}
\end{bmatrix}
$$

python计算结果为

```python
# 计算最后两个维度的 softmax
softmax_score = softmax(score, axis=-1)
print(softmax_score)
'''
[[[0.33281482 0.32866361 0.33852156]
  [0.300503   0.36417739 0.33531961]
  [0.31254078 0.33859622 0.348863  ]]

 [[0.33353778 0.32660643 0.33985578]
  [0.31278383 0.34004841 0.34716777]
  [0.31246009 0.33328805 0.35425186]]]
'''
```

### 应用注意力权重

```python
# 应用注意力权重
output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_size)
```

计算加权输出：$\text{Output} = \text{Attention_Weights} \cdot V$。

test: 计算加权输出：$\text{Output} = \text{Attention_Weights} \cdot V$

- 权重矩阵 (batch_size,seq_len,seq_len)
- $V$: (batch_size,seq_len,embed_size)
- 最终输出形状：(batch_size,seq_len,embed_size)

$$
\begin{aligned}
\text{sotfmax}(\frac{Q\cdot K^T}{\sqrt{4}})\cdot V
&=
\begin{bmatrix}
\begin{bmatrix}
0.3328 & 0.3287 & 0.3385 \\
0.3005 & 0.3642 & 0.3353 \\
0.3125 & 0.3386 & 0.3489 \\
\end{bmatrix}\\
\begin{bmatrix}
0.3335 & 0.3266 & 0.3399 \\
0.3128 & 0.3400 & 0.3472 \\
0.3125 & 0.3333 & 0.3543 \\
\end{bmatrix}
\end{bmatrix}
\cdot
\begin{bmatrix}
\begin{bmatrix}
-0.2116 & -0.1420 & -0.2478 & -0.1694 \\
0.2925 & -0.1160 & -0.6358 & -0.1054 \\
-0.1578 & -0.0537 & -0.6168 &  0.0282 \\
\end{bmatrix}\\
\begin{bmatrix}
-0.2332 & -0.0854 & -0.2655 & -0.1537 \\
0.1524 & -0.0664 & -0.4976 & -0.0751 \\
-0.0576 & -0.1168 & -0.6534 & 0.0231 \\
\end{bmatrix}
\end{bmatrix}\\
&=
\begin{bmatrix}
\begin{bmatrix}
-0.0277 & -0.1035 & -0.5002 & -0.0815 \\
-0.0100 & -0.1029 & -0.5128 & -0.0798 \\
-0.0221 & -0.1024 & -0.5079 & -0.0788 \\
\end{bmatrix}\\
\begin{bmatrix}
-0.0476 & -0.0899 & -0.4731 & -0.0679 \\
-0.0411 & -0.0899 & -0.4791 & -0.0656 \\
-0.0425 & -0.0902 & -0.4803 & -0.0649 \\
\end{bmatrix}
\end{bmatrix}
\end{aligned}
$$

python计算结果为

```python
output = np.matmul(softmax_score, V)
print(f"{output=}")
'''
output=array([[[-0.02770832, -0.10353662, -0.50020991, -0.08147515],
        [-0.00997635, -0.10289728, -0.51280668, -0.07983221],
        [-0.0221438 , -0.10236566, -0.50787888, -0.07879464]],

       [[-0.0475705 , -0.0898791 , -0.47312904, -0.06792539],
        [-0.04110261, -0.08985436, -0.47908553, -0.06557594],
        [-0.04246576, -0.09020536, -0.4802638 , -0.06485452]]])
'''
```

这就是最终的自注意力机制的结果，也就是从一开始的

```
[[[0.9535, 0.0033, 0.7889, 0.8760],  # 我
  [0.1234, 0.1995, 0.0506, 0.4779],  # 爱
  [0.6134, 0.7662, 0.2646, 0.5671]], # 中国

 [[0.8491, 0.1763, 0.7975, 0.6957],  # 太阳
  [0.3699, 0.2550, 0.1919, 0.4196],  # 变
  [0.6227, 0.5930, 0.1368, 0.7236]]] # 红
```

变成了

```
[[[-0.0277, -0.1035, -0.5002, -0.0815],  # 我 = 0.5*我 + 0.4*爱 + 0.1*中国
  [-0.0100, -0.1029, -0.5128, -0.0798],  # 爱 = 0.3*我 + 0.6*爱 + 0.2*中国
  [-0.0221, -0.1024, -0.5079, -0.0788]], # 中国 = 0.1*我 + 0.2*爱 + 0.7*中国

 [[-0.0476, -0.0899, -0.4731, -0.0679],  # 太阳 = 0.5*太阳 + 0.4*变 + 0.1*红
  [-0.0411, -0.0899, -0.4791, -0.0656],  # 变 = 0.2*太阳 + 0.5*变 + 0.3*红
  [-0.0425, -0.0902, -0.4803, -0.0649]]] # 红 = 0.3*太阳 + 0.2*变 + 0.6*红
```

这里的
$$
\begin{bmatrix}
\begin{bmatrix}
0.5 & 0.4 & 0.1 \\
0.3 & 0.6 & 0.2 \\
0.1 & 0.2 & 0.7 \\
\end{bmatrix}\\
\begin{bmatrix}
0.5 & 0.4 & 0.1 \\
0.2 & 0.5 & 0.3 \\
0.3 & 0.2 & 0.6 \\
\end{bmatrix}
\end{bmatrix}
$$
就是
$$
\text{sotfmax}(\frac{Q\cdot K^T}{\sqrt{4}})
$$
，即所谓的注意力机制的权值。

也就是说，注意力机制就是向量的加权求和。

## 自注意力机制的示例代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # 赋值，方便调试学习，后期正式运行请删除
        self.query.weight.data = (torch.tensor([
            [-0.2665, -0.3861, -0.4229, -0.1167],
            [0.0900, 0.0633, 0.0439, -0.3031],
            [0.4027, -0.3294, 0.2227, -0.4405],
            [0.2106, 0.1568, -0.2439, -0.0705]]))
        self.query.bias.data = torch.tensor(
            [0.4796, 0.0029, -0.4205, -0.1166])
        self.key.weight.data = self.query.weight.data
        self.key.bias.data = self.query.bias.data
        self.value.weight.data = self.query.weight.data
        self.value.bias.data = self.query.bias.data

    def forward(self, x):
        # x: (batch_size, seq_len, embed_size)
        # 计算Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)    # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)  # (batch_size, seq_len, seq_len)
        # K.transpose(-2, -1): 互换最后两个维度，从(batch_size, seq_len, embed_size)变成(batch_size, embed_size, seq_len)。
        # 假设Q的形状为 (batch_size, seq_len, embed_size)，我们希望计算Q与K的相似度。
        # 通过转置K为 (batch_size, embed_size, seq_len)，我们可以进行矩阵乘法，使得输出的注意力分数矩阵形状为 (batch_size, seq_len, seq_len)。
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_size)
        return output, attention_weights

# 示例用法
if __name__ == "__main__":
    batch_size = 2
    seq_len = 3
    embed_size = 4

    # 创建无序的行为序列
    behavior_sequences = torch.rand(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)
    # 强制置为固定的值，仅用于调试，可以删除
    behavior_sequences.data = (torch.tensor([
        [[0.9535, 0.0033, 0.7889, 0.8760],
         [0.1234, 0.1995, 0.0506, 0.4779],
         [0.6134, 0.7662, 0.2646, 0.5671]],

        [[0.8491, 0.1763, 0.7975, 0.6957],
         [0.3699, 0.2550, 0.1919, 0.4196],
         [0.6227, 0.5930, 0.1368, 0.7236]]]))
    print("behavior_sequences = ", behavior_sequences)

    # 初始化self-attention层
    attention_layer = SelfAttention(embed_size)

    # 前向传播
    output, attention_weights = attention_layer(behavior_sequences)

    print("Output:", output)
    print("Attention Weights:", attention_weights)
```

