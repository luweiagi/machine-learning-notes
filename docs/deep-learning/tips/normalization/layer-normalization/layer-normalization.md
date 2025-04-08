# Layer Normalization

- [返回上层目录](../tips.md)

对于batch normalization实际上有两种说法，一种是说BN能够解决“Internal Covariate Shift”这种问题。简单理解就是随着层数的增加，中间层的输出会发生“漂移”。另外一种说法是：BN能够解决梯度弥散。通过将输出进行适当的缩放，可以缓解梯度消失的状况。

那么NLP领域中，我们很少遇到BN，而出现了很多的LN，例如bert等模型都使用layer normalization。这是为什么呢？

# BN与LN主要区别

**主要区别在于normalization的方向不同！**

Batch Norm顾名思义是对一个batch进行操作。假设我们有10行3列 的数据，即我们的`batchsize = 10`，每一行数据有三个特征，假设这三个特征是【身高、体重、年龄】。那么BN是针对每一列（特征）进行缩放，例如算出【身高】的均值与方差，再对身高这一列的10个数据进行缩放。体重和年龄同理。这是一种“列缩放”。

而Layer Norm方向相反，它针对的是每一行进行缩放。即只看一笔数据，算出这笔所有特征的均值与方差再缩放。这是一种“行缩放”。

细心的你已经看出来，Layer Normalization对所有的特征进行缩放，这显得很没道理。我们算出一行这【身高、体重、年龄】三个特征的均值方差并对其进行缩放，事实上会因为特征的量纲不同而产生很大的影响。但是BN则没有这个影响，因为BN是对一列进行缩放，一列的量纲单位都是相同的。

那么我们为什么还要使用LN呢？因为NLP领域中，LN更为合适。

如果我们将一批文本组成一个batch，那么BN的操作方向是，对每句话的**第一个**词进行操作。但语言文本的复杂性是很高的，任何一个词都有可能放在初始位置，且词序可能并不影响我们对句子的理解。而BN是**针对每个位置**进行缩放，这**不符合NLP的规律**。

而LN则是针对一句话进行缩放的，且L**N一般用在第三维度**，如[batchsize, seq_len, dims]中的dims，一般为词向量的维度，或者是RNN的输出维度等等，这一维度各个特征的量纲应该相同。因此也不会遇到上面因为特征的量纲不同而导致的缩放问题。

如下图所示：

![bn_vs_ln](pic/bn_vs_ln.jpg)

假如我们的词向量是100维（如图是立方体的高），batchsize是64（立方体中的N）。

BN：固定每句话的第一个位置，则这个切片是 （64， 100）维的矩阵。

LN：固定一句话，则切片是（seq_len, 100）维。

但是，BN取出一条 **（1，64）**的向量（**绿色剪头方向**）并进行缩放，LN则是取出一条**（1， 100）**维（**红色箭头**）进行缩放。

# Layer-normalization代码

Layer normalization代码实现实例如下：

TensorFlow：

```python
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x): # x shape=[batch_size, seq_len, d_model]
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

```

Pytorch：

```python
class LayerNorm(nn.Module):
    def __init__(self, embed_size):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embed_size))  # [embed_size]
        self.beta = nn.Parameter(torch.zeros(embed_size))  # [embed_size]

    def forward(self, x):
        # Layer Normalization 操作
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / (var + 1e-6).sqrt()
        return self.gamma * x_norm + self.beta
```

看代码可知，对于某一条样本，

* mean（单值）是该样本所有元素的平均值
* var（单值）是该样本所有元素的方差
* gamma（向量）是该样本每个元素都分配了一个值
* beta（向量）是该样本每个元素都分配了一个值

# 问题

## layer-norm中的gamma和beta是逐元素学习

> ```python
> self.gamma = nn.Parameter(torch.ones(embed_size))
> self.beta = nn.Parameter(torch.zeros(embed_size))
> ```
>
> layer norm中的gamma和beta还是逐元素学习的啊，我还以为就一个值

你理解得很好，`gamma` 和 `beta` 其实是在 **Layer Normalization** 中逐元素学习的，并不是每个 Batch 共享一个单一的值。

**Layer Normalization 中的 `gamma` 和 `beta`：**

在 **Layer Normalization** 中，`gamma` 和 `beta` 是可学习的参数，它们用于对每个输入样本的标准化结果进行缩放（scaling）和偏移（shifting）。`gamma` 和 `beta` 的形状通常是 `[embed_size]`，其中 `embed_size` 是输入特征的维度。

**具体细节：**

- **gamma（缩放因子）**：用来缩放标准化后的结果，学习一个逐元素的缩放因子。这意味着每个特征维度都会有一个对应的 `gamma` 参数。
- **beta（偏移量）**：用来平移标准化后的结果，同样学习一个逐元素的偏移量，每个特征维度对应一个 `beta`。

**为什么 `gamma` 和 `beta` 不是单一的值？**

- **逐元素的学习**：`gamma` 和 `beta` 是逐元素学习的，是为了确保在标准化后的每个特征维度上都有独立的缩放和偏移。这使得模型能够为每个特征分配独立的学习参数，从而使模型的表达能力更强，能够学习到不同特征的不同尺度和偏移。
- **每个特征都有一个 gamma 和 beta**：因为 Layer Normalization 是对每个样本的所有特征进行标准化，所以我们希望对每个特征都有独立的缩放和偏移。在 `embed_size` 维度上，`gamma` 和 `beta` 都会有一个对应的参数。

**为什么不能只有一个值？**

如果 `gamma` 和 `beta` 只有一个值，那么它们就会在所有的特征维度上共享同一个缩放因子和偏移量，这就限制了模型在每个特征维度上进行不同的学习调整。而在实践中，**每个特征的尺度和偏移是不同的**，所以使用逐元素学习的 `gamma` 和 `beta` 更符合实际的需求，能够更好地适应不同特征的分布。

**总结：**

- **gamma 和 beta 逐元素学习**：它们分别对每个特征维度有独立的学习参数，使得每个特征能够单独进行缩放和偏移。
- 这样做是为了让 **Layer Normalization** 能够更好地处理每个特征的标准化，同时保持足够的灵活性，适应不同特征的分布。

# 参考资料

* [NLP中 batch normalization与 layer normalization](https://zhuanlan.zhihu.com/p/74516930)

本文参考了该知乎博客。

* [Transformer学习总结附TF2.0代码实现](https://blog.csdn.net/qq_43079023/article/details/103301846)

本文中的代码参考此博客。

