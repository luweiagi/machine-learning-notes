# Multi Head Self Attention机制

* [返回上层目录](../attention-mechanism.md)

多头注意力机制（Multi-Head Attention）是自注意力机制的扩展，它通过并行计算多个注意力头（attention heads）来捕获输入信息中的不同子空间的依赖关系。每个头会使用不同的权重矩阵来计算注意力分数，最后将所有头的输出拼接在一起并通过一个线性层进行映射。

下面是多头注意力机制的实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads"

        # 为每个头定义 Q, K, V 的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # 最后输出线性变换
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # 计算 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)    # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        # 将 Q, K, V 分成多个头
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)

        # 将Q, K, V的形状转换为 (batch_size * num_heads, seq_len, head_dim)
        Q = Q.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        K = K.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)
        V = V.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, seq_len, self.head_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size * num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size * num_heads, seq_len, seq_len)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # (batch_size * num_heads, seq_len, head_dim)

        # 将输出的多个头拼接在一起
        output = output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        # 通过一个线性层映射回原始的 embedding 大小
        output = self.fc_out(output)

        return output, attention_weights

# 示例用法
if __name__ == "__main__":
    batch_size = 2
    seq_len = 3
    embed_size = 4
    num_heads = 2  # 假设我们用两个头

    # 创建无序的行为序列
    behavior_sequences = torch.rand(batch_size, seq_len, embed_size)  # (batch_size, seq_len, embed_size)
    # 强制置为固定的值，仅用于调试，可以删除
    behavior_sequences.data = (torch.tensor([[
        [0.9535, 0.0033, 0.7889, 0.8760],
        [0.1234, 0.1995, 0.0506, 0.4779],
        [0.6134, 0.7662, 0.2646, 0.5671]],
        
        [[0.8491, 0.1763, 0.7975, 0.6957],
        [0.3699, 0.2550, 0.1919, 0.4196],
        [0.6227, 0.5930, 0.1368, 0.7236]]]))

    # 初始化multi-head attention层
    attention_layer = MultiHeadAttention(embed_size, num_heads)

    # 前向传播
    output, attention_weights = attention_layer(behavior_sequences)

    print("Output:", output)
    print("Attention Weights:", attention_weights)
```



# 参考资料


