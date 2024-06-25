# 保存加载模型文件

* [返回上层目录](../pytorch.md)
* [保存和加载整个模型](#保存和加载整个模型)
  * [保存整个模型](#保存整个模型)
  * [加载整个模型](#加载整个模型)
* [仅保存和加载模型参数（推荐）](#仅保存和加载模型参数（推荐）)
  * [保存模型参数](#保存模型参数)
  * [加载模型参数](#加载模型参数)



# 保存和加载整个模型

## 保存整个模型

```python
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 1))
print(net)

# 保存整个模型，包含模型结构和参数
torch.save(net, 'sample_model.pt')
```

输出：

```shell
Sequential(
  (0): Linear(in_features=128, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

## 加载整个模型

```python
import torch
import torch.nn as nn

# 加载整个模型，包含模型结构和参数
loaded_model = torch.load('sample_model.pt')
print(loaded_model)
```

输出：

```shell
Sequential(
  (0): Linear(in_features=128, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

# 仅保存和加载模型参数（推荐）

## 保存模型参数

```python
import torch
import torch.nn as nn
 
model = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 1))
 
# 保存整个模型
torch.save(model.state_dict(), 'sample_model.pt')
```

## 加载模型参数

```python
import torch
import torch.nn as nn
 
# 下载模型参数 并放到模型中
loaded_model = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 1))
loaded_model.load_state_dict(torch.load('sample_model.pt'))
print(loaded_model)
```

输出：

```shell
Sequential(
  (0): Linear(in_features=128, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

你会好奇如果只是单纯的`torch.load('sample_model.pt')`会是什么？

那就打印一下看看：

```python
import torch
loaded_model = torch.load('sample_model.pt')
print(loaded_model)
```



```shell
OrderedDict([('0.weight', tensor([[-0.0798,  0.0245,  0.0880,  ..., -0.0812,  0.0253, -0.0277],
        [ 0.0382,  0.0644,  0.0483,  ...,  0.0039, -0.0329, -0.0226],
        [ 0.0399,  0.0307, -0.0601,  ...,  0.0154,  0.0748, -0.0678],
        ...,
        [ 0.0279, -0.0479,  0.0126,  ...,  0.0778,  0.0654,  0.0521],
        [ 0.0613,  0.0283,  0.0219,  ..., -0.0807,  0.0087, -0.0058],
        [ 0.0824,  0.0022,  0.0803,  ..., -0.0146,  0.0389, -0.0284]])), ('0.bias', tensor([ 0.0554,  0.0607, -0.0356,  0.0661, -0.0491, -0.0182, -0.0611,  0.0212,
         0.0386,  0.0012, -0.0663, -0.0005,  0.0487, -0.0223, -0.0781,  0.0154])), ('2.weight', tensor([[-0.0585, -0.0603,  0.0626, -0.2448,  0.0612,  0.0704, -0.0561, -0.0535,
         -0.2328, -0.2104, -0.2206, -0.1715,  0.2299, -0.2423, -0.0247, -0.0756]])), ('2.bias', tensor([0.0428]))])
```

可以发现，是字典（state_dict），包括了每一层的名字和参数，但是不是整个模型结构。

> 是什么state_dict：PyTorch中的state_dict是一个python字典对象，将每个层映射到其参数Tensor。state_dict对象存储模型的可学习参数，即权重和偏差，并且可以非常容易地序列化和保存。



# 参考资料

* [pytorch模型的保存与加载](https://blog.csdn.net/lsb2002/article/details/131969478)

本文参考此资料。

