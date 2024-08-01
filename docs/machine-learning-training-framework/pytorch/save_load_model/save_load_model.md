# 保存加载模型文件

* [返回上层目录](../pytorch.md)
* [保存和加载整个模型](#保存和加载整个模型)
  * [保存整个模型](#保存整个模型)
  * [加载整个模型](#加载整个模型)
* [仅保存和加载模型参数（推荐）](#仅保存和加载模型参数（推荐）)
  * [保存模型参数](#保存模型参数)
  * [加载模型参数](#加载模型参数)
* [使用jit保存为TorchScript格式](#使用jit保存为TorchScript格式)

pytorch官网保存加载模型文件的教程：

[https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

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

# 使用jit保存为TorchScript格式

保存为这个格式是为了进行推理。该格式在推理时可不用定义模型的类。该格式也可以在C++中进行推理。

在[https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)简介如下：

> One common way to do inference with a trained model is to use [TorchScript](https://pytorch.org/docs/stable/jit.html), an intermediate representation of a PyTorch model that can be run in Python as well as in a high performance environment like C++. TorchScript is actually the recommended model format for scaled inference and deployment.
>
> ```
> Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
> ```

保存的代码：

```python
# Export to TorchScript
model_scripted = torch.jit.script(model)
# Save
model_scripted.save('model_scripted.pt')
```

进行推理的代码：

```python
model = torch.jit.load('model_scripted.pt')
model.eval()
```

加eval是为了在推理前设置模型中的dropout和batch norm功能为eval模式。

给一段kimi给出的例子吧：

```python
import torch

# 加载训练好的模型，这里假设模型已经保存为'model.pth'
model = torch.jit.load('model.pth')
model.eval()  # 将模型设置为评估模式

# 准备输入数据
# 这里需要根据你的模型输入进行相应的数据预处理
# 例如，如果你的模型接受的是图像数据，你需要进行图像的读取和预处理
# 示例中使用随机数据作为输入
input_data = torch.randn(1, 3, 224, 224)  # 假设输入是单张3通道224x224的图像

# 进行推理
with torch.no_grad():
    output = model(input_data)

# 处理输出
# 根据你的任务类型，这里可能是分类、回归或其他
# 例如，如果是分类任务，你可能需要找到概率最高的类别索引
_, predicted_class = torch.max(output, 1)

print(f'Predicted class index: {predicted_class.item()}')
```



# 参考资料

* [pytorch模型的保存与加载](https://blog.csdn.net/lsb2002/article/details/131969478)

本文参考此资料。

