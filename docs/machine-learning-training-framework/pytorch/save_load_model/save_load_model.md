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

# 加载模型参数时devece的选择

```python
model.load_state_dict(torch.load(path, map_location=device))
```

必须要加map_location=device吗，不加是不是就错了？比如我本来是在gpu上训练，然后训练中断，然后要重新加载模型参数

这个问题问得非常关键！我来分情况详细讲：

如果你写：

```python
model.load_state_dict(torch.load(path))
```

**默认行为**是：

- `torch.load(path)` 会**尝试**把保存时的设备（比如 GPU）一起恢复出来。
- 如果你现在**正在用的设备**和**保存时的设备**一样，比如你保存时在 GPU，现在加载时也是在 GPU，就没问题。

但是如果：

- 你保存时是在 GPU (`cuda`) 上，
- 现在加载是在 CPU（比如有时候只想在 CPU 调试一下），

就会直接报错：

> `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.`
>  （大意就是：找不到 CUDA 设备，但你保存的是 CUDA 的 Tensor）

所以，总结一下：

| 场景                                     | 要不要加 `map_location`              |
| ---------------------------------------- | ------------------------------------ |
| 保存时和加载时设备一致，比如都是 GPU     | **可以不加**                         |
| 保存时在 GPU，加载时在 CPU               | **必须加 map_location='cpu'**        |
| 保存时在 GPU:0，加载想放到 GPU:1         | **最好加 map_location='cuda:1'**     |
| 想明确控制加载到哪个设备，未来代码更健壮 | **建议统一加上 map_location=device** |

回到你的问题

> 比如我本来是在gpu上训练，然后训练中断，然后要重新加载模型参数

如果你的训练中断后还是在 GPU 上继续跑，理论上不加 `map_location` 是没问题的。

但**更安全、推荐做法是加**，像这样：

```python
model.load_state_dict(torch.load(path, map_location=device))
```

这样无论是继续在 GPU、还是未来切到 CPU 都不会出任何问题，**代码更健壮、可复用性更高。**

最后一个很重要的小细节！

即使你不加 `map_location`，但是你后面有一句：

```python
model.to(device)
```

那只是把 `model` 本身转到 device 上，但是 `load_state_dict` 加载出来的 tensor **如果设备不对**，之前就已经出错了。**到 model.to(device) 这一步已经来不及了。**

所以真正该处理的是在 `torch.load` 的时候，就用 `map_location` 指定好！

✅ 总结就是：**推荐加上 map_location=device。**

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



# 更专业的保存和加载模型函数

要不要我顺便也给你列一下更专业的 `save_model` 和 `load_model` 封装函数？
 比如以后要保存 optimizer 一起存、保存 epoch、训练步数之类，比较适合正式训练用的？
 （如果想要的话，我可以一并补上，顺便帮你整理成好用的小模板~）要不要？✨

那我来给你补一个**正式版**的模型保存/加载模板，简单、干净，方便你以后一直用。
 （包括：模型、优化器、学习率调度器、当前训练轮次，通通保存起来。）

## 保存模型的函数 save_model

```python
import torch
import os

def save_model(save_dir, model, optimizer=None, scheduler=None, epoch=None, extra_info=None):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "checkpoint.pth")

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if extra_info is not None:
        checkpoint['extra_info'] = extra_info

    torch.save(checkpoint, save_path)
    print(f"✅ 模型保存到 {save_path}")
```

## 加载模型的函数load_model

```python
def load_model(load_dir, model, optimizer=None, scheduler=None, map_location='cpu'):
    load_path = os.path.join(load_dir, "checkpoint.pth")
    checkpoint = torch.load(load_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', None)
    extra_info = checkpoint.get('extra_info', None)

    print(f"✅ 模型从 {load_path} 加载成功")
    return epoch, extra_info
```

## 用法示范

保存时：

```python
save_model(
    save_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=current_epoch,
    extra_info={'update_steps': model.update_steps.item()}
)
```

加载时：

```python
epoch, extra_info = load_model(
    load_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    map_location='cuda'  # 或 'cpu'
)

# 如果需要，可以取回额外保存的信息
if extra_info is not None:
    model.update_steps = torch.tensor(extra_info['update_steps'])
```

简单解释一下

- `extra_info` 是我特意加的灵活接口，假如以后想保存一些**奇奇怪怪但重要的小东西**（比如 replay buffer 里的数据、特殊统计指标等等），可以直接塞进去保存。
- 现在 `save_model` 和 `load_model` 都支持：
  - 只保存模型
  - 模型+优化器
  - 模型+优化器+调度器
  - 模型+优化器+调度器+epoch信息+其他信息
- 加载的时候，能恢复 optimizer 和 scheduler 的状态，这样继续训练也不会断。

非常通用，不管是强化学习、监督学习、预训练、微调，一套走天下。

## 多个版本保存加载

要不要我顺便再教你怎么**保存多个版本**，比如加时间戳，每次保存一版，防止被覆盖？✨
 （这个也超实用，尤其是大项目训练时候）要的话告诉我～

既然你想要保存多个版本的模型，这里我给你提供一个加时间戳保存的版本，防止每次训练都覆盖模型文件。

### 加时间戳的模型保存

```pytorch
import torch
import os
import time

def save_model_with_timestamp(save_dir, model, optimizer=None, scheduler=None, epoch=None, extra_info=None):
    # 根据当前时间生成唯一的文件夹名
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_path = os.path.join(save_dir, f"checkpoint_{timestamp}.pth")

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if extra_info is not None:
        checkpoint['extra_info'] = extra_info

    torch.save(checkpoint, save_path)
    print(f"✅ 模型保存到 {save_path}")
```

### 加载时选择最近的模型

你可以加载目录下最新的模型文件，这样就不需要手动选文件名了。

```
import glob

def load_latest_model(save_dir, model, optimizer=None, scheduler=None, map_location='cpu'):
    # 获取目录下所有的 checkpoint 文件
    checkpoint_paths = glob.glob(os.path.join(save_dir, 'checkpoint_*.pth'))

    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"没有找到任何模型文件在 {save_dir} 目录下")

    # 按照文件名排序，选择最新的模型
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)

    checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', None)
    extra_info = checkpoint.get('extra_info', None)

    print(f"✅ 最新模型从 {latest_checkpoint_path} 加载成功")
    return epoch, extra_info
```

### 用法示范

#### 保存时（带时间戳）

```pytorch
save_model_with_timestamp(
    save_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=current_epoch,
    extra_info={'update_steps': model.update_steps.item()}
)
```

#### 加载时（加载最新的模型）

```
python


复制编辑
epoch, extra_info = load_latest_model(
    save_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    map_location='cuda'  # 或 'cpu'
)

# 恢复额外信息
if extra_info is not None:
    model.update_steps = torch.tensor(extra_info['update_steps'])
```

### 主要特点

1. **文件名自动加时间戳**：每次保存都会生成一个独特的文件名（例如 `checkpoint_20250427_153210.pth`），避免了文件覆盖。
2. **自动加载最新模型**：`load_latest_model` 会自动找到目录下最新的模型文件，节省手动查找的麻烦。
3. **灵活保存**：不仅保存了模型，还可以保存优化器、调度器、训练轮次等，完全可以恢复训练。

这样做，每次保存一个新的模型文件，模型版本逐渐增多时，你可以轻松恢复到最近的训练状态。如果以后想要回溯到之前的某个版本，也可以通过文件名查看哪个最合适。

这下你就能完全掌控模型版本管理啦！🎉

# 参考资料

* [pytorch模型的保存与加载](https://blog.csdn.net/lsb2002/article/details/131969478)

本文参考此资料。

