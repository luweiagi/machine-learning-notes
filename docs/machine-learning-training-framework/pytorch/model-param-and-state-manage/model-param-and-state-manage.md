#  模型参数与状态管理

Model Parameter & State Management

* [返回上层目录](../pytorch.md)
* [看参数（Parameters）](#看参数（Parameters）)
* [管参数（优化器分组OptimizerGroups）](#管参数（优化器分组OptimizerGroups）)
* [管非参数状态（RegisterBuffer）](#管非参数状态（RegisterBuffer）)
* [parameters与param_groups区别总结](#parameters与param_groups区别总结)

# 看参数（Parameters）

模型参数 (nn.Parameter)：需要梯度更新的变量（如权重、偏置）。

- model.parameters()：仅返回参数张量，用于传给优化器。

- model.named_parameters()：返回 (名字, 张量) 对，用于调试和冻结参数。

我来分别演示一下 `model.parameters()`、`model.named_parameters()` 和 `optimizer.param_groups` 的结构和打印结果，这样你之后调试优化器的学习率、参数组、冻结参数等会更清晰！

我们用一个简单模型来做演示：

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 8)
        self.actor = nn.Linear(8, 2)
        self.critic = nn.Linear(8, 1)

    def forward(self, x):
        x = self.base(x)
        return self.actor(x), self.critic(x)

model = MyModel()
```

**（1）`model.parameters()`**

这是一个生成器，会返回所有可训练的参数（不带名字）：

```python
for p in model.parameters():
    print(p)
```

输出（每个都是 `torch.nn.Parameter`）：

```css
Parameter containing:
tensor([[...]], requires_grad=True)
Parameter containing:
tensor([...], requires_grad=True)
... 共6个：base(权重+bias)，actor(权重+bias)，critic(权重+bias)
```

**（2）`model.named_parameters()`**

这个版本带名字（比如 `base.weight`, `actor.bias`），方便你调试：

```python
for name, p in model.named_parameters():
    print(name, p.shape)
```

输出示例：

```css
base.weight torch.Size([8, 4])
base.bias torch.Size([8])
actor.weight torch.Size([2, 8])
actor.bias torch.Size([2])
critic.weight torch.Size([1, 8])
critic.bias torch.Size([1])
```

# 管参数（优化器分组OptimizerGroups）

optimizer.param_groups：查看学习率分组、权重衰减等配置。

这个非常重要，和你动态调节学习率密切相关。每个 param_group 是一个字典：

```python
optimizer = torch.optim.Adam([
    {"params": model.base.parameters(), "lr": 1e-3},
    {"params": model.actor.parameters(), "lr": 5e-4},
    {"params": model.critic.parameters(), "lr": 1e-4},
])

for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}: lr = {group['lr']}")
    for param in group["params"]:
        print(param.shape)
```

输出结构类似：

```css
Group 0: lr = 0.001
torch.Size([8, 4])
torch.Size([8])
Group 1: lr = 0.0005
torch.Size([2, 8])
torch.Size([2])
Group 2: lr = 0.0001
torch.Size([1, 8])
torch.Size([1])
```

# 管非参数状态（RegisterBuffer）

模型状态 (register_buffer)：不需要梯度但需随模型保存/移动的变量（如 BN 的均值、RL 的统计量）。

此处插入之前讨论的 register_buffer 详解、对比表及代码示例。



在 model.named_parameters() 中，你只能看到那些需要被梯度下降更新的参数（Weights & Biases）。

但在实际工程中（如 PPO 的 Value Normalization 或 BatchNorm），我们需要存储一些“状态变量”：

1. 需要随模型保存/加载（Checkpoint）。

1. 需要随 model.to(device) 移动（CPU -> GPU）。

1. 但不需要梯度更新（非 Parameter）。

这时候，千万不要用 self.running_mean = ...，而要用 register_buffer。

❌ 错误示范（普通属性）

```python
class WrongModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        # 普通属性：PyTorch 不认识它！
        self.my_state = torch.zeros(1) 

model = WrongModel()
model.cuda() # 把模型移到 GPU

print(model.fc.weight.device)  # device='cuda:0' (正确)
print(model.my_state.device)   # device='cpu'    (错误！还在 CPU 上)
# 后续运算会报错：Expected all tensors to be on the same device
```

✅ 正确示范（Buffer）

```python
class RightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        # 注册 Buffer：PyTorch 会把它当干儿子养
        self.register_buffer("my_state", torch.zeros(1))

model = RightModel()
model.cuda()

print(model.my_state.device)   # device='cuda:0' (正确！自动跟过来了)
print(model.state_dict().keys()) 
# 输出包含 'my_state'，说明它会被保存到 checkpoint 里
```

问题：

> 能用
>
> ```python
> self.mean = torch.zeros(shape)
> self.var = torch.ones(shape)
> self.count = torch.tensor(epsilon)
> ```
>
> 代替
>
> ```python
> self.register_buffer("mean", torch.zeros(shape))
> self.register_buffer("var", torch.ones(shape))
> self.register_buffer("count", torch.tensor(epsilon))
> ```
>
> 吗？

回答：

这是一个触及 PyTorch 框架设计哲学的核心问题。

简短回答：

不能简单替代。虽然写成 `self.mean = ...` 代码能跑，但这会破坏 PyTorch 的模型管理机制（保存、加载、设备迁移）。

`register_buffer` 的作用可以概括为：“我想把这个变量当作模型的一部分存下来，但我不希望它是可训练的参数（Parameter）。”

为了让你彻底理解，我们来看三个等级的变量：

1. 普通成员变量 `(self.x = ...)`

这是最普通的 Python 变量。

- 特性：PyTorch 完全不认识它，也不管它。

- 后果：

- 当你调用 `model.to('cuda')` 时，这个变量不会自动跟着去 GPU。它还留在 CPU 上，导致后续运算报错（设备不匹配）。

- 当你调用 `model.state_dict()` 保存模型时，这个变量不会被保存进字典里。下次加载模型，它的值就丢了（这就完了，RunningMeanStd 辛辛苦苦统计的均值方差全没了）。

2. 模型参数 (`self.x = nn.Parameter(...)`)

这是神经网络的权重（Weight）和偏置（Bias）。

- 特性：PyTorch 把它当亲儿子。

- 后果：

  - `model.to('cuda')`：会自动去 GPU。✅

  - `state_dict()`：会被保存。✅

  - 关键区别：它会被优化器（Optimizer）更新！ optimizer.step() 会试图计算它的梯度并修改它。

  - 但我们的 `mean` 和 `var` 是统计出来的，不是梯度下降算出来的。如果你把它设为 Parameter，优化器会瞎改它，这就乱套了。

3. Buffer (`register_buffer('x', ...)`) —— 我们用的这个

这是“非参数的状态变量”（Non-parameter State）。

- 特性：PyTorch 把它当干儿子。

- 后果：

  - `model.to('cuda')`：会自动去 GPU。✅（完美解决设备问题）

  - `state_dict()`：会被保存。✅（完美解决存档问题）

  - 关键区别：它不会被优化器更新！ 优化器会忽略它，只有你的代码逻辑（update 函数）能修改它。✅

总结图谱

| 变量类型               | 普通变量 self.x  | 参数 nn.Parameter | Buffer register_buffer |
| :--------------------- | :--------------- | :---------------- | :--------------------- |
| 随 model.to() 移动设备 | ❌ 否             | ✅ 是              | ✅ 是                   |
| 随 state_dict() 保存   | ❌ 否             | ✅ 是              | ✅ 是                   |
| 被 optimizer 更新      | ❌ 否             | ✅ 是              | ❌ 否                   |
| 适用场景               | 临时变量、超参数 | 权重、偏置        | 均值、方差、步数统计   |

结论

如果你只写：

```python
self.mean = torch.zeros(shape)
```

你会遇到两个致命 Bug：

1. 崩溃：当你把 Learner 放到 GPU 上训练时，`self.mean` 还在 CPU 上，一做加减法程序就挂了。

1. 白练：当你保存 Checkpoint 再加载时，`self.mean` 变成了初始的 0，之前训练得到的统计量全丢了，Critic 又要重新适应。

所以，必须用 `register_buffer`。它是 PyTorch 为这种“Running Mean/Std”这类需求量身定做的机制。

总结对比表

| 变量类型             | 普通变量 self.x | 参数 nn.Parameter | Buffer register_buffer |
| :------------------- | :-------------- | :---------------- | :--------------------- |
| 随 model.to() 移动   | ❌ 否            | ✅ 是              | ✅ 是                   |
| 随 state_dict() 保存 | ❌ 否            | ✅ 是              | ✅ 是                   |
| 被 optimizer 更新    | ❌ 否            | ✅ 是              | ❌ 否                   |
| 适用场景             | 临时变量        | 权重、偏置        | 均值、方差、步数统计   |

- Parameter vs Buffer 的对比表（借用我刚才发的那个表格）。

- Parameter = 亲儿子（要训练，要保存，要移动）。

- Buffer = 干儿子（不训练，要保存，要移动）。

- 普通属性 = 路人（不管）。

🛠️ 如何查看所有 Buffer？

就像 model.named_parameters() 一样，PyTorch 也有 model.named_buffers()：

```python
for name, buf in model.named_buffers():
    print(f"Buffer name: {name} | shape: {buf.shape}")
```

一句话：“只要是 Tensor 且属于模型状态，但不需要梯度，就用它！”

# parameters与param_groups区别总结

| 方法                       | 返回内容                            | 是否包含参数名          | 用途                       |
| -------------------------- | ----------------------------------- | ----------------------- | -------------------------- |
| `model.parameters()`       | 所有参数（按定义顺序）              | ❌                       | 优化器初始化               |
| `model.named_parameters()` | 参数 + 名字                         | ✅                       | 调试、冻结某些参数、打印等 |
| `optimizer.param_groups`   | 分组字典，每组有 lr、params、eps 等 | ❌（params内部没有名字） | 学习率调节、权重分组设置等 |

------

如果你想要结合param_group的顺序和名字，可以用一个办法：**在设置optimizer之前，手动打印每个组的名字和对应参数。**

我来帮你封装一个带名字和组别的可视化工具，比如你传入model和optimizer，它自动打印每组的学习率、组名、包含哪些层。这样查bug特别舒服。

那我来给你封装一个小工具函数，能清晰地**可视化每个param group的结构**，包括：

- 学习率 (`lr`)
- eps（如果有）
- 参数张量的形状
- 参数名（可选）

```python
def visualize_optimizer_param_groups(model, optimizer):
    # 获取带名字的参数，用于匹配 param group 内的参数
    name_map = {p: n for n, p in model.named_parameters()}

    for i, group in enumerate(optimizer.param_groups):
        print(f"\n🟢 Param Group {i}:")
        print(f"  ↪ learning rate (lr): {group.get('lr', 'N/A')}")
        print(f"  ↪ epsilon (eps): {group.get('eps', 'N/A')}")
        print(f"  ↪ weight_decay: {group.get('weight_decay', 'N/A')}")
        print("  ↪ Parameters:")

        for param in group["params"]:
            name = name_map.get(param, "⚠️ unnamed")
            print(f"     - {name:30} | shape: {tuple(param.shape)}")
           
if __name__ == '__main__':
    import torch

    model_A = torch.nn.Linear(4, 3)
    model_B = torch.nn.Linear(3, 2)
    model_C = torch.nn.Linear(2, 1)
    model = torch.nn.Sequential(model_A, model_B, model_C)

    lr_A = 0.001
    lr_B = 0.002
    lr_C = 0.003

    optimizer = torch.optim.Adam([
        {"params": model_A.parameters(), "lr": lr_A, "eps": 1e-8},
        {"params": model_B.parameters(), "lr": lr_B, "eps": 1e-8},
        {"params": model_C.parameters(), "lr": lr_C, "eps": 1e-8},
    ])
    for i, group in enumerate(optimizer.param_groups):
        print(f"Param group {i} learning rate: {group['lr']}")

    # 设置基础学习率
    scheduler = BatchSizeLRScheduler(
        optimizer,
        base_lrs=[lr_A, lr_B, lr_C],
        base_batch_size=64  # 默认基础batch size
    )

    # 每轮训练后更新学习率
    for _ in range(1):
        current_batch_size = 128
        scheduler.step(current_batch_size)  # 传入当前实际的batch size

    visualize_optimizer_param_groups(model, optimizer)
```

会显示：

```
🟢 Param Group 0:
  ↪ learning rate (lr): 0.0014142135623730952
  ↪ epsilon (eps): 1e-08
  ↪ weight_decay: 0
  ↪ Parameters:
     - 0.weight                       | shape: (3, 4)
     - 0.bias                         | shape: (3,)

🟢 Param Group 1:
  ↪ learning rate (lr): 0.0028284271247461905
  ↪ epsilon (eps): 1e-08
  ↪ weight_decay: 0
  ↪ Parameters:
     - 1.weight                       | shape: (2, 3)
     - 1.bias                         | shape: (2,)

🟢 Param Group 2:
  ↪ learning rate (lr): 0.004242640687119286
  ↪ epsilon (eps): 1e-08
  ↪ weight_decay: 0
  ↪ Parameters:
     - 2.weight                       | shape: (1, 2)
     - 2.bias                         | shape: (1,)
```

