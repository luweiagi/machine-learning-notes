#  模型结构参数及显示

* [返回上层目录](../pytorch.md)

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

**（（1）`model.parameters()`**

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

**（3）`optimizer.param_groups`**

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

**（4）总结一下三者区别：**

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

