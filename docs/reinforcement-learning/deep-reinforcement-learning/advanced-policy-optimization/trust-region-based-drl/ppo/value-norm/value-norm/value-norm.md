# Value Norm

- [返回上层目录](../value-norm.md)



# 常规Value-Norm的优缺点以及与PopArt的区别

## 常规Value-Norm的优缺点

常规Value-Norm很多实现是这样的流程：

```python
# 1. 网络输出真实值
v_pred = critic(obs)  # shape: (B,)

# 2. 对 target 做标准化（比如用滑动 mean/std）
target_norm = (target - mean) / std

# 3. 也对网络输出做标准化（为了计算 loss）
v_pred_norm = (v_pred - mean) / std

# 4. 用归一化的两者计算 loss
loss = MSE(v_pred_norm, target_norm)
```

✅ 这种做法的优点：

- **网络仍然输出真实值**（v_pred 是未归一化的）
- 只是训练时计算 loss 的时候，**做了一个可控范围的归一化**，避免 loss 忽大忽小
- 简单直接、不破坏值函数结构

⚠️ 潜在问题是：

在这个过程中，虽然我们“没动网络结构”，但：

- **梯度是基于归一化后的值计算的**，也就是说：
  - 反向传播时，`v_pred_norm = (v_pred - mean) / std`
  - 所以 `∂loss/∂v_pred = (1 / std) * ∂loss/∂v_pred_norm`

➡️ 这就导致了一个问题：

> 💡 **训练时的梯度 scale 是被 std 控制的，是“隐式依赖”于 mean/std 的！**

这在目标变化剧烈、跨任务或多尺度值域时，仍可能造成训练不稳定。

**PopArt 的改进点是：**

PopArt 不仅归一化 target，还通过调整最后一层的 W 和 b，让网络直接输出的是标准化值，再通过一层“自动反归一化”还原出真实值。

换句话说：

- **PopArt 改变了网络参数的表示方式**（等价地学习在另一个空间）
- **loss 是基于标准化的目标，输出却是非标准化的值**

所以：

| 方法            | 输出是否是原始 value？ | loss 使用归一化？ | 是否修正 W/b？ | 梯度 scale 是否稳定？ |
| --------------- | ---------------------- | ----------------- | -------------- | --------------------- |
| 常规 value norm | ✅ 是                   | ✅ 是              | ❌ 否           | ⚠️ 不一定，受 std 影响 |
| PopArt          | ✅ 是                   | ✅ 是              | ✅ 是           | ✅ 是，主动控制        |

总结一下：

你说得没错：

> ✅ “常规 value norm 不会影响 value 网络输出的值，它仍然输出真实值。”

但：

> ⚠️ 它仍然会让梯度 scale 依赖于 std，会影响训练的稳定性和策略同步性。

而 PopArt 做了更进一步的处理，让输出和 loss 在两个空间中都一致，**显式地解耦了梯度 scale 与 std 的关系**，所以它的训练更健壮，特别是在多任务、多尺度 reward 的环境中。

## 与PopArt的区别

PopArt 与简单的 mean/std 归一化 **本质上是两种策略**，它们的 **职责边界不同**：

| 特性                    | 普通 Mean/Std 归一化  | PopArt                                                  |
| ----------------------- | --------------------- | ------------------------------------------------------- |
| 归一化目标              | 仅作用于 value target | 同时作用于 target 和 value 网络输出（通过变换最后一层） |
| 是否需感知网络结构      | ❌ 不需要              | ✅ 必须知道最后一层                                      |
| 对 value 网络结构有依赖 | ❌ 完全独立            | ✅ 需要修改网络或注册最后一层                            |
| 是否修改权重            | ❌ 不修改              | ✅ 更新 mean/std 的同时修改权重和偏置                    |
| 实现复杂度              | 简单                  | 高（涉及数值稳定性与梯度同步）                          |

所以把这两者「强行封装」进同一个类虽然能实现，但会导致：

- 模块内部逻辑非常复杂（需要判断是否有 `linear`，还得处理各种 `if use_popart`）
- 外部使用时 API 不一致，容易出 bug
- 用户理解成本上升：不清楚何时启用 popart 模式，何时只归一化 target

# 代码实现

下面是**ValueNormalizer**的纯代码实现，支持标准的 mean/std 归一化与反归一化（用于 baseline 的值标准化）

```python
import torch
import torch.nn as nn

class ValueNormalizer(nn.Module):
    def __init__(self, epsilon=1e-5, device="cpu"):
        super().__init__()
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.epsilon = epsilon

    def update(self, targets):
        self.mean.data = targets.mean().detach()
        self.var.data = targets.var(unbiased=False).detach()

    def normalize(self, targets):
        return (targets - self.mean) / (self.var + self.epsilon).sqrt()

    def denormalize(self, values):
        return values * (self.var + self.epsilon).sqrt() + self.mean
```

