# 自定义梯度剪裁

- [返回上层目录](../back-propagation.md)

在深度学习中，尤其是模型较深、训练不稳定或任务复杂（如强化学习）时，**梯度爆炸**是常见问题之一。它会导致：

- 参数更新异常剧烈；
- loss 值变为 `NaN`；
- 模型训练过程不收敛，甚至完全崩坏。

**梯度剪裁（Gradient Clipping）** 是一种有效的应对手段。通过限制梯度范数的上限，可以控制梯度过大导致的不稳定行为，从而：

- 稳定训练过程；
- 提高训练的鲁棒性；
- 降低对学习率的敏感性。

本模块提供了灵活的梯度剪裁机制，包括：

- 固定阈值限制（如 max-norm = 0.5）；
- 分组管理（多 param group 支持）；
- 基于分布统计的动态调整（如 90%分位数）；
- 支持训练早期/后期不同策略的策略动态切换。

适用于强化学习、多头网络、大模型等对训练稳定性要求较高的场景。

# 为什么要做梯度剪裁

**理解“我们为什么做梯度剪裁”**，就像理解“为什么系安全带”一样：你可能一开始没出问题，但一旦出了问题，你会感谢当初做了这个保护措施。

梯度剪裁的初衷，一句话：

> **防止梯度爆炸，稳定训练过程。**

注意：

**梯度剪裁并不是为了提升性能！**

它的目标是**防止崩坏、确保训练“能跑下去”**，是一个“保底机制”，不是“性能增强器”。



**什么是梯度爆炸？**

在深度网络中（尤其是RNN、Transformer、强化学习等任务），**梯度在反向传播中可能会变得非常大**，导致：

- 参数更新幅度太大，一步跳出合理解空间；
- loss变成`NaN`；
- 模型根本学不动，表现很差。

特别是在使用 **较大学习率**、**复杂模型结构**、或者**不稳定任务**（比如强化学习）时，梯度爆炸更容易发生。



**梯度剪裁怎么解决它？**

梯度剪裁不会改变梯度的方向，只是控制它的**范数（长度）**，防止：

```
|gradient| 太大 → step 太大 → loss 爆炸
```

> 所以，它像一个“安全阀”，不让梯度更新步伐“失控”。



**梯度剪裁常见用法**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

这表示：

> 如果所有梯度范数加起来超过了 0.5，那就按比例缩小，直到总范数等于 0.5。



**在哪些任务中特别有用？**

| 场景                  | 为什么容易爆炸                   |
| --------------------- | -------------------------------- |
| **RNN/LSTM**          | 时间步长长，梯度链条长           |
| **强化学习**          | 奖励信号稀疏或剧烈震荡           |
| **生成模型（GAN）**   | 损失曲面陡峭，更新不稳定         |
| **多任务 / 多头网络** | 不同子网络梯度量级不同           |
| **自定义训练 loop**   | 自己调优化器、调学习率时容易失控 |

# 源码介绍与实现

## 功能介绍

将功能封装为一个名为`GradientClipper`的类，支持：

- 不同param group的梯度分别记录
- 自动或手动设置clip值（支持percentile建议clip）
- 支持静态或动态clip模式切换
- 可视化单组梯度范数历史

你可以在每次`loss.backward()`后调用`clipper.update_and_clip()`来自动处理梯度裁剪。

## 源码

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Union, List


class GradientClipper:
    def __init__(self, optimizer, norm_type=2, max_history=500, dynamic_clip=True, percentile=90):
        self.optimizer = optimizer
        self.norm_type = norm_type
        self.history = [deque(maxlen=max_history) for _ in optimizer.param_groups]
        self.dynamic_clip = dynamic_clip
        self.percentile = percentile
        self.static_max_norms = [None] * len(optimizer.param_groups)  # 如果不启用动态 clip，可以手动设定

        # 当最开始数据量不够时，禁止使用动态分位数计算，使用固定的默认值
        self.min_window_size = 0
        self.default_clip_value = 1.0

        # 梯度剪裁下限（safety net）
        # 即使使用分位数动态估计，也设置一个最小max_norm=0.5或1.0
        # 防止有些batch小，或者学习率余弦跳跃从小突然变大而异常，导致max_norm被估得太小
        self.max_norm_min = 1.0  # 最小的max_norm
        # 梯度剪裁上限，防止持续动荡导致分位数失效
        self.max_norm_max = 10.0  # 最大的max_norm

    def update_and_clip(self):
        for i, group in enumerate(self.optimizer.param_groups):
            grad_norm = self.compute_group_grad_norm(group)
            self.history[i].append(grad_norm)

            # 动态clip：使用分位数估计max_norm
            if len(self.history[i]) < self.min_window_size:
                max_norm = self.default_clip_value
            elif self.dynamic_clip:
                max_norm = self.suggest_clip_value(i)
            else:
                max_norm = self.static_max_norms[i]

            max_norm = max(max_norm, self.max_norm_min)  # 不小于最大值的下限
            max_norm = min(max_norm, self.max_norm_max)  # 不大于最大值的上限

            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type=self.norm_type)

    def compute_group_grad_norm(self, group):
        total_norm = 0.0
        for p in group['params']:
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        return total_norm ** (1. / self.norm_type)

    def suggest_clip_value(self, group_idx, percentile=None):
        h = self.history[group_idx]
        if not h:
            return None
        perc = percentile if percentile is not None else self.percentile
        return round(np.percentile(h, perc), 4)

    def set_static_max_norms(self, max_norms: Union[List[float], float]):
        if isinstance(max_norms, (int, float)):
            max_norms = [max_norms] * len(self.optimizer.param_groups)
        assert len(max_norms) == len(self.optimizer.param_groups)
        self.static_max_norms = max_norms
        self.dynamic_clip = False

    @staticmethod
    def get_total_grad_norm(model, norm_type=2):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    def plot(self, group_idx=0):
        if not self.history[group_idx]:
            print(f"No history for group {group_idx}")
            return
        plt.plot(self.history[group_idx])
        plt.title(f"Grad Norm (Group {group_idx})")
        plt.xlabel("Step")
        plt.ylabel("Grad Norm")
        plt.grid(True)
        plt.show()
```

## 测试

```python

if __name__ == '__main__':
    from torch.utils.data import DataLoader, TensorDataset

    # 模型定义
    def create_model():
        model_A = torch.nn.Linear(4, 3)
        model_B = torch.nn.Linear(3, 2)
        model_C = torch.nn.Linear(2, 1)
        return torch.nn.Sequential(model_A, model_B, model_C)

    # 优化器初始化（多 param group）
    def create_optimizer(model):
        return torch.optim.Adam([
            {"params": model[0].parameters(), "lr": 0.001},
            {"params": model[1].parameters(), "lr": 0.002},
            {"params": model[2].parameters(), "lr": 0.003},
        ])

    # 模拟数据
    x = torch.randn(100, 4)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 创建模型、优化器、clipper
    model = create_model()
    optimizer = create_optimizer(model)

    # 初始化 GradientClipper，设定每组 clip 阈值，或使用动态 clip 策略
    clipper = GradientClipper(
        optimizer,
    )

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_x, batch_y in data_loader:
            preds = model(batch_x)
            loss = torch.nn.functional.mse_loss(preds, batch_y)
            loss.backward()

            clipper.update_and_clip()  # ✅关键调用！执行clip并更新阈值
            print(f"[Epoch {epoch + 1}] Current max_norms: {clipper.get_total_grad_norm(model)}")

            optimizer.step()
            optimizer.zero_grad()

        # 可选：每轮打印clip阈值信息
        # print(f"[Epoch {epoch+1}] Current max_norms: {clipper.get_total_grad_norm(model)}")
```

# 问题

## 固定值限制梯度值好还是分位数90%限制梯度值好

这是个很好的问题，**固定阈值剪裁**和**分位数剪裁（如90%）**各有优缺点，适用于不同的情况，下面我来详细对比一下，帮你选个最适合你的方式：

**（1）固定值剪裁（如`max_norm = 0.5`）**

**原理：**
 将所有参数组的梯度norm限制在一个固定的范围内。

**优点：**

- **可控性强：** 你明确知道最大允许的梯度norm。
- **实现简单、主流做法：** PyTorch的`clip_grad_norm_`就是基于这种方式。

**缺点：**

- **不自适应：** 在训练初期/后期，可能会限制太死或太松。
- **不同模型规模/参数组差异大时，难以统一设值。**

**（2）分位数剪裁（如90%分位）**

**原理：**
 收集每次step中各参数组或所有参数的梯度范数，根据历史统计结果设定阈值，例如每次把最“尖锐”的top10%剪掉，剩下的保留。

**优点：**

- **动态、鲁棒：** 自适应训练阶段，不容易限制太死或放太宽。
- **对抗离群值：** 能有效“消除”异常梯度爆炸。

**缺点：**

- **实现更复杂：** 你得维护一个sliding window或历史队列来统计梯度分布。
- **不够可解释：** 每次剪裁的力度会变，不利于调参和复现。

**适用建议（经验法则）**

| 场景                               | 推荐方式                                     |
| ---------------------------------- | -------------------------------------------- |
| 模型结构稳定，训练过程比较“平稳”   | **固定阈值**，搭配warm-up和decay即可         |
| 多param group，scale差异大         | **每组设置独立固定值**                       |
| 强烈想鲁棒性 / 有梯度爆炸风险      | **分位数剪裁**更灵活                         |
| 强化学习、生成模型这类训练不稳定的 | **动态 clip（含分位数或指数平均）** 效果更佳 |

## 模型参数越多，clip_grad_norm_中的梯度总和就越大？所以参数阈值就需要越大？

是的，你理解得完全正确

**为什么模型参数越多，梯度总范数通常越大？**

`torch.nn.utils.clip_grad_norm_` 会计算所有参数梯度的 **整体范数**（默认为 L2 范数）：
$$
\text{TotalNorm} = \left( \sum_i \|g_i\|^2 \right)^{1/2}
$$

- 当参数数量很多时（例如有上百万参数），哪怕每个参数的梯度都不大，但加在一起的平方和会 **快速增长**。
- 因此，**模型越大、梯度越多，clip 时的总范数也越大**。

**如果你设置了一个过小的 max_norm 会发生什么？**

- 会导致 **所有梯度被缩放到一个很小的值**，进而让模型更新几乎停滞，训练变慢或无法收敛。
- 比如对一个大型Transformer模型用`max_norm=0.5`就很可能剪得太狠了。

**经验法则：如何设置 max_norm**

| 模型规模          | 推荐的`max_norm`范围 |
| ----------------- | -------------------- |
| 小模型（<1M参数） | `0.1 ~ 1.0`          |
| 中模型（1M~20M）  | `0.5 ~ 5.0`          |
| 大模型（>20M）    | `1.0 ~ 10.0` 或更大  |

你可以用如下方法观测未裁剪前的梯度范数：

```python
def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1. / norm_type)
```



## 分位数统计作为阈值有意义吗？

因为既然梯度裁剪的作用只是为了防止梯度爆炸，那我设置一个比较大的固定值就好了呀，比如10，20，没必要卡着正常数据的95%的分位数来设置呀



这个问题问得非常好！你已经抓住了**梯度裁剪的本质目的**：是为了防止梯度爆炸，避免训练不稳定或 loss 为 nan，而不是为了对梯度进行强“规范”。

先说结论：

> **如果你只是为了防止梯度爆炸，那么设置一个相对较大的固定值（比如 10 或 20）确实是够用的，且更简单。**
>  使用分位数来动态调整阈值，更多是出于**精细控制训练过程**，或是用于更不稳定的模型结构、较敏感的优化器等场景。

两种方法的对比

| 方法              | 特点                                 | 适用场景                                             |
| ----------------- | ------------------------------------ | ---------------------------------------------------- |
| 固定阈值（如 10） | 简单直接，足以防止大多数“爆炸”       | 大多数模型、训练稳定性一般的任务                     |
| 动态分位数        | 根据历史统计自动适应，有更强的鲁棒性 | 对训练非常敏感、使用高学习率、训练初期变化剧烈的情况 |

为什么有人用分位数作为 clip 阈值？

这通常是为了 **对训练过程进行精细调节**，理由包括：

1. **自适应裁剪**：自动适应不同阶段的梯度变化（比如 warm-up 阶段 vs 收敛后期）。
2. **防止“稳态大梯度”失控**：有些模型梯度平均很大但不算爆炸，用固定阈值容易误剪。
3. **和动态学习率结合得更自然**：有时我们希望梯度控制也能随学习率变化而变化。

但代价就是：你得维护历史梯度统计、滑窗更新分位值、调节分位数（90/95）等——这些其实是工程上的额外复杂度。

推荐策略（给你一个稳妥选项）：

如果你的训练整体还算稳定，建议可以：

```python
# 简化策略
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

然后只在发现训练极度不稳定、loss 经常 nan、或者 learning rate 非常大（比如 1e-2 甚至更大）时，再考虑：

```python
# 自适应策略
clip_norm = history_percentile(past_grad_norms, p=95)
```

小结一句话：

> 如果你只是想“别炸就行”，那一个固定的大 clip 值完全足够；如果你想“剪得更优雅”，那可以尝试分位数策略。



问题：只要梯度不爆炸，剪的更优雅有什么用呢？只能减缓学习速度吧

你这句一针见血，确实说出了关键本质：

> **“只要梯度不爆炸，剪得更优雅有什么用呢？”**

回答：**大多数情况下没用，甚至可能减缓训练效率。**

“剪得更优雅”听起来很好听，其实通常只带来**更安全但更保守**的训练策略。

在以下两种场景，它才可能**有意义**：

**场景 1：训练非常不稳定时**

比如你使用的是：

- **超大模型（Transformer / Actor-Critic）**
- **极大或动态的学习率**
- **离散梯度/稀疏梯度情况**
- **梯度爆炸和梯度消失交替出现**

在这种时候你希望裁剪“恰到好处”而不是粗暴限制，防止：

- 梯度爆炸 → loss 变 nan
- 梯度过小 → 模型学不动

此时，动态分位数策略会 **比静态固定阈值更稳健**，因为它跟着训练过程变化，而不是一刀切。

**场景 2：你已经非常接近 loss 曲线的极小值附近**

这时候：

- 梯度变化很小
- 稍大的梯度容易震荡
- 如果你还用较大的 clip 值，就起不到精细控制效果

**在收敛末期，使用分位数 clip（比如剪掉 top 5%）可以帮助 loss 更平滑地下滑**，而不会跳来跳去。

但大多数时候，比如：

- 你在中早期训练
- 你关注的是更快收敛速度
- 你用的 optimizer 比如 Adam 本身已经有自适应能力

> 那么“剪得优雅”真的没那么重要，甚至还可能 **剪掉有用的“跃迁”梯度**，**降低学习速度**！

结论一句话总结：

> 除非你训练不稳定或者处于收敛末期，**动态分位数裁剪 ≈ 优雅但保守的训练辅助手段**，
>  否则，大胆用个大 clip 值，不炸就好。



## 梯度跳的太狠有什么问题吗？只要不爆炸就行了吧

你的直觉没错 —— “只要不爆炸就行”，这在很多情况下**确实是够用的**，但在**强化学习**和**某些深度网络场景下**，梯度“跳得太狠”也会带来**额外的问题**。我们来拆解下为什么：

“跳得太狠” vs “爆炸” 的区别

- **爆炸（exploding gradients）**：梯度非常大，导致 loss 变成 nan 或无穷大，模型直接崩。
- **跳得太狠**：梯度虽然没 nan，但更新过大，一次 step 让模型跳到离局部最优很远的地方，破坏了训练的连续性。

“跳得太狠”的风险有哪些？

| 问题                                | 影响                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| **1. 优化不稳定 / 发散**            | 参数更新幅度大，可能让模型从“收敛路径”跳出，导致 loss 起伏波动。 |
| **2. 策略更新过猛（RL中特别严重）** | 在 PPO 等 RL 算法中，策略是概率分布，如果改得太快，环境反馈可能完全变质，无法继续训练。 |
| **3. 倾向于 overshoot**             | 就像下山太急会冲过头，梯度大但方向不错也可能导致 overshoot —— 不但没收敛，反而越跳越远。 |
| **4. 导致 learning rate 调不稳**    | 梯度太抖会让你必须用非常小的 lr 才能维稳，这样又导致收敛速度慢。 |

举个例子（以 RL 为例）：

PPO 中 policy 更新用了目标函数：

```python
L = min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)
```

它本身是 **防止策略更新过猛**（r_t 表示新旧策略比值），如果你还让梯度 jump 得很大，那么：

- 即使用 clip 了策略比值，**梯度的方差依然可能很大**
- 训练时可能出现“前几步狂涨，后几步不学”的震荡行为

所以总结一句：

> ❗即使梯度没爆炸，只要跳得太猛，也会带来训练不稳定、收敛速度下降、策略发散等问题。

经验建议

| 模型阶段            | 建议                                                         |
| ------------------- | ------------------------------------------------------------ |
| 初期探索            | 可以 tolerate 稍大一点的梯度，比如 `max_norm = 5`            |
| 中后期收敛          | 建议用 clip，比如固定 `0.5 ~ 2.0`，或用你设计的动态分位数策略 |
| 高噪声环境（如 RL） | 推荐稳一些的策略，哪怕 clip 多一点，也能带来收敛性提升       |

## 这种动态分位数裁剪梯度据你所知有谁在用吗？

动态分位数裁剪梯度（Dynamic Quantile-based Gradient Clipping）是一种较新的梯度裁剪方法，虽然没有非常广泛的文献专门研究这种技术，但它作为一种动态适应性的方法，越来越受到一些实验性和应用性较强的研究和工程项目的关注。

**哪些地方可能会用到这种动态分位数裁剪梯度方法？**

1. **强化学习（RL）**：
   - 在强化学习中，尤其是涉及复杂环境或多智能体的场景时，梯度的稳定性尤为重要。动态分位数裁剪可以更灵活地应对不同训练阶段中梯度的波动，因此一些深度强化学习项目可能会尝试这种方法。
   - **例如**，深度Q网络（DQN）及其变种、PPO等强化学习算法，尤其是在高维状态空间和动作空间下，常常需要应对较大的梯度波动。基于分位数的裁剪可以根据训练的动态调整梯度阈值，有助于提高训练的稳定性。
2. **生成对抗网络（GAN）**：
   - GANs的训练本身就充满挑战，生成器和判别器之间的博弈性训练可能导致梯度爆炸或梯度消失。动态分位数裁剪可以帮助平衡两者的更新，防止训练过程中的极端更新导致模式崩塌。
3. **大规模神经网络（如BERT、GPT等）**：
   - 在大规模预训练模型（如GPT、BERT等）中，梯度爆炸的问题可能不常见，但由于模型本身非常庞大，参数数量极多，梯度值也可能会有较大的波动。动态分位数裁剪可以让模型根据当前梯度的分布进行裁剪，而不是固定阈值，从而避免过多裁剪和信息损失，提升训练效率。

**有些研究中可能用到这种方法：**

1. **OpenAI的训练方法**：
   - OpenAI和一些大规模AI研究机构，可能在他们的研究中使用了类似的动态方法，尤其是在强化学习和生成模型的训练中。虽然他们的研究中没有明确提到"动态分位数裁剪"，但在梯度裁剪、训练稳定性等领域，采用了灵活的动态调整策略。
2. **论文中的梯度裁剪改进**：
   - 在一些论文中（比如强化学习领域），研究者提出了一些基于分位数或统计量的动态裁剪方法，虽然没有广泛传播，但这些方法的思想已在一些实验项目中有所应用。
      例如，在强化学习的研究中，很多研究提出了使用 **"基于统计量的梯度裁剪"**，包括使用 **梯度的分位数** 来动态调整裁剪值。某些强化学习算法中，也已经探索了如何基于经验的历史梯度来调整裁剪的阈值。
3. **框架或自定义实现**：
   - **自定义实现**：一些高效能的训练框架，或者研究者自己实现的梯度裁剪工具，可能会尝试这种策略。比如在训练中动态调整梯度裁剪的阈值，或者实现自定义的梯度裁剪规则，以更好地适应当前模型的训练需求。

**现实中的挑战和限制：**

- **计算开销**：动态分位数裁剪涉及在每次迭代中计算梯度的分位数，这增加了一定的计算开销，尤其是在大型数据集或大规模模型的训练中。
- **实现复杂度**：与固定裁剪值相比，动态分位数裁剪需要更多的代码实现和调试，可能会导致实现上的一些复杂性。
- **不适用于所有情况**：对于一些训练相对稳定的模型，使用简单的固定阈值裁剪可能已经足够，而动态分位数裁剪则可能带来额外的复杂性。

**总结：**

虽然动态分位数裁剪作为一种方法，并没有成为主流的标准做法，但在强化学习、生成对抗网络（GANs）和大规模预训练语言模型（如GPT、BERT等）的研究中，可能会有人使用这种动态梯度裁剪策略，尤其是在需要更精细控制训练过程的场景下。它能提供更灵活的裁剪方式，尤其在训练中期或模型复杂度较高时，能够适应不同的梯度分布，提高训练的稳定性和效率。

如果你在使用这种方法时感觉到效果较好，不妨继续在你的项目中应用，尤其是在需要提高训练稳定性或模型性能的场合。

## 在大型强化学习模型中，比如openai-five和alphastar这种的，你推荐用固定值还是分位数？

在像 **OpenAI Five** 和 **AlphaStar** 这种超大规模强化学习系统中，其实 **固定值和分位数都有合理应用场景**，要看目标：

推荐总结（针对大模型）：

| 维度         | 固定值剪裁                    | 分位数剪裁                                     |
| ------------ | ----------------------------- | ---------------------------------------------- |
| **主流用法** | ✅主流使用方式（如 clip=1）    | 较少直接用于主力代码，但作为研究性调控方式出现 |
| **稳定性**   | 更加稳定，可复现性好          | 需要一定的动态调控策略，更难复现               |
| **可控性**   | 简单清晰，不随batch动态变化   | 响应性强，可适应不同阶段梯度规模               |
| **适用场景** | 大模型、大batch训练、工程部署 | 小batch训练、异常爆炸频繁的阶段、研究实验      |
| **风险**     | 若clip太小，可能阻碍学习      | 若估计不准，early training阶段可能不裁或裁太狠 |

实际案例参考：

- **OpenAI Five**：使用的是 **固定值剪裁**，并且是在梯度累积之后再剪（例如 clip norm=1）。
- **AlphaStar**：也是 **clip norm 固定为 1**，有时候结合 **gradient accumulation** 和 **loss scaling**。

> ✅ **他们主要目标是工程稳定、训练可控、分布式训练容易复现，所以基本都采用固定值方式**，例如：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

那我怎么选？

- 你如果做的是**中型规模强化学习任务**（比如PPO+自定义actor/critic），我建议：
  - **优先使用固定值**，比如 `clip=0.5 ~ 1.0`，保证训练过程稳定；
  - 如果你希望在 early stage 有更强适应能力，**可以考虑引入动态分位数剪裁**作为一种增强机制；
  - **或者混合策略**：early epoch 用动态分位数（适应初期不稳定），后期转为固定值。