# 详解ValueNormalization——驯服Critic的”移动靶“

- [返回上层目录](../value-norm.md)
- [核心痛点：MovingTarget（移动靶问题）](#核心痛点：MovingTarget（移动靶问题）)
- [解决方案：把“绝对值”变成“相对值”](#解决方案：把“绝对值”变成“相对值”)
- [数学推导与实现细节](#数学推导与实现细节)
- [代码实现](#代码实现)
- [为什么Critic输出的是NormalizedValue](#为什么Critic输出的是NormalizedValue)
- [为什么我们的方案优于“LossScaling”？](#为什么我们的方案优于“LossScaling”？)
- [灵魂拷问：极端情况会翻车吗？](#灵魂拷问：极端情况会翻车吗？)
- [进阶对比：ValueNorm-vs-PopArt——谁是最终BOSS？](#进阶对比：ValueNorm-vs-PopArt——谁是最终BOSS？)
- [全文总结](#全文总结)

在强化学习（尤其是 PPO 这类 On-Policy 算法）的工程实践中，我们经常遇到一个令人头秃的现象：Critic Loss 忽高忽低，甚至突然爆炸。

这通常不是因为网络结构不对，而是因为我们陷入了强化学习特有的“非平稳性（Non-stationarity）”陷阱。为了解决这个问题，工业界（OpenAI, DeepMind, 腾讯等）的标准解决方案是引入 Value Normalization。

本文将深入探讨我们项目中采用的 Output Normalization（输出归一化） 方案，剖析其背后的数学原理与工程直觉。

# 核心痛点：MovingTarget（移动靶问题）

## 什么是“移动靶”？

在监督学习中，标签（Target）通常是固定的（比如猫就是猫，狗就是狗）。但在强化学习中，Critic 的学习目标是 累计回报（Return）。

随着 Actor 策略的不断进化：

- 昨天：Actor 很菜，只会乱撞，平均得分是 10 分。Critic 学会了：“这个状态值 10 分”。

- 今天：Actor 顿悟了，学会了连招，平均得分涨到了 100 分。

- 结果：Critic 之前的经验瞬间失效了。它不仅要推翻之前的参数，还要把神经网络的输出强行“拉大” 10 倍。

## 为什么这很糟糕？

1. 梯度爆炸：当 Target 从 10 突变到 100 时，MSE Loss 会瞬间飙升（$(100-10)^2 = 8100$），导致梯度失控。

1. 权重震荡：神经网络的权重需要不断适应新的数值量级（Scale）。Critic 被迫去学习“当前的通货膨胀率是多少”，而不是专注于学习“哪个状态相对更好”。

1. 学习停滞：在 Critic 忙于调整量级时，它无法为 Actor 提供准确的 Advantage 信号，导致训练效率低下。

# 解决方案：把“绝对值”变成“相对值”

我们采用的方案是 基于 RunningMeanStd 的 Output Normalization。

## 核心理念：分工明确

我们将 Critic 的任务拆解为两部分：

1. 会计（RunningMeanStd）：负责记录宏观的“行情”。当前的平均分是多少（$\mu$）？波动范围多大（$\sigma$）？

1. 估价师（Critic Network）：只负责评估微观的“相对好坏”。这个状态比平均水平好多少个标准差？

## 方案全景图

我们不再让 Critic 直接拟合真实的 Return（比如 1000），而是让它拟合 归一化后的 Return（比如 1.5）。

流程如下：

1. Update（统计）：收集最新的 Returns，实时更新全局的均值 $\mu$ 和方差 $\sigma$。

1. Normalize（归一化）：将训练用的 Target 变成标准正态分布 $N(0, 1)$。

1. Train（训练）：Critic 网络学习输出归一化后的值。

1. Denormalize（反归一化）：（关键一步） 在计算 GAE 时，将 Critic 的输出还原为真实物理量纲，以便和 Reward 进行加减运算。

# 数学推导与实现细节

## 训练阶段 (Learner)

我们维护一个全局统计量：
$$
\mu_{new}, \sigma_{new} \leftarrow \text{Update}(\text{Returns})
$$
计算 Loss 时，Critic 的目标是归一化后的 Return：
$$
\text{Target}_{norm} = \frac{\text{Return}_{real} - \mu}{\sigma}
$$
Critic 的 Loss 函数为：
$$
L = \text{HuberLoss}(V_{\theta}(s), \text{Target}_{norm})
$$

> 工程直觉：
>
> 因为 Target 被限制在了 0 附近（通常在 $[-3, 3]$ 之间），Critic 网络的最后一层权重不需要很大，梯度非常稳定。无论游戏得分是 10 分还是 10000 分，Critic 看到的都是同一个难度的“试卷”。

## 推理/采样阶段 (Rollout)

当我们需要计算 GAE（优势函数）时，必须回到物理世界。

GAE 公式：
$$
A_t = r_t + \gamma V_{real}(s_{t+1}) - V_{real}(s_t)
$$
注意：这里的 $r_t$ 是环境给的绝对奖励（未归一化）。因此，$V(s)$ 必须也是绝对值，量纲才能统一。

所以，我们在 preprocess 阶段必须做 Denormalize：
$$
V_{real}(s) = V_{\theta}(s) \cdot \sigma + \mu
$$
其中 $V_{\theta}(s)$ 是 Critic 网络直接输出的“标准分”。

# 代码实现

## 主要修改

### 在learner中实现ValueNorm类

```python
# === Value Normalization Tools ===
class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ValueNorm(nn.Module):
    def __init__(self, input_shape, device=torch.device("cpu")):
        super().__init__()
        self.running_ms = RunningMeanStd(shape=input_shape)
        self.to(device)

    def update(self, x, mask=None):
        if mask is not None:
             x = x[mask.bool()]
        # 防止空数据更新
        if x.numel() > 1:
            self.running_ms.update(x)

    def normalize(self, x):
        return (x - self.running_ms.mean) / torch.sqrt(self.running_ms.var + 1e-8)

    def denormalize(self, x):
        return x * torch.sqrt(self.running_ms.var + 1e-8) + self.running_ms.mean
```

### 在learner中使用ValueNorm类基于当前的return汇报来更新mean和std

```python
    def train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		# ...多余代码忽略        
        # ✅ Value Normalization (Step 1: Update & Normalize Returns)
        # 必须在 Advantage Normalization 之前，因为 returns 应该是 Critic 回归的目标。
        # 如果启用了 Value Norm，Critic 预测的是 Normalized Value，所以回归目标也得是 Normalized Return。
        if self.value_normalizer is not None:
            # 1. 使用当前 Batch 的真实回报更新统计量 (仅使用有效数据)
            flat_returns = batch["returns"].view(-1)
            flat_mask = batch["mask"].view(-1)
            self.value_normalizer.update(flat_returns, flat_mask)

            # 2. 将 returns 归一化，作为 Critic 的回归目标
            # 注意：In-place 修改，影响后续 Loss 计算
            batch["returns"] = self.value_normalizer.normalize(batch["returns"])
```

### 在preprocessor中将critic输出值相关还原来和reward计算

```python
# ✅ Value Denormalization for GAE
# 如果启用了 Value Norm，Critic 输出的是 Normalized Value (N(0,1))
# 而 Rewards 是 Real Scale
# GAE 计算公式：delta = r + gamma * V_next - V
# 必须将 V 还原为 Real Scale，才能与 r 进行计算
if value_normalizer is not None:
    values = value_normalizer.denormalize(values)
    bootstrap_value = value_normalizer.denormalize(bootstrap_value)
    critic_value_scalar = value_normalizer.denormalize(critic_value_scalar)
```

## 修改总结

所有代码修改已完成！现在你的强化学习框架已经集成了工业级的 Value Normalization (RunningMeanStd)。

修改总结：

1. Learner (learner.py)：

- 实现了 RunningMeanStd（算法核心）和 ValueNorm（封装工具）。

- 在 train 函数中，Critic Loss 计算前，先用 ValueNorm 对真实回报 (returns) 进行归一化。这确保了 Critic 网络只需要拟合一个稳定的分布 (Mean=0, Std=1)。

- 在 save_checkpoint / load_checkpoint 中增加了对 ValueNorm 状态的保存和恢复。

1. Preprocess (preprocess.py)：

- 这是最关键的一步。由于 Critic 网络现在输出的是 归一化后的价值 (Normalized Value)，而 GAE 计算需要 真实价值 (Real Value)。

- 我在 preprocess_segments 中增加了逻辑：如果启用了 Value Norm，会先将 Worker 传回来的 values、bootstrap_value 和 critic_value_scalar 反归一化 (Denormalize)，还原成真实数值，然后再计算 GAE。

- 这样保证了数学逻辑的绝对正确：GAE = r_real + gamma * V_real_next - V_real。

1. Trainer (trainer.py) & Collector (collector.py)：

- 打通了管道，将 Learner 中的 value_normalizer 实例一路传递给 preprocess_segments。

1. Config (ppo_config.py)：

- 新增了 use_value_norm = True 开关。

### 预期效果

- Critic Loss 曲线：你会发现 Critic Loss 不再随着 Agent 分数的上涨而剧烈震荡（Moving Target 问题被解决）。它会变得非常平滑，且数值会稳定在较小的范围内（因为拟合目标被归一化了）。

- 训练稳定性：即使 Reward 数值很大（比如 1000 分），Critic 梯度也不会爆炸。

- 收敛速度：由于 Critic 不再需要频繁调整权重来适应数值量级的变化，它的收敛速度会显著加快。

这就如同给你的 Critic 装上了一个“自动变焦镜头”，无论外部世界的分数怎么变，它看到的画面永远清晰稳定。

# 为什么Critic输出的是NormalizedValue

> 这里我没有看明白，为什么Critic 输出的是 Normalized Value (N(0,1))呢？又没有给Critic头的输出加上任何限制，怎么就能知道其输出是Normalized Value (N(0,1))呢？

这是一个非常好的问题！你的直觉很敏锐：我们并没有在 Critic 网络的输出层加 Tanh 或 Sigmoid 等强行限制输出范围的激活函数，甚至没有修改网络结构，那凭什么说它输出的就是 Normalized Value 呢？

答案在于：监督学习的训练目标（Loss Target）决定了网络的输出。

## Critic 是个“听话的学生”

神经网络本质上是一个函数拟合器（Function Approximator）。你给它什么 Target，它就努力去拟合什么。

- 没有 Value Norm 时：

  - Target = 真实回报 (比如 1000)。

  - Loss = $(V(s) - 1000)^2$。

  - Critic 必须输出 1000 才能最小化 Loss。所以它学会输出 Real Value。

- 有 Value Norm 时 (我们在 learner.py 里做的修改)：

  - 我们把真实回报 (1000) 进行了归一化：$Target_{norm} = \frac{1000 - \mu}{\sigma} \approx 1.5$。

  - Loss = $(V(s) - 1.5)^2$。

  - Critic 必须输出 1.5 才能最小化 Loss。

虽然 Critic 网络本身结构没变，但因为我们喂给它的标签（标准答案）全是归一化后的数据（均值0，方差1），所以它被迫学会了输出归一化后的预测值。

## 这里的 Denormalize 是为了什么？

现在 Critic 学会了“打标准分”（比如输出 1.5），而不是“打绝对分”（输出 1000）。

但是，GAE 的公式是物理意义上的“优势累积”：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
在这个公式里：

- $r_t$ (Reward)：是环境给的真实绝对值（比如 +10 分）。

- $V(s)$：如果我们直接用 Critic 的输出（1.5），公式就乱套了。

- 错误计算：$10 + 0.99 \times 1.6 - 1.5 \approx 10$（Advantage 巨大，因为 r 很大）。

- 但实际上 1.5 代表的是 1000 分，1.6 代表的是 1010 分。真实的 Advantage 应该是 $10 + 0.99 \times 1010 - 1000 \approx 9.9$。

量纲必须统一！

因为 $r_t$ 是绝对值（我们通常不对 Reward 做 Norm，只对 Return 做 Norm），所以我们必须把 $V(s)$ 从“标准分”还原回“绝对分”，才能和 $r_t$ 放在同一个公式里加减。

这就是为什么在 preprocess.py 里，计算 GAE 之前必须做 Denormalize (反归一化)。

## 总结

* Learner 训练时：把 Target 变成 Norm，强迫 Critic 学会输出 Norm Value。
* Rollout 采样时：Critic 输出的自然就是 Norm Value。
* GAE 计算时：把 Critic 输出的 Norm Value 还原成 Real Value，以便和 Real Reward 进行物理运算。

# 为什么我们的方案优于“LossScaling”？

有些早期的实现方案（我们称之为“半吊子方案”）是这样的：

- 网络依然尝试输出真实值（比如 1000）。

- 只在算 Loss 时，把 $(V - Target)$ 除以 $\sigma$。

对比分析：

| 特性         | 半吊子方案 (Loss Scaling)          | 我们的方案 (Output Normalization) |
| :----------- | :--------------------------------- | :-------------------------------- |
| 网络输出     | 真实值 (e.g., 1000)                | 标准分 (e.g., 1.0)                |
| 权重大小     | 需要很大的 Weight/Bias             | 权重很小，初始化友好              |
| 收敛速度     | 慢（需要漫长的时间把输出“顶”上去） | 快（无需适应量级变化）            |
| 非平稳性适应 | 差（Target 变大时网络需剧烈调整）  | 强（Target 永远在 0 附近）        |
| 主流采用     | 少见 / 已淘汰                      | OpenAI / DeepMind / CleanRL 标配  |

# 灵魂拷问：极端情况会翻车吗？

场景：假设过程奖励全是微小的 -0.1，突然终局奖励来了个 +100。

- $\mu \approx 0, \sigma \approx 10$（假设统计稳定后）。

- 过程分归一化后变成 $-0.01$，大分归一化后变成 $+10$。

Q：小分会不会被淹没？大分会不会炸梯度？

A：不会。这是“双保险”机制：

1. Value Norm (自适应缩放)：

- 如果大分频繁出现，$\sigma$ 会自动变大，把 +100 压缩回 +1.0，保护网络不炸。

- 如果全是小分，$\sigma$ 会自动变小，把 -0.1 放大回 -1.0，保证细节可学。

1. Huber Loss (硬截断)：

- 在 $\sigma$ 还没来得及更新的突变瞬间（Outlier），Huber Loss 会把 MSE 的平方级惩罚降级为线性惩罚，防止单次梯度爆炸。

# 进阶对比：ValueNorm-vs-PopArt——谁是最终BOSS？

在深度强化学习的高端局（如 DeepMind 的研究）中，你经常会听到一个更响亮的名字：PopArt (Preserving Outputs Precisely Adaptive Robustness Technique)。

很多同学会困惑：我们现在用的 Value Norm 是不是就是 PopArt？如果不是，它俩谁更强？

## 本质上的“亲兄弟”

首先要明确：从数学目标上看，我们现在的方案与 PopArt 是高度一致的。

- 共同目标：都希望网络输出的是 Normalized Value (归一化后的相对值)，从而让神经网络摆脱对数值量级（Scale）的敏感性。

- 共同手段：都依赖 RunningMeanStd 来实时追踪回报的 $\mu$ 和 $\sigma$。

## 实现上的“大不同”

虽然目标一致，但在 “如何更新网络” 这个环节上，两者走向了不同的道路。

**方案 A：Value Norm (我们目前的方案)**

这是 Data-Level (数据层) 的操作。

- 做法：我们修改的是 标签 (Target)。

- 流程：

1. 把 Target 变成 $\frac{Target - \mu}{\sigma}$。

1. 告诉网络：“嘿，你的新目标是这个归一化后的值，快去拟合它！”

1. 网络通过 梯度下降 (Gradient Descent)，慢慢修改权重，使得输出逼近新目标。

- 缺点：滞后性 (Lag)。当 $\mu$ 和 $\sigma$ 突变时，Target 变了，但网络的权重还是旧的。网络需要几步迭代才能“追”上新的 Target。这在 Reward 变化极快极剧烈时，可能会有一瞬间的 Loss 抖动。

**方案 B：PopArt (DeepMind 的黑魔法)**

这是 Model-Level (模型层) 的操作。

- 做法：我们直接 手术修改网络权重 (Weight Surgery)。

- 流程：

  - 当 $\mu$ 和 $\sigma$ 更新时，PopArt 不等待梯度下降。

  - 它根据数学公式，瞬间 修改网络最后一层的 $W$ 和 $b$。
    $$
    \begin{aligned}
    W_{new} &= \frac{\sigma_{old}}{\sigma_{new}} W_{old}\\
    b_{new} &= \frac{\sigma_{old} b_{old} + \mu_{old} - \mu_{new}}{\sigma_{new}}
    \end{aligned}
    $$

  - 魔法效果：在统计量更新的毫秒级瞬间，网络的真实输出值 (Real Output) 保持完全不变！网络甚至感觉不到 Target 变了，因为它已经被“整容”成适应新 Target 的样子了。

- 优点：零滞后。完美解决了非平稳性，收敛极快。

## 该如何抉择？

| 维度       | Value Norm (我们的方案)        | PopArt                               |
| :--------- | :----------------------------- | :----------------------------------- |
| 实现难度   | 低 (只需在 Learner 改几行代码) | 高 (需侵入 Model 定义，手写权重更新) |
| 通用性     | 强 (适用于任何网络结构)        | 弱 (需针对最后一层特殊设计)          |
| 收敛速度   | 快 (足够应对 99% 场景)         | 极快 (应对极端非平稳场景)            |
| 计算开销   | 极低                           | 低，但增加了代码维护成本             |
| 工业界现状 | PPO / CleanRL / SB3 的标准配置 | IMPALA / R2D2 / AlphaStar 的标准配置 |

## 比较后的结论

- 对于 PPO 算法：Value Norm 是性价比之王。因为 PPO 本身就是 On-Policy 的，数据分布变化相对平缓（受限于 Trust Region），Value Norm 的那一点点“滞后”几乎可以忽略不计。这也是为什么 OpenAI 和 CleanRL 都首选此方案。

- 何时需要 PopArt？：如果你在做 Off-Policy (如 IMPALA)，或者任务的 Reward 跨度极其变态（如 Atari 57 游戏全家桶，跨度 $10^6$），且你发现 Critic 无论如何都追不上 Target 的变化时，才是祭出 PopArt 这个大杀器的时候。

一句话总结：我们现在的 Value Norm 方案，就是 PPO 在工业界落地的“黄金标准”。它地的“黄金标准”。

# 全文总结

通过引入 RunningMeanStd 和 Output Normalization，我们成功地：

1. 解决了 Moving Target 问题，让 Critic 训练曲线丝般顺滑。

1. 实现了 Scale Invariant（尺度不变性），同一套超参数可以跑倒立摆（分小），也能跑复杂博弈（分大）。

1. 利用 Denormalize 保证了 GAE 数学物理意义的正确性。

这就是让 PPO 从“能跑”进化到“工业级鲁棒”的关键拼装”的关键拼图”。
