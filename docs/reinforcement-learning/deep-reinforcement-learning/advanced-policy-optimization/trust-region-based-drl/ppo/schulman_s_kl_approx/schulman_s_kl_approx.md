# 深入理解 PPO 中的指标监控 Approx KL 计算公式

* [返回上层目录](../proximal-policy-optimization.md)

在 PPO (Proximal Policy Optimization) 的工程实现中，我们经常会在计算监控指标时看到这样一行“奇怪”的代码：

```python
# ratio = pi_new / pi_old
# log_ratio = log(pi_new) - log(pi_old)

# Approx KL (http://joschu.net/blog/kl-approx.html)
# kl = (old_log_probs - new_log_probs) # 原始公式
# 这里用更精确的无偏估计: (ratio - 1) - log(ratio)
approx_kl = (ratio - 1 - log_ratio).mean()
```

初看非常令人困惑：

* KL 散度的定义不是 $-\sum p \log \frac{q}{p}$ 吗？ 也就是 `mean(-log_ratio)` 应该就够了。注意：这里的$p$ 指的是旧概率
* 为什么要加一个 `ratio - 1`？
* 这个公式到底在算什么？

本文将从数学推导和工程实践两个角度，彻底厘清这个公式的来龙去脉。

# 数学推导：它是如何等价于 KL 散度的？

## 定义

我们在 PPO 中计算的是 Reverse KL，即以旧策略 $\pi_{old}$ 为基准，衡量新策略 $\pi_{new}$ 的偏离程度：
$$
KL(\pi_{old} \| \pi_{new}) = \mathbb{E}_{x \sim \pi_{old}} \left[ \log \frac{\pi_{old}(x)}{\pi_{new}(x)} \right] = \mathbb{E}_{x \sim \pi_{old}} \left[ -\log \frac{\pi_{new}(x)}{\pi_{old}(x)} \right]
$$
令概率比率 $r = \frac{\pi_{new}(x)}{\pi_{old}(x)}$ (即代码中的 ratio)，则：
$$
KL = \mathbb{E}_{\pi_{old}} [-\log r]
$$

## 关键恒等式

注意一个概率分布的积分性质：无论策略如何变化，其概率总和必须为 1。
$$
\int \pi_{new}(x) dx = 1 \implies \int \pi_{old}(x) \frac{\pi_{new}(x)}{\pi_{old}(x)} dx = 1
$$
写成期望形式就是：
$$
\mathbb{E}_{x \sim \pi_{old}} [r] = 1
$$
既然 $r$ 的期望是 1，那么：
$$
\mathbb{E}_{x \sim \pi_{old}} [r - 1] = 0
$$

## 凑项

因为 $\mathbb{E}[r-1]=0$，我们可以把它“无偿”地加到 KL 的公式中：
$$
\begin{aligned}
KL &= \mathbb{E}_{x \sim \pi_{old}}[-\log r] \\
&= \mathbb{E}_{x \sim \pi_{old}}[-\log r] + \mathbb{E}_{x \sim \pi_{old}}[r - 1] \\
&= \mathbb{E}_{x \sim \pi_{old}}[(r - 1) - \log r]
\end{aligned}
$$
这就是代码中 `(ratio - 1 - log_ratio).mean()` 的数学来源。

从数学期望的角度看，它与标准的 KL 散度是严格相等的。

# 几何解释：二阶近似

如果我们对函数 $f(r) = (r - 1) - \log r$ 在 $r=1$ （即新旧策略未发生变化）处进行 泰勒展开：

- $f(1) = 0$

- $f'(r) = 1 - \frac{1}{r} \implies f'(1) = 0$

- $f''(r) = \frac{1}{r^2} \implies f''(1) = 1$

根据泰勒公式 $f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2$：
$$
(r - 1) - \log r \approx \frac{1}{2}(r - 1)^2
$$
这意味着，该公式本质上是在计算 概率比率 $r$ 偏离 1.0 的均方误差的一半。这与信息几何中的 Fisher Information Metric 有着深刻联系。

# 工程本质：为什么要多此一举？

既然 `mean(-log_ratio)` 也是 KL 的无偏估计，为什么 OpenAI (John Schulman) 坚持要用 `mean(ratio - 1 - log_ratio)`？

主要有两个工程原因：方差缩减 和 非负性保证。

## 极大地降低方差 (Variance Reduction)

这是最核心的原因。我们引入的 $(r-1)$ 项，在统计学中被称为 Control Variate（控制变量）。

当 $r \approx 1$ 时（PPO 的 Trust Region 约束下通常如此）：

- 方法 A (直接计算)：$-\log r$ 包含了一阶项 $-(r-1)$。当 $r$ 在 1 附近微小波动（如 0.99 或 1.01）时，一阶项会导致结果在正负之间剧烈跳动。

- 方法 B (优化公式)：$(r-1) - \log r$ 消去了一阶项，只保留了二阶项 $\frac{1}{2}(r-1)^2$。

数值举例：

假设我们采集了两个样本，样本 A 的 ratio 为 0.9，样本 B 的 ratio 为 1.1。

| 指标           | 样本 A ($r=0.9$) | 样本 B ($r=1.1$) | 均值 (估计的 KL) | 方差/波动       |
| :------------- | :--------------- | :--------------- | :--------------- | :-------------- |
| $-\log r$      | $0.105$          | $-0.095$         | $0.005$          | 巨大 (跨越正负) |
| $(r-1)-\log r$ | $0.0053$         | $0.0046$         | $0.005$          | 极小 (非常稳定) |

可以看到，虽然两者的均值一样，但优化公式的样本间方差要小得多。在 Mini-batch 训练中，低方差意味着梯度的估计更稳定，训练曲线更平滑。

## 保证非负性 (Non-negativity)

函数 $f(x) = x - 1 - \ln x$ 具有一个漂亮的性质：

- 它是凸函数。

- 它在 $x=1$ 处取全局最小值 0。

- 对于任意 $x>0$，都有 $f(x) \ge 0$。

如果我们直接用 $-\log r$，对于单个样本，如果 $r > 1$（新策略概率提升），算出来的 KL 贡献值是负数。这在物理意义上很反直觉（距离怎么能是负的？）。

而使用 `ratio - 1 - log_ratio`，我们可以保证每一个样本贡献的 KL 值都是非负的。这对于 Debug、绘制 TensorBoard 曲线以及防止异常值抵消非常重要。

# 总结

代码 `approx_kl = (ratio - 1 - log_ratio).mean()` 并非随意之作，它是 PPO 算法实现中的神来之笔：

1. 数学上：利用 $\mathbb{E}[r-1]=0$ 的恒等式，构建了 KL 散度的等价形式。

1. 几何上：近似于 $\frac{1}{2}(r-1)^2$，衡量策略变化的二阶距离。

1. 工程上：作为 Control Variate 消除了估计的一阶噪声，极大地降低了方差，并保证了指标的非负性。

这就是为什么在 OpenAI SpinningUp、Stable Baselines3 以及各种大厂的 RL 库中，你都会看到这一行代码的原因。

严格来说，这个公式 $f(x) = x - 1 - \ln x$ 并不是 OpenAI "发明" 的数学公式（它在信息论和统计学中作为 Bregman Divergence 的一种特例早已存在），但是 将其用于 PPO 代码中作为标准监控指标 (Approx KL) 的做法，确实是由 John Schulman (OpenAI) 在推广 TRPO/PPO 时确立的“行业标准”。

在早期的 TRPO 代码以及后来的 PPO 论文实现细节中，为了快速、低方差地估算 KL 散度（用于动态调整 Learning Rate 或 Clip Range），他们采用了这种 trick。

所以虽然数学源头不是他们，但在深度强化学习领域，这种写法被打上了深深的 OpenAI 烙印，通常被称为 "Schulman's KL Approximation"。

为什么叫Schulman's ？叫 "Schulman's KL Approximation" 或者 "Schulman's trick"，主要是因为 John Schulman 是 TRPO (Trust Region Policy Optimization) 和 PPO (Proximal Policy Optimization) 这两个定义了现代 Policy Gradient 时代的算法的第一作者。

1. 他是 PPO 之父：PPO 论文 *Proximal Policy Optimization Algorithms (2017)* 的一作就是 John Schulman。

1. 工程实现的推广者：在 PPO 之前，计算 KL 散度通常比较繁琐（需要 Hessian 矩阵或者费劲的采样）。Schulman 在 OpenAI Baselines 的代码实现中，极其务实地使用了 $(ratio - 1) - \log(ratio)$ 这个简单的标量公式来近似 KL，并证明了它在工程监控上的有效性。

1. 行业惯例：后来的 RL 开发者（比如 Stable Baselines 的作者们）在阅读 OpenAI 的源码时，为了区分标准的 KL 计算（full calculation）和这个快速估算，习惯性地将其称为 "Schulman's approx"。

简单说，就像“Adam 优化器”我们虽然知道它是 Kingma & Ba 提出的，但代码里如果不叫 Adam 我们也会觉得奇怪。这个 KL 近似写法就是 John Schulman 在 PPO 工程化过程中留下的个人印记。
