# GAE广义优势估计

- [返回上层目录](../trust-region-based-drl.md)
- [GAE的核心思想](#GAE的核心思想)
  - [n-stepTD残差加权（不带γ）](#n-stepTD残差加权（不带γ）)
  - [单步TD残差加权（带γ）](#单步TD残差加权（带γ）)
- [GAE公式推导](#GAE公式推导)

GAE（Generalized Advantage Estimation，广义优势估计）是一种用于**减少方差**并**提高样本效率**的优势函数估计算法，常用于**PPO**（Proximal Policy Optimization）和**TRPO**（Trust Region Policy Optimization）等强化学习算法中。

# GAE的核心思想

在策略梯度方法中，优势函数（Advantage Function）用于衡量某个动作相对于当前策略的期望回报有多好。标准的优势函数定义如下：
$$
A_t = Q(s_t, a_t) - V(s_t)
$$
其中：

- $Q(s_t, a_t)$表示在状态$s_t$执行动作$a_t$之后的期望总回报
- $V(s_t)$表示状态$s_t$的值函数，即从该状态出发的期望回报

**但问题是：**

- **直接使用优势函数会导致高方差**，尤其是在Monte Carlo估计时
- **引入 Bootstrapping 会导致偏差**，比如$TD(\lambda)$可能会引入一定的估计误差

**GAE旨在通过一个可调节参数$\lambda$来在“低方差高偏差”和“高方差低偏差”之间找到平衡。**

GAE的全名是 **Generalized Advantage Estimation**，它通过加权的方式将多个时间步的 **TD残差** 进行平滑，从而减少方差并提高估计的稳定性。具体来说，GAE是通过多个 **δt\delta_tδt**（也就是多个 **TD残差**）的加权平均来估计 **优势函数 A_t**。

GAE的计算公式有两种（GAE的计算方法可以有两种不同的视角，但它们是**等效的**。）：

## n-stepTD残差加权（不带γ）

$$
A_t^{GAE(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \delta_t^{(n)}
$$

其中$\delta_t^{(n)}$是第n步的TD残差。

这就是为什么GAE中的$\delta_t$被称为**TD残差**，而GAE通过加权平均这些残差，来平衡偏差和方差。

而**n-step TD残差**（n-step TDresidual）的公式是用来估计 **动作值函数$Q(s_t, a_t)$**或**状态值函数$V(s_t)$** 的。它通过观察从当前时间步t到未来n步的回报来计算一个误差（残差），反映当前状态（和动作）的价值与未来回报之间的差异。

对于一个给定的时间步t，n-step TD残差$\delta_t^{(n)}$的计算公式为：

**状态值函数的 n-step TD残差**：
$$
\delta_t^{(n)} = G_t^{(n)}-V(s_t) = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)
$$

- $\sum_{i=0}^{n-1} \gamma^i r_{t+i}$：是从时间步t到$t+n−1$的累计奖励，使用折扣因子$\gamma$进行加权。
- $\gamma^n V(s_{t+n})$：是 **n 步后**的状态$s_{t+n}$的估计值。
- $V(s_t)$：是当前状态 $s_t$ 的值函数估计。

注意，这里有个小细节：GAE是针对动作的，那么，$\delta_t^{(n)}$也应该是针对动作的，由于$\delta_t^{(n)} = G_t^{(n)}-V(s_t)$，所以$G_t^{(n)}$也是针对动作的，但是实际上$G_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n})$，你看，其最后一项是$V(s_{t+n})$，这显然是说明$G_t^{(n)}$是针对状态的呀？

> 你的思考非常敏锐！你指出的这个看似的“矛盾”其实正是GAE在设计上的一个**微妙之处**。这涉及到**值函数（V）** 和 **动作值函数（Q）** 之间的关系，我们来仔细拆解下这个问题：
>
> 1. 先看问题的本质
>
> 你说得没错，$G_t^{(n)}$的确是针对**状态值函数**的，因为其最终形式是：
> $$
> \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n})
> $$
> 最后一项是$V(s_{t+n})$，是对状态的估计，而不是对动作的估计。
>
> 这意味着$G_t^{(n)}$是针对**状态**的估计，通常用于TD学习中的值函数更新。
>
> 2. 但GAE是如何“转化”为针对动作的呢？
>
> GAE的目标是估计**优势函数**，即：
> $$
> A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
> $$
> 问题的关键在于：
>
> * 在策略梯度方法（如 PPO）中，**策略的更新是基于优势函数**。
>
> * 由于$Q(s_t, a_t)$的直接估计很困难（因为涉及到特定动作），所以通过TD残差来间接估计：
>
> $$
> \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \approx Q(s_t, a_t) - V(s_t)​也就是说，**用 TD 残差（基于状态值的）来近似 Q 值和 V 值的差异**，从而得到优势估计。
> $$
>
> * 也就是说，**用TD残差（基于状态值的）来近似 Q 值和 V 值的差异**，从而得到优势估计。

## 单步TD残差加权（带γ）

这是GAE最常见的计算方式。我们通过逐步计算每个单步的TD残差$\delta_t$，然后对这些残差进行加权平均。加权的系数是$(\gamma \lambda)^k$，其中$k$是步数的偏移量。

计算公式是：
$$
A_t^{GAE(\lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
$$
每个$\delta_t$的计算是单步的TD残差：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
在这种计算方式中，每个$\delta_t$是单步的TD残差，然后根据$\gamma \lambda$的加权系数进行平滑，逐步计算出更长时间步的加权平均。

# GAE公式推导

让我们从最基础的概念讲起，确保你能真正理解GAE。我们一步步来，把它拆解成你可以理解的小部分。

## 强化学习的基本目标

在强化学习（RL）中，我们的目标是**让智能体学会如何在环境中行动，以获得最高的奖励**。核心思想是：

- 智能体（Agent）观察到**状态**$s_t$
- 选择一个**动作**$a_t$
- 通过这个动作，环境给出**奖励**$r_t$
- 进入新的**状态** $s_{t+1}$
- 目标是让累计奖励尽可能大

那么，**问题来了**：

- 我们怎么知道某个动作是不是“好”的？
- 一个动作的真正价值不只是当前的奖励，还包括未来可能带来的奖励，我们要怎么估计这个“长期价值”呢？

## 认识值函数V(s)和优势函数A(s,a)

认识值函数$V(s)$和优势函数$A(s,a)$

为了衡量一个动作的好坏，我们有两个重要的概念：

* **值函数$V(s)$**：

它告诉我们，**如果从状态$s$开始，并按照当前策略$\pi$行动，未来的总奖励大概（多次采样的平均）是多少**：
$$
V(s) = \mathbb{E} \left[ r_0 + \gamma r_1 + \gamma^2 r_2 + \dots \right]
$$
其中$\gamma$是折扣因子，表示未来奖励的影响程度。

* **优势函数$A(s, a)$**：

它告诉我们，**采取动作$a$相比于平均水平（值函数）好多少**：
$$
A(s, a) = Q(s, a) - V(s)
$$
直白点说，**优势函数衡量的是“这个动作比其他普通动作好多少”**。

## 计算优势的难点

**如果我们可以准确计算$A(s, a)$，就能更好地训练智能体，但计算它非常困难！**

有两种常见的方法：

1. **Monte Carlo（MC）方法**：
   - 直接跑完整个回合，计算从当前状态出发的实际总奖励
   - **问题**：方差很大，训练很不稳定。就是说，其多次采样的平均是肯定是准确的，但是每次采样的值变化很大，可能本次采样是100，下次是1100，再下次是-900，其平均值是100，而真实的平均值就是100。
2. **时间差分（TD）方法**：
   - 只用一步的奖励加上下一状态的值估计
   - **问题**：虽然方差小了，但会有偏差。就是说，其多次采样的平均值和真实平均值是有偏差的，但是每次采样的值都变化不大，可能本次采样是500，下次是550，再下次是450，采样平均值是500，但是真实的平均值是100，显然存在偏差。

那么，我们能不能**结合这两种方法，既降低方差又减少偏差**？

**这就是GAE（广义优势估计）要解决的问题！**

## GAE的核心思想

GAE通过一个“平衡因子”$\lambda$在Monte Carlo和TD之间找到一个中间值。

**计算方法如下：**

1. **先计算TD误差**：
   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)δ
   $$
   这个公式的意思是：

   “我实际得到的奖励$r_t$加上下一状态的估计值（折扣后的），再减去当前状态的估计值。”

2. **然后，我们用GAE累积多个TD误差：**
   $$
   A_t^{GAE(\lambda)} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
   $$
   这里的$\lambda$就是控制**累积多少步的TD误差**：

   - **$\lambda = 0$**：只用当前TD误差，更新更快但不稳定
   - **$\lambda = 1$**：用所有未来TD误差，结果更准确但方差更大
   - **一般选择$\lambda = 0.95$，在稳定性和准确性之间找平衡**

## 递推计算GAE

为了高效计算，我们用递推公式：
$$
A_t^{GAE(\lambda)} = \delta_t + (\gamma \lambda) A_{t+1}^{GAE(\lambda)}
$$
这个公式的意思是：

- 先算当前TD误差$\delta_t$
- 再加上未来的优势（折扣后的）

这可以让我们**从后往前**递推计算GAE，使计算更加高效。

## GAE在PPO里的作用

GAE计算的优势值$A_t$被用在策略梯度更新中：
$$
\nabla_{\theta} J(\theta) = \mathbb{E} \left[ A_t \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) \right]
$$
然后用于PPO的目标函数：
$$
L(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]
$$
**简单来说：**

- **GAE提供一个更稳定的优势估计**
- **让PPO/ TRPO 训练更快，更稳定**
- **减少高方差问题，提高样本利用率**

## GAE总结

- **GAE是一种更稳定的优势估计算法**
- **它结合了 Monte Carlo（低偏差但高方差）和TD方法（高偏差但低方差）**
- **通过$\lambda$控制优势估计的平滑程度，一般取 0.95**
- **广泛用于PPO/ TRPO 等强化学习算法，提高训练效率**

## 回顾蒙特卡洛MC和时间差分TD方法

我们下面仔细讲清楚**Monte Carlo（MC）方法**和**时间差分（TD）方法**是怎么估计**优势函数**$A(s, a)$的。

### 复习：优势函数A(s, a)是什么

在强化学习（RL）里，我们想知道**某个动作a**到底有多好。
 为了量化它，我们使用**优势函数A(s, a)**：
$$
A(s, a) = Q(s, a) - V(s)
$$
它的含义是：

- **$Q(s, a)$**：从状态$s$执行动作$a$后，未来可能获得的总奖励
- **$V(s)$**：从状态$s$出发，按照策略$\pi$采取所有可能动作的期望奖励

$\Rightarrow$ **优势函数告诉我们，选择$a$这个动作比平均水平（V(s)）要好多少！**

### Monte Carlo（MC）方法：整局回合后计算

**核心思想**

Monte Carlo 方法的思路是：

- 直接从当前状态$s_t$**一路执行到底**，直到整个回合结束
- 把所有的奖励累加起来，作为这个状态-动作的真实价值
- 这样，我们就可以估计$Q(s_t, a_t)$，然后计算$A(s_t, a_t)$

**计算公式**

Monte Carlo 方法直接计算**完整的回报（Return）**：
$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T
$$
然后，用它来估计**动作值函数**：
$$
Q(s_t, a_t) \approx G_t
$$
再计算优势：
$$
A(s_t, a_t) = G_t - V(s_t)
$$
**示例**

假设我们有一个游戏回合，智能体的奖励如下：

- $2r_1 = 1, r_2 = 0, r_3 = 2$，最终结束

如果**折扣因子$\gamma = 0.9$**，那么：
$$
\begin{aligned}
G_1 &= 1 + 0.9(0) + 0.9^2(2) = 1 + 0 + 1.62 = 2.62\\
G_2 &= 0 + 0.9(2) = 1.8\\
G_3 &= 2
\end{aligned}
$$
然后，假设我们的**状态值函数V(s)是**：

- $V(s_1) = 2.3, V(s_2) = 1.5, V(s_3) = 1.9$

那么，**计算优势**：
$$
\begin{aligned}
A(s_1, a_1) &= G_1 - V(s_1) = 2.62 - 2.3 = 0.32\\
A(s_2, a_2) &= G_2 - V(s_2) = 1.8 - 1.5 = 0.3\\
A(s_3, a_3) &= G_3 - V(s_3) = 2 - 1.9 = 0.1
\end{aligned}
$$
**优点和缺点**

**优点**：

- **无偏**，因为它使用了完整的真实回报，不依赖估计值
- 适用于终点明确的环境，比如回合制游戏（围棋、象棋）

**缺点**：

- **方差大**：每次跑完整个回合，奖励的波动很大
- **不适用于无限长的任务**：如果没有终点（比如机器人导航），MC 方法无法使用
- **收敛慢**：因为需要多个完整回合才能更新策略

### 时间差分（TD）方法：一步步估计

**核心思想**

TD 方法的思路是：

- 不等到整个回合结束，而是**一步步更新**
- 只用当前奖励$r_t$和下一个状态的值$V(s_{t+1})$来估计$Q(s_t, a_t)$

**计算公式**

TD 方法使用**时间差分（TD 误差）**来更新：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
然后，我们用TD误差来估计**优势函数**：
$$
A(s_t, a_t) \approx \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
**示例**

假设：

- $r_1 = 1, r_2 = 0, r_3 = 2$，回合结束
- 状态值函数：
  - $V(s_1) = 2.3$
  - $V(s_2) = 1.5$
  - $V(s_3) = 1.9$
  - $V(s_4) = 0$（终止状态）

我们计算**TD 误差**：
$$
\begin{aligned}
\delta_1 = r_1 + \gamma V(s_2) - V(s_1) = 1 + 0.9(1.5) - 2.3 = 1 + 1.35 - 2.3 = 0.05\\
\delta_2 = r_2 + \gamma V(s_3) - V(s_2) = 0 + 0.9(1.9) - 1.5 = 1.71 - 1.5 = 0.21\\
\delta_3 = r_3 + \gamma V(s_4) - V(s_3) = 2 + 0 - 1.9 = 0.1
\end{aligned}
$$
**优点和缺点**

**优点**：

- **样本效率高**，可以一步步更新，不需要等完整回合
- **计算更快**，适用于无限长的任务（如机器人控制）
- **方差小**，比 MC 方法稳定

**缺点**：

- **有偏差**，因为它依赖于当前的值函数$V(s)$
- **可能会收敛到次优解**，如果值函数估计有误，可能会影响学习效果

### Monte Carlo与TD方法的对比

| 方法                  | 计算方式                   | 方差   | 偏差       | 适用场景                       |
| --------------------- | -------------------------- | ------ | ---------- | ------------------------------ |
| **Monte Carlo（MC）** | 跑完整个回合后计算 GtG_tGt | **高** | 低（无偏） | 适用于有终点的任务（如游戏）   |
| **时间差分（TD）**    | 只用当前奖励和下一状态估计 | **低** | 高（有偏） | 适用于连续任务（如机器人控制） |

$\Rightarrow$ **GAE（广义优势估计）就是在 MC 和TD之间找一个折中！**

无限长步的TD是否就相当于MC?

是的，**无限长步的 TD（$n \to \infty$）相当于 Monte Carlo（MC）方法！**

让我们用数学和直觉来解释这个问题。

**1. 先回顾 TD（时间差分）方法**

在TD方法中，我们用**有限步的回报**来估计$Q(s, a)$。如果我们只看**一步TD更新**（$n=1$），它的估计是：
$$
Q(s_t, a_t) \approx r_t + \gamma V(s_{t+1})
$$
如果我们用**多步 TD（n-step TD）**，则考虑接下来的n步：
$$
Q(s_t, a_t) \approx r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$
$\Rightarrow$ **TD 估计是逐步逼近真实回报的！**

**2. Monte Carlo 方法的特点**

Monte Carlo 方法的回报是：
$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T
$$
它考虑的是**从$s_t$开始一直到回合结束的完整回报**。

**3. 当$n \to \infty$时，TD 变成 MC**

如果我们把TD估计的步数n设得越来越大：
$$
Q(s_t, a_t) \approx r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$
$\Rightarrow$ **当n趋向无穷时，$\gamma^n V(s_{t+n})$这一项会消失（如果V(s)有界）**，于是我们得到：
$$
Q(s_t, a_t) \approx r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T
$$
这正是 Monte Carlo 方法的计算公式！

**直觉理解**

- **1-step TD** 只看一步，更新很快，但误差较大
- **n-step TD** 看更多步，误差减少
- **无限步 TD（$n \to \infty$）** 直接看完整回合的回报，就变成 MC 了！

**4. 结论**

- **有限步 TD（$n < \infty$）** 是介于 **1-step TD和 MC 之间的折中方案**。
- **当TD的步数趋于无穷时（$n \to \infty$）**，它就等价于 Monte Carlo 方法。
- 这也是**$TD(\lambda)$ 和$GAE(\lambda)$采用$\lambda$作为权衡因子的原因**，它们希望在 **TD 和 MC 之间找到一个平衡**。

## GAE如何结合 MC 和TD？

GAE采用**加权平均**的方式，把 MC 和TD结合起来：
$$
A_t^{GAE(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

- **$\lambda \approx 0$**：更像 TD，更新更快但有偏差
- **$\lambda \approx 1$**：更像 MC，准确但方差大

**总结**

1. **Monte Carlo（MC）方法**：完整计算回报，准确但方差大
2. **时间差分（TD）方法**：一步步估计，稳定但有偏差
3. **GAE（广义优势估计）** 结合两者，通过 λ\lambdaλ 调节偏差和方差的权衡

加下来一步步讲解GAE（广义优势估计，Generalized Advantage Estimation）是如何**结合 Monte Carlo（MC）和 时间差分（TD）** 的。

### 先回顾GAE为什么重要？

**GAE的目标**：
 我们希望估计**优势函数**$A(s, a)$，但 **MC 方法和TD方法各有优缺点**：

- **MC 方法**：准确，但方差大，收敛慢
- **TD 方法**：稳定，但有偏差

$\Rightarrow$ **GAE试图找到 MC 和TD之间的平衡**，让估计既稳定又准确！

### 先看TD误差（TD Error）

TD 误差（$\delta_t$）是TD方法中**一步估计**的核心：

$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

它表示：

- **“当前奖励” $r_t$** 加上 **“未来的状态值” $\gamma V(s_{t+1})$**
- **减去当前状态值$V(s_t)$**，表示比预期好还是差

**问题**：

- 只看一步（$s_t \to s_{t+1}$）可能不够准确（有偏差）
- 但如果用完整 MC 方法（一直累加到结束）又会导致方差很大

### 未来多步的TD误差

如果我们用 **2-step TD**（看 2 步），那么计算方式变成：
$$
\delta_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)
$$
如果是 **3-step TD**：
$$
\delta_t^{(3)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}) - V(s_t)
$$
如果是 **n-step TD**：
$$
\delta_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)
$$
$\Rightarrow$ **当$n \to \infty$时，$\delta_t^{(n)}$变成 MC 方法！**

### GAE如何平衡 MC 和 TD

GAE采用了一种**衰减平均**的方法，把不同步长的TD估计组合在一起：
$$
A_t^{GAE(\lambda)} = (1 - \lambda) (\delta_t^{(1)} + \lambda \delta_t^{(2)} + \lambda^2 \delta_t^{(3)} + \dots)
$$
这里：

- **$\lambda$是一个超参数，介于 0 和 1 之间**
- **当$\lambda \to 0$，GAE变成 1-step TD**（偏差大但稳定）
- **当$\lambda \to 1$，GAE变成 MC 方法**（准确但方差大）

$\Rightarrow$ **GAE通过$\lambda$调节了TD和 MC 之间的平衡！**

可以把GAE理解为：

- **$\lambda \approx 0$**：主要依赖**短步 TD**，更新更稳定，但有偏差
- **$\lambda \approx 1$**：更像 MC，估计更准确，但方差大

**一般选择$\lambda = 0.95$**，让优势估计更平稳，同时不过度偏离真实值。

### 为什么当 λ趋于1则GAE变成 MC 方法？

请问是为啥，能推导一下公式吗？

我们来一步步推导，**证明GAE当$\lambda \to 1$时会变成 Monte Carlo（MC）方法**，并解释它的直觉含义。

**1. 先回顾GAE的定义**

GAE（广义优势估计）是：
$$
A_t^{GAE(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \delta_t^{(n)}
$$
其中：
$$
\delta_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)
$$
是 n-step TD误差。

现在，我们要研究当$\lambda \to 1$时，$A_t^{GAE(\lambda)}$会变成什么。

**2. 计算GAE在$\lambda \to 1$时的极限**

我们先展开GAE：
$$
A_t^{GAE(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \delta_t^{(n)}
$$
将$\delta_t^{(n)}$代入：
$$
A_t^{GAE(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \left( \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t) \right)
$$
把求和符号拆开：
$$
A_t^{GAE(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \sum_{i=0}^{n-1} \gamma^i r_{t+i} + (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \gamma^n V(s_{t+n}) - (1 - \lambda) V(s_t) \sum_{n=1}^{\infty} \lambda^{n-1}
$$
我们分别计算这三个部分。

**第一部分：回报项**

交换求和顺序：
$$
\sum_{n=1}^{\infty} \lambda^{n-1} \sum_{i=0}^{n-1} \gamma^i r_{t+i} = \sum_{i=0}^{\infty} \gamma^i r_{t+i} \sum_{n=i+1}^{\infty} \lambda^{n-1}
$$
这里的第二个求和：
$$
\sum_{n=i+1}^{\infty} \lambda^{n-1} = \lambda^i \sum_{m=0}^{\infty} \lambda^m = \frac{\lambda^i}{1 - \lambda}
$$
因为根据标准的等比级数求和公式：
$$
\sum_{m=0}^{\infty} \lambda^m = \frac{1}{1 - \lambda} \quad (\text{当 } |\lambda| < 1)
$$
所以：
$$
\sum_{n=1}^{\infty} \lambda^{n-1} \sum_{i=0}^{n-1} \gamma^i r_{t+i} = \frac{1}{1 - \lambda} \sum_{i=0}^{\infty} \gamma^i r_{t+i} \lambda^i
$$
当$\lambda \to 1$时，$\lambda^i \approx 1$，所以：
$$
\begin{aligned}
&(1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \sum_{i=0}^{n-1} \gamma^i r_{t+i} \\
=& (1-\lambda)\frac{1}{1 - \lambda} \sum_{i=0}^{\infty} \gamma^i r_{t+i} \lambda^i\\
=& \sum_{i=0}^{\infty} \gamma^i r_{t+i} \lambda^i\\
\to & \sum_{i=0}^{\infty} \gamma^i r_{t+i} \quad(\lambda^i\approx 1)
\end{aligned}
$$
这个正是**Monte Carlo 方法的回报项**！

**第二部分：状态值项**
$$
(1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \gamma^n V(s_{t+n})
$$
当$\lambda \to 1$时，这个项趋近于 0（因为$(1 - \lambda)$变得非常小，而$V(s_{t+n})$有界）。

**第三部分：初始状态值项**
$$
(1 - \lambda) V(s_t) \sum_{n=1}^{\infty} \lambda^{n-1}
$$
因为几何级数的和是：
$$
\sum_{n=1}^{\infty} \lambda^{n-1} = \frac{1}{1 - \lambda}
$$
所以：
$$
(1 - \lambda) V(s_t) \frac{1}{1 - \lambda} = V(s_t)
$$
**3. 最终结论**

把这三部分加起来：
$$
A_t^{GAE(\lambda)} = \sum_{i=0}^{\infty} \gamma^i r_{t+i} - V(s_t)
$$
这正是 **Monte Carlo（MC）方法的优势估计**：
$$
A_t^{MC} = G_t - V(s_t)
$$
因此，**当$\lambda \to 1$时，GAE变成 MC 方法！**

**4. 直觉理解**

1. **GAE是 TD($\lambda$) 的一种形式**，它用$\lambda$平衡**短步TD和 长步 MC**。
2. **当$\lambda \to 0$**，GAE变成 1-step TD（计算快但偏差大）。
3. **当$\lambda \to 1$**，GAE变成 Monte Carlo（准确但方差大）。
4. 现实中，**$\lambda = 0.95$是一个常见的选择**，既减少了 MC 方法的方差，又降低了TD方法的偏差。

# 参考资料

===

* [GAE——泛化优势估计](https://zhuanlan.zhihu.com/p/356447099)
* [六、GAE广义优势估计](https://zhuanlan.zhihu.com/p/549145459)
* [【Typical RL 13】GAE](https://zhuanlan.zhihu.com/p/402198744)