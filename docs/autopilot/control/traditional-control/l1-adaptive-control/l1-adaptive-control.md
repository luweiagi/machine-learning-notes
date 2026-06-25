下面是一篇偏“从直觉到工程”的 L1 Adaptive Control 博客，我按：

- 为什么需要 L1
- 从刚体动力学开始
- 如何得到状态空间模型
- L1 的本质
- 为什么需要低通滤波
- 工程实现
- 与 ADRC / DOB / INDI 的关系
- 实际问题与调参

这样的逻辑来组织。

我尽量避免“论文味太重”，而是偏向：

```
真正让人“理解”L1
```

而不是：

```
只会背公式
```

# L1 Adaptive Control

* [返回上层目录](../traditional-control.md)

我尽量避免“论文味太重”，而是偏向：真正让人“理解”L1，而不是只会背公式。

# 前言

在无人机控制领域，很多人第一次接触：

- L1 Adaptive Control
- ADRC
- INDI
- Disturbance Observer

时，都会感觉这些控制方法好高级。

但当真正理解它们后，你会发现：

> 它们的核心思想，其实非常朴素。

本文会从：

- 刚体动力学
- 状态空间模型
- 扰动估计
- 自适应补偿

一步一步推导出 L1 Adaptive Control 的本质

并最终给出：

- 工程实现思路
- Python 代码
- 调参方法
- 与 ADRC 的关系

本文尽量避免上来就是 Lyapunov，而是从**物理直觉**开始。

------

# 为什么PID不够？

现代无人机几乎都在使用 PID。

例如：

```
位置环
    ↓
速度环
    ↓
姿态角环
    ↓
角速度环（PID）
```

PID 非常强大。

尤其：

- 简单
- 稳定
- 易实现
- 易调参

但 PID 有一个根本问题：

PID 默认认为系统模型基本不变

但真实世界并不是。

例如：

- 风扰
- 挂载变化
- 电池电压变化
- 电机老化
- 重心偏移
- 空气动力变化

这些都会导致真实系统$\neq$理论模型，于是 PID 的性能会下降。

# 公式推导

## 刚体旋转动力学

我们以无人机 roll 轴为例。

定义：`p = roll angular rate`（横滚角速度），单位：rad/s

刚体旋转最基础公式：
$$
\tau = I \dot p
$$
其中：

- $\tau$：力矩
- $I$：转动惯量
- $\dot p$：角加速度

整理：
$$
\dot p = \frac{\tau}{I}
$$
这意味着：角加速度 = 力矩 / 转动惯量

## 控制输入从哪里来？

无人机通过电机推力差产生roll torque

于是：
$$
\tau_{control} = ku
$$
其中：

- $u$：控制输入
- $k$：控制效能

代入：
$$
\dot p = \frac{k}{I}u
$$
定义：
$$
b = \frac{k}{I}
$$
得到：
$$
\dot p = bu
$$

## 为什么还会有ap？

真实世界里角速度不会永远保持，因为存在：

- 空气阻尼
- 气动力
- 摩擦
- 陀螺耦合

这些会导致系统自然衰减，于是加入阻尼项：
$$
\tau_{damping} = -cp
$$
代入：
$$
\dot p = -\frac{c}{I}p + \frac{k}{I}u
$$
定义：
$$
a=-\frac{c}{I}
$$
最终：
$$
\dot p = ap + bu
$$

## 真实世界永远不完美

但现实中还有：

- 风
- 未建模气动力
- 电机误差
- 推力波动
- 重心变化

这些很难精确建模。

于是统一写成 $d(t)$，最终得到：

## 无人机控制中最经典的形式

$$
\dot p = ap + bu + d(t)
$$

这句话翻译成人话就是：

```
当前角加速度 dot(p)
=
系统自身动态 ap
+
控制输入造成的变化 bu
+
未知扰动 d(t)
```

# L1 Adaptive 的核心思想

现在问题来了。

传统 PID 默认认为：$d(t)=0$

但现实世界：扰动永远存在

于是 L1 Adaptive 的核心思想就是：

**在线估计 d(t) 并实时补偿它**。

## 如何估计扰动？

从：
$$
\dot p = ap + bu + d(t)
$$
直接移项：
$$
d(t)=\dot p-ap-bu
$$
这一步非常关键。

它意味着：**如果我们知道当前角加速度，就能反推出未知扰动。**

## 如何得到dot(p)？

IMU 通常只能直接测量 p（角速度）

而不能直接测 p_dot（角加速度）

于是：

工程中通常使用：
$$
\dot p \approx \frac{p_k-p_{k-1}}{dt}
$$
即：

```python
p_dot_est = (p - prev_p) / dt
```

## 扰动估计

于是：

```python
d_hat = (
    p_dot_est
    - a * p
    - b * u
)
```

其中：`d_hat ` 就是 估计出来的未知扰动。

## 然后补偿扰动

如果风在向左吹，那我们就额外向右补一点控制。于是：
$$
u = u_{pid} + u_{L1}
$$
其中：
$$
u_{L1} \approx -d_{hat}
$$
即：估计到多少扰动，就反向补偿多少

## 但系统会立刻爆炸

因为：
$$
\dot p \approx \frac{p_k-p_{k-1}}{dt}
$$
这个微分非常 noisy。IMU 本身就有噪声。微分后：高频噪声会被无限放大

于是 d_hat 会疯狂抖动，最终控制输入疯狂震荡，飞机直接炸。

## L1 的真正核心：低通滤波器

很多人误以为 L1 的核心是 adaptive law，其实：

L1 最关键的是低通滤波器。

L1 的真正思想是：估计可以很快，但控制不能太快

于是：

```
adaptive estimate
    ↓
low-pass filter
    ↓
control input
```

例如：

```
alpha = dt / (tau + dt)

d_hat_filtered = (
    (1 - alpha) * d_hat_filtered
    + alpha * d_hat
)
```

然后：

```
u_l1 = -k_l1 * d_hat_filtered
```

------

# 14. 完整工程版代码

下面给出一个非常工程化的：

# PID + L1

示例。

```python
class RateControllerL1:

    def __init__(self):

        # PID
        self.kp = 0.2
        self.ki = 0.05
        self.kd = 0.01

        # nominal model
        self.a = -2.0
        self.b = 1.0

        # L1
        self.k_l1 = 0.8

        # low-pass filter
        self.tau = 0.03

        self.integrator = 0.0

        self.prev_error = 0.0
        self.prev_rate = 0.0

        self.d_hat_filtered = 0.0

        self.u_prev = 0.0

    def update(self, rate_cmd, rate, dt):

        # ==========================================
        # PID
        # ==========================================

        error = rate_cmd - rate

        self.integrator += error * dt

        derivative = (
            error - self.prev_error
        ) / dt

        u_pid = (
            self.kp * error
            + self.ki * self.integrator
            + self.kd * derivative
        )

        self.prev_error = error

        # ==========================================
        # disturbance estimation
        # ==========================================

        rate_dot_est = (
            rate - self.prev_rate
        ) / dt

        d_hat = (
            rate_dot_est
            - self.a * rate
            - self.b * self.u_prev
        )

        self.prev_rate = rate

        # ==========================================
        # low-pass filter
        # ==========================================

        alpha = dt / (self.tau + dt)

        self.d_hat_filtered = (
            (1 - alpha) * self.d_hat_filtered
            + alpha * d_hat
        )

        # ==========================================
        # L1 compensation
        # ==========================================

        u_l1 = -self.k_l1 * self.d_hat_filtered

        # ==========================================
        # final control
        # ==========================================

        u = u_pid + u_l1

        self.u_prev = u

        return u
```

------

# 15. L1 的本质到底是什么？

很多人会觉得：

```
L1 很高级
```

但真正本质其实非常简单：

# “模型预测”

和：

# “真实系统响应”

之间的误差补偿。

即：

```
理论应该怎么运动
vs
实际怎么运动
```

它们之间的差值：

```
就是 disturbance
```

L1 做的事情：

就是：

```
在线估计这个差值
并实时补偿
```

------

# 16. 为什么 L1 不需要特别精确模型？

L1 的一个巨大优势：

是：

```
nominal model 不需要特别精确
```

例如：

真实系统：

```
非常复杂
```

但模型：

p˙=−2p+u\dot p = -2p + up˙=−2p+u

也可能已经够用了。

因为：

```
模型解释不了的部分
全部会被吸收到 disturbance
```

------

# 17. 但这不意味着可以乱填参数

例如：

b=kIb=\frac{k}{I}b=Ik

这里：

- III：转动惯量
- kkk：控制效能

如果：

```
b 差一个数量级
```

系统仍然会炸。

所以：

L1 并不是：

```
完全不要模型
```

而是：

```
允许模型不精确
```

------

# 18. 工程上真的会做 system identification 吗？

理论上：

会。

但现实中：

很多无人机系统其实是：

```
rough estimate
+
flight tuning
+
adaptive compensation
```

即：

```
大概模型
+
试飞调参
```

而不是：

```
严格系统辨识
```

------

# 19. 为什么现代控制越来越强调“扰动估计”？

因为真实世界：

```
永远不可能完全建模
```

于是：

现代控制越来越强调：

```
在线估计未知部分
```

例如：

- L1 Adaptive
- ADRC
- DOB
- INDI
- ESO

虽然名字不同。

但本质都非常像：

# “估计系统没建模的部分，并补偿它”

------

# 20. ADRC 和 L1 的区别

两者都在做：

```
disturbance estimation + compensation
```

但：

## L1 更偏：

```
adaptive control
```

强调：

- adaptive law
- reference model
- Lyapunov

------

## ADRC 更偏：

```
observer
```

尤其：

# ESO（Extended State Observer）

ADRC 的思想更加激进：

```
“所有未知东西，
统统打包成 total disturbance”
```

然后：

实时观测并补偿。

------

# 21. 为什么 PID 仍然统治工业？

因为现实世界里：

- 噪声
- 延迟
- 未建模动态
- 共振
- 传感器误差

太复杂。

很多 fancy 控制：

```
理论很强
```

但：

```
工程容易翻车
```

于是最终：

很多工业系统都会变成：

```
PID
+
filter
+
feedforward
+
disturbance compensation
```

已经非常强。

------

# 22. 总结

L1 Adaptive 的真正核心不是：

```
复杂数学
```

而是：

# “实时估计模型解释不了的部分”

然后：

# “实时补偿这个差值”

它并不是：

```
完全不要模型
```

而是：

```
允许模型不完美
```

并通过：

```
在线扰动估计
```

提升系统鲁棒性。

这也是现代控制越来越重要的发展方向。



# 参考资料

## 大模型

[https://chatgpt.com/c/6a01bd65-9a84-83ec-a067-77e7d2e6774f](https://chatgpt.com/c/6a01bd65-9a84-83ec-a067-77e7d2e6774f)