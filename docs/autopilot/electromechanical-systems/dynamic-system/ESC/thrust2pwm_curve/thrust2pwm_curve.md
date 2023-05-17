# 螺旋桨推力与转速的关系

* [返回上层目录](../ESC.md)



螺旋桨推力thrust与转速pwm（假设转速正比于pwm）的关系为：
$$
\begin{aligned}
\text{thrust}&=k_1\times C_l \times \text{rpm}^2\\
&=k_2\times C_l \times \text{pwm}^2
\end{aligned}
$$
其中，$C_l$表示升力系数，$\sqrt{\frac{k_2}{k_1}}$为电调输入pwm和转速rpm的正比系数。

但有的电调比如T-model可能会做pwm到推力的线性化

下面是经验公式，实际上不符合物理模型，但是能用：


![pwm2thrust_curve](pic/pwm2thrust_curve.png)

我们需要一个参数a来对直线和平方曲线进行混合，并根据真实值进行曲线拟合，得到最适合的a，但一般来说，0.65适合大部分情况。

即令转速为`p`，推力为`t`，则有
$$
t=(1-a)\cdot p+a\cdot p^2
$$
现在我们想根据推力`t`来求转速`p`：
$$
\begin{aligned}
t&=(1-a)\cdot p+a\cdot p^2\\
&=a(p^2+\frac{1-a}{a}p)\\
&=a(p^2+2bp+b^2)-ab^2\quad \text{assum}\ b=\frac{1-a}{2a}\\
&=a(p+b)^2-ab^2
\end{aligned}
$$
从而
$$
\begin{aligned}
&t=a(p+b)^2-ab^2\\
\Rightarrow&a(p+b)^2=t+ab^2\\
\Rightarrow&p=\sqrt{\frac{t+ab^2}{a}}-b\\
\Rightarrow&p=\sqrt{\frac{t+a(\frac{1-a}{2a})^2}{a}}-\frac{1-a}{2a}\\
\Rightarrow&p=\sqrt{\frac{at+a^2(\frac{1-a}{2a})^2}{a^2}}-\frac{1-a}{2a}\\
\Rightarrow&p=\sqrt{\frac{at+\frac{(1-a)^2}{4}}{a^2}}-\frac{1-a}{2a}\\
\Rightarrow&p=\sqrt{\frac{4at+(1-a)^2}{4a^2}}-\frac{1-a}{2a}\\
\Rightarrow&p=\frac{\sqrt{4at+(1-a)^2}}{2a}+\frac{a-1}{2a}\\
\Rightarrow&p=\frac{a-1+\sqrt{(1-a)^2+4at}}{2a}\\
\end{aligned}
$$
即

![thrust2pwm_curve](pic/thrust2pwm_curve.png)

上图的python代码如下：

```python
import matplotlib.pyplot as plt
import math
import numpy as np

def apply_thrust_curve_and_volt_scaling(thrust, motors_thrust_curve_expo=0.65):
    motors_lift_max = 1.0
    a = (motors_thrust_curve_expo - 1.0) + math.sqrt((1.0 - motors_thrust_curve_expo)**2 + 4.0 * motors_thrust_curve_expo * thrust)
    b = 2.0 * motors_thrust_curve_expo
    throttle_ratio = a / b
    return throttle_ratio

thrust = np.linspace(0, 1, 1001)
throttle_ratio = [apply_thrust_curve_and_volt_scaling(i) for i in thrust]
sqrt_x = [math.sqrt(i) for i in thrust]

plt.figure()
plt.plot(thrust, sqrt_x, '--g', linewidth=3.0, label='$y=\sqrt{x}$')
plt.plot(thrust, thrust, '--b', linewidth=3.0, label='$y=x$')
LINE_NUM = 5
for i in range(LINE_NUM - 1):
    factor = (i + 1) / LINE_NUM
    thrust = np.linspace(0, 1, 1001)
    throttle_ratio = [apply_thrust_curve_and_volt_scaling(i, factor) for i in thrust]
    plt.plot(thrust, throttle_ratio, '--r')
plt.grid()
plt.xlabel("thrust")
plt.ylabel("throttle_ratio")
plt.title("$pwm=\sqrt{thrust}$")
plt.legend()  # plt.legend(loc='upper left')
plt.show()
```

猜测：

电调电压和转速线性相关？回答：并不线性相关，而是转速和转速平方的加权和。

# 台架实验

使用实验台架测试

电调ECS实时输出PWM值，其范围是$1000us\leqslant \text{ECS} \leqslant 2000us$。

电调ECS输出对应螺旋桨产生的推力是Thrust​，单位为kgf，1kgf表示一公斤质量产生的重力，其实就是9.8N。

电调的实时电压为Voltage，其单位是V。

就只测这三个值，测一组数据（最好能达到100组数据），如下图所示（只列出5个）：

| index | PWM of ECS (us) | Thrust (kgf) | Voltage (V) |
| ----- | --------------- | ------------ | ----------- |
| 1     | 1000            | 0.196        | 21.72       |
| 2     | 1118            | 0.245        | 21.72       |
| 3     | 1512            | 0.915        | 21.68       |
| 4     | 1923            | 2.028        | 21.55       |
| 5     | 2000            | 2.254        | 21.52       |

在计算之前，设定一些阈值

| 参数          | 值   |      | 推出的参数                          | 值          |
| ------------- | ---- | ---- | ----------------------------------- | ----------- |
| MOT_PWM_MIN   | 1000 |      | Data   Rows                         | 76          |
| MOT_PWM_MAX   | 2000 |      | PWM at MOT_SPIN_MIN                 | 1150        |
| MOT_THST_EXPO | 0.5  |      | PWM at MOT_SPIN_MAX                 | 1950        |
| MOT_SPIN_ARM  | 0.1  |      | Thrust at MOT_SPIN_MIN              | 0.271       |
| MOT_SPIN_MIN  | 0.15 |      | Thrust at MOT_SPIN_MAX(应当为scale) | 1.881383512 |
| MOT_SPIN_MAX  | 0.95 |      |                                     |             |

其中，左侧值是给定值，右侧值是根据左侧给定值和实验数据计算得到：

`PWM at MOT_SPIN_MIN`的值为
$$
\begin{aligned}
&\text{PWM_at_MOT_SPIN_MIN}\\
=&\text{PWM_MIN} + (\text{PWM_MIN}-\text{PWM_MIN})\cdot \text{SPIN_MIN}\\
=&1000+(2000-1000)*0.15\\
=&1150
\end{aligned}
$$
`PWM at MOT_SPIN_MAX`的值为
$$
\begin{aligned}
&\text{PWM_at_MOT_SPIN_MAX}\\
=&\text{PWM_MIN} + (\text{PWM_MIN}-\text{PWM_MIN})\cdot \text{SPIN_MAX}\\
=&1000+(2000-1000)*0.95\\
=&1950
\end{aligned}
$$
`Thrust at MOT SPIN_MIN`的值为

```e
=INDEX(B12:B1000, MATCH(SMALL(A12:A1000, COUNTIF(A12:A1000, "<"&$G$5)+1), A12:A1000, 0), 1)
```

该值的含义是，当`SPIN_MIN=0.15`，即`PWM at MOT_SPIN_MIN=1150`时，对应的推力值为0.271kgf。

`Thrust at MOT_SPIN_MAX`的值为1.881383512

其含义是。。。这个值应该是写错了。

1

（1）`Normalized Throttle`为归一化后的油门，归一化的同时，考虑了压降，其定义为
$$
\begin{aligned}
\text{throttle}_{\text{norm}}&=\frac{V}{V_{\text{max}}}max(0, \frac{\text{PWM}-\text{PWM}_{\text{spin_min}}}{\text{PWM}_{\text{spin_max}}-\text{PWM}_{\text{spin_min}}})\\
&=\frac{21.68}{21.72}max(0, \frac{1555-1150}{1950-1150})\\
&=0.9982\times 0.5063\\
&=0.5053
\end{aligned}
$$

（2）`Normalized Thrust`为由归一化后的油门计算得到的推力，其定义为
$$
\begin{aligned}
\text{thrust}_{\text{norm}}&=max((1-\text{EXPO})\cdot \text{throttle}_{\text{norm}} + \text{EXPO}\cdot \text{throttle}_{\text{norm}}^2,0)\\
&=(1-0.5)\times 0.5053 + 0.5\times 0.5053^2\\
&=0.25265+0.12766\\
&=0.3803
\end{aligned}
$$
**注意，该变量不是测出来的推力值，而是根据油门值估算出的推力值**。

（3）`Rescaled Thrust`为
$$
\begin{aligned}
\text{thrust}_{\text{rescaled}}&=\frac{\text{thrust}_{\text{real-kgf}}-{\text{thrust}_{\text{spin_min}}}}{\text{thrust}_{\text{norm}}}\\
&=\frac{1.039-0.271}{0.3803}\\
&=2.0193
\end{aligned}
$$
这其实是，真实推力值和估算推力值的比值。

（4）`Corrected Thrust`为修正后的推力，其定义为
$$
\begin{aligned}
\text{thrust}_{\text{corrected}}&=\text{thrust}_{\text{norm}}\cdot mean(\text{thrust}_{\text{rescaled}}) + \text{thrust}_{\text{spin_min}}\\
&=\text{thrust}_{\text{norm}}\cdot mean(\frac{\text{thrust}_{\text{real-kgf}}-{\text{thrust}_{\text{spin_min}}}}{\text{thrust}_{\text{norm}}}) + \text{thrust}_{\text{spin_min}}
\end{aligned}
$$
下面这个xlsx可以自动计算thrust_curve_expo值。

[原版ArduPilot Motor Thrust Fit.xlsx](https://docs.google.com/spreadsheets/d/1_75aZqiT_K1CdduhUe4-DjRgx3Alun4p8V2pt6vM5P8/edit#gid=0)

[我改进的ArduPilot Motor Thrust Fit.xlsx](https://docs.google.com/spreadsheets/d/1_l8iOQ7tlthABDotDbcDRmZIHLIVbNOiLjNEUL10f0M/edit#gid=0)

改进的点：

G7处改为了：

```
=INDEX(B12:B1000, MATCH(SMALL(A12:A1000, COUNTIF(A12:A1000, "<"&$G$5)+1), A12:A1000, 0), 1)
```

# 参考资料

* [Using “measured” MOT_THST_EXPO: What improvement can one expect?](https://discuss.ardupilot.org/t/using-measured-mot-thst-expo-what-improvement-can-one-expect/26172)

这里作者讲了对thrust_curve_expo的理解。

> 这是调整飞机的一个极其重要的参数。大多数 ESC 的工作原理是指令输入与用于驱动电机的平均电压大致成正比。这导致每个螺旋桨的强烈非线性推力响应。我使用大约 100 个螺旋桨的测量值得出默认的 MOT_THST_EXPO 值 0.65。基本原则是道具越大价值越高。因此，对于普通 ESC，30" 螺旋桨将更接近 0.8，而 5" 螺旋桨将更接近 0.5。同轴螺旋桨似乎也更低，所以如果我没有测量推力特性，我对大多数同轴设计使用 0.5。
>
> 但是，某些 ESC 似乎会对命令输入进行补偿以尝试使推力值线性化。根据 ESC 和螺旋桨的不同，MOT_THST_EXPO 可以在 -0.3 到 0.5 之间变化。
>
> 另一个改变 MOT_THST_EXPO 值的参数是 MOT_SPIN_MIN 值。因此，如果您增加 MOT_SPIN_MIN，则需要减少完美的 MOT_THST_EXPO。
>
> 因此，正确定义此参数的唯一方法是测量推力曲线并计算 MOT_THST_EXPO。然而，对于大多数 ESC 和螺旋桨组合，默认值会很好地工作。
>
> 那么如何判断这个值是否错误呢？
> 全油门时会出现振荡，或者在低油门时控制很差。
> 您发现如果不重新进行调整就无法更改飞机的起飞重量。
>
> 这两个都假设你在悬停时有一个近乎完美的曲调（我的意思是我对一个近乎完美的曲调的想法）。您还应该在 MOT_ 参数中设置电压补偿功能。
>
> 所以给你一些我调过的飞机的例子。我有悬停油门小于 15% 的飞机，在全油门下不会显示任何振荡迹象，在低油门设置下不会调整或妥协 PID。我的飞机起飞重量可以在 12.5 到 50 公斤之间变化，而无需调整 PID 值。
>
> 我希望这能回答你的问题。

* [MOT_THST_EXPO and need for thrust stand](https://discuss.ardupilot.org/t/mot-thst-expo-and-need-for-thrust-stand/78258)

下面这个xlsx可以自动计算thrust_curve_expo值。

[ArduPilot Motor Thrust Fit.xlsx](https://docs.google.com/spreadsheets/d/1_75aZqiT_K1CdduhUe4-DjRgx3Alun4p8V2pt6vM5P8/edit#gid=0)

* [PX4调参 MC PID Tuning Guide](https://www.bookstack.cn/read/px4-user-guide/zh-config_mc-pid_tuning_guide_multicopter.md#%E6%8E%A8%E5%8A%9B%E6%9B%B2%E7%BA%BF%20/%20%E5%A4%A7%E6%B2%B9%E9%97%A8%20PID%20%E8%A1%B0%E5%87%8F(Throttle%20PID%20Attenuation)%20{#thrust_curve})

这一章：推力曲线 / 大油门 PID 衰减(Throttle PID Attenuation) {#thrust_curve}

> 用 **THR_MDL_FAC**参数来调整[推力曲线](https://www.bookstack.cn/read/px4-user-guide/zh-advanced_config-parameter_reference.md#THR_MDL_FAC)(推荐的方式)。 默认情况下的PWM - 推力 对应关系是线性的。 — 你可以把参数`THR_MDL_FAC`设为1来让这种关系变成二次的。 0~1之间的值表示线性和二次之间的一个插值。 这个参数一般在0.3~0.5之间，你可以每次增加0.1。 如果该参数太大，你可以看到低油门下的振荡现象。

===

* [Copter: Voltage compensation to throttle/pwm is used in many places in motor output function, is it used repeatedly? #23575](https://github.com/ArduPilot/ardupilot/issues/23575)

这个Issue详细分析了曲线的来由和电机动力学模型。

