# 多旋翼底层电机混控深度解析——从角速率环输出到电调输入的最后一公里

* [返回上层目录](../control-allocation.md)
* [全链路鸟瞰：控制量的一生](#全链路鸟瞰控制量的一生)
* [预备知识：混控的物理基础](#预备知识混控的物理基础)
* [混控函数逐段精讲](#混控函数逐段精讲)
* [统一数学定理：整个混控就是一次"窗口夹紧"](#统一数学定理整个混控就是一次窗口夹紧)
* [十个直击灵魂的深度问答](#十个直击灵魂的深度问答)
* [抗饱和闭环：limit标志如何反哺上层控制器](#抗饱和闭环limit标志如何反哺上层控制器)
* [混控之后：从推力到PWM](#混控之后从推力到pwm)
* [工程验证方法论：如何证明混控代码没有bug](#工程验证方法论如何证明混控代码没有bug)
* [总结：十条设计哲学](#总结十条设计哲学)

本文以 ArduPilot 开源飞控（`AP_MotorsMatrix::output_armed_stabilizing()`，下称"混控函数"）为分析对象，完整拆解多旋翼飞控中最底层、最安全攸关的一环：**姿态控制器算出的三轴力矩指令和油门指令，是如何被分配到每一个电机上的**。

这段代码不到 200 行，却浓缩了十几年全球机队迭代出来的饱和处理哲学。它出错的代价是飞机从空中掉下来，所以本文不满足于"讲清楚流程"，而是要做到三件事：

**① 每个分支给出严格的数学证明；② 每个设计给出物理直觉；③ 每个取舍给出工程业务上的为什么。**

全文代码均引自 ArduPilot 官方源码（`libraries/AP_Motors/` 与 `libraries/AC_AttitudeControl/`）。

# 全链路鸟瞰：控制量的一生

在深入混控函数之前，先建立全局图景。以自稳模式为例，一个控制量从传感器到电调走过如下链路（括号内为 ArduPilot 中的实际函数）：

```
陀螺仪角速度                 飞手杆量
     │                         │
     ▼                         ▼
┌─────────────────────────────────────┐
│ 角度环 P 控制（attitude_controller）│  期望角度 → 期望角速度
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ 角速率环 PID（rate_controller_run） │  期望角速度 → 无量纲力矩指令 [-1, +1]
│   _motors.set_roll / pitch / yaw    │
│   _motors.set_throttle              │  高度环/飞手 → 无量纲油门指令 [0, 1]
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ ★ 混控 output_armed_stabilizing ★  │  4 个标量指令 → N 个电机推力 [0, 1]
│   （本文主角：推力分配与饱和仲裁） │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ 推力线性化 thrust_to_actuator       │  推力 [0,1] → 电机油门比 [spin_min, spin_max]
│   （expo 曲线 + 电压补偿）          │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ PWM 输出（output_to_motors → hal）  │  油门比 → 1000~2000us 脉宽 → 电调
└─────────────────────────────────────┘
```

角速率环的出口代码如下（`AC_AttitudeControl_Multi::rate_controller_run_dt()`），注意 PID 的输出直接塞给了 motors 对象，**并把** `_motors.limit` **标志作为参数传回 PID 做抗积分饱和**——这个闭环我们在第 6 节展开：

```cpp
_motors.set_roll(get_rate_roll_pid().update_all(ang_vel_body.x, gyro_rads.x, dt, _motors.limit.roll, ...));
_motors.set_pitch(get_rate_pitch_pid().update_all(ang_vel_body.y, gyro_rads.y, dt, _motors.limit.pitch, ...));
_motors.set_yaw(get_rate_yaw_pid().update_all(ang_vel_body.z, gyro_rads.z, dt, _motors.limit.yaw, ...));
```

**混控函数的任务一句话概括**：把 `roll_in / pitch_in / yaw_in ∈ [-1, +1]` 和 `throttle_in ∈ [0, 1]`
这 4 个标量，变成 N 个电机各自的推力 `_thrust_rpyt_out[i] ∈ [0, 1]`——并且当 4 个需求加起来超出电机物理量程时，**按明确的优先级做出仲裁**。

这个优先级是全文的纲：

$$\textbf{滚转/俯仰} \gg \textbf{偏航} \gg \textbf{高度（油门）}$$

为什么是这个顺序？因为多旋翼的一切升力都建立在姿态之上：姿态翻了，油门越大砸得越快；而偏航丢了只是机头暂时偏转，高度丢了只是短暂下沉，都可以在姿态恢复后立即补救。

# 预备知识：混控的物理基础

## 电机的物理量程：[0,1] 的残酷性

混控中每个电机的推力被归一化到 $[0.0,\ 1.0]$：

- **下限 0.0 是硬边界**：普通电调不能反转，负推力做不出来。一旦算法算出负数，只能截断，截断就意味着实际力矩和期望力矩不一致——**力矩失真**；
- **上限 1.0 是硬边界**：电调给到 100% 就是全部，算出 1.2 也只能输出 1.0，同样力矩失真。

**多旋翼产生力矩靠的是电机之间的推力差**。当一侧电机顶在 1.0、另一侧踩在 0.0 时，这个构型物理上能输出的力矩已经到极限了，任何算法都变不出更多。混控的全部智慧，就是在这个残酷的 $[0,1]$ 盒子里做最优分配。

## 混控矩阵与因子归一化：那个神秘的0.5

每个电机对三轴的贡献由几何位置决定。以 X 型四旋翼为例，电机 i 的滚转因子是$-\sin(\theta_i)$、俯仰因子是 $\cos(\theta_i)$（$\theta_i$ 为电机相对机头的方位角），偏航因子由桨的旋向决定（CCW 为 +1，CW 为 -1）。

物理直觉（"知人善任"原理）：离滚转轴（X 轴）越远的电机，滚转力臂越长，产生滚转力矩效率越高，所以分到的滚转因子绝对值越大。让每个电机干它力臂上最擅长的事，才能用最小的推力变化换最大的力矩。

注册完电机后，ArduPilot 会做一次归一化（`AP_MotorsMatrix::normalise_rpy_factors()`）：

```cpp
// scale factors back to -0.5 to +0.5 for each axis
for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
    if (motor_enabled[i]) {
        if (!is_zero(roll_fac)) {
            _roll_factor[i] = 0.5f * _roll_factor[i] / roll_fac;
        }
        if (!is_zero(pitch_fac)) {
            _pitch_factor[i] = 0.5f * _pitch_factor[i] / pitch_fac;
        }
        if (!is_zero(yaw_fac)) {
            _yaw_factor[i] = 0.5f * _yaw_factor[i] / yaw_fac;
        }
    }
}
```

**为什么归一化到 ±0.5 而不是 ±1.0？** 这是很多人第一次读代码时的困惑。答案藏在量程对齐里：

- 角速率 PID 的输出范围是 $[-1, +1]$（满量程指令）；
- 电机推力量程是 $[0, 1]$，绕 0.5 的中点油门上下各只有 **0.5 的行程**；
- 若因子是 ±1.0，PID 输出 ±1 时单轴差速需求就是 ±1.0，在 0.5 油门下电机需要输出 $0.5 \pm 1.0 = [-0.5, 1.5]$，瞬间双向饱和，PID 的线性区间只剩一半，控制器整天在饱和区打滚；
- 归一化到 ±0.5 后，PID 满量程输出对应的差速恰好是 ±0.5，$0.5 \pm 0.5 = [0, 1]$，**PID 的满量程恰好映射到电机的满行程**，线性区最大化。

另一个重要副作用：不管是四轴、六轴还是八轴，归一化保证了"同样的 PID 输出产生同样的角加速度量级"，**PID 参数在不同构型间具有可迁移性**。

对称构型还有一个贯穿全文的关键性质：**每一轴的因子列求和为零**（$\sum_i \text{roll\_factor}[i] = \sum_i \text{pitch\_factor}[i] = \sum_i \text{yaw\_factor}[i] = 0$）。

它意味着：差速指令不改变全机总升力，只改变分布——这是后文所有"平均油门"推导的基石。

## 电压与气压补偿：get_compensation_gain

电池电压下降、海拔升高都会让同样的 PWM 输出更小的推力。ArduPilot 在混控入口把所有指令乘一个补偿增益
（`Thrust_Linearization::get_compensation_gain()`）：

```cpp
float Thrust_Linearization::get_compensation_gain() const
{
    if (get_lift_max() <= 0.0) {
        return 1.0;
    }
    float ret = 1.0 / get_lift_max();   // lift_max 由滤波后的电池电压算出

#if AP_MOTORS_DENSITY_COMP == 1
    const float air_density_ratio = AP::ahrs().get_air_density_ratio();
    if (air_density_ratio > 0.3 && air_density_ratio < 1.5) {
        ret *= 1.0 / constrain_float(air_density_ratio, 0.5, 1.25);
    }
#endif
    return ret;
}
```

直觉：电压掉了 20%，就把指令放大 25%，让"指令 × 实际推力系数"保持不变，
PID 眼中的被控对象增益不随电压漂移——这是保证调好的 PID 全电压域都稳定的关键。

## throttle_avg_max：藏在上层的"下降上限"

`_throttle_avg_max` 由姿态控制器每拍设置（`AC_AttitudeControl_Multi`）：

```cpp
_motors.set_throttle_avg_max(get_throttle_avg_max(MAX(throttle_in, _throttle_in)));

float AC_AttitudeControl_Multi::get_throttle_avg_max(float throttle_in)
{
    throttle_in = constrain_float(throttle_in, 0.0f, 1.0f);
    return MAX(throttle_in,
               throttle_in * MAX(0.0f, 1.0f - _throttle_rpy_mix)
             + _motors.get_throttle_hover() * _throttle_rpy_mix);
}
```

翻译成数学（$mix = \text{throttle\_rpy\_mix}$，默认约 0.5）：

$$avg\_max = \max\big(throttle,\ \ throttle \cdot (1-mix) + hover \cdot mix\big)$$

它的物理含义是：**"为了保姿态，允许混控把全机平均油门最多抬到这里"**。

- 油门高于悬停时：$avg\_max = throttle$，mix 完全不起作用（不需要额外借空间）；
- 油门低于悬停时：$avg\_max$ 是飞手油门和悬停油门的加权折中——**严格小于悬停油门**。
这一条保证了：即使混控为保姿态把油门从飞手值往上抬，抬到 $avg\_max$ 也仍然小于悬停油门，
**飞机必然处于净下降状态，绝不会违抗飞手的下降指令上浮**。

mix 本身还有一个精妙的动态调节（`update_throttle_rpy_mix()`）：**升得快（0.2\~0.4 秒内从 0.1 升到 0.5\~0.9）、
降得慢（1.6 秒）**——起飞/姿态告急时立刻扩权，降落/危机解除后慢慢还权，避免油门权限来回振荡。

# 混控函数逐段精讲

下面按执行顺序拆解 `AP_MotorsMatrix::output_armed_stabilizing()`。
（为聚焦主线，略去 thrust_boost 失效电机处理的分支——四旋翼自由度不够，本来也用不上它。）

## 输入采集与入口钳制

```cpp
const float compensation_gain = thr_lin.get_compensation_gain();
const float roll_thrust  = (_roll_in  + _roll_in_ff)  * compensation_gain;
const float pitch_thrust = (_pitch_in + _pitch_in_ff) * compensation_gain;
float yaw_thrust         = (_yaw_in   + _yaw_in_ff)   * compensation_gain;
float throttle_thrust    = get_throttle() * compensation_gain;
float throttle_avg_max   = _throttle_avg_max * compensation_gain;

// sanity check throttle is above zero and below current limited throttle
if (throttle_thrust <= 0.0f) {
    throttle_thrust = 0.0f;
    limit.throttle_lower = true;
}
if (throttle_thrust >= _throttle_thrust_max) {
    throttle_thrust = _throttle_thrust_max;
    limit.throttle_upper = true;
}

// ensure that throttle_avg_max is between the input throttle and the maximum throttle
throttle_avg_max = constrain_float(throttle_avg_max, throttle_thrust, _throttle_thrust_max);
```

三个要点：

1. `roll/pitch` 是 `const`，之后再也不会被修改；`yaw_thrust` 不是——**偏航是唯一会被中途削减的姿态轴**（3.2 节）；
2. `_throttle_thrust_max` 是电流限制/起转斜坡给出的当前允许最大油门；
3. 最后一行保证了一个全文反复使用的不等式：$throttle\_thrust \le throttle\_avg\_max$。
  **记住它，第 5 节的多个证明都靠它。**

## 第一阶段：roll/pitch先上车，yaw按剩余空间限量

```cpp
// throttle providing maximum roll, pitch and yaw range
float throttle_thrust_best_rpy = MIN(0.5f, throttle_avg_max);

float yaw_allowed = 1.0f;
for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
    if (motor_enabled[i]) {
        // calculate the thrust outputs for roll and pitch
        _thrust_rpyt_out[i] = roll_thrust * _roll_factor[i] + pitch_thrust * _pitch_factor[i];

        if (!is_zero(_yaw_factor[i])) {
            const float thrust_rp_best_throttle = throttle_thrust_best_rpy + _thrust_rpyt_out[i];
            float motor_room;
            if (is_positive(yaw_thrust * _yaw_factor[i])) {
                motor_room = 1.0 - thrust_rp_best_throttle;   // 该电机到天花板的空间
            } else {
                motor_room = thrust_rp_best_throttle;         // 该电机到地板的空间
            }
            const float motor_yaw_allowed = MAX(motor_room, 0.0) / fabsf(_yaw_factor[i]);
            yaw_allowed = MIN(yaw_allowed, motor_yaw_allowed);
        }
    }
}
```

这段体现了优先级的第一层：**roll/pitch 无条件全额分配，yaw 只能吃剩饭**。

逐行拆解：

- `throttle_thrust_best_rpy = MIN(0.5, avg_max)` 是一个**预估基准油门**：

  油门高时取 0.5（居中，上下空间都最大）；油门低时取 $avg\_max$（不违抗下降指令的前提下尽量给姿态腾空间）。

  注意它只是**第一阶段的预估值**，第三阶段会被精确重算——两次计算目的不同：

  这里是"假设油门在这，看 yaw 还能塞多少"，后面是"yaw 塞进来后，精确定位基准"；

  （**为什么这里写死 0.5 而不是当前机型的悬停油门 hover？这是理解本行的关键，详见 [Q7](#q7为什么阶段-1-预估偏航时基准油门写死-05-而不是-hover几何空间-vs-气动平衡点与安全不对称性哲学)**）；

- 每个电机分别算"这台电机还能容纳多少 yaw"：yaw 会把这台电机往上推（`yaw_thrust * _yaw_factor[i] > 0`）

  就看到天花板的距离，往下拉就看到地板的距离，再除以偏航因子换算成 yaw 指令量；

- `yaw_allowed` 取全部电机的最小值——**木桶原理，最紧的那台电机决定全机 yaw 权限**。

然后是 yaw 保底空间（headroom）：

```cpp
float yaw_allowed_min = (float)_yaw_headroom * 0.001f;   // MOT_YAW_HEADROOM 参数，默认 50~200
yaw_allowed = MAX(yaw_allowed, yaw_allowed_min);

if (fabsf(yaw_thrust) > yaw_allowed) {
    yaw_thrust = constrain_float(yaw_thrust, -yaw_allowed, yaw_allowed);
    limit.yaw = true;
}
```

`MOT_YAW_HEADROOM` 是一个纯粹的**业务权衡参数**（详见 [Q10](#q10偏航保底裕度-mot_yaw_headroom-设为-0-好还是-50-好为什么不能设为-0偏航自旋死锁与陀螺力矩耦合链)）：

- 设大（如原生默认 200 → 保底 0.2）：即使 roll/pitch 打满，yaw 也总能保住 0.2 的差速权限，机头指向极佳——代价是极端时 yaw 会反过来挤占 roll/pitch 行程；
- 设 50（保底 5%）：仅消耗 4.8% 姿态力矩，换取强劲的反向阻尼刹车片，彻底掐断陀螺耦合自旋死锁；
- 设 0：完全不留保底，极端时全机失去偏航阻尼，存在自旋死锁风险。

被削掉的 yaw 通过 `limit.yaw = true` 通知偏航角速率 PID 冻结积分（第 6 节）。

## 第二阶段：加入yaw，统计全机峰谷

```cpp
float rpy_low = 1.0f;    // lowest thrust value
float rpy_high = -1.0f;  // highest thrust value
for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
    if (motor_enabled[i]) {
        _thrust_rpyt_out[i] = _thrust_rpyt_out[i] + yaw_thrust * _yaw_factor[i];

        if (_thrust_rpyt_out[i] < rpy_low)  { rpy_low  = _thrust_rpyt_out[i]; }
        if (_thrust_rpyt_out[i] > rpy_high) { rpy_high = _thrust_rpyt_out[i]; }
    }
}
```

至此每台电机上的三轴差速需求齐了。$rpy\_low \le 0 \le rpy\_high$（因子列和为零 $\implies$ 差速有正必有负），后面所有推导只需要这两个数和跨度：

$$S = rpy\_high - rpy\_low \quad (\text{全机差速峰峰跨度})$$

## 第三阶段：两大约束压缩 + 贴底基准 + 油门补偿（全函数的心脏）

```cpp
// calculate any scaling needed to make the combined thrust outputs fit within the output range
float rpy_scale = 1.0f;
if (rpy_high - rpy_low > 1.0f) {
    rpy_scale = 1.0f / (rpy_high - rpy_low);            // 约束 1：跨度压缩
}
if (throttle_avg_max + rpy_low < 0) {
    rpy_scale = MIN(rpy_scale, -throttle_avg_max / rpy_low);   // 约束 2：下降上限
}

// calculate how close the motors can come to the desired throttle
rpy_high *= rpy_scale;
rpy_low *= rpy_scale;
throttle_thrust_best_rpy = -rpy_low;                    // 贴底基准
float thr_adj = throttle_thrust - throttle_thrust_best_rpy;
if (rpy_scale < 1.0f) {
    // Full range is being used by roll, pitch, and yaw.
    limit.set_rpy(true);
    if (thr_adj > 0.0f) {
        limit.throttle_upper = true;
    }
    thr_adj = 0.0f;
} else if (thr_adj < 0.0f) {
    // Throttle can't be reduced to desired value
    // todo: add lower limit flag and ensure it is handled correctly in altitude controller
    thr_adj = 0.0f;
} else if (thr_adj > 1.0f - (throttle_thrust_best_rpy + rpy_high)) {
    // Throttle can't be increased to desired value
    thr_adj = 1.0f - (throttle_thrust_best_rpy + rpy_high);
    limit.throttle_upper = true;
}

// add scaled roll, pitch, constrained yaw and throttle for each motor
const float throttle_thrust_best_plus_adj = throttle_thrust_best_rpy + thr_adj;
for (uint8_t i = 0; i < AP_MOTORS_MAX_NUM_MOTORS; i++) {
    if (motor_enabled[i]) {
        _thrust_rpyt_out[i] = (throttle_thrust_best_plus_adj * _throttle_factor[i])
                            + (rpy_scale * _thrust_rpyt_out[i]);
    }
}
```

逐个概念精讲：

**约束 1（保物理量程）**：若跨度 $S > 1$，差速需求超过了电机全行程，按 $rpy\_scale = 1/S$ 等比压缩，压缩后跨度恰好为 1.0——最低电机贴地板 0.0，最高电机贴天花板 1.0，**上下两端同时精确贴合，不需要任何硬削波，正负力矩比例完全保真**。

**约束 2（保操控意图）**：条件 `throttle_avg_max + rpy_low < 0` 移项即 $|rpy\_low| > avg\_max$——姿态需要的贴底基准（见下）超过了允许抬到的上限。此时按 $rpy\_scale = avg\_max / |rpy\_low|$ 进一步压缩，压缩后 $|rpy\_low'|$ 恰好等于 $avg\_max$。**宁可削姿态力矩，也不违抗下降指令往上蹿。**

两个约束都是 $rpy\_scale$ 的上界，取 min 同时满足。注意一个容易混淆的细节：

进入约束 2 的 `if` 的门槛是 $avg\_max < |rpy\_low|$，但约束 2 真正在 min 中**胜出**

（压得比约束 1 还狠）的条件是 $\frac{avg\_max}{|L|} < \frac{1}{S}$，即 $avg\_max < \frac{|L|}{S}$。

门槛和胜出是两回事。

**贴底基准（`throttle_thrust_best_rpy = -rpy_low`）**：把全机基准油门锚定在压缩后的最低点上，于是最低电机 = 基准 + $rpy\_low' = 0.0$，**结构性地保证绝不踩破地板**。这与老版本"居中式基准"（0.5 减中心偏移）不同，是"贴底式基准"。

**油门补偿（`thr_adj`）**：贴底基准通常不等于飞手想要的油门，差额 $thr\_adj = throttle - best$
就是"油门补偿通道"，它受三分支仲裁：

| 分支 | 条件 | 动作 | 含义 |
| :--- | :--- | :--- | :--- |
| 1 | $rpy\_scale < 1.0$ | $thr\_adj = 0$ | 行程被差速占满（或顶到 avg_max），油门通道关闭（掉高保命） |
| 2 | $thr\_adj < 0$ | $thr\_adj = 0$ | 飞手油门低于贴底基准，再减就踩破地板 |
| 3 | $thr\_adj > 1 - (best + rpy\_high)$ | 截到剩余空间 | 飞手油门太高，再加最高电机顶破天花板 |

分支 3 的条件式初看很绕，做一次移项立刻通透。由 $thr\_adj = throttle - best$：

$$thr\_adj > 1 - (best + rpy\_high) \iff throttle + rpy\_high > 1$$

即**"飞手油门 + 最高电机的正差速 > 1.0（最高电机要冒顶了）"**。代码里写成`1.0f - (throttle_thrust_best_rpy + rpy_high)` 是因为这个量有直接的几何意义——贴底之后最高电机距离天花板的**剩余净空**：

```
 1.0 (天花板) ──────────────────────────
               ↑ 剩余净空 = 1-(best+rpy_high)   ← thr_adj 最多只能加这么多
 best+rpy_high ── [最高电机] ────────────
               ↑ 跨度 S' = rpy_high'-rpy_low'
 best         ── [基准 = -rpy_low'] ─────
               ↑ best
 0.0 (地板)   ── [最低电机贴在这里] ─────
```

**最终合成**：每台电机 = 基准 + 补偿 + 压缩后差速。在正常线性区（$rpy\_scale = 1.0$ 且 `thr_adj` 未截断）：

$$out_i = (-rpy\_low) + (throttle + rpy\_low) + rpy_i = throttle + rpy_i$$

贴底基准在代数上被完全抵消，总升力严格等于飞手期望——**贴底基准不改变悬停特性，**

**它只是一个保证不踩地板的记账方式**。

# 统一数学定理：整个混控就是一次"窗口夹紧"

第 3.4 节的两约束 + 三分支看起来是 5 段独立逻辑，实际上可以**统一成一个公式**。

记 $T$=期望油门（入口钳制后）、$L = rpy\_low \le 0$、$H = rpy\_high \ge 0$、$A = avg\_max$、跨度 $S = H + |L|$：
$$
\boxed{\ rpy\_scale = \min\Big(1,\ \frac{1}{S},\ \frac{A}{|L|}\Big), \qquad
\bar{T} = \mathrm{constrain}\Big(T,\ \underbrace{|L| \cdot rpy\_scale}_{\text{窗口下沿}},\ \underbrace{1 - H \cdot rpy\_scale}_{\text{窗口上沿}}\Big)\ }
$$
其中 $\bar{T}$ 是全机平均油门（对称构型差速求和为零，$\bar{T} = best + thr\_adj$ 精确成立）。

**语言表述**：先压缩差速保证"可达油门窗口"非空，再把飞手油门夹进这个窗口。

下沿是地板约束（平均油门再低，最低电机就负了），上沿是天花板约束（再高，最高电机就冒顶了）。

整个混控函数只干了这一件事。

由此可导出**完整分类表**——用四个量的大小关系 100% 确定落入哪个分支：

| 判断顺序 | 条件 | 落入分支 | 平均油门 $\bar{T}$ | 物理含义 |
| :--- | :--- | :--- | :--- | :--- |
| **①** | $S > 1$ | 分支 1 | $\min\big(\frac{\|L\|}{S},\ A\big)$ | 双向超载或触发上限，窗口退化为一个点，油门通道关闭 |
| **②** | $S \le 1$ 且 $\|L\| > A$ | 分支 1 | $A$ | 单向负差速触碰下降上限，按约束 2 压缩，油门死锁在 $A$ |
| **③** | $S \le 1$ 且 $\|L\| \le A$ 且 $T < \|L\|$ | 分支 2 | $\|L\|$ | 飞手油门低于贴底基准，保地板强制将油门抬升到 $\|L\|$ |
| **④** | $S \le 1$ 且 $\|L\| \le A$ 且 $T > 1 - H$ | 分支 3 | $1 - H$ | 飞手油门过高导致最高电机冒顶，保天花板截断到 $1-H$ |
| **⑤** | 其余（$S \le 1$ 且 $\|L\| \le T \le 1 - H$ 且 $\|L\| \le A$） | 线性区 | $T$ | 无饱和，窗口容纳期望油门，全机平均推力精确等于 $T$ |

三个严密性要点：

1. **③④互斥**：$S \le 1 \iff |L| \le 1 - H$，所以 $T < |L|$ 和 $T > 1 - H$ 不可能同时成立，划分无重叠无遗漏；
2. **①的窗口是一个点**：$S > 1$ 压缩后下沿 $= |L|/S$，上沿 $= 1 - H/S = (S - H)/S = |L|/S$，两沿重合——这就是"thr_adj 必须为 0"的几何本质（可用空间严格为零，不是约等于零）；
3. **$\bar{T}$ 在所有区域边界连续无跳变**——飞机在工况间切换时油门不会"咯噔"，这是这套算法能在 500Hz 循环里裸奔而不引入抖动的数学根源。

# 十个直击灵魂的深度问答

以下十问是笔者逐行啃这段代码时真实卡住的地方，每一问都曾以为"抓到了 bug"，
每一答都以严格推导收场。它们合在一起，就是这段代码的灵魂。

## Q1：rpy_scale<1时为什么thr_adj必须清零？明明飞手还想要油门！

**分两条触发路径严格证明**：

**路径 A（约束 1 触发，$S > 1$）**：压缩后 $best + rpy\_high' = |L'| + H' = S' = 1.0$，最高电机恰好贴 1.0、最低电机恰好贴 0.0，**上下同时顶死**。往上加一分，最高电机冒顶；往下减一分，最低电机负推力。可用空间严格等于 0，清零是精确解。

**路径 B（只有约束 2 触发）**：压缩后最高电机可能远低于 1.0，天花板看似有空间！但注意约束 2 把基准压到 $best = avg\_max$，而入口钳制保证了 $throttle \le avg\_max$，所以：$$thr\_adj = throttle - avg\_max \le 0$$恒成立。

这个场景下飞手本来就在要求下降，"想往上加"的需求数学上不存在，清零一分油门都没丢。（这也解释了分支内 `if (thr_adj > 0.0f)` 置 `throttle_upper` 为什么只可能在路径 A 触发。）

**反证——如果不清零会怎样？** 设 $rpy\_high' = +0.5, rpy\_low' = -0.5$（贴底后 $best = 0.5$），飞手要 0.7，强行给 $thr\_adj = 0.2$：

最高电机 $0.5 + 0.2 + 0.5 = 1.2$ 被电调截到 1.0（正向力矩被砍 0.2），

最低电机 $0.5 + 0.2 - 0.5 = 0.2$（负向力矩足额）

——**正负力矩不对称失真，正是大机动中翻机的经典成因**。

## Q2：满油门爬升时，飞机会不会反而掉高？

**会，而且这是物理必然，任何算法都无解。** 极端算例：悬停油门 0.40，飞手推满 0.80，同时强扰动令三轴角速率环全部打满（$roll = pitch = yaw = 1.0$）。X四旋翼混控后四台电机的差速需求为$\{-0.5, -0.5, +1.5, -0.5\}$，跨度 $S = 2.0$：

- $rpy\_scale = 0.5$，压缩后 $\{-0.25, -0.25, +0.75, -0.25\}$，$best = 0.25$，$thr\_adj$ 强制为 0；
- 最终输出 $\{0,\ 0,\ 1.0,\ 0\}$：**一台电机 100%，三台停转**；
- 全机平均推力 $= 0.25 < 0.40$（悬停）$\to$ **飞机在满油门杆下短暂下沉**。

为什么不能把三台停转的电机开大一点保升力？因为最高的那台已经顶死 100%，抬其它电机 = 缩小推力差 = 削弱改出力矩，飞机会在极限姿态下翻扣倒栽——倒扣后油门越大砸得越快。此时 $\{0, 0, 1, 0\}$ 已经是这个构型物理上能挤出的**最大力矩**，神仙也变不出更多升力。

再叠加几何效应：大机动时机身往往已倾斜 $30^\circ \sim 60^\circ$，垂直升力还要乘 $\cos\theta$，下沉进一步加剧。

**为什么这样是安全的？** 500Hz 控制律下姿态通常 100~200ms 改平，$rpy\_scale$ 回到 1.0，下一拍 $thr\_adj$ 立即解除限制，油门瞬间恢复 0.80 把飞机拉起。代价是掉高几十厘米，换来的是绝不翻扣——**姿态没了，保住的高度毫无意义**。

## Q3：收油门下降时，混控会不会反而把飞机抬上去？

这是约束 2 存在的全部意义。先看没有约束 2 的反例：飞手油门 0.10（要求下降），姿态需要 $rpy\_low = -0.40$（跨度未超 1）。贴底基准会被抬到 $best = 0.40 > hover(0.35)$——**飞手收油门到底，飞机反而轰鸣上蹿，降不下来甚至飞丢**。

有了约束 2 后，平均油门 $\bar{T}$ 随姿态需求 $|L|$ 呈现优美的三段线性（$T \le A$ 的下降工况）：

$$
\bar{T} = \mathrm{constrain}\big(|L|,\ T,\ A\big)
$$

| 姿态需求 | $\bar{T}$ | 姿态力矩 | 飞机状态 |
| :--- | :--- | :--- | :--- |
| $\|L\| \le T$ | $= T$（下边沿） | 100% 满血 | 按飞手期望的速率下降，姿态高度双满足 |
| $T < \|L\| \le A$ | $= \|L\|$（区间内浮动） | 100% 满血 | 油门被姿态"借走"一点，下降略缓但必在降 |
| $\|L\| > A$ | $= A$（上边沿封顶） | 按 $A/\|L\|$ 等比压缩 | 以保底速率下降，力矩打折但绝不上蹿 |

三段连续衔接、无死区无跳变。而 $A = avg\_max < hover$（2.2 节 mix 公式保证），
所以**无论姿态怎么闹，下降工况中飞机的净下降永远有数学保证**。

由此还能看清整套算法**方向不对称的哲学**：允许短暂下坠（改平后立即补救，Q2），
绝不允许违抗指令上蹿（约束 2 封死，本问）——因为前者是自愈的，后者是发散的。

## Q4：分支3（天花板截断）什么时候触发？它不是死代码吗？

绝不是死代码，它是**大油门爬升的日常分支**。算例：飞手推 $throttle = 0.90$ 急爬升，
轻微姿态修正 $rpy\_low = -0.15, rpy\_high = +0.15$（跨度 0.3，无压缩，$rpy\_scale = 1.0$）：

- $best = 0.15$，$thr\_adj = 0.90 - 0.15 = 0.75$；
- 天花板净空 $= 1.0 - (0.15 + 0.15) = 0.70 < 0.75 \to$ **触发分支 3**；
- $thr\_adj$ 截到 0.70，`limit.throttle_upper = true`；
- 最终：最高电机 $0.85 + 0.15 = 1.00$（精确贴顶，零削波），最低电机 $0.85 - 0.15 = 0.70$；
- 飞手要 0.90，实际给 0.85——**在力矩零失真的前提下能给的最大升力**。

对照 Q3 你会发现完美的对偶：分支 2 保地板（下降触底），分支 3 保天花板（爬升触顶），
加上分支 1（双向顶死）和线性区，四个分支恰好覆盖 $[0,1]$ 盒子里的全部物理状态，
无遗漏也无冗余——这正是第 4 节统一分类表的由来。

## Q5：为什么老版本算法被换掉了？（stability-patch演进史）

ArduPilot 历史版本（3.x 时代）的"stability patch"用的是**居中式基准**：

$$
best = \min\Big(0.5 - \frac{rpy\_low + rpy\_high}{2},\ avg\_max\Big), \qquad rpy\_scale = \frac{best}{-rpy\_low}
$$
它只按"最低电机不踩破 0.0"计算压缩比，**没有同时校验最高电机会不会冒顶**。
非对称超载算例：$rpy\_low = -0.8,\ rpy\_high = +0.6$（负向差速大于正向）：

- **老算法**：$best = \min(0.6, avg\_max) = 0.6$，$scale = 0.6 / 0.8 = 0.75$ $\to$ 最高电机 $= 0.6 + 0.75 \times 0.6 = 1.05$ **冒顶**，只能靠末尾 `constrain(0, 1)` 硬削波，正向力矩被砍 0.05 而负向足额——又是力矩不对称失真；
- **新算法**：$scale = 1.0 / 1.4 = 0.7143 \to$ 压缩后 $\{-0.5714, +0.4286\}$，$best = 0.5714$，最低电机 $= 0.0$、最高电机 $= 1.0$，**双端同时精确贴合，零削波**。

而在一切不饱和的常规工况下，新老算法逐位等价。新算法（跨度压缩 + 贴底基准）本质上是把"一维保底"升级成了"二维保盒"，是这段代码十几年演进的点睛之笔。

## Q6：为什么偏航优先级高于高度（油门）？高度下降会撞地，偏航偏一点又不会炸机，难道不是高度更重要？

这是一个直觉上极具诱惑力、但动力学上极其危险的经典疑问。表面上看，机头偏几度似乎无伤大雅，而掉高会撞地。
然而在刚体动力学与导航解算的底层，**"放任偏航"绝不仅仅是机头偏航角变化，它会通过三大致命耦合链条直接摧毁整架飞机**：

1. **动力学致命耦合：高速自旋引发的陀螺力矩打翻横侧（翻扣坠毁）**  
   多旋翼偏航力矩极其微弱（依靠反扭力，仅占垂直推力的 $5\%\sim10\%$）。若高度优先级高于偏航（$\text{Throttle} > \text{Yaw}$），大油门爬升或大风天悬停时，油门会把电机行程完全占满，偏航行程被压缩为 0。此时微小的机械不对称或气流扰动就会让飞机开始绕 Z 轴单向加速自旋。

   一旦自旋角速度达到数百度每秒，高速旋转的螺旋桨（数千转/分）与机体刚体会产生巨大的**陀螺进动力矩**：
   $$
   \boldsymbol{\tau}_{\text{gyro}} = \boldsymbol{\omega}_{\text{yaw}} \times \mathbf{I}_{\text{prop}} \boldsymbol{\omega}_{\text{prop}}
   $$
   这个巨大的交叉耦合力矩会**直接横向打在滚转（Roll）和俯仰（Pitch）轴上**，瞬间击穿横侧 PID 的抗扰极限，机身在空中直接翻扣倒栽。**放任偏航，最终的结局依然是姿态倾覆坠机。**

2. **导航解算致命耦合：EKF 状态估计与位置解算的全面发散崩溃**

   高度与水平位置的控制力，必须依赖机身姿态矩阵（DCM / 四元数）精确投影到地理坐标系（NED）。当偏航发生不可控的剧烈自旋时：

   - 磁力计读数失真，GPS 航向矢量与陀螺仪积分发生严重撕裂；
   - EKF 滤波器创新度爆炸，触发 `EKF Yaw Divergence` / `Compass Variance` 警报并进入失控保护；
   - 坐标投影发散后，高度环计算出的油门本身已经变成了错误的随机量——保住一个错误的期望油门毫无意义。

3. **时间尺度与代价不对称：短暂瞬态掉高（自愈） vs 持续正反馈发散（解体）**  
   - **油门让步给偏航**：只发生在偏航大动作的 $100\sim200\text{ms}$ 瞬态。平均油门暂时让出 $0.1\sim0.2$ 的行程，飞机在 0.2 秒内仅仅下沉**几厘米到十几厘米**；偏航角速度被压住后力矩需求消失，下一拍油门立刻拉回，代价是**微小、确定且完全自愈的**；
   - **偏航让步给油门**：偏航力矩为零，机身在不平衡反扭力作用下持续角加速自旋，角速度正反馈飙升，代价是**不可逆且持续发散的**。

**行业流派与权衡**：
- **ArduPilot 经典流派**（`MOT_YAW_HEADROOM = 200`，保底 20% 偏航权限）：宁可稍微抢一点滚转俯仰的行程，也绝不允许飞机转起来；
- **纯姿态优先流派 / PX4 风格**（`MOT_YAW_HEADROOM = 0`）：当滚转俯仰极限打满时允许将偏航压到 0，全力保横侧；但**只要滚转俯仰未占满，剩余行程依然先给偏航、最后才给高度油门**。

**总结**：`滚转/俯仰(保命) ≫ 偏航(保向/抗陀螺耦合) ≫ 高度(保位)`。用 0.2 秒掉高 10 厘米的微小代价，换取整机永远不陷入陀螺力矩翻扣与 EKF 发散，是多旋翼控制工程中最深刻的底线思维。

## Q7：为什么阶段1预估偏航时基准油门写死0.5而不是hover？（几何空间vs气动平衡点与安全不对称性哲学）

初读 `throttle_thrust_best_rpy = MIN(0.5f, throttle_avg_max)` 时，极易产生直觉疑惑："0.5 看起来很像悬停油门，为什么不直接写成当前机型的实际悬停油门 hover（比如 0.35 或 0.65）？"

**答案：0.5 与 hover 是两个完全不同维度的物理量，这里必须是 0.5，不能是 hover。其背后还蕴含着极其深刻的安全不对称性哲学与“掉高保命”机制。**

1、维度差异：几何量程中心 vs 空气动力学平衡点

- **hover**：是【空气动力学】力平衡点（升力 = 重力），随飞行器载重、桨型尺寸、电池电压而改变；
- **0.5**：是电机推力量程 $[0.0, 1.0]$ 的【严格几何中心】，由量程定义决定，在任何多旋翼机型上永远是 0.5。

2、数学证明：为什么最优点恰好是 0.5？

偏航（yaw）的物理机制是通过对称电机对一正一负差速产生的（一部分电机加速，另一部分等量减速）。因此偏航空间同时受制于"天花板（到 1.0）"和"地板（到 0.0）"的双向夹击。

设试探基准为 $b$，记 $rp_{\max}$ 为被 yaw **上推**的那组电机中承载最重（最正）的 roll+pitch 载荷、$rp_{\min}$ 为被 yaw **下拉**的那组电机中承载最重（最负）的载荷，则两侧离边界最近的那台电机决定了各自的最坏净空：
$$
\text{上推侧最坏净空} = 1 - b - rp_{\max}, \qquad \text{下拉侧最坏净空} = b + rp_{\min}
$$
偏航可用空间取决于两端的最坏情况（木桶原理）：
$$
\text{yaw\_allowed} = \min\big(1 - b - rp_{\max},\ b + rp_{\min}\big)
$$
要使这个最小容限最大化（maximin 优化问题），必须令两端空间对称相等：
$$
1 - b^* - rp_{\max} = b^* + rp_{\min} \implies b^* = \frac{1 - rp_{\max} - rp_{\min}}{2}
$$
由于混控矩阵列和为零，roll+pitch 载荷天然对称 $rp_{\max} \approx -rp_{\min}$，代入即得全局最优解：
$$
b^* = \frac{1}{2} = \mathbf{0.5}
$$
**0.5 是让上推侧与下拉侧同时获得最大最坏空间的唯一对称中心**，纯粹由几何量程决定，推导全程与飞机重不重、悬停油门是多少毫无关系。这也正是 ArduPilot 源码在该行上方注释所说的 *"throttle providing maximum roll, pitch and yaw range"*——通篇强调的是 **control range（控制空间）**，只字未提悬停。

3、hover 并没有缺席，它站在正确的岗位上

悬停油门的信息已经通过 `throttle_avg_max` 参与进来了：
$$
throttle\_avg\_max = \max\big(throttle,\ \ (1 - mix)\cdot throttle + mix\cdot hover\big)
$$
`MIN(0.5f, throttle_avg_max)` 的数学本质是：**将无约束下的几何最优点 0.5，投影到操控意图允许的油门上限 `avg_max` 之中**。

- 大油门爬升时（$avg\_max \ge 0.5$）：取 0.5，获得最充裕的对称控制空间来评估偏航；
- 小油门下降时（$avg\_max < 0.5$）：被 `avg_max` 封顶，绝不为了偏航把基准强行抬到 0.5，死守"不违背下降指令"的红线。

4、背后的安全不对称性哲学（Asymmetric Safety Law）与“掉高保命”

读者可能会进一步追问：*“为什么在大油门爬升时（如飞手给 0.9），飞控敢假定油门降到 0.5 来测算偏航？这难道不会导致飞机掉高吗？为什么在小油门时却死活不准往上走？”*

这体现了飞控在面对饱和时的终极物理考量与非对称安全哲学：

| 飞行工况 | 若不遵从本原则的致命后果 | 飞控选择的正确控制哲学与物理本质 |
| :--- | :--- | :--- |
| **大油门爬升**<br>（例如飞手给 $0.90$） | **若死保 0.90 油门不让步**：<br>单向差速净空只剩 $1.0 - 0.9 = 0.1$。一旦遭遇强气流扰动，姿态环需要的差速力矩给不出来，电调全部顶死在 1.0。**差速被抹平，姿态力矩直接归零**！飞机瞬间失控侧翻倒扣（Rollover）；倒扣后朝向地面的 0.9 大油门将以 $30 \sim 40\ \text{m/s}^2$ 的加速度**把飞机加速倒栽砸向地面**。 | **【姿态绝对优先（掉高保命机制）】**：<br>飞控在第 1 阶段假定油门可降至 0.5，充分评估偏航权限；在第 3 阶段通过减小甚至清零 `thr_adj`，主动把平均油门从 0.9 拉低（哪怕飞机在 100~200ms 内短暂掉高 10~30cm），全力放行 100% 姿态防翻扣力矩。<br>**为什么允许？因为下坠是良性、可自愈的**：多旋翼只要不翻扣，姿态一旦改平，下一拍 $rpy \to 0$，油门补偿瞬间满血恢复，0.9 的大油门立即把飞机拉起。 |
| **小油门下降**<br>（例如飞手给 $0.10$） | **若允许像大油门那样把基准抬到 0.5 保姿态**：<br>悬停若为 0.35，基准一旦跳到 0.5，平均推力超过悬停点，飞机不仅不降反而**剧烈向上暴冲（Ballooning）**；触地降落一瞬间调姿态会导致飞机在地面**剧烈弹跳、翻滚打桨炸机**；大风中永远无法着陆直至耗尽电量失控。 | **【下降指令绝对优先（严禁私自上蹿）】**：<br>飞手/自动航线正在降落，油门上限被 `avg_max`（如 0.25）彻底封死。<br>若姿态差速过大，飞控宁可启动 `rpy_scale < 1.0` **等比削减姿态力矩，也绝不把平均推力往上顶**。<br>**为什么禁止？因为私自爬升是失控性、不可自愈的灾难**。 |

5、最后一层保险：它只是测量尺，不是输出值

这个试探基准仅用于第一阶段预估 `yaw_allowed`，算完即弃，绝不直接输出给电机。实际电机的平均油门由第三阶段精确执行（$\bar{T} = \text{constrain}(T, |L'|, 1-H')$），在无饱和或单向饱和但净空充足时，依然输出飞手期望的高油门，绝不会无缘无故掉高。

**总结**：`0.5` 管"几何最优（控制空间最大化）"，`avg_max` 管"意图边界（保证净下降）"，各司其职，完美解耦。大油门允许向姿态让步、允许短暂掉高保命；小油门严守 `avg_max` 红线、死死封住私自上蹿。

##  Q8：为什么阶段1注释写0.5-(rpy_low+rpy_high)/2而代码只写0.5？如何严格证明没有Yaw时rp_low=-rp_high？

ArduPilot 源码第 1 阶段上方的注释写道：

```cpp
// calculate throttle that gives most possible room for yaw which is the lower of:
//      1. 0.5f - (rpy_low+rpy_high)/2.0 - this would give the maximum possible margin above the highest motor and below the lowest
//      2. motors_throttle_avg_max
```

但紧随其后的代码却只有极简的一行：

```cpp
float throttle_thrust_best_rpy = MIN(0.5f, throttle_avg_max);
```

这常让人困惑：注释里的 `0.5f - (rpy_low+rpy_high)/2.0` 是作者漏写了，还是被化简了？

**答案：这是逻辑因果上的“解耦破局”，也是数学推导上的“必然等价”。**

1、逻辑因果死锁（为什么当前阶段根本不能包含 Yaw）

`rpy_low` 和 `rpy_high` 的严格定义是包含了 Roll、Pitch、Yaw 全部三轴后的电机输出极值。但阶段 1 的任务**恰恰是去计算到底能给 Yaw 分配多少空间（`yaw_allowed`）**！  

如果阶段 1 的基准计算反过来依赖最终的 `rpy_low/high`，就会陷入“先有鸡还是先有蛋”的因果死循环。因此，在评估阶段必须先将 Yaw 从公式中彻底剥离。

2、数学严格证明：纯 Roll + Pitch 极值严格反对称（$\text{rp\_low} + \text{rp\_high} \equiv 0$）

设多旋翼有 $N$ 个电机，第 $i$ 个电机的机体坐标为 $(x_i, y_i)$。由刚体力矩平衡定理，各电机的滚转俯仰混控因子满足：

$$k_{r, i} = -C_r \cdot y_i, \qquad k_{p, i} = C_p \cdot x_i$$

给定任意飞手指令 $(r, p)$，第 $i$ 个电机在无偏航时的输出为 $u_i(r, p) = r \cdot k_{r, i} + p \cdot k_{p, i}$。

- **中心反演对称性（Antipodal Symmetry）**：在所有对称多旋翼构型（Quad-X/+、Hexa、Octo 等）中，对任意电机 $i(x_i, y_i)$，必然存在对角对径电机 $j$ 满足 $(x_j, y_j) = (-x_i, -y_i)$；
- **混控因子相反性**：代入得 $k_{r, j} = -k_{r, i}$，且 $k_{p, j} = -k_{p, i}$；
- **输出集合的对偶镜像**：$u_j(r, p) = -(r \cdot k_{r, i} + p \cdot k_{p, i}) = -u_i(r, p)$，即全机输出集合关于原点对称（$U = -U$）；
- **极值对偶性**：
  $$\text{rp\_low} = \min(U) = \min_i(u_i) = \min_j(-u_j) = -\max_j(u_j) = -\text{rp\_high}$$
  $$\implies \mathbf{\text{rp\_low} + \text{rp\_high} \equiv 0 \quad (\forall r, \forall p)}$$

代入居中公式即得：

$$0.5\text{f} - \frac{\text{rp\_low} + \text{rp\_high}}{2.0} = 0.5\text{f} - \frac{0}{2.0} \equiv \mathbf{0.5f}$$

3、架构与算力智慧

- **注释讲理论**：写出通用居中平移模型（$0.5 - \text{中点偏移}$），指明算法的数学渊源；
- **代码做实现**：利用纯 RP 的天生对称性直接代换为常量 `0.5f`，既斩断了因果死锁，又在早期 8 位单片机上省去了一个完整的遍历循环与浮点除法；
- **第 3 阶段真正落地**：当偏航加上去之后，对角电机的旋转同向性打破了正负对称性（$(u_i+y) + (-u_i+y) = 2y \ne 0$），此时代码在阶段 3 真正统计非对称极值并执行居中与贴底夹紧！

## Q9：起飞在地面时为什么要把mix设为0.1？这样会不会导致起飞离地姿态不稳？

在垂直起降（VTOL）和多旋翼工程中，这是著名的“地面操作安全性（Ground Handling）与空中抗扰权限（Flight Authority）的死锁博弈”。

1、为什么地面必须死守 mix = 0.1（若地面维持 mix = 0.5 会发生什么灾难？）

公式为：$avg\_max = throttle \cdot (1 - mix) + hover \cdot mix$。

假设悬停油门 $hover = 0.50$。

- **若在地面给 $mix = 0.5$**：飞手刚解锁、油门在最低位（$throttle = 0.0$）时，$avg\_max = 0.0 \times 0.5 + 0.5 \times 0.5 = \mathbf{0.25}$。混控器允许把基础油门抬到 0.25！
  
  若地面不平机身微倾 $2^\circ$ 或受到地面扰动，姿态环要求 $\pm 0.25$ 滚转修正力矩：
  
  - 最高电机推力 $= 0.25 + 0.25 = \mathbf{0.50}$（**已达到悬停总推力！**）；
  
  - 最低电机推力 $= 0.25 - 0.25 = 0.0$。
  
    **灾难发生**：飞手油门明明为 0，飞机一侧电机却以 50% 的暴风推力疯狂抬升，飞机以未离地的起落架为支点**瞬间侧翻倒扣（Tip-over）、打碎全部桨叶炸机**！降落触地时同理，会引发剧烈触地弹跳（Landing Bounce）。
  
- **若在地面给 $mix = 0.1$**：$avg\_max = 0.0 \times 0.9 + 0.5 \times 0.1 = \mathbf{0.05}$。力矩被等比压缩到 $0.05 / 0.25 = 0.2$，电机推力仅为怠速微风，飞机像钉子一样死死定在地面，绝不翻滚。

2、为什么 mix = 0.1 绝不会导致起飞离地时姿态不稳？（时间尺度的完美错位）

代码中斜坡滤波器的上升速率是 $2.0/\text{s}$：

$$\Delta t = \frac{0.5 - 0.1}{2.0} = \mathbf{0.20\ \text{秒} \ (200\text{ms})}$$

而电机从怠速加速到产生升力、机身克服重力腾空离地，天然存在 **$200\sim300\text{ms}$ 的电机机械与电气响应时间**：

```
t = 0 ms   : 飞手推油门起飞 -> desired_mix 立即置 0.5 -> 此时推力 10%，飞机重重压在地上
t = 100 ms : 电机加速中（推力 30% < 重力） -> mix 已爬升至 0.30 -> 飞机仍在地面
t = 200 ms : mix 已经 100% 达到 0.50（满额姿态抗扰能力完全就位）！-> 推力达 45%
t = 250 ms+: 总推力 > 50% 超过重力，起落架离开地面升空！
             【此时飞机是在 mix = 0.50 的满血抗风抗扰权限下腾空离地的！】
```

**核心结论**：在起落架脱离地面的那一瞬间，mix 早就完成了 200ms 的爬升并满血就位，空中根本不会以 0.1 的 mix 飞行。

3、终极兜底防线协同

- **大角度逃生**：若在大于 30 度的陡坡起飞，`thrust_error_angle > 30°` 触发，直接越级赋予 $mix = max$，全力防侧翻；
- **地面积分清零**：未离地前姿态 PID 积分项被严格清零，防止地面干耗积累出翻滚力矩。

##  Q10：偏航保底裕度MOT_YAW_HEADROOM设为0好还是50好？为什么不能设为0（偏航自旋死锁与陀螺力矩耦合链）？

`MOT_YAW_HEADROOM` 定义了“在电机行程被极限滚转/俯仰占满时，强制为偏航保留的最低推力裕度”。单位为 PWM 微秒当量（50 对应推力量程的 5%）。

1、若激进地设为 0：触发致命的“偏航自旋死锁（Death Spiral）”

当设为 0 时，一旦强扰动或激烈机动使 Roll+Pitch 占满电机 1.0 的跨度，`yaw_allowed` 将被精确斩断为 **0.000**。此时飞控对偏航轴输出 0 差速，整机完全丧失偏航控制力矩与角速度反向阻尼。随之触发刚体动力学链式崩溃：
1. 侧风或进气不对称令飞机产生微小的被动偏航自旋；

2. 由于阻尼归零，自旋迅速加速至数百度每秒；

3. **刚体陀螺进动效应（致命物理破坏）**：高速自旋的机体与数千转高速旋转的螺旋桨产生剧烈陀螺力矩：

   $$\boldsymbol{\tau}_{\text{gyro}} = \boldsymbol{\omega}_{\text{yaw}} \times \mathbf{I}_{\text{prop}} \boldsymbol{\omega}_{\text{prop}}$$

   该力矩横向生硬打在 Roll 和 Pitch 轴上，激发出巨大的非预期横侧力矩；

4. 姿态环被迫加大 Roll/Pitch 输出以对抗陀螺力矩 $\to$ 电机进一步深度饱和 $\to$ 偏航被死死锁在 0 $\to$ 飞机越转越快 $\to$ EKF 航向发散、姿态失控翻扣坠毁！

2、为什么 50（5%）是黄金平衡点（Sweet Spot）？

- **姿态代价微乎其微**：在 Roll/Pitch 占满 1.0 的极限饱和下，强制加入 0.05 偏航后总跨度为 1.05。混控器等比缩放比为 $rpy\_scale = 1.0 / 1.05 = \mathbf{0.952}$——**滚转/俯仰力矩仅被微幅稀释 4.8%**，抗翻扣能力完全不受影响；
- **偏航收益极其巨大**：这 5% 的推力裕度充当了强有力的**偏航角速率 D 环阻尼刹车片**，在被动自旋刚刚萌芽的十几毫秒内将其瞬间扑灭，彻底斩断陀螺交叉耦合链条，确保大机动后姿态与航向迅速收敛。

| `MOT_YAW_HEADROOM` | 极限饱和时 RP 保留率 | 偏航抗自旋阻尼能力 | 综合评价 |
| :--- | :--- | :--- | :--- |
| **`0`** | 100% | 0%（完全归零） | **存在自旋死锁与陀螺翻扣风险**，不推荐 |
| **`50`（黄金平衡）** | **95.2%（仅微调 4.8%）** | **5%（足以提供关键阻尼）** | **兼顾姿态抗翻与偏航本质安全**，推荐 |
| **`200`（AP 默认）** | 83.3%（姿态削弱 16.7%） | 20%（强力偏航） | 适合标准轻型多旋翼；对重载/垂起稍偏大 |

# 抗饱和闭环：limit标志如何反哺上层控制器

混控不只输出推力，还输出四个标志位，它们是底层向上层"报告饱和"的唯一通道：

| 标志 | 置位时机 | 上层的反应 |
| :--- | :--- | :--- |
| `limit.roll` / `limit.pitch` | $rpy\_scale < 1.0$（`limit.set_rpy(true)`） | 滚转/俯仰角速率 PID 冻结积分 |
| `limit.yaw` | yaw 被 `yaw_allowed` 削减，或 $rpy\_scale < 1.0$ | 偏航角速率 PID 冻结积分 |
| `limit.throttle_upper` | 入口超上限 / 分支 1 中 $thr\_adj > 0$ / 分支 3 | 高度环停止向上累积分 |
| `limit.throttle_lower` | 入口油门 $\le 0$ | 高度环停止向下累积分 |

工作机制：第 1 节引用的 `rate_controller_run_dt()` 把 `_motors.limit.roll` 等作为参数传进
`update_all()`，PID 内部据此实施**条件积分**——饱和时 $|I|$ 只许冻结或向零卸载，不许加深。
若无此机制，电机顶满期间误差持续存在，积分疯长（windup），
饱和解除后需要很长时间反向卸载，表现为大幅超调甚至振荡。

两个值得知晓的细节：

1. **roll/pitch 合用一个饱和信号**（`set_rpy(true)` 同时置两轴）不是偷懒：混控压缩的是三轴合量，无法归因到单轴；且从混控几何上讲，用另一轴把 I 加大救不了已被压到下限的那对电机，"两轴一起不恶化"与混控几何是对齐的。（PX4 的分配器用按轴残差实现了更细的粒度，但四旋翼大机动时两者行为相近。）
2. **分支 2 有一个官方遗留 TODO**：`thr_adj < 0` 被钳制时**不置任何标志**（源码注释：`todo: add lower limit flag ...`）。即"想降降不到位"时高度环收不到通知。全球机队都这么飞了很多年，属于已知可接受行为，但读代码时不要以为是自己看漏了。

# 混控之后：从推力到PWM

混控输出的 `_thrust_rpyt_out[i] ∈ [0,1]` 是**期望推力比**，不能直接线性变成 PWM，因为电机推力对转速（近似对 PWM）是非线性的。`Thrust_Linearization::thrust_to_actuator()` 负责最后一步：

```cpp
float Thrust_Linearization::thrust_to_actuator(float thrust_in) const
{
    thrust_in = constrain_float(thrust_in, 0.0, 1.0);
    return spin_min + (spin_max - spin_min) * apply_thrust_curve_and_volt_scaling(thrust_in);
}
```

- `apply_thrust_curve_and_volt_scaling()` 用 `MOT_THST_EXPO`（默认 0.65）做推力曲线反变换——期望推力经过它以后，"PWM→真实推力"的物理非线性刚好被抵消，**保证混控世界里的一切线性代数（差速成比例、力矩保真）在真实世界里成立**；同时叠加电池电压缩放（掉压时自动加大输出比）；
- `spin_min / spin_max`（`MOT_SPIN_MIN/MAX`）把结果映射到电机实际工作区（怠速转速以下推力无效区被跳过）；
- 最后按 `MOT_PWM_MIN/MAX` 换算成微秒脉宽，写入定时器比较寄存器，电调收到信号。

至此，一个角速率误差完成了它到桨叶推力的全部旅程。

# 工程验证方法论：如何证明混控代码没有bug

混控错误的代价是坠机，仅靠"目视审查 + 试飞"验证是不负责任的。

推荐一套可以在宿主机（PC）上执行的**不变量穷举验证法**，思路对任何飞控代码库都适用：

**方法**：把混控函数原样编译进一个 PC 测试程序（不是抄写副本，而是编译真实生产代码），
同时按参考实现（如 ArduPilot 官方源码）独立实现一份对照版本，然后用两类输入轰炸它们：

- **边界值网格**：$roll/pitch/yaw \in \{0, \pm0.05, \pm0.2, \pm0.5, \pm0.8, \pm1.0, \pm1.5\}$、
  油门/avg_max/thrust_max 各取边界与典型值的全组合（覆盖每一个 if 分支的触发边界）；
- **大范围随机采样**：数百万组，含超出正常范围的鲁棒性输入。

对每组输入校验五条**不变量**（任何一条被打破即是 bug）：

1. **等价性**：被测代码与参考实现逐电机输出一致（误差 < 1e-5）；
2. **界内性**：未做末端削波前，输出天然落在 $[0,1]$ 内——证明算法自身不越界，
   末尾的 constrain 只是保险丝而非遮羞布；
3. **标志一致性**：四个 limit 标志与参考实现完全一致（抗饱和信号错了，坠机方式更隐蔽）；
4. **升力守恒**：thr_adj 未被钳制时，全机平均推力精确等于期望油门；
5. **窗口公式**：全机平均推力恒等于 $\mathrm{constrain}(T,\ |L'|,\ 1-H')$（第 4 节定理的实证）。

笔者在自己的工程上以此法跑了**近千万组用例（覆盖四旋翼 X/+、六旋翼 X 三种构型），全部通过**，]才敢确认移植的混控与 ArduPilot 逐位等价。整套测试单次运行只需数秒，此后任何人动混控代码，跑一遍全绿才允许上真机——**把"不敢动底层代码"变成"改完 3 秒知道对错"**。

另一条经验：**不要为了"更清晰"重构这段代码**。统一窗口公式（第 4 节）虽然优美，但分支结构除了算油门还要分路置 limit 标志，重构后代码量不会减少，却会失去与 ArduPilot 上游的逐行可对照性（未来上游修 bug 无法直接 diff 移植）。数学之美放进注释和测试，代码本身与上游保持一致——这才是安全关键代码的正确姿势。

# 总结：十条设计哲学

1. **姿态 ≫ 偏航 ≫ 高度**。多旋翼的一切升力建立在姿态之上，姿态没了，保住的高度毫无意义。
2. **[0,1] 是铁盒子**。力矩来自差速，一端 100% 一端 0% 就是构型的力矩物理极限，算法只能分配、不能创造。
3. **等比压缩优于硬削波**。跨度超限时全轴等比缩小（力矩方向保真），绝不单独削哪台电机（力矩畸变翻机）。
4. **贴底基准 + 油门补偿**。基准锚定最低点保地板，补偿通道找回升力；不饱和时二者代数抵消，总升力分毫不差。
5. **方向不对称（安全不对称性哲学）**：大油门允许短暂下坠（保住姿态即可自愈并满血拉起），小油门封死违抗指令的私自上蹿（触地弹跳与失控发散不可自愈）——`avg_max` 是那道关键封条。
6. **整个混控 = 一次窗口夹紧**：$\bar{T} = \mathrm{constrain}(T,\ |L'|,\ 1-H')$，四个分支只是夹紧落点不同。
7. **地面安全与空中权限的时间错位解耦**：地面/待命/接地 mix=0.1 锁死抬升上限杜绝自杀式翻扣；200ms 的极速爬升斜坡在离地升空前抢先满血就位，兼得绝对安全与充沛抗扰。
8. **保底偏航阻尼掐死陀螺耦合链**：大机动即便姿态深饱和，也必须强制保留 5%（50）最低偏航裕度，用微不足道的 4.8% 姿态让步提供关键反向阻尼，切断自旋死锁链。
9. **饱和必须上报**。limit 标志与 PID 条件积分构成抗饱和闭环，缺了它，每次饱和都在给下一次超调蓄能。
10. **验证优于信任**。穷举不变量测试 + 与参考实现逐位对比，是对"这段代码错了会摔飞机"唯一诚实的回答。

# 参考资料

*参考源码：ArduPilot（GPLv3），`libraries/AP_Motors/AP_MotorsMatrix.cpp`、
`libraries/AP_Motors/AP_Motors_Thrust_Linearization.cpp`、
`libraries/AC_AttitudeControl/AC_AttitudeControl_Multi.cpp`。*

