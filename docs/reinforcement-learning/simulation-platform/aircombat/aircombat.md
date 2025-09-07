# AirCombat

* [返回上层目录](../simulation-platform.md)



# 仿真环境

## Gor-Ren/gym-jsbsim

Gym-JSBSim项目，是一个运行在开源AI训练环境Open AI GYM下的开源飞行动态模型FDM。其中JSBSim提供模型部分功能，Gym提供AI运行环境和强化学习功能。即运行在Gym环境下，通过强化学习训练JSBSim模型令飞机具备自动驾驶能力。

github地址：[Gor-Ren/gym-jsbsim](https://github.com/Gor-Ren/gym-jsbsim/tree/master)

> Gym-JSBSim provides reinforcement learning environments for the control of fixed-wing aircraft using the JSBSim flight dynamics model. Gym-JSBSim requires a Unix-like OS and Python 3.6.
>
> The package's environments implement the OpenAI Gym interface allowing environments to be created and interacted with in the usual way, e.g.:
>
> ```
> import gym
> import gym_jsbsim
> 
> env = gym.make(ENV_ID)
> env.reset()
> state, reward, done, info = env.step(action)
> ```
>
> Gym-JSBSim optionally provides 3D visualisation of controlled aircraft using the FlightGear simulator.

## liuqh16/LAG

github：[liuqh16/LAG](https://github.com/liuqh16/LAG/tree/master)

An environment based on JSBSIM aimed at one-to-one close air combat.

> Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment with baseline reinforcement learning algorithms
>
> We provide a competitive environment for red and blue aircrafts games, which includes single control setting, 1v1 setting and 2v2 setting. The flight dynamics based on JSBSIM, and missile dynamics based on our implementation of proportional guidance. We also provide ppo and mappo implementation for self-play or vs-baseline training.

该项目的详细介绍：[探索天空的智慧较量：轻量级空战游戏——基于强化学习的飞机竞技场](https://blog.csdn.net/gitblog_00056/article/details/139555482)

参考该环境的项目：

* [【RL+空战】学习记录01：jsbsim 仿真环境初次学习，F16 战机起飞](https://blog.csdn.net/weixin_41369892/article/details/149205956)

记录一下jsbsim + python + tacview 仿真&可视化：F16起飞任务

```python
import jsbsim
import matplotlib.pyplot as plt

# 初始化 JSBSim
sim = jsbsim.FGFDMExec(None)
sim.set_debug_level(0)
sim.load_model('f16')

# 初始状态（地面起飞）
sim['ic/h-sl-ft'] = 35000
sim['ic/vt-fps'] = 2500
sim['ic/psi-true-deg'] = 0
sim['ic/theta-deg'] = 95  # 适度仰角
sim['ic/lat-gc-deg'] = 41.62513
sim['ic/long-gc-deg'] = 41.59104
sim['ic/roc-fpm'] = 0
sim.run_ic()

# 设置油门和升降舵控制
sim['fcs/throttle-cmd-norm'] = 1.0        # 油门最大

# 仿真时间与步长
dt = sim.get_delta_t()
total_sim_time = 90  # 秒
steps = int(total_sim_time / dt)

# ACMI 文件头信息
reference_time = "2011-06-02T05:00:00Z"
acmi_lines = [
    "FileType=text/acmi/tacview\n",
    "FileVersion=2.2\n",
    f"0,ReferenceTime={reference_time}\n",
    "0,Author=JSBSim2Tacview\n",
    "0,Title=F16 Takeoff\n",
    "0,RecordingTime=60\n",
    "0,Comments=Generated from JSBSim\n",
    "0,DataSource=JSBSim\n",
    "0,DataRecorder=PythonScript\n",
    "0,3000102,Type=Aircraft,Name=F16\n"
]

# 存储用于绘图的高度数据
altitudes_m = []
time_sec = []

# 仿真主循环
for step in range(steps):
    sim['fcs/elevator-cmd-norm'] = 0.2  # 保持抬头
    sim.run()
    t = step * dt

    # 读取参数
    lat = sim["position/lat-gc-deg"]
    lon = sim["position/long-gc-deg"]
    alt_ft = sim["position/h-sl-ft"]
    alt_m = alt_ft * 0.3048
    heading = sim["attitude/psi-deg"]
    pitch = sim["attitude/theta-deg"]
    roll = sim["attitude/phi-deg"]

    # 写入 ACMI
    acmi_lines.append(f"#{t:.2f}\n")
    acmi_lines.append(f"3000102,T={lat}|{lon}|{alt_m:.2f},Heading={heading:.2f},Pitch={pitch:.2f},Roll={roll:.2f},Name=F16\n")

    # 存储高度数据
    altitudes_m.append(alt_m)
    time_sec.append(t)

# 输出 ACMI 文件
with open("F16_takeoff.acmi", "w") as f:
    f.writelines(acmi_lines)

# 绘制高度变化图
plt.figure(figsize=(10, 6))
plt.plot(time_sec, altitudes_m, label="Altitude (m)", color="green")
plt.title("F16 Takeoff Altitude Curve")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

## xcwoid/BVRGym

github: [xcwoid/BVRGym](https://github.com/xcwoid/BVRGym/tree/main)

This library is heavily based on JSBSim software (https://github.com/JSBSim-Team/jsbsim). This library's primary purpose is to allow users to explore Beyond Visual Range (BVR) tactics using Reinforcement learning.

> Environment
>
> The environments above mainly use the F16 flight dynamics and BVR missile models. The F16 model has an additional wrapper to control simply, while the BVR missile has a Proportional Navigation guidance law implemented to guide it toward the target. The following commands are equivalent, but they run the process in parallel to speed up convergence. Currently, there are three available environments:
