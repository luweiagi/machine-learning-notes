# Humanoid Strike

- [返回上层目录](../time-chamber.md)

github: [inspirai/TimeChamber](https://github.com/inspirai/TimeChamber)



### Tasks

Source code for tasks can be found in `timechamber/tasks`,The detailed settings of state/action/reward are in [here](https://github.com/inspirai/TimeChamber/blob/main/docs/environments.md). More interesting tasks will come soon.

#### Humanoid Strike

Humanoid Strike is a 3D environment with two simulated humanoid physics characters. Each character is equipped with a sword and shield with 37 degrees-of-freedom. The game will be restarted if one agent goes outside the arena. We measure how much the player damaged the opponent and how much the player was damaged by the opponent in the terminated step to determine the winner.

![humanoid_strike](pic/humanoid_strike.gif)



# 参考资料

[世界上第一个开源的人形机器人物理格斗ai，想象一下未来的玩具是这个[呲牙]](https://www.zhihu.com/pin/1595186795960086528)

