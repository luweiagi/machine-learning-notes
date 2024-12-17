# 异质多智能体强化学习HARL(Heterogeneous-Agent Reinforcement Learning)

* [返回上层目录](../advanced-theme.md)

异质智能体强化学习（Heterogeneous-Agent Reinforcement Learning）是单智能体强化学习还是多智能体强化学习？它和多智能体强化学习的关系与区别是什么？

来自chatgpt的回答：

异质智能体强化学习与多智能体强化学习的关系

（1）**异质智能体强化学习属于多智能体强化学习范畴**

异质智能体强化学习（HARL）是一种多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）的子领域。它专注于**异质性场景**，即智能体之间具有显著差异的任务。

- **多智能体强化学习 (MARL)** 是研究多个智能体如何在共享环境中交互的框架，智能体可以是**同质**（行为和能力相似）或**异质**（能力、目标或信息各不相同）。
- HARL 是 MARL 的细化方向，针对智能体在**能力、状态空间、动作空间或目标**上存在差异的场景。

（2）**HARL 和 MARL 的区别**

| **方面**       | **多智能体强化学习（MARL）**         | **异质智能体强化学习（HARL）**           |
| -------------- | ------------------------------------ | ---------------------------------------- |
| **智能体特性** | 偏重于同质智能体的协作或竞争         | 研究异质智能体间的协作或竞争             |
| **目标任务**   | 多数是统一任务，例如共同获得最大奖励 | 可能有不同任务或局部目标，需协调不同能力 |
| **方法**       | 通常使用参数共享策略优化             | 更倾向于个性化策略优化和理论保障         |
| **应用场景**   | 团队游戏、群体行为建模等             | 异质机器人团队、混合型任务协作           |

（3）**HARL 与单智能体强化学习的关系**

HARL 是多智能体问题的一种特殊情况，其核心挑战来自**多智能体间的交互**，这一点与单智能体强化学习完全不同。

- **单智能体强化学习**研究单个智能体如何从环境中学习最优策略，环境本身是静态或非交互的。
- **HARL 的关键难点**在于：每个智能体的学习目标相互依赖，策略的更新需考虑异质智能体的动态行为。

（4）总结

- HARL 是 MARL 的一个子领域，专注于异质性问题。
- HARL 的研究旨在突破传统多智能体算法对于**同质性假设**的限制，解决更复杂、更贴近真实场景的任务。





北京大学的HARL的官方库：[PKU-MARL/HARL](https://github.com/PKU-MARL/HARL)

此仓库包含基于 PyTorch 的异质智能体强化学习（HARL）算法的官方实现，包括 HAPPO、HATRPO、HAA2C、HADDPG、HATD3、HAD3QN 和 HASAC 等。与 MAPPO 不同，HARL 算法普遍适用于异质智能体，并受到严格理论支持，常常在性能上优于传统方法。本仓库使研究人员和实践者能够轻松复现我们的实验结果（在七个具有挑战性的基准上）或将 HARL 算法应用于特定场景。

以下是所提供内容的中文翻译：

------

### 异质智能体强化学习（Heterogeneous-Agent Reinforcement Learning）

此仓库包含基于 PyTorch 的异质智能体强化学习（HARL）算法的官方实现，包括 HAPPO、HATRPO、HAA2C、HADDPG、HATD3、HAD3QN 和 HASAC 等。与 MAPPO 不同，HARL 算法普遍适用于异质智能体，并受到严格理论支持，常常在性能上优于传统方法。本仓库使研究人员和实践者能够轻松复现我们的实验结果（在七个具有挑战性的基准上）或将 HARL 算法应用于特定场景。

概述

HARL 算法是我们为解决一般异质智能体场景中的多智能体协作问题而提出的新方法。它不依赖于限制性的参数共享技术。

关键特性：

1. HARL 算法通过采用**顺序更新机制**实现智能体之间的协调更新，这与 MAPPO 和 MADDPG 的同时更新机制不同。
2. HARL 算法具备单调改进和收敛到均衡的理论保证，从而有效促进智能体间的协作行为。
3. 无论是基于策略的 HARL 算法（如 HAPPO）还是基于价值的 HARL 算法（如 HASAC），在广泛的基准测试中均表现优异。

以下图示展示了顺序更新机制。

有关更多细节，请参阅我们的论文 **HARL (JMLR 2024)** 和 **MEHARL (ICLR 2024 spotlight)**。

环境依赖安装

除了 HARL 算法外，我们还为七种常见环境（SMAC、SMACv2、MAMuJoCo、MPE、Google Research Football、Bi-DexterousHands 和 Light Aircraft Game）实现了接口，用户可以直接使用（同时也支持 Gym 环境，Gym 是单智能体环境，作为多智能体环境的特例，主要用于参考）。

Along with HARL algorithms, we also implement the interfaces for seven common environments ([SMAC](https://github.com/oxwhirl/smac), [SMACv2](https://github.com/oxwhirl/smacv2), [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco), [MPE](https://pettingzoo.farama.org/environments/mpe/), [Google Research Football](https://github.com/google-research/football), [Bi-DexterousHands](https://github.com/PKU-MARL/DexterousHands), [Light Aircraft Game](https://github.com/liuqh16/CloseAirCombat))。

安装 Bi-DexterousHands 环境依赖

- Bi-DexterousHands 依赖于 IsaacGym。需要满足 IsaacGym 的硬件要求。
- 安装方法：从 IsaacGym 官方网站下载 IsaacGym Preview 4 版本，然后在其 `python` 文件夹下运行 `pip install -e .`。

安装 Light Aircraft Game (LAG)

- Light Aircraft Game 是一个新开发的红蓝飞机对抗环境，提供单控制、1v1 和 2v2 场景。
- 对于 2v2 多智能体场景，目前支持自对弈。为弥补局限性，我们新增了合作非武器任务和发射导弹任务：
  1. **非武器任务**：智能体需协作飞到对方尾部并保持合适距离。
  2. **发射导弹任务**：智能体需躲避对方导弹并发射导弹摧毁敌人。