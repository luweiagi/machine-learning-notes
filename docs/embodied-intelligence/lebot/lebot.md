# lebot中的PushT训练指南

* [返回上级目录](../embodied-intellegence.md)

# 工作准备

## 环境准备

```shell
conda deactivate
conda create -n vla -c conda-forge python=3.12 -y
conda activate vla

pip install lerobot 'lerobot[sim]'
pip install gym-pusht  # 安装 pusht 仿真环境
```

# 训练模型





```
cd ~/vla/lerobot

# 设置镜像站（加速下载数据集）
export HF_ENDPOINT=https://hf-mirror.com

# 安装 FFmpeg 和开发库
sudo apt update
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
# 检查 FFmpeg 版本
ffmpeg -version

# 开始训练 Diffusion 模型
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --policy.repo_id=./my_act_model \
  --policy.push_to_hub=False \
  --steps=10000 \
  --batch_size=32 \
  --output_dir=./outputs
```

可以先试简单的：

```
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --policy.repo_id=./my_act_model \
  --policy.push_to_hub=False \
  --steps=3 \
  --batch_size=4 \
  --output_dir=./outputs
```

训练成功：

```shell
从日志可以看到：
Training: 100%|... 3/3 [00:04<00:00, 1.57s/step] - 3步训练完成
Checkpoint policy after step 3 - 模型已保存
End of training - 正常结束
```





继续训练：

```
lerobot-train \
  --config_path=./pusht_model/checkpoints/last/pretrained_model/train_config.json \
  --resume true \
  --steps=400000 \
  --save_freq=10000
```

训练到 40 万步后，再运行：

```shell
lerobot-eval \
  --policy.path=./pusht_model/checkpoints/400000/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=20 \
  --policy.num_inference_steps=40
```

显示：

```
Stepping through eval batches: 100%|█████████████████████| 1/1 [00:50<00:00, 50.05s/it, running_success_rate=25.0%]
```

评估完的模型就在你服务器上，视频是自动生成的。我教你两个最简单的方法看效果。





# 运行测试

```
pip install gym_aloha
```





```shell
export HF_ENDPOINT=https://hf-mirror.com

# 回到项目根目录
cd ~/vla/lerobot

# 使用 last 检查点评估
lerobot-eval \
  --policy.path=./outputs/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --eval.n_episodes=1 \
  --eval.batch_size=1



# 重新设置镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 回到项目根目录
cd ~/vla/lerobot

# 继续下载（会从断点续传）
lerobot-eval \
  --policy.path=lerobot/diffusion_pusht \
  --env.type=pusht \
  --eval.n_episodes=1 \
  --eval.batch_size=1
```







安装成功后运行 `lerobot-eval`，你会看到：

1. 下载预训练模型（首次运行需要下载，约几百 MB）
2. 弹出一个 Pygame/Gym 仿真窗口
3. 机械臂自动把 T 形块推到目标区域
4. 终端输出评估指标（成功率、平均奖励等）



使用 ACT 模型（同样支持 Pusht）

ACT 是另一个成熟模型，文件完整：

```
lerobot-eval \
  --policy.path=lerobot/act_pusht \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --eval.batch_size=50
```









# PushT 训练指南

基于 LeRobot 的 Diffusion Policy，在 PushT 环境中训练机器人推块任务。

## 目录

- [环境介绍](#环境介绍)
- 环境搭建
- 数据集准备
- 训练模型
- 继续训练
- 评估模型
- 超参数说明
- 常见问题

------

## 环境介绍

### PushT环境

PushT 是一个简化的 2D 仿真环境，用于验证机器人学习算法。

| 要素             | 说明                                       |
| :--------------- | :----------------------------------------- |
| **任务目标**     | 控制灰色小球，将 T 形块推入浅绿色目标区域  |
| **观察空间**     | 图像 (384×384) + 状态向量 (2维：T形块坐标) |
| **动作空间**     | 2 维连续动作（推力方向和大小）             |
| **每集最大步数** | 300 步                                     |
| **成功判定**     | T 形块与目标区域重叠面积 ≥ 95%             |

### 环境中的物体

| 物体          | 颜色/形状 | 角色                         |
| :------------ | :-------- | :--------------------------- |
| 小球（Agent） | 蓝色圆形  | 由模型控制                   |
| T 形块        | 灰色 T 型 | 被推动的物体                 |
| 目标区域      | 绿色 T 型 | 需要推入的位置               |
| 推力指示器    | + 标记    | 显示推力方向（动态，可忽略） |

### 模型架构

- **类型**：Diffusion Policy（扩散策略）
- **视觉编码器**：ResNet18（11M 参数）
- **去噪网络**：条件 U-Net（约 40M 参数）
- **总参数量**：52M
- **训练方式**：从头训练（随机初始化），非微调

------

## 环境搭建

### 创建 Conda 环境

```shell
conda create -n vla python=3.10 -y
conda activate vla
```

> 注意：Python 3.12+ 可能存在兼容性问题，推荐 3.10 或 3.11。

### 安装LeRobot和依赖

```shell
# 安装核心库
pip install lerobot

# 安装 PushT 环境
pip install gym-pusht

# 安装仿真依赖
pip install 'lerobot[sim]'

# 如果网络较慢，可使用清华源
pip install lerobot -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证安装

```shell
python -c "import lerobot; print('LeRobot 安装成功')"
python -c "import gym_pusht; print('PushT 环境安装成功')"
```

## 数据集准备

### 自动下载（推荐）

当运行 `lerobot-train` 命令时，它会自动从 Hugging Face 下载所需的数据集。

**为国内用户加速**：

为了提升下载速度，可以在训练**之前**，通过设置环境变量，将下载源指向一个国内镜像站。

```bash
# 设置镜像站地址
export HF_ENDPOINT=https://hf-mirror.com
```

设置完此变量后，无需任何额外操作。当运行下面的训练命令 `lerobot-train` 时，数据便会自动从这个镜像站下载了。

### 手动下载（可选）

如果自动下载不稳定，或者您希望单独下载数据集以备他用，可以使用以下命令手动下载。

```bash
# 1. 安装 hf 工具 (如果尚未安装)
pip install -U "huggingface_hub[cli]"

# 2. 可选：设置镜像站加速下载
export HF_ENDPOINT=https://hf-mirror.com

# 3. 执行下载命令，将数据集保存到本地目录 ./pusht_dataset
hf download lerobot/pusht --repo-type dataset --local-dir ./pusht_dataset
```

手动下载成功后，在后续的训练命令中，可以通过添加 `--dataset.root=./pusht_dataset` 参数来直接使用本地的数据集，避免重复下载。

## 训练模型

### 基础训练命令

```shell
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=./pusht_model \
  --policy.repo_id=./pusht_model \
  --policy.push_to_hub=False \
  --steps=20000 \
  --batch_size=64
```

### 参数说明

| 参数                   | 含义              | 示例值               |
| :--------------------- | :---------------- | :------------------- |
| `--policy.type`        | 策略类型          | `diffusion` 或 `act` |
| `--dataset.repo_id`    | 数据集来源        | `lerobot/pusht`      |
| `--output_dir`         | 输出目录          | `./pusht_model`      |
| `--policy.repo_id`     | 模型保存标识      | `./pusht_model`      |
| `--policy.push_to_hub` | 是否上传到 HF Hub | `False`              |
| `--steps`              | 总训练步数        | `20000`              |
| `--batch_size`         | 批次大小          | `64`                 |
| `--save_freq`          | 保存检查点频率    | `1000`               |

### 完整示例（推荐配置）

bash

```
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --output_dir=./pusht_model \
  --policy.repo_id=./pusht_model \
  --policy.push_to_hub=False \
  --steps=400000 \
  --batch_size=64 \
  --save_freq=20000
```



------

## 继续训练

### 从最新检查点恢复

bash

```
lerobot-train \
  --config_path=./pusht_model/checkpoints/last/pretrained_model/train_config.json \
  --resume true
```



### 从指定步数恢复并修改目标步数

bash

```
lerobot-train \
  --config_path=./pusht_model/checkpoints/200000/pretrained_model/train_config.json \
  --resume true \
  --steps=400000
```



> 注意：`--steps` 是**总目标步数**，不是新增步数。

------

## 评估模型

### 基础评估命令

bash

```
lerobot-eval \
  --policy.path=./pusht_model/checkpoints/last/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=10 \
  --eval.batch_size=10
```



### 带推理步数调优的评估

bash

```
lerobot-eval \
  --policy.path=./pusht_model/checkpoints/last/pretrained_model \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=20 \
  --policy.num_inference_steps=40
```



### 评估结果解读

输出中的关键指标：

| 指标             | 含义                | 典型值      |
| :--------------- | :------------------ | :---------- |
| `pc_success`     | 成功率（重叠 ≥95%） | 0.25（25%） |
| `avg_max_reward` | 平均最高重叠率      | 0.62        |
| `avg_sum_reward` | 平均累积奖励        | 75          |
| `eval_ep_s`      | 每集平均耗时（秒）  | 2.5         |

### 查看评估视频

bash

```
# 进入评估结果目录
ls outputs/eval/2026-XX-XX-XX-XX-XX_pusht_diffusion/videos/pusht_0/

# 复制到 Windows 桌面（WSL 环境）
cp outputs/eval/.../videos/pusht_0/*.mp4 /mnt/c/Users/你的用户名/Desktop/
```



------

## 超参数说明

### 模型核心参数

| 参数                  | 含义                 | 推荐值 |
| :-------------------- | :------------------- | :----- |
| `horizon`             | 动作轨迹长度         | 16     |
| `n_action_steps`      | 一次生成的动作步数   | 8      |
| `n_obs_steps`         | 执行多少步后重新观测 | 2      |
| `num_train_timesteps` | 训练时扩散步数       | 100    |
| `num_inference_steps` | 推理时扩散步数       | 40     |

### `n_action_steps` 与 `n_obs_steps` 的关系

- 模型一次生成 `n_action_steps` 步动作
- 执行 `n_obs_steps` 步后，重新观测并重新生成
- 剩余 `n_action_steps - n_obs_steps` 步被丢弃

bash

```
# 更频繁地重新规划（更像 ACT）
--policy.n_obs_steps=1

# 更少地重新规划（更平滑）
--policy.n_obs_steps=8
```



### 推理步数调优建议

| 配置                      | 效果             | 推荐场景     |
| :------------------------ | :--------------- | :----------- |
| `num_inference_steps=10`  | 快速，动作较粗糙 | 实时要求高   |
| `num_inference_steps=40`  | **平衡点**       | 通用推荐     |
| `num_inference_steps=100` | 精细，收益递减   | 追求极致精度 |

------

## 常见问题

### Q: 训练时提示 `policy.repo_id missing`

**原因**：新版 LeRobot 要求指定 `--policy.repo_id`。

**解决**：添加 `--policy.repo_id=./模型路径 --policy.push_to_hub=False`

### Q: 评估时提示 `add_collision_handler` 错误

**原因**：pymunk 版本与 gym-pusht 不兼容。

**解决**：

bash

```
pip uninstall pymunk -y
pip install pymunk==6.6.0
```



### Q: 模型下载/数据集下载失败

**原因**：Hugging Face 网络不稳定。

**解决**：

bash

```
export HF_ENDPOINT=https://hf-mirror.com
```



### Q: 训练很慢

**原因**：使用 CPU 训练，且 Pusht 数据集包含视频解码。

**解决**：

- 减小 `--batch_size`
- 使用 `--dataset.streaming=True`
- 或换用 GPU 环境

### Q: 评估时没有弹出可视化窗口

**原因**：WSL/服务器环境无 GUI 支持。

**解决**：评估会自动生成 mp4 视频，复制到有 GUI 的机器上观看。

------

## 预期训练效果

| 训练步数  | 预期成功率 | 说明                     |
| :-------- | :--------- | :----------------------- |
| 500 步    | ~0%        | 随机探索阶段             |
| 5000 步   | ~5-10%     | 开始学会朝目标方向推     |
| 20000 步  | ~10-15%    | 偶尔能推到目标附近       |
| 400000 步 | ~25%       | 稳定推到位，成功率约 1/4 |

> 实际效果受硬件、batch size、学习率等影响，以上为参考值。

------

## 参考命令速查

bash

```
# 训练
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht --output_dir=./pusht_model --policy.repo_id=./pusht_model --policy.push_to_hub=False --steps=400000 --batch_size=64 --save_freq=20000

# 继续训练
lerobot-train --config_path=./pusht_model/checkpoints/last/pretrained_model/train_config.json --resume true --steps=400000

# 评估
lerobot-eval --policy.path=./pusht_model/checkpoints/last/pretrained_model --env.type=pusht --eval.n_episodes=20 --eval.batch_size=20 --policy.num_inference_steps=40
```



------

这份 README 涵盖了你从零到 40 万步训练的全过程。如果还有需要补充的内容，随时告诉我。



#  参考资料



===

* [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
* [Policy Interface and Base Classes](https://deepwiki.com/huggingface/lerobot/4.1-imitation-learning-training)

* [欢迎来到全网最强LeRobot具身智能教程](https://zihao-ai.feishu.cn/wiki/TS6swApHbinx01kHDi5cf5n5n8c)
* [LeRobot训练SO-101机械臂ACT模型全流程解析](https://www.bilibili.com/video/BV1enHizcEk6/)

* [Hugging Face 中 LeRobot 使用的入门指南](https://zhuanlan.zhihu.com/p/1905602739016307192)

##  大模型

* [什么是VLA？我还不清楚，但是我想了解入门，你能让我了解清楚吗](https://chat.deepseek.com/a/chat/s/5edd82f8-92d7-49cc-9d1b-a79ece4afd92)

* [vla入门适合用openvla还是lerobot？](https://chat.deepseek.com/a/chat/s/85f10cd1-4cf2-4b8b-ac47-34d964a96655)



