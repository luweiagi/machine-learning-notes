# TensorBoard

* [返回上层目录](../training-tool.md)



`TensorBoard` 是PyTorch和TensorFlow都支持的可视化工具，用于追踪和展示训练过程中各种指标，比如：

- 损失曲线（Loss）
- 奖励变化（Reward）
- 学习率（Learning Rate）
- 模型权重（Weights & Gradients）
- 网络结构（Graph）
- 训练时的图像、直方图等（Images, Histograms）

在PyTorch中，`SummaryWriter` 是**TensorBoard的接口**，用于把数据写入到TensorBoard可读取的日志文件中。

`SummaryWriter` 会把训练信息**记录到磁盘**，然后用`tensorboard`命令可视化。

# 代码解读

```python
from torch.utils.tensorboard import SummaryWriter

# 创建 TensorBoard 记录器
writer = SummaryWriter(log_dir='runs/time_{}'.format(args.date_time))
```

`SummaryWriter(log_dir=...)`

- 作用：**创建一个TensorBoard记录器**
- `log_dir='runs/time_{}'`：
  - `runs/` 是TensorBoard记录日志的目录。
  - `'time_{}'` 可能是训练时间戳，用 `args.date_time`来区分不同的训练任务。

# 如何使用 SummaryWriter

1. 记录训练损失

```python
for step in range(1000):
    loss = some_loss_function()
    writer.add_scalar('Loss/train', loss, step)  # 记录 loss 到 TensorBoard
```

TensorBoard 里会显示 `Loss/train` 这一项的变化曲线。

2. 记录奖励

```python
writer.add_scalar('Reward/episode', episode_reward, episode_number)
```

3. 记录神经网络的权重直方图

```python
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)
```

# 如何启动TensorBoard

当训练代码运行后，会生成 `runs/` 目录下的日志文件，之后可以用以下命令启动TensorBoard：

```shell
tensorboard --logdir=runs
```

然后在浏览器打开 `http://localhost:6006`，就能看到各种可视化曲线。

# 总结

1. `TensorBoard` 是一个**可视化工具**，用来监控强化学习训练过程。
2. `SummaryWriter` 是PyTorch提供的 API，用来**写入日志数据**。
3. 你可以用它来记录**loss、reward、网络参数等**，然后用`tensorboard`命令查看。

这东西在强化学习中非常有用，特别是**调试和对比不同超参数的影响**时！