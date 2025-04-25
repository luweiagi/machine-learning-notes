# 学习率和BatchSize的关系

* [返回上层目录](../learning-rate.md)

[深度学习 | Batch Size大小对训练过程的影响（转）](https://zhuanlan.zhihu.com/p/86529347)

学习率（Learning Rate）和批量大小（Batch Size）是深度学习中两个关键的超参数，它们的设置会直接影响模型的训练效果和收敛速度。二者的关系可以从**梯度更新的数学本质**和**训练实践的经验法则**两个角度理解。

------

### **1. 数学本质：梯度估计的方差与稳定性**

#### **(1) 基本关系**

- **Batch Size**：决定每次参数更新时使用的样本数量。
- **Learning Rate**：决定参数更新的步长。

当 `batch_size` 增大时：

- **梯度估计的方差更小**（更多样本的平均梯度更接近真实梯度），因此可以容忍更大的学习率。
- **参数更新方向更稳定**，但更新步长需要与梯度方差匹配。

#### **(2) 线性缩放规则（Linear Scaling Rule）**

- **核心假设**：保持梯度更新的期望步长一致。

- **公式**：
  $$
  \text{New Learning Rate}=\text{Base Learning Rate}\times \frac{\text{New Batch Size}}{\text{Base Batch Size}}
  $$
  **适用场景**：

  - 使用 **SGD 或 Momentum 优化器**。
  - 学习率较小时（未达到“临界批量”）。

#### **(3) 平方根缩放规则（Square Root Scaling Rule）**

- **公式**：
  $$
  \text{New Learning Rate}=\text{Base Learning Rate}\times \sqrt{\frac{\text{New Batch Size}}{\text{Base Batch Size}}}
  $$
  **适用场景**：

  - 使用 **自适应优化器（如 Adam）**。
  - 学习率较大或接近“临界批量”（Critical Batch Size）。

------

### **2. 经验法则与实验验证**

#### **(1) 何时需要调整学习率？**

- **增大 Batch Size**：通常需要增大学习率（按线性或平方根比例）。
- **减小 Batch Size**：通常需要减小学习率。

#### **(2) 临界批量（Critical Batch Size）**

- **定义**：当 Batch Size 超过某个阈值时，继续增大 Batch Size 不再允许线性增加学习率（梯度噪声已足够小，进一步增大学习率可能导致不稳定）。
- **经验值**：对于大多数任务，临界批量在 `100~1000` 之间。

#### **(3) 自适应优化器的鲁棒性**

- **Adam、RMSProp 等优化器**：对学习率和 Batch Size 的变化更鲁棒（自动调整梯度幅度的缩放），但仍需根据 Batch Size 调整初始学习率。

------

### **3. 实验调整步骤**

1. **基准设置**：
   - 选择一个合理的基准 Batch Size（如 `32` 或 `256`）和学习率（如 `1e-3`）。
   - 训练模型并记录收敛情况。
2. **调整 Batch Size**：
   - 若新 Batch Size 为 `k × 原 Batch Size`，按以下规则调整学习率：
     - **线性缩放**：`新学习率 = 原学习率 × k`（适用于 SGD）。
     - **平方根缩放**：`新学习率 = 原学习率 × sqrt(k)`（适用于 Adam）。
3. **监控指标**：
   - **训练损失曲线**：若损失震荡或发散，说明学习率过大。
   - **梯度范数**：若梯度范数显著变化，需重新平衡学习率和 Batch Size。

------

### **4. 经典案例**

#### **(1) ImageNet 训练**

- **基准设置**：Batch Size=256，Learning Rate=0.1（SGD + Momentum）。
- **调整规则**：若 Batch Size=512，Learning Rate=0.2。

#### **(2) Transformer 模型**

- **基准设置**：Batch Size=4096，Learning Rate=1e-4（Adam）。
- **调整规则**：若 Batch Size=8192，Learning Rate=1e-4 × sqrt(2) ≈ 1.4e-4。

------

### **5. 注意事项**

- **分布式训练**：在数据并行中，总 Batch Size = 单卡 Batch Size × GPU 数量，需同步调整学习率。
- **小批量训练**：Batch Size 过小（如 `<16`）时，梯度噪声大，需降低学习率或使用梯度累积（Gradient Accumulation）。
- **学习率预热（Warmup）**：大 Batch Size 训练时，初始阶段逐步增大学习率（避免早期梯度爆炸）。

------

### **总结**

- **核心关系**：Batch Size 增大 → 梯度估计方差减小 → 允许增大学习率。
- **调整规则**：
  - 线性缩放（SGD）：学习率 ∝ Batch Size。
  - 平方根缩放（Adam）：学习率 ∝ sqrt(Batch Size)。
- **实验验证**：最终需通过训练曲线确定最佳组合。

通过合理平衡学习率和 Batch Size，可以最大化训练效率和模型性能。



# 参考资料

===

* [理解学习率和batch_size大小的关系](https://zhuanlan.zhihu.com/p/364865720)
* [Batch Size理解、如何调整batch size和学习率之间的关系？](https://blog.csdn.net/weixin_43135178/article/details/114882276)

* [【AI不惑境】学习率和batchsize如何影响模型的性能？](https://cloud.tencent.com/developer/article/1474432)

