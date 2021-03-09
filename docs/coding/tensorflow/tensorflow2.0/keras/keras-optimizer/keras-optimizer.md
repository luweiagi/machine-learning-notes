# 优化器keras.optimizer

* [返回上层目录](../keras.md)



# 根据Epoch调整学习率

在HRNet的论文中，作者提到他们训练模型时的学习率设置为：

> The base learning rate is 0.0001 and is dropped to 0.00001 and 0.000001 at the 30 th and 50 th epochs.

根据模型训练进程调整学习率是一种非常常见的做法。TensorFlow中自定义学习率策略有两种方法。

### 拓展LearningRateSchedule

第一种是通过API `tf.keras.optimizers.schedules` 来实现。当前提供了5种学习率调整策略。如果这5种策略无法满足要求，可以通过拓展类 `tf.keras.optimizers.schedules.LearningRateSchedule` 来自定义调整策略。然后将策略实例直接作为参数传入`optimizer` 中。在官方示例[Transformer model](https://www.tensorflow.org/tutorials/text/transformer#training_and_checkpointing)中展示了具体的示例代码。

不过这种方法存在局限性。类方法 `def__call__(self, step)` 只允许传入一个参数 `step` 。所以你的调整策略要以当前的 `step` 为依据。如果要随epoch改变学习率，需要根据当前step推算epoch。

### 自定义callback 

通过callback来改变学习率是第二种可行的方案。同样，官方提供了完善的文档来说明如何做到这一点。HRNet在第30 epoch和50 epoch时改变了学习率。自定义callback可以帮助我们做到这一点。

#### 全局类方法

Keras.model的 `fit` 方法通过接受 `callback` 对象作为参数，可以获得训练过程的内部状态。在训练开始与结束时，每个batch开始与结束时，每个epoch开始与结束时， `callback` 对应的类方法会被调用。例如在每个epoch开始时会调用

```python
on_epoch_begin(self, epoch, logs=None)
```

在epoch结束时，则会调用

```python
on_epoch_end(self, epoch, logs=None)
```

这种机制给我们调整训练过程提供了机会窗口。

#### 访问模型属性

`callback` 还提供了访问模型本身的方法，进而可以获取、设定模型的训练学习率。例如获取模型当前学习率

```python
lr = float(tf.keras.backend.get_value(
            self.model.optimizer.learning_rate))
```

`on_epoch_begin` 函数中，`epoch`是一个已知参数。此刻可以根据`epoch`来调整学习率。之后再将新的学习率赋值给model

```python
tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
```





