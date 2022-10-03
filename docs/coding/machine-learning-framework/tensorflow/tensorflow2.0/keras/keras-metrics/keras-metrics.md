# 指标keras.metrics

* [返回上层目录](../keras.md)



# 使用步骤总结

```
用法的流程：
1、 新建一个metric
2、 更新数据update_state
3、 取出数据result().numpy()
4、 重置清零数据reset_states
```

**步骤1，新建一个meter**

```python
acc_meter = metrics.Accuracy()   # 新建一个准确度的meter
loss_meter = metrics.mean()    # 求平均值
```

**步骤2，向meter中添加数据**

```python
loss_meter.update_state(loss)
acc_meter.update_state(y_true, y_pred)
```

**步骤3，取出数据**

```python
print(loss_meter.result().numpy())
print('Evaluate Acc:', total_correct/total, acc_meter.result().numpy())
```

在**loss_meter.result()** 会得到一个tensor的数据，再使用.numpy将它转化为numpy的数据

**步骤4，清除缓存meter**

```python
loss_meter.reset_states()
acc_meter.reset_states()
```



# 各种评价指标

## tf.keras.metrics.Accuracy

计算预测与真实值的准确度。

```python
tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
```

例如，如果**y_true**为[1、2、3、4]，而**y_pred**为[0、2、3、4]，则精度为**3/4**或 **.75** 。如果将权重指定为 **[1、1、0、0]** ，则精度将为**1/2**或 **.5** ,权重0是用来屏蔽的。

```python
update_state(y_true, y_pred, sample_weight=None)
```

示例：

```python
import tensorflow as tf

m = tf.keras.metrics.Accuracy()
m.update_state([1, 2, 3, 4], [0, 2, 3, 4])
# 注： m.update_state(y, pred) 中添加的不是一个实时的Accuracy，它是一个专门计算Accuracy 的meter，只需要传递真实的y和预测的y（pred），它会自动的计算Accuracy，并保存。
print(m.result().numpy())
# 0.75
m.reset_states()

m.update_state([1, 2, 3, 4], [0, 2, 3, 4], sample_weight = [1, 1, 0, 0])
print(m.result().numpy())
# 0.5
```

## tf.keras.metrics.Mean

函数说明：计算给定值的（加权）平均值。

用法：

```python
tf.keras.metrics.Mean(name='mean', dtype=None)
```

例如，如果值为[1、3、5、7]，则平均值为4。如果权重指定为[1、1、0、0]，则平均值为2。

示例：

```python
import tensorflow as tf

m = tf.keras.metrics.Mean()
m.update_state([1, 3, 5, 7])
print(m.result().numpy())
# 4.0
m.reset_states()

m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
print(m.result().numpy())
# 2.0
```



# 参考资料

* [tensorflow2中keras.metrics的Accuracy和Mean的用法](https://blog.csdn.net/jpc20144055069/article/details/105324654)

本文参考了此博客。

