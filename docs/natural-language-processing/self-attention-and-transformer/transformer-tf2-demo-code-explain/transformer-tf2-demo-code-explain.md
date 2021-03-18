# Transformer模型TF2官网demo代码解读

- [返回上层目录](../self-attention-and-transformer.md)



TensorFlow官网有一个Transformer的demo，即

[理解语言的 Transformer 模型](https://www.tensorflow.org/tutorials/text/transformer)（如果打不开，需要科学上网），本文针对此略微进行结构调整（但代码不变），然后进行讲解。

整个代码分为三部分：

* params.py  一些配置参数
* transformer_model.py 模型结构
* run_demo.py 运行，包括数据产出、模型加载、训练、预测等

# 配置参数params.py

这块的代码如下所示，不多，也很好理解，不再多说，即便不理解也不用管，接着往下看。

```python
# 数据来源
DATA_PATH = 'ted_hrlr_translate/pt_to_en'
# 读取数据的buffer
BUFFER_SIZE = 20000
# 数据的batch size
BATCH_SIZE = 64

# 为了使训练速度变快，我们删除长度大于40个单词的样本。
MAX_LENGTH = 40
# 训练的轮次
EPOCHS = 1  # 20
```

#  模型结构transformer_model.py

导入模型所需的包：

```python
import tensorflow as tf
import numpy as np

'''
本代码为 理解语言的Transformer模型 中的代码实现
https://www.tensorflow.org/tutorials/text/transformer
本教程训练了一个Transformer模型 用于将葡萄牙语翻译成英语。
'''
```







# 运行run_demo.py



# 参考资料

* [理解语言的 Transformer 模型](https://www.tensorflow.org/tutorials/text/transformer)

这是官网的demo

