# 得到模型参数值

* [返回上层目录](../tools.md)





# 取出tf.layers.dense全连接层的weight和bias参数值

TensorFlow版本：1.14.0

在TensorFlow中，tf.layers.dense 定义了一个全连接层，其实现的是（来自官方文档）：

> This layer implements the operation: outputs = activation(inputs * kernel + bias) Where activation is the activation function passed as the activation argument (if not None), kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is True).

意思就是它实现了 y = activation(x * kernel + bias) 的操作，其中，activation是激活函数。在这里，kernel 就是指我们通常所说的 weight，它被TF称为 kernel 而不是 weight。因此，如果你想从这个模型里取出weight参数的话，就要注意它的名字了，否则会读不到这个参数。
下面我们就来看具体的例子。

下面这段代码的逻辑非常简单，它干了下面这些事情：
* 创建了一个非常简单的2层全连接网络
* 训练网络
* 训练完成后把它保存成checkpoint格式的模型文件
* 从保存的checkpoint读取模型里第一层的 weight 和 bias 参数值，打印出来

```
import tensorflow as tf
import numpy as np
LAYER_1_NAME = 'layer1'  # 第一层的名字
LAYER_2_NAME = 'layer2'  # 第二层的名字

# 创建一个非常简单的神经网络，它有两层
x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
layer1 = tf.layers.dense(x, 5, activation=tf.nn.sigmoid, name=LAYER_1_NAME)
layer2 = tf.layers.dense(layer1, 2, activation=tf.nn.sigmoid, name=LAYER_2_NAME)
loss = tf.reduce_mean((layer2 - x) ** 2)
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
x_values = np.random.normal(0, 1, (5000, 2))  # 生成用于输入的随机数
    for step in range(1000):
_, loss_value = sess.run([optimizer, loss], feed_dict={x: x_values})
if step % 100 == 0:
print("step: %d, loss: %f" % (step, loss_value))
# 把模型保存成checkpoint
    saver = tf.compat.v1.train.Saver()
save_path = saver.save(sess, './checkpoint/model.ckpt')
print("model saved in path: %s" % save_path, flush=True)
# 读取刚保存的checkpoint
    reader = tf.train.NewCheckpointReader(save_path)
weights = reader.get_tensor(LAYER_1_NAME + '/kernel')  # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
    bias = reader.get_tensor(LAYER_1_NAME + '/bias')  # bias的名字
    print(weights)
print(bias)
# 如果想打印模型中的所有参数名和参数值的话，把下面几行取消注释
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor name: ", key)
    #     print(reader.get_tensor(key))
```





# 参考资料

* [如何取出 tf.layers.dense 定义的全连接层的weight和bias参数值](https://blog.csdn.net/fly_time2012/article/details/105861436)

`取出tf.layers.dense全连接层的weight和bias参数值`一节参考了此博客。