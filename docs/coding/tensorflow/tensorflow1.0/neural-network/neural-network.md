# Neural Network

* [返回上层目录](../tensorflow1.0.md)



# embedding_lookup

tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。tf.nn.embedding_lookup（params, ids）:params可以是张量也可以是数组等，id就是对应的索引，其他的参数不介绍。

[tf.nn.embedding_lookup函数的用法](https://www.cnblogs.com/gaofighting/p/9625868.html)



**ids只有一行：**

```python
p=tf.Variable(tf.random_normal([5,1]))#生成5*1的张量
b = tf.nn.embedding_lookup(p, [1, 3])#查找张量中的序号为1和3的
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    #print(c)
    print(sess.run(p))
    print(p)
    print(type(p))
```

结果是

```
[[0.15791859]
 [0.6468804 ]]
[[-0.2737084 ]
 [ 0.15791859]
 [-0.01315552]
 [ 0.6468804 ]
 [-1.0601649 ]]
<tf.Variable 'Variable:0' shape=(10, 1) dtype=float32_ref>
<class 'tensorflow.python.ops.variables.Variable'>
```



**如果ids是多行：**

```python
import tensorflow as tf
import numpy as np

a = [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]
a = np.asarray(a)
idx1 = tf.Variable([0, 2, 3, 1], tf.int32)
idx2 = tf.Variable([[0, 2, 3, 1], [4, 0, 2, 2]], tf.int32)
out1 = tf.nn.embedding_lookup(a, idx1)
out2 = tf.nn.embedding_lookup(a, idx2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(out1)
    print out1
    print '=================='
    print sess.run(out2)
    print out2
```

结果是

```python
[[ 0.1  0.2  0.3]
 [ 2.1  2.2  2.3]
 [ 3.1  3.2  3.3]
 [ 1.1  1.2  1.3]]
Tensor("embedding_lookup:0", shape=(4, 3), dtype=float64)
==================
[[[ 0.1  0.2  0.3]
  [ 2.1  2.2  2.3]
  [ 3.1  3.2  3.3]
  [ 1.1  1.2  1.3]]

 [[ 4.1  4.2  4.3]
  [ 0.1  0.2  0.3]
  [ 2.1  2.2  2.3]
  [ 2.1  2.2  2.3]]]
Tensor("embedding_lookup_1:0", shape=(2, 4, 3), dtype=float64)
```



**如果张量是二维**

```python
p = tf.Variable(tf.random_normal([5, 2]))#生成5*2的张量
b = tf.nn.embedding_lookup(p, [1, 3])  # 查找张量中的序号为1和3的

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    # print(c)
    print(sess.run(p))
    print(p)
    print(type(p))
```

结果是

```
[[-0.5707266   1.31594   ]
 [-0.16886874  0.11318349]]
[[-1.0407234  -1.1149817 ]
 [-0.5707266   1.31594   ]
 [ 0.41421786 -0.28062782]
 [-0.16886874  0.11318349]
 [ 0.7819106  -1.7940577 ]]
<tf.Variable 'Variable:0' shape=(5, 2) dtype=float32_ref>
<class 'tensorflow.python.ops.variables.RefVariable'>
```





















