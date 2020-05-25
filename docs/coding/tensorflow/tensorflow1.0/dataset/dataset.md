# TensorFlow Dataset API

* [返回上层目录](../tensorflow1.0.md)



[TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)

[Tensorflow学习——导入数据](https://blog.csdn.net/weixin_39506322/article/details/82455860)



# 数据集来源

## from_tensor_slices

`from_tensors()` 这个函数会把传入的tensor当做一个元素,但是`from_tensor_slices()` 会把传入的tensor除开第一维之后的大小当做元素个数.比如上面`2x5` 的向量,我们得到的元素是其中每一个形状为`(5,)`的tensor。

它的真正作用是**切分传入Tensor的第一个维度，生成相应的dataset。**

例如：

```python3
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
```

传入的数值是一个矩阵，它的形状为(5, 2)，tf.data.Dataset.from_tensor_slices就会切分它形状上的第一个维度，最后生成的dataset中一个含有5个元素，每个元素的形状是(2, )，**即每个元素是矩阵的一行。**



[TensorFlow学习（十五）：使用tf.data来创建输入流(上)](https://blog.csdn.net/xierhacker/article/details/79002902)

[TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)



# 转换



## repeat

repeat不指定次数的情况下，可无限延伸。

repeat的位置也很关键，如下两种就有明显差别

（1）`repeat()`紧跟在`Dataset`之后，当batch不满足5之后就会重复给。

```python
dataset=tf.data.Dataset.from_tensor_slices(tensors=np.array([0., 1., 2., 3., 4., 5.])).repeat().shuffle(buffer_size=1).batch(5)
iterator=dataset.make_one_shot_iterator()
element=iterator.get_next()
with tf.Session() as sess:
    print("elements of dataset:")
    for i in range(2):
        print("epoch:", i)
        for j in range(6 // 2):
            print("---mini_batch:", j, sess.run(element))
```

运行结果：

```shell
elements of dataset:
epoch: 0
---mini_batch: 0 [0. 1. 2. 3. 4.]
---mini_batch: 1 [5. 0. 1. 2. 3.]
---mini_batch: 2 [4. 5. 0. 1. 2.]
epoch: 1
---mini_batch: 0 [3. 4. 5. 0. 1.]
---mini_batch: 1 [2. 3. 4. 5. 0.]
---mini_batch: 2 [1. 2. 3. 4. 5.]
```

（2）`repeat()`跟在`batch(5)`之后，当batch取没之后就会重复给。

```python
dataset=tf.data.Dataset.from_tensor_slices(tensors=np.array([0., 1., 2., 3., 4., 5.])).shuffle(buffer_size=1).batch(5).repeat()
iterator=dataset.make_one_shot_iterator()
element=iterator.get_next()
with tf.Session() as sess:
    print("elements of dataset:")
    for i in range(2):
        print("epoch:", i)
        for j in range(6 // 2):
            print("---mini_batch:", j, sess.run(element))
```

运行结果：

```shell
elements of dataset:
epoch: 0
---mini_batch: 0 [0. 1. 2. 3. 4.]
---mini_batch: 1 [5.]
---mini_batch: 2 [0. 1. 2. 3. 4.]
epoch: 1
---mini_batch: 0 [5.]
---mini_batch: 1 [0. 1. 2. 3. 4.]
---mini_batch: 2 [5.]
```



[Tensorflow datasets.shuffle repeat batch方法](https://www.cnblogs.com/marsggbo/p/9603789.html)



## interleave

interleave()是Dataset的类方法，所以interleave是作用在一个Dataset上的。

首先该方法会从该Dataset中取出cycle_length个element，然后对这些element apply map_func, 得到cycle_length个新的Dataset对象。然后从这些新生成的Dataset对象中取数据，每个Dataset对象一次取block_length个数据。当新生成的某个Dataset的对象取尽时，从原Dataset中再取一个element，然后apply map_func，以此类推。



cycle_length 和 block_length 参数元素的生成顺序.cycle_length 控制并发处理的输入元素的数量.如果将cycle_length 设置为1,则此转换将一次处理一个输入元素,并将产生与 tf.contrib.data.Dataset.flat_map 相同的结果.一般来说,这种转换将应用 map_func 到 cycle_length 的输入元素,在返回的 Dataset 对象上打开迭代器,并循环通过它们生成每个迭代器的 block_length 连续元素,并在每次到达迭代器结束时使用下一个输入元素.

例如：

```python
dataset = tf.data.Dataset.from_tensor_slices(tensors=np.array([1., 2., 3., 4., 5., 6.]))\
    .interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(6), cycle_length=2, block_length=4)

iterator=dataset.make_one_shot_iterator()
element=iterator.get_next()
with tf.Session() as sess:
    print("elements of dataset:")
    for i in range(2):
        print("epoch:", i)
        for j in range(10):
            print("---mini_batch:", j, sess.run(element))
'''
{
    1, 1, 1, 1,
    2, 2, 2, 2,
    1, 1,
    2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    3, 3,
    4, 4,
    5, 5, 5, 5,
    5, 5,
}
'''
```

输出为：

```shell
elements of dataset:
epoch: 0
---mini_batch: 0 1.0
---mini_batch: 1 1.0
---mini_batch: 2 1.0
---mini_batch: 3 1.0
---mini_batch: 4 2.0
---mini_batch: 5 2.0
---mini_batch: 6 2.0
---mini_batch: 7 2.0
---mini_batch: 8 1.0
---mini_batch: 9 1.0
epoch: 1
---mini_batch: 0 2.0
---mini_batch: 1 2.0
---mini_batch: 2 3.0
---mini_batch: 3 3.0
---mini_batch: 4 3.0
---mini_batch: 5 3.0
---mini_batch: 6 4.0
---mini_batch: 7 4.0
---mini_batch: 8 4.0
---mini_batch: 9 4.0
```



[小M学机器学习 tf.data.Dataset.interleave()](https://zhuanlan.zhihu.com/p/97876668)

[TensorFlow文本文件行的数据集](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-8pwb2dbr.html)

[tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)

[tf.data.Dataset](https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?version=stable#interleave)







# 迭代器





## 创建迭代器

构建了表示输入数据的 `Dataset` 后，下一步就是创建 `Iterator` 来访问该数据集中的元素。[`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) API 目前支持下列迭代器，复杂程度逐渐增大：

- **单次**  **iterator = dataset.make_one_shot_iterator()**
- **可初始化**  **iterator = dataset.make_initializable_iterator()**
- **可重新初始化  iterator = tf.data.Iterator.from_structure()**
- **可馈送  iterator = tf.data.Iterator.from_string_handle()**



### from_string_handle



```python
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(5).map(
    # lambda x: x + tf.random_uniform([], -1, 1, tf.int64)).repeat()
    lambda x: x).repeat()
validation_dataset = tf.data.Dataset.range(11, 14)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()


sess = tf.Session()
# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
# while True:
for i in range(3):
    # Run 200 steps using the training dataset. Note that the training dataset is
    # infinite, and we resume from where we left off in the previous `while` loop
    # iteration.
    print("the %sth epoch:" % i)
    print("    training_handle:")
    for _ in range(7):
        print(sess.run(next_element, feed_dict={handle: training_handle}))

    print("    validation_handle:")
    # Run one pass over the validation dataset.
    sess.run(validation_iterator.initializer)
    for _ in range(3):
        print(sess.run(next_element, feed_dict={handle: validation_handle}))
```

结果为：

```shell
the 0th epoch:
    training_handle:
0
1
2
3
4
0
1
    validation_handle:
11
12
13
the 1th epoch:
    training_handle:
2
3
4
0
1
2
3
    validation_handle:
11
12
13
the 2th epoch:
    training_handle:
4
0
1
2
3
4
0
    validation_handle:
11
12
13
```



## 消耗迭代器

[TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)



## 保存迭代器状态

[TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)





