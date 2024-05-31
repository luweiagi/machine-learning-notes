# 理解Eager模式及tf.function

* [返回上层目录](../tensorflow2.0.md)

在tensorflow1.x的时候，代码默认的执行方式是graph execution（图执行），而从tensorflow2.0开始，改为了eager execution（饥饿执行）。

正如翻译的意思一样，eager execution会立即执行每一步代码，非常的饥渴。而graph execution会将所有代码组合成一个graph（图）后再执行。这里打一个不太恰当的比喻来帮助大家理解：eager execution就像搞一夜情，认识后就立即“执行”，而graph execution就像婚恋，认识后先憋着，不会立即“执行”，要经过了长时间的“积累”后，再一次性“执行”。

在eager 模式下，代码的编写变得很自然很简单，而且因为代码会被立即执行，所以调试时也变得很方便。而graph 模式下，代码的执行效率要高一些；而且由于graph其实就是一个由操作指令和数据组成的一个数据结构，所以graph可以很方便地被导出并保存起来，甚至之后可以运行在其它非python的环境下（因为graph就是个数据结构，里面定义了一些操作指令和数据，所以任何地方只要能解释这些操作和数据，那么就能运行这个模型）；也正因为graph是个数据结构，所以不同的运行环境可以按照自己的喜好来解释里面的操作和数据，这样一来，解释后生成的代码会更加符合当前运行的环境，这里一来代码的执行效率就更高了。

可能有些同学还无法理解上面所说的“graph是个数据结构...”。这里我打个比方来帮助大家理解。假设graph里面包含了两个数据x和y，另外还包含了一个操作指令“将x和y相加”。当C++的环境要运行这个graph时，“将x和y相加”这个操作就会被翻译成相应的C++代码，当Java环境下要运行这个graph时，就会被解释成相应的Java代码。graph里面只是一些数据和指令，具体怎么执行命令，要看当前运行的环境。

除了上面所说的，graph还有很多内部机制使代码更加高效运行。总之，graph execution可以让tensorflow模型运行得更快，效率更高，更加并行化，更好地适配不同的运行环境和运行设备。

graph 虽然运行很高效，但是代码却没有eager 的简洁，为了兼顾两种模式的优点，所以出现了tf.function。使用tf.function可以将eager 代码一键封装成graph。

既然是封装成graph，那为什么名字里使用function这个单词内，不应该是tf.graph吗？因为tf.function的作用就是将python function转化成包含了graph的tensorflow function。所以使用function这个单词也说得通。下面的代码可以帮助大家更好地理解。

```python
import tensorflow as tf
import timeit
from datetime import datetime

# 定义一个 Python function.
def a_regular_function(x, y, b):
	x = tf.matmul(x, y)
	x = x + b
	return x

# `a_function_that_uses_a_graph` 是一个 TensorFlow `Function`.
a_function_that_uses_a_graph = tf.function(a_regular_function)

# 定义一些tensorflow tensors.
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
# 在python中可以直接调用tenforflow Function。就像使用python自己的function一样。
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
assert(orig_value == tf_function_value)
```

tf.function不仅仅只作用于顶层的python function，它也作用于内嵌的python function。看下面的代码你就能明白了。

```python
def inner_function(x, y, b):
	x = tf.matmul(x, y)
	x = x + b
    return x

# 使用tf.function将`outer_function`变成一个tensorflow `Function`。注意，之前的代码是将tf.function当作是函数来使用，这样是被当作了修饰符来使用。这两种方式都是被支持的。
@tf.function
def outer_function(x):
	y = tf.constant([[2.0], [3.0]])
	b = tf.constant(4.0)
	return inner_function(x, y, b)

# tf.function构建的graph中不仅仅包含了`outer_function`还包含了它里面调用的`inner_function`。
outer_function(tf.constant([[1.0, 2.0]])).numpy()
```

输出结果：

```shell
array([[12.]], dtype=float32)
```

如果你之前使用过tenforflow 1.x，你会察觉到，在2.x中构建graph再也不需要tf.Session和Placeholder了。使代码大大地简洁了。

我们的代码里经常会将python代码和tensorflow代码混在一起。在使用tf.function进行graph转化时，tensorflow的代码会被直接进行转化，而python代码会被一个叫做AutoGraph (tf.autograph)的库来负责进行转化。

同一个tensorflow function可能会生成不同的graph。因为每一个tf.Graph的input输入类型必须是固定的，所以如果在调用tensorflow function时传入了新的数据类型，那么这次的调用就会生成一个新的graph。输入的类型以及维度被称为signature（签名），tensorflow function就是根据签名来生成graph的，遇到新的签名就会生成新的graph。下面的代码可以帮助你理解。

```python
@tf.function
def my_relu(x):
	return tf.maximum(0., x)

# 下面对`my_relu`的3次调用的数据类型都不同，所以生成了3个graph。这3个graph都被保存在my_relu这个tenforflow function中。
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1])) #python数组
print(my_relu(tf.constant([3., -3.]))) # tf数组
```

输出结果：

```shell
tf.Tensor(5.5, shape=(), dtype=float32)
tf.Tensor([1. 0.], shape=(2,), dtype=float32)
tf.Tensor([3. 0.], shape=(2,), dtype=float32)
```

如果相同的输入类型被调用了，那么不会再重新生成新的类型。

```shell
# 下面这两个调用就不会生成新的graph.
print(my_relu(tf.constant(-2.5))) # 这个数据类型与上面的 `tf.constant(5.5)`一样.
print(my_relu(tf.constant([-1., 1.]))) # 这个数据类型与上面的 `tf.constant([3., -3.])`一样。
```

因为一个tensorflow function里面可以包含多个graph，所以说tensorflow function是具备多态性的。这种多态性使得tensorflow function可以任意支持不同的输入类型，非常的灵活；并且由于对每一个输入类型会生成一个特定的graph，这也会让代码执行时更加高效！

下面的代码打印出了3种不同的签名

```python
print(my_relu.pretty_printed_concrete_signatures())
```

输出结果：

```shell
my_relu(x)
 Args:
   x: float32 Tensor, shape=()
 Returns:
   float32 Tensor, shape=()

my_relu(x=[1, -1])
 Returns:
   float32 Tensor, shape=(2,)

my_relu(x)
 Args:
   x: float32 Tensor, shape=(2,)
 Returns:
   float32 Tensor, shape=(2,)
```

上面你已经学会了如何使用tf.function将python function转化为tenforflow function。但要想在实际开发中正确地使用tf.function，还需要学习更多知识。下面我就带领大家来学习学习它们。八十八师的弟兄们，不要退缩，跟着我一起冲啊啊啊！

默认情况下，tenforflow function里面的代码会以graph的模式被执行，但是也可以让它们以eager的模式来执行。大家看下面的代码。

```python
@tf.function
def get_MSE():
	print("Calculating MSE!")

#这条语句就是让下面的代码以eager的模式来执行
tf.config.run_functions_eagerly(True)

get_MSE(y_true, y_pred)

#这条代码就是取消前面的设置
tf.config.run_functions_eagerly(False)
```

某些情况下，同一个tensorflow function在graph与eager模式下会有不同的运行效果。python的print函数就是其中一个特殊情况。看下面的代码。

```python
@tf.function
def get_MSE(y_true, y_pred):
	print("Calculating MSE!")
	sq_diff = tf.pow(y_true - y_pred, 2)
	return tf.reduce_mean(sq_diff)

y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)

error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
```

输出结果：

```shell
Calculating MSE!
```

看到输出结果你是不是很惊讶？get_MSE被调用了3次，但是里面的python print函数只被执行了一次。这是为什么呢？因为python print函数只在创建graph时被执行，而上面的3次调用中输入参数的类型都是一样的，所以只有一个graph被创建了一次，所以python print函数也只会被调用一次。

为了将graph和eager进行对比，下面我们在eager模式下看看输出结果。

```shell
# 开启强制eager模式
tf.config.run_functions_eagerly(True)

error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)

# 取消eager模式
tf.config.run_functions_eagerly(False)
```

输出结果：

```shell
Calculating MSE!
Calculating MSE!
Calculating MSE!
```

看！在eager模式下，print被执行了3次。PS：如果使用tf.print，那么在graph和eager模式下都会打印3次。

graph execution模式还有一个特点，就是它会不执行那些无用的代码。看下面的代码。

```python
def unused_return_eager(x):
	# 当传入的x只包含一个元素时，下面的代码会报错，因为下面的代码是要获取x的第二个元素。
    # PS:索引是从0开始的，1代表第二个元素
	tf.gather(x, [1]) # unused
	return x

try:
	print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
	print(f'{type(e).__name__}: {e}')
```

上面的代码是以eager的模式运行，所以每一行代码都会被执行，所以上面的异常会发生并且会被捕获到。而下面的代码是以graph模式运行的，则不会报异常。因为tf.gather(x, [1])这句代码其实没有任何用途（它只是获取了x的第二个元素，并没有赋值也没有改变任何变量），所以graph模式下它根本就没有被执行，所以也就不会报任何异常了。

```python
@tf.function
def unused_return_graph(x):
	tf.gather(x, [1])
	return x

try:
	print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
	print(f'{type(e).__name__}: {e}')
```

前面我们说graph的执行效率会比eager的要高，那到底高多少呢？其实我们可以用下面的代码来计算graph模式到底能比eager模式提升多少效率。

```python
x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)

def power(x, y):
	result = tf.eye(10, dtype=tf.dtypes.int32)
	for _ in range(y):
		result = tf.matmul(x, result)
	return result

print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000))
```

输出结果：

```shell
Eager execution: 1.8983725069999764
```

用tf.function运行：

```python
power_as_graph = tf.function(power)
```

输出结果：

```shell
Graph execution: 0.5891194120000023
```



从上面的代码可以看出graph比eager的执行时间缩短了近3倍。当然，因具体计算内容不同，效率的提升程度也是不同的。

graph虽然能提升运行效率，但是转化graph时也会有代价。对于某些代码，转化graph所需的时间可能比运行graph的还要长。所以在编写代码时要尽量避免graph的重复转化。如果你发现模型的效率很低，那么可以查查是否存在重复转化。可以通过加入print函数来判断是否存在重复转化（还记得前面我们讲过，每次转化graph时就会调用一次print函数）。看下面的代码。

```python
@tf.function
def a_function_with_python_side_effect(x):
	print("Tracing!") # An eager-only side effect.
	return x * x + tf.constant(2)

print(a_function_with_python_side_effect(tf.constant(2)))
print(a_function_with_python_side_effect(tf.constant(3)))
```

输出结果：

```shell
Tracing!
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(11, shape=(), dtype=int32)
```

可以看出，因为上面两次调用的参数类型是一样的，所以只转化了一次graph，print只被调用了一次。

```python
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))
```

输出结果：

```shell
Tracing!
tf.Tensor(6, shape=(), dtype=int32)

Tracing!
tf.Tensor(11, shape=(), dtype=int32)
```

上面print被调用了2次。啊？为什么？你可以会表示不解~~上面两个参数的类型是一样的啊，为什么还调用了两次print。因为，输入参数是python类型，对于新的python类型每次都会创建一个新的graph。所以最好是用tenforflow的数据类型作为function的输入参数。

最后我给出tf.function相关的几点建议：

* 当需要切换eager和graph模式时，应该使用tf.config.run_functions_eagerly来进行明显的标注。

* 应该在python function的外面创建tenforflow的变量（tf.Variables)，在里面修改它们的值。这条建议同样适用于其它那些使用tf.Variables的tenforflow对象（例如keras.layers,keras.Models,tf.optimizers）。

* 避免函数内部依赖外部定义的python变量。

* 应该尽量将更多的计算量代码包含在一个tf.function中而不是包含在多个tf.function里，这样可以将代码执行效率最大化。

* 最好是用tenforflow的数据类型作为function的输入参数。



# 参考资料

* [一文搞懂tf.function](https://www.bilibili.com/read/cv12856573/)

本文复制自此B站博客。





