# NumPy

* [返回上层目录](python.md#python)
* [NumPy介绍](#NumPy介绍)
* [Array数组和矩阵的基本运算](#Array数组和矩阵的基本运算)
  * [函数库的导入](#函数库的导入)
  * [array常用变量及参数](#array常用变量及参数)
  * [创建矩阵](#创建矩阵)
    * [直接创建矩阵array](#直接创建矩阵array)
    * [创建0矩阵zeros](#创建0矩阵zeros)
    * [创建1矩阵ones](#创建1矩阵ones)
    * [区间内按等差创建矩阵arange](#区间内按等差创建矩阵arange)
    * [区间内按元素个数取值linspace](#区间内按元素个数取值linspace)
    * [创建随机数组random](#创建随机数组random)
    * [矩阵拼接按行vstack](#矩阵拼接按行vstack)
    * [矩阵拼接按列hstack](#矩阵拼接按列hstack)
    * [矩阵分割按列hsplit](#矩阵分割按列hsplit)
    * [矩阵分割按行vsplit](#矩阵分割按行vsplit)
    * [多重复制tile](#多重复制tile)
    * [用reshape创建新数组](#用reshape创建新数组)
  * [基本操作和运算](#基本操作和运算)
    * [求和sum](#求和sum)
    * [求最大值max](#求最大值max)
    * [求最小值min](#求最小值min)
    * [求平均值mean](#求平均值mean)
    * [矩阵行求和sum](#矩阵行求和sum)
    * [矩阵列求和sum](#矩阵列求和sum)
    * [元素求平方a2](#元素求平方a2)
    * [元素求e的n次幂exp](#元素求e的n次幂exp)
    * [元素开根号sqrt](#元素开根号sqrt)
    * [向下取整floor](#向下取整floor)
    * [平坦化数组ravel](#平坦化数组ravel)
    * [查找并修改矩阵特定元素](#查找并修改矩阵特定元素)
  * [Array数组的数据类型](#Array数组的数据类型)
    * [查询数据类型dtype](#查询数据类型dtype)
    * [创建时指定元素类型](#创建时指定元素类型)
    * [转换数据类型astype](#转换数据类型astype)
    * [修改数组的shape](#修改数组的shape)
  * [矩阵运算](#矩阵运算)
    * [矩阵乘法dot](#矩阵乘法dot)
    * [矩阵转置T/transpose](#矩阵转置T/transpose)
    * [求特征值和特征向量eig](#求特征值和特征向量eig)
    * [求矩阵的迹trace](#求矩阵的迹trace)
  * [复制](#复制)
    * [共享内存](#共享内存)
    * [浅复制view](#浅复制view)
    * [深复制copy](#深复制copy)
  * [查询矩阵的维度个数形状等](#查询矩阵的维度个数形状等)
    * [查询维度ndim](#查询维度ndim)
    * [查询元素个数size](#查询元素个数size)
    * [查询矩阵的大小shape](#查询矩阵的大小shape)
  * [判断==](#判断==)
    * [利用==判断数组或矩阵中是否存在某个值](#利用==判断数组或矩阵中是否存在某个值)
    * [一次判断多个条件](#一次判断多个条件)
  * [排序和索引](#排序和索引)
    * [排序sort](#排序sort)
    * [按行或按列找到最大值的索引argmax](#按行或按列找到最大值的索引argmax)
  * [数据归一化](#数据归一化)
* [最简洁的NumPy大纲](#最简洁的NumPy大纲)
  * [NumPy函数和属性](#NumPy函数和属性)
    * [NumPy类型](#NumPy类型)
    * [numpy常用函数](#numpy常用函数)
  * [NumPy.ndarray函数和属性](#NumPy.ndarray函数和属性)
    * [ndarray属性](#ndarray属性)
    * [ndarray函数](#ndarray函数)
    * [ndarray索引/切片方式](#ndarray索引/切片方式)
  * [NumPy.random函数](#NumPy.random函数)
  * [NumPy.linalg函数](#NumPy.linalg函数)
* [参考资料](#参考资料)

# NumPy介绍

标准安装的Python中用列表(list)保存一组值，可以用来当作数组使用，不过由于列表的元素可以是任何对象，因此列表中所保存的是对象的指针。这样为了保存一个简单的[1,2,3]，需要有3个指针和三个整数对象。对于数值运算来说这种结构显然比较浪费内存和CPU计算时间。

此外Python还提供了一个array模块，array对象和列表不同，它直接保存数值，和C语言的一维数组比较类似。但是由于它不支持多维，也没有各种运算函数，因此也不适合做数值运算。

NumPy的诞生弥补了这些不足，NumPy提供了两种基本的对象：ndarray（N-dimensional array object）和 ufunc（universal function object）。ndarray(下文统一称之为数组)是存储单一数据类型的多维数组，而ufunc则是能够对数组进行处理的函数。

Numpy为python提供了快速的多维数组处理能力。用于数学计算，提供了数学函数，但往往用ndarray数据结构，ndarray是数据结构，n维数组，本身内核是C/C++，编程和运行速度快。

**(numpy/scipy)官网**：https://www.scipy.org/

**简要介绍**

NumPy是高性能科学计算和数据分析的基础包，用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多，本身是由C语言开发。这个是很基础的扩展，其余的扩展都是以此为基础。数据结构为ndarray,一般有三种方式来创建。

1. Python对象的转换
2. 通过类似工厂函数numpy内置函数生成：np.arange,np.linspace.....
3. 从硬盘读取，loadtxt

快速入门：[Quickstart tutorial](https://link.zhihu.com/?target=https%3A//docs.scipy.org/doc/numpy-dev/user/quickstart.html)

部分功能如下：

- ndarray, 具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。


- 用于对整组数据进行快速运算的标准数学函数（无需编写循环）。
- 用于读写磁盘数据的工具以及用于操作内存映射文件的工具。
- 线性代数、随机数生成以及傅里叶变换功能。
- 用于集成C、C++、Fortran等语言编写的代码的工具。

**百度百科**

NumPy系统是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵，比Python自身的嵌套列表（nested list structure)结构要高效的多（该结构也可以用来表示矩阵matrix）。据说NumPy将Python相当于变成一种免费的更强大的MatLab系统。

一个用python实现的科学计算包。包括：1、一个强大的N维数组对象Array；2、比较成熟的（广播）函数库；3、用于整合C/C++和Fortran代码的工具包；4、实用的线性代数、傅里叶变换和随机数生成函数。numpy和稀疏矩阵运算包scipy配合使用更加方便。

NumPy（Numeric Python）提供了许多高级的数值编程工具，如：矩阵数据类型、矢量处理，以及精密的运算库。专为进行严格的数字处理而产生。多为很多大型金融公司使用，以及核心的科学计算组织如：Lawrence Livermore，NASA用其处理一些本来使用C++，Fortran或Matlab等所做的任务。

**NumPy库简介**

这两年Python特别火，在一些IT网站上转一圈各处都能看到关于Python的技术类文章，引用[官方的说法](http://www.python.org/doc/essays/blurb.html)，Python就是“一种解释型的、面向对象的、带有动态语义的高级程序设计语言”。Python是一种想让你在编程实现自己想法时感觉不那么碍手碍脚的程序设计语言。Python特点是开发快，语言简洁，可以花较少的代价实现想要的功能，并且编写的程序清晰易懂，比如豆瓣、国外视频网站youtube、社交分享网站Reddit、文件分享服务Dropbox就是使用Python开发的网站，如此看来Python在大规模使用方面应该没什么问题；Python从性能方面来看，有速度要求的话，还是用C++改写关键部分吧。Python在特定领域的表现还是很突出的，比如作为脚本语言、网络爬虫、科学算法等方面。我是因为搞深度学习开始接触Python的，之前学的C++，在遇见Python后简直打开了新世界的大门，码代码的幸福感简直爆棚啊。。。。。。下面开始正题 

**NumPy是使用Python进行科学计算的一个基本库。** 其中包括：

1. 一个强大的N维数组对象Array；
2. 用于集成C / C ++和Fortran代码的工具；
3. 实用的线性代数、傅里叶变换和随机数生成函数。

除了其明显的科学用途，NumPy也可以用作通用数据的高效多维容器。 可以定义任意数据类型。 这允许NumPy无缝，快速地与各种各样的数据库集成。

# Array数组和矩阵的基本运算

**numpy还是很强大的，这里把一些矩阵基本操作做一些整理，方便大家，也方便我自己码代码的时候查找。**

有句话对于我这个初学者来说觉得还是挺符合的，翻书看视频浏览教程贴啊什么的，会发现很多知识点，一开始并不用非得记下都有些什么函数，问题是好像也记不住，学过去之后只要知道这个东西它都能实现什么些什么功能能干些什么事就够了，在你写程序的时候你需要实现什么，这时候再去查找就足够了，用着用着自然就记住了。犹记得那时候苦翻C++ Primer Plus那本书时的悲痛，学语言不用的话真是看后面忘前面。

犹记得那时候苦翻C++ Primer Plus那本书时的悲痛，学语言不用的话真是看后面忘前面。

## 函数库的导入

```python
import numpy #或者
import numpy as np
```

这里`np`就是`numpy`的一个别名。在下面的程序中就可以用`np`来代替`numpy`了。

## array常用变量及参数

numpy.array 常用变量及参数

- dtype变量，用来存放数据类型， 创建数组时可以同时指定。
- shape变量， 存放数组的大小， 这人值是可变的， 只要**确保无素个数不变**的情况下可以任意修改。（-1为自动适配， 保证个数不变）
- reshape方法，创建一个改变了形状的数组，与**原数组是内存共享的**，即都指向同一块内存。

## 创建矩阵

### 直接创建矩阵array

首先需要创建数组才能对其进行其它操作，通过给array函数传递Python的序列对象创建数组，如果传递的是多层嵌套的序列，将创建多维数组(如c):

```python
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array((5, 6, 7, 8))
c = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print (a)
print ('---')
print (b)
print ('---')
print (c)
# 输出
#[1 2 3 4]
#---
#[5 6 7 8]
#---
#[[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
```

若导入`numpy`用的是`import numpy`命令，那么在创建数组的时候用`a = numpy.array([1, 2, 3, 4])`的形式
若导入`numpy`用的是`import numpy as np`命令，那么用`a = np.array([1, 2, 3, 4])`

### 创建0矩阵zeros

```python
import numpy as np
a = np.zeros((3, 4)) # 创建3行4列的0矩阵
b = np.zeros((3, 4), dtype=np.str) # 可以在创建的时候指定数据类型
print(a)
print(a.dtype)
print('---')
print(b)
print(b.dtype)
```

### 创建1矩阵ones

```python
import numpy as np
a = np.ones((3, 4), dtype = int) # 创建3行4列的1矩阵
print(a)
print(a.dtype)
# 输出
#[[1 1 1 1]
# [1 1 1 1]
# [1 1 1 1]]
#int32
```

### 区间内按等差创建矩阵arange

左闭右开

```python
import numpy as np

a = np.arange(10, 30, 5) # 10开始到30，没加5生成一个元素
print(a)
# 输出
#[10 15 20 25]
# 可以通过修改shape属性改变维度，参考上文
b = np.arange(0, 2, 0.3) # 0开始到2，没加0.3生成一个元素
print(b)
# 输出
#[ 0.   0.3  0.6  0.9  1.2  1.5  1.8]
c = np.arange(12).reshape(3, 4) # 从0开始每加1共生成12个元素，并通过reshape设定矩阵大小为3行4列
print(c)
# 输出
#[[ 0  1  2  3]
# [ 4  5  6  7]
# [ 8  9 10 11]]
d = np.random.random((2, 3)) # 生成2行3列矩阵，元素为0-1之间的随机值
print(d)
# 输出
#[[ 0.83492169  0.76747417  0.3277655 ]
# [ 0.99115563  0.32029091  0.69754825]]
```

### 区间内按元素个数取值linspace

```python
import numpy as np
from numpy import pi
print(np.linspace(0, 2*pi, 11)) # 0到2*pi，取11个值
#输出
#[ 0.          0.62831853  1.25663706  1.88495559  2.51327412  3.14159265
#  3.76991118  4.39822972  5.02654825  5.65486678  6.28318531]
print(np.linspace(0, 10, 11)) # 0到10，取11个值
#输出
#[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
```

### 创建随机数组random

这里有详细的numpy.random的用法介绍：

[为什么你用不好Numpy的random函数？](#https://www.cnblogs.com/lemonbit/p/6864179.html)

```python
import numpy as np
a = np.random.random(5)
print(a)
#[ 0.01449445  0.61745786  0.47911107  0.80746168  0.48032829]
b = np.random.random([2,3])
print(b)
#[[ 0.00194012  0.6861311   0.06081057]
# [ 0.1238706   0.48659479  0.76274877]]
```

### 矩阵拼接按行vstack

### 矩阵拼接按列hstack

```python
import numpy as np

a = np.floor(10*np.random.random((2, 2)))
b = np.floor(10*np.random.random((2, 2)))

print (a)
print ('---')
print (b)
print ('---')
print (np.vstack((a, b))) # 按行拼接，也就是竖方向拼接
print ('---')
print (np.hstack((a, b))) # 按列拼接，也就是横方向拼接
#输出：
#[[ 9.  4.]
# [ 4.  4.]]
#---
#[[ 8.  3.]
# [ 9.  8.]]
#---
#[[ 9.  4.]
# [ 4.  4.]
# [ 8.  3.]
# [ 9.  8.]]
#---
#[[ 9.  4.  8.  3.]
# [ 4.  4.  9.  8.]]
```

### 矩阵分割按列hsplit

```python
import numpy as np

a = np.floor(10*np.random.random((2, 6)))

print (a)
print (np.hsplit(a, 3)) # 按列分割，也就是横方向分割，参数a为要分割的矩阵，参数3为分成三份
print ('---')
print (np.hsplit(a, (2, 3, 5))) # 参数(3, 4)为在维度3前面也就是第4列前切一下，在维度4也就是第5列前面切一下
# 输出
#[[ 2.  9.  4.  6.  1.  9.]
# [ 7.  1.  7.  9.  3.  5.]]
#[array([[ 2.,  9.],
#       [ 7.,  1.]]), array([[ 4.,  6.],
#       [ 7.,  9.]]), array([[ 1.,  9.],
#       [ 3.,  5.]])]
#---
#[array([[ 2.,  9.],
#       [ 7.,  1.]]), array([[ 4.],
#       [ 7.]]), array([[ 6.,  1.],
#       [ 9.,  3.]]), array([[ 9.],
#       [ 5.]])]
```

### 矩阵分割按行vsplit

```python
import numpy as np

a = np.floor(10*np.random.random((6, 2)))

print (a)
print (np.vsplit(a, 3)) # 按列分割，也就是横方向分割，参数a为要分割的矩阵，参数3为分成三份
print ('---')
print (np.vsplit(a, (2, 3, 5))) # 参数(3, 4)为在维度3前面也就是第4列前切一下，在维度4也就是第5列前面切一下
# 输出
#[[ 4.  3.]
# [ 9.  1.]
# [ 0.  0.]
# [ 8.  8.]
# [ 0.  2.]
# [ 5.  0.]]
#[array([[ 4.,  3.],
#       [ 9.,  1.]]), array([[ 0.,  0.],
#       [ 8.,  8.]]), array([[ 0.,  2.],
#       [ 5.,  0.]])]
#---
#[array([[ 4.,  3.],
#       [ 9.,  1.]]), array([[ 0.,  0.]]), array([[ 8.,  8.],
#       [ 0.,  2.]]), array([[ 5.,  0.]])]
```

### 多重复制tile

```python
import numpy as np
a = np.array([5,10,15])
print(a)
print('---')
b = np.tile(a, (4,3))# 参数(4, 3)为按行复制4倍，按列复制3倍
print(b)
print(b.shape)
print(type(b))
#输出
#[ 5 10 15]
#---
#[[ 5 10 15  5 10 15  5 10 15]
# [ 5 10 15  5 10 15  5 10 15]
# [ 5 10 15  5 10 15  5 10 15]
# [ 5 10 15  5 10 15  5 10 15]]
#(4, 9)
#<class 'numpy.ndarray'>
```

### 用reshape创建新数组

使用数组的reshape方法，可以创建一个改变了尺寸的新数组，原数组的shape保持不变：

```python
import numpy as np

a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(a)
print('---')
b = a.reshape((4, -1))
#b.shape = -1, 3
#b.shape = (4, 3)
#b.shape = 4, 3
print('b = ', b)
print (b.shape)
print('a = ', a)
# 输出
#[[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
#---
#b =  [[ 1  2  3]
# [ 4  4  5]
# [ 6  7  7]
# [ 8  9 10]]
#(4, 3)
#a =  [[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
```



## 基本操作和运算

### 求和sum()

### 求最大值max()

### 求最小值min()

### 求平均值mean()

```python
import numpy as np
test1 = np.array([[5, 10, 15],
                  [20, 25, 30],
                  [35, 40, 45]])
print(test1.sum())
# 输出 225
print(test1.max())
# 输出 45
print(test1.min())
# 输出 5
print(test1.mean())
# 输出 25.0
```

### 矩阵行求和sum(axis=1)

```python
import numpy as np
test1 = np.array([[5, 10, 15],
                  [20, 25, 30],
                  [35, 40, 45]])
print(test1.sum(axis=1))
# 输出 array([30, 75, 120])
```

### 矩阵列求和sum(axis=0)

```python
import numpy as np
test1 = np.array([[5, 10, 15],
                  [20, 25, 30],
                  [35, 40, 45]])
peint(test1.sum(axis=0))
# 输出 array([60, 75, 90])
```

### 元素求平方a2

```python
import numpy as np
a = np.arange(4)
print (a)
print (a**2)
# 输出 [0 1 2 3]
#      [0 1 4 9]
```

### 元素求e的n次幂exp

### 元素开根号sqrt

```python
import numpy as np
test = np.arange(3)
print (test)
print (np.exp(test)) #e的n次幂
print (np.sqrt(test)) #开根号
# 输出 [0 1 2]
#      [1. 2.71828183 7.3890561]
#      [0 1. 1.41421356]
```

### 向下取整floor

```python
import numpy as np

testRandom = 10*np.random.random((2,3))
testFloor = np.floor(testRandom)

print(testRandom)
print (testFloor)

# 输出 [[ 4.1739405   3.61074364  0.96858834]
#       [ 4.83959291  8.786262    0.74719657]]
#      [[ 4.  3.  0.]
#       [ 4.  8.  0.]]
```

### 平坦化数组ravel

```python
import numpy as np

test = np.array([[2,3,4],[5,6,7]])
test.shape = (6, 2)
print(test)
print(test.T)

# 输出 [[2 3]
#      [4 5]]
#     [2 3 4 5]
```

### 查找并修改矩阵特定元素

例如下面代码中，x_data是我代码中的一个矩阵，但是矩阵数据中有缺失值是用?表示的，我要做一些数据处理，就需要把?换掉，比如换成0。(注：这里我换成了更简单的矩阵)

```python
import numpy as np

a = np.arange(5)

print (a)
print('----')

a[a==0]=100

print(a)
# 输出
#[0 1 2 3 4]
#----
#[100   1   2   3   4]
```

## Array数组的数据类型

关于数据类型：List中的元素可以是不同的数据类型，而Array和Series中则只允许存储相同的数据类型，这样可以更有效的使用内存，提高运算效率。

list类型是python自带的类型，下面的结构就是list类型：

```python
list1 = [1, 2, 3, 4, 5 ];
print(list1)
print(type(list1))
#[1, 2, 3, 4, 5]
#<class 'list'>
```

array数组的数据类型有下面几种

```
bool -- True , False
int -- int16 , int32 , int64
float -- float16 , float32 , float64
string -- string , unicode
```

### 查询数据类型dtype

```python
import numpy as np

a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])

print (a)
print (a.dtype)
print(type(a))
# 输出
#[[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
# int32
#<class 'numpy.ndarray'>
```

### 创建时指定元素类型

```python
import numpy as np
a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
b = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype='str')
print (a)
print("a.dtype = ", a.dtype)
print ('---')
print (b)
print("b.dtype = ", b.dtype)
# 输出
#[[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
#a.dtype =  int32
#---
#[['1' '2' '3' '4']
# ['4' '5' '6' '7']
# ['7' '8' '9' '10']]
#b.dtype =  <U2 ???????这是什么鬼？为什么不是str？
```

### 转换数据类型astype

```python
import numpy as np
b = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype='str')
print (b)
print("b.dtype = ", b.dtype)
b = b.astype(int)
print (b)
print("b.dtype = ", b.dtype)
# 输出
#[['1' '2' '3' '4']
# ['4' '5' '6' '7']
# ['7' '8' '9' '10']]
#b.dtype =  <U2
#[[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
#b.dtype =  int32
```

### 修改数组的shape

通过修改数组的shape属性，在保持数组元素个数不变的情况下，改变数组每个轴的长度。下面的例子将数组b的shape改为(4, 3)，从(3, 4)改为(4, 3)并不是对数组进行转置，而只是改变每个轴的大小，数组元素在内存中的位置并没有改变：

```python
import numpy as np

b = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(b)
print('---')
b.shape = 4, -1#当某个轴的元素为-1时，将根据数组元素的个数自动计算该轴的长度
#b.shape = -1, 3
#b.shape = (4, 3)
#b.shape = 4, 3
print(b)
print (b.shape)
# 输出
#[[ 1  2  3  4]
# [ 4  5  6  7]
# [ 7  8  9 10]]
#---
#[[ 1  2  3]
# [ 4  4  5]
# [ 6  7  7]
# [ 8  9 10]]
#(4, 3)
```

## 矩阵运算

### 矩阵乘法dot()

```python
import numpy as np
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])
print (a*b) # 对应位置元素相乘
print (a.dot(b)) # 矩阵乘法
print (np.dot(a, b)) # 矩阵乘法，同上
# 输出 [[5 12]
#       [21 32]]
#      [[19 22]
#       [43 50]]
#      [[19 22]
#       [43 50]] 
```

### 矩阵转置T/transpose()

```python
import numpy as np

test = np.array([[2,3,4],[5,6,7]])
test.shape = (3, -1)

print(test)
print(test.T)
print(test.transpose())
# 输出 [[2 3]
#       [4 5]
#       [6 7]]
#     [[2 4 6]
#      [3 5 7]]
#[[2 4 6]
# [3 5 7]]
```

### 求特征值和特征向量eig()

```python
import numpy as np
import numpy.linalg as nplg
a = np.array([[1,0],[2,3]])

print(a)
print(nplg.eig(a))

eigValues = np.array([ 3.,  1.])
eigVectors = np.array([[ 0.        ,  0.70710678],
                       [ 1.        , -0.70710678]])

print(a.dot(eigVectors[:,0]))
print(eigValues[0]*eigVectors[:,0])

print(a.dot(eigVectors[:,1]))
print(eigValues[1]*eigVectors[:,1])
```

### 求矩阵的迹trace()

```python
import numpy as np

test = np.array([[2,3,4],[5,6,7],[8,9,10]])

print(test)
print(test.trace())
print(np.trace(test))
# 输出
#[[ 2  3  4]
# [ 5  6  7]
# [ 8  9 10]]
#18
#18
```

## 复制

### 共享内存=

a和b共享数据存储内存区域，因此修改其中任意一个数组的元素都会同时修改另外一个数组或矩阵的内容：

```python
import numpy as np
a = np.arange(12)
b = a

print (a)
print (b)
print (b is a) # 判断b是a？
# 输出 [ 0  1  2  3  4  5  6  7  8  9 10 11]
#    [ 0  1  2  3  4  5  6  7  8  9 10 11]
#    True
b.shape = 3, 4
b[0,0] = 100;
print (a.shape)
# 输出 (3, 4)
print(a)
# 输出[[100   1   2   3]
# [  4   5   6   7]
# [  8   9  10  11]]
print (id(a))#内存地址
print (id(b))
# 输出 201372576
#      201372576
```

### 浅复制view()

不是同一地址，但是会被改变

```python
# The view method creates a new array object that looks at the same data.

import numpy as np
a = np.arange(12)
b = a.view() # b是新创建出来的数组，但是b和a共享数据

print(b is a) # 判断b是a？
# 输出 False
print (b)
# 输出 [ 0  1  2  3  4  5  6  7  8  9 10 11]
b.shape = 2, 6 # 改变b的shape，a的shape不会受影响
print (a.shape)
print (b)
# 输出 (12,)
#[[ 0  1  2  3  4  5]
# [ 6  7  8  9 10 11]]
b[0, 4] = 1234 # 改变b第1行第5列元素为1234，a对应位置元素受到影响
print (b)
# 输出 [[   0    1    2    3 1234    5]
#         [   6    7    8    9   10   11]]
print (a)
# 输出 [   0    1    2    3 1234    5    6    7    8    9   10   11]
```

### 深复制copy()

不是同一地址，也不会被改变

```python
# The copy method makes a complete copy of the array and its data.

import numpy as np
a = np.arange(12)
a.shape = 3, 4
a[1, 0] = 1234

c = a.copy()
print(c is a)
c[0, 0] = 9999 # 改变c元素的值，不会影响a的元素
print (c)
print (a)
# 输出
#False
#[[9999    1    2    3]
# [1234    5    6    7]
# [   8    9   10   11]]
#[[   0    1    2    3]
# [1234    5    6    7]
# [   8    9   10   11]]
```

## 查询矩阵的维度个数形状等

### 查询维度ndim

```python
import numpy as np
a = np.array([[5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])
print(a.ndim)
#输出
#2
```

### 查询元素个数size

```python
import numpy as np
a = np.array([[5, 10, 15],
       [20, 25, 30],
       [35, 40, 45]])
print(a.size)
# 输出 9
```

### 查询矩阵的大小shape

```python
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print (a.shape)
print ('---')
print (b.shape)
# 输出
#(4,)
#---
#(3, 4)
```

(4, )shape有一个元素即为一维数组，数组中有4个元素 
(3, 4)shape有两个元素即为二维数组，数组为3行4列

## 判断==

### 利用==判断数组或矩阵中是否存在某个值

```python
import numpy as np
test = np.array([5,10,15,20])
print(test >= 15)
#[False False  True  True]
print(test*(test >= 15) + 15)
#[15 15 30 35]
```

将判断结果赋给某个变量

```python
import numpy as np
test = np.array([[ 5, 10, 15],
                 [20, 25, 30],
                 [35, 40, 45]])
# 判断test中第三列等于30的元素
secondColumn30 = (test[:,2] == 30)
print(secondColumn30)
# 取secondColumn30为True的行，打印
print(test[secondColumn30,:])
# 输出
#[False  True False]
#[[20 25 30]]
```

### 一次判断多个条件

```python
import numpy as np
test = np.array([5,10,15,20])
equalTo15And20 = (test == 15) & (test == 20)#同时等于15又等于20的元素，显然是不存在的
print(equalTo15And20)
#[False False False False]
equalTo15Or20 = (test == 15) | (test == 20)
print(equalTo15Or20)#等于15或者等于20的元素
#[False False  True  True]
test[equalTo15Or20] = 100
print(test)
#[  5  10 100 100]
```

## 排序和索引

### 排序sort

```python
import numpy as np
a = np.array([[4,3,5,],[1,2,1]])
print (a)
#[[4 3 5]
# [1 2 1]]
b = np.sort(a, axis=1) # 对a按每行中元素从小到大排序
print(b)
#[[3 4 5]
# [1 1 2]]
c = a.copy()# 深拷贝
c.sort(axis=0)
print(c)
#[[1 2 1]
# [4 3 5]]
print(a)# a没有随着c变化
```

对单行数组进行排序的另外一种方法

```python
import numpy as np
a = np.array([4, 3, 1, 2])
b = np.argsort(a) # 求a从小到大排序的坐标
print (b)
print (a[b]) # 按求出来的坐标顺序排序
print(a[np.argsort(a)])
```

### 按行或按列找到最大值的索引argmax

```python
import numpy as np
data = np.sin(np.arange(20)).reshape(5, 4)
print (data)
ind = data.argmax(axis=0) # 按列得到每一列中最大元素的索引，axis=1为按行
print (ind)
data_max = data[ind, range(data.shape[1])] # 将最大值取出来
print (data_max)

print(data.max(axis=0)) #也可以直接取最大值
```

## 数据归一化

```python
import numpy as np
a= 10*np.random.random((5,5))
b = a.copy()
print(a)
print("---")
amin, amax = a.min(), a.max()
print(amin, amax)
print("---")
a = (a-amin)/(amax-amin)
print(a)
print("---")
b = (b-b.min())/(b.max() - b.min())
print(b)
# 输出
#[[ 7.33412218  7.62784714  7.06761515  6.56230239  3.76404535]
# [ 2.68197834  6.02335055  4.67169946  5.08454875  6.97170333]
# [ 4.02393841  3.9723266   1.82841784  7.6049149   0.38845819]
# [ 6.55672442  1.40986757  1.14657213  3.0356768   9.55024583]
# [ 1.06007416  1.23600072  0.97610622  8.8232397   0.39996053]]
#---
#0.38845818848 9.5502458299
#---
#[[ 0.75811231  0.7901721   0.72902333  0.67386895  0.36844198]
# [ 0.25033544  0.61504289  0.46751152  0.51257361  0.71855465]
# [ 0.39680905  0.39117567  0.15717016  0.78766907  0.        ]
# [ 0.67326012  0.11148582  0.08274738  0.28894128  1.        ]
# [ 0.07330622  0.09250842  0.0641412   0.920648    0.00125547]]
#---
#[[ 0.75811231  0.7901721   0.72902333  0.67386895  0.36844198]
# [ 0.25033544  0.61504289  0.46751152  0.51257361  0.71855465]
# [ 0.39680905  0.39117567  0.15717016  0.78766907  0.        ]
# [ 0.67326012  0.11148582  0.08274738  0.28894128  1.        ]
# [ 0.07330622  0.09250842  0.0641412   0.920648    0.00125547]]
```

# 最简洁的NumPy大纲

## NumPy函数和属性

### NumPy类型

|      类型      | 类型代码  |            说明             |
| :----------: | :---: | :-----------------------: |
|  int8、uint8  | i1、u1 |     有符号和无符号8位整型（1字节）      |
| int16、uint16 | i2、u2 |     有符号和无符号16位整型（2字节）     |
| int32、uint32 | i4、u4 |     有符号和无符号32位整型（4字节）     |
| int64、uint64 | i8、u8 |     有符号和无符号64位整型（8字节）     |
|   float16    |  f2   |          半精度浮点数           |
|   float32    | f4、f  |          单精度浮点数           |
|   float64    | f8、d  |          双精度浮点数           |
|   float128   | f16、g |          扩展精度浮点数          |
|  complex64   |  c8   |       分别用两个32位表示的复数       |
|  complex128  |  c16  |       分别用两个64位表示的复数       |
|  complex256  |  c32  |      分别用两个128位表示的复数       |
|     bool     |   ?   |            布尔型            |
|    object    |   O   |         python对象          |
|    string    |  Sn   |   固定长度字符串，每个字符1字节，如S10    |
|   unicode    |  Un   | 固定长度Unicode，字节数由系统决定，如U10 |

### numpy常用函数

| **生成函数**                                 | **作用**                                   |
| :--------------------------------------- | :--------------------------------------- |
| np.array( x)<br>np.array( x, dtype)      | 将输入数据转化为一个ndarray<br>将输入数据转化为一个类型为type的ndarray |
| np.asarray( array )                      | 将输入数据转化为一个新的（copy）ndarray                |
| np.ones( N )<br>np.ones( N, dtype)<br>np.ones_like( ndarray ) | 生成一个N长度的一维全一ndarray<br>生成一个N长度类型是dtype的一维全一ndarray<br>生成一个形状与参数相同的全一ndarray |
| np.zeros( N)<br>np.zeros( N, dtype)<br>np.zeros_like(ndarray) | 生成一个N长度的一维全零ndarray<br>生成一个N长度类型位dtype的一维全零ndarray<br>类似np.ones_like( ndarray ) |
| np.empty( N )<br>np.empty( N, dtype)<br>np.empty(ndarray) | 生成一个N长度的未初始化一维ndarray<br>生成一个N长度类型是dtype的未初始化一维ndarray<br>类似np.ones_like( ndarray ) |
| np.eye( N )<br>np.identity( N )          | 创建一个N * N的单位矩阵（对角线为1，其余为0）               |
| np.arange( num)<br>np.arange( begin, end)<br>np.arange( begin, end, step) | 生成一个从0到num-1步数为1的一维ndarray<br>生成一个从begin到end-1步数为1的一维ndarray<br>生成一个从begin到end-step的步数为step的一维ndarray |
| np.mershgrid(ndarray, ndarray,...)       | 生成一个ndarray * ndarray * ...的多维ndarray    |
| np.where(cond, ndarray1, ndarray2)       | 根据条件cond，选取ndarray1或者ndarray2，返回一个新的ndarray |
| np.in1d(ndarray, [x,y,...])              | 检查ndarray中的元素是否等于[x,y,...]中的一个，返回bool数组  |
|                                          |                                          |
| **矩阵函数**                                 | **说明**                                   |
| np.diag( ndarray)<br>np.diag( [x,y,...]) | 以一维数组的形式返回方阵的对角线（或非对角线）元素<br>将一维数组转化为方阵（非对角线元素为0） |
| np.dot(ndarray, ndarray)                 | 矩阵乘法                                     |
| np.trace( ndarray)                       | 计算对角线元素的和                                |
|                                          |                                          |
|                                          |                                          |
| **排序函数**                                 | **说明**                                   |
| np.sort( ndarray)                        | 排序，返回副本                                  |
| np.unique(ndarray)                       | 返回ndarray中的元素，排除重复元素之后，并进行排序             |
| np.intersect1d( ndarray1, ndarray2)<br>np.union1d( ndarray1, ndarray2)<br>np.setdiff1d( ndarray1, ndarray2)<br>np.setxor1d( ndarray1, ndarray2) | 返回二者的交集并排序。<br>返回二者的并集并排序。<br>返回二者的差。<br>返回二者的对称差 |
|                                          |                                          |
| **一元计算函数**                               | **说明**                                   |
| np.abs(ndarray)<br>np.fabs(ndarray)      | 计算绝对值计算绝对值（非复数）                          |
| np.mean(ndarray)                         | 求平均值                                     |
| np.sqrt(ndarray)                         | 计算x^0.5                                  |
| np.square(ndarray)                       | 计算x^2                                    |
| np.exp(ndarray)                          | 计算e^x                                    |
| log、log10、log2、log1p                     | 计算自然对数、底为10的log、底为2的log、底为(1+x)的log      |
| np.sign(ndarray)                         | 计算正负号：1（正）、0（0）、-1（负）                    |
| np.ceil(ndarray)<br>np.floor(ndarray)<br>np.rint(ndarray) | 计算大于等于改值的最小整数<br>计算小于等于该值的最大整数<br>四舍五入到最近的整数，保留dtype |
| np.modf(ndarray)                         | 将数组的小数和整数部分以两个独立的数组方式返回                  |
| np.isnan(ndarray)                        | 返回一个判断是否是NaN的bool型数组                     |
| np.isfinite(ndarray)<br>np.isinf(ndarray) | 返回一个判断是否是有穷（非inf，非NaN）的bool型数组<br>返回一个判断是否是无穷的bool型数组 |
| cos、cosh、sin、sinh、tan、tanh               | 普通型和双曲型三角函数                              |
| arccos、arccosh、arcsin、arcsinh、arctan、arctanh | 反三角函数和双曲型反三角函数                           |
| np.logical_not(ndarray)                  | 计算各元素not x的真值，相当于-ndarray                |
| **多元计算函数**                               | **说明**                                   |
| np.add(ndarray, ndarray)<br>np.subtract(ndarray, ndarray)<br>np.multiply(ndarray, ndarray)<br>np.divide(ndarray, ndarray)<br>np.floor_divide(ndarray, ndarray)<br>np.power(ndarray, ndarray)<br>np.mod(ndarray, ndarray) | 相加<br>相减<br>乘法<br>除法<br>圆整除法（丢弃余数）<br>次方<br>求模 |
| np.maximum(ndarray, ndarray)<br>np.fmax(ndarray, ndarray)<br>np.minimun(ndarray, ndarray)<br>np.fmin(ndarray, ndarray) | 求最大值<br>求最大值（忽略NaN）<br>求最小值<br>求最小值（忽略NaN） |
| np.copysign(ndarray, ndarray)            | 将参数2中的符号赋予参数1                            |
| np.greater(ndarray, ndarray)<br>np.greater_equal(ndarray, ndarray)<br>np.less(ndarray, ndarray)<br>np.less_equal(ndarray, ndarray)<br>np.equal(ndarray, ndarray)<br>np.not_equal(ndarray, ndarray) | ><br>>=<br><<br><=<br>==<br>!=           |
| logical_and(ndarray, ndarray)<br>logical_or(ndarray, ndarray)<br>logical_xor(ndarray, ndarray) | &<br>单根竖线<br>^                           |
| np.dot( ndarray, ndarray)                | 计算两个ndarray的矩阵内积                         |
| np.ix_([x,y,m,n],...)                    | 生成一个索引器，用于Fancy indexing(花式索引)           |
|                                          |                                          |
| **文件读写**                                 | **说明**                                   |
| np.save(string, ndarray)                 | 将ndarray保存到文件名为 [string].npy 的文件中（无压缩）   |
| np.savez(string, ndarray1, ndarray2, ...) | 将所有的ndarray压缩保存到文件名为[string].npy的文件中     |
| np.savetxt(sring, ndarray, fmt, newline=‘\n‘) | 将ndarray写入文件，格式为fmt                      |
| np.load(string)                          | 读取文件名string的文件内容并转化为ndarray对象（或字典对象）     |
| np.loadtxt(string, delimiter)            | 读取文件名string的文件内容，以delimiter为分隔符转化为ndarray |

## NumPy.ndarray函数和属性

### ndarray属性

|   ndarray函数   |        功能         |
| :-----------: | :---------------: |
| ndarray.ndim  |   获取ndarray的维数    |
| ndarray.shape | 获取ndarray各个维度的长度  |
| ndarray.dtype | 获取ndarray中元素的数据类型 |
|   ndarray.T   |   简单转置矩阵ndarray   |

### ndarray函数

| **函数**                                   | **说明**                                   |
| ---------------------------------------- | ---------------------------------------- |
| ndarray.astype(dtype)                    | 转换类型，若转换失败则会出现TypeError                  |
| ndarray.copy()                           | 复制一份ndarray(新的内存空间)                      |
| ndarray.reshape((N,M,...))               | 将ndarray转化为N*M*...的多维ndarray（非copy）      |
| ndarray.transpose((xIndex,yIndex,...))   | 根据维索引xIndex,yIndex...进行矩阵转置，依赖于shape，不能用于一维矩阵（非copy） |
| ndarray.swapaxes(xIndex,yIndex)          | 交换维度（非copy）                              |
|                                          |                                          |
| **计算函数**                                 | **说明**                                   |
| ndarray.mean( axis=0 )                   | 求平均值                                     |
| ndarray.sum( axis= 0)                    | 求和                                       |
| ndarray.cumsum( axis=0)<br>ndarray.cumprod( axis=0) | 累加 <br>累乘                                |
| ndarray.std()<br>ndarray.var()           | 方差<br>标准差                                |
| ndarray.max()<br>ndarray.min()           | 最大值<br>最小值                               |
| ndarray.argmax()<br>darray.argmin()      | 最大值索引<br>最小值索引                           |
| ndarray.any()<br>ndarray.all()           | 是否至少有一个True是否全部为True                     |
| ndarray.dot( ndarray)                    | 计算矩阵内积                                   |
| **排序函数**                                 | **说明**                                   |
| ndarray.sort(axis=0)                     | 排序，返回源数据                                 |

### ndarray索引/切片方式

| ndarray[n]                               | 选取第n+1个元素               |
| ---------------------------------------- | ----------------------- |
| ndarray[n:m]                             | 选取第n+1到第m个元素            |
| ndarray[:]                               | 选取全部元素                  |
| ndarray[n:]                              | 选取第n+1到最后一个元素           |
| ndarray[:n]                              | 选取第0到第n个元素              |
| ndarray[ bool_ndarray ]<br>注：bool_ndarray表示bool类型的ndarray | 选取为true的元素              |
| ndarray[[x,y,m,n]]...                    | 选取顺序和序列为x、y、m、n的ndarray |
| ndarray[n,m]<br>ndarray\[n][m]           | 选取第n+1行第m+1个元素          |
| ndarray[n,m,...]<br>ndarray\[n][m]....   | 选取n行n列....的元素           |

## NumPy.random函数

| **函数**                                   | **说明**                                   |
| ---------------------------------------- | ---------------------------------------- |
| seed()<br>seed(int)<br>seed(ndarray)     | 确定随机数生成种子                                |
| permutation(int)<br>permutation(ndarray) | 返回一个一维从0~9的序列的随机排列<br>返回一个序列的随机排列        |
| shuffle(ndarray)                         | 对一个序列就地随机排列                              |
| rand(int)<br>randint(begin,end,num=1)    | 产生int个均匀分布的样本值<br>从给定的begin和end随机选取num个整数 |
| randn(N, M, ...)                         | 生成一个N*M*...的正态分布（平均值为0，标准差为1）的ndarray    |
| normal(size=(N,M,...))                   | 生成一个N*M*...的正态（高斯）分布的ndarray             |
| beta(ndarray1,ndarray2)                  | 产生beta分布的样本值，参数必须大于0                     |
| chisquare()                              | 产生卡方分布的样本值                               |
| gamma()                                  | 产生gamma分布的样本值                            |
| uniform()                                | 产生在[0,1)中均匀分布的样本值                        |

## NumPy.linalg函数

| **函数**                        | **说明**                         |
| ----------------------------- | ------------------------------ |
| det(ndarray)                  | 计算矩阵列式                         |
| eig(ndarray)                  | 计算方阵的本征值和本征向量                  |
| inv(ndarray)<br>pinv(ndarray) | 计算方阵的逆<br>计算方阵的Moore-Penrose伪逆 |
| qr(ndarray)                   | 计算qr分解                         |
| svd(ndarray)                  | 计算奇异值分解svd                     |
| solve(ndarray)                | 解线性方程组Ax = b，其中A为方阵            |
| lstsq(ndarray)                | 计算Ax=b的最小二乘解                   |

# numpy数据保存与读取

主要介绍了Python Numpy中数据的常用保存与读取方法，本文给大家介绍的非常详细。

在经常性读取大量的数值文件时(比如深度学习训练数据),可以考虑现将数据存储为`Numpy`格式,然后直接使用Numpy去读取,速度相比为转化前快很多。

下面就常用的保存数据到二进制文件和保存数据到文本文件进行介绍:

## 保存为二进制文件(.npy/.npz)

### numpy.save

```python
numpy.save
```

保存一个数组到一个二进制的文件中,保存格式是.npy

参数介绍

> numpy.save(file, arr, allow_pickle=True, fix_imports=True)
> file:文件名/文件路径
> arr:要存储的数组
> allow_pickle:布尔值,允许使用Python pickles保存对象数组(可选参数,默认即可)
> fix_imports:为了方便Pyhton2中读取Python3保存的数据(可选参数,默认即可)

**使用**

```python
>>> import numpy as np 
#生成数据 
>>> x=np.arange(10) 
>>> x 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
 
#数据保存 
>>> np.save('save_x',x) 
 
#读取保存的数据 
>>> np.load('save_x.npy') 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
```

### numpy.savez

```python
numpy.savez
```

这个同样是保存数组到一个二进制的文件中,但是厉害的是,它可以保存多个数组到同一个文件中,保存格式是.npz,它其实就是多个前面`np.save`的保存的npy,再通过打包(**未压缩**)的方式把这些文件归到一个文件上,不行你去解压npz文件就知道了,里面是就是自己保存的多个npy。

参数介绍

> numpy.savez(file, *args, **kwds)
> file:文件名/文件路径
> *args:要存储的数组,可以写多个,如果没有给数组指定Key,Numpy将默认从'arr_0','arr_1'的方式命名
> kwds:(可选参数,默认即可)

**使用**

```python
>>> import numpy as np 
#生成数据 
>>> x=np.arange(10) 
>>> x 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
>>> y=np.sin(x) 
>>> y 
array([ 0.  , 0.84147098, 0.90929743, 0.14112001, -0.7568025 , 
  -0.95892427, -0.2794155 , 0.6569866 , 0.98935825, 0.41211849]) 
  
#数据保存 
>>> np.save('save_xy',x,y) 
 
#读取保存的数据 
>>> npzfile=np.load('save_xy.npz') 
>>> npzfile #是一个对象,无法读取 
<numpy.lib.npyio.NpzFile object at 0x7f63ce4c8860> 
 
#按照组数默认的key进行访问 
>>> npzfile['arr_0'] 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
>>> npzfile['arr_1'] 
array([ 0.  , 0.84147098, 0.90929743, 0.14112001, -0.7568025 , 
  -0.95892427, -0.2794155 , 0.6569866 , 0.98935825, 0.41211849]) 
```

更加神奇的是,你可以不适用Numpy默认给数组的Key,而是自己给数组有意义的Key,这样就可以不用去猜测自己加载数据是否是自己需要的.

```
#数据保存 
>>> np.savez('newsave_xy',x=x,y=y) 
 
#读取保存的数据 
>>> npzfile=np.load('newsave_xy.npz') 
 
#按照保存时设定组数key进行访问 
>>> npzfile['x'] 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
>>> npzfile['y'] 
array([ 0.  , 0.84147098, 0.90929743, 0.14112001, -0.7568025 , 
  -0.95892427, -0.2794155 , 0.6569866 , 0.98935825, 0.41211849]) 
```

简直不能太爽,深度学习中,有时候你保存了训练集,验证集,测试集,还包括他们的标签,用这个方式存储起来,要啥加载啥,文件数量大大减少,也不会到处改文件名去.

### numpy.savez_compressed

```python
numpy.savez_compressed
```

这个就是在前面`numpy.savez`的基础上加了压缩,前面我介绍时尤其注明numpy.savez是得到的文件打包,不压缩的.这个文件就是对文件进行打包时使用了压缩,可以理解为压缩前各npy的文件大小不变,使用该函数比前面的numpy.savez得到的npz文件更小.

注:函数所需参数和`numpy.savez`一致,用法完成一样.

## 保存到文本文件

### numpy.savetxt

```python
numpy.savetxt
```

保存数组到文本文件上,可以直接打开查看文件里面的内容.

参数介绍

> numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
> fname:文件名/文件路径,如果文件后缀是.gz,文件将被自动保存为.gzip格式,np.loadtxt可以识别该格式
> X:要存储的1D或2D数组
> fmt:控制数据存储的格式
> delimiter:数据列之间的分隔符
> newline:数据行之间的分隔符
> header:文件头步写入的字符串
> footer:文件底部写入的字符串
> comments:文件头部或者尾部字符串的开头字符,默认是'#'
> encoding:使用默认参数

**使用**

```python
>>> import numpy as np 
#生成数据 
>>> x = y = z = np.ones((2,3)) 
>>> x 
array([[1., 1., 1.], 
  [1., 1., 1.]]) 
  
#保存数据 
np.savetxt('test.out', x) 
np.savetxt('test1.out', x,fmt='%1.4e') 
np.savetxt('test2.out', x, delimiter=',') 
np.savetxt('test3.out', x,newline='a') 
np.savetxt('test4.out', x,delimiter=',',newline='a') 
np.savetxt('test5.out', x,delimiter=',',header='abc') 
np.savetxt('test6.out', x,delimiter=',',footer='abc') 
```

保存下来的文件都是友好的,可以直接打开看看有什么变化.

### numpy.loadtxt

```python
numpy.loadtxt
```

根据前面定制的保存格式,相应的加载数据的函数也得变化.

参数介绍

> numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes')
> fname:文件名/文件路径,如果文件后缀是.gz或.bz2,文件将被解压,然后再载入
> dtype:要读取的数据类型
> comments:文件头部或者尾部字符串的开头字符,用于识别头部,尾部字符串
> delimiter:划分读取上来值的字符串
> converters:数据行之间的分隔符
> .......后面不常用的就不写了

使用

```
np.loadtxt('test.out') 
np.loadtxt('test2.out', delimiter=',') 
```



# 参考资料

* [NumPy的百度百科](https://baike.baidu.com/item/numpy/5678437?fr=aladdin)


* [如何系统地学习Python 中 matplotlib, numpy, scipy, pandas？-知乎](https://www.zhihu.com/question/37180159)


* [NumPy学习笔记（1）--NumPy简介](http://blog.csdn.net/lwplwf/article/details/55251813)

NumPy介绍就是复制自这里。

* [NumPy学习笔记（2）--Array数组和矩阵基本运算](http://blog.csdn.net/lwplwf/article/details/55506896)


* [NumPy学习笔记（3）--排序与索引](http://blog.csdn.net/lwplwf/article/details/55805602)
* [NumPy学习笔记（4）--数据归一化](http://blog.csdn.net/lwplwf/article/details/55806834)

Array数组和矩阵的基本运算都是复制或改编自这里。

* [Python-NumPy最简洁的大纲](http://www.cnblogs.com/keepgoingon/p/7137448.html)

最简洁的NumPy大纲都是复制自这里。

* [Numpy中数据的常用的保存与读取方法](https://www.cnblogs.com/wushaogui/p/9142019.html)

“numpy数据保存与读取”复制自这里。

===

[搭建模型第一步：你需要预习的NumPy基础都在这了](https://zhuanlan.zhihu.com/p/38786013)

[图解NumPy，这是理解数组最形象的一份教程了](https://mp.weixin.qq.com/s/_r1czXpTRL4zFfaBL6XiVA)

