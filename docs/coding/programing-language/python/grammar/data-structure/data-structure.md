# 数据结构

* [返回上层目录](../python.md)
* [range()](#range())
* [字符串](#字符串)
  * [字符串操作](#字符串操作)
  * [字符串常用内置方法](#字符串常用内置方法)
* [列表](#列表)
  * [对列表进行增删改查](#对列表进行增删改查)
  * [列表去重](#列表去重)
  * [切片操作](#切片操作)
* [元组](#元组)
* [字典](#字典)
  * [基本操作](#基本操作)
  * [字典的增删改查操作](#字典的增删改查操作)
  * [使用循环来遍历字典](#使用循环来遍历字典)
  * [清空字典](#清空字典)
* [集合](#集合)



# range()

range(开始，结束)，返回一个序列，**左闭右开**

```python
for i in range(10): #等价于range(0,10)
	print(i)
```

range(开始，结束，步长)

```python
# 打印0到10的偶数
for i in range(0,11,2):
	print(i)
```

## range函数的作用

python内置range()函数的作用是什么？它能返回一系列连续增加的整数，它的工作方式类似于分片，可以生成一个列表对象。range函数大多数时常出现在**for循环中**，在for循环中可做为索引使用。其实它也可以出现在任何需要整数列表的环境中，在python 3.0中range函数是一个迭代器，不能print(range(4))

Python3.x 中 range() 函数返回的结果是一个**整数序列的对象**，而**不是列表**。

```python
print(type(range(10)))
#<class 'range'>
```

当你 help(range) 时会看到：

```python
help(range)
#Return an object that produces a sequence of integers from start (inclusive)
#to stop (exclusive) by step.
```

所以，不是列表，但是可以利用 list 函数返回列表，即：

```python
print(list(range(5)))
# [0, 1, 2, 3, 4]
```

# 字符串

## 字符串操作

- 字符串变量定义

  s = "hello"

  s = 'hello'

- 组成字符串的方式

  - 使用“+”将两个字符串连接成一个新的字符串
  - 组成字符串格式化符号

- 下标hello[01234]

  - 通过下标获取指定位置的字符，string_name[index]

- 切片

  - 切片的语法：string_name[起始:结束:步长]

代码

- 基本操作

```python
s = "hello"
# 通过交表获取指定位置元素
print(s[1])
# 获取字符串长度
print(len(s))
# 循环遍历字符串
for i in range(0,len(s)):
    print(s[i],end="")
print("")
# 脚标 用负数表示倒着数
print(s[-1])# 倒数第一个
print(s[-2])# 倒数第二个
```

**切片**

- 基本操作

```python
# 切片
# 注意：切片切出来的字符串是左闭右开的
line = "zhangsan,20"
name = line[0:8:1]
print(name)
age = line[9::1]# 截取到最后，可以不加结束序号
print(age)
```

- 切片步长

```python
# 切片步长
s = "abcde"
print(s[1:])# bcde
print(s[1:-1])# bcd
print(s[1:-2])# bc
# 隔一个位置取一个元素
print(s[0::2])# 等价于
print(s[::2])# ace
```

## 字符串常用内置方法

- find

  在字符串中查找指定的子字符串是否存在，如果存在则返回第一个子字符串的起始下标，如果不存在则返回-1

- count

  在字符串中统计包含的子字符串的个数

- replace

  使用新的子字符串替换指定的子字符串，返回新的字符串

- split

  按照指定的分隔符字符串，返回分割之后的所有元素的列表

```python
# find
line = "hello world hello python"
print(line.find("hello"))# 第一个子字符串的起始脚标
print(line.find("hello", 6))# 从第六个脚标其开始查找 12
print(line.find("java")) # 不存在，返回-1
# count
print(line.count("world")) # 出现的次数
# replace 字符串是不可变类型
new_line = line.replace("hello", "qqq")
print(new_line)

# split分割
line_list = line.split(" ")
print(line_list)
```

- startswith

  判断字符串是否以指定前缀开头，返回值为True或False

- endswith

  判断字符串是否以指定后缀结束，返回值为True或False

```python
# startswith
files = ["20171201.txt","20180101.log"]
for item in files:
    if item.startswith("2018") and item.endswith("log"):
        print("2018年待处理日志：{}".format(item))
```

- upper

  字符串所有字符大写

- lower

  字符串所有字符小写

```python
# upper lower 大小写
content = input("是否继续，继续输入yes，退出输入no")
if content.lower() == "yes":
    print("欢迎继续使用")
else:
    print("退出，请取卡")
```

# 列表

可理解为柜子，柜子里有不同的抽屉，可存储不同类型的值。

- 可存储相同或者不同类型数据的集合

- 列表定义

  - name_list = ["zhangsan","lisi","wangwu"]

- 顺序存储，可通过下标获取内部元素

  name_list[0]

  name_list[1]

- 内容可变，可通过下角标修改元素值

  name_list[0] = "xiaobai"

- 使用循环遍历列表

- 嵌套列表

代码

基本操作

```python
name_list = ["zhangsan", "lisi", "wangwu"]
print(name_list)# ['zhangsan', 'lisi', 'wangwu']
print(type(name_list))# 类型<class 'list'>
# 脚标获取列表元素
print(name_list[0])
```

存储不同类型的元素，遍历列表

```python
# 存储不同类型的数据
info_list = ["zhangsan", 20, 180.5, 80, True]
print(info_list[4])# True
info_list[3] = 70
print(info_list)# ['zhangsan', 20, 180.5, 70, True]

# 遍历列表，获取列表所有元素
# while循环
i = 0
while i < len(info_list):
    print(info_list[i])
    i += 1
# for循环，通过脚标
for i in range(len(info_list)):
    print(info_list[i])
# for循环，通过序列的每一个元素
for item in info_list:
    print(item)
```

嵌套列表

```python
# 嵌套列表
info_lists = [["zhangsan", 20], ["lisi", 30], ["wangwu", 40]]
print(info_lists[0])# ['zhangsan', 20]
print(info_lists[0][0])# zhangsan

# 循环遍历嵌套列表
for person in info_lists:
    for item in person:
        print(item)
    print("-----------")
```

## 对列表进行增删改查

- append()/insert()添加元素

  - append()向列表末尾添加元素

    ```python
    # append向列表末尾添加元素
    info_lists = [["zhangsan", 20], ["lisi", 30], ["wangwu", 40]]
    info_lists.append(["xiaobai", 25])
    print(info_lists)
    # [['zhangsan', 20], ['lisi', 30], ['wangwu', 40], ['xiaobai', 25]]
    ```

  - insert()可指定位置添加元素

    ```python
    # insert(index, item)
    info_lists = [["zhangsan", 20], ["lisi", 30], ["wangwu", 40]]
    info_lists.insert(1,["wangmazi", 23])
    print(info_lists)
    # [['zhangsan', 20], ['wangmazi', 23], ['lisi', 30], ['wangwu', 40]]
    ```

- “+”组合两个列表生成新的列表

  ```python
  # 两个列表元素组合生成新的列表
  name_list1 = ["唐僧","悟空","八戒"]
  name_list2 = ["沙僧", "白龙马"]
  new_list = name_list1 + name_list2
  print(new_list)# ['唐僧', '悟空', '八戒', '沙僧', '白龙马']
  ```

- extend()向调用它的列表中添加另外一个列表的元素

  ```python
  name_list1 = ["唐僧","悟空","八戒"]
  name_list2 = ["沙僧", "白龙马"]
  name_list1.extend(name_list2)
  print(name_list1)# ['唐僧', '悟空', '八戒', '沙僧', '白龙马']
  ```


- del()/pop()/remove()删除元素

  ```python
  # 删除
  name_list = ["唐僧","悟空","八戒", "沙僧", "白龙马"]
  # 指定脚标
  del name_list[1]
  print(name_list)
  # 指定元素名称
  name_list.remove("八戒")
  print(name_list)
  # 删除最后一个元素(未指定删除位置)
  name_list.pop()
  print(name_list)
  # 删除某个脚标
  name_list.pop(1)
  print(name_list)
  ```

- 切片

  - 和字符串的切片操作相同

    ```python
    name_list = ["唐僧","悟空","八戒", "沙僧", "白龙马"]
    print(name_list[1::2])# ['悟空', '沙僧']
    ```

- in/not in 判断元素在列表中是否存在

  ```python
  name_list = ["唐僧","悟空","八戒", "沙僧", "白龙马"]
  print("悟空" in name_list)# True
  print("如来" in name_list)# False
  print("悟空" not in name_list)# False
  print("如来" not in name_list)# True
  ```

- sort()列表内元素重排序

  - 默认从小到大排列

    ```python
    # sort()
    num_list = [6, 3, 12, 1]
    num_list.sort() # 默认升序
    print(num_list)# [1, 3, 6, 12]
    num_list.sort(reverse=True) # 倒序排列
    print(num_list)# [12, 6, 3, 1]
    ```

- reverse()列表内容倒置

  ```python
  num_list = [6, 3, 12, 1]
  num_list.reverse()
  print(num_list)#[1, 12, 3, 6]
  ```

- count()统计列表内指定元素个数

  ```python
  num_list = [6, 3, 12, 1, 6, 6]
  print(num_list.count(6))# 3
  ```

## 列表去重

使用内置set方法来去重：

```python
lst1 = [2, 1, 3, 4, 1]
lst2 = list(set(lst1))
print(lst2)
# [1, 2, 3, 4]
```

## 切片操作

```python
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a[-1:] == a[-1]
```

这个输出是True or False？

```python
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a[-1:] == a[-1])  # 这个输出是 False
```

今天来和大家一起讨论一下切片的知识点，列表的切片相信大家都用过，切片操作的基本语法比较简单，但是内在逻辑还是比较绕的，下面我会结合例子来总结切片操作的各种情况。

### Python可切片对象的索引方式

包括：正索引和负索引两部分，如下图所示，以list对象`a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`为例：

![slicing-operation](pic/slicing-operation.png)

### Python切片操作的一般方式

一个完整的切片表达式包含两个“:”，用于分隔三个参数(start_index、end_index、step)，当只有一个“:”时，默认第三个参数step=1。

切片操作基本表达式：object[start_index:end_index:step]

参数解释：

* start_index：表示起始索引（包含该索引对应值）；该参数省略时，表示从对象“端点”开始取值，至于是从“起点”还是从“终点”开始，则由step参数的正负决定，step为正从“起点”开始，为负从“终点”开始。
* end_index：表示终止索引（不包含该索引对应值）；该参数省略时，表示一直取到数据“端点”，至于是到“起点”还是到“终点”，同样由step参数的正负决定，step为正时直到“终点”，为负时直到“起点”。
* step：正负数均可，其绝对值大小决定了切取数据时的‘‘步长”，而正负号决定了“切取方向”，正表示“从左往右”取值，负表示“从右往左”取值。当step省略时，默认为1，即从左往右以步长1取值。

### Python切片操作详细例子

以下示例均以list对象a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]为例：

```python
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

1. 切取单个值

```python
a[0]  # 0

a[-4]  # 6
```

2. 切取完整对象

```python
a[:]  # 从左往右[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

a[::]  # 从左往右[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

a[::-1]  # 从右往左[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

3. start_index和end_index全为正（+）索引的情况

```python
a[1:6]  # [1, 2, 3, 4, 5] step=1，从左往右取值，start_index=1到end_index=6同样表示从左往右取值。

a[1:6:-1]  # [] 输出为空列表，说明没取到数据。step=-1，决定了从右往左取值，而start_index=1到end_index=6决定了从左往右取值，两者矛盾，所以为空。

a[6:2]  # [] 同样输出为空列表。step=1，决定了从左往右取值，而start_index=6到end_index=2决定了从右往左取值，两者矛盾，所以为空。

a[:6]  # [0, 1, 2, 3, 4, 5] step=1，表示从左往右取值，而start_index省略时，表示从端点开始，因此这里的端点是“起点”，即从“起点”值0开始一直取到end_index=6（该点不包括）。

a[:6:-1]  # [9, 8, 7] step=-1，从右往左取值，而start_index省略时，表示从端点开始，因此这里的端点是“终点”，即从“终点”值9开始一直取到end_index=6（该点不包括）。

a[6:]  # [6, 7, 8, 9] step=1，从左往右取值，从start_index=6开始，一直取到“终点”值9。

a[6::-1]  # [6, 5, 4, 3, 2, 1, 0] step=-1，从右往左取值，从start_index=6开始，一直取到“起点”0。
```

4. start_index和end_index全为负（-）索引的情况

```python
a[-1:-6]  # [] step=1，从左往右取值，而start_index=-1到end_index=-6决定了从右往左取值，两者矛盾，所以为空。索引-1在-6的右边（如上图）

a[-1:-6:-1]  # [9, 8, 7, 6, 5] step=-1，从右往左取值，start_index=-1到end_index=-6同样是从右往左取值。索引-1在6的右边（如上图）

a[-6:-1]  # [4, 5, 6, 7, 8] step=1，从左往右取值，而start_index=-6到end_index=-1同样是从左往右取值。索引-6在-1的左边（如上图）

a[:-6]  # [0, 1, 2, 3] step=1，从左往右取值，从“起点”开始一直取到end_index=-6（该点不包括）。

a[:-6:-1]  # [9, 8, 7, 6, 5] step=-1，从右往左取值，从“终点”开始一直取到end_index=-6（该点不包括）。

a[-6:]  # [4, 5, 6, 7, 8, 9] step=1，从左往右取值，从start_index=-6开始，一直取到“终点”。[4, 3, 2, 1, 0]step=-1，从右往左取值，从start_index=-6开始，一直取到“起点”。
```

5. start_index和end_index正（+）负（-）混合索引的情况

```python
a[1:-6]  # [1, 2, 3] start_index=1在end_index=-6的左边，因此从左往右取值，而step=1同样决定了从左往右取值，因此结果正确

a[1:-6:-1]   # [] start_index=1在end_index=-6的左边，因此从左往右取值，但step=-则决定了从右往左取值，两者矛盾，因此为空。

a[-1:6][]  # start_index=-1在end_index=6的右边，因此从右往左取值，但step=1则决定了从左往右取值，两者矛盾，因此为空。

a[-1:6:-1]  # [9, 8, 7] start_index=-1在end_index=6的右边，因此从右往左取值，而step=-1同样决定了从右往左取值，因此结果正确。
```

6. 多层切片操作

```python
a[:8][2:5][-1:]  # [4] 
# 相当于：
# a[:8] = [0, 1, 2, 3, 4, 5, 6, 7]
# a[:8][2:5] = [2, 3, 4]
# a[:8][2:5][-1:] = [4]
# 理论上可无限次多层切片操作，只要上一次返回的是非空可切片对象即可。
```

7. 切片操作的三个参数可以用表达式

```python
a[2+1:3*2:7%3]  # [3, 4, 5] 即：a[2+1:3*2:7%3] = a[3:6:1]
```

8. 其他对象的切片操作

前面的切片操作说明都以list为例进行说明，但实际上可进行的切片操作的数据类型还有很多，包括元组、字符串等等。

```python
(0, 1, 2, 3, 4, 5)[:3]  # (0, 1, 2) 元组的切片操作

'ABCDEFG'[::2]  # 'ACEG' 字符串的切片操作

for i in range(1,100)[2::3][-10:]: 
    print(i)
# 就是利用range函数生成1-99的整数，然后取3的倍数，再取#最后十个。
```

### 常用切片操作

以列表`a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`为说明对象

1. 取偶数位置

```python
b = a[::2]  # [0, 2, 4, 6, 8]
```

2. 取奇数位置

```python
b = a[1::2]  # [1, 3, 5, 7, 9]
```

3. 拷贝整个对象

```python
b = a[:]
print(b) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(id(a)) # 41946376
print(id(b)) # 41921864
# 或
b = a.copy()
print(b)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(id(a)) # 39783752
print(id(b)) # 39759176
# 需要注意的是[:]和.copy()都属于“浅拷贝”，只拷贝最外层元素，内层嵌套元素则通过引用，而不是独立分配内存。

a = [1, 2, ['A','B']]
# 原始a
print('a = {}'.format(a))  # a = [1, 2, ['A', 'B']]
b = a[:]
b[0] = 9  # 修改b的最外层元素，将1变成9
b[2][0] = 'D'  # 修改b的内嵌层元素
# b修改内部元素A为D后，a中的A也变成了D，说明共享内部嵌套元素，但外部元素1没变。
print('a = {}'.format(a))  # a = [1, 2, ['D', 'B']]
# 修改后的b
print('b = {}'.format(b))
print('id(a) = {}'.format(id(a)))  # id(a) = 38669128
print('id(b) = {}'.format(id(b)))  # id(b) = 38669192
```

4. 修改单个元素

```python
a[3] = ['A','B']  # [0, 1, 2, ['A', 'B'], 4, 5, 6, 7, 8, 9]
```

5. 在某个位置插入元素

```python
a[3:3] = ['A','B','C']  # [0, 1, 2, 'A', 'B', 'C', 3, 4, 5, 6, 7, 8, 9]
a[0:0] = ['A','B']  # ['A', 'B', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

6. 替换一部分元素

```python
a[3:6] = ['A','B']  # [0, 1, 2, 'A', 'B', 6, 7, 8, 9]
```

### 总结

（一）start_index、end_index、step三者可同为正、同为负，或正负混合。但必须遵循一个原则，否则无法正确切取到数据，即：

当start_index的位置在end_index的左边时，表示从左往右取值，此时step必须是正数（同样表示从左往右）；

当start_index的位置在end_index的右边时，表示从右往左取值，此时step必须是负数（同样表示从右往左），即两者的取值顺序必须是相同的。

对于特殊情况，当start_index或end_index省略时，起始索引和终止索引由step的正负来决定，这种情况不会有取值方向矛盾（即不会返回空列表[]），但正和负取到的结果顺序是相反的，因为一个向左一个向右。

（二）在利用切片时，step的正负是必须要考虑的，尤其是当step省略时。比如a[-1:]，很容易就误认为是从“终点”开始一直取到“起点”，即a[-1:]= [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]，但实际上a[-1:]=[9]（注意不是9），原因在于step=1表示从左往右取值，而起始索引start_index=-1本身就是对象的最右边元素了，再往右已经没数据了。

（三）需要注意：“取单个元素（不带“:”）”时，返回的是对象的某个元素，其类型由元素本身的类型决定，而与母对象无关，如上面的a[0]=0、a[-4]=6，元素0和6都是“数值型”，而母对象a却是“list”型；“取连续切片（带“:”）”时，返回结果的类型与母对象相同，哪怕切取的连续切片只包含一个元素，如上面的a[-1:]=[9]，返回的是一个只包含元素“9”的list，而非数值型“9”。







# 元组

- 循序存储相同/不同类型的元素

- 元组定义，使用()将元素括起来，元素之间用","隔开

- 特性：**不可变，不支持增删改查**

- 查询：通过下标查询元组指定位置的元素

- 空元组的定义：none_tuple = ()

- 只包含一个元素的元组：one_tuple = ("one",)

  **定义一个元素的元组，括号里一定要有一个逗号(,)**

- 循环遍历元组



列表用于存储可变的元素，一般存储相同类型的元素。

元组不可变，通常存储一些不希望被改变的信息，如用户名，密码等。

```python
#定义元组，存储数据库信息
db_info = ("192.169.1.1", "root", "root123")
# 通过脚标查询元组中的元素
ip = db_info[0]
port = db_info[1]
print("ip:{},port:{}".format(ip, port))
# ip:192.169.1.1,port:root
print(type(ip))#<class 'str'>

# 通过脚标来修改元组指定元素的值，这是不行的
db_info[1] = 8080
print(db_info)
# TypeError: 'tuple' object does not support item assignment

# del删除元组指定位置的元素，这是不行的
del db_info[1]
# TypeError: 'tuple' object doesn't support item deletion

# 定义一个元组
one_tuple = (123,)
print(one_tuple)# (123,)
print(type(one_tuple))# <class 'tuple'>
# 错误的定义只包含一个元素的元组，少了元素后的逗号。
one_tuple1 = (123)
print(one_tuple1)# 123
print(type(one_tuple1))# <class 'int'>

# 定义空元组
none_tuple = ()
print(none_tuple)
print(type(none_tuple))
```

循环遍历元组

```python
# 循环遍历
db_info = ("192.169.1.1", "root", "root123")
# for循环
for item in db_info:
    print(item)

# while循环
i = 0
while i < len(db_info):
    print(db_info[i])
    i += 1
```

# 字典

- 存储Key-Value键值对类型的数据
- 字典定义：{Key1 : value1,  key2 : value2,  ...}
- 查询：根据Key查找Value
- 字典具有添加、修改、删除操作
- 内置方法get、keys、values、items、clear
- 循环遍历字典



## 基本操作

```python
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经"}
print("{}的年龄：{}，性别：{}，工作内容：{}".format(user_if_dict["name"],user_if_dict["age"],user_if_dict["gender"],user_if_dict["job"]))

# 通过Key修改已经存在的值
user_if_dict["job"] = "取经|偷桃"
print(user_if_dict["job"])
```

使用字典的原因：

存储大量数据也能够准确查找和修改。

不支持通过下标来查询指定位置的元素的值。

Key不能重复出现，否则后面的值会覆盖前面的值

```python
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经","name":"白骨精"}
print("{}的年龄：{}，性别：{}，工作内容：{}".format(user_if_dict["name"],user_if_dict["age"],user_if_dict["gender"],user_if_dict["job"]))
```

## 字典的增删改查操作

```python
# 字典的增删改查操作

# 添加一个键值对
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经"}
user_if_dict["tel"] = 13812345678
print(user_if_dict)# 5对
print(len((user_if_dict)))

# 修改字典中的指定的值
user_if_dict["tel"] = 13811118888
print(user_if_dict)
# 删除元素
del user_if_dict["tel"]
print(user_if_dict)

# 查询指定名字的元素
print(user_if_dict["name"])
# 查询不存在的键，会报错
# 解决办法
# 方法一：in or not in
if "tel" in user_if_dict:
    print(user_if_dict["tel"])
else:
    print("\"tel\"不存在")
# 方法二： 字典内置的get方法
# 如果不存在，就会返回一个设定的默认值，用于缺省只补全
print(user_if_dict.get("tel","19911116666"))# None
```

## 使用循环来遍历字典

```python
# 使用循环来遍历字典
# 字典内置的Keys方法，返回所有的Key组成一个序列
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经","name":"白骨精"}
for key in user_if_dict.keys():
    print("{}:{}".format(key, user_if_dict[key]),end="|")
# 字典内置的Values方法，返回所有的Value组成的一个序列
for value in user_if_dict.values():
    print(value)# 只能遍历出字典所有的值
# 返回字典的键值对，组成元组返回
for item in user_if_dict.items():
    print(type(item))# <class 'tuple'>
    print(item)# ('name', '白骨精')
    print(item[0])# Key: name
    print(item[1])# Value: 白骨精
# 用两个变量分别接受字典的Key和Value
for key,value in user_if_dict.items():
    print("{}:{}".format(key,value))
```

## 清空字典

```python
user_if_dict = {"name":"悟空","age":100,"gender":"male","job":"取经","name":"白骨精"}
print(user_if_dict)
user_if_dict.clear()
print(user_if_dict)# {}
```

# 集合

- 无序存储不同数据类型不重复元素的序列

  即使填入多个相同的元素，也会被去重

- 集合定义：name_set={"xiaoming",  "xiaoqiang", "xiaobai"}

- 使set对序列中元素去重，同时创建集合

  例如：name_set = set(["xiaoming", "zhangsan"])

- 创建空集合：none_set = set()

- 使用in和not in判断一个元素在集合中是否存在

- 使用add(元素)方法添加一个元素到集合中

- 使用update(序列)方法将一个序列中的元素添加到集合中，同时对元素去重

- remove(元素)根据元素值删除集合中指定元素，如果元素不存在，则报错

- discard(元素)根据元素值删除集合中指定元素，如果元素不存在，不会引发错误

- pop()随机删除集合中的某个元素，并且返回被删除的元素

- clear()清空集合

- 集合操作

  - 交集intersection(&)
  - 并集union(|)
  - 差集difference(-)
  - 对称差集(^)

基本用法

```python
# 集合的定义，元素去重
student_set = {"zhangsan","lisi","wangwu","zhangsan"}
print(student_set)# 无序，去重
print(len(student_set))
print(type(student_set))
# set(序列)
# set(集合) 对list中的元素去重，并创建一个新的集合
id_list = ["id1", "id2", "id3", "id1", "id2"]
new_set = set(id_list)
print(id_list)
print(new_set)
# set(元组) 对突破了中的元素去重，并创建一个新的集合
id_tuple = ("id1", "id2", "id3", "id1", "id2")
new_set = set(id_tuple)
print(id_tuple)
print(new_set)
```

对于字符串，会打乱元素顺序，并对字符去重。

```python
string_set = set("hello")
print(string_set)
# {'o', 'e', 'l', 'h'}
```

创建空集合

```python
# 创建空集合
none_set = set()
print(none_set)# set()
# 注意，床架空字典是{}
none_dict = {}
print(none_dict)# {}
```

判断存在与否

```python
# in or not in
id_list = ["id1", "id2", "id3", "id1", "id2"]
new_set = set(id_list)
user_id = "id1"
if user_id in new_set:
    print("{}存在".format(user_id))
elif user_id not in new_set:
    print("{}不存在".format(user_id))
```

update添加序列，而add()只能添加一个元素

```python
# update(序列) 重复元素会去重
name_set = {"zhangsan", "lisi"}
# 添加列表元素到集合
name_set.update(["悟空", "八戒"])
print(name_set)
# 添加元组元素到集合
name_set.update(("悟空", "八戒"))
print(name_set)
# 添加多个序列元素到集合
name_set.update(["悟空", "八戒"],["沙僧", "八戒"])
print(name_set)
# 把一个集合并入另一个集合
name_set.update({"张飞","李逵"})
print(name_set)
# add()只能添加一个元素
name_set.add("如来佛")
print(name_set)
```

三种删除操作

```python
# 三种删除操作

#remove(元素)
name_set = {"zhangsan", "lisi", "wangwu"}
name_set.remove("zhangsan")
print(name_set)
# remove删除不存在的元素会报错
# name_set.remove("zhangsan")# KeyError: 'zhangsan'

# discard(元素)
name_set = {"zhangsan", "lisi", "wangwu"}
name_set.discard("zhangsan")
print(name_set)
name_set.discard("zhangsan")

# pop随机删除
name_set = {"zhangsan", "lisi", "wangwu"}
name_set.pop()
print(name_set)
```

交集，并集，差集

```python
# 交集，并集，
# 交集
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
inter_set1 = num_set1 & num_set2
inter_set2 = num_set1.intersection(num_set2)
print(inter_set1)
print(inter_set2)

# 并集
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
union_set1 = num_set1 | num_set2
union_set2 = num_set1.union(num_set2)
print(union_set1)
print(union_set2)

# 差集
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
diff_set1 = num_set1 - num_set2
diff_set2 = num_set1.difference(num_set2)
print(diff_set1)
print(diff_set2)

# 对称差集, 互差再并
num_set1 = {1,2,4,7}
num_set2 = {2,5,8,9}
sym_diff_set1 = num_set1 ^ num_set2
sym_diff_set2 = num_set1.symmetric_difference(num_set2)
print(sym_diff_set1)
print(sym_diff_set2)
```



# 参考资料

* [彻底搞懂Python切片操作](https://baijiahao.baidu.com/s?id=1659565555319388195)

“切片操作”复制自此博客。

