# 函数

* [返回上层目录](../python.md)
* [函数定义和类型](#函数定义和类型)
* [局部变量、全局变量](#局部变量、全局变量)
* [函数缺省参数、不定长参数、命名参数](#函数缺省参数、不定长参数、命名参数)
* [递归函数及应用](#递归函数及应用)
* [匿名函数及应用](#匿名函数及应用)

# 函数定义和类型

- 对某一功能的封装

- 函数定义

  ```python
  def 函数名称（参数）:
  	函数体代码
  	return 返回值
  ```

  别忘了定义函数名后面加冒号“:”

- 函数调用：函数名(参数)

- 函数参数

  形参：定义函数时设置的参数

  实参：调用函数时传入的参数



无参函数

```python
def print_user_info():
    print("name:zhangsan")
    print("age:20")

print_user_info()
```

有参函数

```python
def print_user_info2(name, age):
    print("name:{}".format(name))
    print("age:{}".format(age))

name = "zhangsan"
age = 20
print_user_info2(name, age)
```

带有返回值的函数

```python
# 返回单个值
def x_y_sum(x,y):
    res = x + y
    return res
# 返回多个值
def x_y_comp(x,y):
    rs1 = x + y
    rs2 = x - y
    rs3 = x * y
    rs4 = x / y
    # rs = (rs1, rs2, rs3, rs4)
    # return  rs
    return rs1, rs2, rs3, rs4

z = x_y_sum(10, 40)
print(z)
z = x_y_comp(4, 2)
print(z)# (6, 2, 8, 2.0)
print(type(z))# <class 'tuple'>
```

# 局部变量、全局变量

- 局部变量
  - 函数内部定义的变量
  - 不同函数内的局部变量可以定义相同的名字，互不影响
  - 作用范围：函数体内有效，其他函数不能直接使用
- 全局变量
  - 函数外部定义的变量
  - 作用范围：可以在不同函数中使用
  - 在函数内使用global关键字实现修改全局变量的值
  - 全局变量命名建议以g_开头，如：g_name

基本操作

```python
# 全局变量
g_name = "zhangsan"
def get_name1():
    print(g_name)

def get_name2():
    print(g_name)

get_name1()
print("--------")
get_name2()
```

想通过在函数内直接修改全局变量的方式是错误的，相当于在函数体内定义了一个与全局变量同名的局部变量。

全局变量不能在函数体内被直接通过赋值而修改。函数体内被修改的那个”全局变量“其实只是函数体内定义的一个局部变量，只是名称相同而已。所以，通过在函数体内直接对全局变量赋值是无法改变其值的。

```python
g_age = 25
def change_age():
    g_age = 35
    print("函数内：",g_age)

change_age()# 函数内： 35
print("--------")
print(g_age)# 25
```

应该在函数体内用global声明全局变量，才能修改：

```python
g_age = 25
def change_age():
    global g_age# 必须使用global关键字声明
    print("修改之前：",g_age)
    g_age = 35
    print("修改之后：",g_age)

change_age()# 修改之前： 25 修改之后： 35
print("--------")
print(g_age)# 35
```

全局变量定义的位置应当放在调用它的函数之前，不然会出错。

原因：python解释器从上到下逐行执行，那当执行此函数时，函数之后的变量是不存在的。

```python
g_num1 = 100
def print_global_num():
    print("g_num1:{}".format(g_num1))
    print("g_num2:{}".format(g_num2))
    print("g_num3:{}".format(g_num3))

g_num2 = 200
print_global_num()
g_num3 = 300# 在调用函数之后，没有被定义
```

正确的全局变量定义方法：

在函数调用之前就把全局变量定义好

```python
g_num1 = 100
g_num2 = 200
g_num3 = 300

def print_global_num1():
    print("g_num1:{}".format(g_num1))
def print_global_num2():
    print("g_num2:{}".format(g_num2))
def print_global_num3():
    print("g_num3:{}".format(g_num3))

print_global_num1()
print_global_num2()
print_global_num3()
```

全局变量的类型为字典、列表时，在函数体内修改你值时，可不使用global关键字。

```python
g_num_list = [1,2,3]
g_info_dict = {"name":"zhangsan", "age":20}
def update_info():
    g_num_list.append(4)# 并没有使用global关键字
    g_info_dict["gender"] = "male"
def get_info():
    for num in g_num_list:
        print(num,end= " ")
    for key,value in g_info_dict.items():
        print("{}:{}".format(key, value))

update_info()
get_info()
# 1 2 3 4
# name:zhangsan
# age:20
# gender:male
```

# 函数缺省参数、不定长参数、命名参数

- 缺省(默认)参数
  - 函数定义带有初始值的形参
  - 函数调用时，缺省参数可传，也可不传
  - 缺省参数一定要位于参数列表的最后
  - 缺省参数数量没有限制
- 命名参数
  - 调用带有参数的函数时，通过指定参数名称传入参数的值
  - 可以不按函数定义的参数顺序传入
- 不定长参数
  - 函数可以接受不定个数的参数传入
  - def function([formal_args,]*args)函数调用时，传入的不定参数会被封装成元组
  - def function([formal_args,]**args)函数调用时，如果传入key=value形式的不定长参数，会被封装成字典
- 拆包
  - 对于定义了不定长参数的函数，在函数调用时需要把已定义好的元组或者列表传入到函数中，需要使用拆包方法

缺省参数

```python
def x_y_sum(x, y=20):
    rs = x + y
    print("{}+{}={}".format(x,y,rs))

x_y_sum(10, 30)# 10+30=40
x_y_sum(10)# 10+20=30
```

命名参数

```python
def x_y_sum(x=10, y=20):
    rs = x + y
    print("{}+{}={}".format(x, y, rs))

num1 = 15
num2 = 12
x_y_sum(y=num1, x=num2)
# 12+15=27
```

不定长参数

1、元组：*args

```python
# 计算任意数字的和
def any_num_sum(x, y=10, *args):
    print("args:{}".format(args))
    rs = x + y
    if len(args) > 0:
        for arg in args:
            rs += arg
    print("计算结果：{}".format(rs))

# any_num_sum(20)
# any_num_sum(20,30)
any_num_sum(20,30,40,50)
# args:(40, 50) 元组
# 计算结果：140
```

2、字典：**args

接受key:value对，然后封装到字典里

```python
# 缴五险一金
def social_comp(basic_money, **proportion):
    print("缴费基数：{}".format(basic_money))
    print("缴费比例：{}".format(proportion))

social_comp(8000, e=0.1, a=0.12)
# 缴费基数：8000
# 缴费比例：{'e': 0.1, 'a': 0.12}
```

不定长参数综合使用+拆包

```python
# 工资计算器
def salary_comp(basic_money, *other_money, **proportion):
    print("基本工资：{}".format(basic_money))
    print("其他福利：{}".format(other_money))
    print("计费比例：{}".format(proportion))
other_money = (500,200,100,1000)
proportion_dict = {"e":0.2, "m":0.1, "a":0.12}

# 注意要用*和**来拆包，不然就会把最后两个都当作元组进行封装了
salary_comp(8000, *other_money, **proportion_dict)
# 基本工资：8000
# 其他福利：(500, 200, 100, 1000)
# 计费比例：{'e': 0.2, 'm': 0.1, 'a': 0.12}
```

# 递归函数及应用

- 函数调用自身
- 注意：递归过程中要有用于结束递归的判断

递归函数

```python
# 阶乘
'''
1! = 1
1! = 2*1!
3! = 3*2!
'''
# for循环计算阶乘
def recursive_for(num):
    rs = num
    for i in range(1,num):
        rs *= i
    return rs
# 递归计算阶乘
def recursive(num):
    if num > 1:
        return num * recursive(num-1)
    else:
        return num

num = recursive_for(4)
print(num)
num = recursive(4)
print(num)
```

# 匿名函数及应用

匿名函数：定义的函数没有名称

- 用lambda关键字创建匿名函数
- 定义：lambda[参数列表]:表达式
- 匿名函数可以作为参数被传入其他函数

匿名函数

```python
# 匿名函数
sum = lambda  x,y: x+y
print(sum(10,20))# 30
print(type(sum))# <class 'function'>
```

应用场景：

- 作为函数的参数

  ```python
  def x_y_comp(x,y,func):
      rs = func(x,y)
      print("计算结果：{}".format(rs))

  x_y_comp(3,5,lambda x,y:x+y)# 计算结果：8
  x_y_comp(3,5,lambda x,y:x*y)# 计算结果：15
  ```

- 作为内置函数的参数

  ```python
  user_info = [{"name":"zhangsan","age":20},{"name":"lisi","age":15},{"name":"wangwu","age":30},]
  print(user_info)
  # 按照年龄的降序排列，默认是升序
  user_info.sort(key=lambda info:info["age"], reverse=True)
  print(user_info)
  ```

# python内置函数

## enumerate()

enumerate() 函数用于将一个**可遍历的数据对象**(如**列表**、**元组**或**字符串**)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

以下是 enumerate() 方法的语法:

```python
enumerate(sequence, [start=0])
```

参数：

- sequence -- 一个序列、迭代器或其他支持迭代对象。
- start -- 下标起始位置。

返回值：

返回 enumerate(枚举) 对象。

实例：

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
list(enumerate(seasons, start=1))       # 下标从 1 开始
# [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

for循环使用 enumerate：

```python
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
	print i, element

#0 one
#1 two
#2 three
```

