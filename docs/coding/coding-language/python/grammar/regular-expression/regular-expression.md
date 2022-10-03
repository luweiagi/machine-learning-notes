# 正则表达式

* [返回上层目录](python.md#python)
* [re模块的使用](#re模块的使用)
* [字符匹配、数量表示、边界表示](#字符匹配、数量表示、边界表示)
  * [字符匹配](#字符匹配)
  * [数量表示](#数量表示)
  * [边界表示](#边界表示)
* [正则表达式的高级用法](#正则表达式的高级用法)
* [贪婪与非贪婪模式](#贪婪与非贪婪模式)

# re模块的使用

- re正则表达式模块
  - match（正则表达式，待匹配字符串），
    - 用于正则匹配检查，如果待匹配字符串能够匹配正则表达式，则match方法返回匹配对象，否则返回None
    - 采用从左往右逐项匹配
  - group()方法
    - 用来返回字符串的匹配部分

```python
import re
rs = re.match("chinahadoop", "chinahadoop.cn")
print(rs)
# <_sre.SRE_Match object; span=(0, 11), match='chinahadoop'>
print(type(rs))
#<class '_sre.SRE_Match'>
if rs != None:
    print(rs.group())
    # chinahadoop
```

# 字符匹配、数量表示、边界表示

## 字符匹配

- . ，匹配除“\n”之外的任意单个字符
- \d，匹配0到9之间的一个数字，等价于[0-9]
- \D，匹配一个非数字字符，等价于[~0-9]
- \s，匹配任意空白字符
- \w，匹配任意单词字符（包含下划线），如a-z，A-z，0-9，_
- \W，匹配任意非单词字符，等价于^[a-z，A-z，0-9，_]
* []，匹配[]中列举的字符
- ^，取反

"."匹配除了“\n”之外的任意单个字符

```python
import re
rs = re.match(".", "1")
print(rs.group())# 1
rs = re.match(".", "a")
print(rs.group())# a
rs = re.match(".", "abc")
print(rs.group())# a
rs = re.match("...", "abcd")
print(rs.group())# abc
rs = re.match(".", "\n")
print(type(rs))# <class 'NoneType'>
```

"\s"匹配空格字符

"\S"匹配非空格字符

```python
import re
rs = re.match("\s","\t")
print(rs.group())# tab
rs = re.match("\S","abc")
print(rs.group())# a
```

"\w"匹配单词字符

```python
import re
rs = re.match("\w","Ab")
print(rs.group())# A
rs = re.match("\w","12")
print(rs.group())# 1
rs = re.match("\w","_")
print(rs.group())# _
```

"[]"匹配日中括号列举的字符

```python
rs = re.match("[Hh]","Hello")
if rs != None:
    print(rs.group())# H
rs = re.match("[0123456789]","3n")
if rs != None:
    print(rs.group())# 3
rs = re.match("[0-9]","3n")
if rs != None:
    print(rs.group())# 3
```

## 数量表示

- *，一个字符可以出现任意次，也可以一次都不出现
- +，一个字符至少出现一次
- ？，一个字符至多出现一次
- {m}，一个字符出现m次
- {m,}，一个字符至少出现m次
- {m, n}，一个字符出现m到n次

“*“表示出现任意次

```python
import re
# 以1开头
rs = re.match("1\d*","123456789")
if rs != None:
    print(rs.group())# 123456789
rs = re.match("1\d*","23456789")
if rs != None:
    print(rs.group())# None
rs = re.match("1\d*","12345abcde")
if rs != None:
    print(rs.group())# 12345
```

"+"一个字符至少出现1次

```python
# 至少包含一个字符的数字，那就能匹配上
import re
rs = re.match("\d+","123abc")
print(rs)# 123
```

"?"一个字符至多出现1次

```python
# 最多匹配一个
import re
rs = re.match("\d?","abc")
print(rs)# ''空
rs = re.match("\d?","123abc")
print(rs)# 1
```

自定义匹配字符出现的次数

```python
import re
rs = re.match("\d{3}","12345abc")# 出现3次
print(rs)# 123
rs = re.match("\d{5}","123abc")# 出现5次
print(rs)# None
rs = re.match("\d{3,}","12345abc") #至少出现3次
print(rs)# 12345
rs = re.match("\d{3,}","12abc") #至少出现3次
print(rs)# None
rs = re.match("\d{0,}","abc1234") #至少出现0次 = *
print(rs)# ""
rs = re.match("\d{1,}","123abc1234") #至少出现1次 = +
print(rs)# 123
```

注意：从字符串的第一个字符进行匹配，这是match函数的特点

练习：匹配一个手机号

分析过程：

手机号11位，第一位以1开头，第二位3/5/7/8，第三位到第十一位是0-9数字

```python
import re
rs = re.match("1[3578]\d{9}","13612345678abc")
print(rs) # 13612345678
```

后面我们学到“边界”，才能识别11位。

## 边界表示

- 字符串与单词边界

  - ^，用于匹配一个字符串的开头

  - $，用于匹配一个字符串的结尾

    只是用于匹配是否结束，而不是匹配结束的那个字符。

- 匹配分组

  - |，表示或，匹配|连接的任何一个表达式
  - ()，将括号中字符作为一个分组
  - \\NUM，配合分组()使用，引用分组NUM（NUM表示分组的编号）对应的匹配规则
  - (?P<name>)，给分组起别名
  - (?P=name)，应用指定别名的分组匹配到的字符串

转义字符

```python
# 转义字符
str1 = "hello\\world"
print(str1)# hello\world
str2 = "hello\\\\world"
print(str2)# hello\\world
# 原生字符串 raw(原始)的简写
str3 = r"hello\\world"
print(str3)# hello\\world
```

正则表达式里的字符

```python
import re
str1 = r"hello\\world"
rs = re.match("\w{5}\\\\\w{5}",str1)
print(rs)# None 居然是空，为什么？
# 若需匹配\，则需正则表达式里\\\\个斜杠匹配
rs = re.match("\w{5}\\\\\\\\\w{5}",str1)
print(rs)# hello\\\\world
# 为什么？
# \w{5}\\\\\\\\\w{5}   \\\\\\\\(正则表达式引擎)->\\\\正则对象
rs = re.match(r"\w{5}\\\\\w{5}",str1)# 原生
print(rs) # hello\\\\world
```

字符串的边界

使用结束边界"$"完善手机号匹配

```python
import re
# $ 表示到这里就结束了
rs = re.match("1[3578]\d{9}$","13612345678abc")
print(rs) # None
rs = re.match("1[3578]\d{9}$","13612345678")
print(rs) # 13612345678
```

匹配邮箱

```python
import re
rs = re.match("\w{3,10}@163.com$","luwei123@163.com")
print(rs) # luwei123@163.com
rs = re.match("\w{3,10}@163.com$","lw@163.com")
print(rs) # None
rs = re.match("\w{3,10}@163.com$","luwei123@163.comHaHa")
print(rs) # None
rs = re.match("\w{3,10}@163.com$","luwei123@163Xcom")# .这里是任意字符
print(rs) # luwei123@163Xcom 这里就有问题了,这里的.不是任意字符
rs = re.match("\w{3,10}@163\.com$","luwei123@163Xcom")# 用转义字符
print(rs)# None
```

匹配分组

匹配0-100的数字

```python
import re
rs = re.match("[1-9]?\d?$|100$","0")
print(rs) # None
```

正则表达式中使用括号()来分组

```python
import re
rs = re.match("\w{3,10}@(163|qq|gmail)\.com","hello@gmail.com")
print(rs) # hello@gmail.com
```

\NUM 使用第NUM个分组

```python
import re
html_str = "<head><title>python</title></head>"
rs = re.match(r"<.+><.+>.+</.+></.+>", html_str)
print(rs)# <head><title>python</title1></head>
html_str = "<head><title>python</title111></head222>"
rs = re.match(r"<.+><.+>.+</.+></.+>", html_str)
print(rs)# <head><title>python</title111></head222>
html_str = "<head><title>python</title111></head222>"
rs = re.match(r"<(.+)><(.+)>.+</\2></\1>", html_str)
print(rs)# None
html_str = "<head><title>python</title></head>"
rs = re.match(r"<(.+)><(.+)>.+</\2></\1>", html_str)
print(rs)# <head><title>python</title></head>
# 给每个分组起一个别名，这样方便使用
html_str = "<head><title>python</title></head>"
rs = re.match(r"<(?P<g1>.+)><(?P<g2>.+)>.+</(?P=g2)></(?P=g1)>", html_str)
print(rs)# <head><title>python</title></head>1
```

# 正则表达式的高级用法

- search
  - 从左到右在字符串的任意位置搜索第一次出现匹配给定正则表达式的字符
- findall
  - 在字符串中查找所有匹配成功的组，返回匹配成功的结果列表
- finditer
  - 在字符串中查找所欲正则表达式匹配成功的字符串，返回iterator迭代器
- sub
  - 将匹配到的数据使用新的数据替换
- split
  - 根据指定的分隔符切割字符串，返回切割之后的列表

```python
import re
rs = re.search("hello","haha,hello,python,hello,world")
# 有两个hello，但只找第一个，不会继续往后找
print(rs)# hello

rs = re.findall("hello","haha,hello,python,hello,world")
#  找到所有的hello匹配结果
print(rs)# ['hello', 'hello']

rs = re.finditer("\w{3,20}@(163|qq)\.(com|cn)","hello@163.com---luwei@qq.com")
# 返回和match同类型的迭代器
print(type(rs))
for it in rs:
    print(it.group())
    # hello@163.com
    # luwei@qq.com

str = "java python c cpp java"
rs = re.sub(r"java","python", str)
#  将匹配到的数据使用新的数据替换
print(rs)# python python c cpp python

# 用指定符号进行字符串切割
line1 = "word;Word,emp?hahaha"
print(re.split(r";|,|\?", line1)) #别忘了转义"?"
# ['word', 'Word', 'emp', 'hahaha']
print(re.split(r"[;,?]", line1))
# ['word', 'Word', 'emp', 'hahaha']
```

# 贪婪与非贪婪模式

- 贪婪模式
  - 正则表达式引擎默认是贪婪模式，尽可能多的匹配字符
- 非贪婪模式
  - 与贪婪模式相反，尽可能少的匹配字符
  - 在表示数量的”*“，”?“，”+“，“{m,n}”符号后面加上?，使贪婪变成非贪婪。

```python
import re
rs = re.findall("hello\d*","hello12345")
print(rs)# hello12345
rs = re.findall("hello\d*?","hello12345")
print(rs)# hello
```

