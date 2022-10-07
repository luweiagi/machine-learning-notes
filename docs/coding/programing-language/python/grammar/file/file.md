# 文件操作

* [返回上层目录](../python.md)
* [日期和时间](#日期和时间)
* [文件操作](#文件操作)
* [文件夹操作](#文件夹操作)
* [JSON格式文件操作](#JSON格式文件操作)
* [CSV格式文件操作](#CSV格式文件操作)

# 日期和时间

- time模块

  - time() 函数获取当前时间戳
  - time.sleep(seconds) 睡眠程序等待几秒钟

- datetime模块

  - datetime.datetime.now() 获取当前日期和时间

  - strftime(format) 日期时间格式化

  - datetime.datetime.fromtimestamp(timestamp)将时间戳转换为日期时间

  - datetime.timedelta(时间间隔)返回一个时间间隔对象，通过时间间隔可以对时间进行加减法得到新的时间

    | 格式化符号 | 说明                  |
    | ----- | ------------------- |
    | %y    | 两位数的年份表示(00~99)     |
    | %Y    | 思维说的年份表示(0000~9999) |
    | %m    | 月份(01~12)           |
    | %d    | 月内中的一天(0~31)        |
    | %H    | 24小时制小时数(0~23)      |
    | %I    | 12小时制小时数(01~12)     |
    | %M    | 分钟数(00=59)          |
    | %S    | 秒(00~59)            |

time模块

```python
import time
# 获取当前时间戳
print(time.time())# 从1970年到现在经历的秒数

# 程序等待
start_time = time.time()
print("----------")
time.sleep(2)# 程序等待
print("----------")
end_time = time.time()
print(end_time - start_time)
```

datetime模块

```python
import datetime
print(datetime.datetime.now())
# 2018-05-17 20:11:59.104035
print(type(datetime.datetime.now()))
# <class 'datetime.datetime'>

# 日期格式化 string_formate
print(datetime.datetime.now().strftime("%Y/%m/%d %H/%M/%S"))
# 2018/05/17 20/11/59
```

计算时间差值

```python
# 计算时间差值
import datetime, time
start_time = datetime.datetime.now()
time.sleep(5)
end_time = datetime.datetime.now()
print((end_time - start_time).seconds)
```

时间戳转换为日期

```python
# 时间戳转换为日期
import datetime, time
ts = time.time()
print(ts)
print(datetime.datetime.fromtimestamp(ts))
print(datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d"))
```

根据时间间隔，获取指定日期

```python
# 根据时间间隔，获取指定日期
import datetime, time
today = datetime.datetime.today()
print(today.strftime("%Y-%m-%d %H-%M-%S"))
# 2018-05-17 20-22-56
delta_time = datetime.timedelta(days=1)
print(delta_time)
# 1 day, 0:00:00
yestoday = today - delta_time
print(yestoday.strftime("%Y-%m-%d %H-%M-%S"))
# 2018-05-16 20-22-56
```

# 文件操作

- open(文件路径，访问模式，encoding=编码格式)方法打开一个已存在的文件，或者创建新的文件

- close()方法关闭已打开的文件

- 打开文件常用的三种访问模式

  - r:只读模式（默认）
  - w:只写模式
  - a:追加模式

- write(data)方法向文件中写入字符串

  ```python
  # 如果不存在则创建一个文件
  #相对路径
  f = open("test.txt", "w", encoding="utf-8")
  #绝对路径
  # f = open("d://test.txt", "w", encoding="utf-8")
  # 追加模式 a：append
  # f = open("test.txt", "a", encoding="utf-8")
  f.write("你好")
  f.close()
  ```

- read()方法读取文件全部内容

  ```python
  # 读数据
  f = open("test.txt", "r", encoding="utf-8")
  data = f.read()# 一次性全读出来
  print(data)
  ```

- readlines()方法读取文件全部内容，放回一个列表，每行数据是列表中的一个元素。

  一次性全部读取，非常低效

  ```python
  # readlines() 一次性全部读取，非常低效
  f = open("test.txt", "r", encoding="utf-8")
  data = f.readlines()
  print(data)# ['你好\n', '123\n']
  print(type(data))# <class 'list'>
  for line in data:
      print("--->{}".format(line),end="")
  f.close()
  ```

- readline()方法按行读取文件数据

  ```python
  #readline
  f = open("test.txt", "r", encoding="utf-8")
  data = f.readline()
  print(data)# ['你好\n', '123\n']
  print(type(data))# <class 'list'>
  f.close()
  ```

- writelines(字符串序列)将一个字符串序列（如字符串列表等）的元素写入到文件中

  ```python
  f = open("test.txt", "w", encoding="utf-8")
  # f.writelines(["zhangsan","lisi","wangwu"])
  # zhangsanlisiwangwu
  f.writelines(["zhangsan\n","lisi\n","wangwu\n"])
  # zhangsan
  # lisi
  # wangwu
  f.close()
  ```

- os.rename(oldname,newname)文件重命名

- os.remove(filepath)删除文件

- 安全的打开关闭文件的方式(自动调用close方法)：

  ```python
  with open("d://test.txt","w") as f:
  	f.write("hello python")
  ```

  ```python
  with open("test.txt", "w", encoding="utf-8") as f:
      f.writelines(["zhangsan\n","lisi\n","wangwu\n"])
  ```

# 文件夹操作

- os.mkdir(path)：创建文件夹
- os.getcwd()：获取程序运行的当前目录
- os.listdir(path)：获取指定目录下的文件列表
- os.rmdir(path)：删除空文件夹
- shutil.rmtree(path)：删除非空文件夹
  - shutil：高级的文件、文件夹、压缩包处理模块

```python
import os
# 创建文件夹
os.mkdir("d://test_dir")
# 获取程序运行的当前目录
path = os.getcwd()
print(path)
# 获取指定目录下面所有文件
files_list = os.listdir("d://")
print(files_list)
# 删除空文件夹
os.rmdir("d://test_dir")
# 删除非空文件夹
import shutil
shutil.rmtree("d://test_dir")
```

# JSON格式文件操作

- 引入json模块：import json

- dumps(python_data)：将Python数据转换为JSON编码的字符串

- loads(json_data)：将json编码的字符串转换为python的数据结构

- dump(python_data, file)：将Python数据转换为JSON编码的字符串，并写入文件

- load(json_file)：从JSON数据文件中读取数据，并将JSON编码的字符串转换为python的数据结构

- Python数据类型与JSON类型对比

  | Python       | JSON       |
  | ------------ | ---------- |
  | dict         | {}         |
  | list, tuple  | []         |
  | str          | sring      |
  | int 或者 float | number     |
  | True/False   | true/false |
  | None         | null       |

将Python数据转换为JSON编码的字符串，

然后将json编码的字符串转换为python的数据结构：

```python
import json
disc = {"name":"zhangsan",
        "age":20,
        "language":["python", "java"],
        "study":{"AI":"python","bigdata":"hadoop"},
        "if_vip":True,
        }
# 将Python数据转换为JSON编码的字符串
json_str = json.dumps(disc)
print(json_str)
# {"name": "zhangsan", "age": 20, "language": ["python", "java"], "study": {"AI": "python", "bigdata": "hadoop"}, "if_vip": true}
print(type(json_str))# <class 'str'>

# 将json编码的字符串转换为python的数据结构
py_dict = json.loads(json_str)
print(py_dict)
# {'name': 'zhangsan', 'age': 20, 'language': ['python', 'java'], 'study': {'AI': 'python', 'bigdata': 'hadoop'}, 'if_vip': True}
print(type(py_dict))# <class 'dict'>
```

# CSV格式文件操作

- csv格式文件默认以逗号分隔
- 引入csv模块：import csv
- write写操作
  - writerow([row_data])一次写入一行数据
  - writerows([[row_data],[row_data],...])一次写入多行数据
- read读操作
  - reader(file_object)根据打开的文件对象返回一个可迭代reader对象
  - 可以使用next(reader)遍历reader对象，获取每一行数据
- DictWriter和DictReader对象处理Python字典类型的数据

write写操作：

```python
import csv
datas = [["name","age"],["zhangsan", 20],["lisi", 24]]
with open("d://user_info.csv","w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in datas:
        writer.writerow(row)
        # name,age
        # zhangsan,20
        # lisi,24
    # 一次写入
    writer.writerows(datas)
```

read读操作：

```python
# 读csv数据
with open("d://user_info.csv","r",newline="",encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)# ['name', 'age']
    print("-------")
    for row in reader:
        print(row)
        print(row[0])
        print(row[1])
```

字典数据操作

- 写：

```python
import csv
header = ["age","name"]
rows = [{"name":"zhangsan","age": 20},{"name":"lisi","age": 24}]
with open("d://user_info2.csv","w",newline="",encoding="utf-8") as f:
    writer = csv.DictWriter(f, header)
    writer.writeheader()
    writer.writerows(rows)
```

- 读

```python
import csv
header = ["name","age"]
with open("d://user_info2.csv","r",newline="",encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)# OrderedDict([('age', '20'), ('name', 'zhangsan')])
        print(row["name"],row["age"])# zhangsan 20
```

