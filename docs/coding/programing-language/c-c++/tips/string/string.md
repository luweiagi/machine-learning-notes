# string字符串类

* [返回上层目录](../tips.md)
* [substr提取部分字符串](#substr提取部分字符串)
* [c_str()转为char\*数组](#c_str()转为char\*数组)
* [to_string数值转字符串](#to_string数值转字符串)
* [char[]，char *，string之间转换](#char[]，char *，string之间转换)
* [string赋值](#string赋值)

# substr提取部分字符串

```c++
string str_new = str.substr(0, length);
```

# c_str()转为char\*数组

```c++
const char* cstr = str.c_str();
```

# to_string数值转字符串

功能：将数值转化为字符串。返回对应的字符串。

函数原型：

```c++
string to_string (int val);
string to_string (long val);
string to_string (long long val);
string to_string (unsigned val);
string to_string (unsigned long val);
string to_string (unsigned long long val);
string to_string (float val);
string to_string (double val);
string to_string (long double val);
```

# char[]，char *，string之间转换

[char[]，char *，string之间转换](https://blog.csdn.net/yzhang6_10/article/details/51164300)

# string赋值

```c++
#include <iostream>
using namespace std;

string num1 = string("111");

string s;
s.push_back(1 + '0');
```


