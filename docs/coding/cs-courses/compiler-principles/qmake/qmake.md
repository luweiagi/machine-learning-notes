# QMake生成Makefile

* [返回上层目录](../compiler-principles.md)



# 通过qmake生成Makefile文件

这是在linux系统下生成的，windows系统下还没尝试过。

假设已知有如下文件和文件夹

```
│ ------main.cpp
└─mymath
  ------mycpp.h
  ------mymath.cpp
```

除了一个子文件夹`mymath`和三个程序文件以外，没有其他任何文件，即没有QT使用的`.pro`项目文件。

其中，`main.cpp`中内容为：

```cpp
#include <stdio.h>
#include "mymath.h"

int main() {
	printf("Hello World!\n");
	add(1, 2);
	return 0;
}
```

`mycpp.h`内容为：

```cpp
void add(int a, int b);
```

`mycpp.cpp`内容为：

```cpp
#include <stdio.h>
#include "mymath.h"

void add(int a, int b) {
	int c = a + b;
	printf("c = %d\n", c);
}
```

现在生成Makefile文件，需要两步：

（1）生成`.pro`项目文件

```shell
qmake -project
```

首先你要找到qmake命令所在的路径，在`/home/xxx/Qt5.0.0/5.0.0/gcc_64/bin/`中，需要你添加这个路径。

注意，**需要打开生成的`.pro`项目文件，在里面加上`CONFIG -= qt`**，这是因为生成的是非QT的C++项目，加上这个等效于你创建项目时选择非QT的C++项目（创建非QT的C++项目，其`.pro`项目文件里就有这句话），不然，下一步生成的Makefile文件中会有依赖QT相关的库，不仅多余，还会大概率报错，因为你可能没配置相关库的路径。

（2）生成Makefile文件。

```shell
qmake -o Makefile hello.pro
```

这样就生成了Makefile文件，和`.pro`项目文件在同一个目录下。

还可以加上(yejun)

```
qmake -o Makefile hello.pro -spec linux-g++ CONFIG+=qtquickcompiler
```

或者(zhengen)

```
qmake -spec linux-g++-64 -o Makefile hello.pro
```



# 参考资料

* [Qt环境设置及使用](https://www.95pm.com/index.php/post/130811.html)

完成Qt应用程序的编写后，需要进行编译和运行。以下是一个简单的编译和运行步骤：1. 打开命令提示符，进入Qt应用程序所在的目录。2. 输入命令“qmake -project”生成项目文件。3. 输入命令“qmake”生成Makefile文件。4. 输入命令“make”进行编译。5. 输入命令“.\应用程序名称.exe”运行应用程序。

* [Linux 用qmake快速生成makefile](http://www.taodudu.cc/news/show-369459.html)

Makefile可以像这样由".pro"文件生成：

```shell
qmake -o Makefile hello.pro
```

现在你的目录下已经产生了一个 Makefile 文件，输入"make" 指令就可以开始编译 hello.c 成执行文件，执行 ./hello 和 world 打声招呼吧！打开这个Makefile文件看看，是不是很专业啊！