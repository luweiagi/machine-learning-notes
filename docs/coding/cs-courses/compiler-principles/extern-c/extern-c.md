# 关于extern "C"

* [返回上层目录](../compiler-principles.md)
* [gcc和g++的区别](#gcc和g++的区别)
  * [基础知识](#基础知识)
  * [测试](#测试)
    * [编译使用g++，链接使用g++，成功](#编译使用g++，链接使用g++，成功)
    * [编译使用gcc，链接使用g++，成功](#编译使用gcc，链接使用g++，成功)
    * [编译使用g++，链接使用gcc，失败](#编译使用g++，链接使用gcc，失败)
    * [编译使用gcc/g++，链接使用gcc-lstdc++，成功](#编译使用gcc/g++，链接使用gcc-lstdc++，成功)
* [extern"C"](#extern"C")
  * [C++引用C](#C++引用C)
    * [C++不加extern“C”，失败](#C++不加extern"C"，失败)
    * [C++加上extern“C”，成功](#C++加上extern"C"，成功)
    * [不能给C函数本身加extern"C",gcc不识别会报错](#不能给C函数本身加extern"C",gcc不识别会报错)
  * [C引用C++](#C引用C++)
    * [C++不加extern“C”，失败](#C++不加extern"C"，失败)
    * [C++加上extern“C”，成功](#C++加上extern"C"，成功)
    * [C++加上extern“C”且C++引用不带extern的自身头文件，失败](#C++加上extern"C"且C++引用不带extern的自身头文件，失败)
    * [C程序加上extern“C”，失败](#C程序加上extern"C"，失败)
    * [C程序引用带#ifdef-C++且带extern的C++头文件，成功](#C程序引用带#ifdef-C++且带extern的C++头文件，成功)
    * [C程序调用C++的类，待补充，未完成](#C程序调用C++的类，待补充，未完成)

一句话总结：

1、C++里如果某个函数提前声明了extern "C",那该函数就会以C语言去编译，即函数名不变

2、C语言里禁止出现extern "C",因为C语言编译器不认该关键字

# gcc和g++的区别

一句话总结：

1、gcc和g++均能编译C语言和C++，区别是，gcc遇到.c按C语言编译，遇到.cpp按照C++编译，而g++不管遇到.c还是.cpp均按照C++编译

2、如果C++语言使用了自己的LIB库（如输入输出<iostream>），则gcc无法链接，必须要加`-lstdc++`才行，而g++可以链接C++的LIB库。

## 基础知识

* GNU is a free operating system
* GNU's not unix;递归：G means GNU,整句缩写也是GNU.
* GCC(GNU编译器套件)：**GNU Compiler Collection。**可以编译C、C++、JAVA、Fortran、Pascal、Object-C、Ada等语言
  * gcc是GCC中的GNU C Compiler（**C 编译器**）
  * g++是GCC中的GNU C++ Compiler（**C++编译器**）

gcc既能编译C语言程序（后缀为.c识别为C语言程序，按照C语言来编译），也能编译C++程序（后缀为.cpp识别为C++程序，按照C++来编译），**但无法链接C++的库**，需手动添加`-lstdc++ -shared-libgcc`选项，表示gcc在编译C++程序时可以链接必要的C++标准库。

g++则无论是C语言程序（后缀为.c）还是C++程序（后缀为.cpp）均当做C++程序来编译。

更多关于gcc和g++的知识：

[GCC编译器——GCC编译器的简介](https://blog.csdn.net/oqqHuTu12345678/article/details/125043255)

[gcc和g++是什么，有什么区别？](http://c.biancheng.net/view/7936.html)

## 测试

```c++
// /////math.cpp/////
int add(int a, int b) {
	return a + b;
}

// /////main.cpp/////
#include <stdio.h>
#include <iostream>
using namespace std;

//extern "C" int add(int, int);
int add(int, int);

int main() {
	int a = 1, b = 2;
	int ret = add(a, b);
	printf("ret = %d\n", ret);
	cout << "ret = " << ret << endl;
	return 0;
}
```

### 编译使用g++，链接使用g++，成功

```shell
# 分别编译math.cpp和main.cpp
g++ -c math.cpp -o math.o  # 成功！
g++ -c main.cpp -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 成功！
```

### 编译使用gcc，链接使用g++，成功

```shell
# 分别编译math.cpp和main.cpp
gcc -c math.cpp -o math.o  # 成功！
gcc -c main.cpp -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 成功！
```

### 编译使用g++，链接使用gcc，失败

```shell
# 分别编译math.cpp和main.cpp
g++ -c math.cpp -o math.o  # 成功！
g++ -c main.cpp -o main.o  # 成功！
# 链接
gcc main.o math.o -o app  # 失败！
```

错误原因：

```shell
main.o: In function `main':
main.cpp:(.text+0x48): undefined reference to `std::cout'
...
main.cpp:(.text+0x72): undefined reference to `std::ostream::operator<<(std::ostream& (*)(std::ostream&))'
collect2: error: ld returned 1 exit status
```

意思就是找不到头文件`#include <iostream>`的函数。

其根本原因就在于，main.cpp中使用了C++标准库 `<iostream>` 和提供的类对象，而C++和C文件中标准库STL文件的命名方式不同，gcc默认是无法找到它们的，所以在链接时会出错。

编译执行C++程序，使用gcc和g++也是有区别的。要知道，很多C++程序都会调用某些标准库中现有的函数或者类对象，而单纯的gcc指令是无法自动链接这些标准库文件的。如果想使用gcc指令来编译执行C++程序，需要在使用gcc指令时，手动为其添加`-lstdc++ -shared-libgcc`选项，表示gcc在编译C++程序时可以链接必要的C++标准库。

> C语言标准库和C++标准库有什么不同？
>
> 一、规模不同。C++标准库内容庞大许多，涵盖范围也要广得多。
> 二、功能不同。C++标准库功能更强大。
> 三、使用范围不同。鉴于两种语言本身的区别，这种差别是显而易见的。
> ……
>
> 虽然都是根据编程需要去使用库，如C语言的stdio.h、stdlib.h、string.h、time.h等，C++的algorithm、iostream、vector等，但是后者明显更适应现代编程方法的要求，特别是标准模板库、容器类等标准类库的提出，大大提升了编程的效率。  

### 编译使用gcc/g++，链接使用gcc-lstdc++，成功

```shell
# 分别编译math.cpp和main.cpp
gcc/g++ -c math.cpp -o math.o  # 成功！
gcc/g++ -c main.cpp -o main.o  # 成功！
# 链接
gcc main.o math.o -o app -lstdc++  # 成功！
gcc main.o math.o -o app -lstdc++ -shared-libgcc  # 成功！
```

# extern"C"

## C++引用C

### C++不加extern“C”，失败

```cpp
// /////math.c/////
int add(int a, int b) {
	return a + b;
}

// /////main.cpp/////
#include <iostream>
using namespace std;

//extern "C" int add(int, int);
int add(int, int);

int main() {
    int a = 1, b = 2;
    int ret = add(a, b);
    cout << "ret = " << ret << endl;
    return 0;
}
```

然后编译并链接：

```shell
# 分别编译math.c和main.cpp
gcc -c math.c -o math.o  # 成功！
gcc -c main.cpp -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 失败！
gcc main.o math.o -o app -lstdc++  # 失败！
```

报错：

```shell
main.o: In function `main':
main.cpp:(.text+0x21): undefined reference to `add(int, int)'
collect2: error: ld returned 1 exit status
```

这是因为，

gcc编译main.cpp时（`gcc -c main.cpp -o main.o`），会按照C++语法去编译，编译的main.o二进制文件中，add函数会被改写为add_ini_int，

而gcc编译math.c时（gcc -c math.c -o math.o），是按照C语言语法去编译，编译的math.o二进制文件中，add函数依然是add，

所以，在链接阶段（g++ main.o math.o -o app），main.o中的add_ini_int函数，由于和math.o中的add函数名称不一致，所以找不到匹配add_ini_int的具体实现，就会报错，说main.cpp找不到add(int, int)的定义。

### C++加上extern“C”，成功

```cpp
// /////math.c/////
int add(int a, int b) {
	return a + b;
}

// /////main.cpp/////
#include <iostream>
using namespace std;

extern "C" int add(int, int);

int main() {
    int a = 1, b = 2;
    int ret = add(a, b);
    cout << "ret = " << ret << endl;
    return 0;
}
```

然后编译并链接：

```shell
# 分别编译math.c和main.cpp
gcc -c math.c -o math.o  # 成功！
gcc -c main.cpp -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 成功！
gcc main.o math.o -o app -lstdc++  # 成功！
```

**成功**，这是因为，

gcc编译main.cpp时（`gcc -c main.cpp -o main.o`），会按照C++语法去编译，但是遇到了`extern "C"`修饰的函数add，这就是要求gcc在编译C++文件时，对该函数add保持原名，即add（而不是add_int_int），编译的main.o二进制文件中，add函数就是原名add，而不会被改写为add_ini_int，

而gcc编译math.c时（gcc -c math.c -o math.o），是按照C语言语法去编译，编译的math.o二进制文件中，add函数依然是add，

所以，在链接阶段（g++ main.o math.o -o app），main.o中的add函数，会找到math.o中的add函数，就成功链接生成可执行二进制文件了。

### 不能给C函数本身加extern"C",gcc不识别会报错

```c++
// /////math.c/////
extern "C" int add(int, int);

int add(int a, int b) {
	return a + b;
}

// /////main.cpp/////
#include <iostream>
using namespace std;

extern "C" int add(int, int);

int main() {
    int a = 1, b = 2;
    int ret = add(a, b);
    cout << "ret = " << ret << endl;
    return 0;
}
```

然后编译：

```shell
# 编译math.cpp
gcc -c math.c -o math.o  # 成功！
```

失败：

> math.c:1:8: error: expected identifier or ‘(’ before string constant
>  extern "C" int add(int, int);

这说明gcc不识别`"C"`，会报错，gcc会认为这个句子本应该是类似`extern str = "C"`。

但是g++可以编译：

```shell
# 编译math.cpp
g++ -c math.c -o math.o  # 成功！
```

编译的math.o里，add函数名字就是add，而不是add_int_int。

然后可以成功链接，总结一下，即

```shell
# 分别编译math.c和main.cpp
# 因为math.c里含有extern "C"， gcc不识别
#gcc -c math.c -o math.o  # 失败
g++ -c math.c -o math.o  # 成功！
gcc -c main.cpp -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 成功！
gcc main.o math.o -o app -lstdc++  # 成功！
```

有个小疑问，就是g++编译的math.o里（`g++ -c math.c -o math.o`），会把add函数最终生成为什么呢？

add还是add_int_int?

经过`vim math.o`查看，是add。

所以，当C++引用C时，如果是用gcc编译器，那就不要在C语言文件中加入`extern "C"`这样的语句，哪怕是在头文件中。而是应当在C++文件中引用C语言的头文件时，标注`extern "C"`或者用`extern "C" {}`把C语言的头文件包住。

什么叫用`extern "C" {}`把C语言的头文件包住？即

```c++
// /////main.cpp/////

#include <iostream>
using namespace std;

//extern "C" int add(int, int);
extern "C" {
	int add(int, int);
}

int main() {
	int a = 1, b = 2;
	int ret = add(a, b);
	cout << "ret = " << ret << endl;
	return 0;
}
```

## C引用C++

记住两点：

1、C++里如果某个函数提前声明了extern "C",那该函数就会以C语言去编译，即函数名不变

2、C语言里禁止出现extern "C",因为C语言编译器不认该关键字

### C++不加extern“C”，失败

```c++
// /////math.cpp/////
#include <iostream>
using namespace std;

int add(int a, int b) {
	cout << "c = " << a << " + " << b << " = " << a + b << endl;
    return a + b;
}

// /////main.c/////
#include <stdio.h>

//extern "C" int add(int, int);
int add(int, int);

int main() {
	int a = 1, b = 2;
	int ret = add(a, b);
	printf("ret = %d\n", ret);
	return 0;
}
```

然后编译并链接：

```shell
# 分别编译math.cpp和main.c
gcc -c math.cpp -o math.o  # 成功！
gcc -c main.c -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 失败！
gcc main.o math.o -o app -lstdc++  # 失败！
```

失败：

> main.o: In function `main':
> main.c:(.text+0x21): undefined reference to `add'
> collect2: error: ld returned 1 exit status

gcc编译math.cpp（gcc -c math.cpp -o math.o）时，是按照C++语法编译的，那么math.o里的add函数必然是add_int_int。而main.o里的add函数依然是add，所以main.o里的add找不到对应的实现，就报错了。

### C++加上extern“C”，成功

既然C++不加extern“C”失败了，那要怎么才能成功呢？

我们自然想到了给C++加上extern“C”，即

```c++
// /////math.cpp/////
#include <iostream>
using namespace std;

extern "C" int add(int, int);

int add(int a, int b) {
	cout << "c = " << a << " + " << b << " = " << a + b << endl;
    return a + b;
}

// /////main.c/////
#include <stdio.h>

//extern "C" int add(int, int);
int add(int, int);

int main() {
	int a = 1, b = 2;
	int ret = add(a, b);
	printf("ret = %d\n", ret);
	return 0;
}
```

然后编译并链接：

```shell
# 分别编译math.cpp和main.c
gcc -c math.cpp -o math.o  # 成功！
gcc -c main.c -o main.o  # 成功！
# 链接
g++ main.o math.o -o app  # 成功！
gcc main.o math.o -o app -lstdc++  # 成功！
```

编译成功！

gcc编译math.cpp生成的math.o里的add函数依然是add，而不是add_int_int。

### C++加上extern“C”且C++引用不带extern的自身头文件，失败

**注意！！！注意！！！注意！！！**：

前面的main.c函数，是没有加math.h的，如果存在math.h，且math.h里有`int add(int, int);`，那就得分情况了

如果你的math.h是这样：

```c++
#include <iostream>
using namespace std;

int add(int, int);  // 相当于#include "math.h"
extern "C" int add(int, int);

int add(int a, int b) {
	cout << "c = " << a << " + " << b << " = " << a + b << endl;
    return a + b;
}
```

编译math.cpp，`gcc -c math.cpp -o math.o`，会报错：

> math.cpp:5:16: error: conflicting declaration of ‘int add(int, int)’ with ‘C’ linkage
>  extern "C" int add(int, int);
>                 ^~~
> math.cpp:4:5: note: previous declaration with ‘C++’ linkage
>  int add(int, int);
>      ^~~

意思是，冲突了，你写` extern "C" int add(int, int);`不就是要告诉编译器这个函数按照C语言的语法来编译么，那你头文件里又加了`int add(int, int);`声明，那这意思就是默认按照C++编译器，这不就冲突了么。

### C程序加上extern“C”，失败

卧槽，上面的失败了，那就病急乱投医，给C程序加上`extern "C" int add(int, int);`试试，其实很显然，这会失败，因为C语言程序里不识别extern "C"。

为了彻底死心，还是再试试吧...

```c++
// /////math.cpp/////
#include <iostream>
using namespace std;

int add(int a, int b) {
	cout << "c = " << a << " + " << b << " = " << a + b << endl;
    return a + b;
}

// /////main.c/////
#include <stdio.h>

extern "C" int add(int, int);

int main() {
	int a = 1, b = 2;
	int ret = add(a, b);
	printf("ret = %d\n", ret);
	return 0;
}
```

编译main.c，`gcc -c main.c -o main.o`，报错：

> main.c:3:8: error: expected identifier or ‘(’ before string constant
>  extern "C" int add(int, int);
>         ^~~
> main.c: In function ‘main’:
> main.c:8:12: warning: implicit declaration of function ‘add’ [-Wimplicit-function-declaration]
>   int ret = add(a, b);

很显然，错了，因为C语言编译器会认为你的`extern "C"`应该是`extern char* str = "C"`，记住，只有C++编译器才识别`extern "C"`。

经过了实验，并得出了两点惨痛教训：

1、C++里如果某个函数提前声明了extern "C",那该函数就会以C语言去编译，即函数名不变

2、C语言里禁止出现extern "C",因为C语言编译器不认该关键字

### C程序引用带#ifdef-C++且带extern的C++头文件，成功

经过了这么多挫折，只有一次成功，太打击人了。

成功的地方在于，我只需要给C++源文件里提前声明要调用的函数为extern "C"，该C++文件被编译时，该函数就会以C的方式去编译。

失败的地方在于，C语言不认extern "C"啊，那就不能在C++的头文件里加上extern "C"。

这样会造成不便：

1、如果C++的头文件要给C语言引用，那我只能把C++的头文件声明为不带extern "C"的

2、如果C++的头文件要给C++本身的源码实现引用，那我就必须把C++的头文件声明为带extern "C"的

面对这样的矛盾，解决办法似乎只有一个，那就是生成两个C++的头文件：

* 一个不带extern "C"，这个专门给C语言源码引用
* 另一个带extern "C"，这个专门给C++源码引用

这好坑爹啊，太麻烦了，能不能只生成一个头文件，既能给C语言源码引用，又能给C++源码引用？

啊，想出来了，利用C++编译器有__cplusplus这个宏定义，而C语言编译器没有这个宏定义，我们可以把C++头文件写成：

```c++
// /////math.h/////
#ifdef _cplusplus
extern "C"
{
#endif

int add(int a, int b);

#ifdef _cplusplus
extern "C"
}
#endif
```

这样不就完美了么！

参考资料：[C中如何调用C++函数](https://blog.csdn.net/BoArmy/article/details/8652870)

### C程序调用C++的类，待补充，未完成

[C中如何调用C++函数?](https://blog.csdn.net/qq_29011249/article/details/105402456)

[c 文件中调用 cpp 中函数](https://blog.csdn.net/w839687571/article/details/120252956)