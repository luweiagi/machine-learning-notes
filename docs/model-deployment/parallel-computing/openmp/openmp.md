# OpenMP

* [返回上层目录](../parallel-computing.md)
* [利用OpenMP来矩阵并行计算](#利用OpenMP来矩阵并行计算)
* [IDE配置openMP](#IDE配置openMP)
  * [VisualStudio配置openMP](#VisualStudio配置openMP)
  * [CLion配置openMP](#CLion配置openMP)



# 利用OpenMP来矩阵并行计算

我们想求如下的矩阵和向量乘法
$$
\begin{aligned}
y_{[3,1]}&=M_{[3,2]}x_{[2,1]}+b_{[3,1]}\\
\begin{bmatrix}
2.1\\ 
5.2\\ 
8.3
\end{bmatrix}
&=
\begin{bmatrix}
1 & 2\\ 
3 & 4\\ 
5 & 6
\end{bmatrix}
\begin{bmatrix}
1\\ 
0.5
\end{bmatrix}
+
\begin{bmatrix}
0.1\\ 
0.2\\ 
0.3
\end{bmatrix}
\end{aligned}
$$
但是，给的矩阵和向量的原始形式都是个一维数组，即
$$
\begin{aligned}
y_{[3,1]}&=M_{[3,2]}x_{[2,1]}+b_{[3,1]}\\
\begin{bmatrix}
2.1 & 5.2 & 8.3
\end{bmatrix}
&=
\begin{bmatrix}
1 & 3 & 5 & 2 & 4 & 6
\end{bmatrix}
\begin{bmatrix}
1 & 0.5
\end{bmatrix}
+
\begin{bmatrix}
0.1 & 0.2 & 0.3
\end{bmatrix}
\end{aligned}
$$
需要你把原始形式的转换成适合矩阵向量乘法的形式。

```c++
auto input_dim = 2;
auto output_dim = 3;
const std::vector<float> input = { 1, 0.5 };
const std::array<float, 6> weights = { 1, 3, 5, 2, 4, 6 };
const std::array<float, 3> biases = { 0.1, 0.2, 0.3 };
// 这个size_t W在不同的变，可以不用在模板里指定，真正由函数参数array<float, W>& weights指定
std::vector<float> output(output_dim);

auto input_ptr = input.data();
auto weights_ptr = weights.data();
auto biases_ptr = biases.data();
auto output_ptr = output.data();
#pragma omp parallel for
for (unsigned int i = 0; i < output_dim; i++) {
    for (unsigned int j = 0; j < input_dim; j++) {
    	*(output_ptr + i) += *(weights_ptr + j * output_dim + i) * *(input_ptr + j);
    }
    *(output_ptr + i) += *(biases_ptr + i);
}
```



# IDE配置openMP

## VisualStudio配置openMP

（1）在项目属性里按照下图进行配置，开启OpenMP支持。

![vs-config](pic/vs-config.png)

（2）开始写OpenMP的代码，例子：

```c++
void main() {
#pragma omp parallel for
    for(int i = 0; i < 100; ++i)
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
}
```

## CLion配置openMP

配置CMakeList.txt：

```cmake
cmake_minimum_required(VERSION 3.17)
project(openMP C)
set(CMAKE_C_STANDARD 99)
add_executable(openMP main.c)
# openMP 配置
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()
```

参考：[详解CLion配置openMP的方法](https://www.php1.cn/detail/XiangJie_CLion_P_8884a2b8.html)



## Makefile配置OpenMP

要在Makefile中的两处添加`-fopenmp`标志：

* 在`CXXFLAGS`中添加`-fopenmp`，否则编译器不支持OpenMP并行，并行不起作用。

  `CFLAGS = -pipe -std=c++11 -fopenmp -O2`

加了上面，如果编译报错，那就要在加上：

* 在`LFLAGS`处加上`-foenmp`。

  `LFLAGS = -fopenmp`



# 参考资料



===

* [多线程使用1--#pragma omp parallel for](https://blog.csdn.net/weixin_44210987/article/details/112388379)

讲了OpenMP的使用方法，以及其中的互斥，竞争，保护等操作。

[并行计算笔记(004)-OpenMP的简介](https://zhuanlan.zhihu.com/p/81025414)

[Openmp用法小结](https://zhuanlan.zhihu.com/p/530489840)