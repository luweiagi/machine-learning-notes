# CUDA

* [返回上层目录](../parallel-computing.md)





===

[CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)

目标机只有Nvidia的GPU及GPU驱动，不允许装CUDA，能跑自己写的CUDA吗？

> 这样肯定不行，cuda的编译器是nvcc。除非你在有cuda的环境编译好了，再把程序放在这台机器跑。只需要安装显卡驱动就行。如只是跑简单加法的例子是可以的，因为nvcc默认是静态链接，没有其他依赖。
>
> 跑不起来，你只要安装了显卡驱动就有cudaruntime了，没有cudatoolkit，你没法编译cuda程序
>
> 装了驱动可以跑的，把编译好的二进制拿进去跑，没问题
>
> 有驱动的话，拿到接口交叉编译之后上来跑就可以吧。

