# volatile关键字

* [返回上层目录](../tips.md)



一般说来，volatile用在如下的几个地方：

1、中断服务程序中修改的供其它程序检测的变量需要加volatile；

2、多任务环境下各任务间共享的标志应该加volatile；

3、存储器映射的硬件寄存器通常也要加volatile说明，因为每次对它的读写都可能由不同意义；

另外，以上这几种情况经常还要同时考虑数据的完整性（相互关联的几个标志读了一半被打断了重写），在1中可以通过关中断来实现，2中可以禁止任务调度，3中则只能依靠硬件的良好设计了。

# 参考资料

* [详解C/C++中volatile关键字](https://blog.csdn.net/weixin_44363885/article/details/92838607)

===

* [嵌入式C语言中的关键字volatile](https://blog.csdn.net/Last_Impression/article/details/134821755)

