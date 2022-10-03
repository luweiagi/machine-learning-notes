# Scala简介

* [返回上层目录](../scala.md)



# 单机编译并运行scala

将下述代码

```scala
object HelloWorld extends App { 
    if (args.length > 0) println("hello, " + args(0)) 
    else println("HelloWorld") 
} 
```

写到`HelloWorld.scala`中，然后使用scalac编译，再用scala执行。

```shell
# 编译 
scalac HelloWorld.scala 
# 运行（输出运行所花费的时间） 
scala -Dscala.time HelloWorld 
```









# 参考资料

* [Scala入门系列（六）：面向对象之object](https://www.cnblogs.com/LiCheng-/p/8022289.html)

"单机编译并运行scala"参考此博客。



