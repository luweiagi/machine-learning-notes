# 高级类型

* [返回上层目录](../scala.md)
* [单例类型](#单例类型)



# 单例类型

所有的对象实例都有一个`x.type`的单例类型，它只对应当前对象实例。这么做有什么意义呢？

从[这里](http://scalada.blogspot.com/2008/02/thistype-for-chaining-method-calls.html)看到一种情况，在“链式”调用风格下，有适用的场景：

```scala
class A {def method1: A = this }
class B extends A {def method2: B = this}

val b = new B
b.method2.method1  // 可以工作
b.method1.method2  // 不行，提示：error: value method2 is not a member of A
```

有些人很喜欢用 `x.foo.bar` 这样的方式连续的去操作，这种风格也成为”链式调用”风格，它要求方法返回的必须是当前对象类型，以便连贯的调用方法。不过上面，因为父类中声明的method1方法返回类型限制死了就是A类型(不写返回值类型，用类型推导也一样)，导致子类对象调用完method1之后，类型已经变成了父类型，无法再调用子类型中的方法了。解决方法是：

```scala
class A { def method1: this.type = this } 
class B extends A { def method2 : this.type = this } 

val b = new B
b.method1.method2  // ok
```

把返回类型都改为了 `this.type` 单例类型，就灵了。它利用了`this`关键字的动态特性来实现的，在执行`b.method1` 的时候，`method1`返回值类型`this.type` 被翻译成了`B.this.type`

```scala
scala> b.method1
res0: b.type = B@ca5bdb6
```

这样不同的对象实例在执行该方法的时候，返回的类型也是不同的(都是当前实例的单例类型)。

小结，单例类型是个特殊的类型，单例类型绑定(依赖)在某个对象实例上，每个对象实例都有它的单例类型。不过它的场景并不多见。





# 参考资料

* [scala类型系统：3) 单例类型与this.type](http://hongjiang.info/scala-type-system-singleton-type/)

"单例类型"参考此文章。

