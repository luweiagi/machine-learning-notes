# 数据结构

* [返回上层目录](../scala.md)



# Option选项

Scala Option(选项)类型用来表示一个值是可选的（有值或无值)。

Option[T] 是一个类型为 T 的可选值的容器：

* 如果值存在， Option[T] 就是一个 Some[T] ，
* 如果不存在， Option[T] 就是对象 None 。

接下来我们来看一段代码：

```scala
// 虽然 Scala 可以不定义变量的类型，不过为了清楚些，我还是
// 把他显示的定义上了
 
val myMap: Map[String, String] = Map("key1" -> "value")
val value1: Option[String] = myMap.get("key1")
val value2: Option[String] = myMap.get("key2")
 
println(value1) // Some("value1")
println(value2) // None
```

在上面的代码中，myMap是一个Key的类型是String，Value的类型是String的hash map，但不一样的是他的get()返回的是一个叫Option[String]的类别。

Scala使用Option[String]来告诉你：「我会想办法回传一个 String，但也可能没有String给你」。

myMap里并没有key2这笔数据，get()方法返回None。

Option有两个子类别，一个是Some，一个是None，当他回传Some的时候，代表这个函式成功地给了你一个String，而你可以透过get()这个函数拿到那个String，如果他返回的是None，则代表没有字符串可以给你。

## 模式匹配

你也可以通过模式匹配来输出匹配值。实例如下：

```scala
val sites = Map("runoob" -> "www.runoob.com", "google" -> "www.google.com")

println("show(sites.get( \"runoob\")) : " + show(sites.get( "runoob")) )
println("show(sites.get( \"baidu\")) : " +
show(sites.get( "baidu")) )
}

def show(x: Option[String]) = x match {
    case Some(s) => s
    case None => "?"
}
```

执行以上代码，输出结果为：

```scala
show(sites.get( "runoob")) : www.runoob.com
show(sites.get( "baidu")) : ?
```

## Option常用方法

### getOrElse()

你可以使用getOrElse()方法来获取元组中存在的元素或者使用其默认的值，实例如下：

```scala
val a:Option[Int] = Some(5)
val b:Option[Int] = None 

println("a.getOrElse(0): " + a.getOrElse(0) )
println("b.getOrElse(10): " + b.getOrElse(10) )
```

执行以上代码，输出结果为：

```scala
a.getOrElse(0): 5
b.getOrElse(10): 10
```

### isEmpty()

你可以使用isEmpty()方法来检测元组中的元素是否为None，实例如下：

```scala
val a:Option[Int] = Some(5)
val b:Option[Int] = None 

println("a.isEmpty: " + a.isEmpty )
println("b.isEmpty: " + b.isEmpty )
```

执行以上代码，输出结果为：

```scala
a.isEmpty: false
b.isEmpty: true
```

### foreach()

如果选项包含有值，则将每个值传递给函数f， 否则不处理。

```scala
def foreach[U](f: (A) => U): Unit
```

注意，该函数会自动检测是否为空，无需再在前面加isEmpty()判断。为空就不会继续执行foreach了。

# 字符串

## 字符串插值

自2.10.0版本开始，Scala提供了一种新的机制来根据数据生成字符串：字符串插值。字符串插值允许使用者将变量引用直接插入处理过的字面字符中。如下例：

```scala
val name="James"
println(s"Hello,$name")//Hello,James
```

在上例中， `s”Hello,$name” `是待处理字符串字面，编译器会对它做额外的工作。待处理字符串字面通过“号前的字符来标示（例如：上例中是s）。字符串插值的实现细节在 [SIP-11](https://docs.scala-lang.org/sips/pending/string-interpolation.html) 中有全面介绍。

Scala 提供了三种创新的字符串插值方法：s,f 和 raw.

**s字符串插值器**

在任何字符串前加上s，就可以直接在串中使用变量了。你已经见过这个例子：

```scala
val name="James"
println(s"Hello,$name")//Hello,James 
//此例中，$name嵌套在一个将被s字符串插值器处理的字符串中。
//插值器知道在这个字符串的这个地方应该插入这个name变量的值，
//以使输出字符串为Hello,James。
//使用s插值器，在这个字符串中可以使用任何在处理范围内的名字。
```

字符串插值器也可以处理任意的表达式。例如：

```scala
println(s"1+1=${1+1}") 
//将会输出字符串1+1=2。任何表达式都可以嵌入到${}中。
```

**f插值器**

在任何字符串字面前加上 f，就可以生成简单的格式化串，功能相似于其他语言中的 printf 函数。当使用 f 插值器的时候，所有的变量引用都应当后跟一个printf-style格式的字符串，如%d。看下面这个例子：

```scala
val height=1.9d
val name="James"
println(f"$name%s is $height%2.2f meters tall")//James is 1.90 meters tall 
/*f 插值器是类型安全的。如果试图向只支持int的格式化串传入一个double值，编译器则会报错。例如：

val height:Double=1.9d

scala>f"$height%4d"
<console>:9: error: type mismatch;
 found : Double
 required: Int
           f"$height%4d"
              ^ f 插值器利用了java中的字符串数据格式。这种以%开头的格式在 [Formatter javadoc] 中有相关概述。如果在具体变量后没有%，则格式化程序默认使用 %s（串型）格式。*/
```

**raw插值器**

除了对字面值中的字符不做编码外，raw 插值器与 s 插值器在功能上是相同的。如下是个被处理过的字符串：

```scala
scala>s"a\nb"
res0:String=
a
b 这里，s 插值器用回车代替了\n。而raw插值器却不会如此处理。

scala>raw"a\nb"
res1:String=a\nb 当不想输入\n被转换为回车的时候，raw 插值器是非常实用的。
```

除了以上三种字符串插值器外，使用者可以自定义插值器。

























# 向量和矩阵

## org.apache.spark.mllib.linalg

这个包里有

DenseVector, Vector, Vectors, SparseVector

DenseMatrix, Matrix, SparseMatrix

### org.apache.spark.mllib.linalg.Matrix

初始化并打印矩阵：

```scala
import org.apache.spark.mllib.linalg._
val m1: Matrix = new DenseMatrix(
    3, 4, Array(
        1.0, 0.0, 2.0, 0.0, 
        3.0, 1.0, 2.0, 1.0, 
        0.0, 1.0, 1.0, 0.0)
)
/*m1: org.apache.spark.mllib.linalg.Matrix =
1.0  0.0  2.0  1.0
0.0  3.0  1.0  1.0
2.0  1.0  0.0  0.0*/

// 按列打印每一列
for (col <- m1.colIter) { 		 
    println(col.toArray.mkString(","))
}
/*1.0,0.0,2.0
0.0,3.0,1.0
2.0,1.0,0.0
1.0,1.0,0.0*/
```







## Breeze.linalg

这个包里有DenseVector, SparseVector, DenseMatrix, SparseMatrix。

### breeze.linalg.DenseMatrix

scala 中有一个包叫做breeze.linalg.DenseMatrix，是专门针对矩阵运算的，目前只是用到DenseMatrix这一个，暂时讲解一下他的用法

DenseMatrix即稠密矩阵与SparseMatrix（稀疏矩阵相对应）

为了写起来方便，可以这样对其重命名为：

```scala
import breeze.linalg.{DenseMatrix => BDM}
```

即BDM相当于DenseMatrix。

**一、随机生成一个矩阵**

```scala
val a :BDM[Double] = BDM.rand(2,10)
val b :BDM[Double] = BDM.rand(2,10)
```

a和b均是二维矩阵，数值是随机生成的。

**二、访问矩阵中的元素**

1、按行访问，这样可以获得一行的所有数据

```scala
val a1 = a(0,::)
val b1 = b(0,::)
```

括号中第一个参数为行数，后面::代表所有的列，即这一行的所有数据

2、访问单个元素

```scala
val a5 = a.data(4)
val b7 = b.data(6)
```

data是他的一个方法，参数为我们要访问的第几个元素。

> 注意1、索引是从0开始的，所以第5个元素的索引值为4
>
> 注意2、矩阵的存储是按列存储的，也就是第一列之后接着是第二列的数据，并不是按照行来存储的，这一点和MATLAB一样





# 其他

## 使用Range来填充一个集合

Problem
​    你想要使用Range来填充一个List，Array，Vector或者其他的sequence。

Solution
​    对于支持range方法的集合你可以直接调用range方法，或者创建一个Range对象然后把它转化为一个目标集合。

在第一个解决方案中，我们调用了伴生类的range方法，比如Array，List，Vector，ArrayBuffer等等：

```scala
scala> Array.range(1, 10)
res83: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9)
 
scala> List.range(1, 10)
res84: List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9)
 
scala> Vector.range(0, 10, 2)
res85: scala.collection.immutable.Vector[Int] = Vector(0, 2, 4, 6, 8)
```

对于一些集合，比如List，Array，你也可以创建一个Range对象，然后把它转化为相应的目标集合：

```scala
scala> val a = (1 to 10).toArray
a: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
 
scala> val l = (1 to 10) by 2 toList
warning: there were 1 feature warning(s); re-run with -feature for details
l: List[Int] = List(1, 3, 5, 7, 9)
 
scala> val l = (1 to 10).by(2).toList
l: List[Int] = List(1, 3, 5, 7, 9)
```

 我们来看看那些集合可以由Range直接转化的：

```scala

def toArray: Array[A]
def toBuffer[A1 >: Int]: Buffer[A1]
def toIndexedSeq: IndexedSeq[Int]
def toIterator: Iterator[Int]
def toList: scala.List[Int]
def toMap[T, U]: collection.Map[T, U]
def toParArray: ParArray[Int]
def toSet[B >: Int]: Set[B]
def toStream: Stream[Int]
def toTraversable: collection.Traversable[Int]
def toVector: scala.Vector[Int]
```

使用这种方案我们可以把Range转为Set等，不支持range方法的集合类：

```scala
scala> val set = Set.range(0, 5)
<console>:8: error: value range is not a member of object scala.collection.immutable.Set
       val set = Set.range(0, 5)
                     ^
 
scala> val set = Range(0, 5).toSet
set: scala.collection.immutable.Set[Int] = Set(0, 1, 2, 3, 4)
 
scala> val set = (0 to 10 by 2).toSet
set: scala.collection.immutable.Set[Int] = Set(0, 10, 6, 2, 8, 4)
```

 你也可以创建一个字符序列：

```scala
scala> val letters = ('a' to 'f').toList
letters: List[Char] = List(a, b, c, d, e, f)
 
scala> val letters = ('a' to 'f' by 2).toList
letters: List[Char] = List(a, c, e)
```

Range还能用于for循环：

```scala
scala> for(i <- 0 until 10 by 2) println(i)
0
2
4
6
8
```

**Discussion**

通过对Range使用map方法，你可以创建出了Int，char之外，其他元素类型的集合

```scala
scala> val l = (1 to 3).map(_ * 2.0).toList
l: List[Double] = List(2.0, 4.0, 6.0)
```

使用同样的方案，你可以创建二元祖集合：

```scala
scala> val t = (1 to 5).map(e => (e, e*2))
t: scala.collection.immutable.IndexedSeq[(Int, Int)] = Vector((1,2), (2,4), (3,6), (4,8), (5,10))
```

二元祖集合很容易转换为Map：

```scala
scala> val map = t.toMap
map: scala.collection.immutable.Map[Int,Int] = Map(5 -> 10, 1 -> 2, 2 -> 4, 3 -> 6, 4 -> 8)
```

## Scala中的Infinity和NaN

### Infinity

`i == i + 1`，一个数字永远不会等于它自己加1？

无穷大加1还是无穷大。

你可以用任何被计算为无穷大的浮点算术表达式来初始化i，例如：

```scala
double i = 1.0 / 0.0;
```

不过，你最好是能够利用标准类库为你提供的常量：

```scala
double i = Double.POSITIVE_INFINITY;
```

事实上，很多情况下都不需要真正使用无限大来表示无限大，许多足够大的浮点数都可以实现这一目的，例如：

```scala
double i = 1.0e40;
```

同时Scala提供了一些检测方法：

```scala
Double.isInfinite();
Double.isFinite();
```

举例：

```scala
scala> val a = 22.0
a: Double = 22.0

scala> a.isInfinite
res0: Boolean = false

scala> val b = 2.0/0
b: Double = Infinity

scala> b.isInfinite
res1: Boolean = true

scala> b.isPosInfinity
res4: Boolean = true
```

### NaN

`i==i`，一个数字总是等于它自己?

IEEE 754浮点算术保留了一个特殊的值用来表示一个不是数字的数量。这个值就是NaN（Not a Number），对于所有没有良好的数字定义的浮点计算，例如0.0/0.0，其值都是它。规范中描述道，NaN不等于任何浮点数值，包括它自身在内。

你可以用任何计算结果为NaN的浮点算术表达式来初始化i，例如：

```scala
double i = 0.0 / 0.0;
```

同样，为了表达清晰，你可以使用标准类库提供的常量：

```scala
double i = Double.NaN;
```

NaN还有其他的惊人之处。

```scala
Double.NaN == Double.NaN //结果是false。但是，
Double a = new Double(Double.NaN);
Double b = new Double(Double.NaN);]
a.equals(b);  //true
```

任何浮点操作，只要它的一个或多个操作数为NaN，那么其结果为NaN。这条规则是非常合理的，但是它却具有奇怪的结果。例如，下面的程序将打印false：

```scala
double i = 0.0 / 0.0;
System.out.println(i - i == 0);
```

总之，float 和double 类型都有一个特殊的NaN 值，用来表示不是数字的数量。

同时Java和Scala提供了一些检测方法：

```scala
Double.isNaN();
```

举例如下：

```scala
scala> val a = 22.0
a: Double = 22.0

scala> a.isNaN
res0: Boolean = false

scala> val b = 0.0/0
b: Double = NaN

scala> a.isNaN
res1: Boolean = true
```

## 系统时间和时间戳

### 获取时间戳

```scala
System.currentTimeMillis()
```

### 获取系统时间

```scala
import java.text.SimpleDateFormat
import java.util.Date

val df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
val systemTime1 = df.format(new Date())
val systemTime2 = df.format(System.currentTimeMillis())
```

## MD5Hash

获取md5 hash的16进制的字符串，并转化为Long型整数。

```scala
import java.security.MessageDigest
def hashMD5(content: String): String = {
      val md5 = MessageDigest.getInstance("MD5")
      val encoded = md5.digest((content).getBytes)
      encoded.map("%02x".format(_)).mkString
    }

val a = hashMD5("abcd")

java.lang.Long.decode("0x" + a.slice(0,10))
```

注意：只取hash字符串的前10位是因为为了保证转换后的数值范围处于Long整型范围内。





# 参考资料

* [Scala Option(选项)](https://www.runoob.com/scala/scala-options.html)

"Option选项"参考此博客。

* [字符串插值](https://docs.scala-lang.org/zh-cn/overviews/core/string-interpolation.html)

"字符串插值"参考此博客。

* [Element by Element Matrix Multiplication in Scala](https://stackoverflow.com/questions/51304156/element-by-element-matrix-multiplication-in-scala)

"org.apache.spark.mllib.linalg.Matrix"参考此博客。

* [Scala中的矩阵Breeze.linalg.DenseMatrix](https://blog.csdn.net/qq_18293213/article/details/53308125)

* [快速了解Breeze(二)](https://blog.csdn.net/zhuqing2020/article/details/37605553)
* [spark | scala | 线性代数库Breeze学习](https://blog.csdn.net/xxzhangx/article/details/74066679)
* [Cool Breeze of Scala for Easy Computation: Introduction to Breeze Library](https://blog.knoldus.com/cool-breeze-of-scala-for-easy-computation-introduction-to-breeze-library/)

"breeze.linalg.DenseMatrix"参考此博客。

* [scala使用Range来填充一个集合](https://blog.csdn.net/qq_36330643/article/details/76483551)

"使用Range来填充一个集合"参考此博客。

* [Scala中的Infinity和NaN](https://blog.csdn.net/u013007900/article/details/79225016)

"Scala中的Infinity和NaN"参考此博客。

* [scala获取当前系统时间的两种方式](https://blog.csdn.net/qq_34885598/article/details/86583307)

"系统时间和时间戳"参考此博客。

* [scala Md5加密](https://blog.csdn.net/sunny_xsc1994/article/details/90606893)

"MD5Hash"参考此博客。

