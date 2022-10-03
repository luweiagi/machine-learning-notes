# 模式匹配和样例类

* [返回上层目录](../scala.md)



# 模式匹配

模式匹配是检查某个值（value）是否匹配某一个模式的机制，一个成功的匹配同时会将匹配值解构为其组成部分。它是Java中的`switch`语句的升级版，同样可以用于替代一系列的 if/else 语句。

## 语法

一个模式匹配语句包括一个待匹配的值，`match`关键字，以及至少一个`case`语句。

```scala
import scala.util.Random

val x: Int = Random.nextInt(10)

x match {
  case 0 => "zero"
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}
```

上述代码中的`val x`是一个0到10之间的随机整数，将它放在`match`运算符的左侧对其进行模式匹配，`match`的右侧是包含4条`case`的表达式，其中最后一个`case _`表示匹配其余所有情况，在这里就是其他可能的整型值。

`match`表达式具有一个结果值

```scala
def matchTest(x: Int): String = x match {
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}
matchTest(3)  // other
matchTest(1)  // one
```







# 样例类case class

案例类（Case classes）和普通类差不多，只有几点关键差别，接下来的介绍将会涵盖这些差别。案例类非常适合用于不可变的数据。下一节将会介绍他们在模式匹配中的应用。

## 定义一个案例类

一个最简单的案例类定义由关键字`case class`，类名，参数列表（可为空）组成：

```scala
case class Book(isbn: String)
val frankenstein = Book("978-0486282114")
```

注意在实例化案例类`Book`时，并没有使用关键字`new`，这是因为案例类有一个默认的`apply`方法来负责对象的创建。

当你创建包含参数的案例类时，这些参数是公开（public）的`val`

```scala
case class Message(sender: String, recipient: String, body: String)
val message1 = Message("guillaume@quebec.ca", "jorge@catalonia.es", "Ça va ?")

println(message1.sender)  // prints guillaume@quebec.ca
message1.sender = "travis@washington.us"  // this line does not compile
```

你不能给`message1.sender`重新赋值，因为它是一个`val`（不可变）。在案例类中使用`var`也是可以的，但并不推荐这样。

## 比较

案例类在比较的时候是按值比较而非按引用比较：

```scala
case class Message(sender: String, recipient: String, body: String)

val message2 = Message("jorge@catalonia.es", "guillaume@quebec.ca", "Com va?")
val message3 = Message("jorge@catalonia.es", "guillaume@quebec.ca", "Com va?")
val messagesAreTheSame = message2 == message3  // true
```

尽管`message2`和`message3`引用不同的对象，但是他们的值是相等的，所以`message2 == message3`为`true`。

## 拷贝

你可以通过`copy`方法创建一个案例类实例的浅拷贝，同时可以指定构造参数来做一些改变。

```scala
case class Message(sender: String, recipient: String, body: String)
val message4 = Message("julien@bretagne.fr", "travis@washington.us", "Me zo o komz gant ma amezeg")
val message5 = message4.copy(sender = message4.recipient, recipient = "claire@bourgogne.fr")
message5.sender  // travis@washington.us
message5.recipient // claire@bourgogne.fr
message5.body  // "Me zo o komz gant ma amezeg"
```

上述代码指定`message4`的`recipient`作为`message5`的`sender`，指定`message5`的`recipient`为”claire@bourgogne.fr”，而`message4`的`body`则是直接拷贝作为`message5`的`body`了。

# 样例类的模式匹配

案例类非常适合用于模式匹配。

```scala
abstract class Notification

case class Email(sender: String, title: String, body: String) extends Notification

case class SMS(caller: String, message: String) extends Notification

case class VoiceRecording(contactName: String, link: String) extends Notification
```

`Notification` 是一个虚基类，它有三个具体的子类`Email`, `SMS`和`VoiceRecording`，我们可以在这些案例类(Case Class)上像这样使用模式匹配：

```scala
def showNotification(notification: Notification): String = {
  notification match {
    case Email(sender, title, _) =>
      s"You got an email from $sender with title: $title"
    case SMS(number, message) =>
      s"You got an SMS from $number! Message: $message"
    case VoiceRecording(name, link) =>
      s"you received a Voice Recording from $name! Click the link to hear it: $link"
  }
}
val someSms = SMS("12345", "Are you there?")
val someVoiceRecording = VoiceRecording("Tom", "voicerecording.org/id/123")

println(showNotification(someSms))  // prints You got an SMS from 12345! Message: Are you there?

println(showNotification(someVoiceRecording))  // you received a Voice Recording from Tom! Click the link to hear it: voicerecording.org/id/123
```

`showNotification`函数接受一个抽象类`Notification`对象作为输入参数，然后匹配其具体类型。（也就是判断它是一个`Email`，`SMS`，还是`VoiceRecording`）。在`case Email(sender, title, _)`中，对象的`sender`和`title`属性在返回值中被使用，而`body`属性则被忽略，故使用`_`代替。



# 参考资料

* [模式匹配](https://docs.scala-lang.org/zh-cn/tour/pattern-matching.html)

"模式匹配"参考此博客。

* [案例类（CASE CLASSES）](https://docs.scala-lang.org/zh-cn/tour/case-classes.html)
* [模式匹配](https://docs.scala-lang.org/zh-cn/tour/pattern-matching.html)

"样例类case class"参考此博客。

