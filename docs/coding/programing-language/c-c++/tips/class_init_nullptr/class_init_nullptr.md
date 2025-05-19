# Class(std::nullptr_t) noexcept

* [返回上层目录](../tips.md)



### 能自己在写类时也用 `std::nullptr_t` 构造函数吗？

当然可以，而且 **这是现代 C++ 写资源类、句柄类、封装类的高级技巧之一**。特别适用于：

| 你想干什么                                           | 是否推荐用 `std::nullptr_t` 构造 |
| ---------------------------------------------------- | -------------------------------- |
| 写封装类包一堆外部资源（如：句柄、文件、网络、模型） | ✅ 非常推荐                       |
| 写一个类，但有时候想“先不初始化”                     | ✅ 推荐                           |
| 避免写一个丑陋的默认构造函数                         | ✅ 推荐                           |

🎯 示例：你自己实现一个类似的类

假设你封装一个“图像资源类”：

```c++
class Image {
public:
    Image(std::nullptr_t) noexcept : data_(nullptr) {}

    Image(const std::string& path) {
        LoadFromFile(path);
    }

    void LoadFromFile(const std::string& path) {
        // 假设这里做了图像读取
        data_ = new unsigned char[100];
    }

    ~Image() {
        delete[] data_;
    }

private:
    unsigned char* data_;
};
```

使用示例：

```c++
Image img(nullptr);  // 占位初始化
img.LoadFromFile("xxx.png");
```

✅ 总结

- `Session(std::nullptr_t)` 是一种 **安全的延迟初始化技巧**。
- 它常用于封装类（尤其是资源类），避免默认构造函数无意义。
- 你以后在写模型封装、图像封装、Socket类时，都可以借鉴这种写法。



# 问题

## noexcept为什么要加这个？

非常棒的问题！`noexcept` 是现代 C++ 中的重要关键字，作用是**显式声明函数不会抛出异常**。在 `Ort::Session::Session(std::nullptr_t) noexcept` 这样的声明中加上 `noexcept` 有几个关键理由：

🔍 1. **性能优化**

当编译器知道一个函数是 `noexcept` 的，它可以：

- **更大胆地优化代码路径**（比如省略异常处理结构）；
- **在移动构造 / 赋值等标准容器操作中选择更高效的路径**。

比如：

```cpp
std::vector<Ort::Session> sessions;
sessions.emplace_back(nullptr);  // 如果 Session 构造是 noexcept，vector 就能放心使用 move 构造。
```

2. **异常安全判断机制依赖于它**

标准库的很多容器，比如 `std::vector`、`std::optional`、`std::unique_ptr`，在内部实现中会检测你提供的类型是否有 `noexcept` 构造函数。

如果你声明了 `noexcept`，它们会选择更安全高效的方式管理你的对象（比如在扩容时更愿意移动而非复制）。

✅ 3. **告诉使用者这个函数不会抛异常**

`noexcept` 也具有**文档化的作用**，告诉调用者这个函数：

- 不会失败；
- 不需要你去 catch 或 try。

比如：

```c++
Ort::Session session(nullptr);  // 没必要担心 try/catch，因为构造函数是 noexcept。
```

4. **编译器做静态验证**

如果你写了一个 `noexcept` 的函数，但它内部调用了一个可能抛异常的函数，**编译器会报错**。这是一种静态安全性校验：

```c++
void f() noexcept {
    throw std::runtime_error("bad");  // ❌ 编译器会报错：违反了 noexcept
}
```

🎯 使用场景总结

| 场景                       | 是否建议加 `noexcept`          |
| -------------------------- | ------------------------------ |
| 构造函数用于占位、空初始化 | ✅ 很建议                       |
| 析构函数（尤其管理资源）   | ✅ 很建议                       |
| 移动构造 / 移动赋值        | ✅ 非常建议（标准容器高度依赖） |
| 涉及外部库、会抛异常的操作 | ❌ 不要加，除非你保证不抛       |

🧠 总结一句话：

> 加上 `noexcept` 可以 **提升性能、安全性和可读性**，也是现代 C++ 封装类的“好公民行为”。

你自己封装类的时候，如果某个构造函数或函数 **绝不会抛异常**，那就大胆加上 `noexcept`，编译器和容器都会因此更高效地帮你工作。需要我给你一个例子模板也可以～



## 为什么要加参数std::nullptr_t，又没用到

为什么要这么写呢： 

`mage(std::nullptr_t) noexcept : data_(nullptr) {}` 

写成这样不行吗： 

`mage() noexcept : data_(nullptr) {}`

因为`std::nullptr_t`并没有被用到啊



这是个非常细节但非常有价值的问题！这两种写法看起来都能初始化 `data_ = nullptr`，但它们**语义不同**，**用途不同**，不能随便替换。我们来详细讲清楚：

✅ 两种写法对比：

✅ 写法一：`Mage(std::nullptr_t) noexcept`

```C++
Mage(std::nullptr_t) noexcept : data_(nullptr) {}
```

> 👉 **含义**：这是一个**专门为 nullptr 定义的构造函数**，也就是说你可以这样做：

```C++
Mage m(nullptr);  // ✅ 调用 Mage(std::nullptr_t)
```

这个构造函数明确表明：“如果你用 `nullptr` 来构造这个类，我知道该怎么处理。”
 这对类支持一些 STL 操作或智能指针语义非常重要。

🚫 写法二：`Mage() noexcept`

```C++
Mage() noexcept : data_(nullptr) {}
```

> 👉 **含义**：这是一个**无参构造函数**，只能通过以下方式调用：

```C++
Mage m1;       // ✅ 默认构造
Mage m2();     // ✅ 默认构造（注意是函数声明不是构造）
Mage m3(nullptr);  // ❌ 报错！找不到接受 nullptr 的构造函数
```

🤔 所以为什么用 `std::nullptr_t` 呢？

1. **给 nullptr 单独设计的构造函数重载**

让类支持：

```C++
Mage m(nullptr);  // 明确调用这个构造函数
```

这在 STL 容器、自定义智能指针、自定义资源句柄类型里非常常见，比如：

```C++
std::unique_ptr<Foo> ptr(nullptr);  // unique_ptr 就有 unique_ptr(std::nullptr_t) 构造函数
```

2. **防止歧义**

如果你写的是：

```C++
Mage(int x = 0);
```

那这会导致：

```C++
Mage m(nullptr);  // ❌ 有可能错误调用 Mage(int x = 0)，出现类型转换，隐藏 bug
```

但如果你写：

```C++
Mage(std::nullptr_t);
```

就明确表示这是给 `nullptr` 用的，防止编译器将 `nullptr` 误判为 `0`（老式 `int` 兼容性）引发歧义。

🧠 总结一句话：

> `Mage(std::nullptr_t)` 是 C++11 提供的机制，让类可以专门处理 `nullptr` 构造，这比 `Mage()` 更明确、类型安全，常用于需要表现“空对象”、“无效句柄”的类中。

✅ 建议你今后自己写句柄类 / 智能指针类时，也可以写上：

```C++
MyHandle(std::nullptr_t) noexcept : ptr_(nullptr) {}
```

需要的话我可以给你一个例子，比如一个自定义智能指针支持 `nullptr` 构造的完整写法。

