# JSON

* [返回上层目录](../data-format.md)
* [C++中的nlohmann::json库](#C++中的nlohmann::json库)
  * [安装配置nlohmann::json库](#安装配置nlohmann::json库)
  * [示例代码](#示例代码)
  * [JOSN构造方法](#JOSN构造方法)
  * [多个JSON处理](#多个JSON处理)
* [遍历与查询](#遍历与查询)
  * [遍历](#遍历)
    * [遍历json的键（key）](#遍历json的键（key）)
  * [查询](#查询)
    * [确认某个键是否存在于json中](#确认某个键是否存在于json中)



# C++中的nlohmann::json库

C++中流行的JSON库比如**nlohmann/json**库，它在C++中使用非常方便。

## 安装配置nlohmann::json库

**（1）下载JSON库**

下载单个[json.hpp](https://github.com/nlohmann/json/releases/latest/download/json.hpp)，

**（2）将json.hpp文件添加到你的项目中**

将下载的 `json.hpp` 文件放入你的项目目录中，例如放在 `include` 文件夹下。假设你将 `json.hpp` 放在了 `./external/nlohmann/json.hpp` 路径下。

**（3）在CMake中包含路径**

在你的 `CMakeLists.txt` 文件中，将 `json.hpp` 文件所在的路径添加到 `include_directories` 中：

```c++
include_directories(${CMAKE_SOURCE_DIR}/external/nlohmann)
```

这样，CMake就可以找到`json.hpp`，你可以在C++代码中直接引用：

```c++
#include <json.hpp>  // 引入 nlohmann/json.hpp
```

只下载json.hpp就可以了吗？不用下载源码编译安装吗？

> 是的，**nlohmann/json** 是一个 **头文件库**，因此你只需要下载 `json.hpp` 文件就可以使用它，而不需要单独编译或安装它的源码。这个库的设计方式非常简单：它将所有功能都包含在一个单一的头文件中，这使得它在 C++ 项目中非常容易集成。

## 示例代码

- **C++ 端** 使用`cppzmq`发送JSON格式的数据。我们使用 `nlohmann/json` 库将C++数据结构转化为JSON字符串，然后发送给Python端。
- **Python 端** 使用 `pyzmq` 接收C++端发送的JSON数据，并通过`json`库将其解析为Python对象。

这个方案可以扩展到任何需要在C++和Python之间传递JSON数据的场景，ZeroMQ提供了高效的异步消息传递机制，适合用作跨语言的数据通信。

**（1）C++ 端代码（使用 `cppzmq` 和 `nlohmann/json`）**

首先，我们将C++数据结构转化为JSON格式，并通过`cppzmq`发送出去。

C++代码：

```c++
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

// 使用 nlohmann::json 来处理JSON数据
using json = nlohmann::json;

int main() {
    // 创建 ZeroMQ 上下文和 PUSH 套接字
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::push);
    
    // 绑定到 TCP 地址
    socket.bind("tcp://*:5555");

    // 创建一个 JJSON
    json data;
    data["name"] = "Alice";
    data["age"] = 30;
    data["city"] = "Wonderland";

    // 将 JSOJSON为字符串
    std::string message = data.dump();  // `dump()` 返回一个字符串

    // 打印消息，方便调试
    std::cout << "Sending message: " << message << std::endl;

    // 将消息发送出去
    zmq::message_t zmq_message(message.size());
    memcpy(zmq_message.data(), message.c_str(), message.size());

    socket.send(zmq_message, zmq::send_flags::none);

    return 0;
}
```

**（2）Python 端代码（使用 `pyzmq`）**

Python端接收到JSON格式的消息后，可以将其解析为Python对象并进行处理。

Python代码：

```python
import zmq
import json

# 创建 ZeroMQ 上下文和 PULL 套接字
context = zmq.Context()
socket = context.socket(zmq.PULL)

# 连接到 C++ 端的 TCP 地址
socket.connect("tcp://localhost:5555")

while True:
    # 接收消息
    message = socket.recv_string()

    # 将JSON字符串转换为 Python 对象
    data = json.loads(message)

    # 打印接收到的数据
    print(f"Received message: {data}")

    # 你可以处理 JJSON，例如访问字段：
    print(f"Name: {data['name']}, Age: {data['age']}, City: {data['city']}")
```

## JOSN构造方法

### 1、静态大括号初始化

```c++
json j = {
    {"name", "Alice"},
    {"age", 30},
    {"city", "Wonderland"},
    {"address", {
        {"street", "123 Main St"},
        {"zip", "12345"}
    }}
};
```

### 2、使用键值对动态添加

可以通过 `json` 对象的下标操作符 `[]` 来动态添加键值对，这种方式非常直观和灵活：

```c++
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    json j;

    j["name"] = "Alice";          // 动态添加键值对
    j["age"] = 30;
    j["city"] = "Wonderland";

    j["address"]["street"] = "123 Main St"; // 嵌套JSON对象
    j["address"]["zip"] = "12345";

    std::cout << j.dump(4) << std::endl; // 格式化输出
    return 0;
}
```

**优点**：

- 灵活，按需动态添加键值对。
- 支持嵌套结构，例如 `j["address"]["street"]`。

**输出结果**：

```json
{
    "address": {
        "street": "123 Main St",
        "zip": "12345"
    },
    "age": 30,
    "city": "Wonderland",
    "name": "Alice"
}
```

### 3、使用 emplace 动态插入

如果你想更显式地插入键值对，可以使用 `emplace()` 方法，它类似于 `std::map` 的 `emplace`：

```c++
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    json j;

    j.emplace("name", "Alice");       // 插入键值对
    j.emplace("age", 30);
    j.emplace("city", "Wonderland");

    json address;
    address.emplace("street", "123 Main St");
    address.emplace("zip", "12345");

    j.emplace("address", address);   // 插入子对象

    std::cout << j.dump(4) << std::endl;
    return 0;
}
```

**优点**：

- 显式的插入方式，更适合需要从多个地方动态生成的场景。
- 更清晰地表达插入操作。

### 4、使用数组动态生成

对于JSON数组，可以动态添加元素，例如：

```c++
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    json j;

    j["name"] = "Alice";
    j["age"] = 30;

    // 动态生成数组
    j["hobbies"] = json::array();
    j["hobbies"].push_back("reading");   // 添加到数组中
    j["hobbies"].push_back("swimming");
    j["hobbies"].push_back("coding");

    std::cout << j.dump(4) << std::endl;
    return 0;
}
```

输出：

```json
{
    "age": 30,
    "hobbies": [
        "reading",
        "swimming",
        "coding"
    ],
    "name": "Alice"
}
```

说明：

- `json::array()` 创建一个空数组。
- `push_back()` 动态向数组中添加元素。

### 5、使用迭代器构造

如果有数据源，可以使用迭代器来动态构造JSON。例如：

```c++
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>

using json = nlohmann::json;

int main() {
    // 使用 vector 动态生成JSON数组
    std::vector<std::string> hobbies = {"reading", "swimming", "coding"};
    json j;

    j["name"] = "Alice";
    j["hobbies"] = json(hobbies.begin(), hobbies.end()); // 从迭代器生成数组

    std::cout << j.dump(4) << std::endl;
    return 0;
}
```

输出：

```json
{
    "hobbies": [
        "reading",
        "swimming",
        "coding"
    ],
    "name": "Alice"
}
```

### 6、从std::map或其他容器动态生成

可以直接从 `std::map` 或类似的容器生成JSON对象：

```c++
#include <nlohmann/json.hpp>
#include <iostream>
#include <map>

using json = nlohmann::json;

int main() {
    // 使用 map 生成JSON
    std::map<std::string, std::string> address = {
        {"street", "123 Main St"},
        {"zip", "12345"}
    };

    json j;
    j["name"] = "Alice";
    j["address"] = address; // 直接从 map 生成JSON对象

    std::cout << j.dump(4) << std::endl;
    return 0;
}
```

优点：

- 与标准容器无缝对接，非常方便。
- 动态数据结构如 `std::map` 可以直接生成JSON对象。

### 7、从字符串解析

有时候你可能从其他地方拿到一个JSON字符串，直接解析即可：

```c++
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    std::string json_string = R"({
        "name": "Alice",
        "age": 30,
        "city": "Wonderland",
        "address": {
            "street": "123 Main St",
            "zip": "12345"
        }
    })";

    json j = json::parse(json_string); // 从字符串解析JSON

    std::cout << j.dump(4) << std::endl;
    return 0;
}
```

**输出**： 和原始JSON字符串相同。

### 8、从文件动态生成

如果JSON数据存储在文件中，可以直接从文件中加载：

```c++
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

using json = nlohmann::json;

int main() {
    std::ifstream file("data.json"); // 打开JSON文件
    json j;
    file >> j; // 从文件中读取JSON数据

    std::cout << j.dump(4) << std::endl;
    return 0;
}
```

### 总结：灵活生成JSON的多种方式

| 方法                     | 适用场景                          |
| ------------------------ | --------------------------------- |
| **静态大括号初始化**     | 已知结构固定的场景                |
| **动态键值对添加（[]）** | 动态构造嵌套JSON                  |
| **emplace 插入**         | 显式插入、更复杂的场景            |
| **动态数组生成**         | 动态构造数组                      |
| **从迭代器生成**         | 从容器（`vector` 等）生成JSON数组 |
| **从 std::map 生成**     | 从键值对容器生成JSON对象          |
| **字符串解析**           | 从JSON字符串动态生成JSON对象      |
| **文件读取**             | 从文件读取JSON                    |

你可以根据具体场景选择适合的生成方式。例如：

- 数据量较多且来源于容器时，用迭代器或 `std::map`。
- 需要动态生成复杂结构时，用 `[]` 或 `emplace`。

## 多个JSON处理

### 1、多个JSON合并处理

如果已经有两个JSON了，能把这两个JSON各加一个名字组成一个大的JSON吗？

是的，可以通过给每个已有的JSON对象加一个名字（键），并将它们组合成一个大的JSON对象。以下是实现的方法和示例代码：

**实现方法**

1. 创建一个新的JSON对象。
2. 给每个已有的JSON对象赋予一个名字（键）。
3. 将它们插入到新的JSON对象中。

**示例代码**

```c++
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    // 已有的两个JSON对象
    json json1 = {
        {"name", "Alice"},
        {"age", 30}
    };

    json json2 = {
        {"city", "Wonderland"},
        {"zip", "12345"}
    };

    // 创建一个新的JSON对象，并将两个JSON组合进去
    json big_json;
    big_json["person"] = json1;  // 给第一个JSON添加名字 "person"
    big_json["location"] = json2; // 给第二个JSON添加名字 "location"

    // 打印组合后的JSON对象
    std::cout << big_json.dump(4) << std::endl;

    return 0;
}
```

**输出结果**

运行上述代码的输出结果为：

```json
{
    "person": {
        "name": "Alice",
        "age": 30
    },
    "location": {
        "city": "Wonderland",
        "zip": "12345"
    }
}
```

### 2、灵活处理多个JSON对象

如果需要处理动态数量的JSON对象，可以通过循环将它们添加到一个大的JSON对象中。例如：

示例代码（动态组合多个JSON对象）：

```c++
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>

using json = nlohmann::json;

int main() {
    // 准备多个JSON对象
    std::vector<json> json_objects = {
        {{"name", "Alice"}, {"age", 30}},
        {{"city", "Wonderland"}, {"zip", "12345"}},
        {{"country", "Fantasyland"}, {"language", "Common"}}
    };

    // 创建一个大的JSON对象
    json big_json;

    // 动态给每个JSON添加名字并插入到 big_json 中
    for (size_t i = 0; i < json_objects.size(); ++i) {
        big_json["object" + std::to_string(i + 1)] = json_objects[i];
    }

    // 打印组合后的JSON对象
    std::cout << big_json.dump(4) << std::endl;

    return 0;
}
```

输出结果

```json
{
    "object1": {
        "name": "Alice",
        "age": 30
    },
    "object2": {
        "city": "Wonderland",
        "zip": "12345"
    },
    "object3": {
        "country": "Fantasyland",
        "language": "Common"
    }
}
```

**注意事项**

1. **键的唯一性**：
   - 如果多个JSON对象需要放入一个大JSON中，每个对象需要一个唯一的键。可以手动指定键，也可以根据规则（如循环中的索引）动态生成键。
2. **对象结构**：
   - JSON对象的嵌套没有限制，可以自由组合复杂结构。
3. **JSON动态插入**：
   - 你可以动态地向已有的JSON添加数据，也可以用 `emplace` 或下标 `[]` 操作动态扩展。

### 3、合并多个JSON的键值对

如果希望把两个JSON的键值对合并到一个对象中（而不是嵌套结构），可以直接使用 `merge_patch` 或手动添加：

示例（合并两个JSON的键值对）：

```c++
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

int main() {
    json json1 = {
        {"name", "Alice"},
        {"age", 30}
    };

    json json2 = {
        {"city", "Wonderland"},
        {"zip", "12345"}
    };

    // 合并键值对
    json1.update(json2);

    std::cout << json1.dump(4) << std::endl;
    return 0;
}
```

输出结果：

```json
{
    "name": "Alice",
    "age": 30,
    "city": "Wonderland",
    "zip": "12345"
}
```

### 4、总结

- 如果需要将多个JSON嵌套组合，直接通过 `big_json["key"] = json;` 动态构造。
- 如果需要合并键值对，可以使用 `update` 或遍历插入键值。
- `nlohmann::json` 提供了强大的动态操作能力，适合各种场景下的JSON生成与组合。



# 遍历与查询

## 遍历

### 遍历json的键（key）

你可以通过 `json` 对象的 `items()` 方法或者 `begin()` 和 `end()` 方法来遍历 JSON 对象中的键值对。

**示例代码：**

```c++
#include <iostream>
#include <nlohmann/json.hpp>
using namespace std;
using json = nlohmann::json;

int main() {
    // 创建一个示例 JSON 对象
    json j = {
        {"name", "Alice"},
        {"age", 30},
        {"city", "Wonderland"}
    };

    // 遍历 JSON 的所有键值对
    for (auto& [key, value] : j.items()) {
        cout << "Key: " << key << ", Value: " << value << endl;
    }

    return 0;
}
```

在这个例子中，`j.items()` 返回一个可以迭代的键值对序列，你可以直接通过 `auto& [key, value]` 来解构出每一个键（`key`）和对应的值（`value`）。

## 查询

### 确认某个键是否存在于json中

你可以使用 `json` 对象的 `contains()` 方法来检查一个键是否存在于 JSON 对象中。`contains()` 返回一个 `bool` 值，表示该键是否存在。

**示例代码：**

```c++
#include <iostream>
#include <nlohmann/json.hpp>
using namespace std;
using json = nlohmann::json;

int main() {
    // 创建一个示例 JSON 对象
    json j = {
        {"name", "Alice"},
        {"age", 30},
        {"city", "Wonderland"}
    };

    // 检查是否存在某个键
    string key_to_check = "name";
    if (j.contains(key_to_check)) {
        cout << "The key '" << key_to_check << "' exists. Value: " << j[key_to_check] << endl;
    } else {
        cout << "The key '" << key_to_check << "' does not exist." << endl;
    }

    // 检查另一个不存在的键
    key_to_check = "country";
    if (j.contains(key_to_check)) {
        cout << "The key '" << key_to_check << "' exists. Value: " << j[key_to_check] << endl;
    } else {
        cout << "The key '" << key_to_check << "' does not exist." << endl;
    }

    return 0;
}
```

