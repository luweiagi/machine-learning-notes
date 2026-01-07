# ONNX

- [返回上层目录](../model-deployment-practice.md)
- [模型导出为ONNX格式](#模型导出为ONNX格式)
- [模型推理](#模型推理)
  - [Python端推理](#Python端推理)
  - [C++端CPU推理](#C++端CPU推理)



# 模型导出为ONNX格式

写一个包装模型类，把 `act()` 变成 `forward()`

```python
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.act(x)  # 注意这里是调用 act 而不是 forward
```

然后这样导出：

```python
model = Module()
model.eval()
wrapped_model = ModelWrapper(model)

dummy_input = torch.randn(1, 5)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["a1", "a2", "param"],
                  opset_version=11,
                  dynamic_axes={  # 加上这个就会变为batch_size变为动态的
                      "input": {0: "batch_size"},  # 让第0维（batch）是动态的
                      "a1": {0: "batch_size"},
                      "a2": {0: "batch_size"},
                      "param": {0: "batch_size"}
                  })
```

如何查看 ONNX 模型的输出名字？

你可以用 Python 脚本快速查看：

```python
import onnx

model = onnx.load("your_model.onnx")
for output in model.graph.output:
    print(output.name)
```

# 模型推理

## Python端推理

```python
ort_session = ort.InferenceSession("model.onnx")
input_tensor = np.random.randn(2, 5).astype(np.float32)
outputs = ort_session.run(None, {"input": input_tensor})
print(outputs)
# [array([2, 2], dtype=int64),
#  array([1, 1], dtype=int64),
#  array([[-0.09347501, 0.10457519, 0.09422486, -0.24867839],
#         [-0.09440675, 0.10584655, 0.08844154, -0.24860077]], dtype=float32)
# ]
a1, a2, param = outputs
print(a1)
# [2 2]
print(a2)
# [1 1]
print(param)
# [[-0.09347501  0.10457519  0.09422486 -0.24867839]
#  [-0.09440675  0.10584655  0.08844154 -0.24860077]]
```

## C++端CPU推理

使用 ONNX Runtime C++ API（CPU）

### 下载ONNX-Runtime-C++预编译包

你可以从官方发布页下载 C++ 静态/动态库：

- 官方 Release 页：[onnxruntime](https://github.com/microsoft/onnxruntime/releases)

选择一个适合你平台的包，例如：

```shell
onnxruntime-linux-x64-<version>.tgz
```

你可以在 Linux 中使用 `tar` 命令来解压缩这个 `.tgz` 文件。`.tgz` 是 `.tar.gz` 的缩写形式，表示经过 tar 打包后又用 gzip 压缩的文件。

```shell
tar -xzf onnxruntime-linux-x64-1.22.0.tgz
```

解压后，会看到：

```makefile
include/        # C++ 头文件
lib/            # 静态库或动态库
bin/            # 工具
```

如果是在windows下，可能最新的release版本里不包含win-x64的版本，需要在没那么最新的版本里找，比如：

[ONNX Runtime v1.20.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1) 中的 `onnxruntime-win-x64-1.20.1.zip`

### 项目目录结构

```shell
your_project/
├── onnxruntime-linux-x64-1.22.0/  # 解压后的 ONNX Runtime 文件夹
│   ├── include/
│   └── lib/
├── source
│   └── main.cpp                   # 你的主程序
├── models/
│   └── model.onnx                 # 你的模型文件
└── CMakeLists.txt                 # 构建文件（或 Makefile）
│
└── build/                         # 中间产物
│   ├── Makefile
│   └── CMakeCache.txt
│   └── CMakeFiles/
│   └── main                       # 如果你最终编译的程序名叫main
│   └── bin/
│       └── infer                  # 可执行程序，现在被放到了bin/子目录中
```

### 编写 C++ 推理代码

下面是完整 C++ ONNX Runtime 推理代码示例，**适配你的模型输入为 float32[batch_size, 5]，输出为三个张量**：

- `"a1"`：`int64[batch_size]`
- `"a2"`：`int64[batch_size]`
- `"param"`：`float32[batch_size, 4]`

```c++
int main1() {
    // 设置模型路径
    const char* model_path = "model.onnx";

    // 创建 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");

    // Session 选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // 多线程推理
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 创建 Session
    Ort::Session session(env, model_path, session_options);

    // 输入信息
    const char* input_name = "input";
    size_t batch_size = 1;
    std::vector<float> input_tensor_values(batch_size * 5, 1.0f);  // 每个样本5维，batch个样本
    std::vector<int64_t> input_dims = {static_cast<int64_t>(batch_size), 5};

    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_dims.data(), input_dims.size());

    // 输出信息
    std::vector<const char*> output_names = {
        "a1",
        "a2",
        "param"
    };

    // 推理
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        output_names.data(), output_names.size()
    );

    // 解析输出
    // 1. a1: int64[batch_size]
    int64_t* a1 = output_tensors[0].GetTensorMutableData<int64_t>();
    std::cout << "a1: ";
    for (size_t i = 0; i < batch_size; ++i) {
        std::cout << a1[i] << " ";
    }
    std::cout << std::endl;

    // 2. a2: int64[batch_size]
    int64_t* a2 = output_tensors[1].GetTensorMutableData<int64_t>();
    std::cout << "a2: ";
    for (size_t i = 0; i < batch_size; ++i) {
        std::cout << a2[i] << " ";
    }
    std::cout << std::endl;

    // 3. param: float32[batch_size, 4]
    float* param_mean = output_tensors[2].GetTensorMutableData<float>();
    std::cout << "param: " << std::endl;
    for (size_t i = 0; i < batch_size; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < 4; ++j) {
            std::cout << param_mean[i * 4 + j];
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}
```

#### 多线程推理

如果你想用多线程推理，可以用`session_options.SetIntraOpNumThreads(n)`。

`session_options.SetIntraOpNumThreads(n)` 是什么？

这是 ONNX Runtime 的一个 **线程控制接口**，用于控制**单个操作符（OP）内部的并行线程数**，也叫 **intra-op parallelism**（操作内并行度）。

🌱 举个例子更好理解：

比如模型中有一个 `MatMul`（矩阵乘法）操作，它本身是可以并行执行的：

```
[MatMul]
   └──-> 在一个大矩阵里分块做并行计算
```

当你设置：

```c++
session_options.SetIntraOpNumThreads(4);
```

意思是：ONNX Runtime 在执行 **一个操作（如 MatMul）时可以同时用 4 个线程** 来加速它。

⚙️ 和 `SetInterOpNumThreads` 的区别？

| 接口名                    | 含义                         | 控制范围               |
| ------------------------- | ---------------------------- | ---------------------- |
| `SetIntraOpNumThreads(n)` | 一个 OP 内部能用多少线程     | 控制单个操作的并行性   |
| `SetInterOpNumThreads(n)` | 多个 OP 是否可以同时并发执行 | 控制操作之间的并行执行 |

你当前的模型，如果是典型的推理模型（非计算图非常复杂的那种），**重点在于 SetIntraOpNumThreads**。

🧠 设置多少合适？

| 场景                   | 推荐设置                   |
| ---------------------- | -------------------------- |
| CPU 核心数为 4         | `SetIntraOpNumThreads(4)`  |
| 你要批量推理多个样本   | 可以适度提升线程数提高吞吐 |
| 单个线程够快、资源紧张 | 设置为 1，避免资源竞争     |

> ⚠️ 注意：如果你自己写多线程调用 ONNX 推理（比如多线程并发 Session.Run），那最好把 `SetIntraOpNumThreads` 设小一点，避免线程太多打架。

### 构建CMakeLists编译

示例 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(onnx_cpp_infer)

set(CMAKE_CXX_STANDARD 14)

# 设置可执行文件输出路径（放在 add_executable 之前）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# 设置 ONNX Runtime 安装路径
set(ORT_DIR "./onnxruntime-linux-x64-1.22.0")
# 如果 ORT_DIR 是相对路径，CMake 在某些平台
# （尤其是 macOS 或使用 install 命令时）解析 RPATH 时可能失败或生成错误路径。
get_filename_component(ORT_DIR ${ORT_DIR} ABSOLUTE)

# ONNX Runtime 头文件路径
include_directories(${ORT_DIR}/include)
# ONNX Runtime 库文件路径（用于链接）
link_directories(${ORT_DIR}/lib)

# 添加可执行文件
add_executable(infer source/main.cpp)

# 链接 ONNX Runtime 动态库
target_link_libraries(infer onnxruntime)

# ✅ 显式设置运行时库查找路径 (RPATH)
set_target_properties(infer PROPERTIES
  BUILD_RPATH "${ORT_DIR}/lib;$ORIGIN"  # 开发时查找 ORT_DIR
  INSTALL_RPATH "$ORIGIN"  # 发布时查找可执行文件目录（$ORIGIN）
)
# 设置 BUILD_RPATH 与 INSTALL_RPATH 分离：利于部署不依赖开发路径。

# ✅ 自动复制模型文件到 bin 路径
configure_file(
  ${CMAKE_SOURCE_DIR}/models/model.onnx
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/model.onnx
  COPYONLY
)

# ✅ 自动复制 onnxruntime 的动态库到 bin 目录
file(GLOB ONNX_LIBS "${ORT_DIR}/lib/libonnxruntime.so*")
foreach(libfile ${ONNX_LIBS})
  configure_file(
    ${libfile}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
    COPYONLY
  )
endforeach()
```

编译命令

```shell
cd your_project  # 进入项目目录
cmake -S . -B build  # 生成构建配置
cmake --build build  # 编译你的程序
```

或者写成一个脚本：

```
#!/bin/bash
# 自动构建脚本

set -e  # 有错误就退出

echo "[1] 清理旧构建..."
rm -rf ./build/*

echo "[2] 配置项目..."
cmake -S . -B build

echo "[3] 编译项目..."
cmake --build build

echo "[✅] 编译完成，可执行文件在 build/bin 中（如果你设置了输出路径）"
```

编译结果：

```
root@user:~/onnx_infer_cpu# . gen_exe.sh
[1] 清理旧构建...
[2] 配置项目...
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: /root/onnx_infer_cpu/build
[3] 编译项目...
[ 33%] Building CXX object CMakeFiles/infer.dir/source/main.cpp.o
[ 66%] Building CXX object CMakeFiles/infer.dir/source/onnx_infer/onnx_infer.cpp.o
[100%] Linking CXX executable bin/infer
[100%] Built target infer
[✅] 编译完成，可执行文件在 build/bin 中（如果你设置了输出路径）
```

### 运行

```shell
cd build/bin
./infer
```

运行结果（示例）

```
== a1 ==
2 1 0 3
== a2 ==
0 0 1 0
== param ==
[0.05, 0.21, 0.87, 0.33]
[...]
```

注意：

如果你的模型输入 shape 不同，比如 `{1, 4}`，只需修改 `input_dims` 和数据大小。

### 不使用cmake直接手动编译

方法1：

```shell
g++ source/main.cpp -I./onnxruntime/include -L./onnxruntime/lib -lonnxruntime -o infer
```

 然后从`/onnxruntime/lib`把`libonnxruntime.o.1.22.0`复制到程序根目录，改名或者软连接为`libonnxruntime.o.1`，运行

```shell
LD_LIBRARY_PATH=. ./infer
```

结果为：

```shell
a1: 2 
a2: 1 
param: 
  [-0.098551, 0.107577, 0.0857024, -0.254746]
```

方法 2：直接加 RPATH（推荐）

如果你不想每次都 `LD_LIBRARY_PATH`，可以直接这样编译：

```
g++ source/main.cpp -I./onnxruntime/include -L./onnxruntime/lib -Wl,-rpath='$ORIGIN' -lonnxruntime -o infer
```

其中：

- `$ORIGIN` 表示程序所在的目录
- `-Wl,-rpath=...` 是告诉程序“将这个路径作为运行时查找 `.so` 的地方”

运行时自动找当前目录的 `.so.1` 文件，就不会报错了。

