# CMake

* [返回上层目录](../compiler-principles.md)
* [简化通用版CMakeLists模板](#简化通用版CMakeLists模板)
* [支持OpemMP](支持OpemMP)

# 简化通用版CMakeLists模板

一段代码的结构如下所示，也就是有很多子文件夹。用`tree /f`（windows）或`tree -a`（linux）命令展示代码的目录结构：

```shell
.
├── CMakeLists.txt
├── Makefile
├── src
│   ├── src.cpp
│   └── src.h
├── src1
│   ├── main.cpp
│   ├── src1_1
│   │   └── src1_1.cpp
│   ├── src1.cpp
│   ├── src1.h
│   └── src2.cpp
└── TestTemp1
```

 其最简化通用版的CMakeLists模板如下所示：

```cmake
cmake_minimum_required(VERSION 3.21)
project(TestTemp1)
set(CMAKE_CXX_STANDARD 11)

set(prj_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src1)
file(GLOB_RECURSE root_src_files "${prj_src_dir}/*")
message(STATUS "root_src_files = ${root_src_files}")

set(PRJ_SRC_LIST)
list(APPEND PRJ_SRC_LIST ${root_src_files})
message("PRJ_SRC_LIST = ${PRJ_SRC_LIST}")

# 添加头文件路径搜索目录
include_directories(./)

add_executable(${PROJECT_NAME} ${PRJ_SRC_LIST})
```

此内容简化自[cmake+vscode编译多个子目录c++文件的源代码](https://zhuanlan.zhihu.com/p/409339062)，更全面的请点开看该网址。

注意，

* 上述代码中的`include_directories(./)`意思是添加头文件搜索路径，这里只添加了当前CMakeLists所在的根目录，所以代码里引用的头文件地址，必须要从根目录开始引用，比如需要写成`#include "src1/src1.h"`，而不能只写成`#include "src1.h"`，因为其搜索路径只有当前的根目录，除非你加上该地址，比如`include_directories(./;src1/)`，不同头文件搜索路径用空格或分号分开。

  注意，如果包含的目录下还有子目录，那就需要单独再把子目录包含进去，即`include_directories(./;src1/;src1/src1_1/)`，否则子目录下的头文件无法被识别到。

* `file(GLOB_RECURSE root_src_files "${prj_src_dir}/*")`是指在指定目录下递归搜索所有的文件（包括子文件里的文件或子文件夹里的子文件的文件或文件夹，如此递归直到找出所有的文件）。具体解释请看这里：[CMake : 递归的添加所有cpp文件](https://www.cnblogs.com/yongdaimi/p/14689417.html)

另一个简单的入门CMakelists.txt：

```cmake
#设定Cmake的最低版本要求
cmake_minimum_required(VERSION 3.0.0)
#项目名称，可以和文件夹名称不同
project(Hello VERSION 0.1.0)
#命令指定 SOURE_TEST变量（自己定义就行）来表示多个源文件
set(SOURCE_TEST main.cpp math.cpp math.h)
#例如：set(SOURCE_TEST main.cpp test.cpp test1.cpp)
#将生成的可执行文件保存至bin文件夹中
set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/bin)
#生成可执行文件main.exe(可执行文件名 自己定义就行)，用${var_name}获取变量的值。
add_executable(main ${SOURCE_TEST})
```

具体见[Cmake实现VScode中c++多文件编译（记录）](https://blog.csdn.net/qq_52045548/article/details/127091568)

# 支持OpemMP

在

```cmake
add_executable(${PROJECT_NAME} ${PRJ_SRC_LIST})
```

的后面，加上下面的语句。

```cmake
# openMP 配置
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()
```



# 编译子项目作为链接库

假设你的项目结构如下：

```shell
your_project/
├── CMakeLists.txt       # 主项目的 CMakeLists.txt
└── external/
    └── libzmq/          # libzmq 的源代码目录
        └── CMakeLists.txt
```

在主项目的 `CMakeLists.txt` 中，可以通过 `add_subdirectory` 添加 `libzmq`：

```shell
# 主项目的 CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(MyProject)

# 添加 libzmq 子目录
add_subdirectory(external/libzmq)

# 添加你的项目目标
add_executable(my_target main.cpp)

# 链接 libzmq
target_link_libraries(my_target PRIVATE libzmq)
```

总结

- `add_subdirectory` 是将子目录的 CMake 文件包含进来，以便在主项目中使用该子目录中的构建目标。
- 这样可以方便地管理和使用外部依赖或模块。

疑问1：用`add_subdirectory`将子目录的cmake文件包含进来后，应该还需要对它进行编译吧？但是我没看到相关命令啊?

> 使用 `add_subdirectory` 将子目录包含进来后，CMake 会自动**将子目录中的构建目标添加到主项目的构建流程中**，因此**不需要额外的编译命令**。CMake 会在生成构建系统时自动处理子目录中的目标，并在主项目编译时一并编译。
>
> ### 具体流程说明
>
> 1. **将子目录的构建目标包含进主项目**：
>    - `add_subdirectory` 会让 CMake 递归进入 `external/libzmq` 目录，读取该目录中的 `CMakeLists.txt` 文件。
>    - `libzmq` 的 `CMakeLists.txt` 文件中定义了它的构建目标（通常是一个库，如 `libzmq`），这些目标会被添加到主项目的构建系统中。
> 2. **自动编译子目录中的目标**：
>    - 当在主项目中执行 `cmake` 配置和生成构建文件时，CMake 会自动包含子目录的构建逻辑。
>    - 当执行 `make` 或 `cmake --build .` 编译主项目时，主项目和子目录中的所有目标都会被一并编译。
>
> ### 举个例子
>
> 假设你的主项目的 `CMakeLists.txt` 如下：
>
> ```
> cmake
> 
> 
> 复制代码
> # 主项目的 CMakeLists.txt
> 
> cmake_minimum_required(VERSION 3.12)
> project(MyProject)
> 
> # 添加 libzmq 子目录
> add_subdirectory(external/libzmq)
> 
> # 添加你的项目目标
> add_executable(my_target main.cpp)
> 
> # 链接 libzmq 库
> target_link_libraries(my_target PRIVATE libzmq)
> ```
>
> 这里 `add_subdirectory(external/libzmq)` 加入了 `libzmq` 的构建逻辑，因此在执行构建时会自动编译 `libzmq`。
>
> CMake 会自动识别并编译 `libzmq`，无需手动指定编译 `libzmq`。

疑问2：假设add_executable可以完成编译且生成链接库。那我在执行target_link_libraries(my_target PRIVATE libzmq)命令的时候，是怎么知道libzmq的位置在哪里的呢？

> 当执行 `target_link_libraries(my_target PRIVATE libzmq)` 时，CMake 会通过 `add_subdirectory(external/libzmq)` 引入的构建信息来自动找到 `libzmq` 的位置。让我们详细解释一下这个过程。
>
> ### `add_subdirectory` 如何帮助 CMake 知道 `libzmq` 的位置
>
> 1. **引入子目录的构建信息**：
>    - `add_subdirectory(external/libzmq)` 会让 CMake 进入 `external/libzmq` 目录，执行其中的 `CMakeLists.txt`。
>    - `libzmq` 的 `CMakeLists.txt` 通常包含 `add_library(libzmq ...)`，用于定义一个名为 `libzmq` 的库目标。
>    - 一旦定义了这个目标，CMake 就会知道 `libzmq` 的具体构建信息（包括源文件、编译选项等）。
> 2. **生成编译规则**：
>    - 当执行 `cmake` 命令配置项目时，CMake 会记录所有目标（包括 `libzmq`）的生成规则和路径。
>    - CMake 会将 `libzmq` 的位置和编译方式纳入整体构建系统中。
> 3. **链接库时自动识别位置**：
>    - 当使用 `target_link_libraries(my_target PRIVATE libzmq)` 时，CMake 不需要通过绝对路径来找到 `libzmq`，因为它已经在 `add_subdirectory` 阶段处理了 `libzmq` 的位置。
>    - `target_link_libraries` 可以直接找到并链接 `libzmq`。
>
> ### 示例
>
> 假设 `external/libzmq/CMakeLists.txt` 中有如下内容：
>
> ```
> cmake
> 
> 
> 复制代码
> add_library(libzmq STATIC zmq.cpp)  # 定义 libzmq 库
> target_include_directories(libzmq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
> ```
>
> 在主项目的 `CMakeLists.txt` 中调用 `add_subdirectory(external/libzmq)` 后：
>
> - `libzmq` 作为目标自动加入构建系统，`target_link_libraries(my_target PRIVATE libzmq)` 就可以直接链接它。
> - CMake 会在构建项目时自动生成 `libzmq`，并在最终链接阶段正确找到 `libzmq` 的位置。
>
> ### 总结
>
> 通过 `add_subdirectory`，CMake 已经知道了 `libzmq` 的生成规则和路径，因此在 `target_link_libraries` 中直接写目标名称即可，无需指定具体路径。这是 CMake 管理依赖的一种重要机制。

问题3：target_link_libraries(my_target PRIVATE libzmq)中的private是什么意思？可以不加吗？

> 在 `target_link_libraries` 中，`PRIVATE` 是一种访问控制关键字，用来指定**库的链接范围**，即链接库的影响范围。CMake 提供了三种链接范围：`PRIVATE`、`PUBLIC` 和 `INTERFACE`。每种范围的含义和使用情况如下：
>
> ### 1. `PRIVATE`
>
> - **含义**：当库以 `PRIVATE` 方式链接时，`my_target` 可以使用 `libzmq` 提供的功能，但是任何依赖 `my_target` 的其他目标都**无法访问** `libzmq`。
>
> - **效果**：`libzmq` 的包含路径和编译选项仅对 `my_target` 生效，不会传递给依赖 `my_target` 的其他目标。
>
> - **适用场景**：当库仅在当前目标中使用，不需要被进一步传递时。
>
>   ```
>   cmake
>   
>   
>   复制代码
>   target_link_libraries(my_target PRIVATE libzmq)
>   ```
>
> ### 2. `PUBLIC`
>
> - **含义**：`libzmq` 的包含路径和编译选项不仅对 `my_target` 生效，还会传递给任何依赖 `my_target` 的其他目标。
>
> - **效果**：对外公开 `libzmq`，让依赖 `my_target` 的目标也可以访问 `libzmq`。
>
> - **适用场景**：当 `my_target` 是一个库，而 `libzmq` 的功能需要向上层传播时使用。
>
>   ```
>   cmake
>   
>   
>   复制代码
>   target_link_libraries(my_target PUBLIC libzmq)
>   ```
>
> ### 3. `INTERFACE`
>
> - **含义**：`libzmq` 的包含路径和编译选项只会对依赖 `my_target` 的目标生效，而不会直接应用到 `my_target` 本身。
>
> - **效果**：让 `libzmq` 只对依赖的目标可见，而对 `my_target` 自身不可见。
>
> - **适用场景**：当 `my_target` 是一个接口库（例如仅包含头文件而无实际实现），其依赖只需传递给依赖项时使用。
>
>   ```
>   cmake
>   
>   
>   复制代码
>   target_link_libraries(my_target INTERFACE libzmq)
>   ```
>
> ### 是否可以不加
>
> 在 CMake 3.0 及更高版本中，不加 `PRIVATE`、`PUBLIC` 或 `INTERFACE` 会导致 CMake 报错。因为 CMake 需要知道链接范围来正确管理依赖关系。
>
> ### 总结
>
> 选择适当的链接范围取决于项目的设计需求：
>
> - **仅供内部使用**：使用 `PRIVATE`
> - **需要对外部公开**：使用 `PUBLIC`
> - **只为依赖目标提供接口**：使用 `INTERFACE`
>
> 对于普通的可执行文件或测试文件，通常选择 `PRIVATE`。

问题4：虽然`add_subdirectory(external/libzmq)`能自动编译并指导libzmq的位置，但是我自己并不知道啊，怎么才能打印出来libzmq编译好的位置呢

> 你可以使用 `message` 命令在 CMake 配置阶段输出 `libzmq` 编译好的位置。CMake 提供了一些变量来帮助找到目标的编译位置，如 `TARGET_FILE`、`TARGET_LINKER_FILE` 等，可以用它们来获取 `libzmq` 编译后的具体位置。
>
> 以下是几种方法来打印出 `libzmq` 库的路径：
>
> ### 方法 1：使用 `TARGET_FILE` 变量
>
> CMake 提供了 `TARGET_FILE` 变量，可以用来获取某个目标的完整路径。可以在主项目的 `CMakeLists.txt` 中添加如下代码：
>
> ```cmake
> # 确保 add_subdirectory(external/libzmq) 之后执行
> add_subdirectory(external/libzmq)
> 
> # 输出 libzmq 库的生成路径
> message(STATUS "libzmq compiled path: " $<TARGET_FILE:libzmq>)
> ```
>
> - `$<TARGET_FILE:libzmq>` 会返回 `libzmq` 库的完整路径，包括路径、库文件名和后缀。
>
> ### 方法 2：使用 `TARGET_LINKER_FILE` 或 `TARGET_RUNTIME_FILE`
>
> 如果 `libzmq` 是动态库，你可以使用 `TARGET_LINKER_FILE` 或 `TARGET_RUNTIME_FILE` 变量来获取更具体的库路径：
>
> ```cmake
> message(STATUS "libzmq linker path: " $<TARGET_LINKER_FILE:libzmq>)
> message(STATUS "libzmq runtime path: " $<TARGET_RUNTIME_FILE:libzmq>)
> ```
>
> - **TARGET_LINKER_FILE**：返回用于链接的库文件路径（如 `.lib` 或 `.a`）。
> - **TARGET_RUNTIME_FILE**：返回运行时动态库的路径（如 `.dll` 或 `.so` 文件）。
>
> ### 方法 3：输出目标的构建目录
>
> 如果你只想知道 `libzmq` 库的生成目录，可以使用 `CMAKE_CURRENT_BINARY_DIR`：
>
> ```cmake
> add_subdirectory(external/libzmq)
> message(STATUS "libzmq build directory: ${CMAKE_CURRENT_BINARY_DIR}/external/libzmq")
> ```
>
> 这种方法假设 `libzmq` 在主项目的构建目录中的相对路径是 `external/libzmq`。

## 链接已有的动态链接库

怎么样在cmakelist中指定一个已编译好的动态链接库，然后链接。

> 在 CMake 中，要链接一个已经编译好的动态库，可以使用 `find_library` 或直接指定库的路径，并使用 `target_link_libraries` 将其链接到你的目标。这里是如何实现的详细步骤。
>
> ### 假设
>
> 假设已编译好的动态库文件名是 `mylib.so`，它位于 `/path/to/library` 目录。
>
> ### 方法 1：使用 `find_library` 查找动态库
>
> 1. 使用 `find_library` 查找库的路径。
> 2. 使用 `target_link_libraries` 将找到的库链接到你的目标。
>
> ```cmake
> # 指定库文件所在的目录
> set(MYLIB_PATH "/path/to/library")
> 
> # 查找动态库
> find_library(MYLIB NAMES mylib PATHS ${MYLIB_PATH})
> 
> # 检查库是否找到
> if(NOT MYLIB)
>     message(FATAL_ERROR "mylib not found in ${MYLIB_PATH}")
> endif()
> 
> # 添加可执行文件或目标
> add_executable(my_target main.cpp)
> 
> # 链接找到的动态库
> target_link_libraries(my_target PRIVATE ${MYLIB})
> ```
>
> - `find_library` 会在指定路径 `${MYLIB_PATH}` 中查找 `mylib.so`，并将其路径存储在变量 `MYLIB` 中。
> - 然后在 `target_link_libraries` 中使用 `MYLIB` 变量链接库。
>
> ### 方法 2：直接指定动态库的完整路径
>
> 如果库路径是已知的，可以直接将它传递给 `target_link_libraries`。
>
> ```cmake
> # 添加可执行文件或目标
> add_executable(my_target main.cpp)
> 
> # 直接指定库的完整路径进行链接
> target_link_libraries(my_target PRIVATE "/path/to/library/mylib.so")
> ```
>
> ### 方法 3：设置 `CMAKE_PREFIX_PATH` 或 `CMAKE_LIBRARY_PATH`
>
> 如果有多个已编译好的库，可以通过设置 `CMAKE_PREFIX_PATH` 或 `CMAKE_LIBRARY_PATH` 来指定库的查找路径：
>
> ```cmake
> # 设置库的查找路径
> set(CMAKE_LIBRARY_PATH "/path/to/library")
> 
> # 添加可执行文件或目标
> add_executable(my_target main.cpp)
> 
> # 链接库
> target_link_libraries(my_target PRIVATE mylib)
> ```
>
> ### 注意事项
>
> 1. **库名称和路径的兼容性**：`mylib` 的名称应与实际文件名称匹配，例如 `mylib.so`。
>
> 2. **链接类型（PRIVATE/PUBLIC/INTERFACE）**：`PRIVATE` 表示只对当前目标生效。根据需要选择合适的类型。
>
> 3. **include 目录**：如果已编译好的库还需要头文件目录，可以使用 `target_include_directories` 指定头文件路径：
>
>    ```cmake
>    target_include_directories(my_target PRIVATE "/path/to/library/include")
>    ```
>
> 这样配置后，CMake 就可以找到并链接这个动态库了。





# 参考资料

https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html#link-libs

[cmake+vscode编译多个子目录c++文件的源代码](https://zhuanlan.zhihu.com/p/409339062)

https://zhuanlan.zhihu.com/p/406886060

[多个源文件，多个目录](https://www.shuzhiduo.com/A/nAJvK2go5r/)

