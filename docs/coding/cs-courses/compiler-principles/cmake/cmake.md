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

include_directories(./)

add_executable(${PROJECT_NAME} ${PRJ_SRC_LIST})
```

此内容简化自[cmake+vscode编译多个子目录c++文件的源代码](https://zhuanlan.zhihu.com/p/409339062)，更全面的请点开看该网址。

注意，

* 上述代码中的`include_directories(./)`意思是添加头文件搜索路径，这里只添加了当前CMakeLists所在的根目录，所以代码里引用的头文件地址，必须要从根目录开始引用，比如需要写成`#include "src1/src1.h"`，而不能只写成`#include "src1.h"`，因为其搜索路径只有当前的根目录，除非你加上该地址，比如`include_directories(./;src1/)`，不同头文件搜索路径用空格或分号分开。
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

# 参考资料

https://www.jetbrains.com/help/clion/quick-cmake-tutorial.html#link-libs

[cmake+vscode编译多个子目录c++文件的源代码](https://zhuanlan.zhihu.com/p/409339062)

https://zhuanlan.zhihu.com/p/406886060

[多个源文件，多个目录](https://www.shuzhiduo.com/A/nAJvK2go5r/)

