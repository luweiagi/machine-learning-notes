# 编码规范

* [返回上层目录](../programing-language.md)
* [clang-format](#clang-format)

# clang-format

clang-format 是 LLVM 项目里的代码自动格式化工具，主要面向 C / C++ / Objective-C 等。你写好风格规则后，它按规则改缩进、空格、换行、括号位置等，避免大家手调排版。

你当前工程里还没有 `.clang-format`，本机也还没装 `clang-format`。

怎么用（最短路径）

## 在ubuntu上用

1、安装

在 WSL/Ubuntu 上：

```shell
sudo apt update

sudo apt install clang-format
```

装好后确认：

```shell
clang-format --version
```

2、在项目根目录放配置文件

在仓库根目录创建 `.clang-format`，例如用 Google 风格起步：

```shell
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 120
AlignConsecutiveMacros: false
```

常用基础风格：`Google`、`LLVM`、`Mozilla`、`WebKit`、`Microsoft`。

也可以一键生成再改：

```shell
clang-format -style=google -dump-config > .clang-format
```

3、格式化文件

格式化单个文件（直接改文件）：

clang-format -i BSP/bsp_baro.c

只看会改成什么样（不写回文件）：

clang-format BSP/bsp_baro.c

批量格式化：

```
find BSP -name "*.c" -o -name "*.h" | xargs clang-format -i
```

和你前面讨论的关系

配置里这两项直接对应之前说的风格：

| 选项                            | 含义                           |
| :------------------------------ | :----------------------------- |
| `AlignConsecutiveMacros: false` | 宏定义不对齐，中间只留一个空格 |
| `BasedOnStyle: Google`          | 采用 Google 风格作为基底       |

注意

- 第一次对老工程跑全量格式化，会改很多文件，Diff 会很大。建议：新建模块先跟规范走，或按目录逐步迁移，不要一次改完整个飞控工程。
- 嵌入式代码里若有特殊宏/汇编内联，偶发格式结果不满意，可在那段外包一层：

```c
// clang-format off
... 不格式化的代码 ...
// clang-format on
```

## 在Cursor里用（更方便）

1、安装扩展：Clang-Format（或带 Format Document 的 C/C++ 扩展）

2、先装可执行文件（WSL 里）

插件只是“遥控器”，真正改代码的是系统里的 `clang-format`。你这边目前还没装：

```shell
sudo apt update

sudo apt install clang-format

clang-format --version
```

装好后再回 Cursor 用。

3、在工程根目录放风格文件

在 `fc_code/` 下新建 `.clang-format`，例如：

```shell
BasedOnStyle: Google
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 120
AlignConsecutiveMacros: false
```

没有这个文件时，插件会用默认风格，结果可能和你想的不一致。

4、配置 Cursor（推荐写到工作区）

打开：`Ctrl + Shift + P` → 输入 Preferences: Open Workspace Settings (JSON)
（会打开你工程里的 `.vscode/settings.json`）

在现有内容里追加这些项：

```shell
"[c]": {

  "editor.defaultFormatter": "xaver.clang-format",

  "editor.formatOnSave": false

},

"[cpp]": {

  "editor.defaultFormatter": "xaver.clang-format",

  "editor.formatOnSave": false

},

"C_Cpp.formatting": "disabled",

"clang-format.style": "file"
```

说明：

| 配置                         | 作用                               |
| :--------------------------- | :--------------------------------- |
| `editor.defaultFormatter`    | 格式化时用你装的那个插件           |
| `C_Cpp.formatting: disabled` | 关掉微软 C/C++ 插件抢格式化        |
| `clang-format.style: file`   | 按项目根目录的 `.clang-format` 来  |
| `formatOnSave: false`        | 先手动试，避免老代码一保存全被改乱 |

你项目里还有 clangd。只要默认格式化器设成 `xaver.clang-format`，一般就不会被 clangd 抢走。

4、怎么用

打开例如 `BSP/bsp_baro.c`，然后任选一种：

1. 快捷键：`Shift + Alt + F`（Format Document）
2. 命令面板：`Ctrl + Shift + P` → `Format Document`
3. 右键：编辑器空白处 → Format Document

第一次可能会弹窗问用哪个格式化器，选 Clang-Format。

想先看效果、不立刻写回文件，可在终端：

clang-format BSP/bsp_baro.c | less

确认无了再：

clang-format -i BSP/bsp_baro.c

5、常见问题

* 提示找不到 clang-format

说明 WSL 里没装，或插件找不到路径。可在设置里加：

```
"clang-format.executable": "clang-format"
```

若 `which clang-format` 有路径，就写成完整路径。

* 格式化没反应 / 报错

确认当前语言模式是 C（你工程里 `.h` 已关联成 C），且默认格式化器是 `xaver.clang-format`。

* 老工程不建议一上来就开 Format On Save

会一次改掉大量历史文件。新文件/新模块先用；整仓格式化单独做一次再提交。

