# Git

* [返回上层目录](../coding.md)
* [Git本地化版本管理](#Git本地化版本管理)
  * [第一次初始化](#第一次初始化)
  * [版本管理](#版本管理)
  * [打包输出](#打包输出)

# Git本地化版本管理

在你的项目里，如果之前从来没有过Git，但是你现在想在本地做版本管理，而不想上传到网上，那就可以按照下面的介绍来做。

## 第一次初始化

### 添加.gitignore

先在项目根目录下添加`.gitignore`，内容可以是

```shell
# RL / 项目核心
airsim/
logs/
checkpoints/
runs/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# 模型权重
*.pt
*.pth

# 数据记录
state_json_history*.json

# 打包产物
*.zip
*.tar*

# 测试缓存
.pytest_cache/

# 系统文件
.DS_Store
Thumbs.db
*Zone.Identifier

# IDE（可选）
.vscode/
.cursor/
.idea/
```

### 初始化Git

然后初始化Git

```shell
git init
```

### 添加当前文件

```shell
git add .
```

### 增加注释存储在本地

```shell
git commit -m "xxx"
```

注意，如果是新的提交，Git 还不知道**你是谁**，所以无法提交。每次提交都需要记录一个作者名字和邮箱。

你现在有两种选择：**全局设置**或者**当前仓库单独设置**。

1️⃣ 全局设置（推荐，所有仓库共用）

```shell
git config --global user.name "yourname"
git config --global user.email "your_email@example.com"
```

- `--global` 表示你电脑上所有 Git 仓库都默认用这个名字和邮箱
- 设置完成后，再提交就不会报错了

```
git commit -m "init"
```

2️⃣ 仓库局部设置（只影响当前仓库）

```shell
git config user.name "yourname"
git config user.email "your_email@example.com"
```

- 如果你不想暴露全局信息，或者不同仓库用不同身份，可以用这个
- 设置后，提交同样可以正常工作

✅ 检查是否设置成功

```shell
git config --list
```

你应该能看到类似：

```shell
user.name=yourname
user.email=your_email@example.com
```

🧠 提示

- 邮箱不必真实邮箱，如果只是本地管理，可以写任意字符串：

```shell
git config --global user.email "local@mycomputer"
```

- 一旦设置，提交记录就会带上你设定的名字和邮箱

## 版本管理

### 不同实验开不同的分支

```
main          # 稳定版本
├── exp_reward_v1
├── exp_model_v2
├── exp_xxx
👉 每个实验一条分支
👉 成功了再 merge 回 main
```

## 打包输出

```shell
git archive -o project.zip HEAD
```

具体的：

| 场景                               | archive 命令                                                 |
| ---------------------------------- | ------------------------------------------------------------ |
| 当前在分支上                       | `git archive -o project.zip HEAD` → 打包分支最新 commit      |
| 当前在历史 commit（detached HEAD） | `git archive -o project.zip HEAD` → 打包当前检出的历史 commit |
| 想打包任意 commit，不切换          | `git archive -o project.zip <commit_id>`                     |

> **关键点**：HEAD 总是指当前检出的 commit，不管你是在分支上还是 detached HEAD

该命令介绍：

分解成几个部分：

1. `git archive`

   - Git 内置命令，用来 **把仓库里的某个版本打包成压缩包**
   - 注意：它只打包 **被 Git 跟踪的文件**（受 `.gitignore` 忽略的文件不会打包）
   - 不会把仓库里的 `.git` 文件夹打包进去

2. `-o project.zip`

   - `-o` 表示输出文件
   - `project.zip` 是你生成的压缩包名字
   - 打包结果会直接生成这个 zip 文件

3. `HEAD`

   - Git 的一个引用，指向当前分支的最新 commit

   - 打包的就是你当前最新提交里的源码状态

   - 如果想打包某个历史 commit 或标签，可以写 commit id 或 tag，例如：

     ```
     git archive -o project_v1.zip v1.0
     ```

🔥 核心作用

- 生成干净的源码包

  ，不会包含：

  - `logs/`
  - `checkpoints/`
  - `__pycache__/`
  - `.git/`

- ✅ 非常适合：

  - 离线分享代码
  - 提交给其他团队或实验环境
  - 构建 Docker 镜像前打包源码
