# 大模型通用 API 调用范式详解：以 ChatGPT 为例

* [返回上层目录](../model-deployment-practice.md)
* [什么是ChatGPT-API-key](#什么是ChatGPT-API-key)
* [使用第三方API](#使用第三方API)
* [单次对话Demo](#单次对话Demo)
* [多次对话Demo](#多次对话Demo)
* [老版本的单次对话Demo](#老版本的单次对话Demo)

> **说明**：本文讨论的是一类通用的大模型对话 API 设计模式，ChatGPT 仅作为示例模型，相关原理同样适用于大多数类 OpenAI 风格的大模型服务。

在当前的大模型生态中，无论是 OpenAI 的 ChatGPT，还是各类第三方模型服务，大多数对话型模型 API 在设计上都高度相似：
 都采用 **API Key 鉴权**、**HTTP 请求**、**无状态服务**，并通过在请求中显式传递上下文来实现多轮对话。

本文将以 **ChatGPT API 作为具体示例**，系统讲解这一类**通用大模型对话 API 的调用范式**，包括 API Key 的获取与使用方式、Token 与计费机制、以及单轮与多轮对话在工程上的实现方法。

理解了这套模式后，你可以非常自然地将相同思路迁移到其他大模型服务上，而不局限于某一家平台。

# 什么是ChatGPT-API-key

来自：[ChatGPT API key调用问题和付费版有什么区别？一篇讲清楚](https://zhuanlan.zhihu.com/p/1904231700629816190)

ChatGPT API key调用问题和付费版有什么区别？一篇讲清楚

最近在帮项目接入AI能力的时候，研究了一下ChatGPT的API服务，发现很多人对**API调用和Plus会员**混淆不清。所以这篇就来系统说说，**ChatGPT API Key到底怎么用**、**和普通付费会员Plus有什么区别**，有哪些坑，怎么避雷。

**一、ChatGPT API 和 ChatGPT Plus 是两回事！**

先把最常见的误解纠正：

**ChatGPT Plus**：是你在网页或App中使用ChatGPT聊天时的“升级服务”，主要解锁GPT-4模型（现在是GPT-4 Turbo）。

**API Key 调用**：是开发者通过OpenAI平台获取接口调用权限，把ChatGPT能力集成到自己的应用、网站、插件中。

这两者是**完全独立的产品体系**，互不相通：

| 项目     | ChatGPT Plus          | ChatGPT API                   |
| -------- | --------------------- | ----------------------------- |
| 面向人群 | 普通用户              | 开发者/企业                   |
| 使用方式 | 网页/App聊天界面      | 编程调用接口                  |
| 模型类型 | GPT-3.5 / GPT-4 Turbo | 可选GPT-4、GPT-3.5、Whisper等 |
| 计费方式 | 固定20美元/月         | 按调用token计费               |
| 是否互通 | ❌ 账户不通用          | ❌ 不共享额度                  |

二、API Key 怎么获取？步骤如下：

想调用OpenAI的API，需要以下步骤：

1、注册OpenAI开发者账号：`https://platform.openai.com/`

2、登录后，点击右上角头像 → View API Keys

3、创建新的API Key（注意：**只显示一次**，要保存好）

4、开始使用API（用[Postman](https://zhida.zhihu.com/search?content_id=257542830&content_type=Article&match_order=1&q=Postman&zhida_source=entity)或后端语言进行请求）

5、如果没额度了，充值key可以用：`https://yeka.ai/i/68888`

**提醒：创建API Key不等于开通GPT-4调用权限，默认只有3.5，需要申请或充值才可解锁GPT-4。**

三、ChatGPT API调用常见问题

1）GPT-4 怎么开通？

API默认只开GPT-3.5，需要充值并启用 **“[Pay-as-you-go](https://zhida.zhihu.com/search?content_id=257542830&content_type=Article&match_order=1&q=Pay-as-you-go&zhida_source=entity)”** 模式。操作：

* 打开 Billing 页面，绑定付款方式（Visa/MasterCard）
* 开启用量结算模式（可设置每月上限）
* 充值后系统可能自动开通 GPT-4 API 使用权限（不保证立即）

2）Token 是什么？怎么计费？

OpenAI按“token数”计费，token可以理解为英文的单词/字符片段。

比如你发一句话“Hello, how are you?”，大约是 5 个 tokens。

以下是2024年主流模型API价格参考：

| 模型          | 输入费率（每百万token） | 输出费率 |
| ------------- | ----------------------- | -------- |
| GPT-4 Turbo   | $10                     | $30      |
| GPT-3.5 Turbo | $0.5                    | $1.5     |

3）API调用有限制吗？

* 有，比如：每分钟最大速率（tokens per minute）
* 每天最大消费额度（可以设置预算）
* GPT-4需申请开通，不保证所有账户都有

四、Plus会员和API调用哪个适合你？

这个看你用途：

**个人日常使用、学习、问答** → 用ChatGPT Plus足够了，20刀/月不限次，方便。

**产品开发、批量处理内容、接入业务流程** → 用API才是正解，按需计费，能控制成本。

很多人误以为买了Plus就能用API，其实是两套账。

要做自动化脚本、生成内容、嵌入到网站的，**必须申请API Key。**

**调用案例**

使用 OpenAI 提供的 ChatGPT API，发送一句话让 AI 进行回复（模拟一个最基础的聊天对话功能）。

* 拿到你的 **OpenAI API Key**（注册并生成，保存好）
* 已安装 `openai` Python 库

ChatGPT API 调用示例代码（Python）

```python
import openai

# 替换为你自己的 API Key
openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # 或者使用 "gpt-4"、"gpt-4-turbo"（需开通）
    messages=[
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "能不能帮我写一段 Python 代码，判断一个数是不是质数？"}
    ]
)

# 打印 AI 回复的内容
print(response['choices'][0]['message']['content'])
```

输出效果：

```python
当然可以！以下是判断一个数是否为质数的 Python 代码：

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 示例使用
print(is_prime(7))  # 输出 True
```

# 使用第三方API

> 翻源码论坛发现[方舟 API](https://api.aiyungc.cn/)（[https://api.aiyungc.cn](https://api.aiyungc.cn/)） ，直接 REST，8K 上下文 0.02 元/次，支持流式返回，Python 异步 20 并发，3 分钟跑完 2000 条。
>
> 来自：[ChatGPT 4 太贵？我找到 1 个平替 API，上下文 8K 只要 0.02 元/次](https://zhuanlan.zhihu.com/p/1995101179374482441)

这里测试时使用了`https://api.aiyungc.cn/`作为我的第三方API。登录官网注册，充值2块钱，获取API，比如：`sk-xxxxxxxxxx`，这是一个比较长的字符串，把它复制下来做为自己的API，千万别上传到网上或者GitHub上，这会泄露你的API，也会遭到GitHub封禁。

# 单次对话Demo

先`pip install openai`，然后就可以使用了：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://yunwu.ai/v1",
    api_key='sk-xxxxxxxxxxxx'  # 替换成你的API
)

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，帮我写一段python代码，判断一个整数是不是偶数？"},

  ],
  timeout=100,

)
# print(response)
print(response.choices[0].message.content)
```

运行结果：

````shell
python chat_aiyungc.py

当然可以！判断一个整数是否为偶数的方法之一是检查这个整数除以2的余数是否为0。下面是一个简单的Python函数，可以用来进行这个判断：

```python
def is_even(number):
    """判断一个整数是否为偶数"""
    return number % 2 == 0

# 测试这个函数
number = 4
if is_even(number):
    print(f"{number} 是偶数。")
else:
    print(f"{number} 是奇数。")
```

你可以根据需要将这个代码拷贝到你的Python环境中进行测试。这个函数接受一个整数作为参数，并返回一个布尔值，指示该整数是否为偶数。
````





# 多次对话Demo

> 目前的代码只能进行单次对话，那怎么进行多轮对话呢？一般是怎么做的？

这是一个非常棒的问题！刚接触 AI 开发时，最容易产生的一个误解就是以为 AI 会自动记得你之前说过的话。

**1. 核心原理：AI 是没有“记忆”的**

实际上，OpenAI 的服务器是无状态的（Stateless）。这意味着：

- 你发一个请求过去，它给你一个结果。

- 当你发第二个请求时，它完全不记得你上一个请求问了什么。

**2. 一般是怎么做的？**

要实现多轮对话，其实是靠程序员“喂”给它历史记录。

你每次发请求时，要把之前所有的对话内容（包括你问的和你收到的回复）全部塞进 messages 列表里再发给它。

**3. 代码实现**

我为你新建了一个 `chat_multiturn.py` 文件，你可以直接运行它体验：

```python
# coding=utf-8
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key='sk-xxxxxxxxxxxx'  # 替换成你的API
    base_url="https://yunwu.ai/v1",
)

def chat():
    # 1. 定义一个列表，用来存储所有的对话历史
    # 模型本身没有记忆，每次都要把之前的记录都发给它
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"}
    ]

    print("--- 已进入多轮对话模式（输入 'quit' 退出） ---")

    while True:
        # 获取用户输入
        user_input = input("用户: ")
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            break

        # 2. 将用户的问题添加到历史记录中
        messages.append({"role": "user", "content": user_input})

        try:
            # 3. 将【整个历史记录列表】发送给 AI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                timeout=60
            )

            # 4. 获取 AI 的回复
            ai_reply = response.choices[0].message.content
            print(f"AI: {ai_reply}\n")

            # 5. 重要：将 AI 的回复也添加到历史记录中
            # 这样下次提问时，AI 才知道它刚才说了什么
            messages.append({"role": "assistant", "content": ai_reply})

        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    chat()
```

**关键点解析：**

1. messages 列表：它是你的“对话账本”。

1. role: user：代表你说话的内容。

1. role: assistant：代表 AI 回复的内容。必须存回去，AI 才知道它之前承诺过什么或回答过什么。

1. 循环：每次循环都会让这个“账本”变长，AI 看到的上下文也就越多。

**进阶提示（刚学可以先了解）：**

随着对话次数增多，messages 列表会变得非常长，这会导致：

1. 费用变贵：因为 OpenAI 是按字符数（Token）计费的，每次发送历史记录都要付钱。

1. 超过长度限制：每个模型都有最大字数限制（Context Window）。

所以成熟的应用通常会做“历史记录清理”（比如只保留最近 10 轮对话，或者对旧的对话做摘要）。

**运行结果：**

````shell
python chat_multiturn.py 
--- 已进入多轮对话模式（输入 'quit' 退出） ---
用户: hello
AI: Hello! How can I assist you today?

用户: can you just write a python code which can print hello world?
AI: Certainly! Here's a simple Python code that prints "Hello, World!":

```python
print("Hello, World!")
```

You can run this code in any Python environment to see the output. Let me know if there's anything else you'd like to learn or do with Python!

用户: just a test: how many times I say hello to you? 
AI: So far, you've said "hello" to me twice. Once in your initial message and then in the context of asking for help. If you have more questions or need further assistance, feel free to ask!

用户: But I just say hello to you one time... 
AI: Apologies for the misunderstanding. In your first message, you said "hello" once. If there's anything else you'd like assistance with, feel free to let me know!

用户: quit
````

# 老版本的单次对话Demo

有的代码会涉及到的OpenAI的API的代码还是里2023年年底之前的老版本，为了让你明白老版本的API调用代码怎么写，下面给个demo：

```python
# coding=utf-8
import openai
openai.api_base = 'https://yunwu.ai/v1'
openai.api_key = 'sk-yXjTud24yUOugEG6z075Q8POZiXA2QSi9dthS9RlBCM32HVh'
# openai.proxy = 'http://xxx:port' # 替换为你的代理地址

def use(prompt):
	response = openai.ChatCompletion.create(
		  model="gpt-4o",
		  messages=[
                {"role": "system", "content": "You are a helpful assistant."},
		        {"role": "user", "content": prompt}
		    ]
		)
	return response['choices'][0]['message']['content']

if __name__ == "__main__":
    prompt = "能不能帮我写一段 Python 代码，判断一个整数是不是偶数？"
    result = use(prompt)
    print(result)
```



> @chat_old_version.py  这个代码的内容是不是是openai的老版本？

是的，chat_old_version.py 使用的确实是 OpenAI Python SDK v0.28.1 及更早版本 的写法。

你可以通过以下几个标志性的特征来识别它是老版本：

**1. 核心识别点**

- 写法：openai.ChatCompletion.create(...)
  - ​	老版本 (v0.x)：通过 openai 模块名直接调用静态方法。

- 新版本 (v1.x+)：必须先实例化一个客户端 client = OpenAI()，然后通过 client.chat.completions.create(...) 调用。

- 地址设置：openai.api_base = '...'

  - 老版本：使用 api_base。

  - 新版本：使用 base_url。

- 结果获取：response['choices'][0]['message']['content']

  - 老版本：返回的是一个类似 字典 (Dict) 的对象，必须用方括号 [] 取值。

  - 新版本：返回的是一个 模型对象，支持点号 . 取值（如 response.choices[0].message.content）。

**2. 为什么会有这个版本？**

OpenAI 在 2023 年底对 SDK 进行了大规模重构（升级到 v1.0.0）。这次升级是不向下兼容的，导致很多旧教程和旧项目（比如你提到的斯坦福小镇早期版本）如果没更新，就只能用这种老写法。

**3. 如果你想在自己电脑上运行这个老版本代码**

如果你的环境中安装的是最新的 openai 库（v1.x），运行这个文件会报错。你有两个选择：

- 选择 A（不推荐）：降级安装老版本库：

  `pip install openai==0.28.1`

- 选择 B（推荐）：按照我之前给你的 chat.py 里的写法，把代码重写为新版本语法。新版本更稳定，报错信息也更清晰。

总结：如果你正在看的 GitHub 项目使用的是这种写法，说明那个项目可能有一段时间没维护了，或者它固定了旧版本的依赖。在你自己的新练习中，建议一律使用新版本写法。