# 多智能体：斯坦福AI小镇

* [ 返回上层目录](../llm-based-multi-agent.md)

# 项目介绍

项目介绍：

[机器之心：斯坦福的「虚拟小镇」开源了：25个AI智能体照进《西部世界》](https://baijiahao.baidu.com/s?id=1773826178459815883)

GItHub：[Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents)

# 运行方法

使用方式：

直接看官网GitHub的ReadMe，如果你实在看不懂，也可以参考下中文的说明教程：

[量子位：爆火斯坦福「AI小镇」开源，自己电脑也能跑的西部世界｜GitHub 1.3K星](https://zhuanlan.zhihu.com/p/649183641)



智能体每一步运行都要调用OpenAI的API接口，需要你创建你的账户的API调用：

https://platform.openai.com/api-keys

但是调用API是要钱的，需要你在下面的网址里绑定信用卡：

https://platform.openai.com/settings/organization/billing/overview

注意：在该网址里需要设置金额上限，建议你初步设置3美元。以免长时间调用花了你几千美元。



## 报错提示

### 环境已存在

当你运行到：`python reverie.py`，如果输入仿真名称：`test-simulation`后出现错误：

> FileExistsError: [Errno 17] File exists: '../../environment/frontend_server/storage/test-simulation'

那就就说明你之前已经运行过一次相同名称的环境了，你得把之前的环境删除，即

```shell
(base) user@LAPTOP-xxx:~/generative_agents/environment/frontend_server/storage$ rm -rf test-simulation/
```

## 难题

### 运行后画面不动

第二大坑是AI小镇运行在本机运行起来后，你看不到AI小人在移动。对策是到GitHub上的项目页的Issue栏找到这个链接：

https://github.com/joonspk-research/generative_agents/issues/59

按照要求添加文件夹和添加相应代码，然后就一马平川了。

即：

1、添加文件夹

```python
mkdir ./generative_agents/environment/frontend_server/storage/base_the_ville_isabella_maria_klaus/movement then start the simulation
```

2、更改代码：

add code before File "/Users/zhlei/generative_agents/reverie/backend_server/reverie.py", line 400

```python
          # Example json output: 
          # {"persona": {"Maria Lopez": {"movement": [58, 9]}},
          #  "persona": {"Klaus Mueller": {"movement": [38, 12]}}, 
          #  "meta": {curr_time: <datetime>}}
          
          # 这里是你添加的代码
          curr_move_path = f"{sim_folder}/movement"  # hahaha
          # If the folder doesn't exist, we create it.
          if not os.path.exists(curr_move_path):
            os.makedirs(curr_move_path)
            
          curr_move_file = f"{sim_folder}/movement/{self.step}.json"
          with open(curr_move_file, "w") as outfile: 
            outfile.write(json.dumps(movements, indent=2))
```

### 运行从凌晨开始，而且太慢

修改/home/lw/generative_agents/environment/frontend_server/storage/test-simulation/reverie/meta.json

```
{
  "fork_sim_code": "base_the_ville_isabella_maria_klaus",
  "start_date": "February 13, 2023",
  "curr_time": "February 13, 2023, 07:00:00",
  "sec_per_step": 60,
  "maze_name": "the_ville",
  "persona_names": [
    "Isabella Rodriguez",
    "Maria Lopez",
    "Klaus Mueller"
  ],
  "step": 0
}
```

我已经帮你完成了“时空跳跃”配置：

1. 当前时间：改为 07:00:00 AM。

1. 推进速度：改为 60 秒/步（原来是 10 秒/步），这会显著提升模拟速度。

我已经把修改后的 meta.json 覆盖到了 base_the_ville_isabella_maria_klaus 环境中。现在这个原始模板的起始时间已经固定在 早上 07:00:00，且步长设为 60 秒。

现在你可以开始全新的模拟了：

1. 运行 python reverie.py。

1. Forked simulation: 输入 base_the_ville_isabella_maria_klaus。

1. New simulation: 输入任意你喜欢的新名字（比如 fresh-start）。

1. 运行模拟：输入 run 100。

这样，你刷新浏览器后，画面会直接从早上 7 点开始，再也不会有从凌晨 0 点开始的长长时间回放了。一切都是全新的开始的回放了。

### 如何恢复上次的环境

如果你想接着跑上次那个名为 test-simulation 的环境，操作方法如下：

1. 启动程序：python reverie.py

1. 第一步（Forked simulation）：输入你上次跑的那个名字（例如 test-simulation）。

1. 第二步（New simulation）：输入一个全新的名字（例如 test-simulation-v2）。

注意：这个项目的设计理念是“只增不减”。它不支持在原文件夹上直接继续写（为了防止弄坏之前的记录），而是每次“继续”都相当于在之前进度的基础上“分身”出一个新版本。

### 思考阈值更改

> 现在的输出怎么成了这样子了？
>
> Klaus Mueller persona.scratch.importance_trigger_curr:: 142
> 150
> Isabella Rodriguez persona.scratch.importance_trigger_curr:: 141
> 150
> Maria Lopez persona.scratch.importance_trigger_curr:: 141
> 150
> Klaus Mueller persona.scratch.importance_trigger_curr:: 142
> 150
> Isabella Rodriguez persona.scratch.importance_trigger_curr:: 141
> 150
> Maria Lopez persona.scratch.importance_trigger_curr:: 141
> 150
> Klaus Mueller persona.scratch.importance_trigger_curr:: 142
> 150

你看到的这些重复输出是程序的 “记忆整理”机制 触发了。

发生了什么？

在该项目中，每个人物都有一个“重要性触发器”（importance_trigger_curr）。

1. 累积重要性：每当人物感知到一个事件（比如看到别人在睡觉，或者自己换了一个动作），系统都会给这个事件打分（1-10分）。

1. 触发阈值：当累积的分数达到 150 时，人物就会停下来进行一次“反思”（Reflect），将最近的琐碎记忆总结成更深刻的见解。

1. 当前的死循环感：你看到的 141、142 正在接近 150。由于你把时间步长改成了 60 秒/步，每一分钟人物都会感知到很多环境变化，导致这个分数累积得非常快。

为什么输出这么多？

这是因为代码中为了调试方便，打印了这些中间数值。虽然看起来像是在刷屏或“卡住了”，但实际上它是在疯狂计算：

- 它在为每一个微小的动作（比如“还在床上躺着”）向 LLM 询问：“这件事对 Isabella 有多重要？”

- 由于你有 3 个人物，每分钟（每一步）都要进行多次这样的询问。

建议操作

1. 保持运行（如果你不赶时间）：

这并不是报错，而是程序在执行正常的逻辑。等它跨过 150 这个门槛完成“反思”后，输出就会恢复正常。

2. 调低“灵敏度”（如果你觉得太烦）：

如果你觉得人物反思得太频繁（毕竟现在 1 步顶过去 6 步），可以修改 persona/persona.py 里的触发阈值（从 150 改到 500），或者干脆关掉这个调试打印。

3. 暂时忍受：

因为 gpt-4o-mini 很快，这几百行输出其实在几秒钟内就能跑完。

总结：

这代表 Agent 们正在认真思考！它们正在评估每一分钟发生的点点滴滴。只要没有 ERROR 字样，请让它继续跑，画面很快就会同步。

> 如果你觉得人物反思得太频繁（毕竟现在 1 步顶过去 6 步），可以修改 persona/persona.py 里的触发阈值（从 150 改到 500），或者干脆关掉这个调试打印。

这个值在哪里呢？

我已经按照你的要求完成了修改：

1. 修改触发阈值

文件：reverie/backend_server/persona/memory_structures/scratch.py

将 importance_trigger_max 从 150 改为了 500。

- 效果：人物进行“反思”的频率会降低到原来的 1/3 左右。这样在每步 60 秒的高速模式下，人物不会因为一点小事就停下来思考半天。

2. 关闭调试打印

文件：reverie/backend_server/persona/cognitive_modules/reflect.py

注释掉了那行频繁出现的 print (persona.scratch.name, "persona.scratch.importance_trigger_curr::", ...)。

- 效果：终端输出会变得非常干净，你只会看到关键的行动描述和任务分解，不再会被那一串串数字刷屏。

---

> 对重复性的事件（比如一直“在睡觉”）直接给一个默认的低分，不再询问 GPT
>
> 
>
> 这个怎么做？

我已经为 generate_poig_score 函数添加了逻辑，对于以下常见的重复性、低重要性事件，系统将直接赋予 1 分（最低分），而不再请求 GPT 进行评分：

- is idle（闲置中）

- is sleeping（正在睡觉）

- is occupied（被占用）

- is being used（正在被使用）

- is vacant（空闲的）

- is empty（空的）

修改涉及的文件：

1. reverie/backend_server/persona/cognitive_modules/perceive.py (感知模块)

1. reverie/backend_server/persona/cognitive_modules/reflect.py (反思模块)

1. reverie/backend_server/persona/cognitive_modules/converse.py (对话模块)

这样做的好处：

- 速度极大提升：系统不再需要为“某某正在睡觉”这种毫无意义的重复事件去等待 GPT 的网络响应。

- 节省 Token：减少了大量冗余的 API 调用。

- 防止卡顿：由于你设置了每步 60 秒，周围可能会产生大量这类环境事件。现在的过滤逻辑可以确保系统只关注那些真正有意义的新变化。

现在你可以重启后端尝试运行了，速度应该会变得非常顺滑！

# 更换API接口

## 能把内置的gpt-3.5-turbo换成其他gpt模型吗

> 我想运行github上的斯坦福小镇的ai agent的代码，它里面就是用的gpt-3.5-turbo，我换成gpt-4o可以吗？

完全可以，而且强烈建议你换成 gpt-4o 或者 gpt-4o-mini。

对于像“斯坦福小镇（Generative Agents）”这样复杂的 AI Agent 项目，直接更换模型不仅能运行，通常还会效果更好。以下是几点关键建议：

1. 为什么 gpt-4o-mini 是最佳选择？

斯坦福小镇项目包含大量的智能体对话和思考，API 调用频率非常高。

- 兼容性：gpt-4o-mini 和 gpt-4o 的接口与 gpt-3.5-turbo 完全一致。

- 价格：gpt-4o-mini 比 gpt-3.5-turbo 还要便宜得多（通常只有 3.5 价格的 1/3 甚至更低）。

- 智力：gpt-4o-mini 比 gpt-3.5-turbo 聪明很多，智能体表现会更像“人”，减少胡言乱语。

2. 如果换成 gpt-4o 会怎样？

- 优点：智能体会变得极其聪明，逻辑极强。

- 缺点：太贵了。斯坦福小镇运行一小时可能会产生数千次 API 调用，用 gpt-4o 可能会迅速消耗掉你的余额。

3. 如何在斯坦福小镇代码中修改？

由于该项目代码量较大，建议你使用 全局搜索替换。

1. 在 VS Code 中按下 Ctrl + Shift + F (Windows) 或 Cmd + Shift + F (Mac)。

1. 搜索字符串：gpt-3.5-turbo

1. 替换为：gpt-4o-mini

1. 点击“全部替换”。

4. 注意事项：API 基础路径 (Base URL)

斯坦福小镇的代码通常默认连接 OpenAI 官方服务器。既然你使用的是 yunwu.ai（云舞）的服务，你还需要找到它设置 openai.api_base 或 base_url 的地方，将其改为：

https://yunwu.ai/v1

否则，即使改了模型名字，代码还是会因为连不上官方服务器而报错。

总结：

直接放心大胆地把 gpt-3.5-turbo 替换成 gpt-4o-mini。这不仅能解决你现在的 503 报错，还能让你跑“小镇”跑得更久、更省钱、更智能。

## 能把内置的gpt-3.5-turbo换成其他厂家的大模型吗

> 明白了，那还能改成其他大模型吗？比如deepseek之类的？

当然可以！这正是目前 AI 开发最方便的地方：绝大多数国产大模型（如 DeepSeek、通义千问、文心一言等）都支持“OpenAI 兼容接口”。

这意味着你不需要重写代码，只需要像修改 chat_aiyungc.py 那样，改一下 base_url、api_key 和 model 名字就能直接无缝切换。

1. 以 DeepSeek 为例怎么改？

如果你直接去 DeepSeek 官网（platform.deepseek.com）申请了 Key，代码只需要这样改：

```python
from openai import OpenAI

client = OpenAI(
    # 1. 换成 DeepSeek 的官方 API 地址
    base_url="https://api.deepseek.com", 
    # 2. 换成你申请的 DeepSeek Key
    api_key="sk-xxxxxx", 
)

response = client.chat.completions.create(
    # 3. 换成 DeepSeek 的模型名称
    # deepseek-chat (V3版本，速度快，全能)
    # deepseek-reasoner (R1版本，带深度思考，适合复杂逻辑)
    model="deepseek-chat", 
    messages=[
        {"role": "user", "content": "你好"}
    ]
)
```

**2. 为什么大家现在爱用 DeepSeek 跑 Agent？**

1. 极度便宜：DeepSeek 的价格大约只有 OpenAI 的 1%~5%。跑“斯坦福小镇”这种高频调用的项目，DeepSeek 能让你省下巨额费用。

1. 中文支持更好：对于中文语境的理解，DeepSeek 往往比 GPT 更地道。

1. 推理能力强：特别是最新的 DeepSeek-R1（推理模型），在处理复杂逻辑（比如 Agent 规划下一步要做什么）时，表现不亚于 GPT-4o。

**3. 在“斯坦福小镇”这类项目中使用 DeepSeek 的注意事项**

虽然接口兼容，但有两点需要注意：

- 模型指令遵循度：斯坦福小镇的代码里有很多复杂的 Prompt（提示词）。有些小模型可能理解不了那么复杂的指令。但 DeepSeek V3 或 R1 这种量级的模型通常没问题。

- JSON 格式限制：Agent 代码经常要求 AI 返回 JSON 格式。如果换了模型后报错，通常是因为新模型返回的格式不够严谨，可能需要微调 Prompt。

**4. 总结：如何选择？**

- 追求稳定/最强智力：用 gpt-4o。

- 追求性价比（跑 Agent 首选）：用 deepseek-chat 或 gpt-4o-mini。

- 需要解决高难度逻辑难题：用 deepseek-reasoner (R1)。

操作建议：你可以先在你的 chat.py 里试着把 base_url 改成 DeepSeek 的试一下，如果跑通了，再去大规模修改那个 Agent 项目的代码。

## 老版本的OpenAI-API的代码demo

斯坦福小镇代码里涉及到的OpenAI的API的代码还是里2023年年底之前的老版本，为了让你明白老版本的API调用代码怎么写，下面给个demo：

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

## 更换API接口实际操作

### 使用第三方API

> 翻源码论坛发现[方舟 API](https://api.aiyungc.cn/)（[https://api.aiyungc.cn](https://api.aiyungc.cn/)） ，直接 REST，8K 上下文 0.02 元/次，支持流式返回，Python 异步 20 并发，3 分钟跑完 2000 条。
>
> 来自：[ChatGPT 4 太贵？我找到 1 个平替 API，上下文 8K 只要 0.02 元/次](https://zhuanlan.zhihu.com/p/1995101179374482441)

这里测试时使用了`https://api.aiyungc.cn/`作为我的第三方API。登录官网注册，充值2块钱，获取API，比如：`sk-xxxxxxxxxx`，这是一个比较长的字符串，把它复制下来做为自己的API，千万别上传到网上或者GitHub上，这会泄露你的API，也会遭到GitHub封禁。

修改后端服务器的`utils.py`文件夹

```python
# Copy and paste your OpenAI API Key
openai_api_key = "sk-xxxxx"  # 设置为你的API
# 添加 AI API Base URL
openai_api_base = 'https://yunwu.ai/v1'
# 设定新的 OpenAI 模型，代替默认的 gpt-3.5-turbo
openai_udf_model = 'gpt-4o-mini'

# 剩余的代码不变：
# Put your name
...
```

### 添加api_base

全局搜索`openai_api_key`，找到openai接口的调用处。

 一共发现两处，在所有的`openai.api_key = openai_api_key`后面都加上`openai.api_base = openai_api_base`，即

```python
openai.api_key = openai_api_key
openai.api_base = openai_api_base  # hahaha
```

然后搜`openai.ChatCompletion.create`，更换你自己定义的模型。

一共发现四处，其中有三处是用的`gpt-3.5-turbo`模型，一处用的是`gpt-4`模型，把用`gpt-3.5-turbo`模型的同意更换为你自定义的模型，即

```python
    completion = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo", 
    model=openai_udf_model,  # hahaha
    messages=[{"role": "user", "content": prompt}]
    )
```

 然后按官方说明文档运行即可。

# 第三方汉化版

生成式智能体（Generative Agents）深度汉化版

github：https://github.com/x-glacier/GenerativeAgentsCN

视频教程：[斯坦福大学AI agent小镇，本地零成本部署上车](https://www.bilibili.com/video/BV1CyK8zxEsn/)

看GitHub的ReadMe来操作。

注意：

在`generative_agents/data/config.json`中，把

```json
            "llm": {
                "provider": "ollama",
                "model": "qwen3:4b-instruct-2507-q4_K_M",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": ""
            },

            "embedding": {
                "provider": "ollama",
                "model": "qwen3-embedding:0.6b-q8_0",
                "base_url": "http://127.0.0.1:11434",
                "api_key": ""
            },
```

替换为

```
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://yunwu.ai/v1",
                "api_key": "sk-yXjTud24yUOugEG6z075Q8POZiXA2QSi9dthS9RlBCM32HVh"
            },
            
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "base_url": "https://yunwu.ai/v1",
                "api_key": "sk-yXjTud24yUOugEG6z075Q8POZiXA2QSi9dthS9RlBCM32HVh"
            },
```

重新运行的话，要不要删除上次运行创建的环境文件？

建议删除上次运行产生的存档文件夹，或者换一个新的任务名称。

原因如下：

1. 存档冲突：在你运行 python start.py --name sim-test ... 时，程序会在 results/checkpoints/sim-test 目录下创建存档。如果该目录已存在，程序会因为保护机制提示你“名字已存在”并要求你输入一个新名字，除非你使用了 --resume 参数。

1. 数据完整性：由于上次运行在第一步生成日程时就报错停止了，存档文件夹里可能只有一些初始化的碎片文件（如 storage 目录下的索引），并没有完成任何实际的模拟步骤。为了确保这次修改后的代码能从干净的状态（Clean State）重新初始化智能体的记忆和日程，删除旧文件夹是更好的选择。

### 操作建议：

如果你想继续使用 sim-test 这个名字：

```python
# 删除旧的存档文件夹
rm -rf results/checkpoints/sim-test
# 重新运行
python start.py --name sim-test --start "20250213-09:30" --step 10 --stride 10
```

如果你不介意换个名字，可以直接运行并换个名字：

```python
python start.py --name sim-test-v2 --start "20250213-09:30" --step 10 --stride 10
```

总结：不需要删除你修改过的 config.json（那是你的配置文件），只需要删除 results/checkpoints/ 下对应的那个存档文件夹即可。

但是如果你想接着上一次的保存结果继续运行，那就不要删， 按照GitHub说明里说的那样，加上`--resume`重新运行。

运行结果

```shell
  thought:
    <0>这是 亚当 在 2025年02月13日（星期四）09:30 的计划：凌晨4点醒来，记录一天的思考和灵感；早上5点阅读哲学相关书籍，探索不同观点；早上6点起床，简单整理并准备当天计划；早上7点吃早餐，同时思考创造力与现实世界的联系；上午9点开始撰写关于创造力重要性的书稿；中午12点吃午饭，期间回顾当天写作进展；下午1点小睡一会儿，恢复精力；下午3点继续写作，重点分析创造力如何影响社会变革；下午5点吃晚饭，饭后思考市长选举的潜在影响与公众参与；晚上8点上床睡觉，准备进入深度反思状态
schedule:
  00:00~01:00: 睡觉
  01:00~02:00: 睡觉
  02:00~03:00: 睡觉
  03:00~04:00: 睡觉
  04:00~05:00: 凌晨醒来，记录一天的思考与灵感，重点整理关于创造力如何影响社会结构的初步观察
  05:00~06:00: 阅读哲学经典著作《人类理解论》或《理想国》片段，探索‘创造力是否源于人类对现实的重新定义’这一问题
  06:00~07:00: 起床，简单整理房间，回顾昨日写作内容，制定今日书稿的结构框架与核心论点
  07:00~08:00: 吃早餐，边吃边思考‘创造力与现实世界之间的动态关系’，尝试用哲学视角提出新假设
  08:00~09:00: 开始撰写关于创造力重要性的书稿，聚焦‘创造力如何打破传统认知边界’这一章节
  09:00~10:00: 继续撰写书稿，分析创造力在艺术、科技与社会制度中的具体体现，结合历史案例进行论证:
    09:00~09:15: 收集艺术与科技中创造力的历史案例
    09:15~09:25: 分析创造力在艺术领域的具体体现
    09:25~09:35: 分析创造力在科技领域的具体体现
    09:35~09:50: 分析创造力在社会制度中的具体体现
    09:50~09:55: 查阅相关哲学文献以补充论证
    09:55~10:00: 评估现有论点的偏见风险
    10:00~10:10: 思考是否需要引入其他哲学流派观点
  10:00~11:00: 暂停写作，进行一次深度反思：思考当前观点是否被偏见影响，是否需要引入其他哲学流派的观点进行平衡:
    10:00~10:30: 进行深度反思：审视当前观点是否受偏见影响
    10:30~10:45: 回顾过往相关文献，寻找反例或不同视角的支撑
    10:45~10:55: 记录反思过程中的关键想法
    10:55~11:00: 考虑引入其他哲学流派（如实用主义、存在主义）的观点
  11:00~12:00: 阅读关于社会变革与创新思维的学术论文，挑战‘创造力仅属于少数天才’的先入之见:
    11:00~11:15: 审查已有学术论文的结构与论点
    11:15~11:25: 对比不同哲学流派对创造力的定义
    11:25~11:35: 记录潜在反例或反驳观点
    11:35~11:45: 整理关键概念与理论框架
    11:45~11:55: 提出对‘创造力属于少数天才’这一观点的质疑
    11:55~12:00: 标注需要进一步研究的议题
  12:00~13:00: 吃午饭，期间回顾当天写作进展，评估逻辑连贯性与论点深度，调整结构以增强说服力
  13:00~14:00: 小睡一会儿，恢复思维清晰度，为下午的深度思考与写作做准备
  14:00~15:00: 小睡一会儿，保持思维流动，避免过度疲劳影响批判性思维能力
  15:00~16:00: 继续写作，重点分析创造力如何推动社会变革，如环保运动、科技革命等实际案例
  16:00~17:00: 暂停写作，进行一次‘观点挑战’：主动设想一个反例——如果创造力被系统性压制，世界会如何演变？
  17:00~18:00: 继续写作，将反例分析纳入书稿，形成对创造力与社会自由之间关系的辩证论述
  18:00~19:00: 吃晚饭，饭后思考市长选举的潜在影响，特别是候选人对公共空间、社区创造力的政策设想
  19:00~20:00: 记录市长选举中可能体现的‘公共参与’与‘创意治理’议题，形成初步观察笔记
  20:00~21:00: 观看一段关于城市公共艺术与社区参与的纪录片，反思创造力在基层治理中的角色
  21:00~22:00: 进行一次深度反思：将创造力与市长选举的联系纳入书稿的宏观框架，思考‘谁在定义创造力？谁在使用它？’
  22:00~23:00: 整理当日所有笔记与思考，形成一份结构清晰的思维导图，为明日写作做准备
  23:00~00:00: 睡觉
llm:
  model: qwen3:4b-instruct-2507-q4_K_M
  summary:
    total: S:50,F:0/R:50
    llm_normal: S:50,F:0/R:50
```

运行结果

![game-cn](pic/game-cn.png)

#  参考资料



## 大模型

* [项目介绍、怎么办信用卡充值](https://chatgpt.com/c/69702d33-3f28-8324-aa88-10c691ca6a82)

===========

* [斯坦福小镇开源项目探索](https://blog.csdn.net/m0_61678439/article/details/145032693)

介绍了怎么替换为其他模型的API。