# 自己运行大语言模型

* [返回上层目录](../multimodal-large-model.md)





[M4 Mac mini作为轻量级AI服务器，太香了！](https://mp.weixin.qq.com/s/XUovXDjfO2fhcgTCrg1Q9w)

拿到Mac mini之后，我安装的第一个软件是Ollama，然后下载Qwen 2.5。因为我一直想实现这样一个场景：

一台足够给力、又足够冷静的机子作为轻量级AI服务器，跑本地大模型，然后供给给局域网内的所有设备使用，比如手机。

第二个是关于网络的设置。这个是我问Cursor学来的。

在初始状态下，Ollama只监听Localhost。要让局域网内的其他设备，比如手机也能访问Ollama，需要修改它的监听地址。

在终端里输入这一行命令：OLLAMA_HOST="0.0.0.0:11434" ollama serve

0.0.0.0指的是让Ollama监听所有网络接口。不管活儿从哪来，都接。11434是它默认的端口，没必要改动。这么改动之后，手机、Pad这些设备都可以通过局域网IP地址接入Ollama。

那么，最后一个问题来了：在移动端用什么APP去连接Ollama？

在桌面端有太多选择了，比如经典的Open WebUI，还有Obsidian的一堆AI插件都支持。在iPhone上，我个人的选择是Enchanted