# ColossalAI

* [返回上层目录](../hpcaitech.md)

相信开源社区的强大力量终将实现小规模chatgpt的自搭建。目前来看，1.6g显存对应着1.2亿参数，那么在消费级gpu上，有希望实现10亿参数左右的chatgpt（仅限于推理）。

colossal-ai-chatgpt官网：[hpc-ai.tech/blog/colossal-ai-chatgpt](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt)

github: [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI)



===

[开源方案复现ChatGPT流程！1.62GB显存即可体验，单机训练提速7.73倍](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650868445&idx=1&sn=f8b7db5f226533bc04ca3b729efbf6b8&chksm=84e4cea3b39347b5d443260dfcbf313a28dd5bcae0ff889a03c3c3a49c991e92e9b08195a4fa&scene=21#wechat_redirect)

[ChatGPT低成本复现流程开源！单张消费级显卡、1.62GB显存可体验](https://zhuanlan.zhihu.com/p/606582332)



# ColossalChat

作为当下最受欢迎的开源 AI 大模型解决方案，Colossal-AI 率先建立了包含**监督数据集收集->监督微调->奖励模型训练->强化学习微调的完整RLHF****流程，以LLaMA为基础预训练模型，推出ColossalChat，**是目前最接近ChatGPT原始技术方案的实用开源项目**!

ColossalChat 开源了基于LLaMA模型，复现训练ChatGPT 三个阶段的完整代码。

github地址：https://github.com/hpcaitech/ColossalAI

包含以下内容

1. **Demo**：可直接在线体验模型效果，无需注册或 waitinglist
2. **训练代码**：开源完整 RLHF 训练代码，已开源至含 7B 和 13B 两种模型
3. **数据集**：开源 104K 中、英双语数据集
4. **推理部署**：4bit 量化推理 70 亿参数模型仅需 4GB 显存
5. **模型权重**：仅需单台服务器少量算力即可快速复现
6. 更大规模模型、数据集、其他优化等将保持高速迭代添加

# 参考资料

- [有代码的话本地搭建一个 ChatGPT 可行吗？](https://www.zhihu.com/question/583485737/answer/2958341615)