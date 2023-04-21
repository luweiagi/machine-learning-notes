# ColossalChat

* [返回上层目录](../rlhf.md)

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

* [有代码的话本地搭建一个 ChatGPT 可行吗？](https://www.zhihu.com/question/583485737/answer/2958341615)