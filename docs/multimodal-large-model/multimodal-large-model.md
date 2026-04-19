# LLM大语言模型

* [返回上层目录](../natural-language-processing.md)
* [Scaling Law](scaling-law/scaling-law.md)
* [RLHF基于人工反馈的强化学习方法](rlhf/rlhf.md)
* [LoRA大语言模型的低秩适应](lora/lora.md)
* [Prompt Learning](prompt-learning/prompt-learning.md)
* [Emergence涌现现象](emergence/emergence.md)
* [自己运行大语言模型](run-llm-diy/run-llm-diy.md)
* [自己训练大语言模型](train-llm-diy/train-llm-diy.md)
* [LLM based Multi Agent](llm-based-multi-agent/llm-based-multi-agent.md)
* [业界应用](industry-application/industry-application.md)







但是核心推动是RL后训练，sft是让模型学习特定能力来做到场景适配，偏向于特定模式学习缺少泛化性，而RL可以基于奖励函数（现在主要是可验证的强化学习[RLVR](https://zhida.zhihu.com/search?content_id=753406135&content_type=Answer&match_order=1&q=RLVR&zhida_source=entity)）来进行探索式的学习，能稳定推理路径并保持一定的泛化性(是否能学习探索出超越基础模型的推理能力暂时存疑，大部分研究认为RL基于基模型的推理空间探索到稳固的推理路径[2]，但是可以结合使用，比如long cot可以先进行sft冷启动打下基础，让模型具备生成long cot这种格式数据的能力，接着再上RL进行后训练。从最近的各项基于RL的agent模型训练，

作者：sunnyzhao
链接：https://www.zhihu.com/question/13476251758/answer/1965203644279423833
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
