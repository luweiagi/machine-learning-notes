# ChatGPT

* [返回上层目录](../openai.md)







===

[张俊林：通向AGI之路：大型语言模型（LLM）技术精要](https://zhuanlan.zhihu.com/p/597586623)

感谢大佬的总结，非常全面。LLM 作为新型的知识库或许是一个新机会，原来手工构建的知识库可能都会被大模型取代；从 EMNLP19 "Language Models as Knowledge Bases" 就已经有这个端倪，最近一些工作也尝试从模型 probe 出世界知识用到具体任务中，譬如 KDD22 "Proton: Probing Schema Linking Information from Pre-trained Language Models for Text-to-SQL Parsing" 也有比较好的提升；所以 LLM 第一个 "颠覆" 的领域可能是 KB [捂脸]

写得太好了！把最近的 LLM 需要关注的进展总结得淋漓尽致！唯一有一点我觉得可能还值得商榷，就是大模型稀疏化的必要性。从 Google 论文的结果来看，MOE 稀疏化的大模型通常会被小得多的 dense model 吊打。过去几年的各种稀疏化研究似乎也表明似乎深度学习模型可能不太适合稀疏化。

[一块RTX3090跑ChatGPT体量模型的方法来了！代码已开源](https://mp.weixin.qq.com/s/8FOojNMnCe1D3q0TOgE_EQ)

[当ChatGPT和Stable diffusion碰撞：谷歌用人类反馈提升文生图效果](https://mp.weixin.qq.com/s/FrqpybryiJ-ikO4ZVeISIg)

[ChatGPT 团队背景（共87人）](https://mp.weixin.qq.com/s/VM2SzNyZF2bZDQ7tYvZFkA)

[李宏毅-ChatGPT 原理剖析1對ChatGPT的常見誤解](https://mp.weixin.qq.com/s/gQ6SCJXmcF86o5TH3mW7Sg)

[打造中国版ChatGPT，这是国内最有实力的一批NLP团队与人才](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650869214&idx=1&sn=e8dab20eab64f3f92d42b2e3fd8e2a2b&chksm=84e4c9a0b39340b6e08200354f72c09d589a99e7763ebaf58bd1d3d0510ee0a954b65a040eff&scene=126&sessionid=1677048170#rd)

[ChatGPT是怎么练成的？斯坦福CS224N课程讲解《自然语言生成》等核心技术，附71页Slides](https://mp.weixin.qq.com/s/wxYdUMBDFc7InBi83OBy-A)

[历史最全ChatGPT、LLM相关书籍、论文、博客、工具、数据集、开源项目等资源整理分享](https://mp.weixin.qq.com/s/VoOEw2-cJ-klGMrl7MJeQQ)

本资源整理了有关 ChatGPT、GPT 和大型语言模型 (LLM)的必读论文、博客、工具、数据集、开源项目等资源，需要自取。

[一文读懂ChatGPT模型原理](https://zhuanlan.zhihu.com/p/589621442)

[为什么百度没有做出chatGPT这种产品？](https://www.zhihu.com/question/572106826/answer/2808336053)

GPT一直是openAI在主攻，其他公司大部分都在做bert方向，GPT1、2出来的时候效果也不好，大家都看不上，但是从GPT3开始，大家才发现，好像真给openAI搞出点什么来，不过也仅仅是一点点，那个时候GPT3生成的东西也不够惊艳。后面加入了prompt，让模型有了很强的理解能力，又引入了强化学习，才造就了如今这个惊艳的产品。

ChatGPT基础模型就是GPT3，标注instruct prompt+RLHF。

国内对大模型的研究也并没有落下国外太多，不过应用落地确实差点，下面列下我知道的大模型吧，也不全

| 模型                | 所属     | 参数量                              |
| ------------------- | -------- | ----------------------------------- |
| ChatGPT/InstructGPT | closeAI  | 175B                                |
| OPT/OPT-IML         | Meta     | 175B/30B/13B                        |
| ERNIE 3/2/1         | 百度     | 260B/100B/10B                       |
| 中文GPT-3           | 阿里     | 27B                                 |
| 悟道2               | 北京智源 | 1750B（神威超算训的，不愧是国家队） |
| GLM                 | 清华     | 130B/10B                            |
| PALM/flan-PALM      | google   | 540B                                |
| Gopher              | deepmind | 280B                                |

