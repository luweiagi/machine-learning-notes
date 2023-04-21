# ChatGPT

* [返回上层目录](../openai.md)







===

[张俊林：通向AGI之路：大型语言模型（LLM）技术精要](https://zhuanlan.zhihu.com/p/597586623)

> 感谢大佬的总结，非常全面。LLM 作为新型的知识库或许是一个新机会，原来手工构建的知识库可能都会被大模型取代；从 EMNLP19 "Language Models as Knowledge Bases" 就已经有这个端倪，最近一些工作也尝试从模型 probe 出世界知识用到具体任务中，譬如 KDD22 "Proton: Probing Schema Linking Information from Pre-trained Language Models for Text-to-SQL Parsing" 也有比较好的提升；所以 LLM 第一个 "颠覆" 的领域可能是 KB [捂脸]
>
> 写得太好了！把最近的 LLM 需要关注的进展总结得淋漓尽致！唯一有一点我觉得可能还值得商榷，就是大模型稀疏化的必要性。从 Google 论文的结果来看，MOE 稀疏化的大模型通常会被小得多的 dense model 吊打。过去几年的各种稀疏化研究似乎也表明似乎深度学习模型可能不太适合稀疏化。

[一块RTX3090跑ChatGPT体量模型的方法来了！代码已开源](https://mp.weixin.qq.com/s/8FOojNMnCe1D3q0TOgE_EQ)

[当ChatGPT和Stable diffusion碰撞：谷歌用人类反馈提升文生图效果](https://mp.weixin.qq.com/s/FrqpybryiJ-ikO4ZVeISIg)

[ChatGPT 团队背景（共87人）](https://mp.weixin.qq.com/s/VM2SzNyZF2bZDQ7tYvZFkA)

[李宏毅-ChatGPT 原理剖析1對ChatGPT的常見誤解](https://mp.weixin.qq.com/s/gQ6SCJXmcF86o5TH3mW7Sg)

[打造中国版ChatGPT，这是国内最有实力的一批NLP团队与人才](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650869214&idx=1&sn=e8dab20eab64f3f92d42b2e3fd8e2a2b&chksm=84e4c9a0b39340b6e08200354f72c09d589a99e7763ebaf58bd1d3d0510ee0a954b65a040eff&scene=126&sessionid=1677048170#rd)

[ChatGPT是怎么练成的？斯坦福CS224N课程讲解《自然语言生成》等核心技术，附71页Slides](https://mp.weixin.qq.com/s/wxYdUMBDFc7InBi83OBy-A)

[历史最全ChatGPT、LLM相关书籍、论文、博客、工具、数据集、开源项目等资源整理分享](https://mp.weixin.qq.com/s/VoOEw2-cJ-klGMrl7MJeQQ)

> 本资源整理了有关 ChatGPT、GPT 和大型语言模型 (LLM)的必读论文、博客、工具、数据集、开源项目等资源，需要自取。

[一文读懂ChatGPT模型原理](https://zhuanlan.zhihu.com/p/589621442)

[为什么百度没有做出chatGPT这种产品？](https://www.zhihu.com/question/572106826/answer/2808336053)

> GPT一直是openAI在主攻，其他公司大部分都在做bert方向，GPT1、2出来的时候效果也不好，大家都看不上，但是从GPT3开始，大家才发现，好像真给openAI搞出点什么来，不过也仅仅是一点点，那个时候GPT3生成的东西也不够惊艳。后面加入了prompt，让模型有了很强的理解能力，又引入了强化学习，才造就了如今这个惊艳的产品。
>
> ChatGPT基础模型就是GPT3，标注instruct prompt+RLHF。
>
> 国内对大模型的研究也并没有落下国外太多，不过应用落地确实差点，下面列下我知道的大模型吧，也不全
>
> | 模型                | 所属     | 参数量                              |
> | ------------------- | -------- | ----------------------------------- |
> | ChatGPT/InstructGPT | closeAI  | 175B                                |
> | OPT/OPT-IML         | Meta     | 175B/30B/13B                        |
> | ERNIE 3/2/1         | 百度     | 260B/100B/10B                       |
> | 中文GPT-3           | 阿里     | 27B                                 |
> | 悟道2               | 北京智源 | 1750B（神威超算训的，不愧是国家队） |
> | GLM                 | 清华     | 130B/10B                            |
> | PALM/flan-PALM      | google   | 540B                                |
> | Gopher              | deepmind | 280B                                |

符尧 yao.fu@ed.ac.uk的几篇文章

> - 第一篇
>
>   [万字拆解！追溯ChatGPT各项能力的起源](https://www.163.com/dy/article/HPCSSTSN0511D05M.html)
>
>   [ChatGPT进化的秘密](https://mp.weixin.qq.com/s?__biz=MzU5ODY2MTk3Nw==&mid=2247489981&idx=1&sn=5b7b9e49f6bdc925eae584b6ab7d9229&chksm=fe41978bc9361e9dc263c1cfdaa6ad2d882f5e684e7ec427ca4b15d5b987aa607b1c0b01b4ca&scene=21#wechat_redirect)
>
> - 第二篇
>
>   [ChatGPT的一小步，NLP范式转变的一大步](https://mp.weixin.qq.com/s?__biz=MzU5ODY2MTk3Nw==&mid=2247490043&idx=1&sn=d749ed78b417256a5768b67af0e27bc0&chksm=fe4197cdc9361edbe01370064209c68cc2f4c17642f722462d76fc1e4809d48f1397cd5f236e&cur_album_id=2716948051252510721&scene=189#wechat_redirect)
>
>   [深度学习中，模型大了好还是小了好呢？](https://www.zhihu.com/question/434846017/answer/2822621158)
>
> - 第三篇
>
>   [谁能做出中国版ChatGPT？怎么做？ ](https://gov.sohu.com/a/648882950_129720)
>
>   [探索智能的极限](https://yaofu.notion.site/e1cd16d1fae84f87aeddf872c838e07c)

[ChatGPT的训练过程术语整理](https://zhuanlan.zhihu.com/p/614987279)

> 以ChatGPT为代表的大语言模型训练框架包含五步：
>
> 1. 基座预训练（Base pretrain）
> 2. SFT微调（Supervised Fine-Tuning）
> 3. 奖励函数训练（Reward Modeling, RM），最常用的是基于排序的奖励函数建模（Ranking-Based Reward Modeling，RBRM）
> 4. 基于人类反馈的强化学习（RLHF，基于RM/RBRM进行PPO强化学习训练）
> 5. 与人类对齐（Align AI with human values）

[ChatGPT训练算力估算：1万块A100 GPU是误传，中小创企也可突围](https://zhuanlan.zhihu.com/p/606930232)

> 微软云计算服务平台Azure为OpenAI搭建的用于训练ChatGPT的训练算力集群使用了超过4453颗64核的CPU，以及超过10000个Nvidia Tesla V100 GPU，总计约2227台服务器，成本越8-10亿人民币。如果使用Nvidia最新的A100GPU，大约需要3000-5000块GPU进行训练（一次训练耗时两周），成本大约6-8.5亿人民币，但目前Nvidia A100 GPU对我国禁运。

