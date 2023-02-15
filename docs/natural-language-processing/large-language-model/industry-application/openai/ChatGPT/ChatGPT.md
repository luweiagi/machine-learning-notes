# ChatGPT

* [返回上层目录](../openai.md)







===

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

