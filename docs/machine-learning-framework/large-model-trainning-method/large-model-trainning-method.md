# AI大模型训练框架


* [返回上层目录](../machine-learning-framework.md)

**深度学习理论上人人都可参与研发探索，然而，具有“巨量数据、巨量算法、 巨量算力”三大特征的AI大模型却成了门槛很高的技术竞赛。**对于任何企业包括巨头来说，打造一个大模型都不是一件容易的事情，需要收集海量数据、需要采买海量算力、需要进行大量研发，金钱、时间、人力投入同样“巨量”，正是因为此构建AI大模型的企业几乎都是财力雄厚、技术强悍的巨头——微软甚至宣称其用了价值10亿美元的超级计算机来训练其AI大模型。在“土豪”的科技巨头外，少数有一定科研经费和实力的机构推出了小众的大模型，但不具备工业化条件。

源1.0有底气建立开源开放生态在于技术的底气：单论参数规模其拥有2457亿参数，超过1750亿参数的GPT-3，且其解决了巨量模型训练不稳定的业界难题，提出了稳定训练巨量模型的算法。

在计算效率上，源1.0训练用了2128张GPU、且在16天内就完成了训练，看上去是不小的算力和时间投资，不过相对同等量级的AI大模型效率却高了不少。“巨无霸”“MT-NLG”的训练需要的算力相当于4480块A100显卡，GPT-3的训练则是在超过28.5万个CPU核心以及超过1万个GPU上完成，训练时间均超过一个月。源1.0的训练共消耗约4095PD（PetaFlop/s-day），相较于“GPT-3”的3640PD，计算效率得到大幅提升。源1.0做到这一点的核心原因在于其采用了张量并行、流水线并行和数据并行的三维并行策略，这背后则是用好了浪潮智慧计算的“看家本领”。

# 参考资料

* [AI大模型成巨头比武场，“强人工智能”时代来临？](https://zhuanlan.zhihu.com/p/426482407)

===

[为什么说大模型训练很难？](https://www.zhihu.com/question/498271491)

[大模型参数量和模型大小怎么换算？](https://www.zhihu.com/question/589705235)

[如何在多个GPU上训练很大的模型-并行训练](https://zhuanlan.zhihu.com/p/536304655)

> 本文译自OpenAI研究院Lilian Weng在2021年9月发布，并在2022年4月更新的文章[How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)中的第一部分-并行训练方法。

[如何训练大模型 Dive into Big Model Training](https://zhuanlan.zhihu.com/p/546215261)

[如何训练千亿/万亿大模型](https://zhuanlan.zhihu.com/p/542596233)



[什么是大模型？超大模型和 Foundation Model 呢？](https://www.zhihu.com/question/498275802/answer/2221187242)

>  目前部分深度学习框架，例如Pytorch和Tensorflow，没有办法满足超大规模模型训练的需求，于是微软基于Pytroch开发了DeepSpeed，腾讯基于Pytroch开发了派大星PatricStar，达摩院同基于Tensoflow开发的分布式框架Whale。像是华为昇腾的MindSpore、百度的PaddlePaddle，还有国内的追一科技OneFlow等厂商，对超大模型训练进行了深度的跟进与探索，基于原生的AI框架支持超大模型训练。



[训练ChatGPT需要什么-超大模型训练part1（机器学习系统）](https://zhuanlan.zhihu.com/p/599688923)

[训练ChatGPT需要什么-超大模型训练part2（机器学习系统）](https://zhuanlan.zhihu.com/p/599695149)



[ChatGPT训练算力估算：1万块A100 GPU是误传，中小创企也可突围](https://zhuanlan.zhihu.com/p/606930232)

> 微软云计算服务平台Azure为OpenAI搭建的用于训练ChatGPT的训练算力集群使用了超过4453颗64核的CPU，以及超过10000个Nvidia Tesla V100 GPU，总计约2227台服务器，成本越8-10亿人民币。如果使用Nvidia最新的A100GPU，大约需要3000-5000块GPU进行训练（一次训练耗时两周），成本大约6-8.5亿人民币，但目前Nvidia A100 GPU对我国禁运。

