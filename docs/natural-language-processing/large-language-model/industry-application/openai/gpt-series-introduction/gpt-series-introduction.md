# GTP系列介绍

* [返回上层目录](../openai.md)

GPT-1和GPT-2都有公开的预训练模型和源代码可用，但是这些模型的训练数据和训练代码不是公开的。这些模型的源代码可以在GitHub上找到。

至于GPT-3，目前它仍然是一个私有的商业产品，只有少数被邀请的合作伙伴才能访问它。OpenAI表示他们不会公开GPT-3的预训练模型和源代码，但是他们提供了一些API接口，供开发人员和研究人员使用。

chatgpt应该算是第3代nlp技术了。前两代是预训练+微调，然后是自监督+prompt，现在的是利用了RHLF、instruct等技术。



===



[从 GPT 到 ChatGPT 的演进与应用思考](https://mp.weixin.qq.com/s/3Pr82xKpZ7mAWQcxPPB1xA)

[paper: A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT](https://arxiv.org/abs/2302.09419)

这里[ChatGPT背后的大模型技术如何炼？MSU等最新《预训练基础模型综述》，97页pdf全面阐述BERT到ChatGPT历史脉络](https://www.zhuanzhi.ai/vip/f9ef3cea409e4e561fa87db1821f57d0)提到了上述论文，预训练基础模型(PFMs)被视为具有不同数据模态的各种下游任务的基础。预训练的基础模型，如BERT、GPT-3、MAE、DALLE-E和ChatGPT，在大规模数据上进行训练，为广泛的下游应用提供了合理的参数初始化。**PFMs背后的预训练思想在大型模型的应用中起着重要的作用。**作为一种迁移学习范式，预训练通过冻结和微调技术应用于计算机视觉，显示出良好的性能。词向量在自然语言处理中也可以看作是修饰词的一

[GPT系列的数据集之谜](https://mp.weixin.qq.com/s/p0s6FmEof2gkb0jrHBo3JA)

[ChatGPT背后的经济账]

OneFlow发布了《ChatGPT背后的经济账》，其作者从经济学视角推导了训练大型语言模型的成本。本文作者则整理分析了2018年到2022年初从GPT-1到Gopher的相关大型语言模型的所有数据集相关信息，希望帮助有志于开发“类ChatGPT”模型的团队少走一步弯路。