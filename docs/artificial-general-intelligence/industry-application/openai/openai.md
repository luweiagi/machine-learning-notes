# OpenAI

- [返回上层目录](../industry-application.md)
- [OpenAI介绍](openai-introduction/openai-introduction.md)
- [Emergence of Grounded Compositional Language in Multi-Agent Populations arXiv2017](emergence_lauguage_in_ma/Emergence-of-Grounded-Compositional-Language-in-Multi-Agent-Populations.md)



===

[ChatGPT掀智力革命！OpenAI发布AGI路线图，最终通向超级智能世界](https://mp.weixin.qq.com/s/LRKVE6lcRph8tLe35s-00w)

[OpenAI CEO Sam Altman：OpenAI 对于通用人工智能的未来规划](https://mp.weixin.qq.com/s/ArTmHLZlAo9kXafxyppoUg)



[为什么 OpenAI 可以跑通所有 AGI 技术栈？](https://www.zhihu.com/question/644486081/answer/3398751210)

## 方法论明确

OpenAI的方法论是通往 AGI 的方法论。这个方法论有着非常清晰的逻辑结构，和非常明确的推论。我们甚至可以用公理化的方式来描述它，怎么说呢，感觉上有一种宿命感，。

### 方法论的公理

这套方法论的大厦构建于以下几个“公理”（打引号是因为它们不是真正的“公理”，更多是经验规律，但是在AGI方法论中，它们起到了公理的作用）：

**公理1:** [The bitter lesson](https://link.zhihu.com/?target=https%3A//www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)。我认为所有做AI的人都应该熟读这篇文章。“The bitter lesson” 说的事情是，长期来看，AI领域所有的奇技淫巧都比不过强大的算力夹持的通用的AI算法（这里“强大的算力”隐含了大量的训练数据和大模型）。某种意义上，强大的算力夹持的通用的AI算法才是AGI路径的正道，才是AI技术真正进步的方向。从逻辑主义，到专家系统，到SVM等核方法，到深度神经网络，再到现在的大语音模型，莫不过此。

**公理2: Scaling Law**。这条公理说了，一旦选择了良好且通用的数据表示，良好且通用的数据标注，良好且通用的算法，那么你就能找到一套通用规律，保证数据越多，模型越大，效果越好。而且这套规律稳定到了可以在训练模型之前就能预知它的效果：

如果说 **公理1 The bitter lesson** 是AGI的必要条件——大模型，大算力，大数据，那么**公理2 Scaling Law** 就是AGI充分条件，即我们能找到一套算法，稳定的保证大模型，大算力，大数据导致更好的结果，甚至能预测未来。

而具体来谈，就是我们之前说的“良好且通用的数据表示，良好且通用的数据标注，良好且通用的算法”，在GPT和Sora中都有相应的内容：

- 在GPT中，良好且通用的数据表示，是tokenizer带来的embedding。良好且通用的数据标注是文本清理和去重的一套方法（因为自然语言训练是unsupervised training，数据本身就是标注）。良好且通用的算法就是大家熟知的transformers + autoregressive loss。
- 在Sora中，良好且通用的数据表示，是video compress network带来的visual patch。良好且通用的数据标注是OpenAI自己的标注器给视频详细的描述（很可能是GPT-vision）。良好且通用的算法也是大家熟知的transformers + diffusion

“良好且通用的数据表示，良好且通用的数据标注，良好且通用的算法”同时也为检测scaling law做好了准备，因为你总是可以现在更小规模的模型和数据上检测算法的效果，而不用大幅更改算法。比如GPT1，2，3这几代的迭代路径，以及Sora中OpenAI明确提到visual patch使得他们用完全一样的算法在更小规模的数据上测试。

**公理3: Emerging properties**。这条公理其实是一条检验公理：我怎么知道scaling law带来“质变”，而不仅仅是“量变”？答案是：你会发现，随着scaling law的进行，你的模型突然就能稳定掌握之前不能掌握的能力，而且这是所有人能够直观体验到的。比如GPT-4相比于GPT-3.5，可以完成明显更复杂的任务，比如写一个26行诗来证明素数是无限的，每行开头必须是从A到Z。比如Sora相对于之前的模型，它的时空一致性，以及对现实中物理规律的初步掌握。没有 Emerging properties，我们很难直观感觉到突破性的变化，很难感知“我们真的向AGI前进了一步”，或者是“我们跑通了一个技术栈”。