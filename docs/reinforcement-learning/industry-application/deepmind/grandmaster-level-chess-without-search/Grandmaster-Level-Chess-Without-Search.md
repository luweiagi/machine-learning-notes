# Grandmaster-Level Chess Without Search

- [返回上层目录](../deepmind.md)



论文地址 ：[*Amortized Planning with Large-Scale Transformers: A Case Study on Chess*](https://arxiv.org/pdf/2402.04494)



# 参考资料



===

[聊聊端到端与下一代自动驾驶系统](https://zhuanlan.zhihu.com/p/692302970)

数据驱动和传统方法之间关系如何调和？

其实和自动驾驶非常类似的一个例子就是下棋，刚好在今年2月份的时候[Deepmind](https://zhida.zhihu.com/search?content_id=241975575&content_type=Article&match_order=1&q=Deepmind&zhida_source=entity)发表了一篇文章（Grandmaster-Level Chess Without Search [https://arxiv.org/abs/2402.04494](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.04494)）就在探索只用数据驱动，抛弃AlphaGo和[AlphaZero](https://zhida.zhihu.com/search?content_id=241975575&content_type=Article&match_order=1&q=AlphaZero&zhida_source=entity)中的MCTS search是否可行。类比到自动驾驶中就是，只用一个网络直接输出action，抛弃掉后续所有的步骤。**文章的结论是，在相当的规模的数据和模型参数下，不用搜索仍然可以得到一个还算合理的结果，然而和加上搜索的方法比，还有非常显著的差距。**（文章中这里的对比其实也不尽公平，实际差距应该更大）尤其是在解一些困难的残局上，[纯数据驱动性能](https://zhida.zhihu.com/search?content_id=241975575&content_type=Article&match_order=1&q=纯数据驱动性能&zhida_source=entity)非常糟糕。这类比到自动驾驶中，也就是意味着，需要多步博弈的困难场景或corner case，仍然很难完全抛弃掉传统的优化或者[搜索算法](https://zhida.zhihu.com/search?content_id=241975575&content_type=Article&match_order=1&q=搜索算法&zhida_source=entity)。像AlphaZero一样合理地运用各种技术的优势，才是最为高效提升性能的方式。



* [大模型是否有推理能力？DeepMind数月前的论文让AI社区吵起来了](https://zhuanlan.zhihu.com/p/2466197714)