# Self-Attention机制和Transformer

- [返回上层目录](../natural-language-processing.md)
- [Transformer: Attention Is All You Need  NIPS2017](attention-is-all-you-need/attention-is-all-you-need.md)
- [Transformer模型tensorflow2.0官网demo代码解读](transformer-tf2-demo-code-explain/transformer-tf2-demo-code-explain.md)
- [Transformer的细节问题](transformer-details/transformer-details.md)



===

[如何最简单、通俗地理解Transformer？](https://www.zhihu.com/question/445556653/answer/3272070260)

> 千万不要搁那研究k是建值，q是查询，v是值，如果你看到这种讲解，基本就别看了，那作者自己也没搞明白。
>
> 信我一句，把transformer和gnn，gcn放在一起学，你会看到更加本质的东西。
>
> 这样你就能理解位置嵌入，不管是正弦还是可学习的嵌入，不管是时间嵌入还是其他先验嵌入。
>
> 进而理解什么autoformer，ltransformer，itransformer，graphformer，这样你就会看到transformer在多元时序和图上的应用（二者本就一样）
>
> 然后你就能明白只要改动注意力计算的方式就能造一个新的transformer，至于多头和单头，就非常容易理解。而至于什么多模态cross attention，那也就更加显而易见了。
>
> 而残差和norm只是模型的一种技巧，虽然是小技巧，但实际很有用。
>
> 那个ffn，则更是不值一提。你就算用CNN去平替，在小问题上也毫无压力。
>
> 而至于在cv上的使用，其实就是变着法把图像信息变成token序列。

