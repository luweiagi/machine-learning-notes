# 梯度下降算法

* [返回上层目录](../gradient-update-algorithm.md)
* [梯度下降算法的演化](gradient-descent-algorithms-evolution/gradient-descent-algorithms-evolution.md)
* [随机梯度下降SGD](sgd/sgd.md)
* [动量法Momentum](momentum/momentum.md)
* [牛顿动量Nesterov](nesterov/nesterov.md)
* [AdaGrad](adagrad/adagrad.md)
* [RMSprop](rmsprop/rmsprop.md)
* [Adadelta](adadelta/adadelta.md)
* [Adam](adam/adam.md)
* [Nadam](nadam/nadam.md)
* [AMSGrad](amsgrad/amsgrad.md)
* [AdasMax](adamax/adamax.md)



# 参考资料

* [Deep Learning 之 最优化方法](https://blog.csdn.net/BVL10101111/article/details/72614711)

* [从 SGD 到 Adam —— 深度学习优化算法概览(一)](https://zhuanlan.zhihu.com/p/32626442)

* [深度学习最全优化方法总结比较（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)

* [10个梯度下降优化算法+备忘单](https://ai.yanxishe.com/page/TextTranslation/1603?from=singlemessage)

  [深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）](https://www.cnblogs.com/guoyaohua/p/8542554.html)

===

[优化方法总结以及Adam存在的问题(SGD, Momentum, AdaDelta, Adam, AdamW，LazyAdam)](https://blog.csdn.net/yinyu19950811/article/details/90476956)



[请问在深度学习解决分类问题时，测试集loss出现几乎收敛后突然上升，然后缓慢收敛，这是怎么回事？](https://www.zhihu.com/question/562705011/answer/2761495519)

> 别听那几个答案瞎说，什么 double descent 都出来了，一看就是调参经验不够丰富。这种 jump 就是典型的训练不稳定，是 optimizer 没有调好的结果。要解决这个问题，你可以考虑以下几点：learning rate 是否过大？ optimizer 的 momentum coefficient 是否合适？learning rate decay 是否合适（一般用万能的 cosine schedule)？weight decay 加够了没有？
>
> 如果这些都调过了效果还不好，那就要考虑是否你的 loss 有 numerical stability 的问题。可以考虑改 loss 或者用 gradient clipping 一类的 trick。

评论：

> 王一to作者：不要用adam，这东西没有理论推导，就是错的。要用普通sgd或者加momentum

> 作者to王一：神™不要用 Adam，现在 AdamW 是标准做法，从 CV 到 NLP 都在用。说 SGD 有理论指导的你倒是给我证一个 general non-convex case 的 convergence guarantee 试试，不要拿 NTK，PL 或者 strict-saddle 这种在神经网络里面根本不成立的假设来糊弄。

> 王一to作者：你去看看adam给你优化出来的权重，大的要死，不得不加decay。sgd优化符合最小权重变动的原则。但是adam优化出来不在起始点附近微调，而是大动。你又不是不知道adam优化到后期准确度就是比不上sgd

> 作者to王一：我所了解的很多常用模型，包括 CV 的 ViT, ConvNext, Swin Transformer, SimCLR, MoCo v3, MAE，CLIP，DallE-2，以及 NLP 的 BERT，GPT3, T5, Chinchilla, PaLM 都在用 AdamW 或者 AdamW 的 large scale 变体 (LARS, LAMB, AdaFactor 等)，怎么到了你这里 AdamW 就不好用了？

> 王一to作者：你无非就是说这个东西流行他就是对的。照这么说大家都在得新冠，得新冠就是对的唠？还特别喜欢耍一堆名词，意思你知识很广的样子。对和错是需要独立思考的。你能否从loss函数，通过梯度,hessian等第一性原理推导出adam？sgd是可以推导出来的。别跟我扯什么收敛性证明。我关心的是adam里面除以导数得平方的移动平均，这一项对不对的问题。adam除以这一项以后，其实很小的梯度都被放大了很多倍，几乎是方差为一得正态分布了，这导致小梯度被大幅放大了。例如有一个共享参数，由于共享，正常情况下sgd的时候更新速度要除以sqrt(N)，N是共享得参数个数，这样才能保持稳定。好了，adam一来，都变成同样速度更新了，sqrt(N)项消失，导致这个贡献参数被快速更新，loss很容易爆的知不知道？

> 一勺香菜to作者：你说adam不如sgd能理解，但是说是错的有点太扯了。最多是理论不完善而已，牺牲一点点accuracy换来大幅效率提高我想傻子才不用吧

> 王一to一勺香菜：开始提高效率收敛快，后面就摆烂了。我这么了解adam是因为我不但用过，还自己手撸过，玩烂以后发现他是个坑。就像和女神离过婚的男人，看着大家看女神的眼光，我不禁感慨一句，这是个坏女人

> 大只鱼to王一：实践是检验真理的一标准啊，湍流没解决，也不妨碍他用

> 王一to大只鱼：我比楼主进行了更多的实践啊，这才发现adam是错的。每天一早起来写cuda，写算子，看论文，一直到天黑，几年了。这个不是调包侠可以比的。。。

> 王一：给你们看看第三方的一些说法：[（转）优化时该用SGD，还是用Adam？——绝对干货满满！](https://blog.csdn.net/S20144144/article/details/103417502)

> cycloidzzz to作者：说了这么多，你倒是给神经网络证个sgd收敛的bound呀。adam和sgd都没法对神经网络给出什么令人满意的bound，就不要五十步笑百步了

> 作者to cycloidzzz：对不起，收敛证明领域我不懂，我无法给出证明。知为知，不知为不知。但是我认为收敛是数学家来擦屁股的小事。但是公式没有来由是一个很可怕的事。不要把对错和收敛等价，这是两码事。
>
> 另外这个问题后面我不再回答了，因为相信adam的人太多了，一一回答我没有那么多精力。爱信不信，反正我的技术内容已经说完了，没有更多了，并没有人要逼你们信，如果adam可以让你们经常郁闷，那对我来说是好事，因为我的成果会更容易赚钱。今天被激了才会说出这么多技术细节。



[（转）优化时该用SGD，还是用Adam？——绝对干货满满！](https://blog.csdn.net/S20144144/article/details/103417502)