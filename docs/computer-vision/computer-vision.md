# 前言

* [返回上层目录](../README.md)
* [YOLO](yolo/yolo.md)
* [图像标注工具](image-mark-tool/image-mark-tool.md)



===



[谷歌复用30年前经典算法，CV引入强化学习，网友：视觉RLHF要来了？](https://mp.weixin.qq.com/s/Lr2ySqTu8Hh343yKPc974g)

[CVPR 二十年，影响力最大的 10 篇论文！](https://mp.weixin.qq.com/s/_yRaZvMuv72KI8OrN-GPyA)

[计算机视觉入门路线](https://mp.weixin.qq.com/s?__biz=MzkyMDE2OTA3Mw==&mid=2247494483&idx=1&sn=7069ade230575cfcb1c1f8c8e8763ecb&chksm=c194544df6e3dd5bea7a98723b764c7db8591e292a775c4c465f715acc260d5d1fc53aa6487c&payreadticket=HIjWViK3B_NTMPuq4Zzm_wEIyKtyvPbtyDFiQTwJsqOLFvAW28Qv38O0pcR_VdMzz15Xsb0#rd)



[那些经典的深度学习模型是怎么想到的？比如resnet？](https://www.zhihu.com/question/577724394/answer/2899520935)

本质上resnet、densenet、inception这些网络并没有明显区别，都是反复利用feature maps，例如resnet的跳跃连接，densenet的密集连接，inception的多分支。甚至包括那种即插即用的模块，例如注意力机制，本质上也是对feature maps的反复利用，然后就是不同的融合方式。

如果你认真看一下所有的注意力模块，就会发现形式反正就那个形式，卷积、全连接、池化这几种随便叠加，哪种有效就行。

同样，如果你认真去看一些特征金字塔的改进，同样如此，特征金字塔的形式是现成的，关键就在于怎么做融合，有的就是从上往下，有的从下往上，有的上下其手。感兴趣的可以看看公众号CV技术指南的各种大总结《计算机视觉中的特征金字塔技术总结》

再同样，数据增强的方法也有十几篇论文，能怎么创新？反正就那个形式，然后就搞出了cutmix，mixup这十几种。感兴趣的可以看看公众号CV技术指南的各种大总结《计算机视觉中的数据增强总结》

因此，如果给你一个最基本的CNN网络，让你在那上面创新，其实思路是很明确的。一个固定的单路径前向传播，可以改成有捷径的(resnet的跳跃连接），多岔路的(inception的多分支)，通往多个地点的捷径的(densenet的密集连接)，另一个分支上做一些特殊的操作再重新相乘的(注意力模块)。

如果还不明白，再举个例子，如果给你一个基本的CNN网络，应该怎么设计轻量化网络？反正就是卷积通道上的连接减少，激活函数在底层简化、替换耗时的操作如单元素的操作等。

搞到后面就会发现，已经没得搞了。因为它的组成就这么一些，每个地方都有人去做优化。

同样后面的CNN已经没得多大可以创新的了，因为对feature maps的应用的设计就那么一些，只能从别的领域上弄过来了，例如把transformer里的注意力模块弄过来做一些操作。本质上与现有的那些网络创新的思路没什么区别。