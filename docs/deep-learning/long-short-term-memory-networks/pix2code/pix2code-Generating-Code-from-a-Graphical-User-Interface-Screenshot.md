# pix2code

- [返回上层目录](../long-short-term-memory-networks.md)



![paper](pic/paper.jpg)

pdf: [pix2code: Generating Code from a Graphical User Interface Screenshot](https://arxiv.org/pdf/1705.07962.pdf)

github: [tonybeltramelli/pix2code](https://github.com/tonybeltramelli/pix2code)



# 参考资料

* [深度学习助力前端开发：自动生成GUI图代码（附试用地址）](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650728064&idx=1&sn=adb744e599299916faa23545c2ab436e&chksm=871b22feb06cabe83ba0e3268a7a8c0c486cf6f42427ab41814e69242f919b25bc7de06ea258&scene=21)
* [UI2Code（一）pix2code](http://w4lle.com/2019/03/13/UI2Code-0/)
* 



* [UI Design to Code Skeleton](https://cs.anu.edu.au/courses/CSPROJECTS/18S2/initialTalks/u6013787.pdf)

介绍了当前的几种生成代码的模型原理。



* [I trained your project, but it always output the same DSL content no matter what images are given](https://github.com/floydhub/pix2code-template/issues/2)

复现pix2code论文代码，发现损失在0.38左右就下不去了，然后生成的代码也有问题，不管给什么图片生成的代码都一样。因为显卡不足，把batchsize从默认的64改为了2。这里有个人也遇到了同样的问题。

* [前端智能化漫谈 (1) - pix2code](https://lusing.blog.csdn.net/article/details/97273669)

* [前端智能化漫谈 (2) - pix2code实战篇](https://blog.csdn.net/lusing/article/details/97400787)
* [前端智能化漫谈 (3) - pix2code推理部分解析](https://lusing.blog.csdn.net/article/details/99082951)
* [前端智能化漫谈 (4) - pix2code结果编辑距离分析](https://lusing.blog.csdn.net/article/details/99684397)

这几篇博客大体上介绍了一下pix2code代码的实践。

* [paper: Improving pix2code based Bi-directional LSTM]()

这是用双向lstm代替了论文里的单向lstm，待看。

* [搞定复杂 GUI！西安交大提出前端设计图自动转代码新方法](https://www.infoq.cn/article/cabmj-kx3xcyv9qbv02a)
* [paper: Automatic Graphics Program Generation using Attention-Based Hierarchical Decoder](https://arxiv.org/abs/1810.11536)

这是西交大发的一篇论文，用了attention注意力机制的，但没给代码和数据集，说是和企业合作的，不便给。

* [github samuelmat19/image2source-tf2 using trainsformer](https://github.com/samuelmat19/image2source-tf2)

该代码使用了transformer技术。基本原理大概看了下，就是encoder使用vggnet，decoder使用transformer，感觉思路比较清晰。值得研究一下。

* [阿里imgcook](https://imgcook.taobao.org/)
* [UI2Code（三）imgcook](http://w4lle.com/2019/04/08/UI2Code-2/)

这是阿里的从设计稿一键生成代码的网站。

* [前端要凉？微软开源的Sketch2Code碉堡了！](https://zhuanlan.zhihu.com/p/44263965)

* [github Microsoft Sketch2Code](https://github.com/microsoft/ailab/tree/master/Sketch2Code)

Sketch2Code是一种基于Web的解决方案，它使用AI将手绘用户界面的图片转换为可用的HTML代码。这个还没看，感觉比较复杂，是个大项目，集成了很多东西。