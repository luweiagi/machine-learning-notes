# DarkNet深度学习框架


* [返回上层目录](../yolo.md)





[darknet整体框架](https://blog.csdn.net/weixin_41722370/article/details/90340347?utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-9.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-9.control)

darknet是使用C和CUDA编写的开源的神经网络框架，它快速且使用简单，之前在海康做caffe方面的工作，本想研究caffe的源代码，但是被导师推荐阅读darknet源代码加深对深度学习的理解而且还能巩固C语言，由此记录一下我的darknet源码阅读之路。

[Darknet框架简介](https://blog.csdn.net/mao_hui_fei/article/details/113820303?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.control&spm=1001.2101.3001.4242)

darknet是一个较为轻型的完全基于C与CUDA的开源深度学习框架，其主要特点就是容易安装，没有任何依赖项（OpenCV都可以不用），移植性非常好，支持CPU与GPU两种计算方式。

[Darknet概述](https://blog.csdn.net/u010122972/article/details/83541978)

优点：Darknet是一个比较小众的深度学习框架，没有社区，主要靠作者团队维护，所以推广较弱，用的人不多。而且由于维护人员有限，功能也不如tensorflow等框架那么强大，但是该框架还是有一些独有的优点：

相比于TensorFlow来说，darknet并没有那么强大，但这也成了darknet的优势：

1. darknet完全由C语言实现，没有任何依赖项，当然可以使用OpenCV，但只是用其来显示图片、为了更好的可视化；
2. darknet支持CPU（所以没有GPU也不用紧的）与GPU（CUDA/cuDNN，使用GPU当然更块更好了）；
3. 正是因为其较为轻型，没有像TensorFlow那般强大的API，所以给我的感觉就是有另一种味道的灵活性，适合用来研究底层，可以更为方便的从底层对其进行改进与扩展；

[Darknet框架简介](https://blog.csdn.net/xunan003/article/details/79932888)

darknet深度学习框架源码分析：详细中文注释，涵盖框架原理与实现语法分析

https://github.com/hgpvision/darknet



