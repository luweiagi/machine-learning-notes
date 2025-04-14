# AdaGrad

* [返回上层目录](../gradient-descent-algorithms.md)



[Deep Learning 最优化方法之AdaGrad](https://blog.csdn.net/bvl10101111/article/details/72616097)



1.简单来讲，设置全局学习率之后，每次通过，全局学习率逐参数的除以历史梯度平方和的平方根，使得每个参数的学习率不同

2.效果是：在参数空间更为平缓的方向，会取得更大的进步（因为平缓，所以历史梯度平方和较小，对应学习下降的幅度较小）

3.缺点是,使得学习率过早，过量的减少

4.在某些模型上效果不错。