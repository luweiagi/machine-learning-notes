# 四元数

- [返回上层目录](../navigation.md)

# 为什么要用四元数

四元数的矢量部分直接对应了旋转轴，标量和矢量部分共同决定了旋转角，这是比欧拉角三个角表示旋转更为接近本质的方法。

相较于欧拉角求旋转向量坐标的方法，

* 四元数方法不需要进行求三角函数的运算，因此运算精度更高；
* 四元数不存在欧拉角的框架自锁问题

框架自锁问题如下：

欧拉角速率和机体轴角速率的关系为：
$$
\begin{aligned}
\begin{bmatrix}
\phi'\\
\theta'\\
\psi'\\
\end{bmatrix}
&=
\begin{bmatrix}
1&\tan\theta \sin\phi&\tan\theta \cos\phi\\
0&\cos\phi&-\sin\phi\\
0&\frac{\sin\phi}{\cos\theta}&\frac{\cos\phi}{\cos\theta}\\
\end{bmatrix}
\cdot
\begin{bmatrix}
p\\
q\\
r\\
\end{bmatrix}
\end{aligned}
$$
由上式可看出欧拉角表示方法存在的问题。例如，当俯仰角$\theta\rightarrow\pm90^{\circ}$时，由于方程含有$(\cos\theta)^{-1}$和$\tan\theta$项，使得横滚角变化率$\dot{\phi}$和航向角变化率$\dot{\psi}$无穷大，这种现象在工程上被称为框架自锁现象。解决这个问题的方法是采用四元数法。





# 四元数的性质

Graßmann积

纯四元数有一个很重要的特性：如果有两个纯四元数

[四元数的定义与性质](https://blog.csdn.net/wxc_1998/article/details/119038069)







# 参考资料





===

* [一个对四元数旋转的简单推导](https://zhuanlan.zhihu.com/p/166674954)



* [3Blue1Brown四元数的可视化](https://www.bilibili.com/video/BV1SW411y7W1)

* [3Blue1Brown四元数和三维转动](https://www.bilibili.com/video/BV1Lt411U7og)，[可互动的探索式视频](https://eater.net/quaternions)

直观认识四元数，不过比较难以想象四维空间。。



















