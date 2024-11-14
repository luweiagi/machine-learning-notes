# 磁力计校准

* [返回上层目录](../magnetometer.md)

磁力计的数据在实际中是椭球的形状，在此之前使用了球体拟合进行校准，也就是简化为正球体的模型，得出的结果比较差，航向计算不准，还是需要用椭球的模型来估计偏移量，先使用标准的椭球方程，进行化简与变形，得到最小二乘法可以进行估计的标准形式，之后对原始数据进行最小二乘法矩阵的赋值，求解方程，最终观察拟合效果。

椭球的标准方程为：
$$
\left(\frac{x-O_x}{R_x}\right)^2+\left(\frac{y-O_y}{R_y}\right)^2+\left(\frac{z-O_Z}{R_z}\right)^2=1
$$
椭球方程化简，就是将椭球方程转化为适合最小二乘法求解的形式，先将椭球方程展开，得到：
$$
\frac{x^2-2O_xx+O_x^2}{R_x^2}+\frac{y^2-2O_yy+O_y^2}{R_y^2}+\frac{z^2-2O_zz+O_z^2}{R_z^2}=1
$$
然后可变形为
$$
\begin{bmatrix}
-x^2 & -y^2 & -z^2 & -x & -y & -z 
\end{bmatrix}
\begin{bmatrix}
-\frac{1}{R_x^2}\\
-\frac{1}{R_y^2}\\
-\frac{1}{R_z^2}\\
\frac{2O_x}{R_x^2}\\
\frac{2O_y}{R_y^2}\\
\frac{2O_z}{R_z^2}\\
\end{bmatrix}
=
1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)
$$
但是上式右侧依然包含有未知数，需要把上式右侧变为1，即
$$
\begin{bmatrix}
-x^2 & -y^2 & -z^2 & -x & -y & -z 
\end{bmatrix}
\begin{bmatrix}
-\frac{1}{R_x^2\left(1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)\right)}\\
-\frac{1}{R_y^2\left(1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)\right)}\\
-\frac{1}{R_z^2\left(1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)\right)}\\
\frac{2O_x}{R_x^2\left(1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)\right)}\\
\frac{2O_y}{R_y^2\left(1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)\right)}\\
\frac{2O_z}{R_z^2\left(1-\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)\right)}\\
\end{bmatrix}
=
1
$$
上式可看成
$$
Ax=b
$$
用最小二乘法求解此方程，即
$$
\begin{aligned}
&A^TAx=A^Tb\\
\Rightarrow&x = (A^TA)^{-1}A^Tb
\end{aligned}
$$
然后我们来分别求解未知量$O_x$，$O_y$，$O_z$，$R_x$，$R_y$，$R_z$。

假设我们已经得到了未知量的解$H[6]$，并假设$\left(\frac{O_x^2}{R_x^2}+\frac{O_y^2}{R_y^2}+\frac{O_z^2}{R_z^2}\right)=$G，则有
$$
\begin{bmatrix}
H_1\\
H_2\\
H_3\\
H_4\\
H_5\\
H_6\\
\end{bmatrix}
=
\begin{bmatrix}
-\frac{1}{R_x^2\left(1-G\right)}\\
-\frac{1}{R_y^2\left(1-G\right)}\\
-\frac{1}{R_z^2\left(1-G\right)}\\
\frac{2O_x}{R_x^2\left(1-G\right)}\\
\frac{2O_y}{R_y^2\left(1-G\right)}\\
\frac{2O_z}{R_z^2\left(1-G\right)}\\
\end{bmatrix}
$$
现在我们先求$G$，
$$
\frac{H_4^2}{H_1}=\frac{\left(\frac{2O_x}{R_x^2\left(1-G\right)}\right)^2}{-\frac{1}{R_x^2\left(1-G\right)}}=-\frac{4O_x^2}{R_x^2(1-G)}\\
\frac{H_5^2}{H_2}=\frac{\left(\frac{2O_y}{R_y^2\left(1-G\right)}\right)^2}{-\frac{1}{R_y^2\left(1-G\right)}}=-\frac{4O_y^2}{R_y^2(1-G)}\\
\frac{H_6^2}{H_3}=\frac{\left(\frac{2O_z}{R_z^2\left(1-G\right)}\right)^2}{-\frac{1}{R_z^2\left(1-G\right)}}=-\frac{4O_z^2}{R_z^2(1-G)}
$$
可得
$$
\frac{H_4^2}{H_1}+\frac{H_5^2}{H_2}+\frac{H_6^2}{H_3}=\frac{-4}{1-G}\left(\frac{O_x^2}{R_x^2}+\frac{O_z^2}{R_z^2}+\frac{O_z^2}{R_z^2}\right)=\frac{-4}{1-G}G
$$
现在我们根据上式求$G$值，

为了方便起见，令$\frac{H_4^2}{H_1}+\frac{H_5^2}{H_2}+\frac{H_6^2}{H_3}=H_v$，则有
$$
\begin{aligned}
&H_v=\frac{-4}{1-G}G\\
\Rightarrow &G=\frac{H_v}{H_v-4}
\end{aligned}
$$
前面我们已求得
$$
\begin{bmatrix}
H_1\\
H_2\\
H_3\\
H_4\\
H_5\\
H_6\\
\end{bmatrix}
=
\begin{bmatrix}
-\frac{1}{R_x^2\left(1-G\right)}\\
-\frac{1}{R_y^2\left(1-G\right)}\\
-\frac{1}{R_z^2\left(1-G\right)}\\
\frac{2O_x}{R_x^2\left(1-G\right)}\\
\frac{2O_y}{R_y^2\left(1-G\right)}\\
\frac{2O_z}{R_z^2\left(1-G\right)}\\
\end{bmatrix}
$$
所以，
$$
\begin{aligned}
O_x&=H_4/(\frac{1}{R_x^2(1-G)})/2=-\frac{H_4}{2H_1}\\
O_y&=H_5/(\frac{1}{R_y^2(1-G)})/2=-\frac{H_5}{2H_2}\\
O_x&=H_6/(\frac{1}{R_z^2(1-G)})/2=-\frac{H_6}{2H_3}
\end{aligned}
$$
还有
$$
\begin{aligned}
R_x^2&=-\frac{1}{H_1(1-G)}=-\frac{1}{H_1(1-\frac{H_v}{H_v-4})}=\frac{H_v-4}{4H_1}\\
\Rightarrow \frac{1}{R_x^2}&=\frac{4}{H_v-4}H_1\\
R_y^2&=-\frac{1}{H_2(1-G)}=-\frac{1}{H_2(1-\frac{H_v}{H_v-4})}=\frac{H_v-4}{4H_2}\\
\Rightarrow \frac{1}{R_y^2}&=\frac{4}{H_v-4}H_2\\
R_z^2&=-\frac{1}{H_3(1-G)}=-\frac{1}{H_3(1-\frac{H_v}{H_v-4})}=\frac{H_v-4}{4H_3}\\
\Rightarrow \frac{1}{R_z^2}&=\frac{4}{H_v-4}H_3\\
\end{aligned}
$$

# 参考资料

* [有没有介绍用于椭球拟合的最小二乘算法的文献或者教材？](https://www.zhihu.com/question/330095789/answer/3055890390)

本文参考了此知乎回答。

