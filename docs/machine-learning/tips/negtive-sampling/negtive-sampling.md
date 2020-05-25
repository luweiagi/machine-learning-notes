# 负采样

* [返回上层目录](../tips.md)
* [两种负采样](#两种负采样)
  * [第一种负采样](#第一种负采样)
  * [第二种负采样](#第二种负采样)
* [两种负采样的本质区别](#两种负采样的本质区别)
  * [第一种负采样求梯度](#第一种负采样求梯度)
  * [第二种负采样求梯度](#第二种负采样求梯度)

# 两种负采样

## 第一种负采样

采样方式为：
$$
\begin{aligned}
<u, i_0>,\\
<u, i_1>,\\
<u, i_2>,\\
<u, i_3>,\\
<u, i_4>,\\
<u, i_5>
\end{aligned}
$$
其中，$i_0$为正样本，$i_1,\ i_2, ... , i_5$为负样本。

损失计算方式为
$$
\begin{aligned}
\text{loss}&=-p_0\mathop{\Pi}_{i=1}^5(1-p_i)\\
\Rightarrow\text{loss}&=-log(p_0)-\sum_{i=1}^5log(1-p_i)\\
&=-log\left(\frac{1}{1+\text{exp}(-s_0)}\right)-\sum_{i=1}^5log\left(1-\frac{1}{1+\text{exp}(-s_i)}\right)\\
\end{aligned}
$$

## 第二种负采样

采样方式为：
$$
<u, i_0, i_1, i_2, i_3, i_4, i_5>
$$
其中，$i_0$为正样本，$i_1,\ i_2, ... , i_5$为负样本。

损失计算方式为
$$
\begin{aligned}
&\text{loss}=-\frac{\text{exp}(s_0)}{\sum_{i=0}^5\text{exp}(s_i)}\\
\Rightarrow &\text{loss}=-(s_0-\text{log}\sum_{i=0}^5\text{exp}(s_i))=-s_0+\text{log}\sum_{i=0}^5\text{exp}(s_i)
\end{aligned}
$$

# 两种负采样的本质区别

对两种负采样的损失函数分别求梯度。

## 第一种负采样求梯度

$$
\begin{aligned}
\frac{\partial \text{loss}}{\partial s_0}&=\frac{\partial\left[-log\left(\frac{1}{1+\text{exp}(-s_0)}\right)-\sum_{i=1}^5log\left(1-\frac{1}{1+\text{exp}(-s_i)}\right)\right]}{\partial s_0}\\
&=-\frac{p_0(1-p_0)}{p_0}\\
&=-1+p_0\\

\frac{\partial \text{loss}}{\partial s_i}|_{i=1,2,...,5}&=\frac{\partial\left[-log\left(\frac{1}{1+\text{exp}(-s_0)}\right)-\sum_{i=1}^5log\left(1-\frac{1}{1+\text{exp}(-s_i)}\right)\right]}{\partial s_i}\\
&=-\frac{p_i(1-p_i)}{1-p_i}\\
&=p_i\\
\end{aligned}
$$

其中，上式的推导用到了 sigmoid导数的特点：$f'(z)=f(z)(1-f(z))$。

注意，这里的
$$
\begin{aligned}
\sum_{i=0}^5\frac{\partial \text{loss}}{\partial s_i}=-1+\sum_{i=0}^5p_i\neq0
\end{aligned}
$$
即**所有的变量$s_i$的梯度的和不是1**

## 第二种负采样求梯度

$$
\begin{aligned}
\frac{\partial \text{loss}}{\partial s_0}&=\frac{\partial[-(s_0-\text{log}\sum_{i=0}^5\text{exp}(s_i))]}{\partial s_0}\\
&=-1+\frac{\text{exp}(s_0)}{\sum_{i=0}^5\text{exp}(s_i)}\\
&=-1+p_0\\

\frac{\partial \text{loss}}{\partial s_i}|_{i=1,2,...,5}&=\frac{\partial[-(s_0-\text{log}\sum_{i=0}^5\text{exp}(s_i))]}{\partial s_i}\\
&=\frac{\text{exp}(s_i)}{\sum_{i=0}^5\text{exp}(s_i)}\\
&=p_i\\
\end{aligned}
$$

注意：所有变量的梯度值相加等于1:
$$
\begin{aligned}
\sum_{i=0}^5\frac{\partial \text{loss}}{\partial s_i}=-1+\sum_{i=0}^5p_i=-1+1=0
\end{aligned}
$$
所以，**每个变量$s_i$的梯度更新值是相互制约的，总和等于1**。

**这就是两种负采样的本质区别，即所有的变量$s_i$的梯度的和是否等于1**。

