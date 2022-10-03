# 动态规划

* [返回上层目录](../jianzhi-offer.md)
* [剑指offer14：剪绳子](#剑指offer14：剪绳子)



# 剑指offer14：剪绳子

> 题目：给你一根长度为n绳子，请把绳子剪成m段（m、n都是整数，n>1并且m≥1）。每段的绳子的长度记为k[0]、k[1]、……、k[m]。k[0]\*k[1]\*…\*k[m]可能的最大乘积是多少？例如当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到最大的乘积18。

首先定义函数f(n)为把长度为n的绳子剪成若干段后各段长度乘积的最大值。在剪第一刀的时候，我们有n-1中可能的选择，也就是剪出来的第一段绳子可能长度为1，2，...，n-1。因此f(n) = max(f(i) \* f(n-i))。其中0<i<n。

c++:

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <stack>
 
using namespace std;
 
int maxProductAfterCutting_Solution1(int length)
{
	if (length < 2)
		return 0;
	if (length == 2)
		return 1;
	if (length == 3)
		return 2;
	int *products = new int[length + 1];
	products[0] = 0;
	products[1] = 1;
	products[2] = 2;
	products[3] = 3;
 
	int max = 0;
	for (int i = 4; i <= length; ++i)
	{
		max = 0;
		for (int j = 1; j <= i / 2; j++)
		{
			int product = products[j] * products[i - j];
			if (product > max)
				max = product;
		}
		products[i] = max;
	}
	max = products[length];
	delete[] products;
	return max;
}
int main()
{
	cout <<maxProductAfterCutting_Solution1(10)<<endl;
	system("pause");
}
```



# 参考资料

* [面试题：剪绳子──动态规划 or 贪心算法](https://blog.csdn.net/sinat_36161667/article/details/80785142)

本文参考此博客。

