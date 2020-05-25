# pandas

* [返回上层目录](python.md#python)
* [pandas简介](#pandas简介)

# pandas简介

Pandas(PythoN Data Analysis Library)在NumPy基础上提供了更多的数据读写工具，是python的数据分析库，二维表格数据封装，读取二维表格DataFrame（封装了ndarray），读文件，依赖于Numpy。

pandas处理数据能力比excel强，10G以内都可以用pandas处理，超过后有可能用spark处理。

推荐一本书，是pandas的作者写的，[利用Python进行数据分析 (豆瓣)](https://book.douban.com/subject/25779298/)。pandas主要基于numpy.ndarray构造了更高级的Series和DataFrame数据结构。这本书主要就是说明基于这两种数据结构的API用法。这些API主要是对原本numpy操作的补充。行列Index在DataFrame的加强对于各种数据逻辑操作帮助比较大。对pyplot的绘图函数也和两种数据结构绑定的很好。越来越多的数据分析特别是探索式的分析都会转到Python和R这块来，高性能的部分还是会用c扩展来实现。 

**官网**：http://pandas.pydata.org/

**简单介绍**

Pandas是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。Pandas 纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。最具有统计意味的工具包，某些方面优于R软件。数据结构有一维的Series，二维的DataFrame(类似于Excel或者SQL中的表，如果深入学习，会发现Pandas和SQL相似的地方很多，例如merge函数)，三维的Panel（Pan（el) + da(ta) + s，知道名字的由来了吧）。学习Pandas你要掌握的是：

1. 汇总和计算描述统计，处理缺失数据 ，层次化索引
2. 清理、转换、合并、重塑、GroupBy技术
3. 日期和时间数据类型及工具（日期处理方便地飞起）

快速入门：[10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)。这块内容推荐这个文档，在pandas官网上。















# 参考资料

* [如何系统地学习Python 中 matplotlib, numpy, scipy, pandas？](https://www.zhihu.com/question/37180159)

pandas简介中的简单介绍就是复制的这里。





