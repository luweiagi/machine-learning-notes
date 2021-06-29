# k均值聚类

* [返回上层目录](../clustering.md)
* [模型](#模型)
* [策略](#策略)
* [算法](#算法)
* [算法特性](#算法特性)
  * [总体特点](#总体特点)
  * [收敛性](#收敛性)
  * [初始类的选择](#初始类的选择)
  * [类别个数k的选择](#类别个数k的选择)
* [代码实践](#代码实践)
  * [sklearn中的KMeans实现](#sklearn中的KMeans实现)
  * [word2vec中KMeans实现](#word2vec中KMeans实现)
    * [word2vec中的KMeans源码摘录](#word2vec中的KMeans源码摘录)
    * [对embdding进行KMeans聚类](#对embdding进行KMeans聚类)

k均值聚类是基于样本集合划分的聚类算法。k均值聚类将样本划分为$k$个子集，构成$k$个类，将$n$个样本分到$k$个类中每个样本到其所属类的中心的距离最小。每个样本只能属于一个类，所以$k$均值聚类是硬聚类。下面分别介绍$k$均值聚类的模型、策略、算法，讨论算法的特性及相关问题。

实际上，均值聚类是使用最大期望算法（Expectation-Maximization algorithm）求解的高斯混合模型（Gaussian Mixture Model, GMM）在正态分布的协方差为单位矩阵，且隐变量的后验分布为一组[狄拉克δ函数](https://baike.baidu.com/item/%E7%8B%84%E6%8B%89%E5%85%8B%CE%B4%E5%87%BD%E6%95%B0/5760582)时所得到的特例。

# 模型

给定$n$个样本的集合$X=\{x_1, x_2, ... , x_n\}$，每个样本由一个特征向量表示，特征向量的维数是$m$。k均值聚类的目标是将$n$个样本分到$k$个不同的类或着簇中，这里假设$k<n$。$k$个类$G_1, G_2, ... ,G_k$形成对样本集合$X$的划分，其中
$$
G_i \bigcap G_j = \varnothing ,\ \bigcup_{i=1}^{k}G_i=X
$$
。用$C$表示划分，一个划分对应着一个聚类结果。

划分$C$是一个多对一的函数。事实上，如果把每个样本用一个整数$i \in \{1, 2, ... , n\}$表示，每个类也用一个整数$l  \in \{1, 2, ... , k\}$表示，那么划分或者聚类可以用函数$l = C(i)$表示，其中$i \in \{1, 2, ... , n\}$，$l \in \{1, 2, ... , k\}$。所以**k均值聚类的模型是一个从样本到类的函数**。

# 策略

k均值聚类归结为样本集合$X$的划分，或者从样本到类的函数的选择问题。k均值聚类的策略是通过损失函数的最小化选取最优的划分或函数$C^{*}$。

首先，采用欧式距离平方作为样本之间的距离$d(x_i, x_j)$
$$
\begin{aligned}
d(x_i,x_j)&=\sum_{k=1}^{m}(x_{ki}-x_{kj})^2\\
&= ||x_i-x_j||^2
\end{aligned}
$$
然后，定义样本与其所属类的中心之间的距离的总和为损失函数，即
$$
W(C)=\sum_{l=1}^{k}\sum_{C(i)=l}||x_i-\bar{x}_l||^2
$$
式中，
$$
\bar{x}_l=(\bar{x}_{1l},\bar{x}_{2l},...,\bar{x}_{ml})^T
$$
是第$l$个类的均值或者聚类中心，属于第$l$类的样本个数为$n_l=\sum_{i=1}^nI\left(C(i)=l\right)$，$I(C(i) = l)$是指示函数，取值为1或0。函数$W(C)$也称为能量，表示相同类中的样本相似的程度。

k均值聚类就是求解最优化问题：
$$
\begin{aligned}
C^*&=\text{arg }\mathop{\text{min}}_C W(C)\\
&=\text{arg }\mathop{\text{min}}_C \sum_{l=1}^{k}\sum_{C(i)=l}||x_i-\bar{x}_l||^2
\end{aligned}
$$
相似的样本被聚到同类时，损失函数最小，这个目标函数的最优化能达到聚类的目的。但是，这是一个组合优化问题，$n$个样本分到$k$类，所有可能分法的数目时：
$$
\begin{aligned}
S(n,k)=\frac{1}{k!}\sum_{l=1}^k(-1)^{k-l}
\begin{pmatrix}
k\\ 
l
\end{pmatrix}
k^n
\end{aligned}
$$
这个数字是指数级的。事实上，k均值聚类的最优解求解问题是NP难问题。现实中采用迭代的方法求解。

# 算法

k均值聚类的算法是一个迭代的过程，每次迭代包括两个步骤：

* 首先选择$k$个类的中心，将样本逐个指派到与其最近的中心的类中，得到一个聚类的结果；
* 然后更新每个类的样本的均值，作为类的新的中心

重复以上步骤，直到收敛为止。具体过程如下。

首先，对于给定的中心值$(m_1, m_2, ... , m_k)$，求一个划分$C$，使得目标函数极小化：
$$
\mathop{\text{min}}_C \sum_{l=1}^{k}\sum_{C(i)=l}||x_i-m_l||^2
$$
就是说在类中心确定的情况下，将每个样本分到一个类中，使样本和其所属类的中心之间的距离总和最小。求解结果，将每个样本指派到与其最近的中心$m_l$的类$G_l$中。

然后，对给定的划分$C$，再求各个类的中心$(m_1, m_2, ... , m_k)$，使得目标函数极小化：
$$
\mathop{\text{min}}_{m_1,...,m_k} \sum_{l=1}^{k}\sum_{C(i)=l}||x_i-m_l||^2
$$
就是说在划分确定的情况下，使样本和其所属类的中心之间的距离总和最小。求解结果，对于每个包含$n_l$个样本的类$G_l$，更新其均值$m_l$：
$$
m_l=\frac{1}{n_l}\sum_{G(i)=l}x_i,\quad l=1,...,k
$$
重复以上两个步骤，直到划分不再改变，得到聚类结果。现将k均值聚类算法叙述如下：

**k均值聚类算法**

输入：$n$个样本的集合$X$；

输出：样本集合的聚类$C^*$。

（1）初始化。令$t=0$，随机选择$k$个样本点作为初始聚类中心
$$
m^{(0)}=\left(m_1^{(0)},...,m_l^{(0)},...,m_k^{(0)}\right)
$$
（2）对样本进行聚类。对固定的类中心
$$
m^{(t)}=\left(m_1^{(t)},...,m_l^{(t)},...,m_k^{(t)}\right)
$$
，其中$m_l^{(t)}$为类$G_l$的中心，计算每个样本到类中心的距离，将每个样本指派到与其最近的中心的类中，构成聚类结果$C^{(t)}$。

（3）计算新的类中心。对聚类结果$C^{(t)}$，计算当前各个类中的样本的均值，作为新的类中心
$$
m^{(t+1)}=\left(m_1^{(t+1)},...,m_l^{(t+1)},...,m_k^{(t+1)}\right)
$$
（4）如果迭代收敛或符合停止条件，输出
$$
C^*=C^{(t)}
$$
否则，令$t=t+1$，返回步（2）。

**k均值聚类算法的复杂度是$O(mnk)$，其中$m$是样本维数，$n$是样本个数，$k$是类别个数。在实际代码中，还要乘上迭代次数$iter$**。

例子：

给定含有5个样本的集合
$$
\begin{aligned}
X=
\begin{bmatrix}
0 & 0 & 1 & 5 & 5\\ 
2 & 0 & 0 & 0 & 2
\end{bmatrix}
\end{aligned}
$$
试用k均值聚类算法将样本聚到2个类中。

解：按照上述算法，

（1）选择两个样本点作为类的中心。假设选择
$$
\begin{aligned}
&m_1^{(0)}=x_1=(0,2)^T\\
&m_2^{(0)}=x_2=(0,0)^T
\end{aligned}
$$
（2）以$m_1^{(0)},m_2^{(0)}$为类$G_1^{(0)},G_2^{(0)}$的中心，计算$x_3=(1,0)^T,x_4=(5,0)^T,x_5=(5,2)^T$与$m_1^{(0)}=(0,2)^T,m_2^{(0)}=(0,0)^T$的欧式距离平方。

对$x_3 = (1, 0)^T$，$d(x_3,m_1^{(0)})=5,d(x_3,m_2^{(0)})=1$，将$x_3$分到类$G_2^{(0)}$。

对$x_4= (5, 0)^T$，$d(x_4,m_1^{(0)})=29,d(x_4,m_2^{(0)})=25$，将$x_4$分到类$G_2^{(0)}$。

对$x_5= (5, 2)^T$，$d(x_5,m_1^{(0)})=25,d(x_5,m_2^{(0)})=29$，将$x_5$分到类$G_1^{(0)}$。

（3）得到新的类$G_1^{(1)}=\{x_1,x_5\},G_2^{(1)}=\{x_2,x_3,x_4\}$，计算类的中心$m_1^{(1)}, m_2^{(1)}$：
$$
m_1^{(1)}=(2.5,2.0)^T,m_2^{(1)}=(2,0)^T
$$
（4）重复步骤（2）和步骤（3）。

将$x_1$分到类$G_1^{(1)}$，将$x_2$分到类$G_2^{(1)}$，将$x_3$分到类$G_2^{(1)}$，将$x_4$分到类$G_2^{(1)}$，将$x_5$分到类$G_1^{(1)}$。

得到新的类$G_1^{(2)}=\{x_1,x_5\},G_2^{(2)}=\{x_2,x_3,x_4\}$

由于得到的新的类没有改变，聚类停止。得到聚类结果：
$$
G_1^*=\{x_1,x_5\},G_2^*=\{x_2,x_3,x_4\}
$$

# 算法特性

## 总体特点

k均值聚类有以下特点：基于划分的聚类方法：类别数$k$事先指定；以欧式距离平方表示样本之间的距离，以中心或样本的均值表示类别；以样本和其他所属类的中心之间的距离的总和为最优化的目标函数；得到的类别是平坦的、非层次化的；算法是迭代算法，不能保证得到全局最优。

## 收敛性

k均值聚类属于启发式方法，不能保证收敛到全局最优，**初始中心的选择会直接影响聚类结果**。注意，类中心在聚类的过程中会发生移动，但是往往不会移动太大，因为在每一步，样本被分到与其最近的中心的类中。

## 初始类的选择

选择不同的初始中心，会得到不同的聚类结果。针对上面的例子，如果改变两个类的初始中心，比如选择
$$
m_1^{(0)}=x_1,m_2^{(0)}=x_5
$$
那么$x_2$，$x_3$会分到$G_1^{(0)}$，$x_4$会分到$G_2^{(0)}$，形成聚类结果$G_1^{(1)}=\{x_!,x_2,x_3\},\ G_2^{(1)}=\{x_4,x_5\}$，中心是$m_1^{(1)}=(0.33,0.67)^T,m_2^{(1)}=(5,1)^T$。继续迭代，聚类结果仍然是$G_1^{(1)}=\{x_!,x_2,x_3\},\ G_2^{(1)}=\{x_4,x_5\}$。聚类停止。

初始中心的选择，比如可以用层次聚类对样本进行聚类，得到$k$个类时停止。然后从每个类中选择一个与中心距离最近的点。

## 类别个数k的选择

k均值聚类中的类别数$k$值需要预先指定，而在实际应用中最优的$k$值是不知道的，解决这个问题的一个方法是尝试用不同的$k$值聚类，检验各自的到的聚类结果的质量，推测最优的$k$值。聚类结果的质量可以用类的平均直径来衡量。一般地，类别数目变小时，平均直径会增加；类别数目变大超过某个值以后，平均直径会不变；而这个值正是最优的$k$值。下图说明类别数与平均直径的关系。实验时，可以采用二分查找，快速找到最优的$k$值。

下图是聚类的类别数和平均直径的关系。

![KMeans-classes-and-diameter](pic/KMeans-classes-and-diameter.png)

# 代码实践

## sklearn中的KMeans实现

这个简单的例子来自[`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)。

```python
>>> from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> kmeans.predict([[0, 0], [12, 3]])
array([1, 0], dtype=int32)
>>> kmeans.cluster_centers_
array([[10.,  2.],
       [ 1.,  2.]])
```

## word2vec中KMeans实现

### word2vec中的KMeans源码摘录

下面这段代码是从word2vec.c中的源码摘录出来的KMeans部分，我自己改了一些变量的名称，还加了一些注释，方便看懂。

```c
// Run K-means on the word vectors
int cluster_count = classes, iter = 10, closeid; // cluster_count: clusterCount
int *center_count = (int *)malloc(classes * sizeof(int)); //属于每个中心的单词个数
int *vocab_cluster_id = (int *)calloc(vocab_size, sizeof(int)); // 存放每个单词指派的中心id
real closev, x;
real *center_vector = (real *)calloc(classes * layer1_size, sizeof(real)); //存放每个中心的向量表示
for (a = 0; a < vocab_size; a++) vocab_cluster_id[a] = a % cluster_count; //随机指派每个单词的中心
for (a = 0; a < iter; a++) { //一共迭代iter轮
    for (b = 0; b < cluster_count * layer1_size; b++) center_vector[b] = 0;
    for (b = 0; b < cluster_count; b++) center_count[b] = 1;
    for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) center_vector[layer1_size * vocab_cluster_id[c] + d] += syn0[c * layer1_size + d]; //将属于每个中心的点的每个坐标相加
        center_count[vocab_cluster_id[c]]++; // 分别计算属于每个中心的点个数
    }
    for (b = 0; b < cluster_count; b++) { //更新每个中心的向量表示
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
            center_vector[layer1_size * b + c] /= center_count[b];
            closev += center_vector[layer1_size * b + c] * center_vector[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) center_vector[layer1_size * b + c] /= closev;
    }
    for (c = 0; c < vocab_size; c++) { //更新每个样本的中心
        closev = -10;
        closeid = 0;
        for (d = 0; d < cluster_count; d++) {
            x = 0;
            for (b = 0; b < layer1_size; b++) x += center_vector[layer1_size * d + b] * syn0[c * layer1_size + b];
            if (x > closev) { //选出与单词表示最相近的中心
                closev = x;
                closeid = d;
            }
        }
        vocab_cluster_id[c] = closeid;
    }
}
// Save the K-means classes
for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, vocab_cluster_id[a]);
free(center_count);
free(center_vector);
free(vocab_cluster_id);
```

### 对embdding进行KMeans聚类

word2vec训练出来的embedding结果，以二进制进行存储，其格式为

```
id vec[0] vec[1] ... vec[end]
```

注意：vec[0] vec[1] ... vec[end]需要以二进制格式存储。

将下面KMeans.c文件进行编译，然后用下面的sh脚本运行。

这里面的向量是经过归一化的，其中心向量也经过了归一化，所以其实算的是cos距离，是球面聚类。

```c
// KMeans.c
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <stdlib.h> // mac os x
// #include <malloc.h>

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

typedef float real;                      // Precision of float numbers

void KMeans(const char *vocab, const float *syn0, const int vocab_size, const int layer1_size, const int classes, const char *output_file_name) {
  long a, b, c, d;
  FILE *fo;
  fo = fopen(output_file_name, "wb");
  // Run K-means on the word vectors
  int clcn = classes, iter = 10, closeid;
  int *centcn = (int *)malloc(classes * sizeof(int));
  int *cl = (int *)calloc(vocab_size, sizeof(int));
  real closev, x;
  real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
  char *word = (char *)malloc(max_w * sizeof(char));

  for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
  for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < vocab_size; c++) {
      for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
      centcn[cl[c]]++;
    }
    for (b = 0; b < clcn; b++) {
      closev = 0;
      for (c = 0; c < layer1_size; c++) {
        cent[layer1_size * b + c] /= centcn[b];
        closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
      }
      closev = sqrt(closev);
      for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
    }
    for (c = 0; c < vocab_size; c++) {
      closev = -10;
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
        if (x > closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }
  // Save the K-means classes
  for (a = 0; a < vocab_size; a++) {
    memcpy(word, vocab + a * max_w, max_w);
    fprintf(fo, "%s %d\n", word, cl[a]);
  }
  free(centcn);
  free(cent);
  free(cl);
}

int main(int argc, char **argv) {
  FILE *f;
  float len;
  long long words, size, a, b;
  char vec_file_name[max_size], output_file_name[max_size];
  float *M;
  char *vocab;
  if (argc < 4) {
    printf("Usage: ./KMeans vec_file output_file classes\n");
    return 0;
  }
  strcpy(vec_file_name, argv[1]);
  strcpy(output_file_name, argv[2]);
  int classes = atoi(argv[3]);

  f = fopen(vec_file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));

  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  KMeans(vocab, M, words, size, classes, output_file_name);

  return 0;
}
```

用于编译KMeans.c文件的makefile文件如下，用`make`命令编译。

```makefile
SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: KMeans

KMeans : KMeans.c
    $(CC) KMeans.c -o ${BIN_DIR}/KMeans $(CFLAGS)
    chmod +x ${SCRIPTS_DIR}/*.sh

clean:
    pushd ${BIN_DIR} && rm -rf KMeans; popd
```

在shell脚本中确定输入文件，输出文件，以及聚类的类别个数，下面的sh中指定了聚类类别为30。

```shell
#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

VECTOR_DATA=$DATA_DIR/final_item_vec_v1.bin
OUTPUT_DATA=$DATA_DIR/KMeans_v1.txt

set -x
$BIN_DIR/KMeans $VECTOR_DATA $OUTPUT_DATA 30
```

运行shell文件，则会输出聚类结果文件。至此聚类完成。经过测试，量级在5百万的数据，可在3分钟内聚类完成。

然后如果需要用sql查询做一些分析，就考虑将生成的文件传到hdfs中

```shell
hadoop fs -copyFromLocal KMeans_v1.txt hdfs://nameservice/user/machine_learning/lu/tmp/
```

然后导入sql

```sql
LOAD DATA INPATH 'hdfs://nameservice/user/machine_learning/lu/tmp/' OVERWRITE INTO TABLE machine_learning.tp_embedding_cluster;
```

就可以做想做的分析了

# 参考资料

* 《统计学习方法第二版-李航》

本文的理论部分摘抄自李航书的对应章节。

* [word2vec中k-means学习笔记](https://blog.csdn.net/zhoubl668/article/details/24320153)

“word2vec中的KMeans源码摘录”参考此文。

