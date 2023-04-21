# Linux常用命令

* [返回上层目录](../linux.md)
* [查看文件](#)
  * [head/tail/sed查看或截取超大文件](#head/tail/sed查看或截取超大文件)
  * [less直接查看超大文件](#less直接查看超大文件)
  * [cat文件内容打印到屏幕](#cat文件内容打印到屏幕)
  * [tree展示目录结构](#tree展示目录结构)
* [移动/传输文件](#移动/传输文件)
  * [cp复制文件](#cp复制文件)
  * [mv移动文件/重命名](#mv移动文件/重命名)
  * [scp本地文件上传到开发机](#scp本地文件上传到开发机)
* [压缩文件](#压缩文件)
  * [zip压缩文件](#zip压缩文件)
  * [tar压缩文件](#tar压缩文件)
* [文本处理](#文本处理)
  * [wc计算字数行数](#wc计算字数行数)
  * [awk文本分析](#awk文本分析)
  * [sort文本排序](#sort文本排序)
* [磁盘](#磁盘)
  * [df查看磁盘分区空间](#df查看磁盘分区空间)
  * [du查看当前目录的总大小](#du查看当前目录的总大小)
  * [ln软链接](#ln软链接)
* [权限](#权限)
  * [chmod更改文件权限](#chmod更改文件权限)
  * [chown更改文件拥有者](#chown更改文件拥有者)
* [终端/后台](#终端/后台)
  * [nohup后台运行](#nohup后台运行)
  * [screen会话窗口保留](#screen会话窗口保留)
* [过滤/参数](#过滤/参数)
  * [grep关键字查找](#grep关键字查找)
  * [xargs管道传参](#xargs管道传参)
* [定时任务](#定时任务)
  * [crontab定时运行](#crontab定时运行)



# 查看文件

## ls显示文件夹内容

查看文件夹中的文件总数量

```shell
ls -l | grep "^-" | wc -l
```

## head/tail/sed查看或截取超大文件

Linux下打开超大文件方法
在Linux下用VIM打开大小几个G、甚至几十个G的文件时，是非常慢的。

这时，我们可以利用下面的方法分割文件，然后再打开。

1 查看文件的前多少行

```shell
head -10000 /var/lib/mysql/slowquery.log > temp.log
```

上面命令的意思是：把slowquery.log文件前10000行的数据写入到temp.log文件中。

或者查看：

```
head -10000 /var/lib/mysql/slowquery.log
```

2 查看文件的后多少行

```shell
tail -10000 /var/lib/mysql/slowquery.log > temp.log
```

上面命令的意思是：把slowquery.log文件后10000行的数据写入到temp.log文件中。

3 查看文件的几行到几行

```shell
sed -n ‘10,10000p’ /var/lib/mysql/slowquery.log > temp.log
```

上面命令的意思是：把slowquery.log文件第10到10000行的数据写入到temp.log文件中。



参考资料：

* [Linux打开超大文件方法](https://blog.csdn.net/liurwei/article/details/82225245)



## less直接查看超大文件

less 与 more 类似，但使用 less 可以随意浏览文件，而 more 仅能向前移动，却不能向后移动，而且 **less 在查看之前不会加载整个文件**。

```shell
less abc.txt
```

参考资料：

* [Linux less命令](https://www.runoob.com/linux/linux-comm-less.html)



## cat文件内容打印到屏幕

cat 命令用于连接文件并**直接打印到标准输出设备上**。

```shell
cat abc.txt
```

参考资料：

* [Linux cat命令](https://www.runoob.com/linux/linux-comm-cat.html)



## tree展示目录结构

安装：

```shell
apt-get install tree
```

使用：

```shell
tree [要展示的文件夹路径] -d -L 1
```

解释：

tree: 显示目录树；

-d: 只显示目录；

-L 1: 选择显示的目录深度为1，只显示一层深度。

```shell
tree ~/Desktop/saved_models
```

结果为：

```
saved_models
├── cv_sod
│   └── 1609392632
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
└── lstm
    └── 1609392632
        ├── assets
        │   ├── vocab.labels.txt
        │   └── vocab.words.txt
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index

7 directories, 8 files
```

参考资料：

* [命令展示该目录下的所有子目录及文件结构 tree](https://www.cnblogs.com/oxspirt/p/6278004.html)

## find查找文件

查找文件，从当前目录查找所有后缀名为`md`的文件

```shell
find ./ -name "*.md"
```

找出来的满足要求的文件，可通过`xargs`管道传参来进行后续处理，如：

```shell
find ./ -name "*.md" | xargs sed -i "" "s/abcd/efgh/g"
```

把每一行中所有`abcd`字符串转为`efgh`。

## file显示文件信息

```shell
file libxxx.so
# libxxx.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=a5fb794c5f5a4acdc8d365b4fa0f41a45807f49b, with debug_info, not stripped
```

显示该linux库文件适用于x86-64架构，和arm架构不兼容。

# 移动/传输文件

## cp复制文件

只复制部分文件，就使用 `grep -v`来 过滤掉不想复制的文件/文件夹，文件名之间用`\|`分割。 

```shell
cp -r `ls| grep -v 'data\|input\|output\|tmp'` ../deep_1f/
```



## mv移动文件/重命名

在一些情况下，我们需要将很多个文件同时移动到一个指定的目录下，如果一个一个移动，那太蛋疼了。

今天用man mv查到一个选项-t，是指定目标文件夹，就是我们所要将文件移至的文件夹，很省事儿，分享给大家。

比如当前目录下有a.dir  b.dir   c.dir  1.txt  2.txt  des.dir.我们现在要将a.dir  b.dir   c.dir  1.txt  2.txt这几个文件移动到des.dir 目录下。

执行：

```shell
mv  a.dir  b.dir   c.dir  1.txt  2.txt  -t  des.dir
```

就可以一次将这些文件移动到des.dir下。



参考资料：

* [linux命令mv同时移动多个文件至一个目录](https://jingyan.baidu.com/article/d8072ac4686a1dec95cefd8d.html)



## scp本地文件上传到开发机

```shell
scp /tmp/Anaconda3-2019.03-Linux-x86_64.sh  ml004.dx:/tmp/
```

同样也可以将文件从开发机下载到本地。

## relink找出符号链接指向位置

readlink是linux用来找出符号链接所指向的位置。

```shell
例1：
readlink -f /usr/bin/awk
结果：
/usr/bin/gawk #因为/usr/bin/awk是一个软连接，指向gawk
例2：
readlink -f /home/software/log
/home/software/log  #如果没有链接，就显示自己本身的绝对路径
```

该功能常常和dirname $0连用：

```
path=$(dirname $0)
path2=$(readlink -f $path)
或者一步到位：
BASE_DIR=$(readlink -f `dirname "$0"`)
```

`dirname $0`只是获取的当前脚本的相对路径.

解释：
`readlink -f $path` 如果​\$path没有链接，就显示自己本身的绝对路径



# 压缩文件

## zip压缩文件

```js
-d 从压缩文件内删除指定的文件。
-q 不显示指令执行过程。
-r 递归处理，将指定目录下的所有文件和子目录一并处理。
-v 显示指令执行过程或显示版本信息。
```

实例：

* 将/home/html/ 这个目录下所有文件和文件夹打包为当前目录下的 html.zip：

  ```shell
  zip -q -r html.zip /home/html
  ```

* 如果在我们在 /home/html 目录下，可以执行以下命令：

  ```shell
  zip -q -r html.zip *
  ```

* 从压缩文件 cp.zip 中删除文件 a.c

  ```shell
  zip -dv cp.zip a.c
  ```



对于zip、z01、z02等格式的，需要先将这些压缩包合并后再解压，具体方法如下：

先用zip将其中的文件进行合并，然后再解压，可以获得所有文件。

```shell
# image.zip image.z01 image.z02 ...
zip -s 0 image.zip --out full.zip
unzip full.zip
```



参考资料：

* [Linux zip命令](https://www.runoob.com/linux/linux-comm-zip.html)



## tar压缩文件

- 命令选项：

```
-z(gzip)      用gzip来压缩/解压缩文件
-j(bzip2)     用bzip2来压缩/解压缩文件
-v(verbose)   详细报告tar处理的文件信息
-c(create)    创建新的档案文件
-x(extract)   解压缩文件或目录
-f(file)      使用档案文件或设备，这个选项通常是必选的。
```

- 命令举例：

```shell
#打包
tar -vcf buodo.tar buodo

#拆包
tar -vxf buodo.tar

#打包并压缩
tar -zvcf buodo.tar.gz buodo
tar -jvcf buodo.tar.bz2 buodo 

#解压并拆包
tar -zvxf buodo.tar.gz 
tar -jvxf buodo.tar.bz2
```

参考资料：

- [linux 压缩和解压缩命令gz、tar、zip、bz2](https://blog.csdn.net/capecape/article/details/78548723)



# 文本处理

## wc计算字数行数

利用wc指令我们可以计算文件的Byte数、字数、或是列数，若不指定文件名称、或是所给予的文件名为"-"，则wc指令会从标准输入设备读取数据。

```shell
wc [-clw][--help][--version][文件...]
```

**参数**：

- -c或--bytes或--chars 只显示Bytes数。
- -l或--lines 只显示行数。
- -w或--words 只显示字数。
- --help 在线帮助。
- --version 显示版本信息。

实例：

在默认的情况下，wc将计算指定文件的行数、字数，以及字节数。使用的命令为：

```shell
wc testfile
```

使用 wc统计，结果如下：

```shell
wc testfile  # testfile文件的统计信息  
# 3 92 598 testfile
# testfile文件的行数为3、单词数92、字节数598 
```

其中，3 个数字分别表示testfile文件的行数、单词数，以及该文件的字节数。

如果想同时统计多个文件的信息，例如同时统计testfile、testfile_1、testfile_2，可使用如下命令：

```shell
wc testfile testfile_1 testfile_2   #统计三个文件的信息 
```

输出结果如下：

```shell
#统计三个文件的信息  
wc testfile testfile_1 testfile_2
#3 92 598 testfile
# 第一个文件行数为3、单词数92、字节数598  
# 9 18 78 testfile_1
# 第二个文件的行数为9、单词数18、字节数78  
# 3 6 32 testfile_2
# 第三个文件的行数为3、单词数6、字节数32  
# 15 116 708 总用量
# 三个文件总共的行数为15、单词数116、字节数708 
```



参考资料：

* [Linux wc命令](https://www.runoob.com/linux/linux-comm-wc.html)



## awk文本分析

AWK是一种处理文本文件的语言，是一个强大的文本分析工具。

之所以叫AWK是因为其取了三位创始人 Alfred Aho，Peter Weinberger, 和 Brian Kernighan 的 Family Name 的首字符。

**语法**

```shell
awk [选项参数] 'script' var=value file(s)
或
awk [选项参数] -f scriptfile var=value file(s)
```

**选项参数说明：**

- -F fs or --field-separator fs
  指定输入文件折分隔符，fs是一个字符串或者是一个正则表达式，如-F:。
- -v var=value or --asign var=value
  赋值一个用户定义变量。
- -f scripfile or --file scriptfile
  从脚本文件中读取awk命令。
- -mf nnn and -mr nnn
  对nnn值设置内在限制，-mf选项限制分配给nnn的最大块数目；-mr选项限制记录的最大数目。这两个功能是Bell实验室版awk的扩展功能，在标准awk中不适用。
- -W compact or --compat, -W traditional or --traditional
  在兼容模式下运行awk。所以gawk的行为和标准的awk完全一样，所有的awk扩展都被忽略。
- -W copyleft or --copyleft, -W copyright or --copyright
  打印简短的版权信息。
- -W help or --help, -W usage or --usage
  打印全部awk选项和每个选项的简短说明。
- -W lint or --lint
  打印不能向传统unix平台移植的结构的警告。
- -W lint-old or --lint-old
  打印关于不能向传统unix平台移植的结构的警告。
- -W posix
  打开兼容模式。但有以下限制，不识别：/x、函数关键字、func、换码序列以及当fs是一个空格时，将新行作为一个域分隔符；操作符**和**=不能代替^和^=；fflush无效。
- -W re-interval or --re-inerval
  允许间隔正则表达式的使用，参考(grep中的Posix字符类)，如括号表达式[[:alpha:]]。
- -W source program-text or --source program-text
  使用program-text作为源代码，可与-f命令混用。
- -W version or --version
  打印bug报告信息的版本。



**基本用法**

```shell
# 行匹配语句 awk '' 只能用单引号
awk '{[pattern] action}' {filenames}

#-F相当于内置变量FS, 指定分割字符
awk -F

# 设置变量
awk -v

# 使用脚本
awk -f {awk脚本} {文件名}
```

例如：

log.txt文本内容如下：

```js
2 this is a test
3 Are you like awk
This's a test
10 There are orange,apple,mongo
```

用法如下：

```shell
# 每行按空格或TAB分割，输出文本中的1、4项
awk '{print $1,$4}' log.txt

# 格式化输出
awk '{printf "%-8s %-10s\n",$1,$4}' log.txt

# 使用","分割
awk -F, '{print $1,$2}' log.txt

# 或者使用内建变量
awk 'BEGIN{FS=","} {print $1,$2}' log.txt

# 使用多个分隔符.先使用空格分割，然后对分割结果再使用","分割
awk -F '[ ,]'  '{print $1,$2,$5}' log.txt

awk -v a=1 '{print $1,$1+a}' log.txt
# 2 3

awk -v a=1 -v b=s '{print $1,$1+a,$1b}' log.txt
# 2 3 2s
```



**运算符**

| 运算符                  | 描述                             |
| ----------------------- | -------------------------------- |
| = += -= *= /= %= ^= **= | 赋值                             |
| ?:                      | C条件表达式                      |
| \|\|                    | 逻辑或                           |
| &&                      | 逻辑与                           |
| ~ 和 !~                 | 匹配正则表达式和不匹配正则表达式 |
| < <= > >= != ==         | 关系运算符                       |
| 空格                    | 连接                             |
| + -                     | 加，减                           |
| * / %                   | 乘，除与求余                     |
| + - !                   | 一元加，减和逻辑非               |
| ^ ***                   | 求幂                             |
| ++ --                   | 增加或减少，作为前缀或后缀       |
| $                       | 字段引用                         |
| in                      | 数组成员                         |

过滤第一列大于2的行

```shell
awk '$1>2' log.txt
# 输出
# 3 Are you like awk
# This's a test
# 10 There are orange,apple,mongo
```

过滤第一列等于2的行

```shell
awk '$1==2 {print $1,$3}' log.txt
# 输出
# 2 is
```

过滤第一列大于2并且第二列等于'Are'的行

```shell
awk '$1>2 && $2=="Are" {print $1,$2,$3}' log.txt
# 输出
# 3 Are you
```



**内建变量**

| 变量        | 描述                                                       |
| ----------- | ---------------------------------------------------------- |
| $n          | 当前记录的第n个字段，字段间由FS分隔                        |
| $0          | 完整的输入记录                                             |
| ARGC        | 命令行参数的数目                                           |
| ARGIND      | 命令行中当前文件的位置(从0开始算)                          |
| ARGV        | 包含命令行参数的数组                                       |
| CONVFMT     | 数字转换格式(默认值为%.6g)ENVIRON环境变量关联数组          |
| ERRNO       | 最后一个系统错误的描述                                     |
| FIELDWIDTHS | 字段宽度列表(用空格键分隔)                                 |
| FILENAME    | 当前文件名                                                 |
| FNR         | 各文件分别计数的行号                                       |
| FS          | 字段分隔符(默认是任何空格)                                 |
| IGNORECASE  | 如果为真，则进行忽略大小写的匹配                           |
| NF          | 一条记录的字段的数目                                       |
| NR          | 已经读出的记录数，就是行号，从1开始                        |
| OFMT        | 数字的输出格式(默认值是%.6g)                               |
| OFS         | 输出记录分隔符（输出换行符），输出时用指定的符号代替换行符 |
| ORS         | 输出记录分隔符(默认值是一个换行符)                         |
| RLENGTH     | 由match函数所匹配的字符串的长度                            |
| RS          | 记录分隔符(默认是一个换行符)                               |
| RSTART      | 由match函数所匹配的字符串的第一个位置                      |
| SUBSEP      | 数组下标分隔符(默认值是/034)                               |

```shell
awk 'BEGIN{printf "%4s %4s %4s %4s %4s %4s %4s %4s %4s\n","FILENAME","ARGC","FNR","FS","NF","NR","OFS","ORS","RS";printf "---------------------------------------------\n"} {printf "%4s %4s %4s %4s %4s %4s %4s %4s %4s\n",FILENAME,ARGC,FNR,FS,NF,NR,OFS,ORS,RS}'  log.txt
# FILENAME ARGC  FNR   FS   NF   NR  OFS  ORS   RS
# ---------------------------------------------
# log.txt    2    1         5    1
# log.txt    2    2         5    2
# log.txt    2    3         3    3
# log.txt    2    4         4    4
```



**使用正则，字符串匹配**

**~ 表示模式开始。// 中是模式。**

```shell
# 输出第二列包含 "th"，并打印第二列与第四列
awk '$2 ~ /th/ {print $2,$4}' log.txt
# this a

# 输出包含"re" 的行
awk '/re/ ' log.txt
# 3 Are you like awk
# 10 There are orange,apple,mongo
```



**忽略大小写**

```shell
awk 'BEGIN{IGNORECASE=1} /this/' log.txt
# 2 this is a test
# This's a test
```



**模式取反**

```shell
awk '$2 !~ /th/ {print $2,$4}' log.txt
# Are like
# a
# There orange,apple,mongo
awk '!/th/ {print $2,$4}' log.txt
# Are like
# a
# There orange,apple,mongo
```



**awk脚本**

关于awk脚本，我们需要注意两个关键词BEGIN和END。

- BEGIN{ 这里面放的是执行前的语句 }
- END {这里面放的是处理完所有的行后要执行的语句 }
- {这里面放的是处理每一行时要执行的语句}

假设有这么一个文件（学生成绩表）：

```shell
cat score.txt
# Marry   2143 78 84 77
# Jack    2321 66 78 45
# Tom     2122 48 77 71
# Mike    2537 87 97 95
# Bob     2415 40 57 62
```

我们的awk脚本如下：

```shell
$ cat cal.awk
#!/bin/awk -f
#运行前
BEGIN {
    math = 0
    english = 0
    computer = 0
 
    printf "NAME    NO.   MATH  ENGLISH  COMPUTER   TOTAL\n"
    printf "---------------------------------------------\n"
}
#运行中
{
    math+=$3
    english+=$4
    computer+=$5
    printf "%-6s %-6s %4d %8d %8d %8d\n", $1, $2, $3,$4,$5, $3+$4+$5
}
#运行后
END {
    printf "---------------------------------------------\n"
    printf "  TOTAL:%10d %8d %8d \n", math, english, computer
    printf "AVERAGE:%10.2f %8.2f %8.2f\n", math/NR, english/NR, computer/NR
}
```

我们来看一下执行结果：

```shell
$ awk -f cal.awk score.txt
NAME    NO.   MATH  ENGLISH  COMPUTER   TOTAL
---------------------------------------------
Marry  2143     78       84       77      239
Jack   2321     66       78       45      189
Tom    2122     48       77       71      196
Mike   2537     87       97       95      279
Bob    2415     40       57       62      159
---------------------------------------------
  TOTAL:       319      393      350
AVERAGE:     63.80    78.60    70.00
```



**awk关联数组**

在awk中，数组都是关联数组.

所谓关联数组就是每一个数组元素实际都包含两部分：key和value，类似python里面的字典。在awk中数组之间是无序的，一个数组的key值是数值，例如1，2，3，并不代表该数组元素在数组中的出现的位置。

awk中的数组有以下特性：

1. 数组无需定义，直接使用
2. 数组自动扩展
3. 下标可以是数值型或者字符串型

元素赋值：

```shell
arr[0]=123
arr[“one”]=123
```

数组长度：

```shell
length(arr)
```

data.txt:

```js
a1 b1 1
a2 b1 1
a1 b2 1
a3 b2 1
```

统计第一列和第二列中的去重数量：

```shell
awk -F"\t"  '{hash1[$1]+=1; hash2[$2]+=1}END{print length(hash1) "\t" length(hash2)}' data.txt
```

计算频率：

data.txt:

```js
a1 1
a2 2
a1 5
a3 3
```

shell代码：

```shell
awk -F',' '{all_samp_num+=$2; dict[$1]=$2} END{for(i in dict){printf("%s,%.6f\n", i, dict[i]/all_samp_num)}}' ./data.txt > ./res.txt
```



**另外一些实例**

计算文件大小

```shell
ls -l *.txt | awk '{sum+=$6} END {print sum}'
```



从文件中找出长度大于80的行

```shell
awk 'length>80' log.txt
```



参考资料：

* [Linux awk 命令](https://www.runoob.com/linux/linux-comm-awk.html)
* [awk关联数组](https://blog.csdn.net/qinyushuang/article/details/50342875)



## sort文本排序

**sort命令**是在Linux里非常有用，它将文件进行排序，并将排序结果标准输出。

sort的-n、-r、-k、-t选项的使用：

```shell
[root@mail text]# cat sort.txt
AAA:BB:CC
aaa:30:1.6
ccc:50:3.3
ddd:20:4.2
bbb:10:2.5
eee:40:5.4
eee:60:5.1

#将BB列按照数字从小到大顺序排列：
[root@mail text]# sort -nk 2 -t: sort.txt
AAA:BB:CC
bbb:10:2.5
ddd:20:4.2
aaa:30:1.6
eee:40:5.4
ccc:50:3.3
eee:60:5.1

#将CC列数字从大到小顺序排列：
[root@mail text]# sort -nrk 3 -t: sort.txt
eee:40:5.4
eee:60:5.1
ddd:20:4.2
ccc:50:3.3
bbb:10:2.5
aaa:30:1.6
AAA:BB:CC

# -n是按照数字大小排序，-r是以相反顺序，-k是指定需要爱排序的栏位，-t指定栏位分隔符为冒号
```



参考资料：

* [sort命令](https://man.linuxde.net/sort)

## sed批量文本处理

sed命令可批量处理文本，比如：

```shell
# ==================批量替换内容==================

# 将每行开头为$$换成vvvv
find ./ -name "*.md" | xargs sed -i "" "s/^[$][$]$/vvvv/g"
# 将其余的$$换成$
find ./ -name "*.md" | xargs sed -i "" "s/[$][$]/\$/g"
# 将vvvv换回$$
find ./ -name "*.md" | xargs sed -i "" "s/vvvv/\$\$/g"
```

具体的规则是：

```
sed总结
sed元字符（正则表达式）
^ 匹配行开始，如：/^sed/匹配所有以sed开头的行。 
$ 匹配行结束，如：/sed$/匹配所有以sed结尾的行。 
. 匹配一个非换行符的任意字符，如：/s.d/匹配s后接一个任意字符，最后是d。 
* 匹配0个或多个字符，如：/*sed/匹配所有模板是一个或多个空格后紧跟sed的行。 
[] 匹配一个指定范围内的字符，如/[ss]ed/匹配sed和Sed。 
[^] 匹配一个不在指定范围内的字符，如：/[^A-RT-Z]ed/匹配不包含A-R和T-Z的一个字母开头，紧跟ed的行。 
\(..\) 匹配子串，保存匹配的字符，如s/\(love\)able/\1rs，loveable被替换成lovers。 
& 保存搜索字符用来替换其他字符，如s/love/**&**/，love这成**love**。
\< 匹配单词的开始，如:/\ 匹配单词的结束，如/love\>/匹配包含以love结尾的单词的行。 
x\{m\} 重复字符x，m次，如：/0\{5\}/匹配包含5个0的行。 
x\{m,\} 重复字符x，至少m次，如：/0\{5,\}/匹配至少有5个0的行。
x\{m,n\} 重复字符x，至少m次，不多于n次，如：/0\{5,10\}/匹配5~10个0的行。
```



参考资料：

* [LINUX正则表达式：SED用法详解](https://www.freesion.com/article/4929825348/)

# 磁盘

## df查看磁盘分区空间

df 以磁盘分区为单位查看文件系统，可以获取硬盘被占用了多少空间，目前还剩下多少空间等信息。

例如，我们使用df -h命令来查看磁盘信息， -h 选项为根据大小适当显示：

![df](pic/df.jpg)

显示内容参数说明：

- **Filesystem**：文件系统
- **Size**： 分区大小
- **Used**： 已使用容量
- **Avail**： 还可以使用的容量
- **Use%**： 已用百分比
- **Mounted on**： 挂载点　



参考资料：

* [Linux 查看磁盘空间](https://www.runoob.com/w3cnote/linux-view-disk-space.html)



## du查看当前目录的总大小

**du** 的英文原义为 **disk usage**，含义为显示磁盘空间的使用情况，用于查看当前目录的总大小。

例如查看当前目录的大小：

```shell
# du -sh
605M    .
```

**查看当前目录中一级目录的大小**：

```shell
# du -sh *
26G	anaconda3
517M	Anaconda3-2019.07-Linux-x86_64.sh
28G	crontab
0	fan.ze
308G	han.xu
130G	han.zefeng
324G	huang.dachun
575G	jiao.wei
101G	li.renjie
223G	li.sizhen
348G	liu.xingchen
0	li.yaozong
334G	lu.wei
46G	ma.jintao
107G	shi.hongye
420G	wang.yuan
316G	wu.baoxin
```

显示指定文件所占空间：

```shell
# du log2012.log 
300     log2012.log
```

方便阅读的格式显示test目录所占空间情况：

```shell
# du -h test
608K    test/test6
308K    test/test4
4.0K    test/scf/lib
4.0K    test/scf/service/deploy/product
4.0K    test/scf/service/deploy/info
12K     test/scf/service/deploy
16K     test/scf/service
4.0K    test/scf/doc
4.0K    test/scf/bin
32K     test/scf
8.0K    test/test3
1.3M    test
```

du 命令用于查看当前目录的总大小：

- -s：对每个Names参数只给出占用的数据块总数。
- -a：递归地显示指定目录中各文件及子目录中各文件占用的数据块数。若既不指定-s，也不指定-a，则只显示Names中的每一个目录及其中的各子目录所占的磁盘块数。
- -b：以字节为单位列出磁盘空间使用情况（系统默认以k字节为单位）。
- -k：以1024字节为单位列出磁盘空间使用情况。
- -c：最后再加上一个总计（系统默认设置）。
- -l：计算所有的文件大小，对硬链接文件，则计算多次。
- -x：跳过在不同文件系统上的目录不予统计。
- -h：以K，M，G为单位，提高信息的可读性。



**`du -h --max-depth=1`查看子文件夹大小**

实际上就是超过指定层数（1层）的目录后，予以忽略。

也可以用`du -sh *`



参考资料：

* [Linux 查看磁盘空间](https://www.runoob.com/w3cnote/linux-view-disk-space.html)

## ln软链接

它的功能是为某一个文件在另外一个位置建立一个同步的链接。

当我们需要在不同的目录，用到相同的文件时，我们不需要在每一个需要的目录下都放一个必须相同的文件，我们只要在某个固定的目录，放上该文件，然后在其它的目录下用ln命令链接（link）它就可以，不必重复的占用磁盘空间。

比如，`/mnt/disk4/`的空间特别大，而当前文件夹所在的磁盘空间不足，但是还想把文件放在当前文件夹下，那就可以建一个软连接，看起来是把文件放在当前文件夹的`data`目录下了，其实文件是存放在`/mnt/disk4/data/user/`中的。

```shell
ln -s /mnt/disk4/data/user/ data
```

参考资料：

* [Linux ln命令](https://www.runoob.com/linux/linux-comm-ln.html)

## mount硬盘或U盘挂载

挂载

```shell
sudo fdisk -l
sudo mount /dev/sda1 /mnt/mymount
```

卸载

```shell
# 通过设备名卸载
umount -v /dev/sda1  # /dev/sda1 umounted  
# 通过挂载点卸载
umount -v /mnt/mymount/  # /tmp/diskboot.img umounted 
```



参考资料：

* [mount,umount命令详解](https://blog.csdn.net/qq_42014600/article/details/90404249)





# 权限

权限简介：

Linux系统上对文件的权限有着严格的控制，用于如果相对某个文件执行某种操作，必须具有对应的权限方可执行成功。这也是Linux有别于Windows的机制，也是基于这个权限机智，Linux可以有效防止病毒自我运行，因为运行的条件是必须要有运行的权限，而这个权限在Linux是用户所赋予的。

Linux的文件权限有以下设定：

* Linux下文件的权限类型一般包括读，写，执行。对应字母为 r、w、x。
* Linux下权限的粒度有 拥有者(**U**ser)、群组(**G**roup)、其它以外的人(**O**thers)三种(这三种均为所有人**A**ll)。每个文件都可以针对三个粒度，设置不同的rwx(读写执行)权限。
* 通常情况下，一个文件只能归属于一个用户和组， 如果其它的用户想有这个文件的权限，则可以将该用户加入具备权限的群组，一个用户可以同时归属于多个组。

## chmod更改文件权限

Linux上通常使用chmod命令对文件的权限进行设置和更改。

使用格式

```shell
chmod [可选项] <mode> <file...>
```

```shell
参数说明：
 
[可选项]
  -c, --changes          like verbose but report only when a change is made (若该档案权限确实已经更改，才显示其更改动作)
  -f, --silent, --quiet  suppress most error messages  （若该档案权限无法被更改也不要显示错误讯息）
  -v, --verbose          output a diagnostic for every file processed（显示权限变更的详细资料）
       --no-preserve-root  do not treat '/' specially (the default)
       --preserve-root    fail to operate recursively on '/'
       --reference=RFILE  use RFILE's mode instead of MODE values
  -R, --recursive        change files and directories recursively （以递归的方式对目前目录下的所有档案与子目录进行相同的权限变更)
       --help		显示此帮助信息
       --version		显示版本信息
[mode] 
    权限设定字串，详细格式如下 ：
    [ugoa...][[+-=][rwxX]...][,...]，
    其中
    [ugoa...]
    u 表示该档案的拥有者，g 表示与该档案的拥有者属于同一个群体(group)者，o 表示其他以外的人，a 表示所有（包含上面三者）。
    [+-=]
    + 表示增加权限，- 表示取消权限，= 表示唯一设定权限。
    [rwxX]
    r 表示可读取，w 表示可写入，x 表示可执行，X 表示只有当该档案是个子目录或者该档案已经被设定过为可执行。
 	
[file...]
    文件列表（单个或者多个文件、文件夹）
```

范例：

设置所有用户可读取文件 a.conf

```shell
chmod ugo+r a.conf
或 
chmod a+r  a.conf
```

设置 c.sh 只有 拥有者可以读写及执行

```shell
chmod u+rwx c.sh
```

设置文件 a.conf 与 b.xml 权限为拥有者与其所属同一个群组 可读写，其它组可读不可写

```shell
chmod a+r,ug+w,o-w a.conf b.xml
```

设置当前目录下的所有档案与子目录皆设为任何人可读写

```shell
chmod -R a+rw *
```

**数字权限使用格式：**

在这种使用方式中，首先我们需要了解数字如何表示权限。 首先，我们规定 数字 4 、2 和 1表示读、写、执行权限（具体原因可见下节权限详解内容），即**r=4，w=2，x=1**。此时其他的权限组合也可以用其他的八进制数字表示出来，

如：

```shell
rwx = 4 + 2 + 1 = 7
rw = 4 + 2 = 6
rx = 4 +1 = 5
```

即

* 若要同时设置 rwx (可读写运行） 权限则将该权限位 设置 为 4 + 2 + 1 = 7

* 若要同时设置 rw- （可读写不可运行）权限则将该权限位 设置 为 4 + 2 = 6

* 若要同时设置 r-x （可读可运行不可写）权限则将该权限位 设置 为 4 +1 = 5

由上可以得出，每个属组的所有的权限都可以用一位八进制数表示，每个数字都代表了不同的权限（权值）。如 最高的权限为是7，代表可读，可写，可执行。

故如果我们将每个属组的权限都用八进制数表示，则文件的权限可以表示为三位八进制数

```shell
-rw------- =  600
-rw-rw-rw- =  666
-rwxrwxrwx =  777
```

上面我们提到，每个文件都可以针对三个粒度，设置不同的rwx(读写执行)权限。即我们可以用用三个8进制数字分别表示 拥有者 、群组 、其它组( u、 g 、o)的权限详情，并用chmod直接加三个8进制数字的方式直接改变文件权限。语法格式为：

```shell
chmod <abc> file...
```

其中

>a,b,c各为一个数字，分别代表User、Group、及Other的权限。
>
>相当于简化版的
>
>chmod u=权限, g=权限, o=权限 file...
>
>而此处的权限将用8进制的数字来表示User、Group、及Other的读、写、执行权限

范例：

* 设置所有人可以读写及执行

```shell
chmod 777 file  
# 等价于  chmod u=rwx,g=rwx,o=rwx file 或  chmod a=rwx file
```

设置拥有者可读写，其他人不可读写执行

```shell
chmod 600 file
# 等价于  chmod u=rw,g=---,o=--- file 或 chmod u=rw,go-rwx file
```



参考资料：

* [Linux权限详解（chmod、600、644、666、700、711、755、777、4755、6755、7755）](https://blog.csdn.net/u013197629/article/details/73608613)

## chown更改文件拥有者

linux/Unix 是多人多工作业系统，每个的文件都有拥有者（所有者），如果我们想变更文件的拥有者（利用 chown 将文件拥有者加以改变），**一般只有系统管理员(root)拥有此操作权限**，而普通用户则没有权限将自己或者别人的文件的拥有者设置为别人。

范例：

设置文件 d.key、e.scrt的拥有者设为 users 群体的 tom

```shell
chown tom:users file d.key e.scrt
```

设置当前目录下与子目录下的所有文件的拥有者为 users 群体的 James

```shell
chown -R James:users  *
```



参考资料：

* [Linux权限详解（chmod、600、644、666、700、711、755、777、4755、6755、7755）](https://blog.csdn.net/u013197629/article/details/73608613)



# 终端/后台

## nohup后台运行

后台运行

```shell
nohup python -u xxx.py 1>./log.txt 2>&1 &
```

python需要加-u，这样才会使得所有输出到输出到log里。

## screen会话窗口保留

在VPS中执行一些非常耗时的任务时（如下载，压缩，解压缩，编译，安装等），我们通常是单独开一个远程终端窗口来执行这个任务，且在任务执行过程中不能关闭这个窗口或者中断连接，否则正在执行的任务会被终止掉。而有了screen，我们可以在一个窗口中安装程序，然后在另一个窗口中下载文件，再在第三个窗口中编译程序，只需要一个SSH连接就可以同时执行这三个任务，还可以方便的在不同会话或窗口中切换，即使因为意外导致窗口关闭或者连接中断，也不会影响这三个任务的执行。

screen的说明相当复杂，反正我是看得头晕了。但事实上，我们只需要掌握下面五个命令就足够我们使用了：

* 创建一个名为test1的会话

```shell
screen -S test1
```

* 进入test1后输入下面的命令，退出test1会话，但会话中的任务会继续执行

```shell
screen -d
or
press Ctrl-A 和 D
```

* 列出所有会话

```shell
screen -ls
```

* 恢复名为test1的会话

```shell
screen -r test1
or
screen -r pid(number)
```

* 退出并彻底关闭当前窗口，会话中的任务也会被关闭

```shell
exit
or
Ctrl+D
```



参考资料：

* [screen 命令使用及示例](https://linux.cn/article-8215-1.html)
* [linux文件权限查看及修改-chmod ------入门的一些常识](https://blog.csdn.net/haydenwang8287/article/details/1753883)



# 过滤/参数

## grep关键字查找

```shell
# 从文件abc.txt中查找含有xxx字符串的行
grep "xxx" abc.txt
# 去掉含有xxx的行，例如xxx是异常信息，训练数据中可去除
grep -v "xxx" abc.txt
hadoop fs -text hdfs://x/* | grep -v INFO  > ./abc.txt
# 只保留xxx字符，即便某一行有多个xxx字符，这n个xxx字符会变为n行单个xxx
# 下列语句用于计算xxx在文本中出现的次数
grep -o "xxx" abc.txt | wc -l
```

例子：

```shell
# 过滤掉含有INFO关键字的行
hadoop fs -text hdfs://nameservice3/user/abc/* | grep -v INFO  > ./a.txt
```



## xargs管道传参

xargs 是一个强有力的命令，它能够捕获一个命令的输出，然后传递给另外一个命令。

之所以能用到这个命令，关键是由于很多命令不支持|管道来传递参数，而日常工作中有有这个必要，所以就有了 xargs 命令。

xargs 一般是和管道一起使用。

```shell
# kill含有down的前5条任务
ps -ef | grep down| head -5|awk '{print $2}' | xargs kill -9
# 展示含有down的前5条任务
ps -ef | grep down| head -5|awk '{print $2}' | xargs echo
```



# 任务

## crontab定时运行

Linux crontab是用来定期执行程序的命令。

当安装完成操作系统之后，默认便会启动此任务调度命令。

crond命令每分锺会定期检查是否有要执行的工作，如果有要执行的工作便会自动执行该工作。

而linux任务调度的工作主要分为以下两类：

- 1、系统执行的工作：系统周期性所要执行的工作，如备份系统数据、清理缓存
- 2、个人执行的工作：某个用户定期要做的工作，例如每隔10分钟检查邮件服务器是否有新信，这些工作可由每个用户自行设置

**语法**：

```shell
crontab [ -u user ] { -l | -r | -e }
```

**说明：**

crontab 是用来让使用者在固定时间或固定间隔执行程序之用，换句话说，也就是类似使用者的时程表。

-u user 是指设定指定 user 的时程表，这个前提是你必须要有其权限(比如说是 root)才能够指定他人的时程表。如果不使用 -u user 的话，就是表示设定自己的时程表。

**参数说明**：

- -e : 执行文字编辑器来设定时程表，内定的文字编辑器是 VI，如果你想用别的文字编辑器，则请先设定 VISUAL 环境变数来指定使用那个文字编辑器(比如说 setenv VISUAL joe)
- -r : 删除目前的时程表
- -l : 列出目前的时程表

时程表的格式如下：

```shell
f1 f2 f3 f4 f5 program
```

- 其中 f1 是表示分钟，f2 表示小时，f3 表示一个月份中的第几日，f4 表示月份，f5 表示一个星期中的第几天。program 表示要执行的程序。
- 当 f1 为 * 时表示每分钟都要执行 program，f2 为 * 时表示每小时都要执行程序，其余类推
- 当 f1 为 a-b 时表示从第 a 分钟到第 b 分钟这段时间内要执行，f2 为 a-b 时表示从第 a 到第 b 小时都要执行，其余类推
- 当 f1 为 */n 时表示每 n 分钟个时间间隔执行一次，f2 为 */n 表示每 n 小时个时间间隔执行一次，其余类推
- 当 f1 为 a, b, c,... 时表示第 a, b, c,... 分钟要执行，f2 为 a, b, c,... 时表示第 a, b, c...个小时要执行，其余类推

**另一种语法**：

```shell
crontab [ -u user ] file
```

使用者也可以将所有的设定先存放在文件中，用 crontab file 的方式来设定时程表。

举例：

* 每月每天每小时的第 0 分钟执行一次 /bin/ls

  ```shell
  0 * * * * /bin/ls
  ```

* 在 12 月内, 每天的早上 6 点到 12 点，每隔 3 个小时 0 分钟执行一次 /usr/bin/backup

  ```shell
  0 6-12/3 * 12 * /usr/bin/backup
  ```

* 周一到周五每天下午 5:00 寄一封信给 alex@domain.name

  ```shell
  0 17 * * 1-5 mail -s "hi" alex@domain.name < /tmp/maildata
  ```

* 每月每天的午夜 0 点 20 分, 2 点 20 分, 4 点 20 分....执行 echo "haha"

  ```shell
  20 0-23/2 * * * echo "haha"
  ```

**注意：**当程序在你所指定的时间执行后，系统会寄一封信给你，显示该程序执行的内容，若是你不希望收到这样的信，请在每一行空一格之后加上`> /dev/null 2>&1`即可。

**查看log**：

crontab的运行记录在/var/log/cron-201xxxxx中，直接vim打开这个文件，或者可以用tail -f /var/log/cron.log观察。

参考资料：

* [Linux crontab命令](https://www.runoob.com/linux/linux-comm-crontab.html)



## ll /proc/pid查看pid信息

通过`top`命令查看当前占用cpu资源的任务，然后找到该任务的pid，用`ll /proc/pid`进行查看，就可以找到该任务的信息，就能知道是谁在运行，是哪个程序运行的。



# 安全

## set出错就停止

* set -e

  设置该选项后，当脚本中任何以一个命令执行返回的状态码不为0时就退出整个脚本（默认脚本运行中某行报错会继续往下执行），这样就不必设置很多的判断条件去判断每个命令是否执行成功，特别那些依赖很强的地方，脚本任何一处执行报错就不应继续执行其他语句的时候就特别有用，之前写的一些像LAMP的安装脚本就深有体会。。。

* set +e

  执行的时候如果出现了返回值为非零将会继续执行下面的脚本 

* set -u

  设置该选项后，当脚本在执行过程中尝试使用未定义过的变量时，报错并退出运行整个脚本（默认会把该变量的值当作空来处理），这个感觉也非常有用，有些时候可能在脚本中变量很多很长，疏忽了把某个变量的名字写错了，这时候运行脚本不会报任何错误，但结果却不是你想要的，排查起来很是头疼，使用这个选项就完美的解决了这个问题。

注意：

1. 作用范围只限于脚本执行的当前进行，不作用于其创建的子进程。
2. 另外，当想根据命令执行的返回值，输出对应的log时，最好不要采用set -e选项，而是通过配合exit 命令来达到输出log并退出执行的目的。

# 系统环境

## source点命令

以一个脚本为参数，该脚本将作为当前shell的环境执行，即不会启动一个新的子进程。所有在脚本中设置的变量将成为当前Shell的一部分。

# 进程

## kill

一般使用kill -9直接彻底杀死进程

```shell
ps -ef | grep pidname
kill -9 pid
```



ps联合kill杀死进程（如杀死含有pidname的进程）

```shell
sudo ps -ef | grep lstm | awk '{print $2}' | xargs sudo kill -9
```

# 用户创建与删除

## adduser创建用户

```shell
sudo adduser luwei
```

有些发行版在创建用户的同时，会要求你设定用户密码。

## userdel删除用户

删除用户，“userdel 用户名”即可。最好将它留在系统上的文件也删除掉，可以使用“userdel -r 用户名”来实现。

```shell
sudo userdel -r luwei
```



