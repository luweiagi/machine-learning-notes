# Hadoop常用命令

* [返回上层目录](hadoop.md)



hadoop常用命令： 

* hadoop fs 

  查看Hadoop HDFS支持的所有命令 

* hadoop fs –ls 

  列出目录及文件信息 

* hadoop fs –lsr 

  循环列出目录、子目录及文件信息 

* hadoop fs –put test.txt /user/sunlightcs 

  将本地文件系统的test.txt复制到HDFS文件系统的/user/sunlightcs目录下 

* hadoop fs –get /user/sunlightcs/test.txt

  将HDFS中的test.txt复制到本地文件系统中，与-put命令相反 

* hadoop fs –cat /user/sunlightcs/test.txt

  查看HDFS文件系统里test.txt的内容 

* hadoop fs –tail /user/sunlightcs/test.txt 

  查看最后1KB的内容 

* hadoop fs –rm /user/sunlightcs/test.txt 

  从HDFS文件系统删除test.txt文件，rm命令也可以删除空目录 

* hadoop fs –rmr /user/sunlightcs

  删除/user/sunlightcs目录以及所有子目录 

* hadoop fs –copyFromLocal test.txt /user/sunlightcs/test.txt

  从本地文件系统复制文件到HDFS文件系统，等同于put命令 

* hadoop fs –copyToLocal /user/sunlightcs/test.txt test.txt

  从HDFS文件系统复制文件到本地文件系统，等同于get命令 

* hadoop fs –chgrp [-R] /user/sunlightcs

  修改HDFS系统中/user/sunlightcs目录所属群组，选项-R递归执行，跟linux命令一样 

* hadoop fs –chown [-R] /user/sunlightcs

  修改HDFS系统中/user/sunlightcs目录拥有者，选项-R递归执行 

* hadoop fs –chmod [-R] MODE /user/sunlightcs

  修改HDFS系统中/user/sunlightcs目录权限，MODE可以为相应权限的3位数或+/-{rwx}，选项-R递归执行 

* hadoop fs –count [-q] PATH

  查看PATH目录下，子目录数、文件数、文件大小、文件名/目录名 

* hadoop fs –cp SRC [SRC …] DST

  将文件从SRC复制到DST，如果指定了多个SRC，则DST必须为一个目录 

* hadoop fs –du PATH

  显示该目录中每个文件或目录的大小 

* hadoop fs –dus PATH

  类似于du，PATH为目录时，会显示该目录的总大小 

* hadoop fs –expunge

  清空回收站，文件被删除时，它首先会移到临时目录.Trash/中，当超过延迟时间之后，文件才会被永久删除 

* hadoop fs –getmerge SRC [SRC …] LOCALDST [addnl]

  获取由SRC指定的所有文件，将它们合并为单个文件，并写入本地文件系统中的LOCALDST，选项addnl将在每个文件的末尾处加上一个换行符 

* hadoop fs –touchz PATH

  创建长度为0的空文件 

* hadoop fs –test –[ezd] PATH

  对PATH进行如下类型的检查： 

  * -e PATH是否存在，如果PATH存在，返回0，否则返回1 
  * -z 文件是否为空，如果长度为0，返回0，否则返回1 
  * -d 是否为目录，如果PATH为目录，返回0，否则返回1 

* hadoop fs –text PATH

  显示文件的内容，当文件为文本文件时，等同于cat，文件为压缩格式（gzip以及hadoop的二进制序列文件格式）时，会先解压缩 

* hadoop fs –help ls

  查看某个[ls]命令的帮助文档



# 参考资料

* [hadoop hdfs常用命令](https://blog.csdn.net/gz747622641/article/details/54133728)

本文复制自此csdn博客。


