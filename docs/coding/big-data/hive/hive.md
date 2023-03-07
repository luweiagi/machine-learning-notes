# HIVE常用函数

* [返回上层目录](../big-data.md)




这是工作实践中最常用到的一些sql函数，总结如下：



# 表操作查询

## ALTER TABLE

### 增删

#### 增加列

```sql
ALTER TABLE xxx.xxx ADD COLUMNS (
  xxx bigint COMMENT 'xxx'
)
```

#### 删除列

Hive没有删除指定列的命令，Hive通过replace命令变向实现删除列的功能。
replace命令将用新的列信息替换之前的列信息，相当于删除之前全部的列，再用新的列代替。

```sql
ALTER TABLE xxx.xxx REPLACE COLUMNS (
  xxx bigint COMMENT 'xxx'
)
```

### 内外表转换

外部表转为内部表

```sql
alter table xxx.xxxx set TBLPROPERTIES('EXTERNAL'='false')
```

参考资料：

* [Hive外部表和内部表区别以及相互转换](https://blog.csdn.net/shawnhu007/article/details/83055135)

### 删除分区

```sql
ALTER TABLE xxx.xxxx DROP IF EXISTS PARTITION(partition_date='20191002', partition_version='xxx')
```

注意：如果是外部表，则仅仅会删除表的meta信息，实际的数据是不会删除的。如果非要删除外部表的实际数据，则需要把外部表先转为内部表。

## LOAD DATA INPATH数据导入函数

```sql
LOAD DATA INPATH 'hdfs://nameservice/user/xxxxxxxx/' OVERWRITE INTO TABLE xxx.xxxxxxxx;
```

这样会把原始的hdfs中的数据删除掉，相当于剪切。

## desc table查询表的详细信息

```sql
desc xxx.table
```

## show create table建表语句查询

```sql
show create table xxxx.xxxxxxxxx;
```

## desc formatted表的存储位置查询

用下面的语句查询表级别的存储位置：

```sql
desc formatted xxx.xxxxxx
```

有一类表比较特殊，各个分区是自己用命令load的。因此需要查具体的分区信息：

```sql
desc formatted xxxx partition(partition_date=xxx)
```

## WITH AS临时表

WITH 通常与AS连用，也叫做子查询部分。

1). 可用来定义一个SQL片断，该片断会被整个SQL语句所用到。

2). 为了让SQL语句的可读性更高

```sql
WITH xxxx AS
(
    SELECT
        xxx
    FROM
        xxx
)

INSERT OVERWRITE TABLE xxxx.xxxxxxxx
SELECT
    xxx
FROM
	xxx
```

几个WITH AS连用要加逗号，只写一个WITH：

```sql
WITH sayhi_raw_tmp AS
(
    xxx
)

, active_tab_tmp AS
(
    yyy
)  

SELECT
		...
```

注意，create table的时候，应当把create table语句放在最前面：

```sql
CREATE TABLE abcxxxx AS 

WITH sayhi_raw_tmp AS
(
    xxx
)

SELECT
	z
FROM
	xxx
```







# UDF用户自定义函数

## 举例

```sql
ADD JAR hdfs://nameservice1/user/hive/warehouse/bin/udfs/mxxx-hive-latest.jar;
CREATE TEMPORARY FUNCTION udfLatLngToLocation AS 'com.imxxx.hive.common.udf.UDFLatLngToLocation';
ADD FILE hdfs://nameservice1/user/hive/thirddata/tl_geoCity_all/${hivevar:partition_date}/geoFile;
CREATE TEMPORARY FUNCTION udafMergeMapStr AS 'com.imxxxx.hive.common.udaf.UDAFMergeMapStr';
CREATE TEMPORARY FUNCTION udfToJson AS 'com.mxxx.hive.common.udf.UDFToJson';
```







# 常用内置函数



## A



### array_contains判断数组是否包含元素



判断元素数组是否包含元素:array_contains

语法: array_contains(Array\<T\>, value)

返回值: boolean

说明: 返回 Array\<T\>中是否包含元素 value

举例:

```sql
hive> select array_contains(array(1,2,3,4,5),3) from lxw1234; 
true
```



## C

### COALESCE返回参数中第一个非空值

语法: COALESCE(T v1, T v2, …) 
返回参数中的第一个非空值；如果所有值都为NULL，那么返回NULL

具体举例：https://www.cnblogs.com/luogankun/p/4015280.html



### collect_list列转数组不去重

Hive中collect相关的函数有collect_list和collect_set。

它们都是将分组中的某列转为一个数组返回，不同的是collect_list不去重而collect_set去重，详见[这里](https://blog.csdn.net/AntKengElephant/article/details/83277885)。

```sql
select 
	username, collect_list(video_name) 
from 
	t_visit_video 
group by username;
```



collect_list还有一个重要作用，展示字表排序后结果，而collect_set是乱序的：

```sql
insert overwrite table xxx.yyy partition (partition_date='${hivevar:partition_date}')
select
        concat_ws(':','np_item_cf',user_id) as redis_key
        ,user_id as hash_key,
        ,concat_ws(',',collect_list(item_id)) as value
from (
        select
                user_id,
                item_id,
                score,
        from xxx.zzz
        where partition_date='${hivevar:partition_date}'
        distribute by user_id
        sort by user_id, score desc
) t
group by user_id
;
```









### collect_set列转数组去重

Hive中collect相关的函数有collect_list和collect_set。

它们都是将分组中的某列转为一个数组返回，不同的是collect_list不去重而collect_set去重，详见[这里](https://blog.csdn.net/AntKengElephant/article/details/83277885)。

```sql
select 
	username, collect_set(video_name) 
from 
	t_visit_video 
group by username;
```



### concat字符串连接

语法: CONCAT(string A, string B…)

返回值: string

说明：返回输入字符串连接后的结果，支持任意个输入字符串

举例：

[select concat(‘abc’,'def’,'gh’) from lxw_dual;

abcdefgh



### concat_ws带分隔符字符串连接

语法: concat_ws(string SEP, string A, string B…);  concat_ws(',',array[n]) as value

返回值: string

说明：返回输入字符串连接后的结果，SEP表示各个字符串间的分隔符

举例1：

hive> select concat_ws(‘,’,'abc’,'def’,'gh’) from dual;

abc,def,gh

举例2：

concat_ws(',',collect_list(item_id)) as value

## D

### datediff日期相减天数

语法:   datediff(string enddate, string startdate) 
返回值: int
说明: 返回结束日期减去开始日期的天数。
举例：
hive>   select datediff('2012-12-08','2012-05-09') from dual;
213



### date_sub日期减少函数

语法: date_sub (string startdate, int days)
返回值: string
说明: 返回开始日期startdate减少days天后的日期。

```sql
select date_sub('2012-12-08',10) from iteblog;
```





### distinct去重

distinct单列很简单

distinct多列：select distinct id, type from tablename;

实际返回id与type同时不相同的结果,也就是distinct同时作用了两个字段，必须得id与type都相同的才被排除了。

现在验证一下：

```sql
drop table if exists table_1_luwei;
create table table_1_luwei(
    to_id string,
    from_id string
);
insert into table table_1_luwei values ('A',"L"),('B',"M"),('A',"M"),('A',"L");
select * from table_1_luwei;
```

结果为：

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | B     | M       |
| 3    | A     | M       |
| 4    | A     | L       |

现在distinct一下：

```sql
select distinct to_id, from_id from table_1_luwei;
```

结果变为了

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | A     | M       |
| 3    | B     | M       |





### distribute by A sort by B

distribute by是控制在map端如何拆分数据给reduce端的。hive会根据distribute by后面列，对应reduce的个数进行分发，默认是采用hash算法。

sort by为每个reduce产生一个排序文件。在有些情况下，你需要控制某个特定行应该到哪个reducer，这通常是为了进行后续的聚集操作。distribute by刚好可以做这件事。

因此，distribute by经常和sort by配合使用。

**注：Distribute by和sort by的使用场景**

**1.**Map输出的文件大小不均。

**2.**Reduce输出文件大小不均。

**3.**小文件过多。

**4.**文件超大。


## F

### FROM_UNIXTIME

将unix时间戳转换为自己想要的格式

```sql
SELECT
		FROM_UNIXTIME(act_time,'yyyyMMdd')as partition_date,
FROM 
		xxx.xxxxxx
```



## J

### JOIN

#### INNER JOIN

该函数的意思很容易懂，但是当join自身的时候，会发生什么？

这就不确定了吧。。现在我们来具体看下：

```sql
drop table if exists table_1_luwei;
create table table_1_luwei(
    to_id string,
    from_id string
);
insert into table table_1_luwei values ('A',"L"),('B',"M"),('A',"M"),('A',"N");
select * from table_1_luwei;
```

得到

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | B     | M       |
| 3    | A     | M       |
| 4    | A     | N       |

然后join自身

```sql
SELECT 
    t1.to_id as t1_to
    ,t1.from_id as t1_from
    ,t2.to_id as t2_to
    ,t2.from_id as t2_from
FROM 
    table_1_luwei t1 
INNER JOIN 
    table_1_luwei t2 
ON t1.from_id = t2.from_id;
```

得到

| 行号 | t1_to | t1_from | t2_to | t2_from |
| ---- | ----- | ------- | ----- | ------- |
| 1    | A     | L       | A     | L       |
| 2    | B     | M       | A     | M       |
| 3    | B     | M       | B     | M       |
| 4    | A     | M       | A     | M       |
| 5    | A     | M       | B     | M       |
| 6    | A     | N       | A     | N       |

我们再来看下它的应用（为什么要join自身）

```sql
SELECT
    t1_to
    ,t2_to
    ,count(*) as score
FROM
(
    SELECT 
        t1.to_id as t1_to
        ,t1.from_id as t1_from
        ,t2.to_id as t2_to
        ,t2.from_id as t2_from
    FROM 
        table_1_luwei t1 
    INNER JOIN 
        table_1_luwei t2 
    ON t1.from_id = t2.from_id
 ) aaa
GROUP BY t1_to, t2_to
```

得到

| 行号 | t1_to | t2_to | score |
| ---- | ----- | ----- | ----- |
| 1    | A     | B     | 1     |
| 2    | A     | A     | 3     |
| 3    | B     | A     | 1     |
| 4    | B     | B     | 1     |

看到了吧，结果是同现矩阵，用于协同过滤算法。



我们继续再建立一个表，尝试和上表进行交联。

```sql
drop table if exists table_2_luwei;
create table table_2_luwei(
    from_id string,
    activity int
);
insert into table table_2_luwei values ('L',2),('M',5),('N',7);
select * from table_2_luwei;
```

该表为：

| 行号 | from_id | activity |
| ---- | ------- | -------- |
| 1    | L       | 2        |
| 2    | M       | 5        |
| 3    | N       | 7        |

交联程序为：

```sql
SELECT
    t1_to
    ,t2_to
    ,sum(t1_activity * t2_activity) as score
FROM
(
    SELECT 
        t1.to_id as t1_to
        ,t2.to_id as t2_to
    	,t3.activity as t1_activity
    	,t4.activity as t2_activity
    FROM 
        table_1_luwei t1 
    INNER JOIN 
        table_1_luwei t2 
    ON t1.from_id = t2.from_id
    INNER JOIN 
        table_2_luwei t3
    ON t1.from_id = t3.from_id
    INNER JOIN 
        table_2_luwei t4
    ON t1.from_id = t4.from_id
 ) aaa
GROUP BY t1_to, t2_to
```

交联结果为：

| 行号 | t1_to | t2_to | score |
| ---- | ----- | ----- | ----- |
| 1    | B     | A     | 25    |
| 2    | A     | A     | 78    |
| 3    | B     | B     | 25    |
| 4    | A     | B     | 25    |



#### LEFT OUTER JOIN



创建表1:

```sql
drop table if exists table_1_luwei;
create table table_1_luwei(
    to_id string,
    from_id string
);
insert into table table_1_luwei values ('A',"L"),('B',"M"),('A',"M"),('A',"N");
select * from table_1_luwei;
```

结果如下：

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | B     | M       |
| 3    | A     | M       |
| 4    | A     | N       |

创建表2:

```sql
drop table if exists table_2_luwei;
create table table_2_luwei(
    to_id string,
    from_id string
);
insert into table table_2_luwei values ('A',"L"),('B',"N"),('C',"M"),('D',"M"),('A',"K");
select * from table_2_luwei;
```

结果如下：

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | B     | N       |
| 3    | C     | M       |
| 4    | D     | M       |
| 5    | A     | K       |

然后基于上面的两个表，进行where条件选择：

```sql
select 
    table_1_luwei.to_id as t1_to_id
    ,table_1_luwei.from_id as t1_from_id
    ,table_2_luwei.to_id as t2_to_id
    ,table_2_luwei.from_id as t2_from_id
from
    table_1_luwei
left outer join
    table_2_luwei
on table_1_luwei.from_id = table_2_luwei.from_id
```

结果如下：

| 行号 | t1_to_id | t1_from_id | t2_to_id | t2_from_id |
| ---- | -------- | ---------- | -------- | ---------- |
| 1    | A        | L          | A        | L          |
| 2    | B        | M          | D        | M          |
| 3    | B        | M          | C        | M          |
| 4    | A        | M          | D        | M          |
| 5    | A        | M          | C        | M          |
| 6    | A        | N          | B        | N          |

是不是和你想的一样呢？

而如果多加一个条件，那就是我们想要的：

```sql
select 
    table_1_luwei.to_id as t1_to_id
    ,table_1_luwei.from_id as t1_from_id
    ,table_2_luwei.to_id as t2_to_id
    ,table_2_luwei.from_id as t2_from_id
from
    table_1_luwei
left outer join
    table_2_luwei
on table_1_luwei.from_id = table_2_luwei.from_id and table_1_luwei.to_id = table_2_luwei.to_id
```

结果如下：

|      |          |            |          |            |
| ---- | -------- | ---------- | -------- | ---------- |
| 行号 | t1_to_id | t1_from_id | t2_to_id | t2_from_id |
| 1    | A        | L          | A        | L          |
| 2    | B        | M          |          |            |
| 3    | A        | M          |          |            |
| 4    | A        | N          |          |            |







## L



### log数学函数

语法: log(double base, double a)
返回值: double
说明: 返回以base为底的a的对数

```sql
select log(4,256) from iteblog;
```





## R



### rand随机

Hive实现从表中随机抽样得到一个不重复的数据样本

select  *  from table_a  order by rand() limit 100;



取1/10大小

```sql
select
	*
from
(
    xxx
) tab
where rand() <= 0.1
;
```





### regexp判断字符串是否包含字符串



```sql
recall_type_list regexp '16'
```





### regexp_replace正则表达式解析

语法: regexp_extract(string subject, string pattern, int index)
返回值: string
说明：将字符串subject按照pattern正则表达式的规则拆分，返回index指定的字符。

```sql
partition_date >= regexp_replace(date_sub(concat(substr('${hivevar:partition_date}',1,4),'-',substr('${hivevar:partition_date}',5,2),'-',substr('${hivevar:partition_date}',7,2)),30),'-','')
```



### ROUND四舍五入

**ROUND(X)：** 返回参数X的四舍五入的一个整数。(注意：ROUND 返回值被变换为一个BIGINT!)

```sql
select ROUND(-1.23);
-> -1
select ROUND(-1.58);
-> -2
select ROUND(1.58);
-> 2
```

**ROUND(X,D)：** 返回参数X的四舍五入的有 D 位小数的一个数字。如果D为0，结果将没有小数点或小数部分。

```sql
select ROUND(1.298, 1);
-> 1.3
select ROUND(1.298, 0);
-> 1
```





### row_number排序输出前n个

语法：row_number () over (partition by A order by B desc) as rank

排序输出前n个

说明: DISTRIBUTE BY COLUMN_A 的意思是按照 COLUMN_A 进行分组, SORT BY COLUMN_B 的意思是按照 COLUMN_B 进行排序, 后面跟着 ASC/DESC 指定是按照升序还是降序排序。row_number() 按指定的列进行分组生成行序列, 从 1 开始, 如果两行记录的分组列相同, 则行序列+1。

**[需求实现](https://my.oschina.net/jackieyeah/blog/681274)**

数据表 user_item_score 结构大致如下：

| user_id | item_id | item_score |
| ------- | ------- | ---------- |
| U_AAAA  | I_AAA1  | 0.5        |
| U_BBBB  | I_BBB1  | 0.3        |
| U_AAAA  | I_AAA2  | 0.6        |
| U_CCCC  | I_CCCC  | 0.7        |
| U_AAAA  | I_AAA3  | 0.55       |
| U_BBBB  | I_BBB2  | 0.4        |

实现 SQL 如下:

```sql
select user_id, item_id, item_score from (
    select *, row_number() over ( distribute by user_id sort by item_score desc) rownum from user_item_score
) temp
where rownum <= 50;
```

最终结果如下:

| user_id | item_id | item_score | row_num |
| ------- | ------- | ---------- | ------- |
| U_AAAA  | I_AAA2  | 0.6        | 1       |
| U_AAAA  | I_AAA3  | 0.55       | 2       |
| U_AAAA  | I_AAA1  | 0.5        | 3       |
| U_BBBB  | I_BBB2  | 0.4        | 1       |
| U_BBBB  | I_BBB1  | 0.3        | 2       |
| U_CCCC  | I_CCCC  | 0.7        | 1       |

是不是懂了？





## S

### sort by局部排序

sort by不是全局排序，其在数据进入reducer前完成排序，因此，如果用sort by进行排序，并且设置mapred.reduce.tasks>1，则sort by只会保证每个reducer的输出有序，并不保证全局有序。

sort by不同于order by，它不受hive.mapred.mode属性的影响，sort by的数据只能保证在同一个reduce中的数据可以按指定字段排序。

使用sort by你可以指定执行的reduce个数(通过set mapred.reduce.tasks=n来指定)，对输出的数据再执行归并排序，即可得到全部结果。



### sort_array对数组由小到大排序取极值

```sql
select arr[0] as min_val, arr[4] as max_val 
from(
     select sort_array(array(a,b,c,d,e)) arr 
     from test2
)a;
```





### split分割字符串

**语法**: split(string str, string pat)
**返回值**: array
**说明**: 按照pat字符串分割str，会返回分割后的字符串数组

```sql
select split('abtcdtef','t') from iteblog;
["ab","cd","ef"]
```





### substr,substring字符串截取

语法: substr(string A, int start, int len),substring(string A, int start, int len)

返回值: string

说明：返回字符串A从start位置开始，长度为len的字符串

举例：

hive> select substr(‘abcde’,3,2) from dual;

cd

hive> select substring(‘abcde’,3,2) from dual;

cd

hive>select substring(‘abcde’,-2,2) from dual;

de



```sql
partition_date >= regexp_replace(date_sub(concat(substr('${hivevar:partition_date}',1,4),'-',substr('${hivevar:partition_date}',5,2),'-',substr('${hivevar:partition_date}',7,2)),30),'-','')
```







## T

### trim去掉字符串两端空格

语法：trim(string A)

去掉字符串A两端的空格

## U

### UNIX_TIMESTAMP获取时间戳

MySQL中的UNIX_TIMESTAMP函数有两种类型供调用

**1  无参数调用：UNIX_TIMESTAMP()** 

返回值：自'1970-01-01 00:00:00'的到当前时间的秒数差

例子：SELECT UNIX_TIMESTAMP()  => 1339123415

**2  有参数调用：UNIX_TIMESTAMP(date)**

其中date可以是一个DATE字符串，一个DATETIME字符串，一个TIMESTAMP或者一个当地时间的YYMMDD或YYYMMDD格式的数字

返回值：自'1970-01-01 00:00:00'与指定时间的秒数差

举例说明：

**DATE字符串格式：（日期类型）**

SELECT UNIX_TIMESTAMP(‘2012-06-08’)       => 1339084800

SELECT UNIX_TIMESTAMP(CURRENT_DATE())  =>1339084800

注：CURRENT_DATE ()的返回值是一个DATE字符串格式

以下几种格式返回的结果相同：

SELECT UNIX_TIMESTAMP('20120608');

SELECT UNIX_TIMESTAMP('2012-6-8');

SELECT UNIX_TIMESTAMP('2012-06-08');

结果都是：1339084800

 

**DATETIME字符串格式:（日期和时间的组合类型）**

SELECT UNIX_TIMESTAMP(‘2012-06-08 10:48:55’)  => 1339123415

SELECT UNIX_TIMESTAMP(NOW())  => 1339123415

注：NOW()的返回值是一个DATETIME字符串格式







## W

### where条件过滤

where条件相信大家已经很熟悉了。

但是，对于两个表的where语句，当where的条件数量不足时，会发生什么情况么？可能心里并不是很清楚，那么下面就来验证一下：

创建表1:

```sql
drop table if exists table_1_luwei;
create table table_1_luwei(
    to_id string,
    from_id string
);
insert into table table_1_luwei values ('A',"L"),('B',"M"),('A',"M"),('A',"N");
select * from table_1_luwei;
```

结果如下：

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | B     | M       |
| 3    | A     | M       |
| 4    | A     | N       |

创建表2:

```sql
drop table if exists table_2_luwei;
create table table_2_luwei(
    to_id string,
    from_id string
);
insert into table table_2_luwei values ('A',"L"),('B',"N"),('C',"M");
select * from table_2_luwei;
```

结果如下：

| 行号 | to_id | from_id |
| ---- | ----- | ------- |
| 1    | A     | L       |
| 2    | B     | N       |
| 3    | C     | M       |

然后基于上面的两个表，进行where条件选择：

```sql
select 
    table_1_luwei.to_id as t1_to_id
    ,table_1_luwei.from_id as t1_from_id
    ,table_2_luwei.to_id as t2_to_id
    ,table_2_luwei.from_id as t2_from_id
from
    table_1_luwei
    ,table_2_luwei
where table_1_luwei.from_id = table_2_luwei.from_id
```

结果如下：

| 行号 | t1_to_id | t1_from_id | t2_to_id | t2_from_id |
| ---- | -------- | ---------- | -------- | ---------- |
| 1    | A        | L          | A        | L          |
| 2    | B        | M          | C        | M          |
| 3    | A        | M          | C        | M          |
| 4    | A        | N          | B        | N          |

是不是和你想的一样呢？

这和join是一样的结果





# 参考资料

* [Hive常用函数大全一览](https://www.iteblog.com/archives/2258.html)

“常用内置函数”参考该资料

