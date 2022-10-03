# 流程控制

* [返回上层目录](../shell.md)



# 条件语句

## if

### if语句语法格式

* if

  ```shell
  if condition
  then
      command1 
      command2
      ...
      commandN 
  fi
  ```

  写成一行（适用于终端命令提示符）：

  ```shell
  if [ $(ps -ef | grep -c "ssh") -gt 1 ]; then echo "true"; fi
  ```

  末尾的fi就是if倒过来拼写，后面还会遇到类似的。

* if else

  ```shell
  if condition
  then
      command1 
      command2
      ...
      commandN
  else
      command
  fi
  ```

* if else-if else

  ```shell
  if condition1
  then
      command1
  elif condition2 
  then 
      command2
  else
      commandN
  fi
  ```

**实例**

判断两个变量是否相等：

```shell
a=10
b=20
if [[ $a == $b ]]
then
   echo "a 等于 b"
elif [[ $a -gt $b ]]
then
   echo "a 大于 b"
elif [[ $a -lt $b ]]
then
   echo "a 小于 b"
else
   echo "没有符合的条件"
fi
```

输出结果：

```js
a 小于 b
```



# 循环语句

## for



## while



## until



# 选择语句



## case



## select



# 参考资料

* [linux shell 流程控制（条件if,循环【for,while】,选择【case】语句实例](https://www.cnblogs.com/chengmo/archive/2010/10/14/1851434.html)

本文架构参考此博客。



