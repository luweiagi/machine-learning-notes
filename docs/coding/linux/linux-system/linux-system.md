# linux系统安装

* [返回上层目录](../linux.md)



# ubuntu安装

## ubuntu更新源



```shell
# 备份
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
# 编辑，添加的具体内容见下面的镜像内容
sudo gedit /etc/apt/sources.list
# 更新系统软件 并 更新已安装的包
sudo apt-get update && apt-get upgrade -y
```

要添加的镜像内容：

```shell
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse

deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse

deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse

deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse

deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse

deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty main restricted universe multiverse

deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-security main restricted universe multiverse

deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse

deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse

deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse
```



# 软件安装

## vim安装

执行下列更新以便安装vim。如果速度很慢，需要切换到国内镜像源，怎么切换参看对应章节。

```shell
sudo apt-get purge vim-common
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install vim
```



## 安装requirements.txt

使用豆瓣的源比较快，当然也可以使用其他的源。

```SHELL
pip install -r requirements.txt -i https://pypi.douban.com/simple/
```





