# MistGPU

* [返回上层目录](../cloud-server-platform.md)



[MistGPU](https://mistgpu.com/user/)

![mistgpu](pic/mistgpu.jpg)

（1）配置gpu硬件和软件框架。在`创建服务器`中，选择合适的GPU机型，并选择适合的软件框架，比如是tf1.x还是tf2.x，并设置ssh远程登录密码。

（2）登录服务器。在`服务器管理`中，复制ssh命令，然后粘贴到本地的linux终端，输入密码就可以远程登录到服务器上。

（3）上传代码和数据。在`上传数据集`中，点击`选择文件`然后点击`开始上传`，就可以将数据上传到服务器的`/data`目录中。

（4）下载模型和数据。训练完的模型下载，使用scp命令：

```shell
scp -P 54000 mist@gpu48.mistgpu.com:/data/cloud/file.tar ~/Desktop/
```

或者在win系统上使用WinSCP软件进行下载。

注意，对于大文件，最好只是简单地打个包，不要压缩，否则下载下来会出错。打包命令：

```shell
tar -vcf xxx  # 打包
tar -vxf xxx  # 解包
```

（5）关闭远程服务器。

在网页端的`服务器管理`下点击`关机`即可。或者在远程服务器的`~/`目录下运行`shutdown.sh`关机命令。



