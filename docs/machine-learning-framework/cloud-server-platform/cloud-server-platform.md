# 模型训练云服务器平台


* [返回上层目录](../machine-learning-framework.md)



网上比较便宜的云gpu网站有：

- [MistGPU](https://mistgpu.com/user/)
- [极客云](https://www.jikecloud.net/)

# MistGPU

以[MistGPU](https://mistgpu.com/user/)为例，介绍操作方法。

![mistgpu](D:/2study/%E8%AF%BE%E7%A8%8B%E4%B8%8E%E5%AD%A6%E4%B9%A0/0%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/02%E4%B9%A6%E7%B1%8D/%E4%B9%A6%E7%B1%8D_%E6%95%B0%E6%8D%AE/%E4%B9%A6%E7%B1%8D/%E3%80%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0%E3%80%8B/machine-learning-notes/docs/machine-learning-framework/tensorflow/tools/gpu/pic/mistgpu.jpg)

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



# 参考资料



===

* [【深度学习】模型训练云服务器平台推荐！！！个人心路历程，新手少踩坑](https://zhuanlan.zhihu.com/p/597476907)

介绍了[AutoDL-品质GPU租用平台-租GPU就上AutoDL](https://www.autodl.com/register?code=cd8a7443-6fc2-4ec4-b88a-da35fb2ac603)