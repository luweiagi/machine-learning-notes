# Tensorflow中使用GPU

* [返回上层目录](../tools.md)



# 查看GPU的信息

使用`nvidia-smi`命令查看当前机器的GPU基本信息和被占用信息。

比如下图所示，当前机器有8个GPU，其中[3, 4, 5]被占用。

```shell
Tue Mar 31 20:03:24 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:8A:00.0 Off |                    0 |
| N/A   41C    P0    44W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:8B:00.0 Off |                    0 |
| N/A   37C    P0    45W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:8C:00.0 Off |                    0 |
| N/A   45C    P0    46W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:8D:00.0 Off |                    0 |
| N/A   50C    P0    75W / 300W |   8940MiB / 32480MiB |     49%      Default |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-SXM2...  Off  | 00000000:B3:00.0 Off |                    0 |
| N/A   44C    P0    60W / 300W |   1008MiB / 32480MiB |     28%      Default |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-SXM2...  Off  | 00000000:B4:00.0 Off |                    0 |
| N/A   46C    P0    92W / 300W |   1016MiB / 32480MiB |     41%      Default |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-SXM2...  Off  | 00000000:B5:00.0 Off |                    0 |
| N/A   43C    P0    46W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-SXM2...  Off  | 00000000:B6:00.0 Off |                    0 |
| N/A   42C    P0    44W / 300W |      0MiB / 32480MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    3    320060      C   python                                      8923MiB |
|    4    320060      C   python                                       993MiB |
|    5    320060      C   python                                      1001MiB |
+-----------------------------------------------------------------------------+
```

再通过命令`ll /proc/pid`查看pid信息。

另外，`nvidia-smi -l`命令可以不断自动刷新GPU信息。

# 动态指定空闲GPU

有两种，第一种是github上找的，另一种是自己写的。

* github上的：[gputil](https://github.com/anderskm/gputil)

* 自己写的：见具体的代码，这里就不写了，有点长。

# 程序按需占用GPU显存

在使用过程中，我发现我的机器显存在没有执行复杂操作的时候已经占用很高。经过调查发现，tensorflow默认是占用所有显卡和全部显存，用户可以通过以下设置，让tensorflow按需要占用显存。

以下命令为TensorFlow 2.x的命令：

```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
```

# 网上的云GPU

网上比较便宜的云gpu网站有：

* [MistGPU](https://mistgpu.com/user/)

* [极客云](https://www.jikecloud.net/)

以[MistGPU](https://mistgpu.com/user/)为例，介绍操作方法。

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





# 参考资料



===

[指定当前程序使用的 GPU](https://www.cnblogs.com/king-lps/p/12748459.html)



