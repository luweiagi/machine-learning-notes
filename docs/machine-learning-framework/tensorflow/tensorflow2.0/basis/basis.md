# TensorFlow2.0基础

* [返回上层目录](../tensorflow2.0.md)



# 安装Tensorflow

## Anaconda下安装Tensorflow2.0

anaconda的安装方式不再赘述，首先虚拟出一个tf2.0的环境：

```shell
conda create -n tf20 python=3.6
```

然后激活这个环境：

```shell
conda activate tf20
```

使用pip来安装tf2.0:

```shell
pip install tensorflow
```

安装完以后输入如下命令来测试：

```shell
pip list | grep tensor
```

显示

```shell
tensorboard          2.0.2              
tensorflow           2.0.0              
tensorflow-estimator 2.0.1  
```

或者在python中输入：

```shell
import tensorflow
tensorflow.__version__
tensorflow.keras.__version__
```

output:

```shell
'2.0.0-alpha0'
'2.2.4-tf'
```

至此tensorflow2.0安装完成。

退出tf20环境：

```shell
conda deactivate
```



## linux系统下安装Tensorflow2.1

具体参考：[Centos7 安装Tensorflow2.1 GPU以及Pytorch1.3 GPU（CUDA10.1）](https://blog.csdn.net/qq_37541097/article/details/103933366)





# 测试是否支持GPU

```python
print(tf.test.is_gpu_available())
```







# 参考资料

* [Anaconda下安装Tensorflow2.0](https://blog.csdn.net/PecoHe/article/details/91356275)

"Anaconda下安装Tensorflow2.0"参考此博客。



