# cython

* [返回上层目录](../python.md)



# Cython简介

Cython的本质可以总结如下：Cython是包含C数据类型的Python。

Cython是Python：几乎所有Python代码都是合法的Cython代码。 （存在一些限制，但是差不多也可以。） Cython的编译器会转化Python代码为C代码，这些C代码均可以调用Python/C的API。

Cython可不仅仅包含这些，Cython中的参数和变量还可以以C数据类型来声明。代码中的Python值和C的值可以自由地交叉混合（intermixed）使用, 所有的转化都是自动进行。



# 例子

在同一个目录下创建如下两个文件：

**pytest.pyx**

```cython
import numpy as np
from libc.math cimport pow, tanh
cimport numpy as np
cimport cython
np.import_array()

cdef class FmSGD(object):

    cdef public int vec_dim

    def __init__(self, int vec_dim):
        self.vec_dim = vec_dim

    def fit(self, int num):
        self.vec_dim  = num

```

**setup.py**

```python
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    maintainer='luwei',
    name='pytest',
    version='0.0.1',
    packages=find_packages(),
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("pytest", ["pytest.pyx"],
    						 libraries=["m"],
    						 include_dirs=[numpy.get_include()])]
    )

```

然后通过```python setup.py build_ext --inplace```命令就可以将pytest库安装到python中了。

如下所示：

```python
from pytest import FmSGD
fm = FmSGD(11)
fm.vec_dim
#11
fm.fit(22)
fm.vec_dim
#22
```

注：

* python setup.py build_ext --inplace，这种方式不好，不利于后续的管理和替换，建议安装的时候用pip
* 安装cython文件，和anaconda没有任何关系，是给你的python添加组件，只影响你的python库
* 用conda打包，这样可以保证把依赖的库都打包好，不会有环境不一致的问题
* cython文件是在桌面上的一个文件夹内，cd到该目录下，然后pip安装（pip可以安装本地的包```pip install .```），就会自动把这个文件添加到python环境里

**安装方法：**

通过`conda env list`查看环境列表，激活你想安装的conda环境`source activate xxx`。

然后cd到cyhon文件的目录下，进行安装`pip install .`。

```shell
Looking in indexes: http://mirrors.momo.com/pypi/simple/
Processing /data4/recommend_nearby/lu.wei/fm/cython
Building wheels for collected packages: PySparkFM
  Building wheel for PySparkFM (setup.py) ... done
  Stored in directory: /tmp/pip-ephem-wheel-cache-4hx2m394/wheels/d8/6c/d4/645ca5adc6e249f6dc4e303a044be9c8d2aedb804bb0940b4c
Successfully built PySparkFM
Installing collected packages: PySparkFM
Successfully installed PySparkFM-0.0.1
```

通过`pip install conda-pack`安装打包程序。

通过`conda pack -o ./luwei_environment.tar.gz`将需要的环境打包。

```shell
Collecting packages...
Packing environment at '/home/recommend_nearby/work_space/anaconda3' to './luwei_environment.tar.gz'
[####                                    ] | 12% Completed | 43.2s
```

并拷贝到目标机器或者上传到hdfs上

```
hadoop fs -put ./luwei_environment.tar.gz hdfs://nameservice3/user/recommend_nearby/lu.wei/python3/
```



在目标机器上新建目录`mkdir -p $Anaconda/envs/xxx`。

解压`tar -xzf wrfpy.tar.gz -C $Anaconda/envs/xxx`。

此时`conda env list`就可以看到该环境了。

`conda activate xxx`激活环境，执行`conda-unpack`，大功告成！



====
另外，我发现还有个参数
--conf spark.pyspark.driver.python = /home/barrenlake/tmp/python-2.7.15/bin/python \
这个用该是告诉driver端，所用python的路径吧？
那有了这个设置，是不是就可以不要export PATH=`pwd`/bin:$PATH了？





# 参考资料

* [Cython官方文档中文版](https://moonlet.gitbooks.io/cython-document-zh_cn/content/ch1-basic_tutorial.html)

“Cython简介”参考此博客。

* [Cython - 入门简介](https://zyxin.xyz/blog/2017-12/CythonIntro/)

“例子”参考该博客。

* [官方文档：Using C++ in Cython](http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html)

更多请看官方文档。

* [Spark on Yarn 之Python环境定制](https://www.jianshu.com/p/d77e16008957)

`--archives xxx#xxx`和`--conf spark.pyspark.driver.python`。

* [pyspark使用anaconda后spark-submit方法](https://blog.csdn.net/crookie/article/details/78351095)

python上传后，在进行spark-submit时，会自动分发anaconda2的包到各个工作节点。但还需要给工作节点指定python解压路径。

* [Anaconda环境打包迁移到另一台机器](https://youyou-tech.com/2019/11/03/Anaconda%E7%8E%AF%E5%A2%83%E6%89%93%E5%8C%85%E8%BF%81%E7%A7%BB%E5%88%B0%E5%8F%A6%E4%B8%80%E5%8F%B0%E6%9C%BA%E5%99%A8/)

使用Conda-Pack来进行环境的打包和迁移。

* [Conda-Pack](https://conda.github.io/conda-pack/)

Conda-Pack的官方说明。

