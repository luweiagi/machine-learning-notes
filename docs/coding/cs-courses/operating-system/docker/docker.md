# Docker

* [返回上层目录](../operating-system.md)
* [镜像和容器](#镜像和容器)
  * [镜像和容器的区别](#镜像和容器的区别)
  * [镜像理解](#镜像理解)
  * [容器理解](#容器理解)
* [docker命令](#docker命令)
  * [docker常见命令](#docker常见命令)
  * [docker命令集](#docker命令集)
  * [镜像离线导出与迁移](#镜像离线导出与迁移)
  * [在目标主机加载镜像并启动容器](#在目标主机加载镜像并启动容器)
* [知识点](#知识点)
  * [docker使用privileged参数进入特权模式](#docker使用privileged参数进入特权模式)
    * [docker逃逸原理](#docker逃逸原理)

# 镜像和容器

## 镜像和容器的区别

![image_vs_container](pic\image_vs_contrainer.jpg)

两种理解：

* **光盘操作系统**：简单点说，镜像就类似操作系统光盘介质，容器相当于通过光盘安装后的系统。通过光盘(镜像)，我们能在不同机器上部署系统(容器)，系统内的操作只会保留在当前的系统(容器)中，如果要升级系统，需要使用到光盘，但是可能会导致操作系统的数据丢失。

* **类和对象**：容器是由镜像实例化而来，这和我们学习的面向对象的概念十分相似，我们可以把镜像看作类，把容器看作类实例化后的对象。

## 镜像理解

docker的镜像概念类似虚拟机的镜像。是一个只读的模板，一个独立的文件系统，包括运行容器所需的数据，可以用来创建新的容器。（  docker create <image -id > ：为指定的镜像添加一个可读写层，构成一个新的容器；）

例如：一个镜像可以包含一个完整的ubuntu操作系统环境，里面仅安装了mysql或用户需要的其他应用程序。

docker镜像实际上是由一层一层的系统文件组成，这种层级的文件系统被称为UnionFS( Union file system  统一文件系统)，镜像可以基于dockerfile构建，dockerfile是一个描述文件，里面包含了若干条命令，每条命令都会对基础文件系统创建新的层次结构。

docker提供了一个很简单的机制来创建镜像或更新现有的镜像。用户甚至可以从其他人那里下载一个已经做好的镜像直接使用。（镜像是只读的，可以理解为静态文件）

## 容器理解

docker利用容器来运行应用：

docker容器是由docker镜像创建的运行实例。docker容器类似虚拟机，可以执行包含启动，停止，删除等。

每个容器间是相互隔离的。

容器中会运行特定的应用，包含特定应用的代码及所需的依赖文件。可以**把容器看作一个简易版的linux环境**（包含root用户权限，进程空间，用户空间和网络空间等）和运行在其中的应用程序。

相对于镜像来说容器是动态的，容器在启动的时候创建了一层可写层次作为最上层。（   docker create <image -id > ：为指定的镜像添加一个可读写层，构成一个新的容器；）

```shell
docker 容器=镜像+可读层
```

# docker命令

## docker常见命令

* 新建镜像：

```shell
docker build -t image_01.qc:v0.1 /path/to/Dockerfile
```

* 查看镜像：

```shell
docker images -a
```

* 新建容器：

```shell
docker create  --name myrunoob  nginx:latest 
```

* 查看容器：

```shell
docker ps -a
```

* 运行容器：

说明：这个命令是核心命令，可配置的参数很多。详细的解释可以通过 docker run --help 列出。

```shell
docker run --name mynginx -d nginx:latest
docker run -it -v /test:/soft centos /bin/bash
```

* 进入容器退出，并结束容器运行 

```shell
exit
```

* 退出容器但是容器仍在执行，按`ctrl + p + q`，会回到宿主机桌面 

```shell
ctrl + p + q
```

* 关闭容器：

```shell
docker kill e7c  # 支持模糊查找，只写名称的前三个就可以
```

* 重启容器：

```shell
docker start e7c
docker restart e7c  # 即便容器已经启动，restart也会给重启
```

* **容器启动并进入后台后，这个时候进入容器进行操作，可以使用docker attach命令或docker exec命令**：

（1）docker attach 容器id：

attach是docker自带的命令。**注意**，该命令的前提是，容器是已经被启动的，一旦kill或exit，就先需要start启动容器，否则会报错：`You cannot attach to a stopped container, start it first`。

```shell
docker attach e7c
```

（2）以交互模式进入容器：

从Docker的1.3版本起，Docker提供了更加方便的工具exec命令，可以在运行容器内直接执行任意命令。**注意**，该命令的前提是，容器是已经被启动的，一旦kill或exit，就先需要start来启动，然后再执行该命令，不然会报错：`Error response from daemon: Container e7c is not running`。

```shell
docker exec -it e7c /bin/bash
```

docker attach命令和docker exec命令的区别：

（a）当多个窗口同是attach到同一个容器的时候，所有窗口都会同步显示；当某个窗口因命令阻塞时，其他窗口也无法执行操作。（b）可以使用`docker exec -it 容器id /bin/bash`进入容器并开启一个新的bash终端。 退出容器终端时，不会导致容器的停止。（c）使用`docker attach 容器id`进入正在执行容器，不会启动新的终端， 退出容器时，会导致容器的停止。

* 删除容器：

```shell
docker rm e7c
```

* 删除镜像（若该镜像有关联的容器，需先删除容器或使用 `-f` 强制删除）：

```shell
docker rmi image
```

## docker命令集

[docker官网docker base command line](https://docs.docker.com/engine/reference/commandline/docker/)

[菜鸟：docker命令大全](https://www.runoob.com/docker/docker-command-manual.html)

* 为指定的镜像添加一个可读写层，构成一个新的容器：

```shell
docker create <image -id> 
```

* docker start 命令为容器文件系统创建一个进程的隔离空间。注意，每一个容器只能够有一个进程隔离空间；（运行容器）：

```shell
docker start <container -id>
```

* 这个是先利用镜像创建一个容器，然后运行了这个容器：

说明：这个命令是核心命令，可配置的参数很多。详细的解释可以通过 docker run --help 列出。

```shell
docker run <image -id>
```

* 停止所用的进程：

```shell
docker stop <container -id>
```

* 向所有运行在容器的进程发送一个不友好的 SIGKILL 信号：

```shell
docker kill <container -id>
```

* 将运行中的进程空间暂停：

```shell
docker pause <container -id>
```

* `docker rm`命令会移除构成容器的可读写层。注意，这个命令**只能对非运行态容器执行**：

```shell
docker rm <container -id>
```

* docker rmi是docker image rm的别名。`docker rmi`命令会移除构成镜像的一个只读层。你只能够使用`docker rmi`来移除最顶层（top level layer）（也可以说是镜像），你也可以使用`-f`参数来强制删除中间的只读层：

```shell
docker rmi <image -id>
```

* `docker commit`命令将容器的可读写层转换为一个只读层，这样就把一个容器转换成了不可变的镜像：

```shell
docker commit <container-id>
```

* `docker save`命令会创建一个镜像的压缩文件，这个文件能够在另外一个主机的Docker上使用。和export命令不同，这个命令为每一个层都保存了它们的元数据。这个命令只能对镜像生效：

```shell
docker save <image-id>
```

* `docker export`命令创建一个tar文件，并且移除了元数据和不必要的层，将多个层整合成了一个层，只保存了当前统一视角看到的内容（注：export 后的容器再 import 到 Docker 中，只能看到一个扁平镜像；而 save 后的镜像则不同，可通过 `docker history <image-id>` 查看该镜像的层级历史）：

```shell
docker export <container-id>
```

* `docker history`命令逐层输出指定镜像的构建历史：

```shell
docker history <image-id>
```

* 会列出所有运行中的容器；`docker ps -a`列出运行中和未运行的容器：

```shell
docker ps -a
```

* 列出所用的镜像，也可以说列出所用的可读层：

```shell
docker images -a
```

* 显示容器內运行的进程：

```shell
docker top <container-id>
```

## 镜像离线导出与迁移

以 `quay.io/ascend/vllm-ascend:v0.11.0-310p` 为例，典型的离线迁移流程分三步（在源主机操作）：

**在源主机拉取镜像（docker pull）**

```shell
docker pull quay.io/ascend/vllm-ascend:v0.11.0-310p
```

- 从远端仓库（这里是 `quay.io`）按层（layer）下载镜像，并校验 Digest，最终把镜像保存到本机的 Docker 镜像存储（如 `/var/lib/docker/`）。

**在源主机导出镜像为离线文件（docker save）**

```shell
sudo docker save -o vllm-ascend-v0.11.0-310p.tar quay.io/ascend/vllm-ascend:v0.11.0-310p
```

- `docker save` 会把镜像的各个 layer、manifest、config 等元数据完整打包成 `tar` 文件。
- 生成的 `vllm-ascend-v0.11.0-310p.tar` 可以通过 U 盘、内网传输等方式拷贝到 **没有外网** 的目标主机上。

**在源主机删除镜像（docker rmi，可选）**

```shell
docker rmi quay.io/ascend/vllm-ascend:v0.11.0-310p
```

- 释放源主机磁盘空间，**不影响** 之前导出的 `tar` 文件。
- 也可以用来验证离线包：在目标主机执行 `docker load -i vllm-ascend-v0.11.0-310p.tar`，如果能成功恢复并运行，就说明离线镜像包是可用的。

综上，这三步完成的是：**从公网拉镜像 → 导出为离线包 → 为在其他机器上恢复镜像做好准备**，是常见的离线环境部署流程。

## 在目标主机加载镜像并启动容器

在目标主机上，通常会按下面几步把离线镜像“变成一个可用的 NPU 容器环境”：

**加载离线镜像到 Docker（docker load）**

```shell
docker load -i vllm-ascend-v0.11.0-310p.tar
```

- Docker 从 `tar` 中逐层恢复镜像层（layer）、manifest、config 等信息。
- 看到类似 `Loaded image: quay.io/ascend/vllm-ascend:v0.11.0-310p`，说明目标主机已经成功拥有该镜像。

```shell
3cc982388b71: Loading layer [=================>]  29.54MB/29.54MB
3e5a7e47fd6e: Loading layer [=================>]  130.8MB/130.8MB
c87ba06aad1f: Loading layer [=================>]  183.9MB/183.9MB
59fc0c65c8af: Loading layer [=================>]  4.392GB/4.392GB
4c9b8a055547: Loading layer [=================>]     214B/214B
5303e7da671d: Loading layer [=================>]  1.875kB/1.875kB
d9bba1b583db: Loading layer [=================>]  40.59MB/40.59MB
e28e0b674115: Loading layer [=================>]      97B/97B
ffe33993c4ca: Loading layer [=================>]   2.33MB/2.33MB
b01ba8bfeafa: Loading layer [=================>]     214B/214B
00162ac7dfe5: Loading layer [=================>]  19.28MB/19.28MB
aadfea230c55: Loading layer [=================>]  400.4MB/400.4MB
c154b5d039d8: Loading layer [=================>]  350.9MB/350.9MB
e245b3ff419d: Loading layer [=================>]  95.67MB/95.67MB
Loaded image: quay.io/ascend/vllm-ascend:v0.10.0rc1-310p
```

**确认镜像已加载并获取 image ID（docker images）**

```shell
docker images
```

- 确认仓库名、Tag 是否正确。
- 记下对应的 `IMAGE ID`（如 `dd63478e53a0`），方便在 `docker run` 里直接使用。

```shell
REPOSITORY                 TAG             IMAGE ID     CREATED      SIZE
quay.io/ascend/vllm-ascend v0.10.0rc1-310p dd63478e53a0 6 months ago 16.7GB
```

**通过脚本启动 Ascend 推理容器（docker run）**

例如脚本 `run_vllm_ascend.sh`：

```shell
docker run --name vllm_ascend_10 -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    --device /dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /tmp:/tmp \
    -v /mnt:/mnt \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    dd63478e53a0
```

关键参数含义简要说明：

- **`--name vllm_ascend_10`**：容器名，便于后续 `docker exec`。
- **`-it -d`**：既支持交互（`-it`），又在后台运行（`-d`）。
- **`--net=host`**：共享宿主机网络栈，容器内服务端口无需额外映射，适合分布式 / 推理服务。
- **`--shm-size=500g`**：扩大容器共享内存 `/dev/shm`，防止大模型 / vLLM 在默认 64MB 下出问题。
- **`--privileged=true`**：给予容器接近宿主机 root 的权限，便于访问 NPU 设备（需结合安全策略谨慎使用）。
- **`--device /dev/davinci_manager` 等**：把 Ascend NPU 相关设备节点直通进容器，使容器内能看到并使用 NPU。
- **各类 `-v` 挂载**：复用宿主机的 Ascend 驱动、运维工具，以及 `/home`、`/mnt` 等数据目录，避免重复安装和拷贝数据。
- **`--entrypoint=bash`**：启动后先进入 `bash`，方便在容器内手动验证环境（NPU 可见性、框架 import 等）。

**进入已启动的容器进行操作（docker exec）**

```shell
docker exec -it vllm_ascend_10 /bin/bash
```

- `docker run` 负责“创建并启动”容器。
- `docker exec` 则是“进入一个已经在运行的容器”，不会新建容器实例。

整体来看，你在目标主机上完成的是：**从离线镜像文件恢复镜像 → 基于该镜像启动带 Ascend 驱动直通的推理容器 → 进入容器做后续大模型推理或调试**。

# 知识点

## docker使用privileged参数进入特权模式

```
docker run [选项] 镜像名
选项
-d 后台运行
-it 提供容器交互
--name 设置容器名
--cpus 设置cpu个数
--env 设置环境变量
--mount type=bind,source=/root/target,target=/app或者--mount type=tmpfs,destination=/app 
--volume <host>:<container>:[rw|ro]挂载一个磁盘卷 例如 --volume /home/hyzhou/docker:/data:rw
--restart 设置重启策略on-failure,no,always
--privileged 使用该参数，container 内的 root 拥有真正的 root 权限。否则，container 内的 root 只是外部的一个普通用户权限。privileged 启动的容器可以看到很多 host 上的设备，并可执行 mount，甚至允许在容器中再启 Docker。**生产环境尽量避免使用，以免扩大逃逸风险。**
```

这个是先利用镜像创建一个容器，然后运行了这个容器：

```shell
sudo docker run -ti --privileged ubuntu bash
```

`--privileged`使用该参数，container内的root拥有真正的root权限。否则，container内的root只是外部的一个普通用户权限。privileged启动的容器，可以看到很多host上的设备，并且可以执行mount。甚至允许你在docker容器中启动docker容器。

使用特权模式启动容器后（docker run --privileged），Docker容器被允许可以访问主机上的所有设备、可以获取大量设备文件的访问权限、并可以执行mount命令进行挂载。

![docker-privileged](C:\Users\lw\Desktop\docker\pic\docker-privileged.png)



docker 启动 nvidia/cuda 镜像时使容器能使用主机 GPU 等硬件资源：

```shell
# 仅需 GPU 时推荐使用 --gpus all（需宿主机安装 nvidia-container-toolkit），无需开特权模式
sudo docker run -it --gpus all --name detectron nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 /bin/bash

# 若确有需要才使用 --privileged（权限大、有逃逸风险）
sudo docker run -it --privileged=true --name detectron nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 /bin/bash
```

上述镜像 tag 为示例（CUDA 9 / Ubuntu 16.04 较旧），实际可按需选用新版本。仅需 GPU 时优先用 `--gpus all`，避免滥用 `--privileged`。

- [docker privileged作用_美创安全实验室 | Docker逃逸原理](https://blog.csdn.net/weixin_39664998/article/details/110639657?spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-10-110639657-blog-90576040.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-10-110639657-blog-90576040.pc_relevant_default)

### docker逃逸原理

因为Docker所使用的是隔离技术，就导致了容器内的进程无法看到外面的进程，但外面的进程可以看到里面，所以如果一个容器可以访问到外面的资源，甚至是获得了宿主主机的权限，这就叫做“Docker逃逸”。
目前产生Docker逃逸的原因总共有三种：

1. 由内核漏洞引起。
2. 由Docker软件设计引起。
3. 由特权模式与配置不当引起。

4. 由于特权模式+目录挂载引起的逃逸

这一种逃逸方法较其他两种来说用的更多。特权模式在6.0版本的时候被引入Docker，其核心作用是允许容器内的root拥有外部物理机的root权限，而此前在容器内的root用户只有外部物理机普通用户的权限。

使用特权模式启动容器后（docker run --privileged），Docker容器被允许可以访问主机上的所有设备、可以获取大量设备文件的访问权限、并可以执行mount命令进行挂载。

当控制使用特权模式的容器时，Docker管理员可通过mount命令将外部宿主机磁盘设备挂载进容器内部，获取对整个宿主机的文件读写权限，此外还可以通过写入计划任务等方式在宿主机执行命令。

除了使用特权模式启动Docker会引起Docker逃逸外，使用功能机制也会造成Docker逃逸。Linux内核自版本2.2引入了功能机制（Capabilities），打破了UNIX/LINUX操作系统中超级用户与普通用户的概念，允许普通用户执行超级用户权限方能运行的命令。例如当容器以--cap-add=SYSADMIN启动，Container进程就被允许执行mount、umount等一系列系统管理命令，如果攻击者此时再将外部设备目录挂载在容器中就会发生Docker逃逸。

下面是使用特权模式后，docker可以挂载主机设备的例子：

主机上的设备`/dev/sda3`：

![host-root](C:\Users\lw\Desktop\docker\pic\host-root.png)

docker上挂载主机的设备`/dev/sda3`，然后就能在docker上看到主机的设备`/dev/sda3`了，并且还能直接修改，其实就相当于docker获取了主机的root权限：

![docker-mount-host](C:\Users\lw\Desktop\docker\pic\docker-mount-host.png)

# 参考资料

===

* [容器技术及其应用白皮书（上）-- 容器技术](https://blog.csdn.net/wh211212/article/details/53535881)
* [容器技术及其应用白皮书（下）-- 容器应用](https://blog.csdn.net/wh211212/article/details/53540342)

对容器技术写的不错的文章，待看

* [认识容器，我们从它的历史开始聊起](https://bbs.huaweicloud.com/blogs/285728)

写了对容器的理解。

* [docker -v 挂载问题](https://blog.csdn.net/hnmpf/article/details/80924494)

深入分析了docker -v的挂载问题。

