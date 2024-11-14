# 自动驾驶

* [返回上层目录](../README.md)
* [传感器](sensor/sensor.md)
* [感知](perception/perception.md)
* [导航](navigation/navigation.md)
* [轨迹预测](trajectory-prediction/trajectory-prediction.md)
* [决策](decision/decision.md)
* [规划](planning/planning.md)
* [制导](guidance/guidance.md)
* [控制](control/control.md)
* [动力系统](dynamic-system/dynamic-system.md)

* [集群](swarm/swarm.md)





[大模型能应用在自动驾驶上吗？是否可以让自动驾驶真的走向无人？](https://www.zhihu.com/question/625473884/answer/3258026554)

特斯拉在2023年AI Day公开了occupancy network（占用网络）模型，基于学习进行三维重建，意图为更精准地还原自动驾 驶汽车行驶周围3D环境，可视作BEV视图的升华迭代。



[大模型能应用在自动驾驶上吗？是否可以让自动驾驶真的走向无人？](https://www.zhihu.com/question/625473884)

其实自动驾驶公司已经开始将研究方向转向大模型了，期望能一个模型解决所有，特斯拉据说已经将模型部署到实车上，国内的各大自动驾驶公司也在争先模仿！推荐一些最新自动驾驶+大模型的工作，可以看看~

1. A Survey on Multimodal Large Language Models for Autonomous Driving
2. A Survey of Large Language Models for Autonomous Driving
3. CLIP：Learning Transferable Visual Models From Natural Language Supervision
4. ...



[像人一样开车，大语言模型建攻自动驾驶！自动驾驶迎来ChatGPT时刻](https://new.qq.com/rain/a/20230906A03TOE00)

最近有项研究提出了一种新方法，用大型语言模型来重新思考自动驾驶技术。作者讨论了使用LLM来模拟人类理解驾驶环境，并分析其在处理复杂情况时的推理、解释和记忆能力。他们认为传统的自动驾驶系统在处理边界情况时存在性能限制。为解决这个问题，提出了一个理想的自动驾驶系统，它能像人类一样通过驾驶经验和常识来解决问题。为了实现这个目标，我们确定了三个关键能力：推理、解释和记忆。通过构建一个闭环系统，展示了在驾驶场景中使用LLM的可行性，以展示其理解和与环境交互的能力。实验结果表明，LLM展示了令人印象深刻的推理和解决复杂情况的能力，为开发类似人类的自动驾驶系统提供了宝贵的见解。

代码：

https://github.com/PJLab-ADG/DriveLikeAHuman



[自动驾驶为什么不直接让模型输出下一步的动作?](https://www.zhihu.com/question/598088657/answer/3099636476)

可以当然是可以的，英伟达2016年的那篇[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) 在自动驾驶行业引发了很大振动，几个摄像头输入到神经网络里，直接回归方向盘转动角度，就是这么简单粗暴。他们还进行了路测，使用第一代DrivePX，说是98%的时间都可以正常行驶，中间有一段路连续开了15公里不需要任何人工介入。我记得当时我们的team lead跟我们说我们要失业了。。。

之后的路就好走了，要让端到端能够真正实现，我尽量把中间过程分拆出来单独学不就可以了？说白了就是把已有的自动驾驶系统，拆成一个一个的子系统，给每一个子系统安排一个模型去学，然后把他们连起来就可以了。这方面最成功的是Wayve公司的创始人Alex Kendall，牛津VGG组博士毕业，不确定度学习的一哥，博士论文就是写这个的。2018年就发表了基于不确定度学习的多任务网络。他的做法就是把整个模型弄成一个多任务模型，每一个任务都训练，最后用强化学习输出控制信号。特斯拉2019的那个多任务网络，其实是他的弟弟。。。他们这条路走通了，还比较成功。

这个端到端系统里面子任务的构成和划分，各个任务的连接方式都可以做文章，于是就出现了CVPR2023的UniAd：

[GitHub - OpenDriveLab/UniAD: [CVPR 2023 Best Paper] Planning-oriented Autonomous Driving](https://github.com/OpenDriveLab/UniAD)



[为什么这么多年直到马斯克出来才想到做可复用火箭？](https://www.zhihu.com/question/597238433/answer/3080541702)

因为运载火箭回收中最关键的数学问题之一直到2013年才解决。Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem

[火箭软着陆最优控制问题 (Soft Landing Optimal Control Problem)](https://zhuanlan.zhihu.com/p/673214838)。看到SpaceX火箭精准回收的时候，心情澎湃激动，作为控制优化的学习者，第一反应是想搞明白它究竟是怎么实现的，难点在哪。在这个领域，一篇关于最优控制的文章《Lossless convexification of nonconvex control bound and pointing constraints of the soft landing optimal control problem》做出了很大的贡献，于是尝试研读记录一下，希望我能搞明白一部分，不会的、错误的希望大佬读者们评论区批评指正。





下面是自动驾驶的岗位需求，由此可借鉴自动驾驶的专业划分和知识储备：

自动驾驶高端人才需求：
感知算法总监 200万+
决策规划算法总监 200万+
规控算法总监 200万+
自动驾驶仿真系统负责人 200万+
地图与定位算法总监 200万+
高精地图技术负责人 200万+
自动驾驶软件总监 200万+
自动驾驶首席架构师 200万+
自动驾驶总工 100万+
自动驾驶工程架构技术总监 200万+
深度学习算法总监 200万+
Base地点：北京，上海，深圳，杭州，成都
@职位数量大概100+，隶属于各个自动驾驶公司，有兴趣的朋友请加我好友私聊！
（对于缺乏对市场了解的朋友，可以帮助你了解市场情况以及各家自动驾驶公司的发展状况，根据你的需求来匹配合适的机会！）



深蓝学院课程《自动驾驶规划与控制》

