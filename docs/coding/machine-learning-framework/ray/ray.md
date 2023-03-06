# Ray分布式计算框架

* [返回上层目录](../machine-learning-framework.md)



github: [ray-project/ray](https://github.com/ray-project/ray)

> Ray is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries (Ray AIR) for accelerating ML workloads.



===

[ChatGPT背后的开源AI框架Ray，现在值10亿美元](https://mp.weixin.qq.com/s/6XLWpsCJZRalDs3-yMX4dg)



光说显卡，chatgpt大概用800G显存，你用4090堆比较亏，要30多张并联。但你又买不到性价比高一点的A100，也没有能并联30张4090的主板，更没有多主板还能并行计算的技术

对了比较准确的资料显示训练一次450W美金左右

请问这些技术怎么有啊？国内有吗？openai是怎么解决多个显卡并联训练的？

显卡并联在ai训练是很基础的技术了，没什么稀奇的，gpu计算本来就可以并行

您好，我知道有单机多卡，请问多机多卡，也就是您说的，多主板并行计算，比如10张每张能插8个卡的主板，这10个主板（共10x8个GPU单卡）能并联训练吗？谢谢！

能啊，就说最傻的方法，计算能并行就能搞传统MapReduce，分过去再回来。虽然这样损失很大，现在应该用Ray的比较多吧

