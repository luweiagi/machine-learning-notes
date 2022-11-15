# reinforcement-learning

* [返回上层目录](../advanced-knowledge.md)
* [DRN: A Deep Reinforcement Learning Framework for News Recommendation WWW2018](drn/DRN-A-Deep-Reinforcement-Learning-Framework-for-News-Recommendation.md)



# 为什么要将强化学习用在推荐系统上

作为一个**千亿级数据量**的从业者，我讲讲我认为推荐系统中**最重要的几点**，可能与其他回答都略有不同

1. **不同规模下的工程架构：**特征从**百**到**百万**到**百亿**，不同级别的工程架构相差极大
2. **对目标的选定：**如何选择你的目标，决定了怎么做画像、特征，改变一个目标非常的伤筋动骨，而且也无法说清目标的制定是否科学
3. **对长期目标的学习：**短期的目标可以是一跳（用户的单次成本，付费或者消费），但长期的目标一定是用户付出的长期成本（长期消费，用户粘性），怎么去学习，是非常困难的事情。很多公司、学校都在进行这方面的研究（1、2、3），可以参考

这几个点很难绕过，未来几年也会成为各家推荐的差异点。核心技术说实话大家都非常清楚，Wide & Deep已经应用的非常广泛，这剩余的核心问题就看谁能够解决的足够快、跑的足够前面了。



# 参考文献

* [推荐系统有哪些坑？-Geek An](https://www.zhihu.com/question/28247353/answer/399162539)

"为什么要将强化学习用在推荐系统上"一节参考了此回答。

===

[增强学习在推荐系统有什么最新进展？](https://www.zhihu.com/question/57388498/answer/570874226)



[1] Dulac-Arnold G, Evans R, van Hasselt H, et al. Deep reinforcement learning in large discrete action spaces[J]. arXiv preprint arXiv:1512.07679, 2015.

[2] Liebman E, Saar-Tsechansky M, Stone P. Dj-mc: A reinforcement-learning agent for music playlist recommendation[C]//Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems. International Foundation for Autonomous Agents and Multiagent Systems, 2015: 591-599.

[3] Zheng G, Zhang F, Zheng Z, et al. DRN: A Deep Reinforcement Learning Framework for News Recommendation[C]//Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018: 167-176.

[4] Lixin Zou, Long Xia, Zhuoye Ding, Jiaxing Song, Weidong Liu, Dawei Yin: [Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems](http://export.arxiv.org/abs/1902.05570)[C]KDD 2019

清华大学和京东发表于 KDD 2019 的全新强化学习框架 FeedRec

[5] Youtube RL Recommendation: Top-k Off-Policy Correction for a REINFORCE Recommender System , Google, WSDM, 2019



[【论文复现】Top-K Off-Policy Correction for a REINFORCE RS论文复现](https://mp.weixin.qq.com/s?__biz=MzU0MTgxNDkxOA%3D%3D&chksm=fb256899cc52e18fb17d9adb2898d58360dce1e99b5e9b809bd2ca417d7277948e24a8f8ef52&idx=1&mid=2247487856&scene=21&sn=9ece274646716907367ad1f746c993a4#wechat_redirect)









