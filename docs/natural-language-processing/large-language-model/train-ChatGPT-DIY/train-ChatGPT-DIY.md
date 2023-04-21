# 自己训练ChatGPT

* [返回上层目录](../large-language-model.md)



[怎么训练高性能计算ChatGPT？](https://www.zhihu.com/question/571182694/answer/2942567980)

最近的一些新的模型技术发展让大家发现可以用8.5万美元来训练一个和ChatGPT差不多的模型。因为类似ChatGPT模型本身就是一个大的生成模型GPT-3加上[指令微调](https://www.zhihu.com/search?q=指令微调&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2942567980})变成ChatGPT，因此，只需要找到一个低成本的大模型训练方法加一个低成本指令微调方法即可。也就是本文介绍的2个开源方法：LLaMA+Alpaca



[怎么训练高性能计算ChatGPT？](https://www.zhihu.com/question/571182694/answer/2797310334)

蓝海大脑AI人工智能液冷工作站研究人员表示：

在“人工标注数据+强化学习”框架下，具体而言，ChatGPT的训练过程分为以下三个阶段：

第一阶段：冷启动阶段的监督策略模型。靠[GPT 3.5](https://www.zhihu.com/search?q=GPT 3.5&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})本身，尽管它很强，但是它很难理解人类不同类型指令中蕴含的不同意图，也很难判断生成内容是否是高质量的结果。为了让GPT 3.5初步具备理解指令中蕴含的意图，首先会从测试用户提交的[prompt](https://www.zhihu.com/search?q=prompt&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})(就是指令或问题)中随机抽取一批，靠专业的标注人员，给出指定prompt的高质量答案，然后用这些人工标注好的<prompt,answer>数据来Fine-tune GPT 3.5模型。经过这个过程，我们可以认为GPT 3.5初步具备了理解人类prompt中所包含意图，并根据这个意图给出相对高质量回答的能力，但是很明显，仅仅这样做是不够的。

第二阶段：训练回报模型（Reward Model,RM）。这个阶段的主要目的是通过人工标注训练数据，来训练回报模型。具体而言，随机抽样一批用户提交的prompt(大部分和第一阶段的相同)，使用第一阶段Fine-tune好的冷启动模型，对于每个prompt，由[冷启动模型](https://www.zhihu.com/search?q=冷启动模型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})生成K个不同的回答，于是模型产生出了<prompt,answer1>,<prompt,answer2>….<prompt,answerK>数据。之后，标注人员对K个结果按照很多标准（上面提到的相关性、富含信息性、有害信息等诸多标准）综合考虑进行排序，给出K个结果的排名顺序，这就是此阶段人工标注的数据。

接下来，我们准备利用这个排序结果数据来训练回报模型，采取的训练模式其实就是平常经常用到的pair-wise learning to rank。对于K个排序结果，两两组合，形成图片个训练数据对，ChatGPT采取[pair-wise loss](https://www.zhihu.com/search?q=pair-wise loss&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})来训练Reward Model。RM模型接受一个输入<prompt,answer>，给出评价回答质量高低的回报分数Score。对于一对训练数据<answer1,answer2>，我们假设人工排序中answer1排在answer2前面，那么Loss函数则鼓励RM模型对<prompt,answer1>的打分要比<prompt,answer2>的打分要高。

第三阶段：采用强化学习来增强预训练模型的能力。本阶段无需人工标注数据，而是利用上一阶段学好的RM模型，靠RM打分结果来更新预训练模型参数。具体而言，首先，从用户提交的prompt里[随机采样](https://www.zhihu.com/search?q=随机采样&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})一批新的命令（指的是和第一第二阶段不同的新的prompt，这个其实是很重要的，对于提升LLM模型理解instruct指令的[泛化能力](https://www.zhihu.com/search?q=泛化能力&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})很有帮助），且由冷启动模型来初始化PPO模型的参数。然后，对于随机抽取的prompt，使用PPO模型生成回答answer， 并用上一阶段训练好的RM模型给出answer质量评估的回报分数score，这个回报分数就是RM赋予给整个回答（由单词序列构成）的整体[reward](https://www.zhihu.com/search?q=reward&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2797310334})。有了单词序列的最终回报，就可以把每个单词看作一个时间步，把reward由后往前依次传递，由此产生的策略梯度可以更新PPO模型参数。这是标准的强化学习过程，目的是训练LLM产生高reward的答案，也即是产生符合RM标准的高质量回答。

如果我们不断重复第二和第三阶段，很明显，每一轮迭代都使得LLM模型能力越来越强。因为第二阶段通过人工标注数据来增强RM模型的能力，而第三阶段，经过增强的RM模型对新prompt产生的回答打分会更准，并利用强化学习来鼓励LLM模型学习新的高质量内容，这起到了类似利用伪标签扩充高质量训练数据的作用，于是LLM模型进一步得到增强。显然，第二阶段和第三阶段有相互促进的作用，这是为何不断迭代会有持续增强效果的原因。

尽管如此，我觉得第三阶段采用强化学习策略，未必是ChatGPT模型效果特别好的主要原因。假设第三阶段不采用强化学习，换成如下方法：类似第二阶段的做法，对于一个新的prompt，冷启动模型可以产生k个回答，由RM模型分别打分，我们选择得分最高的回答，构成新的训练数据<prompt,answer>,去fine-tune LLM模型。假设换成这种模式，我相信起到的作用可能跟强化学习比，虽然没那么精巧，但是效果也未必一定就差很多。第三阶段无论采取哪种技术模式，本质上很可能都是利用第二阶段学会的RM，起到了扩充LLM模型高质量训练数据的作用。