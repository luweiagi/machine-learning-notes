# Pytorch基础

* [返回上层目录](../pytorch.md)
* [pytorch代码快速入门](#pytorch代码快速入门)



# pytorch代码快速入门

如果用过tensorflow（其实没用过也没关系），那么可以通过下面一段简单的代码来迅速入门pytorch。

```python
import numpy as np
import random
import torch

import torch
import torch.nn as nn
import torch.optim as optim

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 6)

    def forward(self, input):
        # y = a * x + b
        output = self.linear(input)
        return output

model = DummyModel()  # 实例化model
print("model =", model)
print("模型各层的参数 model.parameters() =")
for i, param in enumerate(model.parameters()):
    print("===>", i, ", type:", type(param), ", size:", param.size(), ",", param)

# 定义loss function
loss_fn = nn.MSELoss()
# 这里把model的参数传入到optimizer中
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

print("训练数据")
input = torch.randn(2, 5)
print("input =", input)
label = torch.randn(2, 6)
print("label =", label)

for epoch in range(100):
    pred = model(input)  # 计算结果
    loss = loss_fn(pred, label)  # 计算损失
    print("epoch: {}, loss: {}".format(epoch, round(float(loss), 5)))

    optimizer.zero_grad()  # 清空过往梯度
    loss.backward()  # 反向传播，计算当前梯度
    optimizer.step()  # 根据梯度更新网络参数
print("pred =", pred)
```

输出：

```
model = DummyModel(
  (linear): Linear(in_features=5, out_features=6, bias=True)
)
模型各层的参数 model.parameters() =
===> 0 , type: <class 'torch.nn.parameter.Parameter'> , size: torch.Size([6, 5]) , Parameter containing:
tensor([[ 0.2304, -0.1974, -0.0867,  0.2099, -0.4210],
        [ 0.2682, -0.0920,  0.2275,  0.0622, -0.0548],
        [ 0.1240,  0.0221,  0.1633, -0.1743, -0.0326],
        [-0.0403,  0.0648, -0.0018,  0.3909,  0.1392],
        [-0.1665, -0.2701, -0.0750, -0.1929, -0.1433],
        [ 0.0214,  0.2666,  0.2431, -0.4372,  0.2772]], requires_grad=True)
===> 1 , type: <class 'torch.nn.parameter.Parameter'> , size: torch.Size([6]) , Parameter containing:
tensor([ 0.1249,  0.4242,  0.2952, -0.4075, -0.4252, -0.2157],
       requires_grad=True)
训练数据
input = tensor([[-1.5727, -0.1232,  3.5870, -1.8313,  1.5987],
        [-1.2770,  0.3255, -0.4791,  1.3790,  2.5286]])
label = tensor([[ 0.4107, -0.9880, -0.9081,  0.5423,  0.1103, -2.2590],
        [ 0.6067, -0.1383,  0.8310, -0.2477, -0.8029,  0.2366]])
epoch: 0, loss: 2.78025
epoch: 1, loss: 1.22228
epoch: 2, loss: 0.14918
epoch: 3, loss: 0.56846
epoch: 4, loss: 1.42182
epoch: 5, loss: 1.39428
epoch: 6, loss: 0.64793
epoch: 7, loss: 0.2631
epoch: 8, loss: 0.60053
epoch: 9, loss: 0.95843
epoch: 10, loss: 0.72269
...
epoch: 15, loss: 0.4925
...
epoch: 20, loss: 0.25056
...
epoch: 40, loss: 0.01546
...
epoch: 50, loss: 0.00616
...
epoch: 90, loss: 0.0
...
epoch: 98, loss: 6e-05
epoch: 99, loss: 2e-05
pred = tensor([[ 0.4057, -0.9840, -0.9035,  0.5389,  0.1093, -2.2489],
        [ 0.6030, -0.1382,  0.8286, -0.2459, -0.8032,  0.2358]],
       grad_fn=<AddmmBackward0>)
```



# DataLoader详解

`torch.utils.data.DataLoader`函数详解

应用实例：

```python
'''
批训练：把数据分为一小批一小批进行训练
Dataloader就是用来包装使用的数据，
比如说该程序中把数据5个5个的打包，
每一次抛出一组数据进行操作。
'''
import torch
import torch.utils.data as Data
torch.manual_seed(1)
BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y) #把数据放在数据库中
loader = Data.DataLoader(
    # 从dataset数据库中每次抽出batch_size个数据
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,#将数据打乱
    num_workers=2, #使用两个线程
)
def show_batch():
    for epoch in range(3): #对全部数据进行3次训练
        for step,(batch_x,batch_y) in enumerate(loader): # 每一次挑选出来的size个数据
            # training
            # 打印出来，观察数据
            print('Epoch:',epoch,'|Step:',step,'|batch x:',
                  batch_x.numpy(),'|batch y:',batch_y.numpy())

if __name__ == '__main__':
    show_batch()
```

结果：

```python
Epoch: 0 |Step: 0 |batch x: [ 5.  7. 10.  3.  4.] |batch y: [6. 4. 1. 8. 7.]
Epoch: 0 |Step: 1 |batch x: [2. 1. 8. 9. 6.] |batch y: [ 9. 10.  3.  2.  5.]
Epoch: 1 |Step: 0 |batch x: [ 4.  6.  7. 10.  8.] |batch y: [7. 5. 4. 1. 3.]
Epoch: 1 |Step: 1 |batch x: [5. 3. 2. 1. 9.] |batch y: [ 6.  8.  9. 10.  2.]
Epoch: 2 |Step: 0 |batch x: [ 4.  2.  5.  6. 10.] |batch y: [7. 9. 6. 5. 1.]
Epoch: 2 |Step: 1 |batch x: [3. 9. 1. 8. 7.] |batch y: [ 8.  2. 10.  3.  4.]
```







# 学习率调整

自定义：

```python
class SchedulerCosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = 0
    
    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total= self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1

# 使用时：
scheduler = SchedulerCosineDecayWarmup(opt, 0.2, 200, 2000)
```

# 模型保存与加载

把原先模型eval，把加载后模型eval。才保证两模型完全一样。

保存模型

```python
torch.save(net,'./model.pth')
# 或者
torch.save(net.state_dict(),'./model-dict.pth')
```

加载模型

```
net=torch.load('./model.pth')
或者
net.load_state_dict(torch.load('./model-dict.pth'))
```

加载后不同

如果不指定模型eval模式，那么加载回来的模型并不是和原先保存的模型相同。

简单说，原先的net你要eval一下，load之后的net也要eval一下，把所有参数freeze掉。才保证两个net完全相同（输入相同tensor得到完全一致的结果）。

```python
#保存
net=net.eval()
torch.save(net,'./model.pth')
torch.save(net.state_dict(),'./model-dict.pth')

#加载
net_load1=torch.load('./model.pth')
net_load1=net_load1.eval()
#或者
net_load2.load_state_dict(torch.load('./model-dict.pth'))
net_load2=net_load2.eval()

#此时net和net_load1、net_load2完全一样。
```

# 基本算符

## tensor.expand()



```python
import torch

x = torch.tensor([1, 2, 3, 4])
print(x)  # tensor([1, 2, 3, 4])
print(x.shape)  # torch.Size([4])

x1 = x.expand(2, 4)
print(x1)
# tensor([[1, 2, 3, 4],
#         [1, 2, 3, 4]])
print(x1.shape)  # torch.Size([2, 4])
```

参数为传入指定shape，在原shape数据上进行高维拓维，根据维度值进行重复赋值。

注意：

* 只能拓展维度，比如A的shape为`2x4`的，不能`A.expend(1,4)`，只能保证原结构不变，在前面增维，比如`A.shape(1,1,4)`
* 可以增加多维，比如x的shape为`[4]`，`x.expend(2,2,1,4)`只需保证本身是4
* 不能拓展低维，比如x的shape为`[4]`，不能`x.expend(4,2)`





## nn.Threshold(threshold, value)

原理如下：
$$
y =
\begin{cases}
x, &\text{ if } x > \text{threshold} \\
\text{value}, &\text{ otherwise }
\end{cases}
$$
代码例子：

```python
import torch
x = torch.tensor([[1, 2, 3, 4], [11, 22, 33, 4]])
t = torch.nn.Threshold(10, 0)
print(t(x))
# tensor([[ 0,  0,  0,  0],
#         [11, 22, 33,  0]])
```



## tensor.item()和tensor.tolist()

- 按照官方文档，可以理解为从只包含**一个**元素的tensor中提取值，注意是只含有one element，其他情况用tolist()
- 在训练时统计loss的变化时，经常用到，否则会累积计算图，造成GPU的额外开销

```python
import torch

x = torch.tensor([1.0])
print(x)  # tensor([1.])
print(x.item())  # 1.0

x = torch.tensor([1.0, 2.0])
print(x.item())
# ValueError: only one element tensors can be converted to Python scalars
print(x.tolist())  # [1.0, 2.0]
```



## torch.no_grad()

被with torch.no_grad()包住的代码，不用跟踪反向梯度计算，来做一个实验：

```python
a = torch.tensor([1.1], requires_grad=True)
b = a * 2  # tensor([2.2000], grad_fn=<MulBackward0>)
print(b.add_(2))  # tensor([4.2000], grad_fn=<AddBackward0>)
with torch.no_grad():
    print(b.mul_(2))  # tensor([8.4000], grad_fn=<AddBackward0>)
    # 看见了吗，可以看到没有跟踪乘法的梯度，还是上面的加法的梯度函数，不过乘法是执行了的
```







# 参考资料

* [请问pytorch自定义的optimizer是怎么通过parameter和model参数联动的？](https://www.zhihu.com/question/446595749/answer/1752209053)

用于pytorch快速入门的代码就来自这里。

* [PyTorch中在反向传播前为什么要手动将梯度清零？](https://www.zhihu.com/question/303070254)

pytorch快速入门的代码中，`optimizer.zero_grad()  # 清空过往梯度`，讲了对这句话的理解。优化器的梯度不会自动清零，默认会累加，所以要手动清零，这样的好处是多了一些玩法，比如小batch多累积几次就可以模拟一个大的batch，但同时学习率要放大。

* [torch.utils.data.DataLoader函数详解](https://blog.csdn.net/seven08290/article/details/83097841?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1.no_search_link&spm=1001.2101.3001.4242.2)

"DataLoader详解"参考了此博客。

* [pytorch中的自带的学习率调整方案（带图示）](https://zhuanlan.zhihu.com/p/349973983)

“学习率调整”参考该文章，讲了几种常见的学习率调整方案。

* [torch load或者load_state_dict（）后模型不相同](https://yonggie.blog.csdn.net/article/details/105882087?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)

“模型保存与加载”参考此博客。

* [一句话理解torch.expand()](https://blog.csdn.net/weixin_42670810/article/details/114278285?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-8.highlightwordscore&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-8.highlightwordscore)

“tensor.expand()”参考此博客。

