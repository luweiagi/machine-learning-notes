# 推理inference

* [返回上层目录](../pytorch.md)
* [使用jit保存加载模型的TorchScript格式推理](#使用jit保存加载模型的TorchScript格式推理)

pytorch官网保存加载模型文件的教程：

[https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

# 使用jit保存加载模型的TorchScript格式推理

保存为这个格式是为了进行推理。该格式在推理时可不用定义模型的类。该格式也可以在C++中进行推理。

在[https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)简介如下：

> One common way to do inference with a trained model is to use [TorchScript](https://pytorch.org/docs/stable/jit.html), an intermediate representation of a PyTorch model that can be run in Python as well as in a high performance environment like C++. TorchScript is actually the recommended model format for scaled inference and deployment.
>
> ```
> Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
> ```

保存的代码：

```python
# Export to TorchScript
model_scripted = torch.jit.script(model)
# Save
model_scripted.save('model_scripted.pt')
```

进行推理的代码：

```python
model = torch.jit.load('model_scripted.pt')
model.eval()
```

加eval是为了在推理前设置模型中的dropout和batch norm功能为eval模式，即使得模型BN层等失效。

> eval()是PyTorch中用来将神经网络设置为评估模式的方法。在评估模式下，网络的参数不会被更新，Dropout和Batch Normalization层的行为也会有所不同。通常在测试阶段使用评估模式。
>
> eval() 可以作为模型推理的性能提升方法，在评估模式下，计算图是不被跟踪的，这样可以节省内存使用，提升性能。还可以使用torch.no_grad()配合使用，在评估阶段关闭梯度跟踪，进一步提升性能。

给一段kimi给出的例子吧：

```python
import torch

# 加载训练好的模型，这里假设模型已经保存为'model.pth'
model = torch.jit.load('model.pth')
model.eval()  # 将模型设置为评估模式

# 准备输入数据
# 这里需要根据你的模型输入进行相应的数据预处理
# 例如，如果你的模型接受的是图像数据，你需要进行图像的读取和预处理
# 示例中使用随机数据作为输入
input_data = torch.randn(1, 3, 224, 224)  # 假设输入是单张3通道224x224的图像

# 进行推理
with torch.no_grad():
    output = model(input_data)

# 处理输出
# 根据你的任务类型，这里可能是分类、回归或其他
# 例如，如果是分类任务，你可能需要找到概率最高的类别索引
_, predicted_class = torch.max(output, 1)

print(f'Predicted class index: {predicted_class.item()}')
```

注意了：对于上面例子中的`input_data`，格式是[[data], ...]这种的，即外面是有个维度的，这就是batch的数量，如果你是从np生成的单个数据，那大概率是没有外面的这个batch维度的，那就需要加上。方法有两种：

```python
data = np.expand_dims(data, axis=0)
# 或者
s = torch.unsqueeze(torch.tensor(input, dtype=torch.float), 0)
```



# 参考资料

* [pytorch模型的保存与加载](https://blog.csdn.net/lsb2002/article/details/131969478)

本文参考此资料。

===

* [pytorch导出模型并使用onnxruntime C++部署加载模型推理](https://zhuanlan.zhihu.com/p/610188093)