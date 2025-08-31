# 测试是否支持pytorch

* [返回上层目录](../pytorch.md)

测试一下新的gpu硬件、cuda、pytorch-gpu版本是否能否支持深度学习。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def test_gpu_pytorch():
    print("=== PyTorch GPU 测试 ===")
    
    # 1. 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 2. 检查CUDA是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 3. 创建一个简单的神经网络
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 4. 创建模型、数据、优化器
        device = torch.device("cuda:0")
        model = SimpleNet().to(device)
        
        # 创建随机数据
        x = torch.randn(100, 10).to(device)
        y = torch.randn(100, 1).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        print(f"\n模型已移动到GPU: {device}")
        print(f"输入数据形状: {x.shape}")
        print(f"目标数据形状: {y.shape}")
        
        # 5. 训练过程
        print("\n开始训练...")
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
        
        print("训练完成!")
        
        # 6. 测试推理
        with torch.no_grad():
            test_input = torch.randn(5, 10).to(device)
            test_output = model(test_input)
            print(f"\n测试推理:")
            print(f"输入: {test_input.shape}")
            print(f"输出: {test_output.shape}")
            print(f"输出值: {test_output.squeeze()}")
        
        # 7. 检查内存使用
        print(f"\nGPU内存使用:")
        print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"已缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
    else:
        print("CUDA不可用，请检查显卡驱动和PyTorch安装")
        print("建议安装支持CUDA的PyTorch版本")

if __name__ == "__main__":
    test_gpu_pytorch()
```

测试结果为

```shell
(torch) user_name@LAPTOP:~$ python test.py
=== PyTorch GPU 测试 ===
PyTorch版本: 2.8.0+cu128
CUDA是否可用: True
CUDA版本: 12.8
GPU数量: 1
当前GPU: 0
GPU名称: NVIDIA GeForce RTX 5090 Laptop GPU

模型已移动到GPU: cuda:0
输入数据形状: torch.Size([100, 10])
目标数据形状: torch.Size([100, 1])

开始训练...
Epoch [2/10], Loss: 1.0887
Epoch [4/10], Loss: 1.0843
Epoch [6/10], Loss: 1.0803
Epoch [8/10], Loss: 1.0765
Epoch [10/10], Loss: 1.0730
训练完成!

测试推理:
输入: torch.Size([5, 10])
输出: torch.Size([5, 1])
输出值: tensor([0.0729, 0.1185, 0.1223, 0.1391, 0.1760], device='cuda:0')

GPU内存使用:
已分配: 16.26 MB
已缓存: 22.00 MB
```

