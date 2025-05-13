# 保存加载模型文件

* [返回上层目录](../pytorch.md)
* [保存和加载整个模型](#保存和加载整个模型)
  * [保存整个模型](#保存整个模型)
  * [加载整个模型](#加载整个模型)
* [仅保存和加载模型参数（推荐）](#仅保存和加载模型参数（推荐）)
  * [保存模型参数](#保存模型参数)
  * [加载模型参数](#加载模型参数)
* [使用jit保存为TorchScript格式](#使用jit保存为TorchScript格式)

pytorch官网保存加载模型文件的教程：

[https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

# 保存和加载整个模型

## 保存整个模型

```python
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 1))
print(net)

# 保存整个模型，包含模型结构和参数
torch.save(net, 'sample_model.pt')
```

输出：

```shell
Sequential(
  (0): Linear(in_features=128, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

## 加载整个模型

```python
import torch
import torch.nn as nn

# 加载整个模型，包含模型结构和参数
loaded_model = torch.load('sample_model.pt')
print(loaded_model)
```

输出：

```shell
Sequential(
  (0): Linear(in_features=128, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

# 仅保存和加载模型参数（推荐）

## 保存模型参数

```python
import torch
import torch.nn as nn
 
model = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 1))
 
# 保存整个模型
torch.save(model.state_dict(), 'sample_model.pt')
```

## 加载模型参数

```python
import torch
import torch.nn as nn
 
# 下载模型参数 并放到模型中
loaded_model = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, 1))
loaded_model.load_state_dict(torch.load('sample_model.pt'))
print(loaded_model)
```

输出：

```shell
Sequential(
  (0): Linear(in_features=128, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

你会好奇如果只是单纯的`torch.load('sample_model.pt')`会是什么？

那就打印一下看看：

```python
import torch
loaded_model = torch.load('sample_model.pt')
print(loaded_model)
```



```shell
OrderedDict([('0.weight', tensor([[-0.0798,  0.0245,  0.0880,  ..., -0.0812,  0.0253, -0.0277],
        [ 0.0382,  0.0644,  0.0483,  ...,  0.0039, -0.0329, -0.0226],
        [ 0.0399,  0.0307, -0.0601,  ...,  0.0154,  0.0748, -0.0678],
        ...,
        [ 0.0279, -0.0479,  0.0126,  ...,  0.0778,  0.0654,  0.0521],
        [ 0.0613,  0.0283,  0.0219,  ..., -0.0807,  0.0087, -0.0058],
        [ 0.0824,  0.0022,  0.0803,  ..., -0.0146,  0.0389, -0.0284]])), ('0.bias', tensor([ 0.0554,  0.0607, -0.0356,  0.0661, -0.0491, -0.0182, -0.0611,  0.0212,
         0.0386,  0.0012, -0.0663, -0.0005,  0.0487, -0.0223, -0.0781,  0.0154])), ('2.weight', tensor([[-0.0585, -0.0603,  0.0626, -0.2448,  0.0612,  0.0704, -0.0561, -0.0535,
         -0.2328, -0.2104, -0.2206, -0.1715,  0.2299, -0.2423, -0.0247, -0.0756]])), ('2.bias', tensor([0.0428]))])
```

可以发现，是字典（state_dict），包括了每一层的名字和参数，但是不是整个模型结构。

> 是什么state_dict：PyTorch中的state_dict是一个python字典对象，将每个层映射到其参数Tensor。state_dict对象存储模型的可学习参数，即权重和偏差，并且可以非常容易地序列化和保存。

# 加载模型参数时devece的选择

```python
model.load_state_dict(torch.load(path, map_location=device))
```

必须要加map_location=device吗，不加是不是就错了？比如我本来是在gpu上训练，然后训练中断，然后要重新加载模型参数

这个问题问得非常关键！我来分情况详细讲：

如果你写：

```python
model.load_state_dict(torch.load(path))
```

**默认行为**是：

- `torch.load(path)` 会**尝试**把保存时的设备（比如 GPU）一起恢复出来。
- 如果你现在**正在用的设备**和**保存时的设备**一样，比如你保存时在 GPU，现在加载时也是在 GPU，就没问题。

但是如果：

- 你保存时是在 GPU (`cuda`) 上，
- 现在加载是在 CPU（比如有时候只想在 CPU 调试一下），

就会直接报错：

> `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.`
>  （大意就是：找不到 CUDA 设备，但你保存的是 CUDA 的 Tensor）

所以，总结一下：

| 场景                                     | 要不要加 `map_location`              |
| ---------------------------------------- | ------------------------------------ |
| 保存时和加载时设备一致，比如都是 GPU     | **可以不加**                         |
| 保存时在 GPU，加载时在 CPU               | **必须加 map_location='cpu'**        |
| 保存时在 GPU:0，加载想放到 GPU:1         | **最好加 map_location='cuda:1'**     |
| 想明确控制加载到哪个设备，未来代码更健壮 | **建议统一加上 map_location=device** |

回到你的问题

> 比如我本来是在gpu上训练，然后训练中断，然后要重新加载模型参数

如果你的训练中断后还是在 GPU 上继续跑，理论上不加 `map_location` 是没问题的。

但**更安全、推荐做法是加**，像这样：

```python
model.load_state_dict(torch.load(path, map_location=device))
```

这样无论是继续在 GPU、还是未来切到 CPU 都不会出任何问题，**代码更健壮、可复用性更高。**

最后一个很重要的小细节！

即使你不加 `map_location`，但是你后面有一句：

```python
model.to(device)
```

那只是把 `model` 本身转到 device 上，但是 `load_state_dict` 加载出来的 tensor **如果设备不对**，之前就已经出错了。**到 model.to(device) 这一步已经来不及了。**

所以真正该处理的是在 `torch.load` 的时候，就用 `map_location` 指定好！

✅ 总结就是：**推荐加上 map_location=device。**

# 适合部署用的保存格式

## TorchScript和ONNX通用介绍

当我们从训练阶段进入部署阶段时，**再靠 Python + PyTorch 原生模型就远远不够了**，这时就需要将模型“导出”为更**独立、跨平台、稳定**的格式，比如：

🔧 一、TorchScript（PyTorch 官方部署格式）

TorchScript 是 PyTorch 的“部署版”，把模型编译成一个可以脱离 Python 的中间格式，适用于 C++、移动端、服务端部署。

✅ 两种方式导出 TorchScript：

1. 使用 `torch.jit.trace()`（适合无分支的网络）

```python
import torch

model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)

traced_model.save("model_traced.pt")
```

2. 使用 `torch.jit.script()`（适合有控制流的网络）

```python
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

✅ 加载并使用：

```python
model = torch.jit.load("model_traced.pt")
output = model(input_tensor)
```

📦 二、ONNX（Open Neural Network Exchange）

ONNX 是微软和 Facebook 主导的通用格式，可导入 TensorRT、OpenVINO、CoreML 等多种后端，是跨平台部署的首选。

✅ 导出为 ONNX：

```python
import torch

model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11)
```

✅ 优点：

- 可部署在 TensorRT、ONNXRuntime、OpenVINO、移动端等；
- 工业部署（自动驾驶、推理服务）大量使用；
- 与其他框架（如 TensorFlow）更好对接。

🧠 总结对比：

| 特性       | TorchScript              | ONNX                               |
| ---------- | ------------------------ | ---------------------------------- |
| 官方支持   | ✅ PyTorch 自家格式       | ✅ 跨平台工业标准                   |
| 控制流支持 | ✅ 好                     | ⚠️ 有限（不推荐太复杂分支）         |
| 部署平台   | C++/Android/iOS/LibTorch | TensorRT、ONNXRuntime、OpenVINO 等 |
| 使用门槛   | ✅ 简单，集成无缝         | ⚠️ 导出时要注意算子兼容性           |

🔚 如果你准备部署模型，建议路线是：

1. ✅ 保存 `state_dict()` 保留权重；
2. ✅ 写一个推理用模型类（只有 `forward()`）；
3. ✅ 使用 `torch.jit.trace()` 导出 TorchScript；
4. ✅ 或使用 `torch.onnx.export()` 导出 ONNX；
5. ✅ 用部署后端加载并做推理。

------

需要我帮你写一个从 `state_dict` 到 TorchScript/ONNX 的完整示例吗

需要

好的，下面我来给你写一个完整的示例：**从训练完成后保存 state_dict，到导出 TorchScript 和 ONNX 格式，用于部署的全过程**。

✅ 假设你有一个模型如下：

```python
# model.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_dim=10, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)
```

✅ 步骤一：训练完保存 `state_dict`

```python
# train_and_save.py
import torch
from model import MyModel

model = MyModel()
# 训练模型...
# model = train(model)

# 保存权重
torch.save(model.state_dict(), "model_weights.pth")
```

✅ 步骤二：加载权重并导出为 TorchScript 和 ONNX

```python
# export_model.py
import torch
from model import MyModel

# 1. 加载模型结构 & 权重
model = MyModel()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# 2. 创建一个示例输入张量
dummy_input = torch.randn(1, 10)

# ========= TorchScript =========
# 使用 trace 导出（适合无控制流）
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# ========= ONNX =========
torch.onnx.export(
    model,                     # 模型
    dummy_input,               # 示例输入
    "model.onnx",              # 保存路径
    export_params=True,        # 是否导出权重
    opset_version=11,          # ONNX 版本
    do_constant_folding=True,  # 常量折叠优化
    input_names=['input'],     # 输入张量名称
    output_names=['output'],   # 输出张量名称
    dynamic_axes={             # 支持动态 batch size
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

✅ 步骤三：部署时使用

🔸 TorchScript 加载与推理

```python
model = torch.jit.load("model_traced.pt")
model.eval()
output = model(torch.randn(1, 10))
```

🔸 ONNX 加载与推理（用 onnxruntime）

```python
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("model.onnx")
input_tensor = np.random.randn(1, 10).astype(np.float32)
outputs = ort_session.run(None, {"input": input_tensor})
print(outputs[0])
```

🧠 小结

| 格式         | 文件    | 适用场景                         |
| ------------ | ------- | -------------------------------- |
| `state_dict` | `.pth`  | 模型训练/保存权重                |
| TorchScript  | `.pt`   | C++/LibTorch/移动端部署          |
| ONNX         | `.onnx` | 跨平台推理引擎部署（TensorRT等） |

我也可以帮你做一个包含推理速度对比的版本，或者帮你封装为 CLI 命令行工具，要不要继续拓展？

需要

太好了，接下来我会为你提供两个部分：

✅ 一、推理速度对比（TorchScript vs ONNX）

我们使用同一个输入，对两个模型分别进行多次推理，测量平均耗时。

```python
# benchmark_inference.py
import time
import torch
import onnxruntime as ort
import numpy as np
from model import MyModel

# 输入数据
input_tensor = torch.randn(32, 10)
input_numpy = input_tensor.numpy()

# -------- TorchScript 推理 --------
script_model = torch.jit.load("model_traced.pt")
script_model.eval()

with torch.no_grad():
    start = time.time()
    for _ in range(1000):
        _ = script_model(input_tensor)
    torchscript_time = time.time() - start

# -------- ONNX 推理 --------
ort_session = ort.InferenceSession("model.onnx")
start = time.time()
for _ in range(1000):
    _ = ort_session.run(None, {"input": input_numpy})
onnx_time = time.time() - start

print(f"TorchScript time: {torchscript_time:.4f} s")
print(f"ONNX time      : {onnx_time:.4f} s")
```

✅ 二、封装为 CLI 工具

你可以写一个命令行脚本 `export_model.py`，支持以下功能：

- `--weights`：指定权重路径；
- `--output-dir`：输出文件保存位置；
- `--torchscript` / `--onnx`：选择导出格式；
- 自动构建并导出模型。

```python
# cli_export_model.py
import argparse
import torch
from model import MyModel
import os

def export(weights_path, output_dir, to_script, to_onnx):
    model = MyModel()
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    dummy_input = torch.randn(1, 10)

    if to_script:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(os.path.join(output_dir, "model_traced.pt"))
        print("✅ Exported TorchScript to model_traced.pt")

    if to_onnx:
        torch.onnx.export(
            model, dummy_input,
            os.path.join(output_dir, "model.onnx"),
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✅ Exported ONNX to model.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exported")
    parser.add_argument("--torchscript", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()

    export(args.weights, args.output_dir, args.torchscript, args.onnx)
```

运行示例：

```bash
python cli_export_model.py --weights model_weights.pth --torchscript --onnx
```

需要我继续拓展 ONNX 转 TensorRT 或封装成 Docker 部署镜像的内容吗？



推理时还需要加 with torch.no_grad(): 吗

是的，**在使用 PyTorch 进行推理时建议加上 with torch.no_grad():**，即使模型已经处于 `eval()` 模式。

✅ 原因如下：

1. **关闭梯度计算**：
    `torch.no_grad()` 会显式关闭梯度追踪，避免计算图的构建，节省内存和计算资源。
2. **更高的推理效率**：
    不使用 `no_grad()`，虽然你没有反向传播，但 PyTorch 默认仍会记录计算图，浪费内存、显存与时间。
3. **更安全**：
    在某些情况下，未使用 `no_grad()` 可能会导致显存泄漏，特别是多次推理的场景。

✅ 推荐写法（用于推理）：

```python
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

🚫 错误写法（虽然能跑但效率低）：

```python
model.eval()
output = model(input_tensor)  # 没加 no_grad，会记录梯度
```

是否还想了解多线程推理 / GPU 推理的一些加速技巧

需要我继续拓展 ONNX 转 TensorRT 或封装成 Docker 部署镜像的内容吗？  需要！都需要！

太好了！我们将继续分为两个部分来讲：

✅ 一、ONNX 转 TensorRT（适用于部署加速）

我们将使用 NVIDIA 的 `onnx-tensorrt` 工具或 `trtexec` 命令将 ONNX 转换为 TensorRT 引擎文件。

🛠 方法 1：使用 `trtexec`（推荐，官方支持）

```shell
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

- `--onnx`：输入 ONNX 模型路径；
- `--saveEngine`：导出 TensorRT 引擎文件；
- `--fp16`：开启半精度（需 GPU 支持），速度更快。

🛠 方法 2：使用 Python API 转换（需要 TensorRT Python bindings）

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30  # 1 GB
        builder.fp16_mode = True

        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        engine = builder.build_cuda_engine(network)

        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("✅ TensorRT engine saved to", engine_file_path)

build_engine("model.onnx", "model.trt")
```

> 💡 注意：需要安装 `tensorrt`, `pycuda`，并使用 NVIDIA GPU。

✅ 二、封装成 Docker 镜像部署

目标：让模型可以通过 HTTP 接收输入、返回推理结果。

1. 项目结构：

```s
project/
├── app.py              # 推理服务入口
├── model_traced.pt     # TorchScript 模型或 .onnx / .trt
├── requirements.txt    # Python依赖
├── Dockerfile          # 镜像定义
```

2. `app.py`（用 FastAPI 快速构建服务）

```python
from fastapi import FastAPI
import torch
from pydantic import BaseModel
import uvicorn

model = torch.jit.load("model_traced.pt")
model.eval()

app = FastAPI()

class InputData(BaseModel):
    values: list[float]  # 假设输入是一维向量

@app.post("/predict")
def predict(data: InputData):
    x = torch.tensor(data.values).unsqueeze(0)  # (1, D)
    with torch.no_grad():
        output = model(x)
    return {"prediction": output.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

3. `requirements.txt`

```
fastapi
uvicorn
torch
```

4. Dockerfile

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
```

5. 构建并运行镜像

```bash
docker build -t mymodel-server .
docker run -p 8000:8000 mymodel-server
```

📦 访问方式：

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"values": [0.1, 0.2, 0.3, ..., 0.9]}'
```

如果你希望用 `TensorRT` 加速推理部署到 Docker 容器中，我可以为你写一套 **基于 TensorRT + Flask 或 FastAPI 的 Docker 化部署方案**，需要我继续写这一版吗？

需要

好的！让我们一起完成 **基于 TensorRT 加速推理的 Docker 部署** 方案。我们将使用 TensorRT 来加速推理，然后通过 **FastAPI** 创建一个 REST API 接口进行模型推理。

✅ 基于 TensorRT + FastAPI 的 Docker 部署方案

1. 项目结构

```bash
project/
├── app.py              # 推理服务入口
├── model.trt           # TensorRT 引擎文件
├── requirements.txt    # Python依赖
├── Dockerfile          # 镜像定义
```

2. `app.py`（使用 FastAPI 和 TensorRT 进行推理）

```python
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from fastapi import FastAPI
from pydantic import BaseModel
import json

# TensorRT加载模型
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# 加载 TensorRT 引擎
engine = load_engine("model.trt")
context = engine.create_execution_context()

# FastAPI 初始化
app = FastAPI()

class InputData(BaseModel):
    values: list[float]  # 假设输入是一维向量

# TensorRT 推理函数
def infer(input_data):
    # 将输入数据转换为 numpy 数组并转为 TensorRT 输入格式
    input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)
    bindings = []

    # 分配 GPU 内存
    input_buffer = cuda.mem_alloc(input_array.nbytes)
    output_buffer = cuda.mem_alloc(input_array.nbytes)
    
    bindings.append(int(input_buffer))
    bindings.append(int(output_buffer))

    # 执行推理
    cuda.memcpy_htod(input_buffer, input_array)
    context.execute_v2(bindings)

    # 获取结果
    output_array = np.empty_like(input_array)
    cuda.memcpy_dtoh(output_array, output_buffer)
    return output_array.tolist()

@app.post("/predict")
def predict(data: InputData):
    result = infer(data.values)
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

3. `requirements.txt`（Python 依赖）

```
fastapi
uvicorn
numpy
pycuda
tensorrt
```

4. Dockerfile（定义 TensorRT 环境）

```Dockerfile
# 使用支持 NVIDIA GPU 的 Python 版本作为基础镜像
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# 安装 PyTorch, PyCUDA 和 TensorRT Python bindings
RUN pip3 install --upgrade pip
RUN pip3 install fastapi uvicorn numpy pycuda tensorrt

# 设置工作目录
WORKDIR /app

# 复制代码和模型到容器中
COPY . .

# 开放端口
EXPOSE 8000

# 启动应用
CMD ["python3", "app.py"]
```

5. 构建并运行 Docker 镜像

首先，确保你已经构建并保存了 TensorRT 引擎文件 `model.trt`（从 ONNX 模型导出为 TensorRT）。接着，在项目根目录下构建 Docker 镜像：

```bash
docker build -t tensorrt-server .
```

然后，使用以下命令运行 Docker 容器：

```bash
docker run --gpus all -p 8000:8000 tensorrt-server
```

这里的 `--gpus all` 确保容器可以访问所有可用的 GPU。

6. 测试 REST API

假设你的模型接收一个包含浮动数值的输入向量，并返回预测结果。可以通过 `curl` 或 Postman 来测试 API：

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"values": [0.1, 0.2, 0.3, 0.4, 0.5]}'
```

你应该能看到类似这样的输出：

```json
{
  "prediction": [0.1234, 0.5678, 0.9101]
}
```

📝 总结

1. **TensorRT** 加速了模型推理，使用 `.trt` 文件代替常规的 ONNX 或 PyTorch 模型。
2. 使用 **FastAPI** 提供了一个高效的 HTTP API 服务，支持与外部系统进行交互。
3. **Docker** 容器化了推理服务，方便在任何环境中部署并使用 NVIDIA GPU 加速。

如果你还需要更多关于 **TensorRT + FastAPI + Docker** 的信息，或者有任何具体的优化问题，可以继续告诉我！

## 使用jit保存为TorchScript格式

保存为这个格式是为了进行推理。该格式在推理时可不用定义模型的类。该格式也可以在C++中进行推理。

在[https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)简介如下：

> One common way to do inference with a trained model is to use [TorchScript](https://pytorch.org/docs/stable/jit.html), an intermediate representation of a PyTorch model that can be run in Python as well as in a high performance environment like C++. TorchScript is actually the recommended model format for scaled inference and deployment.
>
> ```
> Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
> ```

保存的代码：

```python
model.eval()
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

加eval是为了在推理前设置模型中的dropout和batch norm功能为eval模式。

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



# 更专业的保存和加载模型函数

要不要我顺便也给你列一下更专业的 `save_model` 和 `load_model` 封装函数？
 比如以后要保存 optimizer 一起存、保存 epoch、训练步数之类，比较适合正式训练用的？
 （如果想要的话，我可以一并补上，顺便帮你整理成好用的小模板~）要不要？✨

那我来给你补一个**正式版**的模型保存/加载模板，简单、干净，方便你以后一直用。
 （包括：模型、优化器、学习率调度器、当前训练轮次，通通保存起来。）

## 保存模型的函数 save_model

```python
import torch
import os

def save_model(save_dir, model, optimizer=None, scheduler=None, epoch=None, extra_info=None):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "checkpoint.pth")

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if extra_info is not None:
        checkpoint['extra_info'] = extra_info

    torch.save(checkpoint, save_path)
    print(f"✅ 模型保存到 {save_path}")
```

## 加载模型的函数load_model

```python
def load_model(load_dir, model, optimizer=None, scheduler=None, map_location='cpu'):
    load_path = os.path.join(load_dir, "checkpoint.pth")
    checkpoint = torch.load(load_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', None)
    extra_info = checkpoint.get('extra_info', None)

    print(f"✅ 模型从 {load_path} 加载成功")
    return epoch, extra_info
```

## 用法示范

保存时：

```python
save_model(
    save_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=current_epoch,
    extra_info={'update_steps': model.update_steps.item()}
)
```

加载时：

```python
epoch, extra_info = load_model(
    load_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    map_location='cuda'  # 或 'cpu'
)

# 如果需要，可以取回额外保存的信息
if extra_info is not None:
    model.update_steps = torch.tensor(extra_info['update_steps'])
```

简单解释一下

- `extra_info` 是我特意加的灵活接口，假如以后想保存一些**奇奇怪怪但重要的小东西**（比如 replay buffer 里的数据、特殊统计指标等等），可以直接塞进去保存。
- 现在 `save_model` 和 `load_model` 都支持：
  - 只保存模型
  - 模型+优化器
  - 模型+优化器+调度器
  - 模型+优化器+调度器+epoch信息+其他信息
- 加载的时候，能恢复 optimizer 和 scheduler 的状态，这样继续训练也不会断。

非常通用，不管是强化学习、监督学习、预训练、微调，一套走天下。

## 多个版本保存加载

要不要我顺便再教你怎么**保存多个版本**，比如加时间戳，每次保存一版，防止被覆盖？✨
 （这个也超实用，尤其是大项目训练时候）要的话告诉我～

既然你想要保存多个版本的模型，这里我给你提供一个加时间戳保存的版本，防止每次训练都覆盖模型文件。

### 加时间戳的模型保存

```pytorch
import torch
import os
import time

def save_model_with_timestamp(save_dir, model, optimizer=None, scheduler=None, epoch=None, extra_info=None):
    # 根据当前时间生成唯一的文件夹名
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_path = os.path.join(save_dir, f"checkpoint_{timestamp}.pth")

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if extra_info is not None:
        checkpoint['extra_info'] = extra_info

    torch.save(checkpoint, save_path)
    print(f"✅ 模型保存到 {save_path}")
```

### 加载时选择最近的模型

你可以加载目录下最新的模型文件，这样就不需要手动选文件名了。

```
import glob

def load_latest_model(save_dir, model, optimizer=None, scheduler=None, map_location='cpu'):
    # 获取目录下所有的 checkpoint 文件
    checkpoint_paths = glob.glob(os.path.join(save_dir, 'checkpoint_*.pth'))

    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"没有找到任何模型文件在 {save_dir} 目录下")

    # 按照文件名排序，选择最新的模型
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)

    checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', None)
    extra_info = checkpoint.get('extra_info', None)

    print(f"✅ 最新模型从 {latest_checkpoint_path} 加载成功")
    return epoch, extra_info
```

### 用法示范

#### 保存时（带时间戳）

```python
save_model_with_timestamp(
    save_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=current_epoch,
    extra_info={'update_steps': model.update_steps.item()}
)
```

#### 加载时（加载最新的模型）

```python
epoch, extra_info = load_latest_model(
    save_dir="./save_dir",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    map_location='cuda'  # 或 'cpu'
)

# 恢复额外信息
if extra_info is not None:
    model.update_steps = torch.tensor(extra_info['update_steps'])
```

主要特点

1. **文件名自动加时间戳**：每次保存都会生成一个独特的文件名（例如 `checkpoint_20250427_153210.pth`），避免了文件覆盖。
2. **自动加载最新模型**：`load_latest_model` 会自动找到目录下最新的模型文件，节省手动查找的麻烦。
3. **灵活保存**：不仅保存了模型，还可以保存优化器、调度器、训练轮次等，完全可以恢复训练。

这样做，每次保存一个新的模型文件，模型版本逐渐增多时，你可以轻松恢复到最近的训练状态。如果以后想要回溯到之前的某个版本，也可以通过文件名查看哪个最合适。

这下你就能完全掌控模型版本管理啦！🎉

## 普通类的保存和加载

比如一个自己实现的一个lr的scheduler，我该怎么写def state_dict(self)和def load_state_dict(self, state_dict)？

非常好的问题！如果你写一个**自定义的类**，比如一个学习率调度器（scheduler），想要像 PyTorch 模块那样支持 `state_dict()` 和 `load_state_dict()`，你只需要实现以下两个方法：

✅ 一般推荐的写法如下：

```python
from collections import OrderedDict

class MyLRScheduler:
    def __init__(self, base_lr=0.01, decay_rate=0.9):
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.step_num = 0

    def step(self):
        self.step_num += 1

    def get_lr(self):
        return self.base_lr * (self.decay_rate ** self.step_num)

    def state_dict(self):
        return OrderedDict({
            "version": 1,
            "params": OrderedDict({
                "base_lr": self.base_lr,
                "decay_rate": self.decay_rate,
            }),
            "state": OrderedDict({
                "step_num": self.step_num,
            })
        })

    def load_state_dict(self, state_dict):
        assert state_dict["version"] == 1, "Unsupported state version"
        self.base_lr = state_dict["params"]["base_lr"]
        self.decay_rate = state_dict["params"]["decay_rate"]
        self.step_num = state_dict["state"]["step_num"]
```

🧠 使用方式示例：

```python
scheduler = MyLRScheduler(0.01, 0.95)
scheduler.step()
scheduler.step()

# 保存
ckpt = scheduler.state_dict()

# 加载
new_scheduler = MyLRScheduler(0.1, 0.5)
new_scheduler.load_state_dict(ckpt)
```

🧩 设计建议：

| 项目         | 建议说明                                                     |
| ------------ | ------------------------------------------------------------ |
| 可保存的变量 | 仅保存能影响调度器行为的关键变量（如超参、内部状态）         |
| 参数类型     | 推荐使用基本类型（int, float, str, list, dict）              |
| 键的命名     | 使用双引号 `"key"` 风格，和 PyTorch 保持一致                 |
| 向后兼容     | 如果将来你升级版本，可以加上 `"version"` 字段方便兼容老的 state_dict |

🧩 扩展技巧（如果你有多参数组或自定义逻辑）

```python
def state_dict(self):
    return {
        "version": 1,
        "params": {
            "base_lr": self.base_lr,
            "decay_rate": self.decay_rate,
        },
        "state": {
            "step_num": self.step_num,
        }
    }
```

这样将超参数和状态变量分离，便于维护。

那对应的load怎么写呢

你已经构建了一个结构清晰的 `state_dict`。那对应的 `load_state_dict` 应该严格按照相同的结构去读取，推荐如下写法：

✅ 对应的 `load_state_dict` 示例：

```python
def load_state_dict(self, state_dict):
    # 可选：检查版本号，确保兼容性
    version = state_dict.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported state_dict version: {version}")

    # 加载参数
    params = state_dict["params"]
    self.base_lr = params["base_lr"]
    self.decay_rate = params["decay_rate"]

    # 加载状态
    state = state_dict["state"]
    self.step_num = state["step_num"]
```

🔒 加强健壮性（推荐做法）

如果你希望这个调度器未来可以热加载旧版本，或者用于模型迁移，建议加上一些默认值或容错：

```python
def load_state_dict(self, state_dict):
    version = state_dict.get("version", 1)

    if version == 1:
        params = state_dict.get("params", {})
        self.base_lr = params.get("base_lr", self.base_lr)
        self.decay_rate = params.get("decay_rate", self.decay_rate)

        state = state_dict.get("state", {})
        self.step_num = state.get("step_num", 0)
    else:
        raise ValueError(f"Unsupported state_dict version: {version}")
```



# 参考资料

* [pytorch模型的保存与加载](https://blog.csdn.net/lsb2002/article/details/131969478)

本文参考此资料。

