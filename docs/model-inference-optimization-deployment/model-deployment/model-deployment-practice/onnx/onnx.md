# ONNX

- [返回上层目录](../model-deployment-practice.md)



# 模型导出为ONNX格式

写一个包装模型类，把 `act()` 变成 `forward()`

```
class ActorCriticWrapper(nn.Module):
    def __init__(self, actor_critic_model):
        super().__init__()
        self.model = actor_critic_model

    def forward(self, x):
        return self.model.act(x)  # 注意这里是调用 act 而不是 forward
```

然后这样导出：

```python
model = ActorCritic()
model.eval()
wrapped_model = ActorCriticWrapper(model)

dummy_input = torch.randn(1, 5)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["a1", "a2", "param_mean"],
                  opset_version=11,
                  dynamic_axes={  # 加上这个就会变为batch_size变为动态的
                      "input": {0: "batch_size"},  # 让第0维（batch）是动态的
                      "a1": {0: "batch_size"},
                      "a2": {0: "batch_size"},
                      "param_mean": {0: "batch_size"}
                  })
```

# 模型推理

## python端推理

```python
ort_session = ort.InferenceSession("model.onnx")
input_tensor = np.random.randn(2, 5).astype(np.float32)
outputs = ort_session.run(None, {"input": input_tensor})
print(outputs)
# [array([2, 2], dtype=int64),
#  array([1, 1], dtype=int64),
#  array([[-0.09347501, 0.10457519, 0.09422486, -0.24867839],
#         [-0.09440675, 0.10584655, 0.08844154, -0.24860077]], dtype=float32)
# ]
a1, a2, param_mean = outputs
print(a1)
# [2 2]
print(a2)
# [1 1]
print(param_mean)
# [[-0.09347501  0.10457519  0.09422486 -0.24867839]
#  [-0.09440675  0.10584655  0.08844154 -0.24860077]]
```



