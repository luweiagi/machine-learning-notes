# 基于Ray的强化学习并行训练框架

* [返回上层目录](../ray.md)
* [Ray并行训练方式介绍](#Ray并行训练方式介绍)



# Ray并行训练方式介绍

Ray是一个用于并行计算的框架，主要用于加速机器学习和强化学习任务。它可以在多个CPU核心或者多个机器上并行执行任务，极大地提升计算效率。Ray提供了两种主要方式来加速强化学习：

1. **自己基于Ray实现多环境并行（自定义Ray Actor）**
2. **直接使用Ray RLlib（Ray官方强化学习库）**

## 自己基于Ray实现多环境并行

**核心思路**：使用Ray的Actor模型，手动并行多个环境实例，采样数据，并喂给你的PPO算法。

**适用场景**：

- 你已经有自己的PPO代码（不是用Ray RLlib的PPO）。
- 你希望最大程度地控制代码逻辑，比如如何收集数据、如何处理环境交互等。

**实现方式**：

1. 用Ray的`@ray.remote`装饰器，把你的C++环境封装成Ray的Actor。
2. 在Python里启动多个环境实例，让它们并行运行。
3. 收集所有环境的经验数据，并传给你的PPO训练代码。

```python
import ray
import zmq
import numpy as np

ray.init()  # 初始化Ray

# 定义一个Ray的Actor，封装你的C++环境
@ray.remote
class EnvActor:
    def __init__(self, env_id):
        self.env_id = env_id
        np.random.seed(env_id)  # 确保不同环境行为不同

    def step(self, action):
        """模拟交互，返回 (obs, reward, done)"""
        time.sleep(np.random.uniform(0.01, 0.05))  # 模拟不同环境的计算时间
        obs = np.random.rand(3)  # 伪造观测值
        reward = np.random.rand()  # 伪造奖励
        done = np.random.rand() < 0.05  # 伪造终止标志
        return obs, reward, done

    def reset(self):
        """复位环境"""
        return np.random.rand(3)

# 启动多个环境实例
num_envs = 4
envs = [EnvActor.remote(5555 + i) for i in range(num_envs)]

# 采样数据
actions = [np.random.randint(0, 3) for _ in range(num_envs)]
results = ray.get([env.step.remote(action) for env, action in zip(envs, actions)])

print(results)  # 获取多个环境的并行结果
```

这样，你的环境可以并行运行多个实例，加速数据采样。

## 直接使用Ray-RLlib

**核心思路**：Ray RLlib是Ray提供的强化学习库，内部已经实现了PPO算法，并支持自动并行化环境管理。你只需要定义一个自定义的环境适配器，让RLlib调用你的C++环境即可。

**适用场景**：

- 你不想自己写PPO，而是希望用RLlib自带的PPO算法。
- 你愿意适配你的C++环境，让它符合RLlib的环境接口。

**实现方式**：

1. **自定义RLlib环境**，让它通过ZMQ与C++环境交互。
2. **用RLlib训练PPO**，它会自动并行多个环境实例。

```python
from ray.rllib.env import ExternalEnv
import numpy as np

class CustomEnv(ExternalEnv):
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id
        np.random.seed(env_id)  # 确保不同环境行为不同

    def run(self):
        while True:
            """模拟交互，返回 (obs, reward, done)"""
            time.sleep(np.random.uniform(0.01, 0.05))  # 模拟不同环境的计算时间
            obs = np.random.rand(3)  # 伪造观测值
            reward = np.random.rand()  # 伪造奖励
            done = np.random.rand() < 0.05  # 伪造终止标志
            return obs, reward, done

# 用RLlib的PPO进行训练
import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env

def env_creator(env_config):
    return CustomEnv(env_config["zmq_port"])

register_env("CustomEnv-v0", env_creator)

trainer = ppo.PPOTrainer(env="CustomEnv-v0", config={"num_workers": 4})
for _ in range(10):
    result = trainer.train()
    print(result)
```

这样，你的C++环境可以被RLlib管理，并自动进行多线程训练。

**两种方式的区别**

| 方式                          | 适用场景                        | 优势                         | 缺点                      |
| ----------------------------- | ------------------------------- | ---------------------------- | ------------------------- |
| **自己基于Ray实现多环境并行** | 你自己实现PPO，想要加速环境采样 | 灵活性高，不受RLlib约束      | 需要自己实现PPO并管理数据 |
| **直接使用Ray RLlib**         | 你希望用RLlib的PPO算法          | 省时省力，自动管理多线程训练 | 需要适配RLlib的环境接口   |

**建议**：

- 如果你希望完全掌控PPO算法，建议 **自己基于Ray实现多环境并行**（第一种方式）。
- 如果你愿意用RLlib自带的PPO，并希望快速上手，建议 **直接使用Ray RLlib**（第二种方式）。

# Ray知识点介绍

## Ray是怎么知道 worker.step.remote()已经完成的

当你调用：

```python
future = worker.step.remote(action)
```

- `future` 是一个**Ray ObjectRef**（也叫 Future），表示这个远程任务的**唯一标识符**。
- Ray 的**调度器**会把这个任务分配给 **某个 Worker（远程进程）** 执行。

然后，Ray 在后台使用 **消息队列** 机制（基于 **GCS（全局控制存储）和调度器**），不断检查这些任务的状态：

1. **任务刚被提交** → `future` 处于**未完成**状态。
2. **任务执行中** → `future` 仍然**未完成**。
3. **任务执行完毕** → `future` 变为**已完成**，结果存入 **Ray 的对象存储（Object Store）**。

当你调用：

```python
ready_tasks, _ = ray.wait(list(tasks.keys()), num_returns=1)
```

Ray 会：

1. 遍历 `tasks.keys()` 里的所有 Future（正在运行的 `worker.step.remote()`）。
2. 在**后台不断轮询**这些 Future，检查哪些已经完成。
3. 只要**有 num_returns=1 个任务完成**，就立刻返回这个任务的 Future，不会等其他任务。

**ray.wait() vs ray.get()**

| 方法                             | 作用                                        | 是否阻塞主线程               |
| -------------------------------- | ------------------------------------------- | ---------------------------- |
| `ray.wait([...], num_returns=1)` | **等待最少 1 个任务完成**，但不会等所有任务 | **不会阻塞**，主线程继续运行 |
| `ray.get(futures)`               | **获取所有任务的结果**，必须等所有任务完成  | **会阻塞**，直到所有任务完成 |

**ray.wait() 适合异步并行**，不会卡住慢的任务，提高训练效率！

## ray.wait()的原理

前面你提到了：

> 1、任务执行完毕 → future 变为已完成，结果存入 Ray 的对象存储（Object Store）。
> 2、wait在后台不断轮询这些 Future，检查哪些已经完成。

我的理解：

> 从你提到的1和2两点，我推断出，wait会不断查询 Ray 的对象存储（Object Store）中是否有list中的worker标记，如果有，那就说明完成了？

是的，你的理解基本正确！Ray 的 `ray.wait()` 通过**检查 Object Store（对象存储）**，判断哪些任务已经完成，流程如下：

**深入解析 ray.wait() 机制**

1. **任务提交**：

   ```python
   future = worker.step.remote(action)
   ```

   - `future` 是一个 **ObjectRef**，表示这个任务的唯一标识。
   - 任务被派发到某个 **Worker 进程** 处理。

2. **任务执行中**：

   - Worker 运行 `step()` 方法，可能需要 **不同时间** 才能完成。
   - 这时，Object Store 里**还没有**对应的结果，`future` 处于**未完成**状态。

3. **任务执行完毕**：

   - Worker 计算完 `step()` 结果，把 `(obs, reward, done)` 存入 **Object Store**。
   - 此时 `future` 变为 **已完成状态**，Ray 记录它已经完成。

4. **ray.wait() 检测任务完成**：

   ```python
   ready_tasks, _ = ray.wait(list(tasks.keys()), num_returns=1)
   ```

   - `ray.wait()` 在后台**轮询 Object Store**，检查 `tasks.keys()`（所有 Future）对应的结果是否已经存入 Object Store。
   - 只要有 **至少 1 个** Future 的结果**可用**，它就会立即返回该任务的 Future，不等其他任务。

**你的推断和 Ray 内部流程**

你的推断：

> `ray.wait()` 会不断查询 **Ray 的对象存储（Object Store）** 是否有 `list` 中的 `worker` 标记，如果有，那就说明完成了？

**是的，基本正确！**

- `ray.wait()` 查询的其实是 **Future 对应的 ObjectRef 是否有可用结果**。
- 一旦某个 `ObjectRef` **在 Object Store 里找到了结果**，Ray 就知道这个任务完成了。

**ray.wait() vs ray.get()**

| 方法         | 检查方式                  | 是否等所有任务完成？     | 是否并行高效？   |
| ------------ | ------------------------- | ------------------------ | ---------------- |
| `ray.wait()` | **异步轮询 Object Store** | ❌ 只等 **部分** 任务完成 | ✅ 适合异步并行   |
| `ray.get()`  | **直接读取 Object Store** | ✅ **必须等全部任务完成** | ❌ 可能卡住慢任务 |

**Ray 通过 Object Store 让 ray.wait() 变得高效，避免等待慢任务，提升并行效率！**

## Object Store存储数据格式

你提到了：

> 1、Worker 计算完 step() 结果，把 (obs, reward, done) 存入 Object Store。
> 2、ray.wait() 查询的其实是 Future 对应的 ObjectRef 是否有可用结果。

那我理解：

> Object Store存的是类似一个dict字典，即{Future 对应的 ObjectRef:  (obs, reward, done)}，实际是我理解的这样吗？

你的理解**基本正确**，Object Store 确实**类似一个字典**，但它并不是真正的 Python `dict`，而是一个 **分布式的对象存储系统**，可以跨进程、跨机器访问。

**Object Store 的数据结构（类比 Python dict）**

虽然 Ray 的 Object Store 不是严格的 Python `dict`，但它的作用类似：

```python
ObjectStore = {
    ObjectRef1: (obs1, reward1, done1),
    ObjectRef2: (obs2, reward2, done2),
    ObjectRef3: (obs3, reward3, done3),
}
```

- **Key（键）** → `ObjectRef`（类似于 Future）
- **Value（值）** → 任务执行结果（如 `(obs, reward, done)`）

当 `ray.wait([ObjectRef1, ObjectRef2])` 执行时：

- Ray **查询 Object Store**，看 `ObjectRef1` 和 `ObjectRef2` 是否有值。？？？？？？？？？应该是有key啊
- 如果 `ObjectRef1` **有值了**，就返回它，`ObjectRef2` 还没完成就不管。

**Object Store 和 Python dict 的关键区别**

| 特性       | Python `dict`       | Ray Object Store                      |
| ---------- | ------------------- | ------------------------------------- |
| 作用范围   | 仅限于 **当前进程** | **跨进程 / 分布式**（进程间共享数据） |
| 访问方式   | 直接读取            | 通过 `ObjectRef` 查询                 |
| 任务完成前 | 直接存取数据        | 任务未完成时，数据**不可访问**        |
| 任务完成后 | 立即可读            | `ray.get(ObjectRef)` 获取数据         |

**关键点：**

1. **Object Store 允许数据在多个进程/机器之间共享，而不复制**，提高效率。
2. **ray.wait() 轮询这个存储，检查哪些 ObjectRef 已经有数据**。

**深入理解 ray.get()**

当你执行：

```python
result = ray.get(ObjectRef1)
```

- Ray 会查询 **Object Store**，如果 `ObjectRef1` 还**没有结果**，就会**阻塞等待**。
- 一旦 `ObjectRef1` **有数据**，就把 `(obs, reward, done)` **取出来** 返回。

而 `ray.wait()` 是**异步检查**，不会卡住主进程。

**总结**

你的理解方向是对的，Object Store **类似一个 dict**，但它是**分布式存储**，数据的访问方式也不一样。`ray.wait()` **检查 Object Store 里哪些 ObjectRef 已经有结果**，从而实现异步并行处理。

## Object-Store中存储的结构被ray.get()后会被删除吗

我的问题：

> 那在 Object Store中存储的dict结果，一旦被ray.get()后，是不是就会从 Object Store存储中删除？

**不是的！** 在 **Ray 的 Object Store** 中，**对象不会因为 ray.get() 被获取后就自动删除**，它仍然会继续存储在那里，直到满足**一定的删除条件**。

**ray.get() 只是读取，不会删除 Object Store 的数据**

当你调用：

```python
result = ray.get(obj_ref)
```

- **Ray 只是从 Object Store 里读取数据，并不会删除它。**
- **相同的 obj_ref，你可以多次 ray.get()，都能获取相同的结果。**

换句话说，**Object Store 类似于一个缓存，数据默认不会立刻删除**。

**那 Object Store 里的数据什么时候会删除？**

Ray 采用**自动垃圾回收**（Garbage Collection, GC）机制，当**某个对象的 ObjectRef（即 obj_ref）不再被引用时，Ray 可能会清理它**。以下几种情况会导致数据删除：

**没有任何 Python 变量引用 obj_ref**

如果 `obj_ref` 变量在 Python 代码中被删除，或者超出了作用域，Ray **可能会自动释放**它的存储：

```python
obj_ref = worker.step.remote()  # 生成一个 ObjectRef
result = ray.get(obj_ref)  # 获取数据
del obj_ref  # 显式删除引用，Ray 可能回收对象
```

**当 obj_ref 被 del 或者超出作用域时，Ray 可能会清理 Object Store 里的数据。**

**Object Store 内存不足**

- **Ray 采用 LRU（最近最少使用）策略**，如果 Object Store 的**内存快满了**，Ray 会**自动删除最老的对象**。
- **你可以调整 object_store_memory 参数** 来扩大 Object Store 的内存。

**显式删除**

你可以使用 `ray.internal.free()`（或者 `ray.experimental.internal_kv.del()`）来**强制删除**存储对象：

```python
ray.internal.free([obj_ref])  # 立即删除 Object Store 里的数据
```

**实验验证**

你可以尝试以下代码：

```python
import ray
ray.init()

@ray.remote
def foo():
    return [1, 2, 3]

obj_ref = foo.remote()
print(ray.get(obj_ref))  # 读取结果

del obj_ref  # 删除 Python 变量
print(ray.get(obj_ref))  # ❌ 这会报错，因为 obj_ref 已被回收
```

当 `obj_ref` 被删除后，**Object Store 里的数据也可能会被回收**，`ray.get(obj_ref)` 就会报错。

**结论**

1. **ray.get() 只是读取 Object Store，不会删除数据**，你可以多次 `ray.get(obj_ref)` 获取相同的结果。
2. **数据会被 Ray 的 GC 自动清理**，当 `obj_ref` 变量被删除或超出作用域时，Object Store 可能回收它。
3. **Object Store 采用 LRU 策略**，当内存不足时，Ray 可能自动删除最旧的数据。
4. **可以手动删除对象**（`ray.internal.free([obj_ref])`），强制清理存储。

所以，如果你想确保数据一直在 Object Store 里，需要**保存 obj_ref 并确保它不会被删除**。



我的问题：

> obj_ref = worker.step.remote()  # 生成一个 ObjectRef
>
> result = ray.get(obj_ref)  # 获取数据
>
> del obj_ref  # 显式删除引用，Ray 可能回收对象
>
> 我只是在python中把obj_ref删除了，ray中是怎么知道的呢

当你在 Python 中使用 `del obj_ref` 显式删除一个 `ObjectRef` 时，实际上只是删除了 Python 层面上的对这个 `ObjectRef` 的引用，并没有直接删除 Ray 内部的对象。在 Ray 中，Ray 的对象存储（Object Store）是基于引用计数的机制进行垃圾回收的，具体的行为如下：

**1. 引用计数机制**

Ray 会跟踪每个对象的引用计数（Reference Counting）。每当有一个新的引用指向这个对象，引用计数就会增加；当引用被删除或超出作用域时，引用计数会减少。当一个对象的引用计数降为 0 时，Ray 会认为这个对象不再被使用，可以被回收。

因此，即使你在 Python 中使用 `del obj_ref` 删除了对 `ObjectRef` 的引用，Ray 内部仍然会根据对象的引用计数判断这个对象是否可以被回收。如果在删除引用之后，还有其他的地方（比如其他的 `ray.get()` 或其他的 `ObjectRef` 引用）仍然引用着这个对象，Ray 就不会回收它。

**2. del obj_ref 对 Ray 的影响**

- `del obj_ref` 只会移除 Python 代码中的变量 `obj_ref`，它不会直接影响 Ray 的对象存储中的内容。
- Ray 内部的对象（比如存储 `step()` 返回的结果）会继续存在，直到没有其他地方持有该对象的引用，Ray 才会通过其垃圾回收机制将对象从对象存储中删除。

**3. Ray 对象存储和回收**

当你通过 `ray.get(obj_ref)` 获取对象时，Ray 会在后台检查对象是否已经被计算完成，并返回计算结果。Ray 会将这个对象的结果存储在对象存储中，直到没有任何引用指向它时，Ray 才会清除这个对象并释放内存。

**简而言之**

- `del obj_ref` 删除的是你在 Python 中的引用，Ray 并不会立即回收这个对象。
- Ray 会依赖其内部的引用计数机制来决定何时回收对象。如果没有其他引用指向该对象（例如，其他的 `ObjectRef`），Ray 才会回收它。

**总结**

Ray 会跟踪对象的引用计数，并在对象不再被引用时清除它。即使你在 Python 中删除了 `ObjectRef` 的引用，只要 Ray 内部仍然有其他地方持有对该对象的引用，它就不会被回收。



我的问题：

> 如果不删除的话，那再次调用worker.step.remote，Object Store原来该work对应的dict的value是否会被更新？

**不会自动更新！**

每次你调用 `worker.step.remote()`，**都会返回一个新的 ObjectRef，不会覆盖之前的值**。Object Store 里的数据是**不可变的**（Immutable），不会因为你再次调用 `worker.step.remote()` 而自动更新旧的值。

```python
import ray
ray.init()

@ray.remote
class Worker:
    def step(self):
        return "New Data"

worker = Worker.remote()

# 第一次调用 step
obj_ref1 = worker.step.remote()
print(ray.get(obj_ref1))  # 结果："New Data"

# 再次调用 step
obj_ref2 = worker.step.remote()
print(ray.get(obj_ref2))  # 结果仍然是："New Data"

# 检查两个 ObjectRef 是否相同
print(obj_ref1 == obj_ref2)  # 结果：False（不同的 ObjectRef）
```

obj_ref1 == obj_ref2为什么能比较？？？？？？？？？？？？？？？？？？？？

**观察到：**

- **每次 step.remote() 都会返回新的 ObjectRef**（即 `obj_ref1` 和 `obj_ref2` 是不同的）。
- **Object Store 里不会覆盖原来的值，而是创建新的存储对象**。

## 什么是obj_ref

`obj_ref` 是 **Object Reference** 的简称，也可以理解为一个**指向某个存储在 Ray Object Store 中数据的引用**。

在 Ray 中，计算任务（比如调用 `worker.step.remote()`）会生成一个 **ObjectRef**，这个 **ObjectRef** 就是你用来访问计算结果的“句柄”。通过 `ray.get(obj_ref)`，你可以获取任务执行的结果。

**如何理解 obj_ref？**

你可以把它理解为 **指向 Ray 中数据的指针**，它并不存储数据本身，而是指向存储数据的位置。Ray 使用这些引用来管理任务结果和对象存储。

**工作流程：**

1. 当你调用一个 Ray 远程方法（比如 `worker.step.remote()`）时，Ray 会执行该方法并返回一个 `obj_ref`，表示这个任务的结果。

   ```python
   obj_ref = worker.step.remote()  # 这返回的是一个 ObjectRef
   ```

2. 该 `obj_ref` 只是一个指向结果数据的引用。**任务执行的结果会存储在 Ray 的 Object Store 中，而 obj_ref 则是访问这些结果的唯一标识符**。

3. 你可以通过 `ray.get(obj_ref)` 获取数据，这时 Ray 会去 Object Store 中找到对应的结果并返回给你。

   ```python
   result = ray.get(obj_ref)  # 获取对象存储中的数据
   ```

**简单示例：**

```python
import ray

ray.init()

@ray.remote
class Worker:
    def step(self):
        return "Hello from worker!"

worker = Worker.remote()

# 调用远程方法，得到一个 ObjectRef
obj_ref = worker.step.remote()

# 使用 ObjectRef 获取执行结果
result = ray.get(obj_ref)
print(result)  # 输出: "Hello from worker!"
```

**总结：**

- `obj_ref` 是指向 **Ray Object Store** 中的任务结果的引用。
- 它不包含任务的实际数据，而是 Ray 用来追踪任务进度和获取结果的标识符。
- 通过 `ray.get(obj_ref)`，你可以获取到数据，`obj_ref` 就像是一个数据的“地址”。

# 基于Ray并行采样批量推理的强化学习代码解析

```python
import ray
import time
import numpy as np
from collections import deque

ray.init()

class Agent:
    def __init__(self):
        pass  # 初始化策略网络等

    def choose_action(self, states):
        """批量选择动作"""
        states = np.array(states)  # 转换为 NumPy 数组，方便处理
        return np.random.randint(4, size=len(states))  # 这里用随机动作，实际中改成神经网络批量推理

    def update(self, buffer):
        """训练智能体"""
        print(f"训练 PPO，样本数：{len(buffer)}")
        buffer.clear()  # 清空经验池，等待下次收集

@ray.remote
class EnvActor:
    def __init__(self, env_id):
        self.env_id = env_id
        np.random.seed(env_id)  # 确保不同环境行为不同

    def step(self, action):
        """模拟 C++ 交互，返回 (obs, reward, done)"""
        time.sleep(np.random.uniform(0.01, 0.05))  # 模拟不同环境的计算时间
        obs = np.random.rand(3)  # 伪造观测值
        reward = np.random.rand()  # 伪造奖励
        done = np.random.rand() < 0.05  # 伪造终止标志
        return obs, reward, done

    def reset(self):
        """复位环境"""
        return np.random.rand(3)

# 初始化
num_envs = 2
workers = [EnvActor.remote(i) for i in range(num_envs)]
agent = Agent()

# 复位所有环境
obs_dict = {worker: ray.get(worker.reset.remote()) for worker in workers}

# **初始化 step 任务**
tasks = {worker.step.remote(0): worker for worker in workers}  # 先随便给个动作

buffer = deque(maxlen=3)

while len(buffer) < 3:
    # **等待多个环境完成 step**
    num_ready = min(2, len(tasks))
    ready_tasks, _ = ray.wait(list(tasks.keys()), num_returns=num_ready)

    # **批量获取多个任务的 (obs, reward, done) 结果**
    results = ray.get(ready_tasks)

    # **存储新的环境状态**
    new_obs_list = []
    worker_list = []

    for task, (obs, reward, done) in zip(ready_tasks, results):
        worker = tasks.pop(task)  # 找到对应的 worker
        buffer.append((obs_dict[worker], obs, reward, done))  # 存经验 (s, s', r, done)

        if done:
            new_obs = ray.get(worker.reset.remote())  # 复位环境
        else:
            new_obs = obs  # 否则使用 step 返回的 obs 作为新状态

        obs_dict[worker] = new_obs  # 更新状态
        new_obs_list.append(new_obs)
        worker_list.append(worker)

    # **批量推理动作**
    actions = agent.choose_action(new_obs_list)  # 现在是 batch 处理

    # **派发 step 任务**
    for worker, action in zip(worker_list, actions):
        tasks[worker.step.remote(action)] = worker  # 继续派发任务

    if len(buffer) % 10 == 0 or len(buffer) == 3:
        print(f"已收集 {len(buffer)}/3 样本", end="\r")

print("\n数据收集完毕！开始训练 PPO...")
agent.update(buffer)  # 训练智能体
```

