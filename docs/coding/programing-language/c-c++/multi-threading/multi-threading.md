# 多线程编程

* [返回上层目录](../c-c++.md)
* [基本知识](#基本知识)
  * [互斥锁（std::mutex）](#互斥锁（std::mutex）)
  * [lock_guard和unique_lock的区别](#lock_guard和unique_lock的区别)
  * [条件变量（std::condition_variable）](#条件变量（std::condition_variable）)
* [生产者-消费者模型](#)
  * [单生产者-单消费者模型](#单生产者-单消费者模型)
  * [单生产者多消费者的多线程程序](#单生产者多消费者的多线程程序)
* [知识点](#)
  * [等待互斥锁队列和等待条件队列的区别](#等待互斥锁队列和等待条件队列的区别)
  * [条件变量的等待队列被唤醒线程是否从头运行](#条件变量的等待队列被唤醒线程是否从头运行)
  * [互斥锁等待队列被释放线程是否从头运行](#互斥锁等待队列被释放线程是否从头运行)



# 基本知识

## 互斥锁（std::mutex）

互斥锁是实现线程间资源独占访问的基础手段。一旦一个线程获得了锁，其他试图获取同一锁的线程将会被阻塞，直到锁被释放。

```c++
std::mutex mtx;
// 加锁
mtx.lock();
// 执行临界区代码
// ...
// 解锁
mtx.unlock();
```

**易错点与避免策略**

1. **忘记解锁**：使用`std::lock_guard`或`std::unique_lock`自动管理锁的生命周期，确保即使发生异常也能解锁。
2. **死锁**：避免在持有锁的情况下调用可能阻塞的函数，或按相同的顺序获取多个锁。

## lock_guard和unique_lock的区别

`std::lock_guard<std::mutex>` 和 `std::unique_lock<std::mutex>` 都是 C++ 中用于互斥锁（`mutex`）管理的工具，它们都用来确保线程安全，防止多个线程同时访问共享资源。虽然它们的功能类似，但有一些关键的区别，下面我会逐步解释：

**（1）`std::lock_guard<std::mutex>`**：

`std::lock_guard` 是一种简单的、轻量级的锁管理工具，通常用于**不需要手动解锁的场合**。它会在作用域内自动锁住给定的 `mutex`，并在 `lock_guard` 超出作用域时自动解锁。

**特点**：

- **自动锁定**：当 `std::lock_guard` 对象被创建时，它会自动尝试获取互斥锁。
- **自动释放**：当 `std::lock_guard` 对象超出作用域时，它会自动释放互斥锁，不需要显式地调用解锁。
- **不支持手动解锁**：`lock_guard` 没有显式的 `unlock()` 方法，解锁只能在作用域结束时自动发生。

**使用场景**：

- 适用于临界区非常简单、没有复杂的锁控制需求（比如加锁后直接在代码中执行某些操作后立刻退出）。
- 对于普通的资源保护，`lock_guard` 是一个简单且有效的选择。

```c++
#include <iostream>
#include <mutex>

std::mutex mtx;

void printMessage() {
    std::lock_guard<std::mutex> lck(mtx);  // 自动锁住 mtx
    std::cout << "Hello from thread!" << std::endl;
}  // lck 离开作用域时会自动解锁 mtx
```

**（2）`std::unique_lock<std::mutex>`**：

`std::unique_lock` 是比 `std::lock_guard` 更加灵活的锁管理工具，提供了更多的功能。它在很多场合下都可以替代 `lock_guard`，但也允许更多的手动控制。

**特点**：

- **手动解锁**：`unique_lock` 提供了显式的 `unlock()` 方法，你可以在锁住资源后，在需要的地方手动释放锁。
- **延迟锁定**：`unique_lock` 支持延迟锁定，即你可以先创建 `unique_lock` 对象而不立即锁定互斥量，直到你明确调用 `lock()` 时才会锁定。
- **可重复锁定**：`unique_lock` 支持在同一个线程内多次加锁和解锁（即便是不同的锁操作），适用于一些需要复杂锁控制的情况。
- **支持条件变量**：`unique_lock` 能够与条件变量（`condition_variable`）配合使用，而 `lock_guard` 不能。因为 `unique_lock` 支持锁的释放和重新加锁的操作（这在条件变量等待时非常有用）。

**使用场景**：

- 适用于需要更细粒度控制锁的情况，比如在临界区内部控制锁的获取和释放时。
- 如果你需要与**条件变量**一起使用，`unique_lock` 是必须的，因为条件变量通常需要先释放锁才能让其他线程访问资源。

```c++
#include <iostream>
#include <mutex>

std::mutex mtx;

void printMessage() {
    std::unique_lock<std::mutex> lck(mtx);  // 锁住 mtx
    std::cout << "Hello from thread!" << std::endl;
    // 可以在这里手动解锁
    lck.unlock();  // 手动解锁 mtx
    // 你也可以在此后重新加锁
    lck.lock();  // 重新加锁 mtx
}  // lck 离开作用域时会自动解锁 mtx
```

**（3）何时使用 `std::lock_guard` 和 `std::unique_lock`**

- **使用 std::lock_guard**：如果你只需要一个简单的锁机制，并且没有复杂的需求，`std::lock_guard` 是更简洁、高效的选择。它适合加锁和解锁的时机非常明确，不需要手动控制锁的释放。
- **使用 std::unique_lock**：如果你需要在某些情况下手动解锁、延迟加锁、或者与条件变量一起使用，`std::unique_lock` 更加灵活。它适用于复杂的锁控制，或者在需要对锁的获取和释放有精细控制的场合。

总结：

- `std::lock_guard` 是轻量级的自动锁管理工具，适合简单的锁控制。
- `std::unique_lock` 是更为灵活和强大的工具，适用于需要更多锁控制的场景，比如手动解锁、延迟加锁、与条件变量的配合使用等。



## 条件变量（std::condition_variable）

条件变量用于线程间同步，允许一个线程等待（挂起）直到另一个线程通知某个条件为真。

```c++
std::condition_variable cv;
std::mutex mtx;

void waitingFunction() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{return conditionToWaitFor;}); // 条件满足前挂起
    // 条件满足后执行的代码
}

void notifyingFunction() {
    // 修改状态使得conditionToWaitFor为真
    std::lock_guard<std::mutex> lock(mtx);
    cv.notify_one(); // 唤醒一个等待的线程
}
```

**常见问题与避免策略**

1. **无条件唤醒**：不要在没有改变条件的情况下调用`notify_*`函数，这可能导致不必要的线程唤醒和重新检查条件。
2. **虚假唤醒**：即使没有调用`notify_*`，等待的线程也可能被唤醒。因此，总是使用条件来检查是否真正满足继续执行的条件。
3. **死锁**：确保在调用`wait`之前已经获得了锁，并且在`wait`之后立即检查条件，避免在持有锁的情况下执行耗时操作。

# 生产者-消费者模型

## 单生产者-单消费者模型

```c++
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

std::queue<int> producedItems;
std::mutex mtx;
std::condition_variable condVar;

bool doneProducing = false;

void producer(int n) {
    for (int i = 0; i < n; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 模拟生产时间
        // 检测能获取mutex的独占权限吗？
        // 能到话，那就锁定独占mutex
        std::lock_guard<std::mutex> lock(mtx);
        producedItems.push(i);
        condVar.notify_one(); // 通知消费者
        if (i == n - 1) doneProducing = true;
        // 释放mutex权限
    }
}

void consumer() {
    while (true) {
        // 检测能获取mutex的独占权限吗？
        // 能到话，那就锁定独占mutex
        std::unique_lock<std::mutex> lock(mtx);
        condVar.wait(lock, []{return !producedItems.empty() || doneProducing;});
        if (!producedItems.empty()) {
            int item = producedItems.front();
            producedItems.pop();
            std::cout << "Consumed: " << item << std::endl;
        } else if (doneProducing) {
            break;
        }
    }
}

int main() {
    std::thread producerThread(producer, 10);
    std::thread consumerThread(consumer);

    producerThread.join();
    consumerThread.join();

    return 0;
}
```



## 单生产者多消费者的多线程程序

以下是一个单生产者多消费者的多线程程序。在这个程序中，生产者线程会不断地生产数据，多个消费者线程会从队列中取数据并进行消费。我们将使用 `std::mutex` 来保护共享资源（队列），并且使用 `std::condition_variable` 来协调生产者和消费者的工作。

```c++
#include <iostream>
#include <thread>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>

// 定义队列和锁
std::queue<int> buffer;
std::mutex mtx;
std::condition_variable cv;

// 队列的最大容量
const int max_buffer_size = 10;

// 生产者线程函数
void producer() {
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 模拟生产过程
        
        std::unique_lock<std::mutex> lock(mtx);
        // 如果队列满了，生产者就等待
        cv.wait(lock, []{ return buffer.size() < max_buffer_size; });

        // 生产一个数据并将其放入队列
        buffer.push(i);
        std::cout << "Produced: " << i << std::endl;
        
        // 通知消费者可以消费了
        cv.notify_all();
    }
}

// 消费者线程函数
void consumer(int id) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        // 如果队列为空，消费者就等待
        cv.wait(lock, []{ return !buffer.empty(); });

        // 消费一个数据并从队列中取出
        int data = buffer.front();
        buffer.pop();
        std::cout << "Consumer " << id << " consumed: " << data << std::endl;

        // 通知生产者可以继续生产了
        cv.notify_all();

        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // 模拟消费过程
    }
}

int main() {
    std::vector<std::thread> consumers;

    // 启动一个生产者线程
    std::thread producer_thread(producer);

    // 启动多个消费者线程
    for (int i = 0; i < 3; ++i) {
        consumers.push_back(std::thread(consumer, i));
    }

    // 等待生产者线程完成
    producer_thread.join();

    // 注意：这里为了简单起见，我们没有设置消费者的退出条件
    // 通常，你可以设计一种方式让消费者线程在生产者结束后退出，例如使用标志变量
    for (auto& consumer_thread : consumers) {
        consumer_thread.join();
    }

    return 0;
}
```

**代码解释**：

1. **生产者线程**：
   - 生产者线程在循环中生产数据（我们假设生产的数据是一个整数，生产者在每次生产时都会将一个整数放入队列中）。
   - 在生产之前，生产者会检查队列是否已满。如果队列已满，生产者会等待，直到消费者消费掉一些数据，队列有空位为止。
   - 生产者生产数据后，会通知消费者线程，告诉它们可以消费数据了。
2. **消费者线程**：
   - 每个消费者线程会从共享队列中取数据进行消费。
   - 如果队列为空，消费者会等待，直到生产者生产新的数据。
   - 当消费者消费完数据后，通知生产者可以继续生产。
3. **同步机制**：
   - 使用 `std::mutex` 来保护对共享队列的访问，防止多个线程同时访问队列而导致竞争条件。
   - 使用 `std::condition_variable` 来协调生产者和消费者的工作。`cv.wait()` 会使线程在特定条件下挂起，直到满足条件后才会继续执行。
   - `cv.notify_all()` 会通知所有等待条件变量的线程，让它们重新检查条件并继续执行。
4. **程序结构**：
   - 我们使用一个生产者线程和三个消费者线程来模拟生产和消费过程。
   - 生产者每次生产一个数据，消费者每次消费一个数据。

**注意事项**：

- 这只是一个简单的示例，现实中你可能需要在消费者退出时加入一些机制，例如在生产者结束后通知消费者停止工作。
- `std::condition_variable` 中的 `cv.wait()` 会释放 `std::mutex` 锁，并且将线程放入条件变量的等待队列，直到接收到 `cv.notify_all()` 或 `cv.notify_one()` 才会被唤醒，重新获取锁并继续执行。

**编译运行**：

如果你使用的是 `g++` 编译器，你可以使用以下命令编译并运行代码：

```shell
g++ -std=c++11 -o producer_consumer producer_consumer.cpp -pthread
./producer_consumer
```

在 Windows 上，可以使用 Visual Studio 或其他支持 C++11 及线程库的编译器来运行。

# 知识点

## 等待互斥锁队列和等待条件队列的区别

等待互斥锁会把线程放进一个等待队列1中，等待条件也会把线程放进一个等待队里2中，请问1和2是同一个等待队列吗？

> 你的问题很有道理，实际上“等待互斥锁”和“等待条件”是有不同的等待队列的。我们来详细分析一下。
>
> ### 1. **等待互斥锁**和**等待条件**的等待队列不是同一个队列
>
> 在多线程编程中，互斥锁和条件变量是不同的机制，它们有自己的队列来管理等待的线程。
>
> #### 1.1 **互斥锁的等待队列**
>
> - 当一个线程尝试获取一个已经被占用的互斥锁时，它会被放入**互斥锁的等待队列**。
> - 线程会在这个队列中阻塞，直到能够成功获取到锁。获取锁的线程会从等待队列中被移除，并且重新进入运行状态。
> - **互斥锁的等待队列**是由操作系统或者线程库（如 C++ 标准库实现）管理的，它专门管理那些等待锁的线程。
>
> #### 1.2 **条件变量的等待队列**
>
> - 条件变量则是另一种机制，用于线程之间的通信和同步。**条件变量的等待队列**是管理那些**等待特定条件发生**的线程的。
> - 当一个线程调用 `cv.wait()` 时，它会在等待队列中阻塞，直到其他线程通过 `notify_one()` 或 `notify_all()` 通知它该条件已经满足。
> - 在调用 `cv.wait()` 的时候，线程会**释放互斥锁**并进入条件变量的等待队列。当条件满足时，线程会被唤醒，重新尝试获取锁。
>
> #### 1.3 **两者的区别**
>
> - **互斥锁等待队列**和**条件变量等待队列**是完全不同的队列。
> - **互斥锁等待队列**仅涉及获取锁的操作，它在锁的管理和调度中起作用，通常是由操作系统或线程库管理。
> - **条件变量等待队列**则涉及的是线程等待某种条件满足，通常是由条件变量的实现（如 `std::condition_variable`）管理。
>
> ### 2. **它们如何协同工作**
>
> 虽然它们是不同的队列，但**它们是协同工作的**：
>
> 1. **线程在获取互斥锁时**：一个线程在尝试获取互斥锁时，如果锁被其他线程占用，它会被放入互斥锁的等待队列，等待其他线程释放锁。
> 2. **线程在等待条件时**：如果线程在条件不满足的情况下调用 `cv.wait(lock)`，它会释放已经持有的互斥锁，进入条件变量的等待队列，并阻塞。直到其他线程调用 `cv.notify_*()` 来通知条件满足，线程才会被唤醒。
> 3. **被唤醒后重新获取互斥锁**：当线程被唤醒后，它会重新尝试获取互斥锁，如果获取成功，线程就可以继续执行。如果没有获取到锁，它会继续被放回互斥锁的等待队列，直到它能够获得锁为止。
>
> ### 3. **总结**
>
> - **互斥锁的等待队列**是专门用于管理那些等待锁的线程。
> - **条件变量的等待队列**是专门用于管理那些等待条件满足的线程。
> - 它们是两个**不同的队列**，分别用于不同的目的，但它们共同协作，确保线程同步和顺序执行。

## 条件变量的等待队列被唤醒线程是否从头运行

你提到了：在调用 cv.wait() 的时候，线程会释放互斥锁并进入条件变量的等待队列。当条件满足时，线程会被唤醒，重新尝试获取锁。

我的疑问：当条件变量的等待队列里的线程被notify_all唤醒时，该线程会重头从最开始运行吗？

> 你提出的问题非常重要，涉及到线程在 `cv.wait()` 和 `notify_all()` 等操作中的行为和状态。让我们详细解释一下当一个线程被 `cv.notify_all()` 唤醒时，线程的行为以及它是否会“从头开始运行”。
>
> ### 1. **线程被 cv.wait() 阻塞时的行为**
>
> - 当线程调用 `cv.wait(lock)` 时，它会释放已经持有的互斥锁，并进入 **条件变量的等待队列**，此时它处于阻塞状态。具体来说，线程会被挂起，直到其他线程通知它，条件满足。
> - 这里的 `lock` 是一个 `std::unique_lock<std::mutex>` 类型的对象，它会确保线程在进入等待队列前释放互斥锁。
>
> ### 2. **被 notify_all() 或 notify_one() 唤醒时的行为**
>
> 当其他线程调用 `cv.notify_all()` 或 `cv.notify_one()` 时，等待队列中的线程将被唤醒。此时，唤醒的线程并不会直接从 **最开始** 运行，而是会有以下行为：
>
> 1. **重新获取锁**：
>    - 在被唤醒之后，线程必须重新获取原先释放的互斥锁。由于 `cv.wait()` 已经释放了锁并进入等待状态，所以唤醒后线程会重新尝试获取这个锁。
>    - 如果有其他线程正在持有互斥锁，线程会被挂起在互斥锁的队列中，直到锁可用为止。
> 2. **线程不从头开始运行**：
>    - 线程不会从 **最开始** 重新运行。相反，线程会从它上次被暂停的位置继续执行。也就是说，它会 **恢复** 到调用 `cv.wait()` 后的下一行代码继续执行。
>    - 唤醒的线程不会重新开始执行它的所有代码，它只是继续执行 `cv.wait()` 后面的部分。
> 3. **可能重新检查条件**：
>    - 唤醒的线程需要重新检查等待条件是否满足。如果条件仍然不满足，它会再次调用 `cv.wait()`，并继续阻塞等待。
>
> ### 3. **为什么要重新检查条件？**
>
> C++ 的条件变量机制遵循一种叫做 **“虚假唤醒”**（spurious wakeup）的设计原则。即使条件变量被通知了，线程也不能直接认为条件一定满足。为了防止线程在条件未满足时继续执行，它们会在每次唤醒后再次检查条件。
>
> 例如：
>
> ```c++
> cv.wait(lock, []{ return condition; });
> ```
>
> - 如果 `condition` 仍然不满足，线程会继续等待。
> - 如果 `condition` 满足，线程会继续执行。
>
> ### 4. **总结：**
>
> - 线程在 `cv.wait()` 时会释放互斥锁，进入条件变量的等待队列。
> - 当调用 `notify_all()` 或 `notify_one()` 时，等待队列中的线程被唤醒，它们不会从头开始，而是从调用 `cv.wait()` 之后的位置继续执行。
> - 线程在被唤醒后会重新获取锁，并检查条件是否满足。如果条件未满足，线程会再次进入等待队列。

## 互斥锁等待队列被释放线程是否从头运行

那请问如果`mutex`正在被占用，`std::unique_lock<std::mutex> lock(mtx);`把当前线程挂起后，然后等到了`mutex`被释放，挂起的当前线程会从线程的开头重新运行，还是从`std::unique_lock<std::mutex> lock(mtx);`后开始运行？

> 当你使用 `std::unique_lock<std::mutex> lock(mtx);` 来尝试获取互斥锁（mutex）时，发生了线程挂起和重新唤醒的情况。让我们详细讲解一下在互斥锁被占用、当前线程挂起、再恢复时的执行流程。
>
> ### 关键概念：`std::unique_lock<std::mutex> lock(mtx);`
>
> 1. **线程尝试获取锁**：
>    - 当 `std::unique_lock<std::mutex> lock(mtx)` 被执行时，线程会尝试获取 `mtx` 锁。如果 `mtx` 目前被其他线程持有，当前线程就会被挂起，并且进入等待状态，直到它能够成功获取锁。
> 2. **线程被挂起**：
>    - 线程在获取不到锁时会被挂起，系统会让出 CPU 资源，允许其他线程继续运行。这时线程并没有消耗 CPU 资源。
> 3. **锁被释放后，挂起的线程恢复执行**：
>    - 一旦其他线程释放了锁（也就是锁 `mtx` 被释放），挂起的线程就有机会重新获得锁，并继续执行。线程不会从头开始执行，而是从 `std::unique_lock<std::mutex> lock(mtx);` 之后的地方继续执行。
>
> ### 具体细节：
>
> - **线程恢复执行的位置**：当线程恢复时，程序会从 `std::unique_lock<std::mutex> lock(mtx);` 后面开始执行，而不是从头开始。也就是说，线程会继续执行锁定成功后的部分代码。
> - **从锁被成功获取后开始执行**：
>   - 在线程重新获取到锁后，`std::unique_lock<std::mutex> lock(mtx)` 语句会成功执行，然后代码会继续执行 `lock` 后面的部分。这意味着线程并没有“重头开始”，而是从此处继续。
>
> ### 为什么线程不会从头开始？
>
> 线程是基于 **上下文切换**（context switching）机制来运行的。当线程被挂起时，它的状态（包括程序计数器、寄存器等）被保存在内存中。当线程恢复时，它会根据保存的状态从挂起之前的指令位置继续执行。