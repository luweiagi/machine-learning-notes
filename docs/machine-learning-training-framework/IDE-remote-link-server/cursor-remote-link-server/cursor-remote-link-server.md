# cursor远程连接服务器

* [返回上层目录](../IDE-remote-link-server.md)

# 使用大模型开发代码

具体看这里：

[【Cursor】Cursor的基本使用方式](https://zhuanlan.zhihu.com/p/32081323818)

# 问题

## cursor无法进行代码补全

Python Language Server 服务器的设置，很重要

尝试**Pyright**，请安装该插件。它通常提供最佳的代码补全体验。如果遇到性能问题，可以回退到更轻量的**Jedi**。选择一个选项后需要重启Cursor才能完全生效

为啥不用VSCode中的**Pylance**呢？因为这是微软开发的，已经不让cursor用了，因为和VSCode构成了竞争。

# 参考资料

* [cursor无法进行代码补全时候的修正(折磨我2小时）](https://blog.csdn.net/ranzhiyimu/article/details/146153804)

cursor无法进行代码补全部分参考这里。

