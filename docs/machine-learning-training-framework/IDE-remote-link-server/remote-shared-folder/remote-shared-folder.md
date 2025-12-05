# 远程共享文件夹

* [返回上层目录](../IDE-remote-link-server.md)



# 共享文件夹

我自己的笔记本和另一台windows台式机通过网线连接了，然后怎么用网上邻居或者共享的方式访问我笔记本上的文件夹？

要通过网线共享文件夹，在两台Windows电脑之间使用网络邻居（SMB共享）访问，请按照以下步骤操作：

## 步骤 1：设置IP地址（确保两台电脑在同一局域网）

1. **在你的笔记本（主机）上**：

   - 打开 **控制面板 > 网络和共享中心 > 更改适配器设置**。

   - 右键点击 **以太网（有线网卡）** → **属性** → 双击 **IPv4**。

   - 设置静态IP，例如：

     ```
     IP地址：192.168.1.1
     子网掩码：255.255.255.0
     默认网关：留空
     ```

   - 点击 **确定** 保存。

2. **在台式机上**：

   - 同样设置静态IP，确保在同一网段，例如：

     ```
     IP地址：192.168.1.2
     子网掩码：255.255.255.0
     默认网关：留空
     ```

## 步骤 2：启用网络发现和文件共享

1. **在两台电脑上均操作**：
   - 打开 **控制面板 > 网络和共享中心 > 高级共享设置**。
   - 确保以下选项启用：
     - **启用网络发现**
     - **启用文件和打印机共享**
     - **关闭密码保护共享**（若需免密访问，否则需账号密码）。

## 步骤 3：共享笔记本上的文件夹

1. **在笔记本上**：
   - 右键点击要共享的文件夹 → **属性** → **共享** 选项卡。
   - 点击 **共享...**，添加用户 `Everyone`，设置权限为 **读取/写入**。
   - 记录共享路径（如 `\\192.168.1.1\共享文件夹名`）。

## 步骤 4：从台式机访问共享文件夹

1. **在台式机上**：

   - 打开 **文件资源管理器**，地址栏输入笔记本的共享路径：

     ```
     \\192.168.1.1\共享文件夹名
     ```

   - 按回车，即可访问（若提示密码，输入笔记本的用户名和密码）。

## 常见问题解决

1. **无法访问共享**：

   - 检查防火墙：在 **控制面板 > Windows Defender 防火墙** 中，确保 **文件和打印机共享** 规则已启用。

   - 运行命令测试连通性：

     ```
     ping 192.168.1.1  # 在台式机执行，检查是否能通笔记本IP
     ```

2. **找不到网络路径**：

   - 在两台电脑上运行以下命令重启网络服务：

     ```
     net stop workstation
     net start workstation
     ```

## 替代方案：使用飞鸽传书等局域网工具

如果共享设置复杂，可用 **飞鸽传书（IPMessenger）** 或 **HFS（HTTP文件服务器）** 通过浏览器传输文件，无需共享文件夹。

完成上述步骤后，台式机即可通过网线直接访问你笔记本的共享文件夹。如有具体报错，可进一步排查！

# 在Windows7上配置SSH服务

在 Windows 7 台式机上配置 SSH 服务，以便通过 VS Code 远程连接，需手动安装 OpenSSH 并进行配置。以下是详细步骤：

## 步骤 1：在台式机（Win7）上安装 OpenSSH

Windows 7 默认不包含 OpenSSH 服务，需手动安装OpenSSH

1. 下载 OpenSSH for Windows 的旧版本（如 [openssh-win32.zip](https://github.com/PowerShell/Win32-OpenSSH/releases)）。

2. 解压到 `C:\Program Files\OpenSSH`。

3. 以 **管理员身份** 打开 **CMD**，执行以下命令安装服务：

   ```
   cd "C:\Program Files\OpenSSH"
   powershell.exe -ExecutionPolicy Bypass -File install-sshd.ps1
   ```

注意：如果是win11系统，参考这个安装OpenSSH服务：

[windows11 openssh服务开启；第三方ping不通局域网windows电脑；ssh连接内部ubuntu系统_win11 打开openssh](https://blog.csdn.net/m0_60635176/article/details/137417472)

## 步骤 2：配置 SSH 服务

1. **修改配置文件**：

   - 打开 `C:\Program Files\OpenSSH\sshd_config` 文件（用记事本编辑）。

   - 确保以下配置生效：

     ```
     Port 22
     ListenAddress 0.0.0.0
     PermitRootLogin yes
     PasswordAuthentication yes
     ```

   - 保存文件。

2. **启动 SSH 服务**：

   ```
   net start sshd
   ```

3. **设置防火墙规则**（允许端口 22）：

   ```
   netsh advfirewall firewall add rule name="OpenSSH" dir=in action=allow protocol=TCP localport=22
   ```

## 步骤 3：从笔记本连接台式机

1. **在 VS Code 中**：

   - 安装 **Remote - SSH** 扩展。

   - 按 `F1`，输入 `Remote-SSH: Connect to Host`。

   - 输入连接信息：

     ```
     ssh <用户名>@<台式机IP> -p 22
     ```

     （如 `ssh admin@192.168.1.2`）

2. **首次连接需确认指纹**，输入密码后即可访问。

## 问题：

### 好像只有Administrator，然后密码的话，我锁定后只要回车就能打开，似乎没有密码啊。。

如果您的 Windows 7 台式机的 `Administrator` 账户 **没有密码**（直接按回车即可登录），而 SSH 连接 **必须使用密码或密钥**，以下是解决方案：

**1. 为 Administrator 设置密码（必须步骤）**

在台式机上操作：

1. **以管理员身份运行 CMD**：

   - 按 `Win + R`，输入 `cmd`，然后按 `Ctrl + Shift + Enter`（以管理员权限运行）。

2. **设置密码**：

   ```
   net user Administrator 你的密码
   ```

   - 例如：

     ```
     net user Administrator MySSHpassword123!
     ```

   - 密码需符合复杂性要求（至少包含大小写字母、数字、符号中的两种）。

3. **验证密码是否生效**：

   - 锁定计算机（`Win + L`），尝试用 `Administrator` 和新密码登录。

**2. 修改 SSH 配置允许空密码（不推荐）**

如果坚持不设密码（仅限绝对信任的内网环境）：

1. 编辑 `sshd_config`：

   ```
   notepad "C:\Program Files\OpenSSH\sshd_config"
   ```

2. 添加或修改以下行：

   ```
   PermitEmptyPasswords yes
   PasswordAuthentication yes
   ```

3. 重启 SSH 服务：

   ```
   net stop sshd && net start sshd
   ```

   **⚠️ 警告**：此配置极不安全，可能被恶意攻击。

**3. 测试 SSH 连接**

在笔记本上执行：

```
ssh Administrator@台式机IP
```

- 如果设置了密码，输入密码后登录。
- 如果允许空密码，直接按回车。



# FTP/Samba共享 + VSCode本地编辑

**终极方案：SFTP/Samba共享 + VSCode本地编辑**

✅ **优势**

- **零依赖**：无需在Win7安装`vscode-server`或OpenSSH
- **实时同步**：文件保存后自动同步到Win7
- **跨版本兼容**：Win10可用最新版VSCode

🔧 **具体步骤**

## 1. Win7设置文件共享（Samba协议）

1. **启用共享功能**
   - 进入`控制面板 > 网络和共享中心 > 更改高级共享设置`
   - 启用：
     - 网络发现
     - 文件和打印机共享
     - 关闭密码保护共享（若无需密码）
2. **共享代码文件夹**
   - 右键代码文件夹 → **属性 → 共享 → 高级共享**
   - 勾选**共享此文件夹**，权限设为`完全控制`
   - 记录共享路径：`\\Win7_IP\共享名`

## 2. Win10映射网络驱动器

1. **永久挂载共享文件夹**
   - 打开`此电脑` → **映射网络驱动器**
   - 选择盘符（如`Z:`），输入`\\Win7_IP\共享名`
   - 勾选**重新连接时重新连接**
2. **测试访问**
   - 双击`Z:`盘应能直接看到Win7的文件

## 3. VSCode配置自动同步（两种方式）

方案A：直接编辑网络驱动器（最简单）

- 直接在VSCode打开`Z:\代码文件夹`，编辑后自动保存到Win7

