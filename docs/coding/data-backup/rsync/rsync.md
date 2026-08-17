# rsync

* [返回上层目录](../data-backup.md)
* [rsync进行增量备份](#rsync进行增量备份)
* [设置定时任务（Cron）](#设置定时任务（Cron）)



# rsync进行增量备份

如果备份的是代码、配置文件等小文件，可以直接用 `rsync` 进行增量备份，只复制有变化的部分。

```shell
#!/bin/bash

# 定义源目录（你要备份的 WSL 目录）
SOURCE_DIR="/home/你的用户名/projects"

# 定义目标目录（Windows D 盘下的备份文件夹）
DEST_DIR="/mnt/d/WSL_Backup/projects"

# 记录日志
LOG_FILE="/home/你的用户名/backup.log"

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

# 使用 rsync 进行增量同步（-av 表示归档模式 + 详细信息，--delete 表示删除目标端多余文件）
# rsync -av --delete "$SOURCE_DIR/" "$DEST_DIR/"
# 使用 rsync，并记录详细日志
if rsync -av --delete \
    --exclude="miniconda3" \
    --exclude="__pycache__" \
    --exclude=".*" \
    "$SOURCE_DIR/" "$DEST_DIR/" >> "$LOG_FILE" 2>&1; then
    echo "$(date): ✅ 备份成功" >> "$LOG_FILE"
else
    echo "$(date): ❌ 备份失败，请检查" >> "$LOG_FILE"
fi
```



# 设置定时任务（Cron）

在 WSL 里，用 Linux 标准的 `cron` 来定时执行脚本。

（1）**编辑 crontab**：

```shell
crontab -e
```

（2）**添加定时任务**（例如每天凌晨 2 点执行）：

```shell
0 2 * * * /home/你的用户名/backup_to_d.sh
```

**常用时间格式参考**：

| 时间            | Cron 表达式   |
| :-------------- | :------------ |
| 每天凌晨 2 点   | `0 2 * * *`   |
| 每 6 小时       | `0 */6 * * *` |
| 每周一凌晨 3 点 | `0 3 * * 1`   |
| 每小时          | `0 * * * *`   |

⚠️ 重要注意事项

| 问题           | 建议                                                         |
| :------------- | :----------------------------------------------------------- |
| **权限问题**   | WSL 访问 `/mnt/d/` 时，文件权限可能会映射为 Windows 的权限。如果遇到 Permission denied，可以在 WSL 里用 `sudo` 执行，或者调整 Windows 文件夹的共享权限。 |
| **路径大小写** | Windows 文件系统不区分大小写，而 WSL 是区分大小写的。如果两个文件同名仅大小写不同，在 D 盘上可能会互相覆盖。建议备份时先打包成 `.tar` 归档文件。 |
| **性能影响**   | 如果备份整个 WSL 虚拟硬盘（`ext4.vhdx` 文件），而它正好在使用中，直接复制可能会失败或产生不一致。建议备份内容型数据（如 `/home/`），而不是系统镜像文件本身。 |
| **D 盘格式**   | 如果 D 盘是 NTFS 格式，WSL 里对 `/mnt/d/` 的读写性能是足够好的，不用担心。 |

**如果关机时脚本正在执行，备份会中断，但不会造成数据损坏。下次开机后，它会在预定时间重新运行。**

