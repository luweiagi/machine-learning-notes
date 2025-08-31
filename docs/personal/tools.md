# 工具

* [返回上层目录](../README.md)



# KATEX在线编辑

http://latex.codecogs.com/eqneditor/editor.php



# docsify

[docsify](https://docsify.js.org/#/zh-cn/quickstart)

```shell
cd xxx  # xxx.docs
docsify init ./
docsify serve docs
```



# shell

批量替换内容

```shell
# ==================批量替换内容==================

find ./ -name "*.md" | xargs perl -pi -e "s/SUMMARY.md/README.md/gi"
find ./ -name "*.md.bak" | xargs rm -f
```

批量替换内容

```shell
# ==================批量替换内容==================

find ./ -name "*.md" | xargs sed -i "" "s/^[$][$]$/vvvv/g"

find ./ -name "*.md" | xargs sed -i "" "s/[$][$]/\$/g"

find ./ -name "*.md" | xargs sed -i "" "s/vvvv/\$\$/g"
```

批量替换内容

```shell
# ==================批量替换内容==================

find ./ -name "*.md" | xargs sed -i "" "s/- \[/\* \[/g"
```

批量删除内容

```shell
# ==================批量删除内容==================
find ./ -name "*.md" | xargs sed -i "" "s/xxx/d"
```



# python

## 统计字数

把三类统计都加上：

1. **汉字个数**：`[\u4e00-\u9fff]`
2. **英文单词个数**：用正则 `\b[a-zA-Z]+\b`（匹配连续的英文字母，算一个单词）
3. **数字个数**：用正则 `\b\d+\b`（匹配连续的数字串，算一个数字）

这样可以保证：

- 连续的数字算 **1 个**；
- 连续的英文算 **1 个单词**；
- 每个汉字算 **1 个**。

```python
import os
import re

def count_text_in_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    # 汉字数量
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # 英文单词数量
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    # 数字数量（连续数字算一个）
    numbers = re.findall(r'\b\d+\b', text)

    return len(chinese_chars), len(english_words), len(numbers)

def main():
    file_stats = []  # 保存每个文件的统计
    total_chinese = total_english = total_numbers = 0

    for root, _, files in os.walk("./machine-learning-notes"):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                c, e, n = count_text_in_file(filepath)
                total_chinese += c
                total_english += e
                total_numbers += n
                total = c + e + n
                file_stats.append((filepath, c, e, n, total))

    # 按总字数排序（从大到小）
    file_stats.sort(key=lambda x: x[4], reverse=True)

    # 输出每个文件统计
    print("📊 每个文件字数统计（按总字数排序）：")
    for filepath, c, e, n, total in file_stats:
        print(f"{os.path.basename(filepath):40} | 汉字:{c:6} | 英文单词:{e:6} | 数字:{n:6} | 总字数:{total:6}")

    # 输出汇总结果
    print("\n📈 全部文件汇总：")
    print("  汉字总数   :", total_chinese)
    print("  英文单词数 :", total_english)
    print("  数字总数   :", total_numbers)
    print("  总字数（合计）:", total_chinese + total_english + total_numbers)

if __name__ == "__main__":
    main()
```

把这段代码`wc.py`放在`machine-learning-notes`的外面，运行即可。

即

```shell
---- machine-learning-notes
---- wc.py
```

运行：

```python
python wc.py
```

