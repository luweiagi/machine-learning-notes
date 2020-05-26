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

