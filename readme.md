# AnnotatedTransformer中文pycharm项目版

原项目代码来自于[Annotated-Transformer](https://github.com/harvardnlp/annotated-transformer)，本项目主要是将原文中的jupyter版本的代码转换成pycharm项目的形式来进行注释和运行（jupyter中的可视化部分没有添加到项目中）。

另外，就运行该代码中间出现的问题进行了记录，希望帮助到后面学习的人。

## 出现的问题

只要按照原项目的环境进行安装，最容易出现的问题就是以下两个方面的问题（或许多卡训练部分的代码运行也会出问题，但是我没有多卡\doge）。

### spacy库

完全按照原项目中的requirements进行库的安装后，spacy库调用过程中会出现一系列问题，其中有可能会出现一个报错：

```
TypeError: issubclass() arg 1 must be a class
```

这个错误可以根据[这个链接](https://stackoverflow.com/questions/77037891/typeerror-issubclass-arg-1-must-be-a-class)得到解决。

另外，还有可能出现分词器模型下载出错的问题，这个需要到[这个github页面](https://github.com/explosion/spacy-models/releases)中找到匹配spacy的模型版本进行下载（在原项目中spacy的版本是3.2，所以下载的模型也应该是对应这个版本）。

### multi30k下载出错

这个主要原因是原数据下载网址已经失效了，在本项目的代码中已经增加了修改代码解决了这个问题，解决方案来源于[这个链接](https://github.com/harvardnlp/annotated-transformer/issues/96)。