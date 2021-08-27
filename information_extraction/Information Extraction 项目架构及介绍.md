## Information Extraction 项目架构及介绍

------



**写在开始：**

信息抽取模型的思路来源于@苏剑林的一篇bert4keras的信息抽取文章，链接：

[用bert4keras做三元组抽取 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/7161)

思路就是来源于苏神的这篇文章





**模型介绍：**

构建三个模型：subject模型，object模型和predicate模型

其中subject模型和object模型以序列标注的方式进行构建因此采用的模型是bert-crf，其中bert的编码向量将会共享与三个模型

因此

subject模型：bert-crf

object模型：layer-normal-crf

predicate模型：layer-normal-dense

其中张量的融合与交互采用(Conditional) Layer Normalization的方法（原方法来源是bert4keras，此处简单重写了该类）





**处理流程：**

content输入-------->tokenizer-------->bert-crf模型-------->获取subject和对应的bert编码向量-------->获取subject起始位置处的张量-------->

与bert编码混合作为object模型的输入-------->获取object-------->subject和object的起始张量同bert编码融合-------->获取predicate





**模型架构：**

![模型架构](E:\tang_nlp\information_extraction\模型架构.png)





**label的格式：**

例如：张大三在电影中扮演罪犯

subject需要获取的是**张大三**

因此它的输入label是

[1,1,10,0,0,0,0,0,0,0] 



object需要获取的是**罪犯**

因此它的输入label是

[0,0,0,0,0,0,0,0,0,1,1]



predicate是一个多分类模型因此它的输入格式取决于数据所含有的所有predicate种类：

假设在数据中总计有6中predicate那么输入格式就是

[[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0]]

每个小**[]**代指一种predicate，其中

**[0]**指的是：不是该predicate，

**[1]**指的是：是该predicate

那么在该例子中第四个小[]就代表了**扮演**这一predicate因此[]显示的是[0, 1]

选用该种输入的原因是同一组subject和object可能包含多种predicate

比如**张三**和**李四**既可以有**朋友**关系，也可以有**同学**关系





**后续拓展：**

重点集中在predicate模型上面，假如给定的是固定的数据有固定的predicate类型，那么将其做成一个分类模型也无可厚非。

但是假如给定的是一个开放的数据，没有给你predicate的所有类型，或者说每一组subject和object都有可能产生新的predicate，那么做成分类模型显然是不可以的，因此predicate模型就可以从分类模型变成一个文本生成的模型，也就是说可以将其做成一个seq2seq的模型（从某种意义上来说，文本生成可以做nlp的所有种类的任务，前提是机器够牛逼，数据够大且质量有保障）



