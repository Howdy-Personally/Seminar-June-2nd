# Seminr-June-2nd

## content
### 一、NLP Transformer
### 二、Visual Transformer
### 三、Detection Transformer
### 四、My Work


## Introduction
### 1. NLP Transformer
Transformer由论文《Attention is All You Need》提出
这是论文的链接https://arxiv.org/abs/1706.03762

Transformer改进了RNN最被人诟病的训练慢的缺点，利用self-attention机制实现快速并行。并且Transformer可以增加到非常深的深度，充分发掘DNN模型的特性，提升模型准确率。
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/TheTransformerModelArchitecture.png)
#### 从宏观的视角开始
在机器翻译中，就是输入一种语言，输出另一种语言。
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic1.jpg)
transformer这个黑箱，是由编码组件、解码组件和它们之间的连接组成。
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic2.jpg)
编码组件部分由一堆encoder构成。解码组件部分也是由相同数量的decoder组成的。
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic3.jpg)
每个解码器都可以分解成两个子层，self attention和feed forward neural network
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic4.jpg)
self attention帮助编码器在对每个单词编码时关注输入句子的其他单词，编码器中还有一个注意力层，用来关注输入句子的相关部分。
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic5.jpg)
然后肯定有人想问注意力机制和自注意力机制的区别
注意力机制是发生在编码器和解码器之间，也可以说是发生在输入句子和生成句子之间。而自注意力模型中的自注意力机制则发生在输入序列内部，或者输出序列内部，可以抽取到同一个句子内间隔较远的单词之间的联系，比如句法特征
解释一下什么是自注意力机制self attention
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/attention.gif)
这篇解读来源于 https://blog.csdn.net/longxinchen_ml/article/details/86533005


### 二、Detection Transformer
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/detrstruct.png)
### 三、我的工作
感觉DETR更重要的意义应当是让NLP任务和CV任务之间的协同融合变得更加值得期待，倒不是建立了更有效的目标检测新范式。主流的目标检测算法可以说是一种分类任务，而transfomer将目标检测任务转化为一个序列预测的任务，使用transformer编码-解码器结构和双边匹配的方法，由输入图像直接得到预测结果序列。
