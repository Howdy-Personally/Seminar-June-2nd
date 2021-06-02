# Seminr-June-2nd

## content
### 一、NLP Transformer
### 二、Vision Transformer
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

### 二、Vision Transformer
ViT将Transformer巧妙的应用于图像分类任务，更少计算量下性能跟SOTA相当。
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic9.jpg)
ViT将输入图片拆分成16x16个patches，每个patch做一次线性变换降维同时嵌入位置信息，然后送入Transformer，避免了像素级attention的运算。类似BERT[class]标记位的设置，ViT在Transformer输入序列前增加了一个额外可学习的[class]标记位，并且该位置的Transformer Encoder输出作为图像特征。
ViT舍弃了CNN的归纳偏好问题，更加有利于在超大规模数据上学习知识，即大规模训练优归纳偏好，在众多图像分类任务上直逼SOTA(state of the art)。

### 三、Detection Transformer
DETR使用set loss function作为监督信号来进行端到端训练，然后同时预测所有目标，其中set loss function使用bipartite matching算法将pred目标和gt目标匹配起来。直接将目标检测任务看成set prediction问题，使训练过程变的简洁，并且避免了anchor、NMS等复杂处理。

DETR主要有两个部分：architecture和set prediction loss
#### 1.Architecture
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/detrstruct.png)
DETR先用CNN将输入图像embedding成一个二维表征，然后将二维表征转换成一维表征并结合positional encoding一起送入encoder，decoder将少量固定数量的已学习的object queries(可以理解为positional embeddings)和encoder的输出作为输入。最后将decoder得到的每个output embdding传递到一个共享的前馈网络(FFN)，该网络可以预测一个检测结果(包括类和边框)或着“没有目标”的类。
##### 1.1 Transformer
![avater](https://github.com/Howdy-Personally/Seminar-June-2nd/blob/main/pic/pic10.jpg)
##### 1.1.1 Encoder
将Backbone输出的feature map转换成一维表征，得到 特征图，然后结合positional encoding作为Encoder的输入。每个Encoder都由Multi-Head Self-Attention和FFN组成。
和Transformer Encoder不同的是，因为Encoder具有位置不变性，DETR将positional encoding添加到每一个Multi-Head Self-Attention中，来保证目标检测的位置敏感性。
##### 1.1.2 Decoder
因为Decoder也具有位置不变性，Decoder的N个object query(可以理解为学习不同object的positional embedding)必须是不同，以便产生不同的结果，并且同时把它们添加到每一个Multi-Head Attention中。N个object queries通过Decoder转换成一个output embedding，然后output embedding通过FFN独立解码N个预测结果，包含box和class。对输入embedding同时使用Self-Attention和Encoder-Decoder Attention，模型可以利用目标的相互关系来进行全局推理。
和Transformer Decoder不同的是，DETR的每个Decoder并行输出N个对象，Transformer Decoder使用的是自回归模型，串行输出N个对象，每次只能预测一个输出序列的一个元素。
##### 1.1.3 FFN
FFN由3层perceptron和一层linear projection组成。FFN预测出box的归一化中心坐标、长、宽和class。
DETR预测的是固定数量的N个box的集合，并且N通常比实际目标数要大的多，所以使用一个额外的空类来表示预测得到的box不存在目标。
### 四、我的工作
感觉DETR更重要的意义应当是让NLP任务和CV任务之间的协同融合变得更加值得期待，倒不是建立了更有效的目标检测新范式。主流的目标检测算法可以说是一种分类任务，而transfomer将目标检测任务转化为一个序列预测的任务，使用transformer编码-解码器结构和双边匹配的方法，由输入图像直接得到预测结果序列。
