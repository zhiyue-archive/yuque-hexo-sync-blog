
---

title: Deep Learning重要发展脉络

urlname: wgmy6r

date: 2018-12-26 22:32:12 +0800

tags: []

---
Deep Learning（深度学习）的概念源于人工神经网络的研究，它的概念由Hinton等人于2006年提出，但它的模型经历了怎样的发展和演化，本文将为您深度解读Deep Learning的前世今生。<br />![](https://static.aminer.cn/rcd/article/expertpic/aminer.gif#align=left&display=inline&height=1205&originHeight=1205&originWidth=1694&search=&status=done&width=1694)<br />
<a name="tfacwu"></a>
### 脉络一  cv/tensor

1943年  心理学家麦卡洛可（W·McCulloch）和数理逻辑学家皮茨（W·Pitts）参考了生物神经元的结构，发表了《神经活动中思想内在性的逻辑运算》一文，提出了抽象的神经元模型MP，该模型可以看做深度学习的雏形。

1957年  认知心理学大师 Frank Rosenblatt 发明了感知机（Perceptron，又称感知器），感知机是当时首个可以学习的人工神经网络，掀起了一股学习的热潮。

1969年  人工智能大师 Marvin Minksy 和 Seymour Papert 在《Perceptron》一书中，用详细的数学证明了感知机的弱点，没有隐层的简单感知机在许多像XOR问题的情形下显得无能为力，并证明了简单感知机只能解决线性分类问题和一阶谓诃同题。神经网络研究进入冰河期。

1984年  日本学者福岛邦彦（Kunihiko Fukishima）提出了卷积神经网络的原始模型神经感知机（Neocognitron），产生了卷积和池化的思想（当时不叫卷积和池化）。

1986年  Hinton等人正式提出一般 Delta 法则，即反向传播（BP）算法，并用反向传播训练MLP（多层感知机）。但其实在他提出之前，已经有人将其付诸实际。（1985年 Parter 也独立地得出过相似的算法，他称之为学习逻辑。此外，1985年 Lecun 也研究出大致相似的学习法则。）

1998年  以 Yann LeCun 为首的研究人员实现了一个5层的卷积神经网络——LeNet-5，以识别手写数字。LeNet-5 标志着 CNN（卷积神经网络）的真正面世，LeNet-5 的提出把 CNN 推上了一个小高潮。

之后SVM（支持向量机）兴起，SVM在计算及准确度上都有较大的优势，导致卷积神经网络的方法在后来的一段时间并未能火起来。

2012年  Hinton组的 AlexNet 在 ImageNet 上以巨大优势夺冠，掀起了深度学习的热潮。AlexNet 可以算是 LeNet 的一种更深更宽的版本，并加上了 relu、dropout 等技巧。

这条思路被后人发展，出现了 VGG，GoogLeNet 等网络。

2016年  青年计算机视觉科学家何恺明在层次之间加入了跳跃连接，Resnet 极大增加了网络深度，效果有很大提升。另一个将这条思路继续发展下去的是去年cvpr best paper densenet。

除此之外，cv领域的特定任务还出现了各种各样的模型（Mask-RCNN等），这里不一一介绍。

2017年  Hinton认为反省传播和传统神经网络有缺陷，继而提出了 Capsule Net。但是目前在 cifar 等数据集上效果一般，这个思路还需要继续验证和发展。

<a name="orhvmd"></a>
### 脉络二  生成模型

传统的生成模型是要预测联合概率分布P(x,y)。

RBM（受限玻尔兹曼机）这个模型其实是一个基于能量的模型，1986年的时候就有，2006年将其重新拿出来作为一个生成模型，并且堆叠成为deep belief network，使用逐层贪婪或者wake-sleep的方法训练，不过这个模型效果一般，现在已经没什么人提了。但是Hinton等人却从此开始使用深度学习重新包装神经网络。

Auto-Encoder是上个世纪80年代hinton提出的模型，如今由于计算能力的进步重新登上舞台。2008年，Bengio等人又搞了denoise Auto-Encoder。

Max welling等人使用神经网络训练一个有一层隐变量的图模型，由于使用了变分推断，最后长得跟auto-encoder有点像，因而被称为Variational auto-encoder。此模型可以通过隐变量的分布采样，经过后面的decoder网络直接生成样本。

GAN（生成对抗网络）是于2014年提出的模型，如今炙手可热。它是一个生成模型，通过判别器D（Discriminator）和生成器G（Generator）的对抗训练，直接使用神经网络G隐式建模样本整体的概率分布。每次运行便相当于从分布中采样。

DCGAN是一个相当好的卷积神经网络实现，而WGAN则是通过维尔斯特拉斯距离替换原来的JS散度来度量分布之间的相似性的工作，使得训练稳定。PGGAN则逐层增大网络，生成机器逼真的人脸。

<a name="rkublv"></a>
### 脉络三 sequencelearning

1982年  出现的hopfield network有了递归网络的思想。

1997年  Jürgen Schmidhuber发明LSTM，并做了一系列的工作。但是更有影响力的还是2013年由Hinton组使用RNN做的语音识别工作，这种方法比传统方法更强。

文本方面，Bengio在svm最火的时期提出了一种基于神经网络的语言模型，后来Google提出的word2vec也包含了一些反向传播的思想。在机器翻译等任务上，逐渐出现了以RNN为基础的seq2seq模型（序列模型），模型通过一个encoder（编码器）把一句话的语义信息压成向量再通过decoder（解码器）输出，当然更多的还要和Attention Model（注意力模型）结合。

后来，大家发现使用以字符为单位的CNN模型在很多语言任务也有不俗的表现，而且时空消耗更少。LSTM/RNN 模型中的Attention机制是用于克服传统编码器-解码器结构存在的问题的。其中，self-attention（自注意力机制）实际上就是采取一种结构去同时考虑同一序列局部和全局的信息，Google就有一篇耸人听闻的Attention Is All You Need的文章。

<a name="65uhmf"></a>
### 脉络四 deepreinforcement learning

该领域最出名的是DeepMind，这里列出的David Silver则是一直研究rl（强化学习）的高管。

q-learning是很有名的传统rl算法，deep q-learning则是将原来的q值表用神经网络代替。利用deep q-learning制作的打砖块的游戏十分有名。后来David Silver等人又利用其测试了许多游戏，发在了Nature上。

增强学习在double duel的进展，主要是Qlearning的权重更新时序上。

DeepMind的其他工作诸如DDPG、A3C也非常有名，它们是基于policy gradient和神经网络结合的变种。

大家都知道的一个应用是AlphaGo，里面不仅使用了rl的方法，也包含了传统的蒙特卡洛搜索技巧。Alpha Zero 则是他们搞了一个用Alphago框架来打其他棋类游戏的游戏，而且这个“打”还是吊打的打。



