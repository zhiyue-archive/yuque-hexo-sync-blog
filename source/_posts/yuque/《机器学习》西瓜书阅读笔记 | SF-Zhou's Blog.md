
---

title: 《机器学习》西瓜书阅读笔记 | SF-Zhou&#39;s Blog

urlname: 66b45610-4514-4c48-98d5-2be4de113575

date: 2019-04-03 18:31:13 +0800

tags: []

---
第 1 章 绪论
--------

### 基本术语

`机器学习`：在计算机上从`数据`（data）中产生`模型`（model）的算法，即`学习算法`（learning algorithm）。

A computer program is said to learn from experience EEE with respect to some class of tasks TTT and performance measure PPP, if its performance at tasks in TTT, as measured by PPP, improves with experience EEE.

一般地，令 D\={x⃗1,x⃗2,⋯,x⃗m}D = \\left \\{ \\vec {x}\_1, \\vec {x}\_2, \\cdots, \\vec {x}\_m \\right \\}D\={x1​,x2​,⋯,xm​} 表示包含 mmm 个`样本`（sample）的数据集，每个示例由 ddd 个`属性`（attribute）描述，则每个样本 x⃗i\={xi1;xi2;⋯;xid}\\vec x\_i = \\left \\{x\_{i1}; x\_{i2}; \\cdots; x\_{id} \\right \\}xi​\={xi1​;xi2​;⋯;xid​} 是 ddd 维样本空间 X\\mathcal{X}X 中的一个向量，x⃗i∈X\\vec x\_i \\in \\mathcal{X}xi​∈X，其中 xijx\_{ij}xij​ 是 x⃗i\\vec x\_ixi​ 在第 jjj 个属性上的取值，ddd 称为样本 x⃗i\\vec x\_ixi​ 的`维数`（dimensionality）。

属性张成的空间称为`样本空间`（sample space），每个样本都可在这个空间中找到唯一的坐标位置，因此也把一个样本称为一个`特征向量`（feature vector）。

从数据中学得模型的过程称之为`学习`（learning）或`训练`（training），学得模型适用于新样本的能力称为`泛化`（generalization）能力。

### 假设空间

`归纳`（induction）与`演绎`（deduction）是科学推理的两大基本手段。前者是从特殊到一般的泛化（generalization）过程，后者是从一般到特殊的特化（specialization）过程。从样例中学习是一个归纳的过程，亦称`归纳学习`（inductive learning）。

狭义的归纳学习是从数据中学得`概念`（concept），最基本的概念学习是布尔概念学习。可以把学习的过程看作一个在所有`假设`（hypothesis）组成的空间中进行搜索的过程，搜索目标是找到与训练集`匹配`（fit）的假设。

假设的表示一旦确定，`假设空间`（hypothesis space）及其规模大小就确定了。现实问题中通常面临很大的假设空间，但样本训练集是有限的，因此可能有多个假设与训练集一致，即存在一个与训练集一致的假设集合，称之为`版本空间`（version space）。

### 归纳偏好

机器学习算法在学习过程中对某种类型假设的偏好，称为`归纳偏好`（inductive bias）。归纳偏好可看作是学习算法在庞大的假设空间中对假设进行选择的价值观。

`奥卡姆剃刀`（Occam's Razor）是自然科学研究中常用的原则，即若存在多个假设与观察一致，则选最简单的那个。如无必要，勿增实体。

但奥卡姆剃刀原则并不平凡，“简单”的评价标准无法量化。事实上归纳偏好对应了学习算法本身所做出的关于“什么样的模型更好”的假设。`没有免费的午餐定理`（No Free Lunch Theorem，NFL）证明了在真实目标函数 fff 均匀分布的情况下，所有学习算法学得的模型期望性能是一致的。

脱离实际问题，空谈“什么学习算法更好”毫无意义。

第 2 章 模型评估与选择
-------------

### 经验误差与过拟合

学习器的实际输出与样本的真实输出之间的差异称为`误差`（error），训练集上的误差称为`训练误差`（training error），新样本上的误差称为`泛化误差`（generalization error）。

为了使泛化误差最小化，应该从训练样本中尽可能学出适用于所有潜在样本的“普遍规律”。而将训练样本的特点当作了所有潜在样本的一般性质，导致泛化性能下降的现象，称为`过拟合`（overfitting），相对地没有充分习得训练样本的一般性质的现象，称为`欠拟合`（underfitting）。

现实任务中，存在多种学习算法、不同参数配置，产生不同的模型，需要选择其中合适的模型，该问题称为`模型选择`（model selection）问题。理想状态下使用泛化误差作为模型选择的评价标准，但泛化误差无法直接获得。

### 评估方法

通常使用`测试集`（testing set）来测试学习器对新样本的判别能力，以测试集上的`测试误差`（testing error）作为泛化误差的近似。通常假设测试样本是从样本真实分布中独立同分布采样而得。

对于包含 mmm 个样本的数据集 D\={(x⃗1,y1),(x⃗2,y2),⋯,(x⃗m,ym)}D = \\left \\{ (\\vec x\_1, y\_1), (\\vec x\_2, y\_2), \\cdots, (\\vec x\_m, y\_m) \\right \\}D\={(x1​,y1​),(x2​,y2​),⋯,(xm​,ym​)}，需要将其分解为训练集 SSS、验证集 VVV 和测试集 TTT，常用的方法有留出法、交叉验证法和`自助法`（bootstrapping）。

自助法即从数据集中进行 mmm 次可重复采样，可以选出约 36.8% 的样本作为测试集，在数据集较小时较为有效。

机器学习常涉及两类参数：一是算法的参数，称为`超参数`（hyper parameter），一是模型的参数。对超参数进行设定调优的过程称为`调参`（parameter tuning）。通常使用验证集进行模型选择和调参，使用测试集评估模型的泛化能力。

### 性能度量

性能度量（performance measure），即为模型泛化能力的评价标准。给定数据集 D\={(x⃗1,y1),(x⃗2,y2),⋯,(x⃗m,ym)}D = \\left \\{ (\\vec x\_1, y\_1), (\\vec x\_2, y\_2), \\cdots, (\\vec x\_m, y\_m) \\right \\}D\={(x1​,y1​),(x2​,y2​),⋯,(xm​,ym​)}，其中 yiy\_iyi​ 是样本 x⃗i\\vec x\_ixi​ 的真实标记。

回归任务常用的性能度量是`均方误差`（mean squared error）：

E(f;D)\=∫x⃗∼D(f(x⃗)−y)2p(x⃗)dx⃗ E(f; \\mathcal{D}) = \\int\_{\\vec x \\sim \\mathcal D} (f(\\vec x) - y)^2 p(\\vec {x}) d\\vec x E(f;D)\=∫x∼D​(f(x)−y)2p(x)dx

分类任务常用的性能度量较多，常用的错误率：

E(f;D)\=∫x⃗∼DI(f(x⃗)≠y)p(x⃗)dx⃗ E(f; \\mathcal{D}) = \\int\_{\\vec x \\sim \\mathcal D} \\mathbb I(f(\\vec x) \\neq y) p(\\vec {x}) d\\vec x E(f;D)\=∫x∼D​I(f(x)̸​\=y)p(x)dx

`准确率`（percision）和`召回率`（recall）：

P\=TPTP+FPR\=TPTP+FN \\begin{aligned} P &= \\frac {TP} {TP + FP} \\\\ R &= \\frac {TP} {TP + FN} \\end{aligned} PR​\=TP+FPTP​\=TP+FNTP​​

预测正例

预测负例

真实正例

TP

FN

真实负例

FP

TN

准确率和召回率不可得兼。以准确率作为纵轴、召回率作为横轴，可以得到`P-R曲线`，曲线中“准确率=召回率”的点成为`平衡点`（Break-Even Point）。

准确率和召回率的`调和平均`（harmonic mean）称为`F1`度量：

1F1\=12(1P+1R)F1\=2PRP+R \\begin{aligned} \\frac {1} {F1} &= \\frac {1} {2} (\\frac {1} {P} + \\frac {1} {R}) \\\\ F1 &= \\frac {2PR} {P + R} \\end{aligned} F11​F1​\=21​(P1​+R1​)\=P+R2PR​​

由多组混淆矩阵计算多组准确率和召回率，再求平均值，可得`宏准确率`（macro-P）和`宏召回率`（macro-R）；将多组混淆矩阵求平均值，再求准确率和召回率，可得`微准确率`（micro-P）和`微召回率`（micro-R）。

`ROC` 全称受试者工作特征（Receiver Operating Characteristic），该曲线以`真正例率`（True Positive Rate）为纵轴，以`假正例率`（False Positive Rate）为横轴：

TPR\=TPTP+FNFPR\=FPTN+FP \\begin{aligned} TPR &= \\frac {TP} {TP + FN} \\\\ FPR & = \\frac {FP} {TN + FP} \\end{aligned} TPRFPR​\=TP+FNTP​\=TN+FPFP​​

ROC 曲线下的面积称为`AUC`（Area Under ROC Curve），通常使用 AUC 作为ROC 曲线优劣的判断依据。

不同类型的错误所造成的后果不同，为权衡不同类型错误所造成的不同损失，可为错误赋予`非均等代价`（unequal cost）。令 D+D^+D+ 与 D−D^-D− 代表数据集 DDD 中的正例子集和反例子集，则`代价敏感`（cost-sensitive）错误率为：

E(f;D;cost)\=1m(∑x⃗i∈D+I(f(x⃗i)≠yi)cost01+∑x⃗i∈D−I(f(x⃗i)≠yi)cost10) E(f; D; cost) = \\frac {1} {m} \\left ( \\sum\_{\\vec x\_i \\in D^+} \\mathbb I (f(\\vec x\_i) \\neq y\_i) cost\_{01} + \\sum\_{\\vec x\_i \\in D^-} \\mathbb I (f(\\vec x\_i) \\neq y\_i) cost\_{10} \\right ) E(f;D;cost)\=m1​⎝⎛​xi​∈D+∑​I(f(xi​)̸​\=yi​)cost01​+xi​∈D−∑​I(f(xi​)̸​\=yi​)cost10​⎠⎞​

### 偏差与方差

`偏差-方差分解`（bias-veriance decomposition）是解释学习算法泛化性能的一种重要工具。对测试样本 x⃗\\vec xx，令 yDy\_DyD​ 为 x⃗\\vec xx 在数据集中的标记，yyy 为 x⃗\\vec xx 的真实标记，f(x⃗;D)f(\\vec x; D)f(x;D) 为训练集 DDD 上学的模型 fff 在 x⃗\\vec xx 上的预测输出。以回归任务为例，学习算法的期望预测为：

fˉ(x⃗)\=ED\[f(x⃗;D)\] \\bar f(\\vec x) = \\mathbb E\_D \\left \[ f(\\vec x; D) \\right \] fˉ​(x)\=ED​\[f(x;D)\]

期望输出与真实标记的差别称为`偏差`（bias）：

bias2(x⃗)\=(fˉ(x⃗)−y)2 bias^2(\\vec x) = \\left ( \\bar f(\\vec x) - y \\right )^2 bias2(x)\=(fˉ​(x)−y)2

使用样本数相同的不同训练集产生的`方差`（variance）为：

var(x⃗)\=ED\[(f(x⃗;D)−fˉ(x⃗))2\] var(\\vec x) = \\mathbb E\_D \\left \[ \\left (f(\\vec x; D) - \\bar f(\\vec x) \\right )^2 \\right \] var(x)\=ED​\[(f(x;D)−fˉ​(x))2\]

噪声为：

ε2\=ED\[(yD−y)2\] \\varepsilon ^2 = \\mathbb E\_D \\left \[ (y\_D - y)^2 \\right \] ε2\=ED​\[(yD​−y)2\]

假定噪声的期望为零，可得：

E(f;D)\=bias2(x⃗)+var(x⃗)+ε2 E(f; D) = bias^2(\\vec x) + var(\\vec x) + \\varepsilon ^2 E(f;D)\=bias2(x)+var(x)+ε2

即泛化误差可以分解为偏差、方差和噪声之和。偏差和方差间存在`偏差-方差窘境`（bias-variance dilemma），当学习算法训练不足时，学习器的拟合能力不够强，偏差主导了泛化错误率；当训练程度加深后，学习器的拟合能力足够，方差主导了泛化错误率。

第 3 章 线性模型
----------

### 基本形式

给定由 ddd 个属性描述的示例 x\=(x1;x2;⋯;xd)\\boldsymbol x = (x\_1; x\_2; \\cdots; x\_d)x\=(x1​;x2​;⋯;xd​)，其中 xix\_ixi​ 是 x\\boldsymbol xx 在第 iii 个属性上的取值，`线性模型`（linear model）试图学得一个通过属性的线性组合来进行预测的函数：

f(x)\=wTx+b f(\\boldsymbol x) = \\boldsymbol w^T \\boldsymbol x + b f(x)\=wTx+b

其中 w\=(w1;w2;⋯;wd)\\boldsymbol w = (w\_1; w\_2; \\cdots; w\_d)w\=(w1​;w2​;⋯;wd​)。

线性模型形式简单，易于建模，且 w\\boldsymbol ww 直观表达了各属性在预测中的重要性，因此线性模型有很好的 `可解释性`（comprehensibility）。

在线性模型的基础上可通过引入层级结构或高维映射而得到更为强大的`非线性模型`（nonlinear model）。

### 线性回归

给定数据集 D\={(x1,y1),(x2,y2),⋯,(xm,ym)}D = \\{(\\boldsymbol x\_1, y\_1), (\\boldsymbol x\_2, y\_2), \\cdots , (\\boldsymbol x\_m, y\_m)\\}D\={(x1​,y1​),(x2​,y2​),⋯,(xm​,ym​)}，其中 xi\=(xi1,xi2,⋯,xid)\\boldsymbol x\_i = (x\_{i1}, x\_{i2}, \\cdots, x\_{id})xi​\=(xi1​,xi2​,⋯,xid​)，y∈Ry \\in \\mathbb Ry∈R，`线性回归`（linear regression）试图学得一个线性模型以尽可能准确地预测实际输出标记。

考虑最简单的单属性情形，D\={(xi,yi)}i\=1mD = \\left \\{ (x\_i, y\_i) \\right \\}\_{i=1}^mD\={(xi​,yi​)}i\=1m​，线性回归试图学得

f(xi)\=wxi+b f(x\_i) = wx\_i+b f(xi​)\=wxi​+b

以使得 f(xi)≃yif(x\_i) \\simeq y\_if(xi​)≃yi​。使用均方误差作为衡量 f(x)f(x)f(x) 与 yyy 之间差别的性能度量：

E(w,b)\=∑i\=1m(f(xi)−yi)2 E\_{(w, b)} = \\sum\_{i=1}^{m} {\\left (f(x\_i) - y\_i \\right )^2} E(w,b)​\=i\=1∑m​(f(xi​)−yi​)2

则：

(w∗,b∗)\=arg⁡min⁡(w,b)E(w,b) (w^\*, b^\*) = \\underset {(w, b)} {\\arg \\min} E\_{(w, b)} (w∗,b∗)\=(w,b)argmin​E(w,b)​

均方误差有非常好的几何意义，它对应了 `欧氏距离`（Euclidean distance）。基于均方误差最小化来进行模型求解的方法称为 `最小二乘法`（least square method）。在线性回归中，最小二乘法试图找到一条直线，使得所有样本到直线上的欧氏距离之和最小。

E(w,b)E\_{(w, b)}E(w,b)​ 是关于 www 和 bbb 的凸函数。对于区间 \[a,b\]\[a, b\]\[a,b\] 上定义的函数 fff，若它区间中任意两点 x1x\_1x1​ 和 x2x\_2x2​ 均有 f(x1+x22)≤f(x1)+f(x2)2f(\\frac {x\_1 + x\_2} {2}) \\le \\frac {f(x\_1) + f(x\_2)} {2}f(2x1​+x2​​)≤2f(x1​)+f(x2​)​，则称 fff 为区间 \[a,b\]\[a, b\]\[a,b\] 上的凸函数。对实数集上的函数，可以通过求二阶导数的方式来判断，二阶导数在区间上非负则称为凸函数。

求解 www 和 bbb 使均方误差最小化的过程，称为线性回归模型的最小二乘 `参数估计`（parameter estimation）。将 E(w,b)E\_{(w, b)}E(w,b)​ 分别对 www 和 bbb 求导，得到：

∂E(w,b)∂w\=2(w∑i\=1mxi2−∑i\=1m(yi−b)xi)∂E(w,b)∂b\=2(mb−∑i\=1m(yi−wxi)) \\begin {aligned} \\frac {\\partial E\_{(w, b)}} {\\partial w} &= 2 \\left ( w \\sum\_{i=1}^{m} x\_i^2 - \\sum\_{i=1}^{m} (y\_i - b)x\_i \\right ) \\\\ \\frac {\\partial E\_{(w, b)}} {\\partial b} &= 2 \\left ( mb - \\sum\_{i=1}^{m} (y\_i - wx\_i) \\right ) \\end {aligned} ∂w∂E(w,b)​​∂b∂E(w,b)​​​\=2(wi\=1∑m​xi2​−i\=1∑m​(yi​−b)xi​)\=2(mb−i\=1∑m​(yi​−wxi​))​

对 www 和 bbb 的偏导置零可得到 www 和 bbb 最优解的 `闭式解`（closed-form solution）：

w\=∑i\=1mxiyi−mxˉyˉ∑i\=1mxi2−mxˉ2b\=yˉ−wxˉ \\begin {aligned} w &= \\frac {\\sum\_{i=1}^{m}x\_i y\_i - m \\bar x \\bar y} {\\sum\_{i=1}^{m} {x\_i^2} - m \\bar x^2} \\\\ b &= \\bar y - w \\bar x \\end {aligned} wb​\=∑i\=1m​xi2​−mxˉ2∑i\=1m​xi​yi​−mxˉyˉ​​\=yˉ​−wxˉ​

更一般的情形，给定数据集 D\={(xi,yi)}i\=1mD = \\left \\{ (\\boldsymbol x\_i, y\_i) \\right \\}\_{i=1}^mD\={(xi​,yi​)}i\=1m​，其中 xi\=(xi1,xi2,⋯,xid)\\boldsymbol x\_i = (x\_{i1}, x\_{i2}, \\cdots, x\_{id})xi​\=(xi1​,xi2​,⋯,xid​)，y∈Ry \\in \\mathbb Ry∈R，线性回归试图学得：

f(xi)\=wTxi+b f(\\boldsymbol x\_i) = \\boldsymbol w^T \\boldsymbol x\_i+b f(xi​)\=wTxi​+b

使得 f(xi)≃yif(\\boldsymbol x\_i) \\simeq y\_if(xi​)≃yi​。这称为 `多变量线性回归`（multivariate linear regression）。

\[未完待续\]

### 参考文献

1.  周志华. "机器学习." 清华大学出版社，北京.
