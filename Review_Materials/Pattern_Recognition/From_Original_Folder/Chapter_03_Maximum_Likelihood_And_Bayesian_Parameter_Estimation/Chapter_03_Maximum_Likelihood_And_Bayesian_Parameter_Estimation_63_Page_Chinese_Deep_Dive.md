# 《模式识别》第三章：极大似然与贝叶斯参数估计——63 页逐页中文精讲

> **Chapter 3 — Maximum-Likelihood and Bayesian Parameter Estimation**  
> 基于上传的 `Chapter 3.pdf`。按 **PDF 实际页序 1–63** 编排。课件右下角的原幻灯片编号有跳号：PDF 第 15 页对应课件 16 页，PDF 第 27 页对应课件 33 页，PDF 第 63 页对应课件 69 页。此处一律以 PDF 页序为主。  
> 结构：**① 原文中文翻译／关键板书 ② 零基础概念解释 ③ 推导、图示或例题与易错点**。原文中可见的公式尽量完整转写；“拓展”均为辅助教学，不冒充课件原文。公式用 LaTeX 表示。

## 阅读前先建立知识地图

第二章提出分类规则：

$$
P(\omega_j\mid\mathbf x)=\frac{p(\mathbf x\mid\omega_j)P(\omega_j)}{\sum_i p(\mathbf x\mid\omega_i)P(\omega_i)}.
$$

第三章要解决的是：**这些先验概率和类条件分布从哪里来？** 主要分为四段：

1. **PDF 第 1–14 页**：极大似然估计（MLE），把参数当作固定但未知的数。
2. **PDF 第 15–27 页**：贝叶斯参数估计，把参数当作随机变量，并综合先验与样本。
3. **PDF 第 28–50 页**：相关专题一，隐马尔可夫模型（HMM），学习序列概率、前向／后向算法、Viterbi 解码、Baum–Welch 学习。
4. **PDF 第 51–63 页**：相关专题二，贝叶斯信念网络（BBN），理解有向无环图、联合分布分解、基于证据的概率推理；最后四页为总结。

符号约定：训练集 $\mathcal D=\{\mathbf x_1,\ldots,\mathbf x_n\}$；参数 $\boldsymbol\theta$；$p$ 常表示概率密度，$P$ 常表示离散事件概率；$\hat\theta$ 表示估计量。注意课件的字母 $p$ 也可以指参数维度，要根据上下文区分。

---

## 第一部分：极大似然估计（PDF 第 1–14 页）

### PDF 第 1 页｜章节标题：Maximum-Likelihood and Bayesian Parameter Estimation

**① 原文中文翻译**

第三章：**极大似然与贝叶斯参数估计**。课程名称为 Pattern Recognition（模式识别）。

**② 概念解释**

第二章我们知道了“若已知各类别概率分布，如何分类”；第三章研究“若分布里有未知参数，如何用训练数据估计它们”。例如正态分布由均值 $\mu$ 和方差 $\sigma^2$ 决定，未知参数可以写为 $\theta=(\mu,\sigma^2)$。

**③ 课堂提示**

本章有两个读音近似但含义不同的词：**Likelihood（似然）**描述固定观测数据对参数的支持程度；**Probability（概率）**通常描述给定模型后数据出现的可能性。后续将严格区分“参数固定，数据变化”和“数据已观测，参数变化”的视角。

### PDF 第 2 页｜贝叶斯分类定理回顾

**① 原文中文翻译**

课件重新写出贝叶斯公式：

$$
P(\omega_j\mid\mathbf x)=\frac{p(\mathbf x\mid\omega_j)P(\omega_j)}{p(\mathbf x)},\qquad 1\le j\le c.
$$

要计算后验概率，需要先验概率 $P(\omega_j)$ 和似然（类条件概率密度）$p(\mathbf x\mid\omega_j)$。训练样本由 $c$ 个数据集组成：$\mathcal D_j$ 对应类别 $\omega_j$。其中每个样本来自该类别的类条件分布，且同一个 $\mathcal D_j$ 中样本被视为 **i.i.d.（独立同分布）**。

**② 概念解释**

**独立（Independent）**：知道一个训练样本的具体值，不会在模型假设下改变另一个样本的条件分布。**同分布（Identically Distributed）**：同一类别数据按照相同的概率规律产生。比如某类别下 100 条鱼的长度，被视为从同一个 $p(x\mid\omega_j)$ 独立抽取。

**③ 推导与记忆**

若 $x_1,\ldots,x_n$ 独立同分布：

$$
p(x_1,\ldots,x_n\mid\theta)=\prod_{k=1}^np(x_k\mid\theta).
$$

这个“乘起来”的公式，是极大似然估计能够建立目标函数的基础。真实数据可能存在时间相关性，届时 i.i.d. 假设不再适用，后半章 HMM 正是研究有时序结构的另一类模型。

### PDF 第 3 页｜参数形式与非参数形式

**① 原文中文翻译**

先验概率容易通过样本数量估计：

$$
P(\omega_j)\approx\frac{|\mathcal D_j|}{\sum_{i=1}^c|\mathcal D_i|}.
$$

这里 $|\cdot|$ 表示集合包含的元素个数。估计类条件密度有两类情况：**情况 I**：密度具有某种确定的参数形式，例如高斯分布 $p(\mathbf x\mid\omega_j)\sim N(\boldsymbol\mu_j,\Sigma_j)$，参数为 $\theta_j=\{\boldsymbol\mu_j,\Sigma_j\}$；为了明确依赖关系写成 $p(\mathbf x\mid\omega_j,\theta_j)$。**情况 II**：不假设具体参数形式，将在第四章讨论。

课件补充：$d$ 维高斯分布若使用完整对称协方差矩阵，参数数量为 $d+d(d+1)/2$。

**② 概念解释**

例如你认为身高数据“近似钟形”，只需要估计均值与方差，便可得到整条高斯密度曲线——这是**参数法**。如果不预先指定钟形分布，而直接用样本估计曲线形状，则属于非参数估计思路。

**③ 为什么参数数目是这个公式？**

均值向量需要 $d$ 个数字；对称的 $d\times d$ 协方差矩阵只需提供 $d$ 个对角元素和 $d(d-1)/2$ 个不同的非对角元素，所以总数是 $d+d(d+1)/2$。例如 $d=2$ 时共 $2+3=5$ 个自由参数。

### PDF 第 4 页｜参数估计的两种假设

**① 原文中文翻译**

参数化的类条件概率密度 $p(\mathbf x\mid\omega_j,\theta_j)$ 可以使用两种估计思路：

- **假设 I：极大似然估计（Maximum-Likelihood, ML）**。将参数视为数值固定但未知的量。通过最大化实际训练样本的似然，估计参数值。
- **假设 II：贝叶斯估计（Bayesian Estimation）**。将参数视为有已知先验分布的随机变量。通过观测训练样本，根据贝叶斯公式把参数的先验分布更新为后验分布。

**② 概念解释**

MLE 说：“虽然我不知道真实均值是多少，但它是一项固定的数；找一个最能解释观测数据的数字。”贝叶斯估计说：“我对均值有一个不确定性分布，数据来了以后，这个分布应该如何变化？”

**③ 对比表**

| 问题 | ML | 贝叶斯估计 |
|---|---|---|
| 参数 $\theta$ | 固定但未知 | 用随机变量描述不确定性 |
| 核心 | 最大化 $p(\mathcal D\mid\theta)$ | 计算 $p(\theta\mid\mathcal D)$ |
| 结果 | 通常先得到一个点估计 $\hat\theta$ | 获得参数的后验分布，并可得到预测分布 |
| 是否需要参数先验 | 通常不需要 | 需要 |

### PDF 第 5 页｜MLE 的任务与简化条件

**① 原文中文翻译**

每个类别的似然函数由未知但固定的参数控制，形式为 $p(\mathbf x\mid\omega_j,\theta_j)$。任务是根据 $\{\mathcal D_j\}_{j=1}^c$ 估计 $\{\theta_j\}_{j=1}^c$。为了简化讨论，课件假设**一个类别的数据不提供其他类别参数的信息**，因此每个类别可以分别处理。简化记号为：

$$
\mathcal D_j\to\mathcal D,\qquad\theta_j\to\theta.
$$

**② 概念解释**

假设区分鲈鱼和鲑鱼，分别采用一个高斯分布建模。我们先用鲈鱼样本估计鲈鱼的 $\mu_1,\sigma_1^2$，再用鲑鱼样本估计鲑鱼的 $\mu_2,\sigma_2^2$。这样不用同时处理所有类别。

**③ 注意**

“不同类别样本相互不提供信息”只是本页为方便推导引入的**建模简化**，不是所有模型的普遍真理。共享协方差、多任务模型或者层级贝叶斯模型可以在不同类别之间共享信息。

### PDF 第 6 页｜似然函数与极大似然估计定义

**① 原文中文翻译**

参数记为 $\theta$，同一类别的独立同分布训练集为 $\mathcal D=\{x_1,x_2,\ldots,x_n\}$。目标函数：

$$
\boxed{L(\theta)=p(\mathcal D\mid\theta)=\prod_{k=1}^np(x_k\mid\theta)}.
$$

极大似然估计：

$$
\boxed{\hat\theta_{\rm ML}=\arg\max_\theta p(\mathcal D\mid\theta)}.
$$

课件的直观解释：最优参数与已经观测到的实际训练样本最一致。

**② 概念解释**

$\arg\max$ 返回的是“使函数取得最大值的参数”，而不是最大函数值本身。例：若 $L(\theta)$ 在 $\theta=5$ 处最大，则 $\hat\theta=5$。

**③ 小例子：估计硬币正面概率**

假设一枚硬币出现正面的未知概率是 $\theta$，投 10 次有 7 次正面。忽略与 $\theta$ 无关的组合系数：

$$
L(\theta)\propto\theta^7(1-\theta)^3.
$$

最大点位于 $\hat\theta=7/10=0.7$。这不是证明硬币的真实正面概率必然是 0.7，而是基于当前观测的点估计。

### PDF 第 7 页｜梯度算子与对数似然

**① 原文中文翻译**

给定 $p$ 维参数向量 $\theta=(\theta_1,\ldots,\theta_p)^T$ 和实值函数 $f:\mathbb R^p\to\mathbb R$，梯度算子为：

$$
\nabla_\theta f=\begin{bmatrix}\partial f/\partial\theta_1\\\vdots\\\partial f/\partial\theta_p\end{bmatrix}.
$$

课件举例：$f(\theta)=\theta_1^2+3\theta_1\theta_2$，所以

$$
\nabla_\theta f=\begin{bmatrix}2\theta_1+3\theta_2\\3\theta_1\end{bmatrix}.
$$

定义**对数似然函数**：

$$
\ell(\theta)=\ln p(\mathcal D\mid\theta).
$$

因为对数单调递增：

$$
\arg\max_\theta p(\mathcal D\mid\theta)=\arg\max_\theta\ell(\theta).
$$

**② 概念解释**

偏导数 $\partial f/\partial\theta_1$ 是“其他参数暂时不变时，函数随 $\theta_1$ 如何变化”；梯度把所有偏导数集合起来。对数似然使很长的乘积变成简单的求和，尤其适合求导。

**③ 易错点**

写作“取对数之后最大值保持不变”不够准确：**最大点的位置不变，但最大函数值数值会改变**。

### PDF 第 8 页｜对数似然的梯度与必要条件

**① 原文中文翻译**

对数似然：

$$
\ell(\theta)=\ln\prod_{k=1}^np(x_k\mid\theta)=\sum_{k=1}^n\ln p(x_k\mid\theta).
$$

因此：

$$
\boxed{\nabla_\theta\ell(\theta)=\sum_{k=1}^n\nabla_\theta\ln p(x_k\mid\theta)}.
$$

课件给出极大似然估计的必要条件：

$$
\boxed{\left.\nabla_\theta\ell(\theta)\right|_{\theta=\hat\theta}=\mathbf0}.
$$

即得到包含 $p$ 个方程的方程组。

**② 概念解释**

就像一维函数在内部极值点可能满足 $f'(x)=0$，多维函数的内部极值点也可能要求所有偏导数同时为零。

**③ 数学严谨性**

梯度为零只是**可微内部极大值的必要条件**，不是充分条件；驻点可能是极小值、鞍点，也可能不存在内部驻点（最大值在边界）。实际题目要检查二阶条件、函数形状和参数取值范围。

### PDF 第 9 页｜高斯例子：只知道协方差，估计均值

**① 原文中文翻译**

设 $x_k\sim N(\mu,\Sigma)$（$k=1,\ldots,n$）。假设协方差矩阵 $\Sigma$ 已知，待估参数仅为 $\theta=\{\mu\}$。

$$
p(x_k\mid\mu)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left[-\frac12(x_k-\mu)^T\Sigma^{-1}(x_k-\mu)\right].
$$

取对数并求梯度：

$$
\boxed{\nabla_\mu\ln p(x_k\mid\mu)=\Sigma^{-1}(x_k-\mu)}.
$$

**② 概念解释**

这里的 $\mu$ 是向量参数。对高斯密度取对数后，指数函数的复杂外壳消失，得到一个关于 $\mu$ 的二次式；求梯度就很容易。

**③ 推导关键**

由于 $\Sigma^{-1}$ 是对称矩阵，

$$
\nabla_\mu\left[-\frac12(x_k-\mu)^T\Sigma^{-1}(x_k-\mu)\right]=\Sigma^{-1}(x_k-\mu).
$$

第一项的归一化常数与 $\mu$ 无关，求导为零。

### PDF 第 10 页｜高斯均值 MLE = 样本均值

**① 原文中文翻译**

对于 $n$ 个独立训练样本：

$$
\nabla_\mu\ell(\mu)=\sum_{k=1}^n\Sigma^{-1}(x_k-\mu).
$$

令梯度为零：

$$
\sum_{k=1}^n\Sigma^{-1}(x_k-\hat\mu)=0.
$$

两边左乘 $\Sigma$，得到：

$$
\sum_{k=1}^n(x_k-\hat\mu)=0.
$$

因此：

$$
\boxed{\hat\mu_{\rm ML}=\frac1n\sum_{k=1}^n x_k}.
$$

课件指出这是“直观的结果”：未知均值的 ML 估计正好是训练样本的算术平均值。

**② 概念解释**

如果长度为 $[10,12,14]$，则 $\hat\mu=(10+12+14)/3=12$。如果样本是多维向量，就对每一维分别求平均。

**③ 考试提示**

写出三步即可：写对数似然 → 对 $\mu$ 求导并置零 → 解出样本均值。理解推导比孤立记公式更稳。

### PDF 第 11 页｜高斯均值与协方差都未知：先考虑一维

**① 原文中文翻译**

当均值 $\mu$ 和协方差 $\Sigma$ 都未知时，参数写作 $\theta=\{\mu,\Sigma\}$。先考虑一维情况：

$$
\theta=\begin{bmatrix}\theta_1\\\theta_2\end{bmatrix}=\begin{bmatrix}\mu\\\sigma^2\end{bmatrix}.
$$

一维高斯密度为：

$$
p(x_k\mid\theta)=\frac{1}{\sqrt{2\pi\theta_2}}\exp\left[-\frac{(x_k-\theta_1)^2}{2\theta_2}\right].
$$

对数密度：

$$
\ln p(x_k\mid\theta)=-\frac12\ln(2\pi\theta_2)-\frac{(x_k-\theta_1)^2}{2\theta_2}.
$$

对参数求导：

$$
\nabla_\theta\ln p(x_k\mid\theta)=\begin{bmatrix}(x_k-\theta_1)/\theta_2\\-1/(2\theta_2)+(x_k-\theta_1)^2/(2\theta_2^2)\end{bmatrix}.
$$

**② 概念解释**

注意 $\theta_2=\sigma^2$，是**方差**而不是标准差。求第二个分量的导数时，$1/\theta_2$ 用的是幂函数求导法。

**③ 校勘说明**

原课件此页的正态密度归一化常数排版容易误读；上面写的是与后续对数似然一致的**标准正确形式** $1/\sqrt{2\pi\sigma^2}$，不要误写成 $1/\sqrt{2\pi\sigma}$。

### PDF 第 12 页｜建立均值与方差的两个驻点方程

**① 原文中文翻译**

对数似然：

$$
\ell(\theta)=\sum_{k=1}^n\ln p(x_k\mid\theta).
$$

令两个偏导数为零：

$$
\boxed{\sum_{k=1}^n\frac{x_k-\hat\theta_1}{\hat\theta_2}=0}
$$

以及：

$$
\boxed{-\frac{n}{2\hat\theta_2}+\sum_{k=1}^n\frac{(x_k-\hat\theta_1)^2}{2\hat\theta_2^2}=0}.
$$

**② 概念解释**

以前只求一个未知参数，所以只需要一个方程。现在同时求均值和方差，就要解两个联立方程。

**③ 推导小技巧**

参数域为 $\hat\theta_2>0$，因此可以乘以 $\hat\theta_2$ 或 $2\hat\theta_2^2$ 消去分母，不改变方程解。由第一个方程先解均值，再代入第二个方程解方差。

### PDF 第 13 页｜一维高斯 MLE 的最终结果

**① 原文中文翻译**

从上一页的方程：

$$
\sum_{k=1}^n(x_k-\hat\theta_1)=0\quad\Rightarrow\quad\hat\theta_1=\frac1n\sum_{k=1}^nx_k.
$$

第二个方程化简为：

$$
-n\hat\theta_2+\sum_{k=1}^n(x_k-\hat\theta_1)^2=0.
$$

所以：

$$
\boxed{\hat\mu=\frac1n\sum_{k=1}^nx_k},\qquad\boxed{\hat\sigma^2_{\rm ML}=\frac1n\sum_{k=1}^n(x_k-\hat\mu)^2}.
$$

**② 概念解释**

第一式是样本均值，第二式是“每个数据点离均值的平方偏差，求平均”。例：$[1,2,3]$ 的均值是 2，MLE 方差为 $(1+0+1)/3=2/3$。

**③ 高频易错点：为什么分母是 $n$ 而非 $n-1$？**

课件求的是**正态模型下的极大似然方差估计**，分母为 $n$。统计学中常见的无偏样本方差使用 $n-1$。两种估计满足的优化目标不同，不能混淆；当样本量很大时差异逐渐减小。

### PDF 第 14 页｜多维高斯的 ML 结果

**① 原文中文翻译**

高斯均值与协方差均未知时，多维极大似然估计为：

$$
\boxed{\hat{\boldsymbol\mu}=\frac1n\sum_{k=1}^n\mathbf x_k}
$$

$$
\boxed{\hat\Sigma=\frac1n\sum_{k=1}^n(\mathbf x_k-\hat{\boldsymbol\mu})(\mathbf x_k-\hat{\boldsymbol\mu})^T}.
$$

课件解释：第一式是 $n$ 个向量的算术平均；第二式是 $n$ 个外积矩阵的算术平均。

**② 概念解释**

如果 $\mathbf x=(x_1,x_2)^T$，则

$$
(\mathbf x-\mu)(\mathbf x-\mu)^T=\begin{bmatrix}(x_1-\mu_1)^2&(x_1-\mu_1)(x_2-\mu_2)\\(x_2-\mu_2)(x_1-\mu_1)&(x_2-\mu_2)^2\end{bmatrix}.
$$

平均后对角线给出各特征方差，非对角线给出协方差。

**③ 现实使用提醒**

当特征维度较高、样本数很少时，$\hat\Sigma$ 可能奇异不可逆。使用高斯判别器时需要考虑协方差正则化、对角假设或降维。

---

## 第二部分：贝叶斯参数估计（PDF 第 15–27 页）

### PDF 第 15 页（课件印刷 16）｜贝叶斯估计的设定

**① 原文中文翻译**

每个类别的似然函数参数形式已知：$p(\mathbf x\mid\omega_j,\theta_j)$。但现在 $\theta_j$ 不再被视为固定的未知数，而被看作**随机变量**。因此不应只得到一个 ML 点估计 $\hat\theta_j$ 再代入分类。课件提出要充分利用训练数据 $\mathcal D^{*}=\mathcal D_1\cup\cdots\cup\mathcal D_c$，计算：

$$
P(\omega_j\mid\mathbf x,\mathcal D^{*}).
$$

**② 概念解释**

MLE 最终通常给出“均值估计为 10.5”；贝叶斯估计还可以回答“均值有多大概率位于 10 到 11 之间”。即保留对未知参数的**不确定性**。

**③ 要注意的区别**

原文“不能再做单一的 ML 估计”应理解为：**为了执行本章的贝叶斯预测，我们不再只依赖 ML 插值式的点估计**；不是数学上禁止再计算 MLE。

### PDF 第 16 页（课件 17）｜含训练数据的贝叶斯分类公式

**① 原文中文翻译**

采用联合概率分解：

$$
P(\omega_j\mid\mathbf x,\mathcal D^{*})=\frac{p(\omega_j,\mathbf x,\mathcal D^{*})}{\sum_i p(\omega_i,\mathbf x,\mathcal D^{*})}.
$$

由链式法则：

$$
p(\omega_j,\mathbf x,\mathcal D^{*})=p(\mathcal D^{*})P(\omega_j\mid\mathcal D^{*})p(\mathbf x\mid\omega_j,\mathcal D^{*}).
$$

所以：

$$
P(\omega_j\mid\mathbf x,\mathcal D^{*})=\frac{P(\omega_j\mid\mathcal D^{*})p(\mathbf x\mid\omega_j,\mathcal D^{*})}{\sum_iP(\omega_i\mid\mathcal D^{*})p(\mathbf x\mid\omega_i,\mathcal D^{*})}.
$$

课件采用两个简化假设：$P(\omega_j\mid\mathcal D^{*})=P(\omega_j)$，以及 $p(\mathbf x\mid\omega_j,\mathcal D^{*})=p(\mathbf x\mid\omega_j,\mathcal D_j)$。

**② 概念解释**

这和第二章贝叶斯公式相似，但是“似然”已经由没有训练信息的 $p(\mathbf x\mid\omega_j)$，变成了**利用训练数据更新后的预测密度** $p(\mathbf x\mid\omega_j,\mathcal D_j)$。

**③ 结果**

$$
\boxed{P(\omega_j\mid\mathbf x,\mathcal D^{*})=\frac{P(\omega_j)p(\mathbf x\mid\omega_j,\mathcal D_j)}{\sum_i P(\omega_i)p(\mathbf x\mid\omega_i,\mathcal D_i)}}.
$$

### PDF 第 17 页（课件 18）｜关键：后验预测密度

**① 原文中文翻译**

对一个类别独立处理，简写 $p(\mathbf x\mid\omega_j,\mathcal D_j)$ 为 $p(\mathbf x\mid\mathcal D)$。因为 $\theta$ 是随机变量，可以对它积分：

$$
p(\mathbf x\mid\mathcal D)=\int p(\mathbf x,\theta\mid\mathcal D)\,d\theta
$$

$$
=\int p(\mathbf x\mid\theta,\mathcal D)p(\theta\mid\mathcal D)\,d\theta
$$

假设给定 $\theta$ 后，新样本 $\mathbf x$ 与训练数据独立：

$$
\boxed{p(\mathbf x\mid\mathcal D)=\int p(\mathbf x\mid\theta)p(\theta\mid\mathcal D)\,d\theta}.
$$

**② 概念解释**

这叫**后验预测分布（Posterior Predictive Distribution）**。核心思想：不只使用最可能的一组参数，而是把所有可能参数的预测结果按照参数后验概率做加权平均。

**③ 和 ML 的对比**

MLE 插入式预测：$p(\mathbf x\mid\hat\theta)$；贝叶斯预测：$\int p(\mathbf x\mid\theta)p(\theta\mid\mathcal D)d\theta$。后者将参数不确定性直接传播到新样本预测中。

### PDF 第 18 页（课件 19）｜贝叶斯估计一般步骤：阶段 I

**① 原文中文翻译**

**阶段 I：参数先验密度 → 参数后验密度**。已知参数化模型 $p(\mathbf x\mid\theta)$、先验 $p(\theta)$ 与训练数据 $\mathcal D$，根据贝叶斯公式：

$$
p(\theta\mid\mathcal D)=\frac{p(\theta)p(\mathcal D\mid\theta)}{\int p(\theta)p(\mathcal D\mid\theta)d\theta}
$$

其中：

$$
p(\mathcal D\mid\theta)=\prod_{k=1}^np(\mathbf x_k\mid\theta).
$$

**② 概念解释**

先验 $p(\theta)$ 表示看到训练样本之前如何看待参数。训练数据通过似然 $p(\mathcal D\mid\theta)$ 调整这些可能性，得到后验 $p(\theta\mid\mathcal D)$。

**③ 公式记忆**

$$
\boxed{\text{参数后验}\propto\text{参数先验}\times\text{训练数据似然}}.
$$

“正比”后还需要归一化，才是积分为 1 的合法后验密度。

### PDF 第 19 页（课件 20）｜贝叶斯估计一般步骤：阶段 II、III

**① 原文中文翻译**

阶段 II：由参数的后验密度得到新样本的类条件预测密度：

$$
\boxed{p(\mathbf x\mid\mathcal D)=\int p(\mathbf x\mid\theta)p(\theta\mid\mathcal D)\,d\theta}.
$$

阶段 III：代入贝叶斯分类公式：

$$
\boxed{P(\omega_j\mid\mathbf x,\mathcal D^{*})=\frac{P(\omega_j)p(\mathbf x\mid\omega_j,\mathcal D_j)}{\sum_iP(\omega_i)p(\mathbf x\mid\omega_i,\mathcal D_i)}}.
$$

**② 概念解释**

三个阶段形成一条清晰链路：

```text
训练数据 D + 参数先验 p(θ)
           ↓ 贝叶斯更新
    参数后验 p(θ | D)
           ↓ 积分边缘化 θ
    新样本预测 p(x | D)
           ↓ 贝叶斯分类
   类别后验 P(ωj | x,D*)
```

**③ 学习提示**

这页适合整体背诵“先估参数分布，再求预测分布，最后分类”的顺序。注意这是教学流程，不意味着所有实际贝叶斯模型都必须显式计算所有积分，复杂模型常需数值近似。

### PDF 第 20 页（课件 21）｜高斯贝叶斯估计：均值未知、方差已知

**① 原文中文翻译**

考虑一维情况，参数 $\theta=\{\mu\}$，观测噪声方差 $\sigma^2$ 已知。似然模型：

$$
x\mid\mu\sim N(\mu,\sigma^2).
$$

假设均值参数本身也服从高斯先验：

$$
\mu\sim N(\mu_0,\sigma_0^2).
$$

问题：观测训练集 $\mathcal D$ 后，$p(\mu\mid\mathcal D)$ 是什么形式？课件说明也可使用其他先验，但此处选高斯先验便于求解。

**② 概念解释**

请严格区分两个方差：$\sigma^2$ 是**单个观测相对于真实均值的波动**；$\sigma_0^2$ 是**我们对未知均值 $\mu$ 的先验不确定性**。二者单位相同、角色不同。

**③ 共轭先验**

这里的高斯先验与高斯均值似然形成**共轭（Conjugate）**关系：更新后的参数后验仍然是高斯分布。

### PDF 第 21 页（课件 22）｜从先验和数据写出参数后验

**① 原文中文翻译**

根据贝叶斯公式：

$$
p(\mu\mid\mathcal D)=\frac{p(\mu)p(\mathcal D\mid\mu)}{\int p(\mu)p(\mathcal D\mid\mu)d\mu}
$$

引入与 $\mu$ 无关的归一化常数 $\alpha$：

$$
p(\mu\mid\mathcal D)=\alpha p(\mu)\prod_{k=1}^{n}p(x_k\mid\mu).
$$

高斯先验与观测似然分别为：

$$
p(\mu)=\frac{1}{\sqrt{2\pi}\sigma_0}\exp\left[-\frac{(\mu-\mu_0)^2}{2\sigma_0^2}\right],
$$

$$
p(x_k\mid\mu)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{(x_k-\mu)^2}{2\sigma^2}\right].
$$

**② 概念解释**

全部训练样本都提供关于 $\mu$ 的证据，因 i.i.d. 假设这些证据的似然可以相乘。由于指数函数相乘时指数相加，所以推导会转化成一个关于 $\mu$ 的二次多项式。

**③ 易错点**

求 $p(\mu\mid\mathcal D)$ 时，$\mu$ 是**变量**，观测值 $x_k$ 是固定数据；不要反过来求导。

### PDF 第 22 页（课件 23）｜为什么参数后验仍然是高斯？

**① 原文中文翻译**

把一维高斯先验和 $n$ 个高斯似然相乘：

$$
p(\mu\mid\mathcal D)=\alpha'\exp\left\lbrace-\frac12\left[\frac{(\mu-\mu_0)^2}{\sigma_0^2}+\sum_{k=1}^n\frac{(\mu-x_k)^2}{\sigma^2}\right]\right\rbrace.
$$

把括号中的平方项展开，关于 $\mu$ 的二次项系数为：

$$
\frac{n}{\sigma^2}+\frac1{\sigma_0^2},
$$

一次项中包含：

$$
\frac1{\sigma^2}\sum_{k=1}^nx_k+\frac{\mu_0}{\sigma_0^2}.
$$

因为指数中仍是负的二次式，所以后验 $p(\mu\mid\mathcal D)$ 仍属于正态分布族。

**② 概念解释**

一维正态密度总可以写成：

$$
p(\mu)\propto\exp\left[-\frac{(\mu-m)^2}{2s^2}\right].
$$

只要通过**配方（Completing the Square）**把后验指数整理成这种形式，就能读出新的均值 $m$ 和方差 $s^2$。

**③ 本页重点**

不必机械背诵长展开式；只要抓住“先验的平方项 + 样本的平方项 → 另一个平方项”的结构。

### PDF 第 23 页（课件 24）｜贝叶斯后验均值与方差公式

**① 原文中文翻译**

令后验：

$$
p(\mu\mid\mathcal D)=N(\mu_n,\sigma_n^2).
$$

比较平方项与一次项系数：

$$
\frac{1}{\sigma_n^2}=\frac{n}{\sigma^2}+\frac{1}{\sigma_0^2},
$$

$$
\frac{\mu_n}{\sigma_n^2}=\frac{\sum_{k=1}^nx_k}{\sigma^2}+\frac{\mu_0}{\sigma_0^2}.
$$

所以：

$$
\boxed{\sigma_n^2=\frac{\sigma^2\sigma_0^2}{n\sigma_0^2+\sigma^2}}
$$

$$
\boxed{\mu_n=\frac{\sigma_n^2}{\sigma^2}\sum_{k=1}^nx_k+\frac{\sigma_n^2}{\sigma_0^2}\mu_0}.
$$

**② 概念解释：它是加权平均**

设样本均值 $\bar x=\tfrac1n\sum x_k$，则：

$$
\boxed{\mu_n=\underbrace{\frac{n\sigma_0^2}{n\sigma_0^2+\sigma^2}}_{\text{样本权重}}\bar x+\underbrace{\frac{\sigma^2}{n\sigma_0^2+\sigma^2}}_{\text{先验权重}}\mu_0}.
$$

两个权重相加为 1。数据多时更信任样本；先验方差很小时更信任先验。

**③ 算例**

若 $\mu_0=10,\sigma_0^2=4,\sigma^2=9,n=9,\bar x=13$，则数据权重 $36/45=0.8$、先验权重 $9/45=0.2$，于是 $\mu_n=0.8\times13+0.2\times10=12.4$；后验方差 $\sigma_n^2=36/45=0.8$。

### PDF 第 24 页（课件 25）｜从参数后验进入预测阶段

**① 原文中文翻译**

课件回顾前一阶段结果：

$$
p(\mu\mid\mathcal D)\sim N(\mu_n,\sigma_n^2)
$$

其中 $\mu_n$ 和 $\sigma_n^2$ 已由训练数据计算。已知：

$$
p(x\mid\mu)\sim N(\mu,\sigma^2).
$$

下一步的任务是求：

$$
p(x\mid\mathcal D).
$$

**② 概念解释**

我们还没有完成“预测新样本”这件事。目前得到的是“参数均值有多大不确定性”。下一步把这种参数不确定性与新样本自身的随机噪声合在一起。

**③ 两种随机性的区别**

新样本 $x$ 会因真实观测过程产生波动，方差为 $\sigma^2$；此外，我们对真实均值 $\mu$ 还不完全确定，方差为 $\sigma_n^2$。因此预测分布应比“直接使用已知均值的单一高斯”更宽。

### PDF 第 25 页（课件 26）｜计算后验预测分布

**① 原文中文翻译**

阶段 II 使用：

$$
p(x\mid\mathcal D)=\int p(x\mid\mu)p(\mu\mid\mathcal D)\,d\mu.
$$

将两个高斯密度相乘并积分，得到：

$$
\boxed{p(x\mid\mathcal D)\sim N(\mu_n,\sigma^2+\sigma_n^2)}.
$$

课件说明，最终预测密度仍然是高斯分布，之后即可自然进入第三阶段分类。

**② 概念解释**

这是非常值得记忆的结论：**预测均值等于参数后验均值；预测方差等于观测噪声方差 + 参数后验方差。**

**③ 不使用积分也能理解**

设 $x=\mu+\varepsilon$，其中 $\mu\mid\mathcal D\sim N(\mu_n,\sigma_n^2)$，$\varepsilon\sim N(0,\sigma^2)$ 且独立。两个独立高斯变量相加仍是高斯：

$$
E[x\mid\mathcal D]=\mu_n,\qquad\mathrm{Var}(x\mid\mathcal D)=\sigma_n^2+\sigma^2.
$$

### PDF 第 26 页（课件 27）｜多维高斯贝叶斯估计

**① 原文中文翻译**

在多维情况下，假设协方差矩阵 $\Sigma$ 已知：

$$
\mathbf x\mid\boldsymbol\mu\sim N(\boldsymbol\mu,\Sigma),\qquad\boldsymbol\mu\sim N(\boldsymbol\mu_0,\Sigma_0).
$$

则后验仍是高斯：

$$
p(\boldsymbol\mu\mid\mathcal D)\sim N(\boldsymbol\mu_n,\Sigma_n),
$$

新样本预测：

$$
\boxed{p(\mathbf x\mid\mathcal D)\sim N(\boldsymbol\mu_n,\Sigma+\Sigma_n)}.
$$

可用精度矩阵（协方差的逆矩阵）表示后验：

$$
\boxed{\Sigma_n=(\Sigma_0^{-1}+n\Sigma^{-1})^{-1}}
$$

$$
\boxed{\boldsymbol\mu_n=\Sigma_n\left(\Sigma_0^{-1}\boldsymbol\mu_0+\Sigma^{-1}\sum_{k=1}^n\mathbf x_k\right)}.
$$

课件另给出使用 $\Sigma_0+(1/n)\Sigma$ 的等价矩阵表达式。

**② 概念解释**

一维用“方差倒数”表达信息权重；多维用“协方差逆矩阵”，也就是精度矩阵来表达权重。直观上，哪个来源越确定，对估计结果贡献就越大。

**③ 易错点**

矩阵乘法一般不能交换次序：$AB\neq BA$。推导时不应把一维的标量加权公式直接照搬到矩阵表达式里而随意调整相乘顺序。

### PDF 第 27 页（课件 33）｜参数估计方法比较及分类误差来源

**① 原文中文翻译**

课件比较 ML 与贝叶斯估计：样本数趋于无穷时两者估计可能逐渐一致；ML 通常更简单、计算复杂度更低；课件认为其可解释性较强；贝叶斯估计更能利用先验知识。分类误差来源包括：

$$
\text{Bayes error} + \text{Model error} + \text{Estimation error}.
$$

**② 概念解释**

**贝叶斯误差（Bayes Error）**：即使知道真实数据分布和最优决策规则，也可能因类别特征重叠而无法避免的错误。**模型误差（Model Error）**：假设的数据分布族不适合真实世界，比如实际分布很偏却硬用一个高斯。**估计误差（Estimation Error）**：模型形式正确，但样本有限导致参数估计不准。

**③ 科学理解**

该页是课堂层面的概括，不能视为所有场景都成立的数学排序。例如可解释性无法用统一的 $>$ 比较；贝叶斯方法与 MLE 在有限数据下也不必有统一优劣。关键是识别**模型假设是否合理、样本是否足够、先验是否可信**。

---
## 第三部分：相关专题 I——隐马尔可夫模型 HMM（PDF 第 28–50 页）

### PDF 第 28 页（课件 34）｜相关主题 I：隐马尔可夫模型

**① 原文中文翻译**

Related Topic I — Hidden Markov Model（相关专题一：隐马尔可夫模型）。

**② 概念解释**

前面的高斯模型处理独立同分布训练样本，而很多真实任务具有时间顺序：例如一段语音的音素序列、机器运行状态、人体动作。我们能够观察某些信号，却不能直接看到产生信号的真实状态。HMM 通过“隐藏状态 + 状态转移 + 可见观测”建立这类时序模型。

**③ 初步例子**

隐藏状态可设为“晴天/雨天”，可见符号为“是否带伞”。我们看到一个人连续几天带不带伞，尝试推断每天的天气；也可以反过来计算某段带伞记录出现的概率。

### PDF 第 29 页（课件 35）｜一阶马尔可夫模型：符号与转移矩阵

**① 原文中文翻译**

状态集合 $\Omega=\{\omega_1,\ldots,\omega_c\}$；长度为 $T$ 的状态序列 $\boldsymbol\omega^T=\{\omega(1),\ldots,\omega(T)\}$，其中 $\omega(t)\in\Omega$。课件示例 $\boldsymbol\omega^6=\{\omega_1,\omega_4,\omega_2,\omega_2,\omega_1,\omega_4\}$。

状态转移概率矩阵：

$$
\boxed{A=[a_{ij}]_{c\times c}},\qquad\boxed{a_{ij}=P(\omega(t+1)=\omega_j\mid\omega(t)=\omega_i)}.
$$

在时间齐次模型中 $a_{ij}$ 与时刻 $t$ 无关，且每一行必须满足：

$$
\sum_{j=1}^ca_{ij}=1.
$$

**② 概念解释**

$a_{ij}$ 表示**当前处在第 $i$ 个状态，下一时刻转入第 $j$ 个状态的概率**。例如 $a_{12}=0.3$ 表示今天晴天时明天下雨的概率为 30%。为什么按行求和等于 1？因为从同一个当前状态出发，下一时刻必须进入某个可能状态。

**③ 一阶马尔可夫性**

“预测下一步，只需知道当前这一步，不需要记住此前整个历史”即：

$$
P(\omega(t+1)\mid\omega(1),\ldots,\omega(t))=P(\omega(t+1)\mid\omega(t)).
$$

这是一种建模假设，不代表现实中所有现象都严格满足。

### PDF 第 30 页（课件 36）｜马尔可夫状态转移图与链式法则

**① 原文中文翻译**

课件用状态转移图展示三种状态及有向转移，例如 $a_{11},a_{12},a_{13},\ldots,a_{33}$。依据概率链式法则：

$$
P(\boldsymbol\omega^T)=P(\omega(1))\prod_{t=2}^{T}P(\omega(t)\mid\omega(1),\ldots,\omega(t-1)).
$$

采用一阶马尔可夫假设：

$$
\boxed{P(\boldsymbol\omega^T)=P(\omega(1))\prod_{t=2}^{T}P(\omega(t)\mid\omega(t-1))}.
$$

**② 概念解释**

计算一整条路径的概率，就是“第一步出现的概率 × 后续每一步的转移概率”。

**③ 小例子**

设 $P(\omega(1)=\text{晴})=0.6$，$P(\text{雨}\mid\text{晴})=0.3$，$P(\text{雨}\mid\text{雨})=0.7$。那么“晴 → 雨 → 雨”的路径概率为：

$$
0.6\times0.3\times0.7=0.126.
$$

### PDF 第 31 页（课件 37）｜HMM 基本假设与三组参数

**① 原文中文翻译**

HMM 的基本假设：每一步的状态不可直接观察；每个不可见状态在该步产生一个可见符号。设可能观察符号集合 $\mathcal V=\{v_1,\ldots,v_K\}$，可见序列为 $\mathbf V^T=\{v(1),\ldots,v(T)\}$。

观测符号概率矩阵：

$$
\boxed{B=[b_{jk}]_{c\times K}},\qquad b_{jk}=P(v_k\mid\omega_j),\qquad\sum_{k=1}^{K}b_{jk}=1.
$$

初始状态概率：

$$
\boxed{\boldsymbol\pi=(\pi_1,\ldots,\pi_c)},\qquad\pi_j=P(\omega(1)=\omega_j).
$$

**② 概念解释**

HMM 有三个核心参数：$A$ 管“隐藏状态之间如何转移”；$B$ 管“一个隐藏状态更容易产生哪个观测”；$\pi$ 管“序列刚开始时处于各隐藏状态的可能性”。常写作：

$$
\boxed{\theta=\{A,B,\pi\}}.
$$

**③ 样例**

两个隐藏天气状态“晴/雨”，两个观测“带伞/不带伞”：$B$ 的一行可能是 $[0.2,0.8]$（晴天时带伞 20%），另一行可能是 $[0.9,0.1]$（雨天时带伞 90%）。注意 $B$ 的行对应状态、列对应观测符号。

### PDF 第 32 页（课件 38）｜HMM 的发射概率与状态图

**① 原文中文翻译**

课件用黑色状态转移边和红色发射边展示 HMM。假设可见符号在时刻 $t$ 的产生概率只依赖当时的隐藏状态：

$$
P(\mathbf V^T\mid\boldsymbol\omega^T)=\prod_{t=1}^{T}P(v(t)\mid\omega(t)).
$$

若 $\omega(t)=\omega_j$ 且 $v(t)=v_k$，则对应概率就是 $b_{jk}$。

**② 概念解释**

HMM 同时使用两个条件独立性假设：**状态的一阶马尔可夫性**以及**观测在给定当前状态后的条件独立性**。

```text
隐藏状态：  ω(1)  ---->  ω(2)  ---->  ω(3)
             |           |           |
             v           v           v
可见观测：  v(1)        v(2)        v(3)
```

**③ 易错点**

不能把观测转移 $P(v(t+1)\mid v(t))$ 直接当成隐藏状态转移 $a_{ij}$；观测之间的相关性是经由隐藏状态间接产生的。

### PDF 第 33 页（课件 39）｜盒子取球：HMM 的直观例子

**① 原文中文翻译**

课件给出三个盒子（Box 1、Box 2、Box 3）以及各盒子中的不同颜色球：**隐藏状态是盒子编号；可见符号是取出的球的颜色**。从第 $i$ 个盒子抽到某种颜色球的概率就是相应观测概率。给定观察到的颜色球序列，HMM 的核心问题有哪些？

**② 概念解释**

你不能看到每次究竟选了哪只盒子，但能看到取出的球是什么颜色。盒子可以按一定转移规律切换。这样，只凭连续颜色观测，就可能反推最可能的盒子切换顺序。

**③ 三种问题预告**

已知盒子切换规则，求这段颜色序列出现概率；已知颜色序列，求最可能的盒子切换路径；只有颜色序列，反过来学习每个盒子颜色组成与切换规则。

### PDF 第 34 页（课件 40）｜HMM 的三个核心问题

**① 原文中文翻译**

完整模型参数：$\theta=\{A,B,\pi\}$；观测符号序列为 $\mathbf V^T$。

1. **Evaluation（评估）**：给定 $\theta$，计算 $P(\mathbf V^T\mid\theta)$。
2. **Decoding（解码）**：给定 $\theta$ 与 $\mathbf V^T$，求最可能的隐藏状态序列 $\boldsymbol\omega^{*}$。
3. **Learning（学习）**：给定 $\mathbf V^T$，寻找使 $P(\mathbf V^T\mid\theta)$ 最大的模型参数 $\theta$。

**② 概念解释**

| 任务 | 已知 | 要求 | 本章对应算法 |
|---|---|---|---|
| 评估 | 模型、观测序列 | 观测出现概率 | 前向／后向算法 |
| 解码 | 模型、观测序列 | 最佳状态路径 | Viterbi |
| 学习 | 观测序列 | 模型参数 | Baum–Welch |

**③ 考试记忆**

评估是“概率是多少”；解码是“状态路径是哪条”；学习是“模型参数是什么”。三题不要混淆。

### PDF 第 35 页（课件 41）｜评估问题：直接枚举所有状态路径

**① 原文中文翻译**

把所有可能隐藏状态序列的贡献相加：

$$
P(\mathbf V^T\mid\theta)=\sum_{\boldsymbol\omega^T}P(\mathbf V^T\mid\boldsymbol\omega^T,\theta)P(\boldsymbol\omega^T\mid\theta).
$$

对于一条固定路径：

$$
P(\boldsymbol\omega^T\mid\theta)=\pi_{\omega(1)}\prod_{t=2}^Ta_{\omega(t-1),\omega(t)},
$$

$$
P(\mathbf V^T\mid\boldsymbol\omega^T,\theta)=\prod_{t=1}^Tb_{\omega(t),v(t)}.
$$

课件指出：对很多隐藏状态序列逐条计算不可行；例如 $c=10,T=20$ 时路径数量达 $10^{20}$。

**② 概念解释**

每个时刻有 $c$ 种隐藏状态，长度为 $T$ 的状态序列总数为 $c^T$。即使每条路径只计算 $O(T)$，总复杂度也约为 $O(Tc^T)$，呈指数增长。

**③ 校勘说明**

原幻灯片把复杂度写得更夸张（可见含 $T!$ 的表达式）；按本页这种逐条枚举 $c^T$ 条路径、每条计算长度 $T$ 的朴素实现，通常应计为 **$O(Tc^T)$**，不需要额外的 $T!$ 因子。结论“不可行”仍正确。

### PDF 第 36 页（课件 42）｜前向算法（Forward Algorithm）

**① 原文中文翻译**

定义前向变量 $\alpha_j(t)$：在第 $t$ 步处于隐藏状态 $\omega_j$，并已生成前 $t$ 个观测符号的联合概率。

$$
\boxed{\alpha_j(t)=P(v(1),\ldots,v(t),\omega(t)=\omega_j\mid\theta)}.
$$

初始化：

$$
\boxed{\alpha_j(1)=\pi_jb_{j,v(1)}}.
$$

递推：

$$
\boxed{\alpha_j(t)=\left[\sum_{i=1}^c\alpha_i(t-1)a_{ij}\right]b_{j,v(t)}}.
$$

最终：

$$
\boxed{P(\mathbf V^T\mid\theta)=\sum_{j=1}^c\alpha_j(T)}.
$$

**② 概念解释**

前向算法的想法是：要在第 $t$ 时刻到达状态 $j$，可能从上一刻任意状态 $i$ 转移过来，因此把所有前一状态的贡献**求和**，再乘当前观测的发射概率。

**③ 三步掌握**

先按初始概率和第一枚符号算 $\alpha(1)$；然后按时间从左往右递推；最后把第 $T$ 时刻所有状态的概率相加。

### PDF 第 37 页（课件 43）｜前向算法的伪代码与复杂度

**① 原文中文翻译**

课件伪代码的含义如下：

```text
输入：A, B, π；观测序列 v(1),...,v(T)
for j = 1,...,c:
    α[j,1] = π[j] * B[j,v(1)]
for t = 2,...,T:
    for j = 1,...,c:
        α[j,t] = B[j,v(t)] * sum_i(α[i,t-1] * A[i,j])
return sum_j α[j,T]
```

复杂度从枚举路径的指数级降为：

$$
\boxed{O(c^2T)}.
$$

**② 概念解释**

一共有约 $T$ 个时刻、每个时刻 $c$ 个当前状态，每个当前状态需要检查 $c$ 个前驱状态，因此有 $T\times c\times c$ 的计算量。

**③ 算法思想**

前向算法是一种**动态规划（Dynamic Programming）**：不同路径若在同一时刻汇聚到同一状态，其公共后续计算可以复用，不必重新枚举所有完整路径。

### PDF 第 38 页（课件 44）｜前向算法网格图（Trellis Diagram）

**① 原文中文翻译**

课件通过网格图展示：每一列对应时刻 $t$，每一行对应可能状态 $\omega_j$，网格点标注前向变量 $\alpha_j(t)$。例如计算第三时刻状态 $\omega_2$：

$$
\boxed{\alpha_2(3)=\left[\sum_{i=1}^c\alpha_i(2)a_{i2}\right]b_{2,v(3)}}.
$$

**② 概念解释**

这张图的箭头说明当前状态可以从上一时刻所有候选状态过来。所有通向 $\omega_2$ 的路径先累加，之后再乘 $\omega_2$ 产生第三个可见符号的概率。

**③ 避免一个常见混淆**

网格图显示的是**动态规划的计算连接**，不是说全部路径会真实发生。它只是高效累积概率的一种数据结构或计算视图。

### PDF 第 39 页（课件 45）｜HMM 数值计算实例：模型参数

**① 原文中文翻译**

课件实例有 $c=4$ 个状态、$K=5$ 种观测符号。转移矩阵为：

$$
A=\begin{bmatrix}1&0&0&0\\0.2&0.3&0.1&0.4\\0.2&0.5&0.2&0.1\\0.8&0.1&0&0.1\end{bmatrix}.
$$

观测矩阵：

$$
B=\begin{bmatrix}1&0&0&0&0\\0&0.3&0.4&0.1&0.2\\0&0.1&0.1&0.7&0.1\\0&0.5&0.2&0.1&0.2\end{bmatrix}.
$$

初始分布：

$$
\pi=(0,1,0,0).
$$

特殊性质：$\omega_1$ 是**吸收状态**（进入后不会跳出）；只有状态 $\omega_1$ 会发射符号 $v_1$；初始状态确定是 $\omega_2$。

**② 概念解释**

$A$ 的第一行 $[1,0,0,0]$ 表示处在状态 1 后一定留在状态 1。$B$ 的第一行 $[1,0,0,0,0]$ 表示状态 1 一定产生 $v_1$。$\pi_2=1$ 表示第一时刻必定从状态 2 开始。

**③ 计算准备**

本例观测序列为 $\mathbf V^5=(v_4,v_2,v_4,v_3,v_1)$。计算时可利用零概率大量简化求和。

### PDF 第 40 页（课件 46）｜前向算法实例：算出观测序列概率

**① 原文中文翻译**

给定：

$$
\mathbf V^5=\{v_4,v_2,v_4,v_3,v_1\}.
$$

课件在网格图里给出了不同状态和时刻的 $\alpha_j(t)$，最后只有状态 $\omega_1$ 的非零概率，因此：

$$
\boxed{P(\mathbf V^5\mid\theta)=\sum_{j=1}^4\alpha_j(5)\approx0.00011}.
$$

**② 逐步解释**

第一步观测 $v_4$、初始状态 $\omega_2$：

$$
\alpha_2(1)=\pi_2b_{2,4}=1\times0.1=0.1.
$$

其他初始状态前向概率都为零。第二步观测 $v_2$：

$$
\alpha_2(2)=0.1\times a_{22}\times b_{2,2}=0.1\times0.3\times0.3=0.009.
$$

同理可得到其他状态。最后观测 $v_1$ 只能从状态 1 发射，所以末列只有 $\alpha_1(5)$ 非零。

**③ 注意**

课件在网格图中的中间结果进行了小数截断，因此直接使用这些显示值反算最后一位会有微小舍入差别；实际编程应使用未提前四舍五入的浮点数。

### PDF 第 41 页（课件 47）｜后向算法（Backward Algorithm）

**① 原文中文翻译**

定义后向变量 $\beta_j(t)$：已知第 $t$ 时刻隐藏状态为 $\omega_j$，从 $t+1$ 到 $T$ 时刻的剩余观测序列出现的条件概率。

$$
\boxed{\beta_j(t)=P(v(t+1),\ldots,v(T)\mid\omega(t)=\omega_j,\theta)}.
$$

终止条件：

$$
\boxed{\beta_j(T)=1}.
$$

向后递推：

$$
\boxed{\beta_j(t)=\sum_{i=1}^ca_{ji}b_{i,v(t+1)}\beta_i(t+1)}.
$$

最后：

$$
\boxed{P(\mathbf V^T\mid\theta)=\sum_{j=1}^c\pi_jb_{j,v(1)}\beta_j(1)}.
$$

**② 概念解释**

前向算法问“到当前为止，怎样累积前面的观测概率”；后向算法问“如果当前处于这个状态，后面剩下的观测会以多大概率出现”。

**③ 为什么终止值等于 1？**

到了最后一个时刻，未来已没有剩余观测需要生成；生成空序列的概率按约定为 1。

### PDF 第 42 页（课件 48）｜后向算法伪代码与复杂度

**① 原文中文翻译**

```text
输入：A, B, π；观测序列 v(1),...,v(T)
for j = 1,...,c:
    β[j,T] = 1
for t = T-1,...,1:
    for j = 1,...,c:
        β[j,t] = sum_i(A[j,i] * B[i,v(t+1)] * β[i,t+1])
return sum_j π[j] * B[j,v(1)] * β[j,1]
```

复杂度同样为：

$$
\boxed{O(c^2T)}.
$$

**② 概念解释**

前向：初始化第 1 列，然后从左往右算。后向：初始化最后一列，然后从右往左算。两种方法都能得到同一个观测序列似然。

**③ 重要延伸**

把前向 $\alpha$ 与后向 $\beta$ 结合起来，可以计算“已经看到完整观测后，某个时刻最可能处于哪个状态”，这在 Baum–Welch 参数学习里非常关键。

### PDF 第 43 页（课件 49）｜HMM 解码问题

**① 原文中文翻译**

给定模型参数与观测序列，寻找最有可能的隐藏状态序列：

$$
\boxed{\boldsymbol\omega^{*}=\arg\max_{\boldsymbol\omega^T}P(\boldsymbol\omega^T\mid\mathbf V^T,\theta)}.
$$

因为 $P(\mathbf V^T\mid\theta)$ 与候选状态路径无关，所以等价于：

$$
\boxed{\boldsymbol\omega^{*}=\arg\max_{\boldsymbol\omega^T}P(\boldsymbol\omega^T,\mathbf V^T\mid\theta)}.
$$

朴素方法枚举全部路径，计算量随 $c^T$ 指数增长，不可行。

**② 概念解释**

评估问题对**所有路径求和**，解码问题要从所有路径中**找概率最大的一条**。这是两道不同的题目。

**③ 示例**

如果观察是“带伞、带伞、不带伞”，我们想知道最可能的天气轨迹是否是“雨、雨、晴”；这个“最可能完整轨迹”的问题就是解码。

### PDF 第 44 页（课件 50）｜Viterbi 算法：动态规划状态定义

**① 原文中文翻译**

课件介绍维特比算法（Viterbi Algorithm），定义：

$$
\boxed{\delta_j(t)=\max_{\omega(1),\ldots,\omega(t-1)}P(\omega(1),\ldots,\omega(t-1),\omega(t)=\omega_j,v(1),\ldots,v(t)\mid\theta)}.
$$

$\delta_j(t)$ 表示**截至第 $t$ 步、最后处在状态 $\omega_j$ 的最佳路径及观测联合概率**。与前向／后向算法类似，$\delta$ 可以利用动态规划递推。课件附有 Andrew J. Viterbi 的照片和介绍。

**② 概念解释**

关键区别是：$\alpha_j(t)$ 把所有到达 $j$ 的路径概率**加起来**，而 $\delta_j(t)$ 只保留到达 $j$ 的**最大的一条**路径概率。

**③ 简单对比**

如果两条路径到状态 $j$ 的分数分别为 0.03 与 0.04：前向算法合计为 0.07；Viterbi 保留 0.04，并记录它来自哪个前驱状态。

### PDF 第 45 页（课件 51）｜Viterbi 递推公式

**① 原文中文翻译**

初始化：

$$
\boxed{\delta_j(1)=\pi_jb_{j,v(1)}}.
$$

递推：

$$
\boxed{\delta_j(t)=\left[\max_{1\le i\le c}\delta_i(t-1)a_{ij}\right]b_{j,v(t)}}.
$$

**② 逐步理解**

若想在第 $t$ 步以状态 $j$ 结束，就考虑前一时刻所有候选状态 $i$，计算：前一时刻最佳路径概率 $\delta_i(t-1)$ × 从 $i$ 转移到 $j$ 的概率 $a_{ij}$；挑出最大值，再乘当前状态发射当前观测的概率。

**③ 与前向算法只有一个操作不同**

前向：

$$
\alpha_j(t)=\underbrace{\left[\sum_i\alpha_i(t-1)a_{ij}\right]}_{\text{求和}}b_{j,v(t)}.
$$

Viterbi：

$$
\delta_j(t)=\underbrace{\left[\max_i\delta_i(t-1)a_{ij}\right]}_{\text{取最大}}b_{j,v(t)}.
$$

这个对比是 HMM 最常考、也最容易弄清楚的知识点。

### PDF 第 46 页（课件 52）｜Viterbi 伪代码与路径回溯

**① 原文中文翻译**

除最佳分数 $\delta_j(t)$ 外，还需要保存最佳前驱状态：

$$
\boxed{\psi_j(t)=\arg\max_i\delta_i(t-1)a_{ij}}.
$$

```text
初始化：δ[j,1] = π[j] * B[j,v(1)]
for t = 2,...,T:
    for j = 1,...,c:
        δ[j,t] = max_i(δ[i,t-1] * A[i,j]) * B[j,v(t)]
        ψ[j,t] = argmax_i(δ[i,t-1] * A[i,j])
终点：ω*(T) = argmax_j δ[j,T]
回溯：for t = T-1,...,1:
    ω*(t) = ψ[ω*(t+1), t+1]
```

计算复杂度：

$$
\boxed{O(c^2T)}.
$$

**② 概念解释**

只知道最后的最大概率并不等于知道整条路径。$\psi_j(t)$ 相当于记录“我是从哪个状态过来的”，最后从终点逐步逆向找回所有状态，这叫**路径回溯（Backtracking）**。

**③ 为什么有效？**

到达同一时刻、同一状态的路径中，分数较小的路径不可能在共享相同未来延伸时超越分数较大的路径。因此可以安全丢弃较差的前缀，仅保留最佳前缀。

### PDF 第 47 页（课件 53）｜HMM 的学习问题和 Baum–Welch

**① 原文中文翻译**

给定观测序列 $\mathbf V^T$，寻找使观测概率最大的模型参数：

$$
\boxed{\hat\theta=\arg\max_\theta P(\mathbf V^T\mid\theta)}.
$$

课件指出一般不能保证求得全局最优解，因此利用迭代更新寻找局部最优，使每轮更新后的似然不下降。介绍 **Baum–Welch 算法**，又称 Forward–Backward 训练算法，是著名 **期望最大化（Expectation-Maximization, EM）** 过程的一种实现。

**② 概念解释**

为什么训练困难？因为我们只看到颜色球序列，却没有真实的盒子路径。隐藏路径属于**潜变量（Latent Variables）**，无法像已标注状态那样直接数转移次数。

**③ EM 思想**

E 步：在当前模型下，计算隐藏状态与状态转移可能性的条件期望。M 步：把这些“软计数”用来重新估计 $A,B,\pi$。重复直到收敛。

### PDF 第 48 页（课件 54）｜Baum–Welch 的核心变量：两时刻联合后验

**① 原文中文翻译**

定义：

$$
\boxed{\gamma_{ij}(t)=P(\omega(t)=\omega_i,\omega(t+1)=\omega_j\mid\mathbf V^T,\theta)}.
$$

利用前向、后向变量：

$$
\boxed{\gamma_{ij}(t)=\frac{\alpha_i(t)a_{ij}b_{j,v(t+1)}\beta_j(t+1)}{P(\mathbf V^T\mid\theta)}}.
$$

课件也写为全部 $i,j$ 组合的归一化形式：

$$
\gamma_{ij}(t)=\frac{\alpha_i(t)a_{ij}b_{j,v(t+1)}\beta_j(t+1)}{\sum_{r=1}^c\sum_{s=1}^c\alpha_r(t)a_{rs}b_{s,v(t+1)}\beta_s(t+1)}.
$$

**② 概念解释**

$\gamma_{ij}(t)$ 是“观察了**全部**符号之后，我们认为时刻 $t$ 从状态 $i$ 转移到状态 $j$ 的概率”。

**③ 为什么四项相乘？**

$\alpha_i(t)$ 负责前面的观测；$a_{ij}$ 负责这次状态转移；$b_{j,v(t+1)}$ 负责下一时刻的符号；$\beta_j(t+1)$ 负责剩余观测。整体再除以 $P(\mathbf V^T\mid\theta)$，完成条件概率归一化。

### PDF 第 49 页（课件 55）｜Baum–Welch 参数更新伪代码

**① 原文中文翻译**

课件给出的流程：随机初始化 $\theta=\{A,B,\pi\}$；重复利用前向算法求 $\alpha$、后向算法求 $\beta$；由此计算 $\gamma_{ij}(t)$；再更新初始概率 $\pi$、转移矩阵 $A$ 和发射矩阵 $B$；直到参数变化小于指定阈值等收敛条件。

转移矩阵的更新形式：

$$
\boxed{\hat a_{ij}=\frac{\sum_{t=1}^{T-1}\gamma_{ij}(t)}{\sum_{t=1}^{T-1}\sum_{r=1}^{c}\gamma_{ir}(t)}}.
$$

初始概率更新：

$$
\boxed{\hat\pi_i=\sum_{j=1}^c\gamma_{ij}(1)}.
$$

**② 概念解释**

分子是“预期从状态 $i$ 跳到状态 $j$ 的次数”，分母是“预期从状态 $i$ 向任何状态发生转移的次数”。它们的比值自然可以当作转移概率的新估计。

**③ 发射矩阵的标准实现与校勘**

定义单时刻状态后验：

$$
\eta_i(t)=P(\omega(t)=\omega_i\mid\mathbf V^T,\theta)=\frac{\alpha_i(t)\beta_i(t)}{P(\mathbf V^T\mid\theta)}.
$$

则对**离散 HMM**，常用的发射概率更新为：

$$
\boxed{\hat b_{ik}=\frac{\sum_{t=1}^{T}\mathbf1\{v(t)=v_k\}\eta_i(t)}{\sum_{t=1}^{T}\eta_i(t)}}.
$$

其中 $\mathbf1\{\cdot\}$ 是指示函数，条件成立取 1，否则取 0。**课件原图第 49 页的发射概率更新式中，$\gamma_{ij}(t)$ 的索引与观测时刻看起来不完全一致**；实际计算时应使用上面的单时刻占有概率表达式，避免把发射状态和状态转移的时刻搞反。以上是教学补充，不是原稿逐字转写。

### PDF 第 50 页（课件 56）｜HMM 延伸阅读

**① 原文中文翻译**

“如果要完整了解 HMM……” 课件推荐经典文献：

L. R. Rabiner. *A tutorial on hidden Markov models and selected applications in speech recognition*. **Proceedings of the IEEE**, 1989, 77(2): 257–286.

**② 概念解释**

这篇经典教程系统介绍 HMM 的建模、三大核心问题、前向／后向、Viterbi、参数估计和语音识别应用。

**③ 建议阅读顺序**

先把本章三个问题和四组公式分清楚，再看原论文的算法流程图，会比直接一页页读英文经典文献容易很多。尤其要明确：前向/后向在做**边缘化求和**，Viterbi 在做**最佳路径最大化**，Baum–Welch 在做**带隐变量的参数迭代估计**。

---
## 第四部分：相关专题 II——贝叶斯信念网络（PDF 第 51–59 页）

### PDF 第 51 页（课件 57）｜相关专题 II：贝叶斯信念网络

**① 原文中文翻译**

Related Topic II — Bayesian Belief Network（相关专题二：贝叶斯信念网络／贝叶斯置信网络）。

**② 概念解释**

当存在多个相关变量时，完整联合概率表可能非常庞大。贝叶斯网络用一个**有向无环图（DAG）**表示变量之间的条件依赖结构，并用一组局部条件概率表描述联合分布。

**③ 与 HMM 的联系**

HMM 可以看作一种具有特殊链式结构的概率图模型。贝叶斯网络允许更一般的图结构，例如“季节与地区同时影响鱼种，鱼种再影响亮度和厚度”。

### PDF 第 52 页（课件 58）｜有向无环图 DAG

**① 原文中文翻译**

有向无环图（Directed Acyclic Graph，DAG）记为：

$$
G=(V,E).
$$

$V$ 是图中的节点集合；$E$ 是图中的有向边集合；基本条件是图中不存在**有向回路**。课件示例包含 7 个节点：

$$
V=\{A,B,C,D,E,F,G\},\qquad |V|=7.
$$

边集合为：

$$
E=\{(A,C),(B,D),(D,C),(C,E),(E,F),(E,G),(F,G)\},\quad |E|=7.
$$

**② 概念解释**

有向边 $(A,C)$ 表示从 $A$ 指向 $C$。**有向无环**的意思是：沿箭头一直走，不可能最终回到出发点；这不禁止两节点之间存在多条不同的有向路径。

**③ 原图文字化示意**

```text
A ──→ C ←── D ←── B
      │
      ▼
      E ─────→ G
      │        ▲
      └──→ F ─┘
```

例如 $C$ 的父节点是 $A,D$；$G$ 的父节点是 $E,F$。没有父节点的根节点为 $A,B$。

### PDF 第 53 页（课件 59）｜贝叶斯网络的目标：联合分布分解

**① 原文中文翻译**

贝叶斯信念网络的目标：利用网络的 DAG 结构，为一组随机变量建模联合概率分布。设随机变量为 $X_1,\ldots,X_m$，第 $i$ 个节点的父节点集合记为 $\mathrm{Pa}(X_i)$。联合概率分布可分解为每个变量在其父变量条件下的概率之积：

$$
\boxed{P(x_1,\ldots,x_m)=\prod_{i=1}^{m}P(x_i\mid\mathrm{Pa}(X_i))}.
$$

**② 概念解释**

一般联合概率的链式法则可能需要越来越多的历史条件，而 DAG 的局部马尔可夫性质让我们只需考虑直接父节点。这可以减少模型参数数量，并使复杂概率计算更有结构。

**③ 简单例子**

若 $A\to B\to C$：

$$
P(a,b,c)=P(a)P(b\mid a)P(c\mid b).
$$

它不代表 $A$ 与 $C$ 独立，而是描述**给定 $B$ 时，$C$ 与 $A$ 的条件独立性**。

### PDF 第 54 页（课件 60）｜DAG 例 I：链式网络

**① 原文中文翻译**

课件的网络为：

```text
A ──→ B ──→ C ──→ D
```

联合分布分解为：

$$
\boxed{P(a,b,c,d)=P(a)P(b\mid a)P(c\mid b)P(d\mid c)}.
$$

若只求末端变量 $D$ 的边缘概率，则对其他变量求和：

$$
P(d)=\sum_{a,b,c}P(a)P(b\mid a)P(c\mid b)P(d\mid c).
$$

**② 概念解释**

对 $a,b,c$ 求和意味着“我不关心它们究竟分别取什么值，只关心最后 $D=d$ 的总概率”。逐级求和可以复用中间结果，例如先算 $P(b)$、再算 $P(c)$、最后算 $P(d)$。

**③ 逐步求和**

$$
P(b)=\sum_aP(a)P(b\mid a),
$$

$$
P(c)=\sum_bP(b)P(c\mid b),
$$

$$
P(d)=\sum_cP(c)P(d\mid c).
$$

这也是概率图模型里**消息传递**思想的简单形式。

### PDF 第 55 页（课件 61）｜DAG 例 II：分叉后汇合的网络

**① 原文中文翻译**

课件图为：

```text
      E
     / \
    v   v
    F   G
     \ /
      v
      H
```

即 $E\to F$、$E\to G$、$F\to H$、$G\to H$。联合分布：

$$
\boxed{P(e,f,g,h)=P(e)P(f\mid e)P(g\mid e)P(h\mid f,g)}.
$$

对 $E$ 边缘化：

$$
P(f,g,h)=P(h\mid f,g)\sum_eP(e)P(f\mid e)P(g\mid e).
$$

求 $H=h$ 的边缘概率：

$$
P(h)=\sum_{e,f,g}P(e)P(f\mid e)P(g\mid e)P(h\mid f,g).
$$

**② 概念解释**

$F,G$ 共享父节点 $E$，因此即使 $F,G$ 间没有直接箭头，也可能统计相关；但**给定 $E$ 之后**，在本图结构下 $F$ 与 $G$ 条件独立。反过来，$H$ 同时依赖两个父节点，其条件概率表必须考虑 $F,G$ 的组合。

**③ 理解重点**

“没有箭头”不等于“无条件独立”。贝叶斯网络表达的是由整张图决定的**条件独立关系**。

### PDF 第 56 页（课件 62）｜DAG 例 III：鱼分类贝叶斯网络

**① 原文中文翻译**

课件构建了鱼分类的贝叶斯网络：

```text
 A：季节 (season)          B：海域 (locale)
        \                  /
         \                /
             X：鱼种
            /       \
 C：亮度 (lightness)  D：厚薄 (thickness)
```

变量取值：

- $A$: $a_1$ 冬、$a_2$ 春、$a_3$ 夏、$a_4$ 秋；$P(a_i)=0.25$。
- $B$: $b_1$ 北大西洋、$b_2$ 南大西洋；$P(b_1)=0.6,P(b_2)=0.4$。
- $X$: $x_1$ 鲑鱼、$x_2$ 鲈鱼。
- $C$: $c_1$ 亮、$c_2$ 中、$c_3$ 暗。
- $D$: $d_1$ 宽／厚、$d_2$ 窄／薄。

联合分布分解：

$$
\boxed{P(a,b,x,c,d)=P(a)P(b)P(x\mid a,b)P(c\mid x)P(d\mid x)}.
$$

课件通过**条件概率表（Conditional Probability Tables, CPT）**给出每个节点的局部概率，例如：

| 条件 | $P(x_1\mid a,b)$（鲑鱼） | $P(x_2\mid a,b)$（鲈鱼） |
|---|---:|---:|
| $a_1,b_1$ | 0.5 | 0.5 |
| $a_1,b_2$ | 0.7 | 0.3 |
| $a_2,b_1$ | 0.6 | 0.4 |
| $a_2,b_2$ | 0.8 | 0.2 |
| $a_3,b_1$ | 0.4 | 0.6 |
| $a_3,b_2$ | 0.1 | 0.9 |
| $a_4,b_1$ | 0.2 | 0.8 |
| $a_4,b_2$ | 0.3 | 0.7 |

| 鱼种 | $P(c_1\mid x)$ 亮 | $P(c_2\mid x)$ 中 | $P(c_3\mid x)$ 暗 |
|---|---:|---:|---:|
| $x_1$ 鲑鱼 | 0.6 | 0.2 | 0.1 |
| $x_2$ 鲈鱼 | 0.2 | 0.3 | 0.5 |

| 鱼种 | $P(d_1\mid x)$ 宽 | $P(d_2\mid x)$ 薄 |
|---|---:|---:|
| $x_1$ 鲑鱼 | 0.3 | 0.7 |
| $x_2$ 鲈鱼 | 0.6 | 0.4 |

**② 概念解释**

季节和海域首先影响鱼种，鱼种再影响亮度和厚薄。只要这些概率表已知，就能计算任意完全指定事件的联合概率，并可以在观测部分变量后反过来判断鱼种。

**③ 课件表格中的重要数据问题**

原图中 **$P(c_1\mid x_1)=0.6, P(c_2\mid x_1)=0.2, P(c_3\mid x_1)=0.1$**，其和为 $0.9$，不满足条件概率表一行应和为 1 的标准要求。这里**忠实抄录了课件中的 0.1**，没有擅自替换。后续例题只使用 $c_1$ 的概率，不受这一处表格行总和问题影响，但若要将整套数值作为合法概率模型运行，应先核实原课件这一格是否为笔误。

### PDF 第 57 页（课件 63）｜完整事件联合概率的计算

**① 原文中文翻译**

课件提问：一条鱼在**夏季的北大西洋**被捕获，属于**鲈鱼**，并且**颜色暗、身体薄**，这样的联合事件发生概率是多少？事件写作：

$$
a=a_3,\quad b=b_1,\quad x=x_2,\quad c=c_3,\quad d=d_2.
$$

根据贝叶斯网络的分解：

$$
P(a_3,b_1,x_2,c_3,d_2)=P(a_3)P(b_1)P(x_2\mid a_3,b_1)P(c_3\mid x_2)P(d_2\mid x_2).
$$

代入原图数据：

$$
=0.25\times0.6\times0.6\times0.5\times0.4=\boxed{0.018}.
$$

**② 概念解释**

这是**联合事件概率**：每个变量的取值都指定了，所以用对应 CPT 的概率乘起来。这里的结果 0.018 等于 1.8%。

**③ 易错点**

它不是“已知暗且薄，鱼为鲈鱼的后验概率”，因为题目同时指定了季节、海域、鱼种、亮度和厚薄全部取值。若只给其中几个观测，需要对未观测变量求和并归一化。

### PDF 第 58 页（课件 64）｜有证据时如何分类？

**① 原文中文翻译**

问题：已知某条鱼**颜色亮**，而且在**南大西洋**捕获，该如何判断鱼种？已观察到的证据为：

$$
b=b_2,\qquad c=c_1.
$$

需要比较：

$$
P(X=x_1\mid b_2,c_1)\quad\text{和}\quad P(X=x_2\mid b_2,c_1).
$$

**② 概念解释**

我们目前并不知道季节 $A$，也没有观察身体厚薄 $D$。这些都是未观测变量，必须对其可能取值求和（边缘化）；不能随便指定一个取值，更不能把它们当作零概率。

**③ 思考步骤**

第一，列出目标变量 $X$ 和观测证据 $B,C$；第二，写出全联合分解；第三，对未观察变量 $A,D$ 求和；第四，对 $x_1,x_2$ 的结果归一化。

### PDF 第 59 页（课件 65）｜贝叶斯网络分类：完整计算

**① 原文中文翻译**

求：

$$
P(x_1\mid b_2,c_1)=\alpha\sum_{a,d}P(a,x_1,b_2,c_1,d),
$$

其中 $\alpha$ 表示最终归一化常数（与鱼种候选值无关）。按 DAG 分解：

$$
=\alpha\sum_{a,d}P(a)P(b_2)P(x_1\mid a,b_2)P(c_1\mid x_1)P(d\mid x_1).
$$

利用 $\sum_dP(d\mid x_1)=1$：

$$
=\alpha P(b_2)P(c_1\mid x_1)\sum_aP(a)P(x_1\mid a,b_2).
$$

课件数值为：

$$
=\alpha(0.4)(0.6)[(0.25)(0.7)+(0.25)(0.8)+(0.25)(0.1)+(0.25)(0.3)]
$$

$$
=\boxed{\alpha\times0.114}.
$$

类似地，另一种鱼：

$$
P(x_2\mid b_2,c_1)=\boxed{\alpha\times0.042}.
$$

归一化：

$$
\boxed{P(x_1\mid b_2,c_1)=\frac{0.114}{0.114+0.042}\approx0.731}
$$

$$
\boxed{P(x_2\mid b_2,c_1)=\frac{0.042}{0.114+0.042}\approx0.269}.
$$

所以选择 $x_1$：**鲑鱼（salmon）**。

**② 概念解释**

你可能会疑惑：明明没有季节信息，为什么计算中仍然包含 $a_1,\ldots,a_4$？因为不知道季节，并不意味着它对鱼种没有影响。必须按各季节的概率加权，合并所有可能情况。

**③ 两个重要数学技巧**

1. **边缘化**：对未知变量求和，例 $\sum_dP(d\mid x_1)=1$，可以把无关的厚薄变量消去。
2. **归一化**：先计算两个未归一化分数 $0.114$ 和 $0.042$，再除以它们的总和，得到和为 1 的后验概率。

---

## 第五部分：课件总结（PDF 第 60–63 页）

### PDF 第 60 页（课件 66）｜总结：模式识别的参数估计问题

**① 原文中文翻译**

模式识别的重要问题是：从训练集中估计先验概率和类条件密度。训练数据的基本假设是 i.i.d.。对于估计类条件密度有两种路径：如果使用参数化分布，则采用最大似然估计或贝叶斯估计；如果不规定参数形式，则使用非参数估计方法。

**② 概念解释**

本章的主线并不是增加一种全新分类器，而是回答“**概率模型的参数如何通过数据得到**”。分布一旦得到，就可以继续使用第二章的贝叶斯决策规则。

**③ 关联总结**

```text
训练样本
   ├─ 统计类别比例 → 先验 P(ω)
   └─ 建立类条件密度 p(x|ω)
          ├─ 参数法 → MLE / 贝叶斯估计（第 3 章）
          └─ 非参数法 → 后续章节
                       ↓
                   贝叶斯分类
```

### PDF 第 61 页（课件 67）｜总结：极大似然估计

**① 原文中文翻译**

极大似然估计把参数当作固定但未知的数值。目标函数是对数似然函数；在常规可微内部极大值情形下，必要条件是目标函数关于参数的梯度为零向量。高斯分布例子包括只未知 $\mu$，以及 $\mu$ 和 $\Sigma$ 都未知的情况。

**② 必背公式**

$$
\hat\theta=\arg\max_\theta\sum_k\ln p(x_k\mid\theta),
$$

$$
\hat\mu=\frac1n\sum_kx_k,
$$

$$
\hat\Sigma_{\rm ML}=\frac1n\sum_k(x_k-\hat\mu)(x_k-\hat\mu)^T.
$$

**③ 易错点**

记住 $\hat\Sigma_{\rm ML}$ 分母为 $n$；“梯度为零”并非对所有最优化问题都充分成立，还需核验最大值性质和参数边界。

### PDF 第 62 页（课件 68）｜总结：贝叶斯参数估计

**① 原文中文翻译**

贝叶斯估计将参数视为随机变量。一般流程分三阶段：

- 阶段 I：参数先验密度 $\to$ 参数后验密度；
- 阶段 II：参数后验密度 $\to$ 观测数据的类条件预测密度；
- 阶段 III：利用更新的预测密度进行贝叶斯分类。

高斯例子：未知均值 $\mu$ 的一维和多维情形。

**② 必背公式**

$$
p(\theta\mid\mathcal D)\propto p(\theta)\prod_kp(x_k\mid\theta),
$$

$$
p(x\mid\mathcal D)=\int p(x\mid\theta)p(\theta\mid\mathcal D)d\theta.
$$

**③ 记忆**

MLE：单点 $\hat\theta$；贝叶斯：参数后验 $p(\theta\mid\mathcal D)$；贝叶斯预测：把参数后验积分掉。

### PDF 第 63 页（课件 69）｜总结：HMM 与贝叶斯信念网络

**① 原文中文翻译**

HMM 的参数为 $\theta=\{A,B,\pi\}$，数据为可见符号序列 $\mathbf V^T$。HMM 主要有三个问题：

- 评估：计算 $P(\mathbf V^T\mid\theta)$，使用前向／后向算法；
- 解码：寻找最优隐藏路径，使用 Viterbi；
- 学习：通过 Baum–Welch 更新参数。

贝叶斯信念网络：使用 DAG 结构建模联合分布，通过条件概率表给出各节点在其父节点条件下的分布。

**② 概念解释**

这一页把本章两个相关专题汇总在一起：HMM 用链式隐藏状态表示时间过程，贝叶斯网络用更一般的 DAG 结构表示变量依赖。二者都依赖概率模型，并通过高效的计算策略处理隐变量。

**③ 本章最终结构**

```text
概率模型的学习与推断
├── 参数估计
│   ├── 极大似然估计 MLE
│   └── 贝叶斯参数估计
├── 隐马尔可夫模型 HMM
│   ├── 前向／后向：评估
│   ├── Viterbi：最优路径
│   └── Baum–Welch：学习
└── 贝叶斯信念网络 BBN
    ├── DAG 与条件独立
    ├── 联合概率分解
    └── 证据推理与边缘化
```

---

## 全章复习手册：从零基础到能做题

### 一、最重要的八个公式

**1. 极大似然目标**

$$
\boxed{\hat\theta=\arg\max_\theta\prod_{k=1}^np(x_k\mid\theta)=\arg\max_\theta\sum_{k=1}^n\ln p(x_k\mid\theta)}.
$$

**2. 一维高斯的 ML 均值与方差**

$$
\boxed{\hat\mu=\frac1n\sum_kx_k,\qquad\hat\sigma^2_{\rm ML}=\frac1n\sum_k(x_k-\hat\mu)^2}.
$$

**3. 贝叶斯参数后验**

$$
\boxed{p(\theta\mid\mathcal D)=\frac{p(\theta)p(\mathcal D\mid\theta)}{\int p(\theta)p(\mathcal D\mid\theta)d\theta}}.
$$

**4. 贝叶斯后验预测**

$$
\boxed{p(x\mid\mathcal D)=\int p(x\mid\theta)p(\theta\mid\mathcal D)d\theta}.
$$

**5. 高斯均值后验与预测**

$$
\boxed{\sigma_n^2=\left(\frac1{\sigma_0^2}+\frac n{\sigma^2}\right)^{-1},\qquad\mu_n=\sigma_n^2\left(\frac{\mu_0}{\sigma_0^2}+\frac{\sum_kx_k}{\sigma^2}\right)}.
$$

$$
\boxed{x\mid\mathcal D\sim N(\mu_n,\sigma^2+\sigma_n^2)}.
$$

**6. HMM 前向与后向**

$$
\boxed{\alpha_j(t)=\left[\sum_i\alpha_i(t-1)a_{ij}\right]b_{j,v(t)}}.
$$

$$
\boxed{\beta_i(t)=\sum_j a_{ij}b_{j,v(t+1)}\beta_j(t+1)}.
$$

**7. HMM Viterbi 与 Baum–Welch**

$$
\boxed{\delta_j(t)=\left[\max_i\delta_i(t-1)a_{ij}\right]b_{j,v(t)}}.
$$

$$
\boxed{\gamma_{ij}(t)=\frac{\alpha_i(t)a_{ij}b_{j,v(t+1)}\beta_j(t+1)}{P(\mathbf V^T\mid\theta)}}.
$$

**8. 贝叶斯网络联合分布**

$$
\boxed{P(x_1,\ldots,x_m)=\prod_{i=1}^mP(x_i\mid\mathrm{Pa}(X_i))}.
$$

### 二、四组容易混淆的概念

| 对比 | 左边 | 右边 | 核心区别 |
|---|---|---|---|
| 概率 vs 似然 | 给定参数看数据的概率 | 给定数据比较参数的似然 | 数学式可相同，但优化视角不同 |
| MLE vs 贝叶斯估计 | 输出点估计 $\hat\theta$ | 获得参数后验分布 | 是否显式传播参数不确定性 |
| 前向 vs Viterbi | 到达同一状态的路径**求和** | 到达同一状态的路径**取最大** | 评估全部路径 vs 解码最佳路径 |
| 联合概率 vs 条件概率 | $P(A,B)$ 同时发生 | $P(A\mid B)$ 在 B 已知下 | 是否需要除以 $P(B)$ 归一化 |

### 三、练习题与讲解

**题 1：离散参数估计（基础）** 10 次独立投掷硬币，正面 8 次，反面 2 次。求正面概率 $\theta$ 的 MLE。

解：$L(\theta)\propto\theta^8(1-\theta)^2$，取对数：$\ell(\theta)=8\ln\theta+2\ln(1-\theta)+C$。求导并置零：

$$
\frac8\theta-\frac2{1-\theta}=0\implies\boxed{\hat\theta=0.8}.
$$

**题 2：一维高斯 MLE（基础）** 样本 $[2,4,6,8]$，设来自高斯分布且均值、方差未知。求 MLE。

解：$\hat\mu=(2+4+6+8)/4=5$，

$$
\hat\sigma^2=\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}=\boxed{5}.
$$

注意若题目问无偏样本方差才是 $20/3$，不要混淆。

**题 3：贝叶斯均值更新（核心）** 已知观测方差 $\sigma^2=4$，均值先验 $\mu\sim N(0,1)$；看到 2 个样本 $[2,4]$，求后验均值和方差。

解：$n=2,\sum x_k=6$，

$$
\frac1{\sigma_n^2}=1+\frac24=1.5\implies\boxed{\sigma_n^2=\frac23}.
$$

$$
\mu_n=\frac23\left(\frac04+\frac64\right)=\boxed{1}.
$$

后验预测为 $x\mid\mathcal D\sim N(1,4+2/3)$。虽然样本均值为 3，但先验均值为 0，样本又少，所以后验均值被拉向先验。

**题 4：前向概率（核心）** 已知初始状态 $\pi=(0.6,0.4)$，两个状态对第一观测的发射概率分别为 $0.5,0.2$。求 $\alpha_1(1),\alpha_2(1)$ 和第一观测的概率。

解：$\alpha_1(1)=0.6\times0.5=0.3$，$\alpha_2(1)=0.4\times0.2=0.08$，所以 $P(v(1))=0.3+0.08=\boxed{0.38}$。

**题 5：前向和 Viterbi 的区别（理解）** 前一步到状态 1 和状态 2 的分数为 0.2 和 0.3，转移到状态 3 的概率分别为 0.4 和 0.1，状态 3 发射当前符号的概率为 0.5。分别计算前向递推和 Viterbi 递推的结果。

前向：$\alpha_3(t)=[0.2\times0.4+0.3\times0.1]\times0.5=\boxed{0.055}$。

Viterbi：$\delta_3(t)=\max(0.2\times0.4,0.3\times0.1)\times0.5=\boxed{0.04}$，最佳前驱为状态 1。

**题 6：贝叶斯网络（理解）** 若 $A\to X\to C$，已知 $P(A=a_1)=0.6$、$P(A=a_2)=0.4$、$P(X=x_1\mid a_1)=0.7$、$P(X=x_1\mid a_2)=0.2$、$P(C=c_1\mid x_1)=0.8$，求 $P(X=x_1,C=c_1)$。

先消去未知的 $A$：

$$
P(X=x_1)=0.6\times0.7+0.4\times0.2=0.5.
$$

再乘亮度证据：

$$
P(X=x_1,C=c_1)=0.5\times0.8=\boxed{0.4}.
$$

### 四、按基础循序渐进的学习路线

| 学习阶段 | PDF 页码 | 应达到的能力 | 建议练习 |
|---|---|---|---|
| 第 1 阶段 | 1–8 | 理解 i.i.d.、似然、对数、梯度 | 推导硬币 MLE |
| 第 2 阶段 | 9–14 | 会推高斯均值与方差的 MLE | 计算 $\hat\mu,\hat\sigma^2$ |
| 第 3 阶段 | 15–27 | 能说清参数后验与预测分布的区别 | 计算高斯共轭更新 |
| 第 4 阶段 | 28–42 | 能解释 HMM 参数和前向后向递推 | 手算 2 状态、3 时刻 |
| 第 5 阶段 | 43–50 | 能区分 Viterbi 与 Baum–Welch | 实现 Viterbi 回溯 |
| 第 6 阶段 | 51–63 | 能根据 DAG 写联合分解并计算后验 | 完成鱼分类例题 |

**最终检查表**：

- [ ] 能从独立同分布推导 $p(\mathcal D\mid\theta)=\prod p(x_k\mid\theta)$。
- [ ] 能说明为什么对数不会改变 MLE 的最优参数位置。
- [ ] 会推导高斯均值 MLE 与方差 MLE，并分清 $n$ 和 $n-1$。
- [ ] 能解释高斯贝叶斯估计中的先验、后验及预测三种方差。
- [ ] 能写出 HMM 的 $A,B,\pi$ 并解释每个下标。
- [ ] 能区分前向求和、Viterbi 取最大、Baum–Welch 更新参数。
- [ ] 能根据 DAG 写出联合分布分解，并对未知变量边缘化。

---

## 课件校勘与使用说明

1. **页码不连续不是遗漏**：上传的 PDF 只有 63 页，原课件编号 1–69，中间存在原始幻灯片编号跳跃；本笔记对上传文件的每个 PDF 页面都单独设置了条目。
2. **数学公式中的纠正与拓展**：第 11 页正态密度分母的规范写法、第 35 页枚举复杂度的通常计法、第 49 页 Baum–Welch 发射概率更新的标准实现，都单独标注，未将教学补充伪装成原课件原文。
3. **贝叶斯网络 CPT 可能的笔误**：PDF 第 56 页亮度条件概率表的第一行按图面数据相加为 0.9，而合法 CPT 行应为 1；原样保留并提示核实。
4. **PDF 第 50 页是参考论文页**，第 60–63 页是总结页，仍依顺序完整保留。
5. **图像阅读**：推荐配套 ZIP 中的原始课件逐页截图；公式有歧义时，优先对照原图。全部补充内容用于学习教学，不应替代实际统计建模时的假设检验或数值验证。

**原始材料**：`Chapter 3.pdf`，《Pattern Recognition》课程第三章。  
**整理格式**：UTF-8 Markdown；数学式使用 `$...$` / `$...$` LaTeX；可在 Typora、Obsidian（启用数学）或 VS Code Markdown Preview Enhanced 等编辑器阅读。