# Makemore 4：成为反向传播忍者

[视频](https://www.youtube.com/watch?v=q8SA3rM6ckI)
[代码仓库](https://github.com/karpathy/makemore)
[Eureka Labs Discord](https://discord.com/invite/3zy8kqD9Cp)

## 目录

- [为什么我们要关心反向传播到这种程度？](#为什么我们要关心反向传播到这种程度)
    - [示例 - 用于双向图文映射的深度片段嵌入](#示例---用于双向图文映射的深度片段嵌入)
- [练习 1 - 反向传播](#练习-1---反向传播)
    - [1 - Logprobs](#1---logprobs)
    - [2 - Probs](#2---probs)
    - [3 - Counts_Sum_Inv](#3---counts_sum_inv)
    - [5.1 - Counts](#51---counts)
    - [4 - Counts_Sum](#4---counts_sum)
    - [5.2 - Counts](#52---counts)
    - [6 - Norm_Logits](#6---norm_logits)
    - [7 - Logit_Maxes](#7---logit_maxes)
    - [8 - Logits](#8---logits)
    - [9 - h、W2 和 b2（线性层的反向传播）](#9---hw2-和-b2线性层的反向传播)
    - [10 - hpreact](#10---hpreact)
    - [11 - bngain、bnraw、bnbias](#11---bngainbnrawbnbias)
    - [12 - bndiff、bnvar_inv（批归一化层）](#12---bndiffbnvar_inv批归一化层)
    - [13 - bnvar](#13---bnvar)
    - [贝塞尔校正](#贝塞尔校正)
    - [14 - bndiff2](#14---bndiff2)
    - [15 - bndiff](#15---bndiff)
    - [16 - bnmeani](#16---bnmeani)
    - [17 - hprebn](#17---hprebn)
    - [18 - embcat、W1、b1（线性层）](#18---embcatw1b1线性层)
    - [19 - emb](#19---emb)
    - [20 - C](#20---c)
- [练习 2 - 简化损失](#练习-2---简化损失)
    - [间奏：dlogits 是什么？](#间奏dlogits-是什么)
- [练习 3 - 简化批归一化](#练习-3---简化批归一化)
    - [步骤 1](#步骤-1)
    - [步骤 2](#步骤-2)
    - [步骤 3](#步骤-3)
    - [步骤 4](#步骤-4)

## 为什么我们要关心反向传播到这种程度？

[你应该花时间*真正*理解反向传播。](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

在过去两讲中，我们搭建了一个相当扎实的多层感知机（MLP），用来生成更多名字。
我们之所以这样做，是因为在 [makemore 的第一讲](../N002%20-%20Makemore%201/N002%20-%20Makemore.ipynb)中，我们发现直接的 bigram 模型还不够强大，无法捕捉数据的复杂性。基于 [\[Bengio 等人 2003\]](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 的思路，MLP 方法可以非常灵活地实现一个更强大的 trigram 方法。

我们了解 MLP 的架构及其工作原理。其训练的一个关键部分是 `loss.backward()` 调用，它触发 PyTorch 的反向传播例程来计算梯度，最终用于改进模型参数。**我们还没有探究过它实际是如何工作的。**

我们已经用 [Micrograd](../N001%20-%20Building%20Micrograd/N001%20-%20Micrograd.ipynb) 简要探索过反向传播，但*仅针对标量值*。
**为了真正理解反向传播，我们现在需要把它从标量扩展到张量。**

最好的探索方式是**用 PyTorch 从头实现我们自己的、基于张量的 Makemore-MLP 的反向传播**。

> 反向传播的主要问题在于它 [**是一个有泄漏的抽象。**](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

反向传播并不是某种*莫名其妙地*就为我们的 MLP 提供梯度以获得最佳结果的运算。
把它当作某种“黑魔法”，以为可以借此抽象掉学习过程中的所有复杂性和挑战，是搬起石头砸自己脚的好办法。

> 准确知道反向传播在做什么、以及如何调试它，是一个*巨大的优势*。

没有对反向传播的扎实理解，一些微妙的错误就可能悄悄潜入，并真正拖累性能。
**我们绝不应该仅仅因为 PyTorch 内置了 Autograd 来替我们处理这些细节，就跳过反向传播的细节。**

![](./img/memelol.png)

### 示例 - 用神经网络降低数据维度

Geoff Hinton 和 Ruslan Salakhutdinov 的 [2006 年论文](http://www.cs.toronto.edu/~hinton/absps/science.pdf)展示了如何用深度神经网络以可学习的方式降低数据的维度。
这篇论文实际上引入了“受限玻尔兹曼机（restricted Boltzmann machines）”，其形状非常像今天的自编码器。如果你不知道那是什么，不必在意。该论文的代码库是用 MATLAB 写的，可以在[这里](https://code.google.com/archive/p/matrbm/)找到。

早期的 MLP 并不是用 Tensorflow 或 PyTorch 乃至 Python 语言这样的框架来实现的。
那时*没有* GPU 加速，*没有*自动微分，*没有*针对层或优化器的高层抽象。
相反，当时的网络主要用 MATLAB 实现，**全靠手工**。

### 示例 - 用于双向图文映射的深度片段嵌入

[Karpathy 及其同事 2014 年的这篇论文](https://arxiv.org/abs/1406.5679)用 Python 实现，但只用到了 NumPy。
代价函数、激活函数和反向传播都由作者亲自完整实现。

**题外话 - 论文的简要 TL;DR：**
该论文训练了一个模型，输入是一组图像及关联的自然语言描述，使得之后给定一张图像可以指定一组保留的句子，反之亦然。这被称为*多模态学习（multi-modal learning）*。可以说，该论文的设置使模型能够学习到描述与图像之间的联系。

**好，现在，让我们进入反向传播的细节。**
我们将首先搭建并实现 [上一讲](../N004%20-%20Makemore%203%20-%20Activations,%20BatchNorm/N004%20-%20Makemore_3.ipynb)中 MLP 的前向传播。

```python
# 前几个单元和上一讲相比没有变化
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
# 读入所有名字
words = open('../names.txt', 'r').read().splitlines()
print('Samples:', words[:8])
print('Size:', len(words))
print('Largest:', max(len(w) for w in words))
```

```
Samples: ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia']
Size: 32033
Largest: 15
```

```python
# 构建字符词表并映射为整数
chars = sorted(list(set(''.join(words))))  # set(): 去除重复字母

stoi = {s:i+1 for i,s in enumerate(chars)} # 构造 (字符, 计数) 形式的元组
stoi['.'] = 0                              # 显式添加这个特殊字符的条目
itos = {i:s for s,i in stoi.items()}       # 把 (字符, 计数) 的顺序交换为 (计数, 字符)

vocab_size = len(itos)

# 展示这两个映射，它们其实互为镜像
print(itos)
print(stoi)
print('Training vocabulary size:', vocab_size)
```

```
{1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}
{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
Training vocabulary size: 27
```

我们再次以这样的方式构建数据集：`X` 中的输入是字符三元组（character-trigrams），而 `Y` 中的目标是下一个字符。
我们将使用和之前一样的架构：一个嵌入层、一个线性层、批归一化、一个非线性层，以及一个最后的线性层来产生下一个字符预测的 `logits`。
我们也将使用同样的损失函数，即交叉熵损失。

```python
# 构建数据集
block_size = 3 # 上下文长度：用多少个字符来预测下一个

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # 裁剪并追加

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

random.seed(42)

# 打乱并划分为三（!）个子集
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])   # 80% 训练
Xdev, Ydev = build_dataset(words[n1:n2]) # 10% 开发
Xte,  Yte  = build_dataset(words[n2:])   # 10% 测试
```

```
torch.Size([182625, 3]) torch.Size([182625])
torch.Size([22655, 3]) torch.Size([22655])
torch.Size([22866, 3]) torch.Size([22866])
```

**好，样板部分完成。现在进入正题。**
为了学习如何计算梯度，我们将利用 PyTorch 通过新函数 `cmp()` 来做对比：

```python
# 稍后会用到的工具函数
# 用于把手算的梯度与 PyTorch 计算的梯度做比较

# s  是描述该张量的字符串
# dt 是手算的梯度
# t  是 PyTorch 为之计算梯度的张量
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()        # 精确匹配（无差异）
  app = torch.allclose(dt, t.grad)           # 近似匹配（足够接近）
  maxdiff = (dt - t.grad).abs().max().item() # 两者之间的最大差异 -> 应该为零
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

现在我们像之前那样铺开 MLP 的架构。
注意，许多参数（比如偏置）都是以非标准方式初始化的。
这是因为有时用例如全零（我们之前就这么做过）来初始化可能会*“掩盖”*一个错误的反向传播实现。
通过加入 `0.1` 这个因子，会出现一些参数化的表现力，使有缺陷的实现更容易暴露出来。

**此时只需记住：**
MLP 接收一个由 $3$ 个分词字母组成的上下文，并输出 $27$ 个可能的输出 token 之一，代表最可能跟在这 $3$ 字母输入之后的那个字母。

```python
n_embd = 10   # 神经网络字符嵌入向量的维度
n_hidden = 64 # 每个 MLP 隐藏层的神经元数量

g = torch.Generator().manual_seed(2147483647)       # 用于可复现性
C  = torch.randn((vocab_size, n_embd), generator=g) # 字符嵌入表

prop_scale = 0.1 # 加入缩放因子以避免掩盖反向传播中潜在的错误

# 第 1 层
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) # (3x10)x64
W1 *= (5/3)/((n_embd * block_size)**0.5)              # 初始权重缩放
b1 = torch.randn(n_hidden, generator=g) * prop_scale  # 用 64 个偏置 b1 图个乐子，其实并非必需

# 批归一化
bngain = torch.randn((1, n_hidden))*prop_scale + 1.0
bnbias = torch.randn((1, n_hidden))*prop_scale

# 第 2 层
W2 = torch.randn((n_hidden, vocab_size), generator=g) * prop_scale # 64x27
b2 = torch.randn(vocab_size, generator=g) * prop_scale # 与 b1 不同，这些偏置是必需的，用以暴露梯度中潜在的细微计算误差

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # 总参数数

for p in parameters:
  p.requires_grad = True
```

```
4137
```

权重和偏置矩阵被初始化为小、随机但有表现力的值。
和之前一样，我们会使用分批输入。批次大小设为 $32$。
**对于下面的情况，我们只构造一个这样的批次。**

把一个批次前向传过我们的网络就够了：

```python
# 从训练集构造一个批次
batch_size = 32 # 一次处理多少个样本
n = batch_size  # 也把 N 作为更短的变量名，图个方便

# 构造小批量本身
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g) # 随机索引
Xb, Yb = Xtr[ix], Ytr[ix] # 从 Xtr,Ytr 中组装这个随机批次
```

**现在我们把这个批次传过网络。**

这次前向传播，尤其是批归一化和交叉熵损失，被改写成了其 PyTorch 实现的“分步版本”，以便我们能手工对其反向传播：

```python
# 前向传播，被“切块”成更小的、可逐个 'backward' 的单步

emb = C[Xb]                         # 把字符嵌入为向量
embcat = emb.view(emb.shape[0], -1) # 拼接向量，形成一个大的

# 线性层 1
hprebn = embcat @ W1 + b1           # 隐藏层预激活

# 批归一化层
bnmeani = 1/n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1 / (n-1) * (bndiff2).sum(0, keepdim=True) # 注意：贝塞尔校正（除以 n-1 而非 n）
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

# 非线性
h = torch.tanh(hpreact) # 隐藏层

# 线性层 2
logits = h @ W2 + b2 # 输出层

# 交叉熵损失（所有这些之前都浓缩在 F.cross_entropy(logits, Yb) 里）
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # 减去最大值以保证数值稳定性
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # 如果改用 (1.0 / counts_sum)：无法让反向传播做到逐位精确
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch 反向传播
for p in parameters:
  p.grad = None # 清零所有梯度，清除之前反向传播的残留

# 保留梯度，它们不会在反向传播结束后被释放，以便我们做对比
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()

loss.backward()
loss
```

```
tensor(3.3344, grad_fn=<NegBackward0>)
```

好，我们搭建了 MLP、构造了一个批次、并把它传过了网络。
我们告诉网络为所有中间张量保留梯度。
然后我们运行 PyTorch 的 `loss.backward()` 自动计算所有这些梯度。

我们前向传播中的各变量以下列方式相互连接，最终汇成 `loss`：
![](./img/ninja_graph.png)

上面这张依赖与运算图不太上相，所以在接下来几节里我们只会逐步过一遍。
**现在梯度已就绪可供参照。**
**接下来就看我们去搭建并对比自己的反向传播实现了。**

## 练习 1 - 反向传播
**对整个前向传播做反向传播，按上面定义的顺序，逐个、从后往前地过一遍所有变量。**

```python
# 提醒一下，从 Micrograd 我们知道：
#
# 加法的梯度 c = a + b：
# a.grad = c.grad * (1)
# b.grad = c.grad * (1)
#
# 乘法的梯度 c = a * b：
# a.grad = c.grad * b.data
# b.grad = c.grad * a.data
```

### 1 - Logprobs

`loss` 直接且仅通过 `loss = -logprobs[range(n), Yb].mean()` 依赖于对 `logprobs` 的运算。

由于梯度的形状与为其计算梯度的变量相同，`dlogprobs` 也必须是 `(32, 27)` 的形状，和 `logprobs` 一样。
对于 `-logprobs[range(n), Yb]`，我们针对每行 $i \in [0;n)$ 取出 `Yb` 中该行条目所指索引处的值。
正是这些、且仅这些特定索引构成了我们随后计算负均值的基础。

对于梯度计算而言，这意味着我们仅通过逐行“拔出”的那些索引真正影响到 `loss`。
**因此 `logprobs` 中所有未被 `[range(n), Yb]` 选中的值梯度为零。我们可以直接断言，因为这些值不影响 `loss`。**

三个值 $a,\ b,\ c$ 的负均值为 $(-\frac{a+b+c}{3}) = (-\frac{1}{3}) \cdot a + (-\frac{1}{3}) \cdot b + (-\frac{1}{3}) \cdot c$。
由此可推 $\frac{\partial (-mean)}{\partial a} = \frac{\partial (-mean)}{\partial b} = \frac{\partial (-mean)}{\partial c} = -\frac{1}{3}$。

$$\frac{\partial\ \text{loss}}{\partial\ \text{logprobs}} = \underline{\underline{-\frac{1}{n}}}$$

**`logprobs` 中所有被选中的值梯度为 $-\frac{1}{n}$，其中 $n$ 是 `range(n)` 调用给出、用于“拔出”索引的行数：**

```python
# LOGPROBS
dlogprobs = torch.zeros_like(logprobs) # (32x27)，全零，形状与 logprobs 一致（这意味着没有元素参与 loss，但我们下一步会修正）
dlogprobs[range(n), Yb] = -1.0 / n     # (32x27)，只有正确的索引被赋予 -1/n 的梯度（其余保持 0）
cmp('logprobs', dlogprobs, logprobs)
```

```
logprobs        | exact: True  | approximate: True  | maxdiff: 0.0
```

### 2 - Probs

`probs` 直接且仅通过 `logprobs = probs.log()` 用于计算 `logprobs`，其形式即 $\log_e(\text{probs})$。
**这里我们需要小心，不要忘了额外的一步。**

由于 `probs` 作用于 `logprobs`，**相应的梯度 `dlogprobs` 会通过链式法则作用于 `dprobs`：**
$$d\text{probs} = \frac{d \log_e(\text{probs})}{d \text{probs}} \cdot d\text{logprobs}$$

我们也应看看 `logprobs` 和 `probs` 的形状，但这里没什么可担心的。
两者都是 `(32, 27)`。$\log_e(\text{probs})$ 对 $\text{probs}$ 的导数为：
$$\frac{d \log_e(\text{probs})}{d\text{probs}} = \frac{1}{\text{probs}}$$

因此，与 `dlogprobs` 乘法链接后，得到：
$$d\text{probs} = \frac{1}{\text{probs}} \cdot d\text{logprobs}$$

写成代码就是：

```python
# PROBS
dprobs = (1 / probs) * dlogprobs # (32x27)，逐元素相乘
cmp('probs', dprobs, probs)
```

```
probs           | exact: True  | approximate: True  | maxdiff: 0.0
```

### 3 - Counts_Sum_Inv

`counts_sum_inv` 只在一处被使用：`probs = counts * counts_sum_inv`。

`counts_sum_inv` 只有 `probs` 依赖于它，但它与 `counts` 相互作用以计算 `probs`。
由于两者相互影响，`counts_sum_inv` 和 `counts` 的梯度是相互依赖的。

`counts` 是一个 `(32, 27)` 的矩阵。
`counts_sum_inv` 是一个 `(32, 1)` 的向量。
在做乘法时，PyTorch 会把向量 `counts_sum_inv` 沿 `counts` 的列方向广播，从而由该向量生成一个同样大小为 `(32, 27)` 的矩阵。
利用链式法则知识，我们可以针对导向 `counts_sum_inv` 的*两*个运算（广播和乘法）来计算 `dcounts_sum_inv`。

对于乘法部分，我们得到：
$$\frac{\partial (\text{counts} \cdot \text{counts\_sum\_inv})}{\partial \text{counts\_sum\_inv}} = \text{counts} \cdot \text{dprobs}$$
由于链式法则，`dprobs` 被 `counts` 乘。

**对于广播部分，我们需要运用一些直觉。**
从 [micrograd 那一讲](../N001%20-%20Building%20Micrograd/N001%20-%20Micrograd.ipynb)我们知道，如果一个值在多处被使用，那么梯度会在这些使用点上加总。
如果 `counts_sum_inv` 被广播到 `counts` 的所有列，那么 `counts_sum_inv` 的梯度就是第一步结果沿所有列的求和：
$$\frac{\partial\ \text{probs}}{\partial\ \text{counts\_sum\_inv}} = \sum_i \left( \text{counts} \cdot \text{dprobs} \right)_{:,i}$$

这可以这样实现：

```python
# COUNTS_SUM_INV
# probs = counts * counts_sum_inv (counts = 32x27, counts_sum_inv = 32x1) 或
# c = a * b，但用张量
# a[3x3] * b[3x1] ---->
# a11*b1 a12*b1 a13*b1
# a21*b2 a22*b2 a23*b2
# a31*b3 a32*b3 a33*b3
# ----> c[3x3]

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True) # (32x1)，对行求和（轴 1，即列方向，keepdim=True 以保持 2D 形状）
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv) # true
```

```
counts_sum_inv  | exact: True  | approximate: True  | maxdiff: 0.0
```

### 5.1 - Counts

正如我们所见，`probs` 是用 `counts` 计算的。

有两个变量依赖于 `counts`：
- `probs`，通过 `probs = counts * counts_sum_inv`
- `counts_sum_inv`，通过 `counts_sum = counts.sum(1, keepdims=True)`（以及 `counts_sum_inv = counts_sum**-1`）

**`counts` 是一个被使用两次的节点，一次用于 `probs`，一次用于 `counts_sum_inv`。**

这两条计算路径都影响 `counts` 的梯度，但此时我们只覆盖了 `probs` 这条路径，还没有处理 `dcounts_sum`。

`counts` 的形状是 `(32, 27)`。
`counts_sum_inv` 的形状是 `(32, 1)`。
我们可以直接套用乘法的梯度计算规则：对于 $c = a \cdot b$，$\frac{\partial c}{\partial a}$ 为 $b$。

同样，不能忘了链式法则：

```python
# 第 1 步：
dcounts = counts_sum_inv * dprobs # (32x1) * (32x27) = (32x27)，逐元素相乘（自动广播）

# 第 2 步：
# 暂时搁置
cmp('counts', dcounts, counts) # False，但稍后会修好
```

```
counts          | exact: False | approximate: False | maxdiff: 0.0057856254279613495
```

**这不对，因为它只覆盖了 `counts` 影响损失的两条路径中的一条。**
我们会在下一步算出 `dcounts_sum` 之后再回到 `dcounts`。

### 4 - Counts_Sum

`counts_sum` 通过 `counts_sum_inv = counts_sum**-1` 用于计算 `counts_sum_inv`。
由于这是它唯一的用途，我们可以直接计算 `counts_sum_inv` 对 `loss` 的偏导数。

应用链式法则，得到：

```python
# COUNTS_SUM
dcounts_sum = -counts_sum**(-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)
```

```
counts_sum      | exact: True  | approximate: True  | maxdiff: 0.0
```

### 5.2 - Counts

现在我们可以处理 `counts` 的第二个用途，即通过 `counts_sum = counts.sum(1, keepdims=True)`。
- `counts` 形状为 `(32, 27)`
- `counts_sum` 形状为 `(32, 1)`

我们执行这个运算：

$$\begin{bmatrix}
a_{11} + a_{12} + a_{13}\\
a_{21} + a_{22} + a_{23}\\
a_{31} + a_{32} + a_{33}\\
\end{bmatrix} = \begin{bmatrix}
b_1\\
b_2\\
b_3\\
\end{bmatrix}$$

`b`（即 `counts_sum` 向量）如何依赖于 `a`（即 `counts` 矩阵）？

只看 $b_1$，它只依赖于 $a_{11}$、$a_{12}$ 和 $a_{13}$。所有其它 $a$ 相对 $b_1$ 的梯度为零。
$a_{11}$、$a_{12}$ 和 $a_{13}$ 的梯度为 $1.0$。（偏导数 $\frac{\partial b_1}{\partial a_{11}}$ 为 $1.0$）

在链式法则中，我们有局部导数乘以 $b_1$ 的导数。由于局部导数为 $1.0$，可以省略，结果就是 $b_1$。
$b_1$ 的导数会等量地流向所有用于计算它的 $a$。这等同于说 $b_1$ 的导数为 $1.0$。

由链式法则，我们已知 `dcounts_sum`。

由于所有局部导数都是 $1$，我们只需把 `dcounts_sum` 按行复制到 `counts` 的每一行，从而把它拉伸到 `counts` 的形状。
此外要记住，这是 `counts_sum_inv` 对 `loss` 偏导数的第二步。所以我们要做的是用 `+=` 加到 `dcounts` 已有的那一半上。

```python
# COUNTS

# a11 a12 a13 ---> b1 (= a11 + a12 + a13)
# a21 a22 a23 ---> b2 (= a21 + a22 + a23)
# a31 a32 a33 ---> b3 (= a31 + a32 + a33)

dcounts += torch.ones_like(counts) * dcounts_sum # (32x27)，全 1，乘以 (32x1) = (32x27)（自动广播）
cmp('counts', dcounts, counts)
```

```
counts          | exact: True  | approximate: True  | maxdiff: 0.0
```

### 6 - Norm_Logits

`norm_logits` 通过 `counts = norm_logits.exp()` 用于计算 `counts`。
$e^x$ 的局部导数就是 $e^x$，而既然我们写的是 `counts = norm_logits.exp()`，可以直接用 `counts` 作为局部导数：

```python
dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)
```

```
norm_logits     | exact: True  | approximate: True  | maxdiff: 0.0
```

### 7 - Logit_Maxes

`logits` 和 `norm_logits` 通过 `norm_logits = logits - logit_maxes` 用于计算 `norm_logits`。
`norm_logits` 是 `(32, 27)`，`logits` 也是，但 `logit_maxes` 是 `(32, 1)`。

我们基本上在执行这个运算：

$$\begin{bmatrix}
c_{11} & c_{12} & c_{13}\\
c_{21} & c_{22} & c_{23}\\
c_{31} & c_{32} & c_{33}\\
\end{bmatrix} = \begin{bmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}\\
\end{bmatrix} - \begin{bmatrix}
b_{1} & b_1 & b_1\\
b_{2} & b_2 & b_2\\
b_{3} & b_3 & b_3\\
\end{bmatrix}$$
$$e.g.\ c_{32} = a_{32} - b_3$$

我们得看看每个 `c` 是怎么来的。

每个 `c` 都是一个 `a` 与某个 `-b` 之和。
每个 `c` 对其关联 `a` 的导数为 `1`。
每个 `c` 对其关联 `b` 的导数为 `-1`。

换句话说，每个 `c` 中所含的导数会等量地流向相应的 `a`，同时也以负号流向 `b`。
这意味着 $dc = da$，即 $\text{dlogits} = \text{dnorm\_logits}$ 且 $\text{dlogit\_maxes} = -\text{dnorm\_logits}$。
$\text{dlogit\_maxes}$ 已经完成，$\text{dlogits}$ 只完成了一部分。

```python
dlogits = dnorm_logits.clone()
dlogit_maxes = -1 * dnorm_logits.sum(1, keepdim=True)

cmp('logit_maxes', dlogit_maxes, logit_maxes)
```

```
logit_maxes     | exact: True  | approximate: True  | maxdiff: 0.0
```

### 8 - Logits

除了在 `norm_logits = logits - logit_maxes` 中的用途外，`logits`（其形状为 `(32, 27)`）还被用于 `logit_maxes = logits.max(1, keepdim=True).values`。

通过调用 `.max(1)`，PyTorch 不仅给出每行的最大值，还给出它们在 `logits` 中各自的索引，这对反向传播很实用。

导数在“拔出”位置应为 $1$，其余处处为 $0$。
换句话说，我们需要把 `dlogit_maxes` 散布到 `dlogits` 上，并在取出最大值的那些索引处加上它们的值。

```python
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

# 取每行 logits.max 的索引，并把它 one-hot 编码为“在 27（logits.shape[1]）个可能类中恰好出现一次的类 'index'”
# 这会（按行）在取出最大值的索引处置 1
# 乘以 dlogit_maxes 把这些 1 替换成正确的梯度值
# += 用于完成第二个分支

cmp('logits', dlogits, logits)
```

```
logits          | exact: True  | approximate: True  | maxdiff: 0.0
```

### 9 - h、W2 和 b2（线性层的反向传播）

`h` 仅通过 `logits = h @ W2 + b2` 用于计算 `logits`。
- `dlogits` 形状为 `(32, 27)`
- `h` 是 `(32, 64)`
- `W2` 是 `(64, 27)`
- `b2` 是 `(27,)`

我们如何计算 `h`、`W2`、`b2` 对 `loss` 的偏导数？
要理解发生了什么，我们来看一个具体的小例子并写出来：

$$\begin{bmatrix}d_{11} & d_{12}\\ d_{21} & d_{22}\\ \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12}\\ a_{21} & a_{22}\\ \end{bmatrix} \cdot \begin{bmatrix} b_{11} & b_{12}\\ b_{21} & b_{22}\\ \end{bmatrix} + \begin{bmatrix} c_{1} & c_{2}\\ c_{1} & c_{2}\\ \end{bmatrix}$$
这等价于：
$$d_{11} = a_{11} \cdot b_{11} + a_{12} \cdot b_{21} + c_1\\
d_{12} = a_{11} \cdot b_{12} + a_{12} \cdot b_{22} + c_2\\
d_{21} = a_{21} \cdot b_{11} + a_{22} \cdot b_{21} + c_1\\
d_{22} = a_{21} \cdot b_{12} + a_{22} \cdot b_{22} + c_2$$

有了 `dlogits`，$\frac{\partial L}{\partial d_{11}}$、$\frac{\partial L}{\partial d_{12}}$、$\frac{\partial L}{\partial d_{21}}$ 和 $\frac{\partial L}{\partial d_{22}}$ 就都已知了。
现在我们想要 `L` 对 `h`（即我们例子中的 `a`）的偏导数。

$$\frac{\partial L}{\partial a_{11}} = \frac{\partial L}{\partial d_{11}} \cdot b_{11} + \frac{\partial L}{\partial d_{12}} \cdot b_{12}$$
$$\frac{\partial L}{\partial a_{12}} = \frac{\partial L}{\partial d_{11}} \cdot b_{21} + \frac{\partial L}{\partial d_{12}} \cdot b_{22}$$
$$\frac{\partial L}{\partial a_{21}} = \frac{\partial L}{\partial d_{21}} \cdot b_{11} + \frac{\partial L}{\partial d_{22}} \cdot b_{12}$$
$$\frac{\partial L}{\partial a_{22}} = \frac{\partial L}{\partial d_{21}} \cdot b_{21} + \frac{\partial L}{\partial d_{22}} \cdot b_{22}$$
例如，$a_{11}$ 用于计算 $d_{11}$ 和 $d_{12}$，所以对 $\frac{\partial L}{\partial a_{11}}$，我们需要把 $\frac{\partial L}{\partial d_{11}}$ 和 $\frac{\partial L}{\partial d_{12}}$ 分别乘以 $b_{11}$ 和 $b_{12}$，依此类推。

我们可以继续，把它改写成矩阵乘法的形式：
$$\begin{bmatrix} \frac{\partial L}{\partial d_{11}} & \frac{\partial L}{\partial d_{12}}\\ \frac{\partial L}{\partial d_{21}} & \frac{\partial L}{\partial d_{22}}\\ \end{bmatrix} \cdot \begin{bmatrix} b_{11} & b_{21}\\ b_{12} & b_{22}\\ \end{bmatrix} = \begin{bmatrix} \frac{\partial L}{\partial a_{11}} & \frac{\partial L}{\partial a_{12}}\\ \frac{\partial L}{\partial a_{21}} & \frac{\partial L}{\partial a_{22}}\\ \end{bmatrix} = \frac{\partial L}{\partial d}\ \times\ b^T$$

如果我们退一步对 `b` 做同样的事，得到：$\frac{\partial L}{\partial b} = a^T\ \times\ \frac{\partial L}{\partial d}$。
对 `c`，得到：$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \cdot sum(0)$。（$sum(0)$ 指沿第一维求和，即沿列求和。）

> 矩阵乘法的反向传播，就是前向传播（一次矩阵乘法）的转置。

---

**还有一种更直观的理解方式。**

我们再看一下维度：

最初的运算是：`logits = h @ W2 + b2`
- `dlogits` 形状为 `(32, 27)`
- `h`  是 `(32, 64)`
- `W2` 是 `(64, 27)`
- `b2` 是 `(27,)`

`dh` 应与 `h` 形状相同，即 `(32, 64)`。
`dh` 必然是某种涉及 `dlogits` 和 `W2` 的矩阵乘法。
要得到 `(32, 64)` 的形状，唯一的办法就是把 `(32, 27)` 的矩阵 `dlogits` 与 `W2` 转置后 `(27, 64)` 的矩阵相乘。

把这些都想清楚后，我们就可以计算所有必要的偏导数了：

```python
dh = dlogits @ W2.T
cmp('h', dh, h)

dW2 = h.T @ dlogits
cmp('W2', dW2, W2)

db2 = dlogits.sum(0)
cmp('b2', db2, b2)
```

```
h               | exact: True  | approximate: True  | maxdiff: 0.0
W2              | exact: True  | approximate: True  | maxdiff: 0.0
b2              | exact: True  | approximate: True  | maxdiff: 0.0
```

### 10 - hpreact

`hpreact` 仅通过 `h = torch.tanh(hpreact)` 用于计算 `h`。
从 [micrograd](../N001%20-%20Building%20Micrograd/N001%20-%20Micrograd.ipynb)我们知道，`tanh` 的导数为：

$$a = \tanh(z) = \frac{e^{2z} - 1}{e^{2z} + 1}$$
$$\frac{\partial a}{\partial z} = 1 - a^2 = 1 - \tanh^2(z)$$

带着这个和链式法则，`hpreact` 对 `loss` 的偏导数为：

```python
dhpreact = (1.0 - h ** 2) * dh
cmp('hpreact', dhpreact, hpreact)
```

```
hpreact         | exact: True  | approximate: True  | maxdiff: 0.0
```

### 11 - bngain、bnraw、bnbias

`bngain`、`bnraw` 和 `bnbias` 仅通过 `hpreact = bngain * bnraw + bnbias` 用于计算 `hpreact`。
（`bngain` 和 `bnbias` 用于批归一化）

- `hpreact` 形状为 `(32, 64)`
- `bngain` 是 `(1, 64)`
- `bnraw` 是 `(32, 64)`
- `bnbias` 是 `(1, 64)`

我们要确保 PyTorch 的广播被正确地反向传播：

```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True) # sum 处理广播带来的影响
cmp('bngain', dbngain, bngain)

dbnraw = bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

dbnbias = dhpreact.sum(0, keepdim=True) # sum 处理广播带来的影响
cmp('bnbias', dbnbias, bnbias)
```

```
bngain          | exact: True  | approximate: True  | maxdiff: 0.0
bnraw           | exact: True  | approximate: True  | maxdiff: 0.0
bnbias          | exact: True  | approximate: True  | maxdiff: 0.0
```

### 12 - bndiff、bnvar_inv（批归一化层）

`bndiff` 和 `bnvar_inv` 仅通过 `bnraw = bndiff * bnvar_inv` 用于计算 `bnraw`。
（`bndiff` 和 `bnvar_inv` 用于批归一化）

- `bndiff` 形状为 `(32, 64)`
- `bnvar_inv` 是 `(1, 64)`
- `bnraw` 是 `(32, 64)`

把广播考虑进去，这就比较直接了：

```python
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True) # sum 处理广播带来的影响
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

# 第 1 步
dbndiff = bnvar_inv * dbnraw
cmp('bndiff', dbndiff, bndiff) # False，但稍后会修好
```

```
bnvar_inv       | exact: True  | approximate: True  | maxdiff: 0.0
bndiff          | exact: False | approximate: False | maxdiff: 0.0011395297478884459
```

### 13 - bnvar

`bnvar` 仅通过 `bnvar_inv = (bnvar + 1e-5) ** -0.5` 用于计算 `bnvar_inv`。
（`bnvar` 和 `bnvar_inv` 用于批归一化）

由于这里涉及幂运算，我们对其导数应用幂法则：

$$\frac{d}{dx}x^n=n \cdot x^{n-1}$$

```python
dbnvar = (-0.5 * (bnvar + 1e-5) ** (-1.5)) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)
```

```
bnvar           | exact: True  | approximate: True  | maxdiff: 0.0
```

### 贝塞尔校正

在对 `bndiff2` 的用法 `bnvar = 1 / (n-1) * (bndiff2).sum(0, keepdim=True)` 做反向传播之前，我们需要理解其中应用的 [贝塞尔校正](https://mathcenter.oxford.emory.edu/site/math117/besselCorrection/)。
贝塞尔校正之所以有趣，是因为它偏离了原始的批归一化论文，该论文对我们这个方程给出的是：$\sigma^2_B \leftarrow \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$。

我们用 `n`，它就是论文里的 `m`，但随后从中减去 `1`。
这是因为我们想使用对抽取样本所来自的总体的**无偏估计**。
它提供了对总体方差更准确的估计。

*最初*，论文提议只在评估时使用这个无偏估计，而不是在训练时。
PyTorch 也是在训练时使用有偏估计，而在评估时使用无偏估计。
这会在训练与评估之间引入一个*可避免的差异*，Andrej 在[这里](https://youtu.be/q8SA3rM6ckI?t=4051)指出了这一点。

有趣的是，随着批次大小增大，出错的潜力会降低。*尽管如此，这还是有点怪。*
对此的一个解决办法（正如这里所做的）是在训练时*和*评估时都使用*无偏*估计。

### 14 - bndiff2

`bndiff2` 仅通过 `bnvar = 1 / (n-1) * (bndiff2).sum(0, keepdim=True)` 用于计算 `bnvar`。
（`bndiff2` 和 `bnvar` 用于批归一化）

- `bndiff2` 是 $(32\times 64)$
- `bnvar` 是 $(1\times 64)$

> 对 `bndiff2` 的列求和，在反向传播中会变成复制/广播。
> 反过来，如果前向传播中是复制，那么反向传播中就变成沿同一维的求和。

`bndiff2` 与 `bnvar` 相互关系的一个小例子：
$$\begin{bmatrix}a_{11} & a_{12}\\ a_{21} & a_{22}\end{bmatrix} \rightarrow \begin{bmatrix}b_1\\ b_2\end{bmatrix} = \begin{matrix}\frac{1}{(n-1)} \cdot (a_{11} + a_{21})\\ \frac{1}{(n-1)} \cdot (a_{12} + a_{22})\end{matrix}$$

我们已有 $b_1$ 和 $b_2$ 的导数（即 `bnvar` 的各列）。
我们要把它们反向传播到 `a`。
$\frac{b_1}{b_2}$ 的导数必须流经 $a$ 的第一/第二列，并按 $\frac{1}{(n-1)}$ 缩放。

结合链式法则，得到：

```python
dbndiff2 = (1.0 / (n-1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)
```

```
bndiff2         | exact: True  | approximate: True  | maxdiff: 0.0
```

### 15 - bndiff

`bndiff` 在两处被使用：
- `bndiff2 = bndiff**2`
- `bnraw = bndiff * bnvar_inv`

这让计算 `dbndiff` 变得*非常容易*。
但要记住，我们已经在第 $12$ 步为 `dbndiff` 算过第 $1$ 步了。

我们只需把两者相加：

```python
# 第 2 步
dbndiff += 2 * bndiff * dbndiff2
cmp('bndiff', dbndiff, bndiff)
```

```
bndiff          | exact: True  | approximate: True  | maxdiff: 0.0
```

### 16 - bnmeani

`bnmeani` 仅通过 `bndiff = hprebn - bnmeani` 用于计算 `bndiff`。

- `bnmeani` 形状为 `(1, 64)`
- `bndiff` 是 `(32, 64)`
- `hprebn` 是 `(32, 64)`

这里发生了一次广播。这意味着 `bnmeani` 被重复使用。
因此我们可以取 `dbndiff`，沿行求和并取负。
`hprebn` 与 `bndiff` 形状相同，所以直接把导数复制过去即可。

```python
dbnmeani = (-1.0) * dbndiff.sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

# 第 1 步
dhprebn = dbndiff.clone() # 复制张量（而不只是引用）
cmp('hprebn', dhprebn, hprebn) # False，但稍后会修好
```

```
bnmeani         | exact: True  | approximate: True  | maxdiff: 0.0
hprebn          | exact: False | approximate: False | maxdiff: 0.0009513851255178452
```

### 17 - hprebn

如第 $16$ 步所见，`hprebn` 用于计算 `bndiff`。
它的第二个用途是通过 `bnmeani = 1/n * hprebn.sum(0, keepdim=True)` 计算 `bnmeani`。
- `hprebn` 形状为 `(32, 64)`
- `bnmeani` 是 `(1, 64)`

这与广播正好相反。我们要把 `bnmeani` 的导数乘以 `1/n` 后复制到 `hprebn` 的各行，并追加到 `dhprebn` 已有的部分上：

```python
dhprebn += (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
cmp('hprebn', dhprebn, hprebn)
```

```
hprebn          | exact: True  | approximate: True  | maxdiff: 0.0
```

### 18 - embcat、W1、b1（线性层）

`embcat` 用于线性层 `hprebn = embcat @ W1 + b1`。
这其实与第 $9$ 步中已描述的内容非常相似。

我们直接复用该情形下的计算：

```python
dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)

db1 = dhprebn.sum(0)
cmp('b1', db1, b1)
```

```
embcat          | exact: True  | approximate: True  | maxdiff: 0.0
W1              | exact: True  | approximate: True  | maxdiff: 0.0
b1              | exact: True  | approximate: True  | maxdiff: 0.0
```

### 19 - emb

`emb` 通过 `embcat = emb.view(emb.shape[0], -1)` 用于线性层 `hprebn = embcat @ W1 + b1`。

- `embcat` 形状为 `(32, 30)`
- `emb` 是 `(32, 3, 10)`

这是一个重塑（reshape）操作。
我们只需把导数复制过去并重塑回原始形状即可：

```python
# 把 32x30 变成 32x3x10（仅此而已）
demb = dembcat.view(n, -1, emb.shape[2])
cmp('emb', demb, emb)
```

```
emb             | exact: True  | approximate: True  | maxdiff: 0.0
```

### 20 - C

`C` 通过 `emb = C[Xb]` 用于线性层 `hprebn = embcat @ W1 + b1`。
- `emb` 形状为 `(32, 3, 10)`
- `C` 是 `(27, 10)`
- `Xb` 是 `(32, 3)`

`Xb` 里有三个整数，每个指定使用 `C` 的哪一行。
我们需要为 `Xb` 中每个整数找到 `C` 的对应行，然后把导数路由到那一行。
此外，如果某一行被多次使用，到达那里的梯度要加起来。

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]        # 某个字符的整数索引
        dC[ix] += demb[k, j] # 把该字符嵌入的梯度加到字符计数的梯度上

cmp('C', dC, C)
```

```
C               | exact: True  | approximate: True  | maxdiff: 0.0
```

## 练习 2 - 简化损失

练习 1 很深入，***但远谈不上现实***。
所反向传播经过的那些步骤过于细致。
单看损失，在更高层次上做的反向传播其实可以简化。

这可以通过对比“整体式”损失函数与一个简化计算来看：

```python
# 前向传播

# 之前：
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # 减去最大值以保证数值稳定性
# counts = norm_logits.exp()
# counts_sum = counts.sum(1, keepdims=True)
# counts_sum_inv = counts_sum**-1 # 如果改用 (1.0 / counts_sum)，就无法让反向传播逐位精确...
# probs = counts * counts_sum_inv
# logprobs = probs.log()
# loss = -logprobs[range(n), Yb].mean()

# 现在：
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())
```

```
3.334364652633667 diff: -2.384185791015625e-07
```

那么，这个简化后、与我们的“整体式”几乎没有差异的前向传播，其反向传播是否也简化了？
**要完成练习 2，请查看损失的数学表达式，求导，化简表达式，然后写出来。**

`dlogits` 理想情况下会变成一个依赖 `logits` 和 `Yb`、但更短的函数。

我们有来自神经网络的 `logits`。然后一个 `softmax` 函数把 `logits` 变形为 `probs`（形状相同）。
对于 `probs`，我们用正确下一个字符的身份来“拔出”一行概率。
我们对刚刚“拔出”的值取负对数。
然后我们把所得的负对数概率取平均。这就是 `loss`。

我们要求的是 $\frac{\partial \text{loss}}{\partial l_i}$。
它按如下方式计算：

$$\frac{\partial \text{loss}}{\partial l_i} = \frac{\partial}{\partial l_i} \begin{bmatrix}-\log\frac{e^{l_y}}{\sum_j e^{l_j}}\end{bmatrix}$$

我们要对这里针对 $l_i$ 表达式求导。
既然 $\text{loss} = -\log(p_y)$ 且 $p_i = \frac{e^{l_i}}{\sum_je^{l_j}}$，我们可以改写为：
$$\text{loss} = -\log\left(\frac{e^{l_y}}{\sum_je^{l_j}}\right)\\ $$
我们的目标是得到这个表达式对 $l_i$ 的偏导数：
$$\frac{\partial \text{loss}}{\partial l_i} = \frac{\partial}{\partial l_i} \begin{bmatrix}-\log \left(\frac{e^{l_y}}{\sum_je^{l_j}}\right)\end{bmatrix}$$
利用法则 $\frac{\partial}{\partial x} \log(x) = \frac{1}{x}$，可以消去外层的 log：
$$= -\frac{\sum_j e^{l_j}}{e^{l_y}} \cdot \frac{\partial}{\partial l_i} \begin{bmatrix}\frac{e^{l_y}}{\sum_je^{l_j}}\end{bmatrix}$$

$i$ 与 $y$ 之间可能有两种关系：
- $i \neq y \rightarrow p_i$
- $i = y \rightarrow p_i - 1$

我们需要计算 softmax 的结果 `p`，并在正确的维度 `i` 上，要么把值拔出就完事，要么从中减去 `1`：

```python
# 反向传播

dlogits = F.softmax(logits, dim=1) # 沿 logits 的行做 Softmax
dlogits[range(n), Yb] -= 1 # 在 dlogits 中正确的位置处，总是需要减去一个 1
dlogits /= n               # 由于取了均值，梯度按 n 缩小

cmp('logits', dlogits, logits) # 我只能让 approximate 为 true，maxdiff 为 6e-9
```

```
logits          | exact: False | approximate: True  | maxdiff: 7.916241884231567e-09
```

### 间奏：dlogits 是什么？

`dlogits` 是一个 `(32, 27)` 的矩阵。
除此之外，`dlogits` 就是前向传播中的概率矩阵。
黑点显示了正确的索引，也就是我们在上一步中减去 `1` 的位置。
换句话说，对每一行，我们都在把正确索引的概率往上*拉*，而把所有其它的往下*压*。
如果你取 `dlogits[0].sum()`，结果是 $0$。
推和拉的总量相等，但分布不均。一个被拉，所有其余的被推。

> 你的预测错误程度，正是你在该维度上被拉或被推的程度。
> 一个被自信地误判的元素会被更重地往下压。
> 梯度与误判的程度成正比。
> 这在 `dlogits` 中按行发生。

```python
plt.figure(figsize=(8,8))
plt.imshow(dlogits.detach(), cmap='gray');
```

## 练习 3 - 简化批归一化

要完成这个挑战，请查看批归一化输出的数学表达式，对其输入求导，化简表达式，然后写出来。
批归一化论文：[\[Ioffe, Sergey; Szegedy, Christian. 2015\]](https://arxiv.org/abs/1502.03167)

我们要找到单一的一个运算，一次性地反向传过构成批归一化的整堆方程。
我们有 `dhpreact`（论文中的 $y_i$），想*高效地*由它产生 `dhprebn`。

```python
# 前向传播

# 之前：
# bnmeani = 1/n*hprebn.sum(0, keepdim=True)
# bndiff = hprebn - bnmeani
# bndiff2 = bndiff**2
# bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # 注意：贝塞尔校正（除以 n-1 而非 n）
# bnvar_inv = (bnvar + 1e-5)**-0.5
# bnraw = bndiff * bnvar_inv
# hpreact = bngain * bnraw + bnbias

# 现在：
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
print('max diff:', (hpreact_fast - hpreact).abs().max())
```

```
max diff: tensor(4.7684e-07, grad_fn=<MaxBackward1>)
```

我们直接把批归一化论文里的方程写出来：
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i\\
\sigma_B^2 = \frac{1}{m-1} \sum_{i=1}^m(x_i-\mu_B)^2\ \ \ \ \ \small{(\text{贝塞尔校正})}\\
\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}\\
y_i = \gamma\hat{x_i}+\beta \equiv BN_{\gamma,\beta}(x_i)$$
我们已有 $\frac{\partial l}{\partial y_i}$，并据此想要 $\frac{\partial l}{\partial x_i}$。

### 步骤 1

第一部分很直接。
对 $y_i = \gamma\hat{x_i}+\beta$，可推出 $\frac{\partial l}{\partial \hat{x_i}} = \frac{\partial l}{\partial y_i} \cdot \gamma$。


### 步骤 2

对 $\frac{\partial l}{\partial \sigma^2}$，要考虑在 $\hat{x}$ 中存在许多 $\hat{x_i}$。
这些值中的每一个都各自依赖 $\sigma^2$。
“从 $\sigma^2$ 指向 $\hat{x}$ 的箭头有很多。”

这就是为什么我们要对所有 $i$ 的 $\hat{x_i}$ 求和。

$$\frac{\partial l}{\partial \sigma^2} = \sum_{i}\left( \frac{\partial l}{\partial \hat{x_i}} \cdot \frac{\partial \hat{x_i}}{\partial \sigma^2}\right) $$

把它算出来，得到：

$$= \gamma \cdot \sum_i \frac{\partial}{\partial \sigma^2}\begin{bmatrix}(x_i - \mu)(\sigma^2 + \epsilon)^{-1/2}\end{bmatrix} \cdot \frac{\partial l}{\partial y_i}\\
= \frac{1}{2} \gamma \sum_i (x_i - \mu)(\sigma^2+\epsilon)^{-3/2} \cdot \frac{\partial l}{\partial y_i}$$


### 步骤 3

$\frac{\partial l}{\partial \mu}$ 是什么？
$\mu$ 与 $\hat{x}$ 之间的关系是 $32$ 重（就像之前 $\sigma^2$ 与 $\hat{x}$ 之间那样）。
$\mu$ 与 $\sigma^2$ 之间的关系是 $1$ 重，因为 $\sigma^2$ 只是一个标量（*见上文*）。

所有这 $33$ 个汇入的梯度都要在 $\mu$ 内加总。

$$\frac{\partial l}{\partial \mu} = \sum_i \left( \frac{\partial l}{\partial \hat{x}} \cdot \frac{\partial \hat{x_i}}{\partial \sigma^2}\right) + \left( \frac{\partial l}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu}\right)$$

第一部分是 $32$ 重关系。
附加的第二部分是 $1$ 重关系。

### 步骤 4

现在，对 $x$ 中的每个 $x_i$，会发出三支箭头：
- 一支指向 $\mu$
- 一支指向 $\sigma^2$
- 一支指向 $\hat{x}$ 中的*每一个单独的* $\hat{x_i}$

```python
# 反向传播

# 之前我们有：
# dbnraw = bngain * dhpreact
# dbndiff = bnvar_inv * dbnraw
# dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
# dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
# dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
# dbndiff += (2*bndiff) * dbndiff2
# dhprebn = dbndiff.clone()
# dbnmeani = (-dbndiff).sum(0)
# dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

# 给定 dhpreact 计算 dhprebn（即对批归一化做反向传播）
# （你还需要用到上面前向传播中的一些变量）

# 要为上述最终方程写出这段 python 代码并不容易
dhprebn = bngain * bnvar_inv / n * (n * dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
cmp('hprebn', dhprebn, hprebn) # 我只能让 approximate 为 true，maxdiff 为 9e-10
```

```
hprebn          | exact: False | approximate: True  | maxdiff: 9.313225746154785e-10
```

```python
# 练习 4：把所有东西整合到一起！
# 用你自己的反向传播训练这个 MLP 神经网络

# 初始化
n_embd = 10 # 字符嵌入向量的维度
n_hidden = 200 # MLP 隐藏层的神经元数量

g = torch.Generator().manual_seed(2147483647) # 用于可复现性
C  = torch.randn((vocab_size, n_embd),            generator=g)
# 第 1 层
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
# 第 2 层
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# 批归一化参数
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # 总参数数
for p in parameters:
  p.requires_grad = True

# 与上次相同的优化
max_steps = 200000
batch_size = 32
n = batch_size # 图个方便
lossi = []

# 一旦你写好了反向传播，就用这个上下文管理器提升效率（TODO）
with torch.no_grad():
  # 启动优化
  for i in range(max_steps):

    # 构造小批量
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y

    # 前向传播
    emb = C[Xb] # 把字符嵌入为向量
    embcat = emb.view(emb.shape[0], -1) # 拼接向量
    # 线性层
    hprebn = embcat @ W1 + b1 # 隐藏层预激活
    # 批归一化层
    # -------------------------------------------------------------
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # -------------------------------------------------------------
    # 非线性
    h = torch.tanh(hpreact) # 隐藏层
    logits = h @ W2 + b2 # 输出层
    loss = F.cross_entropy(logits, Yb) # 损失函数

    # 反向传播
    for p in parameters:
      p.grad = None
    # loss.backward() # 用它做正确性对比，之后删掉！

    # 手工反向传播！#swole_doge meme
    # -----------------
    dlogits = F.softmax(logits, dim=1) # 沿 logits 的行做 Softmax
    dlogits[range(n), Yb] -= 1 # 在 dlogits 中正确的位置处，总是需要减去一个 1
    dlogits /= n               # 由于取了均值，梯度按 n 缩小
    # 第 2 个线性层
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    # Tanh
    dhpreact = (1.0 - h**2) * dh
    # 批归一化反向传播
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = bngain * bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
    # 第 1 个线性层
    dembcat = dhprebn @ W1.T
    dW1 = dembcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    # 嵌入层
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
      for j in range(Xb.shape[1]):
        ix = Xb[k,j]
        dC[ix] += demb[k,j]
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
    # -----------------

    # 更新
    lr = 0.1 if i < 100000 else 0.01 # 学习率分步衰减
    for p, grad in zip(parameters, grads):
      #p.data += -lr * p.grad # 旧方式 cheems doge（使用来自 .backward() 的 PyTorch 梯度）
      p.data += -lr * grad # 新方式 swole doge TODO: 启用

    # 跟踪统计量
    if i % 10000 == 0: # 偶尔打印
      print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

    # 准备好训练完整网络时，把下面的提前退出注释掉
    # if i >= 100:
    #   break
```

```
12297
      0/ 200000: 3.7918
  10000/ 200000: 2.2118
  20000/ 200000: 2.4945
  30000/ 200000: 2.7126
  40000/ 200000: 2.1289
  50000/ 200000: 2.8434
  60000/ 200000: 2.5684
  70000/ 200000: 2.3247
  80000/ 200000: 2.5403
  90000/ 200000: 2.4744
 100000/ 200000: 2.7280
 110000/ 200000: 3.0945
 120000/ 200000: 2.7420
 130000/ 200000: 2.8820
 140000/ 200000: 2.8666
 150000/ 200000: 2.8361
 160000/ 200000: 2.8772
 170000/ 200000: 2.6707
 180000/ 200000: 2.7903
 190000/ 200000: 2.8445
```

```python
# 用于检查你的梯度

# for p,g in zip(parameters, grads):
#   cmp(str(tuple(p.shape)), g, p)
```

```python
# 在训练结束时校准批归一化

with torch.no_grad():
  # 把训练集传过网络
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 + b1
  # 在整个训练集上测量均值/标准差
  bnmean = hpreact.mean(0, keepdim=True)
  bnvar = hpreact.var(0, keepdim=True, unbiased=True)
```

```python
# 评估训练和验证损失

@torch.no_grad() # 这个装饰器关闭梯度跟踪
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
  hpreact = embcat @ W1 + b1
  hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
  h = torch.tanh(hpreact) # (N, n_hidden)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.8233625888824463
val 2.8218600749969482
```

| 损失     | 值          |
| -------- | ----------- |
| **训练** | 2.8233626   |
| **验证** | 2.8218601   |

接下来，我们从训练好的模型中采样输出：

```python
# 从模型采样
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # 用全 `.` 初始化
    while True:
      # 前向传播经过神经网络
      emb = C[torch.tensor([context])] # (1,block_size,d)      
      embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
      hpreact = embcat @ W1 + b1
      hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
      h = torch.tanh(hpreact) # (N, n_hidden)
      logits = h @ W2 + b2 # (N, vocab_size)
      # 采样
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
```

```
ernaaimyaahreelmnd.
ryalaretmrsjejdrleg.
adeeedieliihemy.
realekeiseananarneatzimhlkaa.
n.
sadbvrgahimies.
.
n.
jr.
eelklxnteuoanu.
amnedar.
yirle.
ehs.
laajaysknyaaahya.
nalyaisun.
zajelveuren.
.
.
t.
nsveaoec.
```

<center>本笔记本由 <a href="https://github.com/mk2112" target="_blank">mk2112</a> 编写。</center>
