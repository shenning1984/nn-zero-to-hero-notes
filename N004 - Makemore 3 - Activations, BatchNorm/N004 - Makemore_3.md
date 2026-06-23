# Makemore 3

[视频](https://www.youtube.com/watch?v=P6sfmUTpUmc)
[代码仓库](https://github.com/karpathy/makemore)
[Eureka Labs Discord](https://discord.com/invite/3zy8kqD9Cp)

## 目录

- [目标与意图](#目标与意图)
- [准备数据集](#准备数据集)
- [问题 1：初始损失高得不真实](#问题-1初始损失高得不真实)
- [问题 2：频繁出现 Tanh 极值](#问题-2频繁出现-tanh-极值)
  - [为什么 hpreact 暴露了问题？](#为什么-hpreact-暴露了问题)
- [问题 1+2：权重与偏置的缩放](#问题-12权重与偏置的缩放)
- [批归一化](#批归一化)
  - [批归一化的代价](#批归一化的代价)
- [题外话：卷积层](#题外话卷积层)
- [PyTorch 中的线性层](#pytorch-中的线性层)
- [PyTorch 中的批归一化](#pytorch-中的批归一化)
- [用 PyTorch 重建示例](#用-pytorch-重建示例)
  - [带 Tanh 且 gain 正确的 MLP 的激活分布](#带-tanh-且-gain-正确的-mlp-的激活分布)
  - [带 Tanh 但 gain 过小的 MLP 的激活分布](#带-tanh-但-gain-过小的-mlp-的激活分布)
  - [带 Tanh 且 gain 正确的 MLP 的梯度分布](#带-tanh-且-gain-正确的-mlp-的梯度分布)
  - [带 Tanh 但 gain 过小的 MLP 的梯度分布](#带-tanh-但-gain-过小的-mlp-的梯度分布)
  - [不带 Tanh 且 gain 过小的 MLP 的激活分布与梯度分布](#不带-tanh-且-gain-过小的-mlp-的激活分布与梯度分布)
  - [带 Tanh 且 gain 正确的 MLP 的激活分布与梯度分布](#带-tanh-且-gain-正确的-mlp-的激活分布与梯度分布)
  - [带 Tanh、批归一化且关闭 gain 的 MLP 的激活分布与梯度分布](#带-tanh批归一化且关闭-gain-的-mlp-的激活分布与梯度分布)
- [附加内容（课程未涉及）](#附加内容课程未涉及)

## 目标与意图

在[上一讲](../N003%20-%20Makemore%202%20-%20MLP/N003%20-%20Makemore_2.ipynb)中，我们把 bigram 语言模型的想法从一个直接的、基于查找表的方法转变为一个可学习的、基于 MLP 的方法。我们依据的是 [\[Bengio 等人 2003\]](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 的思路。
乍看之下，我们的努力只是让 MLP 模型达到了*与之相近的性能*。
但它也给我们带来了*大得多的灵活性*。
通过对超参数的调整、模型的扩展以及更多的训练，我们最终能够*轻松而又显著地*超过基于查找表的方法的性能。

我们将继续沿用基于 MLP 的方法，并具体考察梯度*如何*流动、它们*长什么样*，以及可能出现*哪些潜在问题*。

**目标：**
- **介绍批归一化（Batch Normalization）** 及其在稳定深度神经网络训练中的作用
- 进一步“PyTorch 化”并模块化我们的神经网络（NN）代码
- **介绍神经网络数值的诊断与解读**

**明确*不是*我们的目标：**
- 对神经网络做真正彻底的性能优化
- 对反向传播的深入解释（使这些诊断有意义的那些关系）（这将在[下一讲](../N005%20-%20Makemore%204%20-%20Backprop%20Ninja/N005%20-%20Makemore_4.ipynb)中讨论）

**在自然语言处理（NLP）的某些方面，我们这一讲其实已经触及了当前研究的前沿。**
语言任务的最优初始化和反向传播这个问题，目前尚未被解决。

> 这一讲会给你一些工具，让你判断自己的神经网络训练设置是否走在正确的轨道上。

**剧透：**
在反向传播过程中，激活值和梯度之间可能会相互配合得不好。
这会引发问题，并对 MLP 的学习过程产生负面影响，甚至使其停滞。
两个这样的主要问题是*梯度消失*或*梯度爆炸*。
[RNN](https://apps.dtic.mil/sti/tr/pdf/ADA164453.pdf)、[GRU](https://d2l.ai/chapter_recurrent-modern/gru.html) 和 [Transformer](https://arxiv.org/abs/1706.03762) 等架构正是为了越来越成功地应对这些训练困难而设计的。

## 准备数据集

```python
# 导入和之前一样
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline
```

这段起步代码只是从[上一份笔记本](../N003%20-%20Makemore%202%20-%20MLP/N003%20-%20Makemore_2.ipynb)复制过来的：

```python
words = open('../names.txt', 'r').read().splitlines() # 读取所有名字
print(words[:8])           # 展示前八个名字
print(len(words), 'words') # 数据集中名字的总数
```

```
['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia']
32033 words
```

我们再次从 `names.txt` 文本文件中加载数据并做校验。
然后我们梳理出实际出现的字符词表，并为每个字符关联一个唯一的索引。
最后，我们创建 `stoi` 和 `itos` 字典，以便在字符与其索引之间来回转换。

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
print(vocab_size)
```

```
{1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}
{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
27
```

接下来，我们把数据打包成输入张量和目标张量的配对。
输入张量包含由前导字符索引组成的 $3$ 元组，而目标张量则包含紧随该序列之后的下一个字符的索引。
我们打乱这些数据（但保持配对不散），并按 80/10/10 的训练/验证/测试比例把全部数据划分为训练集、验证集和测试集。

```python
# 构建数据集
block_size = 3 # 上下文长度：用多少个字符来预测下一个？

def build_dataset(words, split):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # 裁剪并追加

  X, Y = torch.tensor(X), torch.tensor(Y)
  print(f"{split} {X.shape}, {Y.shape}")
  return X, Y

# 随机化数据集（可复现）
import random
random.seed(42)
random.shuffle(words)

# 这些是我们用来划分数据集的“标记”
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))

# 将数据集划分为训练、验证和测试集
Xtr, Ytr = build_dataset(words[:n1], 'train:\t   ') # 80%
Xdev, Ydev = build_dataset(words[n1:n2], 'validation:') # 10%
Xte, Yte = build_dataset(words[n2:], 'test:\t   ')  # 10%
```

```
train:	    torch.Size([182625, 3]), torch.Size([182625])
validation: torch.Size([22655, 3]), torch.Size([22655])
test:	    torch.Size([22866, 3]), torch.Size([22866])
```

注意，我们*立刻*就把数据集划分成了训练、验证和测试子集。

>**问：为什么立刻划分数据集很重要？**
>**答：** 如果你推迟划分数据集，就有信息泄露的风险。假设你在处理一个尚未归一化的数值数据集。如果在划分之前就用整个数据集计算归一化参数（例如均值），那么训练集、验证集和测试集就变得相互依赖了，因为现在测试集通过均值间接地影响了训练数据。这会导致产生过于乐观的性能估计。更糟的是，训练好的模型在真正未见的数据上可能泛化得很差。**永远永远先划分数据！**

现在我们来搭建 MLP。尺寸和属性都和之前一样。
我们也顺便用随机值初始化模型的权重和偏置：

```python
n_embd = 10    # 字符嵌入向量的维度
n_hidden = 200 # MLP 隐藏层的神经元数量

g  = torch.Generator().manual_seed(2147483647)                 # 用于可复现性
C  = torch.randn((vocab_size, n_embd), generator=g)            # 27 个字母 x 10 个值，输入层
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) # 10 个值 x 3 个前导 x 200 个神经元
b1 = torch.randn(n_hidden, generator=g)                        # 200 个神经元的 200 个偏置
W2 = torch.randn((n_hidden, vocab_size), generator=g)          # 200 个输入 x 27 个输出神经元
b2 = torch.randn(vocab_size, generator=g)                      # 27 个输出偏置

parameters = [C, W1, b1, W2, b2] # 把所有参数聚集到一个结构中
print(sum(p.nelement() for p in parameters), 'parameters') # 总参数数（之前是 3481）

for p in parameters:
    p.requires_grad = True
```

```
11897 parameters
```

优化过程也和[之前](../N003%20-%20Makemore%202%20-%20MLP/N003%20-%20Makemore_2.ipynb)完全一样，只是稍微清理了一下。

```python
max_steps = 200000 # 训练多少个批次
batch_size = 32    # 一次处理多少个样本
lossi = []         # 每个批次的损失历史，供稍后绘图

for i in range(max_steps):
    ## 小批量构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    ## 前向传播
    emb = C[Xb]                         # 把字符嵌入为向量
    embcat = emb.view(emb.shape[0], -1) # 拼接向量
    hpreact = embcat @ W1 + b1          # 隐藏层预激活
    h = torch.tanh(hpreact)             # 隐藏层
    logits = h @ W2 + b2                # 输出层
    loss = F.cross_entropy(logits, Yb)  # 损失函数
    
    ## 反向传播，计算梯度
    for p in parameters:
        p.grad = None # 确保上一轮的梯度已清除
    loss.backward()   # 这会计算当前（新的）梯度
    
    ## 参数更新，把缩放后的梯度应用到参数上
    lr = 0.1 if i < (max_steps / 2) else 0.01 # 学习率衰减
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 27.8817
  10000/ 200000: 2.9571
  20000/ 200000: 2.5722
  30000/ 200000: 2.7646
  40000/ 200000: 2.0305
  50000/ 200000: 2.5794
  60000/ 200000: 2.3242
  70000/ 200000: 2.1174
  80000/ 200000: 2.2993
  90000/ 200000: 2.2481
 100000/ 200000: 2.0146
 110000/ 200000: 2.4534
 120000/ 200000: 1.9164
 130000/ 200000: 2.3719
 140000/ 200000: 2.2377
 150000/ 200000: 2.1441
 160000/ 200000: 2.2661
 170000/ 200000: 1.7855
 180000/ 200000: 2.1126
 190000/ 200000: 1.8004
```

```python
plt.plot(lossi);
```

[上次](../N003%20-%20Makemore%202%20-%20MLP/N003%20-%20Makemore_2.ipynb)我们已经评估过模型在训练集和验证集上的损失。
这里我们做的完全一样，只是整洁地封装成了一个函数：

```python
# 见 Makemore #2
# 我们之前做过，这里为了方便而重构：
@torch.no_grad() # 装饰器，关闭梯度跟踪（torch 一侧不再做“记账”）
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]    # 这是一个 switch（没见过这么用的，挺巧妙！）
    
    emb = C[x] # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    h = torch.tanh(embcat @ W1 + b1)    # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)   # 损失函数
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.1270017623901367
val 2.169905662536621
```

| 损失      | 原始       |
| --------- | ---------- |
| **训练**  | 2.1270018  |
| **验证**  | 2.1699057  |

接下来，我们从训练好的模型中采样输出：

```python
# 从模型采样
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # 用全 `.` 初始化
    while True:
      # 前向传播经过神经网络
      emb = C[torch.tensor([context])] # (1,block_size,n_embd)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      # 从分布中采样
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # 滑动上下文窗口并记录采样
      context = context[1:] + [ix]
      out.append(ix)
      # 一旦采样到特殊字符 '.' 就跳出
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # 解码并打印生成的词
```

```
mona.
mayah.
seel.
nah.
yam.
ren.
ruchadrael.
adeero.
elin.
shy.
jen.
eden.
eson.
arleigh.
malkia.
noshubergihamies.
kendretzy.
panthona.
uszayven.
kylene.
```

我们发现这比最初那个直接基于查找表的 Bigram 模型要好得多。**而且我们还能让它表现得更好。**
**这一讲就是通过逐个解决仍存在的问题来进一步优化我们的 MLP。**

## 问题 1：初始损失高得不真实

**网络的初始化有点问题。**
训练开始时 `loss` 太高了。虽然它很快又很显著地下降，但我们为什么要以这种方式开始训练呢？
我们得在最开始浪费资源把损失砸到一个合理的水平，然后才能真正开始精细地逼近最优模型。

**这很低效。**

> 在训练神经网络时，你几乎总能大致知道初始化时可以期望什么样的损失。
> 初始损失取决于损失函数和问题的设定。

实际上，我们很清楚新初始化的模型 `loss` 应该是多少：
$27$ 个字符中的任何一个都可能是下一个字符。
一个完美初始化的模型应该让我们能随机选择这下一个字符，而不带任何学到的经验/偏置。
初始的模型状态应该分配**均匀的似然**。
选中正确字符的机会应当与选中任何某个不同的错误字符的机会相等。
这使得选中任意下一个字符的似然为 $\frac{1}{27}$。

上面，我们用**负对数似然**来表示模型对一个批次的 `loss`：
$$\text{loss} = -\frac{1}{\text{batch\_size}} \sum_{i=1}^{\text{batch\_size}} \log \left( \frac{e^{\text{logits}_{i,\ y_i}}}{\sum_{k=1}^{27} e^{\text{logits}_{i,\ k}}}\right)$$

如果模型现在对所有 $27$ 个字符产生相等的 logits，那么 softmax 会为每个字符给出 $\frac{1}{27}$ 的概率。
由于求和中的每一项因此都变成 $\log (\frac{1}{27})$，计算负对数似然的批次平均就得到：
$$-\log \left(\frac{1}{27} \right) \approx 3.3$$

```python
-torch.tensor(1/27.0).log()
```

```
tensor(3.2958)
```

照此衡量，我们应当期望 $3.3$ 是未训练、刚初始化的模型的初始损失。

>**问：等等，为什么不直接用 $\frac{1}{27}$ 的似然作为损失？为什么要引入负对数似然？**
>**答：** 我们把似然“包裹”在一个负对数里，以获得一个在优化时更易处理的损失值。正因如此，负对数似然是分类任务中损失函数的常见选择。像这样“包裹”似然，是因为它在后续优化中充当一个有用的载体。

>**问：那为什么我们模型的初始损失 $27.8817$ 偏离这个 $3.3$ 的期望将近一个数量级？**
>**答：** 神经网络并不是以构成均匀分布的参数开始的。相反，当前的神经网络不仅由随机参数值组成，还由这些参数内随机表示的概率分布组成。而这些分布绝不被保证是均匀的。

神经网络对“所有 $27$ 个输出一开始都同等可能”这个概念*一无所知*。初始化可能让某些输出被随机地偏袒，因为权重中表达的随机分布扭曲了完美的均匀性。所以，目前单个字母有时纯粹出于偶然地比其他字母明显更可能被选中，把损失推得很高。

> 换句话说，目前神经网络是随机地、但又非常自信地错了。这导致初始损失非常高。

下面是同一问题的另一个例子：

```python
# 这个问题的 4 维示例（而不是 27 维）
logits = torch.tensor([-8.0, 5.0, 1.0, 10.0]) # 一个 MLP 的“输出”
probs = torch.softmax(logits, dim=0)          # 把 logits 转成归一化的概率分布
loss = -probs[2].log()                        # 这种情况下标签为 2 的损失

probs, loss
```

```
(tensor([1.5126e-08, 6.6920e-03, 1.2257e-04, 9.9319e-01]), tensor(9.0068))
```

由于允许 logits 一开始就呈现偶然的分布，
神经网络不太可能快速学到一组能降低损失的好 logits 组合。
首先，这些分布必须通过训练克服，然后才搭好舞台去真正形成一组好的 logits。

我们需要避免极端的偶然输出分布。换句话说，**我们希望初始化神经网络时 logits 彼此相当接近。** 理论上，logits 可以选择围绕任意数字聚拢。但默认地，我们把权重初始化为紧密聚集在 $0$ 这个中心周围。这样，我们把初始化紧密放在正态分布的范围内。

**问题：** 我们如何让输出层的 logits 一开始就接近 $0$？

**答案：** 我们把 `W2` 缩放到接近 $0$，并通过把 `b2` 设为 $0$ 来关掉它。

```python
n_embd = 10
n_hidden = 200

g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                        generator=g)
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0 # 初始化为 0

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True
```

现在输出层的初始化被压缩到一个非常接近 $0$ 的范围内。
而对 `b2`，所有值实际上都被初始化为 $0$ *（有点难看，但它奏效。）*
这有效地消除了 logits 过高的方差以及概率非常错误的初始分布。

**这现在如何影响神经网络的性能？**

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    hpreact = embcat @ W1 + b1          
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 3.3221
  10000/ 200000: 2.1900
  20000/ 200000: 2.4196
  30000/ 200000: 2.6067
  40000/ 200000: 2.0601
  50000/ 200000: 2.4988
  60000/ 200000: 2.3902
  70000/ 200000: 2.1344
  80000/ 200000: 2.3369
  90000/ 200000: 2.1299
 100000/ 200000: 1.8329
 110000/ 200000: 2.2053
 120000/ 200000: 1.8540
 130000/ 200000: 2.4566
 140000/ 200000: 2.1879
 150000/ 200000: 2.1118
 160000/ 200000: 1.8956
 170000/ 200000: 1.8645
 180000/ 200000: 2.0326
 190000/ 200000: 1.8417
```

```python
plt.plot(lossi);
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]                            # switch 语句
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    h = torch.tanh(embcat @ W1 + b1)    # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.069589138031006
val 2.1310746669769287
```

| 损失      | 原始       | 已解决问题 (1)     |
| --------- | ---------- | ------------------ |
| **训练**  | 2.1270018  | **2.0695891**      |
| **验证**  | 2.1699057  | **2.1310747**      |

损失曲线不再像曲棍球杆了。**我们不再有这个高初始损失。**

> 现在这个曲棍球杆问题解决了，训练过程把更多时间花在真正的优化上，而不是去砸平那些不真实的 logits。

**所以三件好事是：**
1. 损失看起来像我们在问题情境中期望的那样
2. 训练期间的损失曲线看起来更平稳了。
3. 训练损失和验证损失都下降了一些。

有趣的是，我们毫不犹豫地把 `b2` 设为 $0$，却没有对 `W2` 做同样的事。**为什么？**
- 我们想保留权重中的随机性，因此它们必须有某种值
- 值为 $0$ 的权重会导致梯度为 none，而偏置只是叠加在上面；值为 $0$ 的权重因此对训练过程不利

## 问题 2：频繁出现 Tanh 极值

除了**问题 1**，神经网络内部还有一个更深层的问题。
logits/输出现在没问题了，但 `h`，即隐藏层激活值组成的向量，有问题。

$\text{tanh}$ 把任意值压缩到 $[-1; 1]$ 范围内。我们来看看 `h`：

```python
print(h.shape)
print(h.view(-1).shape)
```

```
torch.Size([32, 200])
torch.Size([6400])
```

```python
print(h, "\n") # 隐藏层激活 (32 x 200)
plt.hist(h.view(-1).tolist(), 50); # (6400 x 1)，把张量转为列表，对 50 个 bin 展示分布直方图
```

```
tensor([[ 0.7100, -0.7878, -0.7424,  ..., -1.0000,  1.0000,  1.0000],
        [-0.5615, -1.0000, -1.0000,  ..., -0.8078,  0.9971, -0.9209],
        [-1.0000,  0.9940, -1.0000,  ...,  0.3850, -0.9303,  0.3262],
        ...,
        [-0.9992, -1.0000, -0.9999,  ..., -0.4351, -0.8976, -0.3768],
        [ 0.9939,  0.8976,  1.0000,  ..., -0.9538, -1.0000,  0.9987],
        [-1.0000, -0.9995, -1.0000,  ..., -0.6407, -0.8208,  0.5101]],
       grad_fn=<TanhBackward0>)
```

看起来绝大多数值都是 $\text{tanh}$ 的极值。

**这是为什么？**

我们需要回溯 `h`。
`h` 基于 `hpreact`，`hpreact` 怎么样？

```python
plt.hist(hpreact.view(-1).tolist(), 50);
```

我们看到 `hpreact` 内的值大致在 $[-23; 21]$ 范围内。
$\text{tanh}$ 必须把所有这些都压缩到 $[-1; 1]$ 范围内。

### 为什么 hpreact 暴露了问题？

上面这种 `hpreact` 呈窄分布的场景，会在反向传播时反过来给我们制造麻烦。
反向传播时，我们“倒着”走过 MLP 的各个运算和值。
我们也会对构成 `h` 的 $\text{tanh}$ 函数这样做。
在我们的 MLP 中，$\text{tanh}$ 总共对 $200$ 个激活进行计算。

$\text{tanh} = \frac{e^{2x} - 1}{e^{2x} + 1}$ 的导数是 $\left(1 - t^2\right)$。
通过链式法则拼接后，下面就是 $\text{tanh}$ 的梯度：

```python
self.grad += (1 - t ** 2) * out.grad
```

那么，如果我们存储在 `h` 中的 $\text{tanh}$ 输出 `t` 是 $-1$ 或 $1$，会发生什么？
**没错。这是个问题。** 梯度变成 $0$。

> 无论梯度如何，只要 $\text{tanh}$ 的结果是 $-1$ 或 $1$，穿过 $\text{tanh}$ 的反向传播就会把梯度狠狠压平，从而极大地削弱梯度学习的效果。
> **梯度变成 $0$，扼杀任何学习进展。梯度消失了。**

因为 $\text{tanh}$ 如此频繁地如此极端，我们在隐藏层里只能学得极其缓慢，或者经常什么也学不到，因为梯度如此频繁地反向传播为 $0$。
我们可以用下面的代码展示期望的学习效果有多少次被极端的 $\text{tanh}$ 值（以及随之而来的 $0$ 梯度）所抑制：

```python
plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest'); # true: 白，false: 黑（h 是 32 x 200）
```

> 如果有一列完全是白色的，那么它所代表的神经元就**永远**学不到东西。**它会是一个“死神经元”。**

幸运的是，我们在这里看不到这种情况，尽管在某些情况下已经很接近了。
这种“梯度砸平现象”不仅在 $\text{tanh}$ 中能观察到，在 `sigmoid`、`ReLU` 和 `ELU` 中也一样，也就是说在所有在某处变平的激活函数中都能看到。

*函数越平，导数（变化率）越小，梯度越小，能有效发生的学习就越少。*

![](https://miro.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png)
来源：[Shruti Jadon - Medium](https://medium.com/@shrutijadon/survey-on-activation-functions-for-deep-learning-9689331ba092)

> 死神经元可能由初始化造成，也可能由极其偏向边界的 $\text{tanh}$ 造成，使受影响的神经元的学习变得非常非常困难。*这好比脑损伤。*

**简而言之：** 我们**紧急**需要解决 $\text{tanh}$ 砸平梯度的问题！

对于高得不真实的初始损失，我们已经看到可以通过缩放权重和偏置来平抑（至少是初始的）分布。
由于上面的图显示问题在隐藏层中普遍存在，我们现在也可以尝试把 `W1` 和 `b1` 压缩到一个接近 $0$ 的范围。

这样进入 $\text{tanh}$ 的 `hpreact` 输入就会更平缓。
$\text{tanh}$ 就不再倾向于走向极端。

**值得一试！**

```python
n_embd = 10
n_hidden = 200

g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2  # 小，但不为 0
b1 = torch.randn(n_hidden,                        generator=g) * 0.01 # 小，但不为 0 的偏置
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0    # 偏置现在全为 0

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True
```

我们继续再次训练模型，但只跑一个批次：

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    hpreact = embcat @ W1 + b1          
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
    
    break # 只跑第一个批次
```

```
      0/ 200000: 3.3135
```

```python
# 我们其实允许一些值达到极值，但现在很少见了
plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99 , cmap='gray', interpolation='nearest'); # true: 白，false: 黑
```

**这看起来好多了！**
`h` 的分布现在平缓得多，极值仍触手可及，但被触达的频率低得多。
接下来，我们看看 `h` 的分布：

```python
print(h, "\n")
plt.hist(h.view(-1).tolist(), 50);
```

```
tensor([[ 0.5503, -0.1064, -0.6658,  ..., -0.3477, -0.9756,  0.8880],
        [-0.9081, -0.1924, -0.1833,  ...,  0.0494,  0.4942,  0.4397],
        [ 0.8016,  0.1173,  0.8237,  ...,  0.2890,  0.6476,  0.8827],
        ...,
        [-0.9190,  0.5208, -0.0346,  ..., -0.0830,  0.8660,  0.8849],
        [-0.9362,  0.0930, -0.2810,  ..., -0.1260,  0.7874,  0.9102],
        [-0.9190,  0.5208, -0.0346,  ..., -0.0830,  0.8660,  0.8849]],
       grad_fn=<TanhBackward0>)
```

```python
plt.hist(hpreact.view(-1).tolist(), 50);
```

解决了这个问题之后，我们继续完整地训练这个新版本的模型：

```python
n_embd = 10
n_hidden = 200

g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0    # 偏置现在全为 0

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True
```

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    hpreact = embcat @ W1 + b1          
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 3.3135
  10000/ 200000: 2.1648
  20000/ 200000: 2.3061
  30000/ 200000: 2.4541
  40000/ 200000: 1.9787
  50000/ 200000: 2.2930
  60000/ 200000: 2.4232
  70000/ 200000: 2.0680
  80000/ 200000: 2.3095
  90000/ 200000: 2.1207
 100000/ 200000: 1.8269
 110000/ 200000: 2.2045
 120000/ 200000: 1.9797
 130000/ 200000: 2.3946
 140000/ 200000: 2.1000
 150000/ 200000: 2.1948
 160000/ 200000: 1.8619
 170000/ 200000: 1.7809
 180000/ 200000: 1.9673
 190000/ 200000: 1.8295
```

```python
plt.plot(lossi);
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    h = torch.tanh(embcat @ W1 + b1)    # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.0355966091156006
val 2.1026785373687744
```

这些是我们到目前为止见过的损失：

| 损失      | 原始       | 已解决问题 (1)     | 已解决问题 (2)     |
| --------- | ---------- | ------------------ | ------------------ |
| **训练**  | 2.1270018  | 2.0695891          | **2.0355966**      |
| **验证**  | 2.1699057  | 2.1310747          | **2.1026785**      |

>**有进展！**
>随着这些问题被解决，我们的训练变得更聚焦、更有产出。
## 问题 1+2：权重与偏置的缩放

我们之前的优化对于示例 MLP 来说还相对可控。
我们这里只有一个很扁平的 MLP。
MLP 越大，优化就越困难。

嗯……我们现在有了一大堆“魔法数字”，即权重和偏置的因子：

```
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0
```

**对于大型多层 MLP，你如何确定并设置这些值？**

为了便于说明，假设我们用一层，权重和输入都均匀分布。
不施加激活函数，也不加偏置。

```python
x = torch.randn(1000, 10) # 1000 个向量，每个 10 维（输入）（高斯分布）
w = torch.randn(10, 200)  # 10 个向量，每个 200 维（200 个神经元，每个看 10 个输入）

# [!] 这个示例中我们略去偏置 b

y = x @ w # 得到神经元的预激活

print('X:', x.mean(), x.std())  # 均值 ~0，标准差 ~1（因为是高斯分布）
print('W:', w.mean(), w.std())  # 均值 ~0，标准差 ~1（也是高斯分布）
print('Y:', y.mean(), y.std())  # 均值 ~0，标准差 ~3 ?!

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('X')
plt.hist(x.view(-1).tolist(), 50, density=True);
plt.subplot(122)
plt.title('Y')
plt.hist(y.view(-1).tolist(), 50, density=True);
```

```
X: tensor(-0.0089) tensor(1.0081)
W: tensor(-0.0199) tensor(0.9826)
Y: tensor(0.0088) tensor(3.1328)
```

预激活现在有了*非*高斯的标准差。这是通过逐元素相乘发生的，**但我们不想要这样**。
正态分布理想情况下应被保持，因为一个被压平的分布会导致越来越多的极值出现，这反过来又导致梯度越来越频繁地变得极端。

**我们的问题变成：如何缩放 `w`，才能让 `x @ w` 之后标准差仍为 ~1？**

我们通过缩放 `w` 来获得杠杆。如果乘以例如 $5$，标准差会进一步增大。
如果乘以例如 $\frac{1}{5}$，标准差会减小。

数学上，给定权重 `w` 时正确的因子是：$\frac{1}{\sqrt{\text{w.shape}[0]}}$，其中 **`w.shape[0]` 被称为该层的 fan-in（扇入）**。

```python
x = torch.randn(1000, 10) 
w = torch.randn(10, 200) * 10**-0.5 # 即 / sqrt(10) 或 0.316228

y = x @ w

print('X:', x.mean(), x.std())
print('W:', w.mean(), w.std())
print('Y:', y.mean(), y.std())

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('X')
plt.hist(x.view(-1).tolist(), 50, density=True);
plt.subplot(122)
plt.title('Y')
plt.hist(y.view(-1).tolist(), 50, density=True);
```

```
X: tensor(0.0022) tensor(1.0037)
W: tensor(-0.0154) tensor(0.3140)
Y: tensor(-0.0016) tensor(0.9962)
```

有几篇论文研究了如何最好地初始化 MLP 的问题。
这些论文探讨的是如何最好地设置/初始化权重（并针对不同的 MLP 尺度缩放它们），使它们在学习过程中不会变得无穷小或无比巨大。

[这篇论文 \[Kaiming He 等人 2015\]](https://arxiv.org/abs/1502.01852) 针对 CNN 以及 ReLU 和 PReLU 非线性深入考察了这个问题。
*（提醒一下，它并不算轻松读物。）*

**但是：** 论文中描述的初始化技术在 PyTorch 中已经实现 —— `torch.nn.init.kaiming_normal_`。**这大概是在 PyTorch 中初始化权重最常用的方式。**

在这篇论文发表之前，深度 MLP 相当脆弱。
但这篇论文开启了一股“稳定化潮流”，由此涌现出各种各样的优化和稳定化技术，例如：

- 残差连接（Residual Connections），
- 批归一化（Batch-Normalization），
- 层归一化（Layer-Normalization），
- Adam 优化器（以及其他优化器使精确的初始化变得没那么重要）

论文指出，我们必须把权重乘以一个因子，以重新获得期望的标准差。（另见[这里](https://pytorch.org/docs/stable/nn.init.html)）

![](./img/kaiming_normal.PNG)

按照论文，$\text{gain}$ 的值要根据激活函数来选择。
在下表中，我们可以看到，对于我们上面的**线性方法**（完全没有激活函数），$\text{gain}$ 为 $1$，这正是我们已经用 $\frac{1}{\sqrt{\text{fan-in}}}$ 应用的：

![](./img/gains.PNG)
来源：[PyTorch 文档](https://pytorch.org/docs/stable/nn.init.html)

如果我们现在把这种用于 $\text{tanh}$ 激活的初始化技术应用到一个更整洁的 MLP 上，就得到下面的代码：

```python
n_embd = 10
n_hidden = 200

g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5) # 这个因子使输出的标准差 ~1
b1 = torch.randn(n_hidden,                        generator=g) * 0.01 # 小，但不为 0
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0    # 偏置现在全为 0

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True
```

现在我们也优化了权重的值初始化，当然必须再训练一次：

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    hpreact = embcat @ W1 + b1          
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 3.3179
  10000/ 200000: 2.1910
  20000/ 200000: 2.3270
  30000/ 200000: 2.5396
  40000/ 200000: 1.9468
  50000/ 200000: 2.3331
  60000/ 200000: 2.3852
  70000/ 200000: 2.1173
  80000/ 200000: 2.3159
  90000/ 200000: 2.2010
 100000/ 200000: 1.8591
 110000/ 200000: 2.0881
 120000/ 200000: 1.9389
 130000/ 200000: 2.3913
 140000/ 200000: 2.0949
 150000/ 200000: 2.1458
 160000/ 200000: 1.7824
 170000/ 200000: 1.7249
 180000/ 200000: 1.9752
 190000/ 200000: 1.8614
```

```python
plt.plot(lossi);
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    h = torch.tanh(embcat @ W1 + b1)    # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.0376641750335693
val 2.106989622116089
```

| 损失      | 原始       | 已解决问题 (1)     | 已解决问题 (2)     | 已解决问题 (1+2)    |
| --------- | ---------- | ------------------ | ------------------ | ------------------- |
| **训练**  | 2.1270018  | 2.0695891          | **2.0355966**      | 2.0376642           |
| **验证**  | 2.1699057  | 2.1310747          | **2.1026785**      | 2.1069896           |

解决了问题 $2$ 之后，我们大致回到了出发的地方。

**但这其实是件好事**，因为我们不再有缩放权重的“魔法数字”了。
我们只是计算出了现在正在使用的缩放因子。

## 批归一化

这个概念由 Ioffe 和 Szegedy 在[这篇论文](https://arxiv.org/abs/1502.03167)中提出，它使我们在上面所做的精确权重初始化几乎变得过时/无关紧要。

在我们的神经网络中，我们有隐藏状态 `hpreact`，我们用 $\text{tanh}$ 把它形成为激活值。
`hpreact` 应当（我们之前在 $\text{tanh}$ 优化时已经讨论过）不能太小也不能太大，所以理想情况下呈正态分布（标准差为 $1$）。

再往上，我们用了一种相当啰嗦的方法来实现这一点。
但现在想象一下我们没那么做，而只是随机地初始化了权重和偏置。

> **批归一化说：** 如果我们希望隐藏状态 `hpreact` 呈正态分布，我们可以就这样把它从当前分布变换/重整成正态分布。
> 这是一个可微的操作，所以反向传播对此毫无问题。这是允许的！

论文列出了批归一化如何工作的一种“配方”：
![](./img/batch_norm_recipe.PNG)

我们先来看 `hpreact`：

```python
print('hpreact shape:', hpreact.shape) # (32, 200)
```

```
hpreact shape: torch.Size([32, 200])
```

为了引入批归一化，首先，我们沿着 `hpreact` 的第 0 维（批次维）计算均值：

```python
# 步骤 1：沿第 0 维计算均值
# 0 表示我们沿 32 个输入计算均值，所以得到 200 个均值，每个神经元一个
print('mean:', hpreact.mean(0, keepdim=True).shape) # (1, 200)
```

```
mean: torch.Size([1, 200])
```

对于步骤 2，我们沿着 `hpreact` 的第 0 维计算标准差：

```python
# 步骤 2：沿第 0 维计算标准差
# 0 表示我们沿 32 个输入计算均值，所以得到 200 个标准差，每个神经元一个
print('stddev:', hpreact.std(0, keepdim=True).shape) # (1, 200)
```

```
stddev: torch.Size([1, 200])
```

有了这些工具，我们现在可以在步骤 3 中计算批归一化后的 `hpreact`：

```python
# 步骤 3：按配方所述归一化
hpreact = (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) # (32, 200)
print(hpreact.shape)
```

```
torch.Size([32, 200])
```

上面的运算对批次中 $32$ 个输入上的 `hpreact` 中 $200$ 个神经元的值进行归一化，确保神经元激活与正态分布所允许的强度对齐。

如果你把上面 `步骤 3` 中的 `hpreact` 原样用于前向传播，
即 `hpreact = (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True)`，MLP 的性能实际上会**更差**。
原因在于，我们其实只想要这些（受限地）呈正态分布的隐藏层激活**一次，即 MLP 初始化时**。

> 批归一化应当在初始化时通过归一化激活来为训练过程创造一个公平的起点，**而且只在初始化时**。
> 训练期间，激活应当完全且独占地由反向传播来针对输入进行调整。

这正是“配方”中的步骤 4 所允许的。
我们在步骤 4 中需要新的变量。我们需要一个缩放器 `bngain` 和一个偏移 `bnbias`。

**最好通过重新搭建 MLP 来展示：**

```python
n_embd = 10
n_hidden = 200

g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0    # 偏置现在全为 0

bngain = torch.ones((1, n_hidden))  # (1, 200)，每个隐藏神经元一个 1
bnbias = torch.zeros((1, n_hidden)) # (1, 200)，每个隐藏神经元一个偏置

parameters = [C, W1, b1, W2, b2, bngain, bnbias] # gain 和 bias 由反向传播改变!!

for p in parameters:
    p.requires_grad = True
```

初始化时，`bngain` 只由 1 组成，每个神经元一个。
`bnbias` 为每个神经元持有一个 0。我们第一次像这样把 `bngain` 和 `bnbias` 应用于 `hpreact`：

```python
hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
```

结果和之前*完全*一样，**尤其是因为 `bngain` 只由 1 组成、`bnbias` 只由 0 组成**。

> 诀窍在于，从初始化之后 `bngain` 和 `bnbias` 就会改变，因为我们把它们放进了 `parameters`。
> 这意味着——*正合所需*——训练期间 `hpreact` 不再有严格的正态分布被强制施加，
> 而是网络现在可以按自己的意愿调整 `bngain` 和 `bnbias`。
> 这为训练过程提供了一个对激活分布本身可调的抓手。

为了完成“配方”中的步骤 4，我们现在把 `bngain` 和 `bnbias` 应用到 `hpreact`：

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 3.3147
  10000/ 200000: 2.1984
  20000/ 200000: 2.3375
  30000/ 200000: 2.4359
  40000/ 200000: 2.0119
  50000/ 200000: 2.2595
  60000/ 200000: 2.4775
  70000/ 200000: 2.1020
  80000/ 200000: 2.2788
  90000/ 200000: 2.1862
 100000/ 200000: 1.9474
 110000/ 200000: 2.3010
 120000/ 200000: 1.9837
 130000/ 200000: 2.4523
 140000/ 200000: 2.3839
 150000/ 200000: 2.1987
 160000/ 200000: 1.9733
 170000/ 200000: 1.8668
 180000/ 200000: 1.9973
 190000/ 200000: 1.8347
```

```python
plt.plot(lossi);
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    # 这部分是新的，就像我们在新的批归一化神经网络中做的那样
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True) + bnbias
    
    h = torch.tanh(hpreact)     # (N, n_hidden)
    logits = h @ W2 + b2        # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.0668270587921143
val 2.104844808578491
```

| 损失      | 原始       | 已解决问题 (1)     | 已解决问题 (2)     | 已解决问题 (1+2)    | 批归一化   |
| --------- | ---------- | ------------------ | ------------------ | ------------------- | ---------- |
| **训练**  | 2.1270018  | 2.0695891          | **2.0355966**      | 2.0376642           | 2.0668271  |
| **验证**  | 2.1699057  | 2.1310747          | **2.1026785**      | 2.1069896           | 2.1048448  |

事实上，我们**不**期望批归一化给模型的性能/质量带来很大提升。
**这仅仅是因为我们的 MLP 太扁平了，批归一化起不到太大作用。**

> MLP 越深、越交织，就越难以操纵权重（就像我们上面在做批归一化之前用因子做的那样）。
> 但越是这种情况，批归一化就越划算。你不必在每一层都用批归一化，但应当在线性层（$W \cdot x+b$）或卷积层上用它。

### 批归一化的代价

**批归一化是有代价的，而且代价不低：**
到目前为止，我们在每一点上看到的是单个输入样本，它的激活是确定性地计算的，即由这单个输入唯一确定。
我们之前已经把训练分成输入批次。这是为了效率（随机梯度下降）在 [Makemore 2](../N003%20-%20Makemore%202%20-%20MLP/N003%20-%20Makemore_2.ipynb) 中做的，它也让训练过程更稳定。

**批归一化搅乱了这些想法。**
因为 `hpreact` 通过反向传播逐渐偏离正态分布，身处一个批次中现在有了意义。批次中的输入之间存在联系。

> logits 现在不再是“仅仅”单个输入的确定性结果；它们是输入**以及**训练批次中先前输入的结果，批归一化把后者“形变”成了当前状态。**批次算数，而不只是原子训练样本。**

这意味着如果你改变批次组成，然后给 MLP 同一个输入，logits 的值会随批次组成的不同而“抖动”。
这种*抖动*不仅出现在 `logits` 中，也出现在 `h` 中。

**这是好事吗？是的，其实是！**

由于这种与训练历史相关的“抖动”，MLP 现在有了天然的**熵**。
而这反过来又让过拟合更不容易发生。噪声扼杀过拟合。

![](./img/entropy.png)
来源：[Andre Ye - Towards Data Science](https://towardsdatascience.com/understanding-entropy-the-golden-measurement-of-machine-learning-4ea97c663dc3)

> 批归一化是一个非常好的正则化器，以至于你无法想象没有它该怎么训练。它稳定了训练并让过拟合更不容易发生。

**事实上，这还不是批归一化的唯一效果。**

**让我们在 MLP 的生命周期中再往前走一步：** 当我们部署一个 MLP 时，我们期望对*一个*输入得到*一个*输出/预测。
我们期望确定性的行为。

**问题：**
在激活的批归一化下，如果它评估先前输入的历史（生产环境里我们没有这些历史），并因此期望批次作为输入并对输出施加批次相关的影响，这怎么行得通？

**解决方案：**
在实际训练之后、上线之前，在批归一化层中一次性设定均值和标准差。
一次性地，在整个训练集上。固定下来，这有效地终止了批次依赖。

**我们上面已经训练了一个批归一化的 MLP。**
**我们现在与这个新的确定和性能评估过程衔接起来：**

```python
# 新增：在训练结束时校准批归一化
with torch.no_grad(): # 这让下面的所有运算更快，因为“记账”被关闭了
    # 把（整个）训练集前向推过
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    # 一次性测量整个训练集上的均值/标准差（把它们刻在石上）
    bnmean = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    # 这部分是新的，就像我们在新的批归一化神经网络中做的那样
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias # 固定它们，握住它们，技术上如此
    
    h = torch.tanh(hpreact)             # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

# 与之前相同的损失
split_loss('train')
split_loss('val')
```

```
train 2.0668270587921143
val 2.1049270629882812
```

我们得到了与直接测量批归一化模型相同的性能。
**这很好**，我们现在可以通过它来评估单个输入了。

然而，我们当前的过程——定义、训练、冻结 `bnmean` 和 `bnstd`、评估——在现实中其实并不是这样用的。
事实上，`bnmean` 和 `bnstd` 已经可以在 MLP 的训练**期间**被估计出来。

为此需要调整定义和训练过程。
知道这点有帮助：`hpreact` 在初始化时呈正态分布，所以均值会 $\sim 0$、标准差会 $\sim 1$。
我们新增两个变量 `bnmean_running` 用零初始化、`bnstd_running` 用 1 初始化，
以在训练期间跨批次保持 `hpreact` 的均值和标准差的滑动平均。对每个单独的步骤/批次，我们为该批次计算 `bnmeani` 和 `bnstdi`，然后在梯度计算之外用下面的公式更新 `bnmean_running` 和 `bnstd_running`：

```python
bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```

定义和训练过程现在长这样：

```python
n_embd = 10
n_hidden = 200

# 初始化
g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.01 
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0    # 偏置现在全为 0

bngain = torch.ones((1, n_hidden))  # 1 x 200，每个隐藏神经元一个 1
bnbias = torch.zeros((1, n_hidden)) # 1 x 200，每个隐藏神经元一个偏置

# 这两个不需要求梯度（不加入 'parameters'）
bnmean_running = torch.zeros((1, n_hidden)) # hpreact 起始为正态分布 -> 均值为 0
bnstd_running = torch.ones((1, n_hidden))   # hpreact 起始为正态分布 -> 标准差为 1

parameters = [C, W1, b1, W2, b2, bngain, bnbias] # gain 和 bias 由反向传播改变!!

for p in parameters:
    p.requires_grad = True
```

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    hpreact = embcat @ W1 + b1
    
    # 新增：均值和标准差的定义外包到这里
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    
    with torch.no_grad(): # 这与梯度更新无关
        # 沿当前均值和标准差方向累加小更新
        # 训练一结束就会累积形成一个总的均值和标准差
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    
    
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 3.3147
  10000/ 200000: 2.1984
  20000/ 200000: 2.3375
  30000/ 200000: 2.4359
  40000/ 200000: 2.0119
  50000/ 200000: 2.2595
  60000/ 200000: 2.4775
  70000/ 200000: 2.1020
  80000/ 200000: 2.2788
  90000/ 200000: 2.1862
 100000/ 200000: 1.9474
 110000/ 200000: 2.3010
 120000/ 200000: 1.9837
 130000/ 200000: 2.4523
 140000/ 200000: 2.3839
 150000/ 200000: 2.1987
 160000/ 200000: 1.9733
 170000/ 200000: 1.8668
 180000/ 200000: 1.9973
 190000/ 200000: 1.8347
```

```python
plt.plot(lossi);
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    # 这部分是新的，就像我们在新的批归一化神经网络中做的那样
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    
    h = torch.tanh(hpreact)             # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.06659197807312
val 2.1050572395324707
```

| 损失      | 原始       | 已解决问题 (1)     | 已解决问题 (2)     | 已解决问题 (1+2)    | 批归一化   |
| --------- | ---------- | ------------------ | ------------------ | ------------------- | ---------- |
| **训练**  | 2.1270018  | 2.0695891          | **2.0355966**      | 2.0376642           | 2.0668271  |
| **验证**  | 2.1699057  | 2.1310747          | **2.1026785**      | 2.1069896           | 2.1048448  |

`bnmean` 和 `bnmean_running` 以及 `bnstd` 和 `bnstd_running` **并不相同，但非常接近**。*这对我们来说够了。*
我们实际上跳过或合并了“冻结”步骤，不再需要它本身了。

实际上，我们当前的方法**仍然不完美，但以一种*非常*微妙的方式**。
目前，我们通过计算 `embcat @ W1 + b1` 来确定 `hpreact`。
然后我们为 `hpreact` 确定均值（所有当前神经元的所有预激活值一起的）并从 `hpreact` 中减去它。
所以我们加了 `b1`，有意地按每个神经元平移（所有神经元的）均值，但随后又从 `hpreact` 中减去这个均值，使该神经元得到保证为 $0$ 的均值。
因此通过 `b1` 的有意平移无法发生。

批归一化给了我们在实际归一化之后“照常扭曲”的机会，靠的是 `bnbias`。
换句话说，我们不再需要 `b1` 了，因为 `bnbias` 能做同样的工作。

**最后，给出一个不带 `b1` 的、更优的批归一化完整实现：**

```python
n_embd = 10
n_hidden = 200

# 初始化
g  = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
# b1 = torch.randn(n_hidden,                        generator=g) * 0.01 # 关掉，因为不再需要（bnbias 干这个活）
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01 # 小，但不为 0
b2 = torch.randn(vocab_size,                      generator=g) * 0    # 偏置现在全为 0

bngain = torch.ones((1, n_hidden))  # 1 x 200，每个隐藏神经元一个 1
bnbias = torch.zeros((1, n_hidden)) # 1 x 200，每个隐藏神经元一个偏置

# 这两个不需要求梯度（不加入 'parameters'）
bnmean_running = torch.zeros((1, n_hidden)) # hpreact 起始为正态分布 -> 均值为 0
bnstd_running = torch.ones((1, n_hidden))   # hpreact 起始为正态分布 -> 标准差为 1

parameters = [C, W1, W2, b2, bngain, bnbias] # gain 和 bias 由反向传播改变!!

for p in parameters:
    p.requires_grad = True
```

```python
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 本批次的 X,Y
    
    # 前向传播
    emb = C[Xb]                         
    embcat = emb.view(emb.shape[0], -1) 
    
    # 线性层
    hpreact = embcat @ W1 #+ b1
    
    # 批归一化层
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    with torch.no_grad(): # 这与梯度更新无关
        # 沿当前均值和标准差方向累加小更新
        # 训练一结束就会累积形成一个总的均值和标准差
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    # 非线性
    h = torch.tanh(hpreact)             
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 更新
    lr = 0.1 if i < (max_steps / 2) else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # 偶尔打印跟踪统计量
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    
    # 把当前损失值加入历史损失集（供稍后展示）
    lossi.append(loss.log10().item())
```

```
      0/ 200000: 3.3239
  10000/ 200000: 2.0322
  20000/ 200000: 2.5675
  30000/ 200000: 2.0125
  40000/ 200000: 2.2446
  50000/ 200000: 1.8897
  60000/ 200000: 2.0785
  70000/ 200000: 2.3681
  80000/ 200000: 2.2918
  90000/ 200000: 2.0238
 100000/ 200000: 2.3673
 110000/ 200000: 2.3132
 120000/ 200000: 1.6414
 130000/ 200000: 1.9311
 140000/ 200000: 2.2231
 150000/ 200000: 2.0027
 160000/ 200000: 2.0997
 170000/ 200000: 2.4949
 180000/ 200000: 2.0199
 190000/ 200000: 2.1707
```

```python
plt.plot(lossi);
```

```python
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    
    emb = C[x]                          # (N, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1) # 拼接成 (N, block_size * n_embd)
    
    # 这部分是新的，就像我们在新的批归一化神经网络中做的那样
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    
    h = torch.tanh(hpreact)             # (N, n_hidden)
    logits = h @ W2 + b2                # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    
    print(split, loss.item())

split_loss('train')
split_loss('val')
```

```
train 2.0674195289611816
val 2.105670690536499
```

| 损失      | 原始       | 已解决问题 (1)     | 已解决问题 (2)     | 已解决问题 (1+2)    | 批归一化   |
| --------- | ---------- | ------------------ | ------------------ | ------------------- | ---------- |
| **训练**  | 2.1270018  | 2.0695891          | **2.0355966**      | 2.0376642           | 2.0668271  |
| **验证**  | 2.1699057  | 2.1310747          | **2.1026785**      | 2.1069896           | 2.1048448  |
---

## 题外话：卷积层

大多数当前模型的架构（见[示例](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)）都遵循上面这个小模型也有的基本结构：

- 重复 $x$ 次：
    - 线性层
    - 批归一化
    - 激活函数（例如 $\text{tanh}$、ReLU；一个非线性）

很多时候，这个线性层是一个卷积层。

> 卷积层是线性层，但它们专门用于图像数据。
> 它们具有空间结构，意思是 $W \cdot x+b$ 这个运算是针对局部图像块而不是整张图像一次进行的。这些块可以部分重叠。这是可配置的。

附带一提，在数学上，一个卷积层可以表达为：
$$\text{Conv}(x)_{i,j,c,{\text{out}}} = \sum_{c_{\text{in}}=1}^{C_{\text{in}}} \sum_{u=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} \sum_{v=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} W_{c_{\text{out}}, c_{\text{in}}, u, v} \cdot x_{i+u, j+v, c_{\text{in}}} + b_{c_{\text{out}}}$$

在 PyTorch 中，卷积层在 `nn.Conv2d` 类中定义。

---

## PyTorch 中的线性层

[线性层](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)执行计算 $W \cdot x+b$。为了能通过 PyTorch 创建这样的层，你需要知道以下先决条件：

- `in_features`（**fan-in**，单个输入的维度/尺寸（上面是 $10\times 3$））
- `out_features`（**fan-out**，单个输出的维度/尺寸（上面是 $200$））
- `bias`（`True` 或 `False`，默认为 `False`）

这会创建一个 `Linear.weight` 和一个 `Linear.bias` 变量。

下面是官方文档中的示例：

```python
m = torch.nn.Linear(10*3, 200) # 维度选择得和我们的示例一致
input = torch.randn(100, 30)   # 100 个 30 维输入，随机生成
output = m(input)              # 把输入喂给层
print(output.size())           # 我们现在应该有 100 乘 200 个“神经元反应”
```

```
torch.Size([100, 200])
```

## PyTorch 中的批归一化

PyTorch 也有一个[类](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html?highlight=batchnorm1d)用于这个神经网络层。

我们为一个这样的 `BatchNorm1d` 层需要：
- `num_features` - 一个输入的特征/维度数
- `epsilon` - 已经设了一个小的默认值，就让它这样吧
- `momentum` - 这是当前的 `mean` 和 `std` 贡献给 `running_mean` 和 `running_std` 的因子（对我们来说这个因子是 $0.001$，默认是 $0.1$）

> 数据集越小，`momentum` 的设置就应当越小。

- `affine` - （默认）$\text{true}$，如果应当存在 `bngain` 和 `bnbias`
- `track_running_stats` - （默认）$\text{true}$，如果应当存在 `bnmean_running` 和 `bnstd_running` 并在内部聚合

下面是官方文档中的示例：

```python
# 带 200 个可学习参数
m = torch.nn.BatchNorm1d(200) # 批归一化接收 200 个输入，和上面示例一样
# 不带可学习参数（老实说不是常规做法）
# m = nn.BatchNorm1d(200, affine=False)
input = torch.randn(20, 200)  # 我们扔进去 20 个输入，每个 200 维
output = m(input)
print(output)
print(output.shape) # 20 x 200，不变
```

```
tensor([[ 0.0468, -0.8650,  1.0644,  ...,  2.9211,  0.0032,  0.2861],
        [-0.4923, -0.9520, -0.2783,  ..., -0.4517,  0.5321,  1.4931],
        [ 0.1886,  0.0394,  0.0787,  ..., -0.9693,  1.1258, -0.6727],
        ...,
        [-0.4707,  1.0870, -0.1021,  ...,  0.2545,  0.2568,  0.0315],
        [ 0.6869,  1.8652, -0.4082,  ...,  0.9654,  0.6693,  0.1761],
        [ 0.1561,  0.1158, -1.1409,  ..., -1.1584,  1.1738,  0.0483]],
       grad_fn=<NativeBatchNormBackward0>)
torch.Size([20, 200])
```

> 实际上，由于批归一化把对数据的视角“批量化”及其训练效果，它真的可能很快反过来变成问题。
> **尽量避开批归一化。改用 Group Normalization 或 Layer Normalization。**

如果你在想，“批归一化相当出名且成熟。为什么要避开它？”，那你*一定要*继续读下去。

## 用 PyTorch 重建示例

下面的示例很大程度上基于我们之前的模型。
但现在我们也更仔细地看一看批归一化层的效果。

为此，我们首先把网络层构建为模块化的类。
然后我们构建一个*不带*批归一化的神经网络，考察其性质和可能的问题。
然后我们构建第二个 PyTorch 模型，但带批归一化层。

```python
# 线性层定义（模仿 torch.nn.Linear 的结构）
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    # 权重用 Kaiming 初始化，就像上面会写的 (W1 * (5/3)/((n_embd * block_size)**0.5))
    # 这里还缺少 (5/3) 这一项，那是因为我们这里还没有非线性
    # 我们稍后再加上
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
    self.bias = torch.zeros(fan_out) if bias else None # 偏置在这里是可选的
  
  def __call__(self, x):
    self.out = x @ self.weight # W*x
    if self.bias is not None:  # 如果需要就加偏置
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias]) # 返回该层的张量
```

```python
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps            # Epsilon 设为 PyTorch 默认值，你可以改
    self.momentum = momentum  # Momentum 设为 PyTorch 默认值，你可以改
    self.training = True
    # 初始化参数（由反向传播训练）
    # (bngain -> gamma, bnbias -> beta)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # 初始化缓冲区
    # (用滑动 'momentum 更新' 来训练)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  

  def __call__(self, x):
    # 前向传播
    if self.training:
      xmean = x.mean(0, keepdim=True) # 批次均值
      xvar = x.var(0, keepdim=True)   # 批次方差
    else:
      xmean = self.running_mean # 用滑动均值作为基准
      xvar = self.running_var   # 用滑动方差作为基准
    
    # 归一化到单位方差
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta  # 应用批 gain 和 bias
    
    # 更新滑动缓冲区
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    
    return self.out
  

  def parameters(self):
    return [self.gamma, self.beta] # 返回该层的张量
```

```python
# 类似 torch.tanh()，但用类结构以方便后续步骤
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []
```

### 带 Tanh 且 gain 正确的 MLP 的激活分布

```python
n_embd = 10    # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, vocab_size),
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].weight *= 0.1
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
46497
```

```python
# 和上次一样的优化
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # 小批量构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
    # 前向传播
    emb = C[Xb] # 把字符嵌入为向量
    x = emb.view(emb.shape[0], -1) # 拼接向量
    # 按原始顺序把层叠起来（一个接一个）
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # 损失函数
  
    # 反向传播
    for layer in layers:
        # 声明非叶变量梯度要被保留/留存以供评估
        layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # 更新
    lr = 0.1 if i < 100000 else 0.01 # 阶梯式学习率衰减
    for p in parameters:
        p.data += -lr * p.grad

    # 跟踪统计量
    if i % 10000 == 0: # 偶尔打印
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        
    break # 去掉这行以跑完整的优化/训练
```

```
      0/ 200000: 3.2962
```

```python
# 可视化 Tanh 层的直方图（前向传播激活）
# （下面是需要批归一化时的样子）
# 我们可以看到多少张量值取哪些 x 轴值
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层，这些都是 Tanh 层
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Activation Distribution')
plt.legend(legends);
```

```
layer 1 (      Tanh): mean -0.02, std 0.75, saturated: 20.25%
layer 3 (      Tanh): mean -0.00, std 0.69, saturated: 8.38%
layer 5 (      Tanh): mean +0.00, std 0.67, saturated: 6.62%
layer 7 (      Tanh): mean -0.01, std 0.66, saturated: 5.47%
layer 9 (      Tanh): mean -0.02, std 0.66, saturated: 6.12%
```

这张图向我们展示了各层（除输出层外）的取值范围。
也就是说我们可以看到每层有多少张量取哪些 $\text{x}$ 轴值。

**尤其在第 $1$ 层，我们看到了我们熟悉的问题：** 大量神经元被压缩到 $\text{tanh}$ 取值范围的边界。
你也能在百分比中看到这一点，因为 $20.25\%$ 的神经元的 $\text{tanh}$ 值绝对值大于 $0.97$。
这很糟糕，因为我们以此阻断了梯度的学习效果，使学习过程变得迟钝。

有趣的是，这种效果仅限于第一层。后续各层看起来不错。**但这是为什么？**

$\text{gain}$ 被设为 $\frac{5}{3}$，因为我们用的是 $\text{tanh}$ 激活函数。
这是 [\[Kaiming He 等人 2015\] 论文](https://arxiv.org/abs/1502.01852) 中为 $\text{tanh}$ 推荐的值，但也是原因所在。

我们来看看如果把 $\text{gain}$ 设为 $1.0$ 会怎样：

### 带 Tanh 但 gain 过小的 MLP 的激活分布

```python
n_embd = 10    # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].weight *= 0.1
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1 # gain 5/3 被替换为 1

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
46497
```

```python
# 和上次一样的优化
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # 小批量构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
    # 前向传播
    emb = C[Xb] # 把字符嵌入为向量
    x = emb.view(emb.shape[0], -1) # 拼接向量
    # 按原始顺序把层叠起来（一个接一个）
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # 损失函数
  
    # 反向传播
    for layer in layers:
        # 声明非叶变量梯度要被保留/留存以供评估
        layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # 更新
    lr = 0.1 if i < 100000 else 0.01 # 阶梯式学习率衰减
    for p in parameters:
        p.data += -lr * p.grad

    # 跟踪统计量
    if i % 10000 == 0: # 偶尔打印
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        
    break # 去掉这行以跑完整的优化/训练
```

```
      0/ 200000: 3.2988
```

```python
# 可视化 Tanh 层的激活直方图（即激活值）
# （下面是最坏情况的样子）
# 我们可以看到多少张量值取哪些 x 轴值
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Activation Distribution')
plt.legend(legends);
```

```
layer 1 (      Tanh): mean -0.02, std 0.62, saturated: 3.50%
layer 3 (      Tanh): mean -0.00, std 0.48, saturated: 0.03%
layer 5 (      Tanh): mean +0.00, std 0.41, saturated: 0.06%
layer 7 (      Tanh): mean +0.00, std 0.35, saturated: 0.00%
layer 9 (      Tanh): mean -0.02, std 0.32, saturated: 0.00%
```

没有 $\text{gain}$，激活的标准差现在随每层**递减**。
它不再被保持为 $1$。现在第一层看起来最“正常”，后续各层越来越被压缩到 $\text{tanh}$ 取值范围的中心 $0$。

事实上，这个趋势会一直持续到标准差为 $0$。
线性层之间的 $\text{tanh}$ 函数负责压缩，因此也负责这个效果及其加剧。

> 对于“线性层与 $\text{tanh}$ 层三明治”式的模型，$\frac{5}{3}$ 是 $\text{gain}$ 的一个好值。
> 它从哪儿来？不知道。
> 但我们在倒数第二张图中看到 $\text{gain} = \frac{5}{3}$ 的效果正是我们需要的，即标准差被保持在 $1$。

**好。** 这些是每层神经元的**激活**。
现在我们来看看 $\text{gain} = \frac{5}{3}$ 模型每层的**梯度**：

### 带 Tanh 且 gain 正确的 MLP 的梯度分布

```python
n_embd = 10 # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), Tanh(),
  Linear(n_hidden, vocab_size),
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].weight *= 0.1
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
46497
```

```python
# 和上次一样的优化
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # 小批量构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
    # 前向传播
    emb = C[Xb] # 把字符嵌入为向量
    x = emb.view(emb.shape[0], -1) # 拼接向量
    # 按原始顺序把层叠起来（一个接一个）
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # 损失函数
  
    # 反向传播
    for layer in layers:
        # 声明非叶变量梯度要被保留/留存以供评估
        layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # 更新
    lr = 0.1 if i < 100000 else 0.01 # 阶梯式学习率衰减
    for p in parameters:
        p.data += -lr * p.grad

    # 跟踪统计量
    if i % 10000 == 0: # 偶尔打印
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        
    break # 去掉这行以跑完整的优化/训练
```

```
      0/ 200000: 3.2962
```

```python
# 可视化梯度直方图（下面是理想的样子）
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Gradient Distribution')
plt.legend(legends);
```

```
layer 1 (      Tanh): mean +0.000010, std 4.205588e-04
layer 3 (      Tanh): mean -0.000003, std 3.991179e-04
layer 5 (      Tanh): mean +0.000003, std 3.743020e-04
layer 7 (      Tanh): mean +0.000015, std 3.290473e-04
layer 9 (      Tanh): mean -0.000014, std 3.054035e-04
```

这看起来*相当不错*，因为跨层没有任何异常发生。
梯度在各层上应形成（共同的）钟形曲线。*这基本就是这种情况。*

**什么时候你会看到与上图不同的情形？**
如果 $\text{gain}$ 选得太小或太大，钟形曲线看起来会不同。
它会是这样：

### 带 Tanh 但 gain 过小的 MLP 的梯度分布

```python
n_embd = 10    # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, vocab_size),
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].weight *= 0.1
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 0.5

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
46497
```

```python
# 和上次一样的优化
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # 小批量构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
    # 前向传播
    emb = C[Xb] # 把字符嵌入为向量
    x = emb.view(emb.shape[0], -1) # 拼接向量
    # 按原始顺序把层叠起来（一个接一个）
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # 损失函数
  
    # 反向传播
    for layer in layers:
        # 声明非叶变量梯度要被保留/留存以供评估
        layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # 更新
    lr = 0.1 if i < 100000 else 0.01 # 阶梯式学习率衰减
    for p in parameters:
        p.data += -lr * p.grad

    # 跟踪统计量
    if i % 10000 == 0: # 偶尔打印
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        
    break # 去掉这行以跑完整的优化/训练
```

```
      0/ 200000: 3.2960
```

```python
# 可视化梯度直方图
# （下面是你不想要的样子）
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Gradient Distribution')
plt.legend(legends);
```

```
layer 1 (      Tanh): mean +0.000000, std 1.892402e-05
layer 3 (      Tanh): mean -0.000001, std 3.943546e-05
layer 5 (      Tanh): mean +0.000004, std 8.035369e-05
layer 7 (      Tanh): mean +0.000009, std 1.561152e-04
layer 9 (      Tanh): mean -0.000014, std 3.053498e-04
```

$\text{gain} = \frac{5}{3}$ 避免了这种每层（跨其神经元）不均匀的梯度分布。**好。**

> 我们看到，**如果我们*不*使用批归一化**，就必须非常精确地寻找 $\text{gain}$，以获得对激活和梯度的正确效果。

好的，但在加入批归一化层之前，我们应该看看如果只用线性层、连 $\text{tanh}$ 层都不用会发生什么：

### 不带 Tanh 且 gain 过小的 MLP 的激活分布与梯度分布

```python
n_embd = 10 # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden),
  Linear(n_hidden, n_hidden),
  Linear(n_hidden, n_hidden),
  Linear(n_hidden, n_hidden),
  Linear(n_hidden, n_hidden),
  Linear(n_hidden, vocab_size),
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].weight *= 0.1
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 0.5

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
46497
```

```python
# 和上次一样的优化
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    
    # 小批量构造
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
    # 前向传播
    emb = C[Xb] # 把字符嵌入为向量
    x = emb.view(emb.shape[0], -1) # 拼接向量
    # 按原始顺序把层叠起来（一个接一个）
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # 损失函数
  
    # 反向传播
    for layer in layers:
        # 声明非叶变量梯度要被保留/留存以供评估
        layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # 更新
    lr = 0.1 if i < 100000 else 0.01 # 阶梯式学习率衰减
    for p in parameters:
        p.data += -lr * p.grad

    # 跟踪统计量
    if i % 10000 == 0: # 偶尔打印
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        
    break # 去掉这行以跑完整的优化/训练
```

```
      0/ 200000: 3.2959
```

```python
# 可视化线性层的激活直方图（即激活值）
# （下面是你不想要的样子）
# 只有在网络的更深层才有高激活

# 我们可以看到多少张量值取哪些 x 轴值
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Linear):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Activation Distribution')
plt.legend(legends);
```

```
layer 0 (    Linear): mean -0.01, std 0.49, saturated: 5.06%
layer 1 (    Linear): mean -0.00, std 0.25, saturated: 0.00%
layer 2 (    Linear): mean -0.00, std 0.13, saturated: 0.00%
layer 3 (    Linear): mean +0.00, std 0.06, saturated: 0.00%
layer 4 (    Linear): mean -0.00, std 0.03, saturated: 0.00%
```

```python
# 可视化梯度直方图
# （下面是你不想要的样子）
# 只有在浅层才有高梯度（因此有大更新）

plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Linear):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Gradient Distribution')
plt.legend(legends);
```

```
layer 0 (    Linear): mean +0.000000, std 1.990759e-05
layer 1 (    Linear): mean -0.000001, std 3.997084e-05
layer 2 (    Linear): mean +0.000004, std 8.058026e-05
layer 3 (    Linear): mean +0.000009, std 1.562154e-04
layer 4 (    Linear): mean -0.000014, std 3.053490e-04
```

我们在这里可以观察到最坏情况。

没有 $\text{tanh}$，网络越深，激活和学习效果就越弱。
因此这种方法不可用，因为它不可扩展。此外，学习效果明显更低。
**如果在这里不用 $\text{tanh}$，性能会显著下降。**

（顺便说一句，纯线性层神经网络的正确 $\text{gain}$ 应当是 $1$）

没有批归一化时，要做的不仅是找到最有效的层，**还要**设置正确的 $\text{gain}$，这使整个过程更像一种走钢丝式的平衡。

![](https://image.shutterstock.com/image-photo/balancing-pencil-on-index-finger-260nw-242528134.jpg)
来源：[Shutterstock](https://www.shutterstock.com/image-photo/balancing-pencil-on-index-finger-242528134)

**等一下……**

……如果你有一个由纯线性层组成、没有 $\text{tanh}$ 的神经网络，而最优的 $\text{gain}$ 又是 $1$、因而完全没有效果，那为什么……我们还要用 $\text{tanh}$ 层呢？**这不是把一切都搞复杂了吗？** 毕竟一个纯线性神经网络也能被最优地调整，然后照常训练。

> 一堆纯线性层能学到的复杂关系，仅相当于单个大线性层所能学到的。
> 所有的部分 $W \cdot x+b$ 都可以合并成一个大的 $W \cdot x+b$。
> **但是：** 对单个大线性层的反向传播，和对三明治里多个线性层的反向传播，在行为上是两回事。

**$\text{Tanh}$ 给这个“大型线性函数三明治”扩展了表示非线性关系的能力！**

所以让我们回到*带* $\text{tanh}$ 层的神经网络架构：

### 带 Tanh 且 gain 正确的 MLP 的激活分布与梯度分布

```python
n_embd = 10 # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, n_hidden),  Tanh(),
  Linear(n_hidden, vocab_size),
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].weight *= 0.1 # 这让该参数比率成为离群点（见图）
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
46497
```

```python
# 和上次一样的优化
max_steps = 200000 
batch_size = 32    
lossi = []         # 跟踪损失
ud = []            # 跟踪更新与数据之比

for i in range(max_steps):
  # 小批量构造
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
  # 前向传播
  emb = C[Xb] # 把字符嵌入为向量
  x = emb.view(emb.shape[0], -1) # 拼接向量
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # 损失函数
  
  # 反向传播
  for layer in layers:
    layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # 更新
  lr = 0.1 if i < 150000 else 0.01 # 阶梯式学习率衰减
  for p in parameters:
    p.data += -lr * p.grad

  # 跟踪统计量
  if i % 10000 == 0: # 偶尔打印
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

  if i >= 1000:
    break # AFTER_DEBUG: 显然会去掉以跑完整优化
```

```
      0/ 200000: 3.2962
```

```python
# 可视化 Tanh 层的直方图（前向传播激活）
# 我们可以看到多少张量值取哪些 x 轴值
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Activation Distribution')
plt.legend(legends);
```

```
layer 1 (      Tanh): mean -0.04, std 0.76, saturated: 21.97%
layer 3 (      Tanh): mean -0.01, std 0.72, saturated: 11.00%
layer 5 (      Tanh): mean +0.01, std 0.73, saturated: 13.00%
layer 7 (      Tanh): mean -0.05, std 0.73, saturated: 13.34%
layer 9 (      Tanh): mean +0.00, std 0.72, saturated: 10.53%
```

```python
# 可视化梯度直方图（这张图看起来不错）
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.title('Gradient Distribution')
plt.legend(legends);
```

```
layer 1 (      Tanh): mean +0.000024, std 3.353992e-03
layer 3 (      Tanh): mean +0.000012, std 3.157344e-03
layer 5 (      Tanh): mean -0.000004, std 2.925863e-03
layer 7 (      Tanh): mean +0.000036, std 2.715700e-03
layer 9 (      Tanh): mean +0.000020, std 2.308167e-03
```

```python
# 可视化线性层的权重直方图（不是偏置、gamma 或 beta）
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('Weights Gradient Distribution');
```

```
weight   (27, 10) | mean +0.000980 | std 1.189171e-02 | grad:data ratio 1.189149e-02
weight  (30, 100) | mean +0.000118 | std 1.005291e-02 | grad:data ratio 3.214556e-02
weight (100, 100) | mean +0.000033 | std 7.821212e-03 | grad:data ratio 4.653362e-02
weight (100, 100) | mean -0.000107 | std 6.655620e-03 | grad:data ratio 3.925851e-02
weight (100, 100) | mean -0.000017 | std 6.086041e-03 | grad:data ratio 3.605768e-02
weight (100, 100) | mean -0.000077 | std 5.075620e-03 | grad:data ratio 3.015269e-02
weight  (100, 27) | mean -0.000000 | std 2.056585e-02 | grad:data ratio 2.909910e-01
```

我们在这里额外展示了梯度数据值比。
如果这个比太大，我们就有问题了。那就说明数据和/或神经元的一般特性出了问题。

**但这里并非如此。**

（输出不参与比率评估，它只是被忽略。）

### 带 Tanh 但 gain 过小的 MLP 的激活分布与梯度分布

好吧，为了完整起见，我们来看另一张能帮助分析这类模型性能和结构的图：更新与数据之比。
这不像梯度那样是某个值的纯变化强度，而是神经元被调整的实际数据值。

为此我们在上面的代码中扩展了更新与数据之比变量 `ud`。

```python
# 更新与数据之比的直方图（下面是理想的样子）
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # 这些比率应 ~1e-3，在图上标出
plt.legend(legends);
```

我们绘制各参数。用黑线给出理想走向，即我们期望的参数走向（对数应在 $-3$ 附近的某处变平）。输出层再次是个离群点，因为我们在初始化时把它缩放了 $0.1$ 倍以让这一层更灵活/不那么“自信地错”。这人为地降低了权重，所以第 $11$ 层才会这样偏离。

> 如果一层的对数在 $-3$ 附近变平，这层就学得不错。
> 如果参数稳定在显著低于 $-3$ 的位置，则**学习率太低**。

**现在我们来正视这个显而易见的问题：**
**前向传播激活分布暗示激活倾向于走向 $\text{tanh}$ 的取值边界，从而拖慢学习效果。** 所以我们需要批归一化：

### 带 Tanh、批归一化且关闭 gain 的 MLP 的激活分布与梯度分布

```python
n_embd = 10 # 字符嵌入向量的维度
n_hidden = 100 # MLP 隐藏层的神经元数量
g = torch.Generator().manual_seed(2147483647) # 用于可复现性

C = torch.randn((vocab_size, n_embd), generator=g)

layers = [
  Linear(n_embd * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size), BatchNorm1d(vocab_size), 
]

with torch.no_grad():
  # 最后一层：让它不那么自信
  layers[-1].gamma *= 0.1 # 因为最后一层是批归一化
  # 所有其他层：应用 gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0

# 嵌入矩阵 + 所有层中的所有参数 = 涉及的总参数
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # 总参数数

# 这些参数将受反向传播影响
for p in parameters:
  p.requires_grad = True
```

```
47551
```

```python
# 和上次一样的优化
max_steps = 200000 
batch_size = 32    
lossi = []         # 跟踪损失
ud = []            # 跟踪更新与数据之比

for i in range(max_steps):
  # 小批量构造
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # 批次 X,Y
  
  # 前向传播
  emb = C[Xb] # 把字符嵌入为向量
  x = emb.view(emb.shape[0], -1) # 拼接向量
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # 损失函数
  
  # 反向传播
  for layer in layers:
    layer.out.retain_grad() # AFTER_DEBUG: 会去掉 retain_graph
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # 更新
  lr = 0.1 if i < 150000 else 0.01 # 阶梯式学习率衰减
  for p in parameters:
    p.data += -lr * p.grad

  # 跟踪统计量
  if i % 10000 == 0: # 偶尔打印
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

  if i >= 1000:
    break # AFTER_DEBUG: 显然会去掉以跑完整优化
```

```
      0/ 200000: 3.2870
```

```python
# 可视化 Tanh 层的直方图（前向传播激活）
# 这是理想图，非常均匀
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.title('Activation Distribution')
plt.legend(legends);
```

```
layer 2 (      Tanh): mean -0.00, std 0.63, saturated: 2.78%
layer 5 (      Tanh): mean +0.00, std 0.64, saturated: 2.56%
layer 8 (      Tanh): mean -0.00, std 0.65, saturated: 2.25%
layer 11 (      Tanh): mean +0.00, std 0.65, saturated: 1.69%
layer 14 (      Tanh): mean +0.00, std 0.65, saturated: 1.88%
```

```python
# 可视化梯度直方图（这张图看起来不错）
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i, layer in enumerate(layers[:-1]): # 注意：排除输出层
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.title('Gradient Distribution')
plt.legend(legends);
```

```
layer 2 (      Tanh): mean +0.000000, std 2.640701e-03
layer 5 (      Tanh): mean +0.000000, std 2.245584e-03
layer 8 (      Tanh): mean +0.000000, std 2.045741e-03
layer 11 (      Tanh): mean -0.000000, std 1.983132e-03
layer 14 (      Tanh): mean -0.000000, std 1.952381e-03
```

```python
# 可视化线性层的权重直方图（不是偏置、gamma 或 beta）
# 理想图
plt.figure(figsize=(20, 4)) # 图的宽和高
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('Weights Gradient Distribution');
```

```
weight   (27, 10) | mean -0.000000 | std 8.020527e-03 | grad:data ratio 8.012623e-03
weight  (30, 100) | mean +0.000246 | std 9.241064e-03 | grad:data ratio 4.881084e-02
weight (100, 100) | mean +0.000113 | std 7.132873e-03 | grad:data ratio 6.964613e-02
weight (100, 100) | mean -0.000086 | std 6.234302e-03 | grad:data ratio 6.073738e-02
weight (100, 100) | mean +0.000052 | std 5.742181e-03 | grad:data ratio 5.631477e-02
weight (100, 100) | mean +0.000032 | std 5.672201e-03 | grad:data ratio 5.570121e-02
weight  (100, 27) | mean -0.000082 | std 1.209415e-02 | grad:data ratio 1.160105e-01
```

```python
# 更新与数据之比的直方图（下面是理想的样子）
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # 这些比率应 ~1e-3，在图上标出
plt.legend(legends);
```

在某种程度上，我们对 $\text{gain}$ 的鲁棒性强多了。
尽管如此，它对数据比率直方图仍有影响。
你仍然得寻找一个既不太大又不太小的 $\text{gain}$，尽管是有针对性地找。
（在我们这里就是 $1$，即没有效果。*那正合适。*）

## 附加内容（课程未涉及）

```python
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import torch
```

```python
# 作为小部件的批归一化前向传播
# （在你的机器上本地运行）

def normshow(x0):
  
  g = torch.Generator().manual_seed(2147483647+1)
  x = torch.randn(5, generator=g) * 5
  x[0] = x0 # 用滑块覆盖第 0 个样本
  mu = x.mean()
  sig = x.std()
  y = (x - mu)/sig

  plt.figure(figsize=(10, 5))
  # 画 0
  plt.plot([-6,6], [0,0], 'k')
  # 画均值和标准差
  xx = np.linspace(-6, 6, 100)
  plt.plot(xx, stats.norm.pdf(xx, mu, sig), 'b')
  xx = np.linspace(-6, 6, 100)
  plt.plot(xx, stats.norm.pdf(xx, 0, 1), 'r')
  # 画连接输入和输出的小线
  for i in range(len(x)):
    plt.plot([x[i],y[i]], [1, 0], 'k', alpha=0.2)
  # 画输入和输出值
  plt.scatter(x.data, torch.ones_like(x).data, c='b', s=100)
  plt.scatter(y.data, torch.zeros_like(y).data, c='r', s=100)
  plt.xlim(-6, 6)
  # 标题
  plt.title('input mu %.2f std %.2f' % (mu, sig))

interact(normshow, x0=(-30,30,0.5));
```

```python
# 线性层：前向和反向传播的激活统计

g = torch.Generator().manual_seed(2147483647)

a = torch.randn((1000,1), requires_grad=True, generator=g)          # a.grad = b.T @ c.grad
b = torch.randn((1000,1000), requires_grad=True, generator=g)       # b.grad = c.grad @ a.T
c = b @ a
loss = torch.randn(1000, generator=g) @ c
a.retain_grad()
b.retain_grad()
c.retain_grad()
loss.backward()
print('a std:', a.std().item())
print('b std:', b.std().item())
print('c std:', c.std().item())
print('-----')
print('c grad std:', c.grad.std().item())
print('a grad std:', a.grad.std().item())
print('b grad std:', b.grad.std().item())
```

```
a std: 0.9875972270965576
b std: 1.0006722211837769
c std: 31.01241683959961
-----
c grad std: 0.9782556295394897
a grad std: 30.8818302154541
b grad std: 0.9666601419448853
```

```python
# 线性层 + 批归一化：前向和反向传播的激活统计

g = torch.Generator().manual_seed(2147483647)

n = 1000
# 线性层 ---
inp = torch.randn(n, requires_grad=True, generator=g)
w = torch.randn((n, n), requires_grad=True, generator=g) # / n**0.5
x = w @ inp
# 批归一化层 ---
xmean = x.mean()
xvar = x.var()
out = (x - xmean) / torch.sqrt(xvar + 1e-5)
# ----
loss = out @ torch.randn(n, generator=g)
inp.retain_grad()
x.retain_grad()
w.retain_grad()
out.retain_grad()
loss.backward()

print('inp std: ', inp.std().item())
print('w std: ', w.std().item())
print('x std: ', x.std().item())
print('out std: ', out.std().item())
print('------')
print('out grad std: ', out.grad.std().item())
print('x grad std: ', x.grad.std().item())
print('w grad std: ', w.grad.std().item())
print('inp grad std: ', inp.grad.std().item())
```

```
inp std:  0.9875972270965576
w std:  1.0006722211837769
x std:  31.01241683959961
out std:  1.0
------
out grad std:  0.9782556295394897
x grad std:  0.031543977558612823
w grad std:  0.031169468536973
inp grad std:  0.9953053593635559
```

<center>本笔记本由 <a href="https://github.com/mk2112" target="_blank">mk2112</a> 编写。</center>
