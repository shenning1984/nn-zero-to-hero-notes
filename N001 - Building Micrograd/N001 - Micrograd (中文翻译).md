# 构建 Micrograd（Building Micrograd）

[视频](https://www.youtube.com/watch?v=VMj-3S1tku0)
[代码仓库](https://github.com/karpathy/micrograd)
[Eureka Labs Discord](https://discord.com/invite/3zy8kqD9Cp)

## 目录

- [目标](#目标)
- [什么是 Micrograd](#什么是-micrograd)
- [问题剖析](#问题剖析)
    - [理解导数](#理解导数)
    - [神经网络中的导数](#神经网络中的导数)
        - [Value 类 - 基础搭建](#value-类---基础搭建)
        - [Value 类 - 前向传播](#value-类---前向传播)
        - [Value 类 - 图的生成](#value-类---图的生成)
        - [小结回顾](#小结回顾)
        - [Value 类 - 搭建反向传播](#value-类---搭建反向传播)
- [神经网络](#神经网络)
    - [手动反向传播](#手动反向传播)
    - [自动反向传播](#自动反向传播)
        - [Value 类 - 找 Bug 与扩展](#value-类---找-bug-与扩展)
    - [一切汇聚于此](#一切汇聚于此)
- [用 PyTorch API 做同样的事](#用-pytorch-api-做同样的事)
- [回到神经网络](#回到神经网络)

```python
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## 目标

**本系列旨在从零开始理解、构建并训练神经网络。**
我们将从最基础的概念讲起，一路实现一个功能完整的 GPT 克隆。
首先，我们会过一遍 [micrograd](https://github.com/karpathy/micrograd) 项目，借此学习神经网络的基础知识。

> **Micrograd 是一个自动求导引擎（Automatic Gradient Engine）。**
> 它仅用约 $150$ 行代码就包含了训练神经网络所需的全部核心要素。
> 为此，Micrograd *从零开始* 实现了一个被称为**反向传播（backpropagation）** 的过程。

**什么是神经网络？**
神经网络由所谓的神经元（neuron）组成。多个神经元构成一个网络层（layer）。
多个网络层堆叠在一起，便构成了一个基础的神经网络。
一个神经元可能接收来自上一层神经元的输入连接，并向下一层神经元发出输出连接。
这些连接的强度各不相同。对于每一条连接，都有一个定义其强度的参数，这就是所谓的*权重（weight）*。
神经元会计算其所有输入连接的加权和，对其施加一个所谓的非线性函数，然后通过输出连接把结果传给下一层。
这样，一般来说，基础的神经网络会接收某种输入数据，并让数据穿过它的各个网络层，最终产生输出数据。

**什么是反向传播？**
神经网络通过它在输出中所犯的错误来学习。*反向传播* 允许我们迭代地调整网络的参数，使网络逐渐收敛，从而最小化网络自身输出与应该达到的已知参考值（即期望的输出值）之间的差距。
于是，反向传播实际上是在对网络的输出错误进行惩罚。
这种惩罚会以一种特定的方式改变网络的参数，使得下一次给网络同样的输入时，它产生的输出会更接近期望值。

<b>读到这儿，如果上面任何一段话听起来太复杂，[别担心](https://miro.medium.com/v2/resize:fit:1280/1*E4_pTJctmAofSRpZCZbv-g.jpeg)。</b>
我们会在适当的时候，逐个地、*从零开始* 地去接触这些概念。
现在只要记住，这个 *反向传播* 过程是神经网络如何学习的核心，而 Micrograd 从零实现了它。

## 什么是 Micrograd?

<b>自动求导引擎是训练神经网络*最*核心的组件。</b>
Micrograd 是一个微型的自动求导引擎，它支持自动微分以及高阶梯度的计算。
它仅仅 150 行 Python 代码，但作为理解自动求导引擎*是什么*、*做什么* 以及*如何工作* 的工具却非常有效。

让我们先来看一个使用[原版](https://github.com/karpathy/micrograd) Micrograd 库的简单例子：

```python
from micrograd.engine import Value


# 创建两个 "Value" 对象，
# 各自包裹一个浮点数
a = Value(-4.0)
b = Value(2.0)

# 对 "Value" 对象应用算术运算
# 创建一个新的 "Value" 对象 'c'，并对其覆盖两次
c = a + b
c += c + 1
c += 1 + c + (-a)

# 打印 -1.0，即对这些算术表达式做一次前向传播的结果
print(f"Data contained within c:   {c.data:4.1f}")

# 应用反向传播（即沿表达式图做一次反向遍历）
# 计算所有对 c 有贡献的 "Value" 对象（也就是 a 和 b）的梯度
c.backward()

# dc/da = 3.0
print(f"Gradient calculated for a: {a.grad:4.1f}")
# dc/db = 4.0；b 的一点点小变化会导致 c 发生 4.0 倍大小的变化
print(f"Gradient calculated for b: {b.grad:4.1f}")
```

```
Data contained within c:   -1.0
Gradient calculated for a:  3.0
Gradient calculated for b:  4.0
```

> **梯度是一个值，它表示最终结果对各个影响因素变化的敏感程度。**
> 如果我们知道某个值的变化会在多大程度上影响最终结果，就可以据此调整该值，使最终结果朝着期望的方向移动。这就是**反向传播**背后的核心概念。

**Micrograd 允许我们定义数值，并对这些数值应用通常的算术运算。**
在后台，Micrograd 还会随着运算的进行，在所谓的**表达式图（expression graph）** 中记录每个值的使用情况以及它们对后续值的贡献。
随后，这张表达式图会被*反向* 遍历，以**计算**由*所有* 应用过的运算所导致的**梯度**。
自动求导引擎的目标，就围绕着如何正确、快速且自动地为各种算术运算计算出**梯度**。

在上面的代码中，`a` 和 `b` 通过若干不同的运算影响 `c`。
之后，我们运行 `c.backward()` 来计算 `a` 和 `b` 相对于 `c` 的梯度。
这些梯度表示最终结果 `c` 对影响因素 `a` 和 `b` 变化的敏感程度。
例如，初始值 `b` 的一点点小变化，会导致最终结果 `c` 发生 $4.0 \times$ 大小的变化。

上面的例子非常基础，但反向传播可以用于各种算术运算。对于多层感知机（MLP，神经网络的一个子类）来说，情况稍微具体一些：输入和权重通过矩阵乘法和加法相互发生作用。

> 回到我们上面关于反向传播的解释：从 `c` 出发，通过对表达式图中所有影响 `c` 值的节点递归地应用[**链式法则**](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/a/chain-rule-review)来计算梯度。在这条表达式链条中，一个值影响下一个值的方式由*局部梯度*（local gradient）刻画，而局部梯度是相对于它所在的局部运算来计算的。链式法则随后允许我们把局部梯度与链条中前面值的梯度结合起来，这就是我们能够为每个值计算出相对于 `c` 的整体梯度的方式。


**但这到底意味着什么？**
**什么是表达式图？**
**我们为什么使用表达式图？**
**链式法则在这里如何应用？**
让我们一步一步地来看这个一般性问题。

## 问题剖析

### 理解导数

如果我们想要推导做出贡献的值对其他值的局部影响，就需要计算导数。
**导数量化了输入的变化在多大程度上影响输出。**

先慢慢来；我们先在 Python 里实现任意一个二次函数 $f(x) = 3x^2-4x+5$，并动手玩一玩它：

```python
def f(x):
    # 我们的示例函数
    return 3*x**2 - 4*x + 5

# 打印 20.0
f(3.0)
```

```
20.0
```

我们再把示例函数画出来：

```python
# 一个从 -5 到 5、步长为 0.25 的取值范围
xs = np.arange(-5, 5, 0.25)

# 对每个 x 应用 y = f(x)
# 把所有结果收集到 ys 中
ys = f(xs)

# 为每个 x 绘制 y
plt.plot(xs, ys)
plt.grid(True)
plt.show();
```

```
（此处省略图表输出）
```

**那么，我们的函数 $f(x)$ 在任意一点 $x$ 处的导数是多少？**
要解决这个问题，我们首先要理解，导数到底告诉了我们关于函数 $f(x)$ 的什么信息。

**这是教科书上关于函数 $f(x)$ 可微的定义：**
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/aae79a56cdcbc44af1612a50f06169b07f02cbf3)

我们被要求向我们的 $a$ 加上一个接近 $0$ 的正值 $h$，看看这种"从 $f(a)$ 到 $f(a+h)$ 的移动"是增加了还是减少了函数返回的值。
如果值在 $f(a)$ 与 $f(a+h)$ 之间增加了，导数就是正的。
如果函数减小了，导数就是负的。

我们*原原本本地* 这样实现：

```python
h = 0.00000001 # 1e-8
x = 3.0

# 近似 f 在 x=3 处的导数
print((f(x+h)-f(x))/h)
```

```
14.00000009255109
```

这个结果告诉我们，$f$ 在 $x=3$ 处关于 $x$ 的导数可近似为 $m \approx 14$。
（微积分当然教过我们，$f(x) = 3x^2-4x+5$ 的导数是 $f'(x) = 6x-4$，而 $f'(3) = 14$，但我们在这里严格遵循近似定义行事。）

**现在让我们把复杂度提高一点，用 $3$ 个输入产生 $1$ 个输出：**

```python
a = 2.0
b = -3.0
c = 10.0

d = a * b + c

print(f"a * b + c = {d}")
```

```
a * b + c = 4.0
```

给定上面的代码，`a * b + c = d` 仍然是一个函数。
乍一看，你可能会觉得这个函数比之前的二次函数"更简单"。
**但 `d` 关于 `a`、`b` 或 `c` 的导数是多少？**

我们再次采用原原本本的方式：

```python
h = 0.00000001 # 1e-8

# 这是我们想求 d 的导数的那个点 (a, b, c)
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c # (a, b, c) 处的函数值

a += h       # 把 a 抬高 h
d2 = a*b + c # (a+h, b, c) 处的函数值

a -= h       # 恢复 a
b += h       # 把 b 抬高 h
d3 = a*b + c # (a, b+h, c) 处的函数值

b -= h       # 恢复 b
c += h       # 把 c 抬高 h
d4 = a*b + c # (a, b, c+h) 处的函数值

print('Function value d1 for (a,b,c):\t ', d1)
print('Function value d2 for (a+h,b,c): ', d2)
print('Slope between d2 and d1:\t', (d2 - d1)/h)  # 仅把 a 抬高 h 后，函数值增加了多少
print()
print('Function value d3 for (a,b+h,c): ', d3)
print('Slope between d3 and d1:\t ', (d3 - d1)/h) # 仅把 b 抬高 h 后，函数值增加了多少
print()
print('Function value d4 of (a,b,c+h):\t ', d4)
print('Slope between d4 and d1:\t ', (d4 - d1)/h) # 仅把 c 抬高 h 后，函数值增加了多少
```

```
Function value d1 for (a,b,c):	  4.0
Function value d2 for (a+h,b,c):  3.99999997
Slope between d2 and d1:	 -2.999999981767587

Function value d3 for (a,b+h,c):  4.00000002
Slope between d3 and d1:	  1.999999987845058

Function value d4 of (a,b,c+h):	  4.000000010000001
Slope between d4 and d1:	  1.000000082740371
```

通过把每个输入分别抬高 `h`，我们就能看出原始函数的值相应地是*如何*变化的。

更具体地说，我们可以观察到：
- $(d2 - d1) / h \approx -3.0$，它近似于 `b`。
    - 把 `a` 抬高 `h` 会被 `b` 缩放，因为在 `f = ab + c` 中值 `a` 总是被 `b` 缩放。
- $(d3 - d1) / h \approx 2.0$，它近似于 `a`。
    - 把 `b` 抬高 `h` 会让输出被 `a` 缩放，因为在 `f = ab + c` 中 `b` 总是被 `a` 缩放（逻辑同上）。
- $(d4 - d1) / h \approx 1.0$。
    - 把 `c` 抬高 `h` 会让输出正好变化 `h`，因为 `c` 是以加法方式进入函数的：`f = ab + c`。
    - 这个变化不会被任何其他值缩放（这等同于说该变化被 `1` 缩放）。

> **每个斜率 `(dx - d1) / h` 都是 `f` 关于某个被抬高了 `h` 的输入的偏导数：**
> 斜率准确地告诉我们，当某一个输入变化一个单位时输出会变化多少，而其他输入保持不变。

### 神经网络中的导数

到目前为止，你对什么是导数、以及如何为非常简单的函数计算导数，已经有了不错的直觉。
你知道一个函数关于其某一个输入的导数告诉你：在该输入变化一个单位时输出会变化多少，而其他输入保持固定。
这就是反向传播背后的核心概念。Micrograd 现在从零实现了反向传播，这让我们能够为各种算术运算计算导数，包括那些与神经网络相关的运算。但这必须以一种通用的方式来完成，这正是我们接下来要看的。

#### Value 类 - 基础搭建

我们想把导数的逻辑迁移到神经网络上去。
为此，我们首先需要合适的数据结构。

`Value` 类接收一个单一的数值并对它进行跟踪。
你可以这样定义值：`a = Value(3.0)` 或 `b = Value(-2.0)`，
但你接着也应该能够执行 `a + b` 或 `a * b`，从而构建出一个由运算和各个值之间关系组成的图。
而基于此，我们最终应该能够求出最终结果相对于初始值的导数。

回到 `Value` 类，我们可以这样定义它：

```python
class Value:
    
    # 对象初始化（构造函数）
    def __init__(self, data):
        self.data = data

    # 告诉如何优雅地打印这个对象  
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Python 特有的加法运算符函数，a+b == a.__add__(b)
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
    
    # 乘法运算符函数
    # 更多参考：https://docs.python.org/3/library/operator.html
    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out
```

这已经让我们可以像这样使用 `Value` 实例了：

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c  # 等价于 a.__mul__(b).__add__(c)

print(d)       # 调用 d.__repr__()，打印 "Value(data=4.0)"
print(d.data)  # d 中实际包含的数值：4.0
```

```
Value(data=4.0)
4.0
```

#### Value 类 - 前向传播

数据的存储与展示，以及乘法和加法，到现在都安排好了。
但我们仍然缺少一种结构，用来知道应用了哪些运算、应用的顺序是什么，以及在算术运算的过程中，`Value` 对象之间形成了哪些连接。**换句话说，我们现在想要一种方式，来记录特定的 `Value` 是*如何* 产生其他 `Value` 的，以及它们彼此之间是如何关联的。**

我们可以通过给 `Value` 类增加一个 `_children` 属性来实现这种跟踪能力。
`_children` 是一个由直接影响当前 `Value` 对象的 `Value` 对象组成的空 `tuple`。
例如，对于 `c = a + b`，`c` 的 `_children` 元组里就会包含 `a` 和 `b`。

**你可能会问自己，为什么我们把这个属性叫 `_children` 而不是 `_parents`。**
这是一个设计选择，更重要的是，这并不是一个错误。
原因在于，我们稍后会从结果到输入*反向* 遍历这张图。
在前向穿过运算时，技术上看起来像是"父节点"的东西，在反向传播时就变成了"子节点"。

```python
class Value:
    
    # 构造函数现在扩展为可选地接收 _children 元组
    def __init__(self, data, _children=()):  # _children 是一个由 Value 组成的元组
        self.data = data                     # 这个 Value 对象中实际包含的数值
        self._prev = set(_children)          # _prev 就是 _children，但被转成 set 以便处理
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # 加法，a+b == a.__add__(b)
    def __add__(self, other):
        # 我们把结果的 _children 初始化为 self 和 other 这两个 Value
        out = Value(self.data + other.data, (self, other))
        return out
    
    # 乘法，a*b == a.__mul__(b)
    def __mul__(self, other):
        # 我们把结果的 _children 初始化为 self 和 other 这两个 Value
        out = Value(self.data * other.data, (self, other))
        return out
```

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c

print(d)        # Value(data=4.0)
print(d._prev)  # {Value(data=-6.0), Value(data=10.0)}
```

```
Value(data=4.0)
{Value(data=-6.0), Value(data=10.0)}
```

我们现在知道了紧邻的前驱 `Value`，即 `_children`，但我们并不知道自己的 `Value` 究竟是*如何* 用这些 `Value` 对象创建出来的。
为了存储这一信息，我们进一步扩展 `Value` 类。
我们添加两个额外的属性：
- 一个运算属性 `_op`，表示连接 `_children` 内部各 `Value` 的运算；
- 一个标签属性 `label`，用于下面的图生成。

`label` 属性纯粹是为后续可视化而用的。

```python
class Value:
    
    # 构造函数现在扩展为可选地接收 _children 元组、_op 字符串以及 label 字符串
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # 加法 (a+b == a.__add__(b))
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+') # 注意这里把 _op 参数设为 '+'
        return out
    
    # 乘法 (a*b == a.__mul__(b))
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a*b; e.label='e'
d = e+c; d.label='d'

f = Value(-2.0, label='f')
L = d*f; L.label = 'L'  # L = (a*b+c) * f

print(L)        # Value(data=-8.0)
print(L._prev)  # {Value(data=-2.0), Value(data=4.0)}
print(L._op)    # *
```

```
Value(data=-8.0)
{Value(data=-2.0), Value(data=4.0)}
*
```

#### Value 类 - 图的生成

我们现在可以追踪 `d` 是由两个 `Value` 相加得到的：`e` 和 `c`，然后再乘以 `f`。而 `e` 和 `c` 本身又是其他 `Value` 经过算术运算的结果。
更一般地说，我们现在可以追溯某个 `Value` 是由哪个 `Value`、经过怎样的运算创建出来的，就像一棵最终汇聚到某个解节点上的树。

**这种树状结构增长得很快。理想情况下，我们希望能有一种方法把这个表达式图可视化出来。**

对应的代码看起来有点吓人，但别担心：

```python
from graphviz import Digraph
# 关于 graphviz 库的更多信息见 https://graphviz.readthedocs.io/en/stable/api.html

# 枚举所有节点和边 -> 为它们构建集合
def trace(root):
    # 构建图中所有节点和边的集合
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# 绘制图
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = 从左到右
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # 为图中的任意一个值，创建一个矩形（'record'）节点
        dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
        if n._op:
          # 如果这个值是某个运算的结果，则为它创建一个 op 节点
          dot.node(name = uid + n._op, label = n._op)
          # 并把这个节点连接到它上面
          dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # 把 n1 连接到 n2 的 op 节点上
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

现在，只需把我们表达式图的最终 `Value` 对象 `L` 交给 `draw_dot` 函数，我们就能把整个决定 `L` 值的表达式图可视化出来：

```python
draw_dot(L)
```

---

### 小结回顾

到目前为止，
- 我们了解了导数是什么、以及如何为简单函数计算导数，
- 我们了解了多输入函数的偏导数，
- 我们可以用 `Value` 对象通过 $+$ 和 $*$ 来**构建数学表达式**，
- 我们能够**跟踪**哪些 `Value` 对象通过什么运算相互连接，并产生一个带有 `_children` 属性和 `_op` 属性的新 `Value`，
- 我们能够把与某个作为根节点的 `Value` 关联的**表达式图可视化**出来

> 目前我们只可视化出了**前向传播**。

接下来，也是最重要的，我们需要讲解**反向传播**过程。

---

#### Value 类 - 搭建反向传播

我们继续沿用上面 `L` 是如何被创建出来的例子。
我们从*前向传播* 的结果（即 `L`）开始。
然后反向地沿着依赖树走，为中间的各个 `Value` 计算梯度。

> 本质上，对于每个 `Value`，我们计算的是 `L` 相对于这个 `Value` 的导数。

`L` 相对于 `L` 自己的导数是 $1$。
改变 `L` 对 `L` 自身产生的影响是严格成比例的，不言而喻。
这很简单，但 `L` 相对于 `f` 等等的导数又是多少呢？

一个 `Value`（如 `L`）相对于对它做出贡献的另一个 `Value`（如 `f`）的导数，被称为**偏导数（partial derivative）**，也就是所谓的**梯度（gradient）**。

> **梯度**是损失函数相对于当前 `Value` 的导数。

对于每个 `Value`，导数都存储在它的 `grad` 属性中。默认情况下，这个属性初始值为 `0`。
对于 `Value` 所参与的每一次算术运算，其 `grad` 属性都会在反向传播中被相应地修改。

```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # 这个值的梯度，初始为 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # 我们希望如何打印这个对象
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # 加法 (a+b == a.__add__(b))
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    # 乘法 (a*b == a.__mul__(b))
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    # Tanh 函数 (e^(2x)-1)/(e^(2x)+1)
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        return Value(t, (self, ), 'tanh')
```

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'

f = Value(-2.0, label='f')
L = d*f; L.label = 'L'

print(L)       # Value(data=-8.0)
print(L._prev) # {Value(data=-2.0), Value(data=4.0)}
print(L._op)   # *
```

```
Value(data=-8.0)
{Value(data=-2.0), Value(data=4.0)}
*
```

我们来看看，把新增的 `grad` 属性按原样可视化到图里会是什么样子：

```python
# 枚举所有节点和边 -> 为它们构建集合
def trace(root):
    # 构建图中所有节点和边的集合
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

# 绘制图
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = 从左到右
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # 为图中的任意一个值，创建一个矩形（'record'）节点
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
          # 如果这个值是某个运算的结果，则为它创建一个 op 节点
          dot.node(name = uid + n._op, label = n._op)
          # 并把这个节点连接到它上面
          dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # 把 n1 连接到 n2 的 op 节点上
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

```python
draw_dot(L)
```

我们刚刚构建了基础的依赖树可视化，其中目前还为空的 `grad` 属性也被一起显示了出来。
但还没有任何反向传播逻辑。

让我们就例子里梯度的若干基本事实，稍微玩一玩：

```python
L.grad = 1.0    # L 相对于自身的梯度总是 1.0
d.grad = f.data # 由于 L = d * f，我们可以确定 dL/dd = f，因为 f 直接缩放了对 d 的改变
f.grad = d.data # 由上面可知，可以说 dL/df = d，因为 d 相应地缩放了对 f 的改变
```

到目前为止都还不错。**但现在，我们来到反向传播的核心了。**

> **如果你理解了接下来的部分，你就在最根本的层面上理解了训练神经网络是如何运作的。**

为了真正触及反向传播的核心，我们引入一些数学记号。
假设我们需要确定 $L$ 相对于 $c$ 的偏导数，即 $\frac{\partial L}{\partial c}$。

我们也假设沿用上面那组示例计算，即：
- $e = a \times b$
- $d = e + c$
- $L = d \times f$

根据我们上面对于这组算术运算已经讨论过的内容，我们知道 `d.grad = f.data`。
也就是说，我们知道 $L$ 相对于 $d$ 的偏导数是 $\frac{\partial L}{\partial d} = f = -2.0$。

现在我要声称，仅仅从这个设定出发，我们就能直接说——不是偏导数，而是 $c$ 和 $e$ 的*局部导数*（local derivative）分别为 $1$。

**怎么做到的？为什么？怎么讲？** *局部导数* 这个术语指的是使用了 $c$ 和 $e$ 的那个紧邻的运算，也就是 $d = c + e$。
直观上，对 $c$ 或 $e$ 施加任意特定大小的改变，会以完全相同的大小影响 $d$，而不会因为另一个参与值的缩放而改变，因为它们之间是通过加法（而非乘法）算术地连接在一起的。
因此，*局部导数*，即"变化的"影响因子"，对于 $c$ 和 $e$ 都是 $1$：

$\frac{\partial d}{\partial c} = 1.0$
$\frac{\partial d}{\partial e} = 1.0$

由于我们的目标是求 $\frac{\partial L}{\partial c}$，我们需要*以某种方式* 把已知的中介结果 $\frac{\partial L}{\partial d}$
与我们新找到的局部导数 $\frac{\partial d}{\partial c}$ 和 $\frac{\partial d}{\partial e}$ 拼接起来，以构造出 $\frac{\partial L}{\partial c}$ 和 $\frac{\partial L}{\partial e}$。

**这是微积分链式法则（chain rule）大显身手的时候。**

链式法则长这样：
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e1a610aa8446be002e2e30d7121f6a87273d4caa)

它指出，函数 $f$ 关于 $x$ 的导数，可以通过把 $f$ 关于某个中间变量 $u$ 的导数乘以 $u$ 关于 $x$ 的导数来计算。
这正是适用于我们情况的内容。我们可以借此把局部导数的信息与已知的 $L$ 关于 $d$ 的偏导数结合起来，计算出 $L$ 关于 $c$ 和 $e$ 的偏导数。

有趣的是，$\frac{\partial d}{\partial c}$ 对最终结果 $\frac{\partial L}{\partial c}$ 没有任何影响，因为它就是 $1.0$。
对于 $\frac{\partial d}{\partial e}$ 及其对 $\frac{\partial L}{\partial e}$ 的影响也是如此。

> 由于相加得到的 $c$ 和 $e$ 具有 *局部导数* $1.0$，我们可以看到，加法运算把梯度均等地路由到被相加的那些值上。换句话说，**加法会把它到此为止所累积的梯度均匀地分配给各个加数。**

最后，我们可以说 $c$ 和 $e$ 通过以下偏导数与 $L$ 关联：
$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \times \frac{\partial d}{\partial c} = -2.0 \times 1.0 = \frac{\partial L}{\partial d} = \underline{\underline{-2.0}}$
$\frac{\partial L}{\partial e} = \frac{\partial L}{\partial d} \times \frac{\partial d}{\partial e} = -2.0 \times 1.0 = \frac{\partial L}{\partial d} = \underline{\underline{-2.0}}$

沿着依赖树继续往上走一步，我们看到 $e$ 是通过 $e = a \times b$ 由 $a$ 和 $b$ 组成的。
现在我们要真正处理乘法了。

在上一步中，我们得知 $\frac{\partial L}{\partial e} = -2.0$。
现在，同样地，我们先来构造*局部导数*：$\frac{\partial e}{\partial a}$ 和 $\frac{\partial e}{\partial b}$。

从 $e = a \times b$ 可以看出，$\frac{\partial e}{\partial a} = b = -3.0$，$\frac{\partial e}{\partial b} = a = 2.0$。

> 两个因子中任一者的变化对结果影响的缩放倍数，完全由另一个因子决定。

有了这个认识，我们再次使用链式法则来计算 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial b}$：
$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \times \frac{\partial e}{\partial a} = -2.0 \times -3.0 = \underline{\underline{6.0}}$
$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} \times \frac{\partial e}{\partial b} = -2.0 \times 2.0 = \underline{\underline{-4.0}}$

把我们关于 $\frac{\partial L}{\partial a}$、$\frac{\partial L}{\partial b}$、$\frac{\partial L}{\partial c}$ 和 $\frac{\partial L}{\partial e}$ 的结果汇总起来，我们现在就可以为图中所有 `Value` 对象填写 `grad` 属性，像这样：

```python
# 从 c 和 e 相加得到 d 的梯度
c.grad = d.grad * (1) # dL/dc = dL/dd * dd/dc = dL/dd * 1
e.grad = d.grad * (1) # dL/de = dL/dd * dd/de = dL/dd * 1

# 从 a 和 b 相乘得到 e 的梯度
a.grad = e.grad * b.data # dL/da = dL/de * de/da = dL/de * b
b.grad = e.grad * a.data # dL/db = dL/de * de/db = dL/de * a

draw_dot(L)
```

我们来看看，当把所有贡献变量沿着各自的梯度方向只更新一小点时，`L` 是如何变化的：

```python
# 沿梯度方向移动叶节点
# 这是梯度上升（gradient ascent，与梯度下降相反）
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

# 前向传播
e = a * b
d = e + c
L = d * f

print(L.data) # L 增大了
```

```
-7.286496
```

**为什么我们看到 `L` 增大了？**

梯度总是指向**最陡上升**的方向。

把一个值沿着梯度方向移动，会通过这个贡献值对 `L` 产生最大可能的增大效果。
把所有值都沿其梯度方向移动，会最大化 `L`。
**因此，把所有值都沿其梯度的精确反方向移动，就会最小化 `L`。**

**注意我们到现在还没有把反向传播过程自动化。**
我们不得不手动计算梯度，然后再手动更新贡献值。
但我们现在已经知道它是怎么运作的了。把这个过程自动化还有一点点内容，我们接下来会在神经网络的应用场景里看。一旦我们这么做了，就能把所有东西串到一起，看看如何只用区区几行代码实现整个反向传播过程。

## 神经网络

到目前为止，我们已经看了如何为简单函数构建表达式图，并且讲解了如何通过反向传播为该图中所有贡献值计算梯度，即通过应用局部导数和偏导数的概念，并经由链式法则把它们关联起来。
**现在，让我们把前向传播和反向传播的概念拿出来，应用到神经网络上。**
**这将是在我们为 micrograd 自动化梯度计算之前的最后一次手动迭代。**

最终，我们想要构建出像下面这样一个能正常工作的神经网络（NN）：
![](https://cs231n.github.io/assets/nn1/neural_net2.jpeg)

首先，我们应该看看定义神经网络内部一个神经元的数学模型：
![](https://cs231n.github.io/assets/nn1/neuron_model.jpeg)

**图中出现了"激活函数"（activation function）这个术语。**
激活函数，也叫非线性函数，被施加在我们神经元输入的加权和上。
这个被激活的值随后作为我们神经元的输出，传给下一层的带权连接。
存在不同的、具有不同性质的激活函数。例如，上面那个就叫 `tanh`。
更一般地说，**向神经网络的神经元引入像 `tanh` 这样的激活函数，使得该网络能够从输入数据中学习/优化更复杂的、*非线性的* 关系。**
为此，激活函数实际上把输入的加权和映射到一个不同取值范围内的值。

作为参考，下面是 `tanh` 和 `sigmoid` 这两种激活函数的样子：

```python
def sigmoid(x):
    a = []
    for i in x:
        a.append(1/(1+math.exp(-i)))
    return a

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Tanh vs. Sigmoid')

lower = -5
upper = 5
step = 0.2

ax1.grid()
ax1.set_title('Tanh')
ax1.plot(np.arange(lower, upper, step), np.tanh(np.arange(-5, 5, 0.2))) # Tanh 激活函数
ax2.grid()
ax2.set_title('Sigmoid')
ax2.plot(np.arange(lower, upper, step), sigmoid(np.arange(-5, 5, 0.2))) # Sigmoid 激活函数（备选）

plt.show();
```

```
（此处省略图表输出）
```

`tanh` 函数把它的数值输入映射到范围 $[-1,1]$，而 `sigmoid` 函数的输出范围是 $[0,1]$。
在我们的例子里，我们将使用 `tanh` 作为神经元的激活函数。

让我们继续，为 $n=2$ 个输入实现 $\text{tanh}(\sum_{i=1}^{n} w_ix_i + b)$，构建一个微型 MLP：

```python
# 输入 x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# 权重 w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# 偏置 b
# 确保后面反向传播算出来的数字好看
b = Value(6.8813735870195432, label='b')

# 神经元值 m：x1w1+x2w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
m = x1w1x2w2 + b; m.label='m'

# 压缩激活 o：tanh(m)
o = m.tanh(); o.label='o'
```

我们实际上实现了下面这个确切的神经元：
![](./img/mlp_2_inp.png)

我们现在可以画出这个神经元输出 `o` 的依赖图：

```python
draw_dot(o)
```

### 手动反向传播

我们已经见过并讲透了反向传播的基本规则。
现在，让我们为这个神经元执行手动反向传播：从 `o` 出发向后走，我们要找出所有贡献变量的梯度。

> 到最后，我们将能回答 `o` 相对于 `x1`、`w1`、`x2` 或 `w2` 中任意一个的导数是多少。

被激活的根值 `o` 相对于自身的梯度总是 $1.0$，正如我们在前面例子里看到的那样。
但 `o` 相对于加权和 `m` 的梯度，即 $\frac{\partial o}{\partial m}$，又是多少呢？
在这里，我们必须穿过激活函数 `tanh` 进行反向传播。

**`tanh` 的局部导数定义为 $1 - \text{tanh}^2(m)$。**
有了它，我们终于可以开始反向传播了：

```python
# 总是已知
o.grad = 1.0

# 问：o = tanh(m)，do/dm 是多少？
# 答：do/dm = 1 - tanh(m)**2，其中 o = tanh(m) 来自前向传播
m.grad = (1 - o.data**2) * o.grad  # （乘 o.grad 只是为了展示链式法则；并非必需）

# 问：m = x1w1x2w2 + b，do/dx1w1x2w2 和 do/db 是多少？
# 答：由于加法的导数就是 1.0，它通过链式法则把梯度分裂开来
#    我们实际上可以用这种方式把从 m 链过来的梯度传给 x1w1x2w2 和 b
x1w1x2w2.grad = 1.0 * m.grad 
b.grad = 1.0 * m.grad

# 问：x1w1x2w2 = x1w1 + x2w2，do/dx1w1 和 do/dx2w2 是多少？
# 答：再一次，加法通过链式法则把 x1w1x2w2 的梯度分裂开来；我们只需把它传下去
x1w1.grad = 1.0 * x1w1x2w2.grad
x2w2.grad = 1.0 * x1w1x2w2.grad

# 问：x1w1 = x1*w1，do/dx1 和 do/dw1 是多少？
# 答：乘法稍微复杂一点，我们应用乘积法则
#    对于 w1 * x1，w1 是 x1 的局部导数，x1 是 w1 的局部导数
#    我们通过链式法则把局部导数与 x1w1 的梯度联系起来
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

# 问：x2w2 = x2*w2，do/dx2 和 do/dw2 是多少？
# 答：同样，对 x1、w1 应用与上面相同的乘积法则
#    同样地，我们把局部导数相应地与 x2w2 的梯度联系起来
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad # 这会变成 0；改变 w2 没有任何影响，因为 x2 是 0，与 w2 相乘
```

### 自动反向传播

**手动做反向传播是新手才干的事。**
我们要把它自动化并一般化。

为此，我们需要相当大程度地改写并扩展 `Value` 类。
更确切地说，我们必须给 `Value` 对象加上一个 `_backward` 属性，我们可以把它实现为一个用 `None` 初始化的 [lambda 表达式](https://www.bomberbot.com/python/lambda-expressions-in-python-a-comprehensive-guide/)。
对于每个运算，`_backward` 都会被填入一个具体的梯度计算步骤。

下面这个修订过的 `Value` 类把这一概念实现得非常优美：

```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0                # 该 Value 的梯度，初始为 0.0 
        self._backward = lambda: None  # 默认什么也不做，持有把梯度反向传给 _children 的函数
        self._prev = set(_children)    # 做出贡献的 _children Value，转成 set
        self._op = _op                 # 产生该 Value 的运算符
        self.label = label             # 用于 graphviz 的标签
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # 加法 (a+b == a.__add__(b))
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # 把从 out 来的梯度路由给 self 和 other
            # 加法对两者的局部导数都是 1.0
            # 通过链式法则把局部导数与 out 的梯度联系起来
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward # 把梯度计算行为设为 lambda 表达式
        return out
     
    # 乘法 (a*b == a.__mul__(b))
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # 把从 out 来的梯度路由给 self 和 other
            # 乘法的局部导数对 self.grad 是 other.data，对 other.grad 是 self.data
            # 同样地，通过链式法则与 out 的梯度联系起来
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        
        out._backward = _backward
        return out
    
    # tanh 激活函数被当作它自己的一种运算来处理
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            # tanh 的局部导数乘以从子 Value 传进来的梯度
            self.grad = (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
```

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'

f = Value(-2.0, label='f')
L = d*f; L.label = 'L'

print(L)       # Value(data=-8.0)
print(L._prev) # {Value(data=-2.0), Value(data=4.0)}
print(L._op)   # *
```

```
Value(data=-8.0)
{Value(data=-2.0), Value(data=4.0)}
*
```

现在，我们用这个新的 `Value` 类对早先那个简单的 MLP 做一次"前向传播"：

```python
# 输入
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# 权重
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# 偏置
b = Value(6.8813735870195432, label='b') # 确保后面反向传播算出来的数字好看

## 前向传播 
# 目标：求单个输出神经元 'm' 的值
# 神经元值 m = x1*w1+x2*w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
m = x1w1x2w2 + b; m.label='m'

# 压缩激活：tanh(m)
o = m.tanh(); o.label='o'

draw_dot(o)
```

最后，我们不再需要手动进行反向传播了（除了那个永远一样的 `o.grad`，请忽略它）。
反向传播时请记住运算的精确顺序：

```python
o.grad = 1.0  # 让反向传播相乘生效的基例
o._backward()
m._backward()
# b._backward()  # 偏置项 b 不向任何更高层 Value 传播，它只是叶节点，所以不用运行这个
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
# x1._backward() # 不需要运行这些，理由同上 b
# w1._backward()
# x2._backward()
# w2._backward()

draw_dot(o)
```

**我们还有最后一件事要摆脱：** 必须以这种特定的、繁琐的、与 `Value` 计算出现顺序相反的顺序手动调用 `_backward()`。

沿表达式图反向走，意味着对于图中的每个节点，所有在它之后（更靠近根）的东西都必须已经计算好。
*这需要顺序。* **要为图的节点确立一种顺序，我们可以使用拓扑排序（topological sort）。**
拓扑排序把依赖图的节点排成一种边总是指向*同一个公共方向* 的排列：
![](https://assets.leetcode.com/users/images/63bd7ad6-403c-42f1-b8bb-2ea41e42af9a_1613794080.8115625.png)

由拓扑排序产生的顺序保证了梯度的计算和传播方式避免了冗余计算，并最大化了计算并行度。
这反过来通过让梯度在网络中最顺畅地流动，带来了更高的效率，从而加速了处理和收敛速度。[更多关于拓扑排序……](https://www.geeksforgeeks.org/topological-sorting/)

```python
# 拓扑排序
# 把图整理成可以按尊重依赖的顺序遍历它的结构

topo = []
visited = set()

def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v) # 只有当所有前驱节点都被处理后，才添加该节点

build_topo(o)

for t in topo:
    print(f'{t.label:13} : {t}')
```

```
b             : Value(data=6.881373587019543)
x2            : Value(data=0.0)
w2            : Value(data=1.0)
x2*w2         : Value(data=0.0)
w1            : Value(data=-3.0)
x1            : Value(data=2.0)
x1*w1         : Value(data=-6.0)
x1*w1 + x2*w2 : Value(data=-6.0)
m             : Value(data=0.8813735870195432)
o             : Value(data=0.7071067811865476)
```

这就是我们要对图应用 `_backward()` 的*精确* 顺序，从根 `o` 开始。

```python
o.grad = 1.0

topo = []
visited = set()

def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v) # 只有当所有节点都被处理后，才添加该节点
build_topo(o)

for node in reversed(topo):
    node._backward()

draw_dot(o)
```

我们现在就把这种拓扑排序的逻辑实现到 `Value` 类里。
**这是更新后的 `Value` 类结构：**

```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0                # 该 Value 的梯度，初始为 0.0 
        self._backward = lambda: None  # 默认什么也不做，持有把梯度反向传给 _children 的函数
        self._prev = set(_children)    # 做出贡献的 _children Value，转成 set
        self._op = _op                 # 产生该 Value 的运算符
        self.label = label             # 用于 graphviz 的标签
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # 加法 (a+b == a.__add__(b))
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    # 乘法 (a*b == a.__mul__(b))
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data
        
        out._backward = _backward
        return out
    
    # tanh 激活函数
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad = (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) # 只有当所有子节点都被处理后，才添加该节点
        
        # 构建本 Value 上游/导致本 Value 的计算图中所有 Value 的拓扑序
        build_topo(self)
        
        self.grad = 1.0 # 根节点的梯度总是 1.0
        
        for node in reversed(topo):
            # 按反向拓扑序遍历所有 Value，
            # 应用它们的链式法则来填充每个 Value 的 .grad 属性
            # 如果当前 Value 不是根节点，这会覆盖上面的 self.grad = 1.0
            node._backward()
```

```python
# 输入 x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# 权重 w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# 偏置 b
b = Value(6.8813735870195432, label='b') # 确保后面反向传播算出来的数字好看

# 前向传播
# 神经元值 m：x1w1+x2w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
m = x1w1x2w2 + b; m.label='m'

# 压缩激活 o：tanh(m)
o = m.tanh(); o.label='o'

draw_dot(o)
```

现在我们只需调用 `o.backward()`，整个反向传播过程就会自动执行，
对于我们这个简单的 MLP，我们完全不必再操心运算顺序：

```python
o.backward()
draw_dot(o)
```

#### Value 类 - 找 Bug 与扩展

我们刚刚构建了反向传播。至少是在一个简单的单神经元设定下，为*一个* 神经元 `o` 做的。**但这里仍然存在一个相当严重的 bug。**

这个 bug 在下面这个简单例子中浮现出来：

```python
a = Value(3.0, label='a')
b = a + a; b.label = 'b'
b.backward()
draw_dot(b)
```

即便我们施加的是加法，梯度也应该是 $2$，因为 `a + a` 和 `2 * a` 是一样的。
**这种错误行为在这里依然存在：**

```python
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b; d.label='d'
e = a + b; e.label='e'
f = d * e; f.label='f'

f.backward()
draw_dot(f)
```

**一旦一个 `Value` 在依赖图中被使用超过一次，就会出现一致性问题。**
而这在真实场景的例子里是大多数时候都会发生的情况。

> **实际上，我们需要在 `_backward` 函数里*累加*（`+=`）梯度，而不是*设置/覆盖*（`=`）它们。**

而且，*既然都到这一步了*，让我们进一步扩展 `Value` 类。
例如，我们做不了 `a = Value(2.0) + 1.0` 或 `a = Value(2.0) * 2.0`。
另外，我们还要把除法以及 `tanh` 背后那些详细的运算也整合进来。

修复并扩展后的 `Value` 类长这样（看 `_backward` 函数里针对 `+=` 的 bug 修复）：

```python
class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # 默认什么也不做
        self._prev = set(_children)
        self._op = _op
        self.label = label
            
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # 加法 (a+b == a.__add__(b))
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # 扩展
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad  # Bug 修复
            other.grad += 1.0 * out.grad # Bug 修复
        
        out._backward = _backward
        return out
 
    # 乘法 (a*b == a.__mul__(b))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # 扩展
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += out.grad * other.data # Bug 修复
            other.grad += out.grad * self.data # Bug 修复
        
        out._backward = _backward
        return out
    
    # 取负（当作特殊乘法处理）
    def __neg__(self): # -self
        return -1 * self
    
    # 减法（当作特殊加法处理）
    def __sub__(self, other): # self - other
        return self + (-other)

    # 幂运算（当作特殊乘法处理）
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers (for now)"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    # 当 self 在 * 右侧时被调用
    def __rmul__(self, other): # other * self
        return self * other
    
    # 当 self 在 + 右侧时被调用
    def __radd__(self, other): # other + self
        return self + other
    
    # 真除法（当作特殊乘法处理）
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    # tanh 激活函数
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad # Bug 修复
        
        out._backward = _backward
        return out
    
    # 指数函数
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) # 只有当所有节点都被处理后，才添加该节点
        build_topo(self)
        
        self.grad = 1.0 # 种子梯度总是 1.0
        
        for node in reversed(topo):
            node._backward()
```

```python
a = Value(-2.0, label='a')
b = Value(3.0, label='b')
d = a * b; d.label='d'
e = a + b; e.label='e'
f = d * e; f.label='f'

f.backward()
draw_dot(f)
```

```python
# 对新实现的算术运算做合理性检查

a = Value(2.0)
b = Value(4.0)

print(a + 2)
print(2 + a)
print(a * 2)
print(2 * a)
print(-a)
print(a - b)
print()
print(a.exp())
print(a / b) # 除法：a/b = a * (1/b) = a * (b**(-1))，所以我们用一个实现 x**k 的函数
```

```
Value(data=4.0)
Value(data=4.0)
Value(data=4.0)
Value(data=4.0)
Value(data=-2.0)
Value(data=-2.0)

Value(data=7.38905609893065)
Value(data=0.5)
```

### 一切汇聚于此

我们现在要从上面一直使用的单神经元例子里走出来了。
更确切地说，我们要用现在已经可用的、`tanh` 的详细前向和反向运算来改变 `o` 的定义方式。

```python
# 输入 x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# 权重 w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# 偏置 b
b = Value(6.8813735870195432, label='b') # 确保后面反向传播算出来的数字好看

# 神经元值 m：x1w1+x2w2 + b
x1w1 = x1*w1; x1w1.label='x1*w1'
x2w2 = x2*w2; x2w2.label='x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1*w1 + x2*w2'
m = x1w1x2w2 + b; m.label='m'

# 压缩激活 o：tanh(m) 现在显式实现
e = (2*m).exp()
o = (e - 1)/(e + 1); o.label='o'

draw_dot(o)
```

```python
# x1, x2,... 的梯度应当保持不变
o.backward()
draw_dot(o)
```

**我们刚刚为什么这么做？**
我们本质上是改变了这一轮实现的抽象层次。
到底是只实现一个 `tanh`，还是继续用 `Value` 对象显式地实现该函数的原子步骤，这由我们决定。
到现在，大概已经很清楚了：构建这个简单的自动求导引擎，主要是一项处理算术运算、随着其演进在实现的不同抽象层次之间切换、并最终稳健地推导出反向传播所需梯度的练习。
**我们已经搭建好了训练神经网络的基础！**

---

## 用 PyTorch API 做同样的事

[原版 micrograd 项目](https://github.com/karpathy/micrograd) 大致上是以 PyTorch 的语法为模板的。
事实上，它的核心逻辑同样也可以用 PyTorch 来实现。
我们接下来要进入的这种 PyTorch 原生方式，乍看可能有点凌乱，因为它要求把值存储在 PyTorch 的 Tensor 对象里。

```python
x1 = torch.Tensor([2.0]).double();  x1.requires_grad = True  # 单元素 tensor
x2 = torch.Tensor([0.0]).double();  x2.requires_grad = True  # tensor 数据类型现在是 double
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True  # 默认 dtype 是 float32
w2 = torch.Tensor([1.0]).double();  w2.requires_grad = True  # 现在它是 float64 即 double
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True

m = x1*w1 + x2*w2 + b # 执行算术运算，构建依赖图，和 micrograd 一样
o = torch.tanh(m)

print(o.data.item())
o.backward() # backward() 是 pytorch 内置的自动求导函数

print('---') # 下面这些值可以直接拿来与 micrograd 的 x1.grad、w1.grad 等做比较
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
```

```
0.7071066904050358
---
x2 0.5000001283844369
w2 0.0
x1 -1.5000003851533106
w1 1.0000002567688737
```

```python
o.item() # 从 tensor o 里取出标量值
```

```
0.7071066904050358
```

> 使用 PyTorch 的关键优势在于，得益于底层大量的优化，它能让计算**显著更高效**。在实践中，你通常会用 PyTorch 来做真实的应用。然而，*理解反向传播背后的逻辑* 依然很重要，而这正是我们迄今为止用 Micrograd 和我们的 `Value` 类所探索的内容。

## 回到神经网络

既然我们已经有了一些可以用来建模复杂数学表达式的工具，我们就可以构建*分层的神经网络* 了。
我们会一块一块地做，最终得到一个两层感知机（MLP）。

为了完整起见，这里再放一次 MLP 的示意图：

![](https://cs231n.github.io/assets/nn1/neural_net2.jpeg)

```python
# 一个神经元能接收多个输入，并产生一个激活标量
class Neuron:
    def __init__(self, nin):
        # nin -> 该神经元的输入个数
        # 每个输入一个随机权重 [-1, 1]
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        # 偏置控制神经元整体的"触发积极性"
        self.b = Value(np.random.uniform(-1,1))
        
    def __call__(self, x): # 运行 neuron(x) -> 触发 __call__
        # w * x + b
        # zip() 创建一个在两个迭代器的元组上逐项运行的迭代器
        # self.b 作为求和的起始值，然后在其上累加
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # 用 tanh 压缩激活值
        out = act.tanh()
        return out
    
    # 收集该神经元参数列表的便捷代码
    def parameters(self):
        return self.w + [self.b]


# 一组神经元构成一个（隐藏/输入/输出）NN 层
# 例如 n = Layer(2, 3) -> 3 个二维神经元
class Layer:
    # nout -> 这一层应该有多少个神经元/输出
    # nin -> 每个神经元预期有多少个输入
    def __init__(self, nin, nout):
        # 字面上就是按需创建一个神经元列表
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x): # 运行 layer(x) -> 触发 __call__
        # 返回该层所有神经元的激活值
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    # 收集该层所有神经元参数的便捷代码
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


# MLP -> 多层感知机 -> NN
class MLP:
    # nin -> 该 NN 的输入个数
    # nouts -> 数字列表，定义所有想要的层的大小
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x): # mlp(x) -> 在 NN 中调用所有的 layer(x) 值
        for layer in self.layers:
            # 整洁的前向传播实现
            x = layer(x)
        return x
    
    # 收集所有层所有神经元参数的便捷代码
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

```python
x = [2.0, 3.0, -1.0]  # 输入值
n = MLP(3, [4, 4, 1]) # 3 个输入进入两个各有 4 个神经元的层，再加一个输出层
print(n(x))
```

```
Value(data=0.4798905757861232)
```

```python
draw_dot(n(x))
```

借助 micrograd，我们现在能够轻松地反向传播穿过这一大坨东西。
**让我们就在一个训练场景里真正这么做一次。**首先，我们定义一个带有特征和标签的示例训练集：

```python
# 特征/输入
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

# 期望目标
ys = [1.0, -1.0, -1.0, 1.0]

# 取出 NN 当前对 xs 的预测
ypred = [n(x) for x in xs]

for i in range(len(ypred)):
    print(f'{ypred[i]}\t --> {ys[i]}')
```

```
Value(data=0.4798905757861232)	 --> 1.0
Value(data=-0.28379942981470657)	 --> -1.0
Value(data=0.0063444255569660105)	 --> -1.0
Value(data=0.44668279025600416)	 --> 1.0
```

**神经网络此时表现并不好。这是因为它还没有被训练过。**
我们需要衡量神经网络表现得好/坏的程度，以便朝着提升它性能的方向迈步。

要衡量预测到底有多好/多坏，**我们需要一个损失函数（loss function）。**

```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
print('loss:', loss.data)
```

```
loss: 2.102346107338291
```

随着训练进行，这个损失函数输出的损失需要变得尽可能低。
更低的损失表明预测值与真实标签之间的差距更小。

```python
loss.backward()
```

```python
# 示例：现在已算出梯度的神经元权重
print(n.layers[0].neurons[0].w[0].grad) # 第一层第一个神经元的第一个权重的梯度
print(n.layers[0].neurons[0].w[0].data) # 第一层第一个神经元的第一个权重的值
```

```
0.03096638432575599
0.39461898950675023
```

```python
# 用反向传播得到的梯度更新权重
for p in n.parameters(): 
    p.data += -0.01 * p.grad # 沿梯度的反方向移动一小点，以免对这一个样本过拟合
```

```python
# 显示更新后的权重
print(n.layers[0].neurons[0].w[0].data)
```

```
0.3943093256634927
```

好了。我们现在有了一个损失函数来衡量神经网络的表现。
`loss` 这一项直接与参数相连，也就是 MLP 各层的权重和偏置的激活值。
如果我们计算梯度，我们确定的是
为了让损失*最大化*，我们需要对所有这些权重和偏置做出改变的方向和强度。

是的，是*最大化*。

而这恰恰是我们在这里不想要的。
我们想要*最小化* 损失。为此，我们不加上梯度，而是在把它们乘以 $0.01$ 的系数后减去它们。
通过把每个参数沿其梯度反方向移动一小点，我们让损失降低一点点。
保持这一沿梯度的步长很小，可以防止模型对任何一个训练样本反应过度。

这个过程被称为**梯度下降（gradient descent）**，它是训练神经网络的核心过程。

**还有一件事。**

通常，训练期间网络不会只接触数据集一次。
相反，训练会让网络多次遇到每个数据点。
完整地遍历一遍训练数据集被称为一个**轮次（epoch）**。

我们现在可以为我们的网络训练 $5$ 个轮次：

```python
# 运行轮次并显示相应的预测
for t in range(5):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    print(f'Epoch {t}\t - Loss: {loss.data}\t - Predictions: {[y.data for y in ypred]}')

    loss.backward()
    for p in n.parameters(): 
        p.data += -0.01 * p.grad
```

```
Epoch 0	 - Loss: 1.900763610107178	 - Predictions: [0.5000250274166136, -0.3358250283470125, -0.03863695569175478, 0.4657328936144803]
Epoch 1	 - Loss: 1.5317123524466187	 - Predictions: [0.5386788076093043, -0.43291967500948697, -0.13414012061861533, 0.5024040954909761]
Epoch 2	 - Loss: 1.074383854751221	 - Predictions: [0.5926421288880276, -0.5585994023380326, -0.28258999408538277, 0.5539822700927804]
Epoch 3	 - Loss: 0.6466374361457443	 - Predictions: [0.6569231440955614, -0.6878963700947666, -0.4667313238552082, 0.6163966236707142]
Epoch 4	 - Loss: 0.34179405916553623	 - Predictions: [0.7246830507435083, -0.7974699779955747, -0.6463524347079066, 0.6839151595417836]
```

**恭喜你，你刚刚学会了如何从零开始构建并训练一个神经网络！**

---

<center>Notebook 作者：<a href="https://github.com/mk2112" target="_blank">mk2112</a>。</center>
