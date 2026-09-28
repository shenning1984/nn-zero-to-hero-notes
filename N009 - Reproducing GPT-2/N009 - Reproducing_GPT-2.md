# 从零复现 GPT-2

[视频](https://www.youtube.com/watch?v=l8pRSuU81PU)<br>
[Build-NanoGPT 仓库](https://github.com/karpathy/build-nanogpt)<br>
[NanoGPT 仓库](https://github.com/karpathy/nanogpt)<br>
[勘误](https://github.com/karpathy/build-nanogpt?tab=readme-ov-file#errata)<br>
[讨论区](https://github.com/karpathy/build-nanogpt/discussions)<br>
[Eureka Labs Discord](https://discord.com/invite/3zy8kqD9Cp)

## 目录

- [GPT-2 架构概览](#gpt-2-架构概览)
  - [注意力就是全部所需吗？](#注意力就是全部所需吗)
  - [配置](#配置)
  - [GPT 类](#gpt-类)
  - [Block 类](#block-类)
  - [MLP 类](#mlp-类)
  - [CausalSelfAttention 类](#causalselfattention-类)
  - [分词](#分词)
- [重置权重](#重置权重)
- [构造分词后的训练输入](#构造分词后的训练输入)
- [计算损失](#计算损失)
- [优化](#优化)
  - [合理性检查](#合理性检查)
  - [数据加载](#数据加载-1)
- [修 Bug](#修-bug)
  - [Token 嵌入层和线性输出层竟然是同一个？！](#token-嵌入层和线性输出层竟然是同一个)
  - [权重初始化](#权重初始化)
- [优化模型训练](#优化模型训练)
  - [NVIDIA Tensor Core 与 TensorFloat32](#nvidia-tensor-core-与-tensorfloat32)
  - [减少显存搬运](#减少显存搬运)
  - [PyTorch 编译](#pytorch-编译)
  - [FlashAttention 及编译的一个坑](#flashattention-及编译的一个坑)
  - [去掉"丑数"](#去掉丑数)
  - [算法层面的适配](#算法层面的适配)
    - [Adam / AdamW 优化器配置](#adam--adamw-优化器配置)
    - [全局范数裁剪](#全局范数裁剪)
    - [学习率衰减](#学习率衰减)
    - [动态增大批大小](#动态增大批大小)
    - [数据加载](#数据加载-2)
    - [权重衰减与 Fused AdamW](#权重衰减与-fused-adamw)
    - [用梯度累积模拟大批大小](#用梯度累积模拟大批大小)
- [多 GPU 训练](#多-gpu-训练)
  - [引入 DistributedDataParallel](#引入-distributeddataparallel)
  - [适配梯度累积](#适配梯度累积)
  - [适配 DataLoader](#适配-dataloader)
  - [完整接入 DistributedDataParallel](#完整接入-distributeddataparallel)
  - [分发训练循环](#分发训练循环)
- [我们确实需要更多数据](#我们确实需要更多数据)
  - [DataLoader 调整](#dataloader-调整)
- [评估、日志与可视化](#评估日志与可视化)
  - [HellaSwag 评估](#hellaswag-评估)
  - [可视化训练进度](#可视化训练进度)
- [进一步优化与增强的空间](#进一步优化与增强的空间)

<br>

---

2019 年，OpenAI 发布了其生成式预训练 Transformer 语言模型架构（GPT）的第二版。<br><br>
这次发布包括：
- [一篇博文](https://openai.com/blog/better-language-models/)，
- [一篇研究论文 \[Radford et al. 2019\]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，以及
- [一个 GitHub 仓库](https://github.com/openai/gpt-2)。

此外，GPT-2 并非作为单一模型发布，而是作为一组四个模型发布，每个尺寸不同：<br>
- `124M`（1.24 亿参数），
- `355M`，
- `774M`，以及
- `1558M`。

将 GPT-2 作为一组不同尺寸的模型发布，便于做实验比较不同模型规模在某些任务上的性能差异。<br>
这些结果有助于说明要获得某些能力所需的容量，也能暗示<br>
未来在哪个方向上做设计改进最能带来进一步提升。

**在本笔记本中，我们将复现 GPT-2 `124M` 模型，之后你可以很容易地把它扩展到任意更大的尺寸。**<br>
令人瞩目的是，GPT-2 在 2019 年是当时最先进的，而如今你只需约 1 小时、约 10 美元就能训练它（如果你租些 GPU 的话）。<br>
为确保训练过程的准确性，我们不仅会参考 GPT-2 论文 [[Radford et al. 2019]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，还会看 [[Brown et al. 2020]](https://arxiv.org/abs/2005.14165)——引入后续模型系列 GPT-3 的那篇论文。

## GPT-2 架构概览

原论文对 GPT-2 模型系列的描述如下：<br>
![](./img/gpt2_table_sm.png)<br>
来源：[\[Radford et al. 2019\]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**表格把 `124M` 模型称作 `117M`**，因为当时 OpenAI 一开始其实把参数数错了。<br>
不过表格正确指出，小模型有 $12$ 个 transformer 层、隐藏大小为 $768$ 个单元。

**对我们而言，这意味着：**
- 小号 GPT-2 模型由 $12$ 个 transformer 解码器块依次堆叠而成，信息逐块流过
    - 每个块各自包含带掩码的多头注意力、前馈层和层归一化
    - 每个块内部，在归一化后的注意力层和前馈层周围使用了残差连接
    - 每个块的总输出传递给下一个块，作为它的输入
- 每个 transformer 块还有 $768$ 个隐藏单元
    - $768$ 这个值代表每一层内部嵌入的维度，也就是我们把 token 映射进去以表示其含义的空间
    - $768$ 也设定了自注意力机制和前馈层的输入/输出向量大小
    - 换句话说，**隐藏单元数代表了模型跨层的带宽**

如果以上任何一点不清楚，[别担心](https://cdn-images-1.medium.com/v2/resize:fit:2560/1*yIPIuNIn6ar7MvQnNqlWlQ.jpeg)。<br>
详情见课程 [N007 - GPT From Scratch](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb)。

我们从标准 transformer 架构出发。GPT-2 模型一开始<br>
看起来有点像解码器部分（不完全一样，后面会讲）：<br>
<br>
![](./img/gpt2_layout_general.png)<br>
来源：[dzlab.github.io](https://dzlab.github.io/ml/2020/07/25/gpt3-overview/)

原始的 [GPT-2 仓库](https://github.com/openai/gpt-2) 发布的是 TensorFlow 模型实现。<br><br>
我们不看它，而是坚持用 PyTorch。为此，我们来看 [HuggingFace 的 GPT-2 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)。<br>
可读性稍微好一点。*稍微。*

不过我们现在还不想通读那个文件。
HuggingFace 为其模型提供了相当漂亮的抽象，所以我们可以直接用它来窥探一下他们的 GPT-2 实现：

```python
from transformers import GPT2LMHeadModel
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M parameters, for "gpt2-xl" the gold model
sd_hf = model_hf.state_dict() # Dictionary of the model's weight/bias tensors

for identifier, tensor in sd_hf.items():
    print(identifier, tensor.shape)

# Print some GPT-2 weights
print('\n', sd_hf["transformer.wpe.weight"].view(-1)[:20])
```
```
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])
transformer.h.0.ln_1.weight torch.Size([768])
transformer.h.0.ln_1.bias torch.Size([768])
transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.0.attn.c_attn.bias torch.Size([2304])
transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias torch.Size([768])
transformer.h.0.ln_2.weight torch.Size([768])
transformer.h.0.ln_2.bias torch.Size([768])
transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.0.mlp.c_fc.bias torch.Size([3072])
transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.0.mlp.c_proj.bias torch.Size([768])
transformer.h.1.ln_1.weight torch.Size([768])
transformer.h.1.ln_1.bias torch.Size([768])
transformer.h.1.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.1.attn.c_attn.bias torch.Size([2304])
transformer.h.1.attn.c_proj.weight torch.Size([768, 768])
transformer.h.1.attn.c_proj.bias torch.Size([768])
transformer.h.1.ln_2.weight torch.Size([768])
transformer.h.1.ln_2.bias torch.Size([768])
transformer.h.1.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.1.mlp.c_fc.bias torch.Size([3072])
transformer.h.1.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.1.mlp.c_proj.bias torch.Size([768])
transformer.h.2.ln_1.weight torch.Size([768])
transformer.h.2.ln_1.bias torch.Size([768])
transformer.h.2.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.2.attn.c_attn.bias torch.Size([2304])
transformer.h.2.attn.c_proj.weight torch.Size([768, 768])
transformer.h.2.attn.c_proj.bias torch.Size([768])
transformer.h.2.ln_2.weight torch.Size([768])
transformer.h.2.ln_2.bias torch.Size([768])
transformer.h.2.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.2.mlp.c_fc.bias torch.Size([3072])
transformer.h.2.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.2.mlp.c_proj.bias torch.Size([768])
transformer.h.3.ln_1.weight torch.Size([768])
transformer.h.3.ln_1.bias torch.Size([768])
transformer.h.3.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.3.attn.c_attn.bias torch.Size([2304])
transformer.h.3.attn.c_proj.weight torch.Size([768, 768])
transformer.h.3.attn.c_proj.bias torch.Size([768])
transformer.h.3.ln_2.weight torch.Size([768])
transformer.h.3.ln_2.bias torch.Size([768])
transformer.h.3.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.3.mlp.c_fc.bias torch.Size([3072])
transformer.h.3.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.3.mlp.c_proj.bias torch.Size([768])
transformer.h.4.ln_1.weight torch.Size([768])
transformer.h.4.ln_1.bias torch.Size([768])
transformer.h.4.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.4.attn.c_attn.bias torch.Size([2304])
transformer.h.4.attn.c_proj.weight torch.Size([768, 768])
transformer.h.4.attn.c_proj.bias torch.Size([768])
transformer.h.4.ln_2.weight torch.Size([768])
transformer.h.4.ln_2.bias torch.Size([768])
transformer.h.4.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.4.mlp.c_fc.bias torch.Size([3072])
transformer.h.4.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.4.mlp.c_proj.bias torch.Size([768])
transformer.h.5.ln_1.weight torch.Size([768])
transformer.h.5.ln_1.bias torch.Size([768])
transformer.h.5.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.5.attn.c_attn.bias torch.Size([2304])
transformer.h.5.attn.c_proj.weight torch.Size([768, 768])
transformer.h.5.attn.c_proj.bias torch.Size([768])
transformer.h.5.ln_2.weight torch.Size([768])
transformer.h.5.ln_2.bias torch.Size([768])
transformer.h.5.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.5.mlp.c_fc.bias torch.Size([3072])
transformer.h.5.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.5.mlp.c_proj.bias torch.Size([768])
transformer.h.6.ln_1.weight torch.Size([768])
transformer.h.6.ln_1.bias torch.Size([768])
transformer.h.6.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.6.attn.c_attn.bias torch.Size([2304])
transformer.h.6.attn.c_proj.weight torch.Size([768, 768])
transformer.h.6.attn.c_proj.bias torch.Size([768])
transformer.h.6.ln_2.weight torch.Size([768])
transformer.h.6.ln_2.bias torch.Size([768])
transformer.h.6.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.6.mlp.c_fc.bias torch.Size([3072])
transformer.h.6.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.6.mlp.c_proj.bias torch.Size([768])
transformer.h.7.ln_1.weight torch.Size([768])
transformer.h.7.ln_1.bias torch.Size([768])
transformer.h.7.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.7.attn.c_attn.bias torch.Size([2304])
transformer.h.7.attn.c_proj.weight torch.Size([768, 768])
transformer.h.7.attn.c_proj.bias torch.Size([768])
transformer.h.7.ln_2.weight torch.Size([768])
transformer.h.7.ln_2.bias torch.Size([768])
transformer.h.7.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.7.mlp.c_fc.bias torch.Size([3072])
transformer.h.7.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.7.mlp.c_proj.bias torch.Size([768])
transformer.h.8.ln_1.weight torch.Size([768])
transformer.h.8.ln_1.bias torch.Size([768])
transformer.h.8.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.8.attn.c_attn.bias torch.Size([2304])
transformer.h.8.attn.c_proj.weight torch.Size([768, 768])
transformer.h.8.attn.c_proj.bias torch.Size([768])
transformer.h.8.ln_2.weight torch.Size([768])
transformer.h.8.ln_2.bias torch.Size([768])
transformer.h.8.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.8.mlp.c_fc.bias torch.Size([3072])
transformer.h.8.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.8.mlp.c_proj.bias torch.Size([768])
transformer.h.9.ln_1.weight torch.Size([768])
transformer.h.9.ln_1.bias torch.Size([768])
transformer.h.9.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.9.attn.c_attn.bias torch.Size([2304])
transformer.h.9.attn.c_proj.weight torch.Size([768, 768])
transformer.h.9.attn.c_proj.bias torch.Size([768])
transformer.h.9.ln_2.weight torch.Size([768])
transformer.h.9.ln_2.bias torch.Size([768])
transformer.h.9.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.9.mlp.c_fc.bias torch.Size([3072])
transformer.h.9.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.9.mlp.c_proj.bias torch.Size([768])
transformer.h.10.ln_1.weight torch.Size([768])
transformer.h.10.ln_1.bias torch.Size([768])
transformer.h.10.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.10.attn.c_attn.bias torch.Size([2304])
transformer.h.10.attn.c_proj.weight torch.Size([768, 768])
transformer.h.10.attn.c_proj.bias torch.Size([768])
transformer.h.10.ln_2.weight torch.Size([768])
transformer.h.10.ln_2.bias torch.Size([768])
transformer.h.10.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.10.mlp.c_fc.bias torch.Size([3072])
transformer.h.10.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.10.mlp.c_proj.bias torch.Size([768])
transformer.h.11.ln_1.weight torch.Size([768])
transformer.h.11.ln_1.bias torch.Size([768])
transformer.h.11.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.11.attn.c_attn.bias torch.Size([2304])
transformer.h.11.attn.c_proj.weight torch.Size([768, 768])
transformer.h.11.attn.c_proj.bias torch.Size([768])
transformer.h.11.ln_2.weight torch.Size([768])
transformer.h.11.ln_2.bias torch.Size([768])
transformer.h.11.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.11.mlp.c_fc.bias torch.Size([3072])
transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.11.mlp.c_proj.bias torch.Size([768])
transformer.ln_f.weight torch.Size([768])
transformer.ln_f.bias torch.Size([768])
lm_head.weight torch.Size([50257, 768])

 tensor([-0.0188, -0.1974,  0.0040,  0.0113,  0.0638, -0.1050,  0.0369, -0.1680,
        -0.0491, -0.0565, -0.0025,  0.0135, -0.0042,  0.0151,  0.0166, -0.1381,
        -0.0063, -0.0461,  0.0267, -0.2042])
```

这个输出**乍一看似乎没告诉我们多少东西。**<br>
但**可以像这样理解上面的输出：**
- `wte` 是 **token 嵌入的权重矩阵**， 
- `wpe` 是 **位置嵌入的权重矩阵**， 
- `h` 是 **隐藏状态**。 
- 其余的是 transformer 块。

从 `wte` 的维度来看，我们可以说 GPT-2 的词表大小为 $50257$ 个 token，每个 token 被映射到一个大小为 $768$ 的表示嵌入向量。<br>
此外，序列中每个 token 最多可以被前面 $1024$ 个 token 注意到。GPT-2 的上下文大小是 $1024$ 个 token。

**如果你不知道这些意味着什么，别担心，参考 [N008 - GPT Tokenizer](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb)。**

我们实际来画一下位置嵌入矩阵看看：

```python
plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray");
```

我们需要学习这些位置嵌入，因为**默认情况下，transformer 并不按顺序处理数据。**<br>

> 事实上，如果我们不提供位置信息，transformer 对序列这个概念会**毫无察觉**（这跟 RNN/LSTM 不同）。

为解决这个问题，我们不仅编码 token 本身的含义，还编码它的位置，以便 transformer 能更好地把语义纳入考量。<br>
我们上面画出了告知每个 token 位置的位置嵌入矩阵。<br>
位置嵌入是针对 $1024$ 的上下文大小、并在完整的 $768$ 隐藏大小上学习出来的。<br>
位置嵌入直接加到 token 嵌入值上，构成 transformer 的输入。

> 有点反直觉的是，位置嵌入并不是固定的，而是像 token 嵌入本身一样，从一个随机初始化的起点学习出来的。<br>

在训练过程中，对每个隐藏位置、跨所有上下文条目，会浮现出一个正弦函数的编码。<br>
每个编码所表示的函数略有不同，因为每个位置都携带了关于位置的特征信息，transformer 对其进行加工，以恢复出那些被所遇模式认为重要的相对细微差别。<br>

可以说，编码出的正弦函数通过上下文大小那么宽的嵌入里的参数来表达相对重要性。<br>
此外，[GPT-2 的位置嵌入矩阵编码出了一条螺旋线](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix)。

---

**Q:** **为什么位置嵌入不一开始就硬编码？学习 token 的顺序难道不是我们自己能直接表达的事吗，为它做优化岂不是浪费时间？**

**A:** 把按每个隐藏状态的信息硬编码进模型是可能的。<br>
例如原始的 [Transformer 模型 \[Vaswani et al. 2017\]](https://arxiv.org/abs/1706.03762) 就通过硬编码正弦函数做到了这一点。<br>

然而，一个学习过程能直接从 token 分布中发现有用的性质（比如那些固定正弦编码所具有的）。<br>
随着训练集规模和质量的提升，如今这一技术能跨位置嵌入提供超出那种"刚性"的位置关系嵌入本身的额外细微信息。

---

**Q:** **为什么是正弦函数？为什么用这些结构来编码位置，而且它们的估计又是从学习里浮现出来的？**

**A:** 先把一件事说清楚：每个位置嵌入是 $768$ 维的。上下文大小是 $1024$，意味着我们有 $1024$ 个这样的 $768$ 维嵌入。<br>
正弦特性并不是在每个位置嵌入上浮现的，而是按每个隐藏维度、跨所有位置嵌入来看时浮现的。我下面画了图来展示这一点。
因此，这些正弦函数必然表示的是某种类似于"位置间关系"的东西，而它是由该隐藏维度上"位置内"的值浮现出来的。

把这点说清楚后，正弦函数由正弦波和余弦波的组合构成。<br>
可能组合的空间是无限的，这意味着每个嵌入间的关系都能被唯一地编码。<br>
这很好，这正是我们这里需要的。<br>
我们也许会争辩说用随机初始化的嵌入也能行。但这种说法很快就不成立了。<br>
你看，当我们优化嵌入时，所有嵌入其实都是从随机初始化开始的，但我们所做的优化仍然让正弦函数跨嵌入浮现出来。<br>
因此，这些函数必然在某种程度上比随机嵌入更好地表示了关系。但具体怎么做到的？

首先，每个正弦函数都表达在完整上下文大小上。每个隐藏状态一个正弦函数，意味着我们有 $768$ 个这样的函数。<br>
正弦函数在被学习出来时，是梯度下降优化的结果，而梯度下降本身要求函数可微。<br><br>
这一过程产生的优化必然是连续的，具体地使（连续且可微的）正弦函数估计能在嵌入空间里浮现。<br>
由于正弦函数是连续的，上下文位置之间不同的重要性也以连续的方式被表达。<br>
这意味着正弦函数让我们能轻松计算上下文大小内任意两个位置之间的相对重要性，无论局部还是长距离。<br>

> 正弦函数是对语言连续本质的一种模拟，并且由于它是我们为获得它们所做优化的结果，因而是可微的，顺带让模型能非常顺畅地表达任意两个位置之间的相对重要性。

---

**Q:** **我对 GPT-2 的那个螺旋线怎么也想不通。到底是怎么回事？**

**A:** 我们说过，在位置嵌入里跨 $768$ 个隐藏状态的每一个，都浮现出一个正弦函数。<br>
现在，如果我们把目光从"跨位置嵌入地看"切回到"按每个嵌入地看"，并把每个 $768$ 维嵌入压缩到只有 $3$ 维，我们就能画出这些嵌入，它们形成一条连续曲线，一边绕某条轴旋转、一边沿轴推进。<br><br>
每个位置嵌入有 $768$ 个值，每个值各自贡献于一个正弦函数。于是位置嵌入捕捉的是跨多个函数（每个隐藏状态一个）的一个时间步。该嵌入因此既捕捉了序列内位置之间、也捕捉了跨位置的连续本质，而所有嵌入合在一起形成了我们在位置编码中所期望的周期性（跨序列、即嵌入之间的周期性）。

---

现在我们来画出三个单独的嵌入维度（列）：

```python
plt.figure(figsize=(10, 6))
plt.plot(sd_hf["transformer.wpe.weight"][0, :], label="Embedding Pos. 0")
plt.plot(sd_hf["transformer.wpe.weight"][50, :], label="Embedding Pos. 50")
plt.plot(sd_hf["transformer.wpe.weight"][100, :], label="Embedding Pos. 100")
plt.legend()
plt.title("Positional Embeddings")
plt.xlabel("Hidden Dimension")
plt.ylabel("Weight Value")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sd_hf["transformer.wpe.weight"][:, 150], label="Weight Index 150")
plt.plot(sd_hf["transformer.wpe.weight"][:, 200], label="Weight Index 200")
plt.plot(sd_hf["transformer.wpe.weight"][:, 250], color="green", label="Weight Index 250")
plt.legend()
plt.title("Positional Hidden Dim Values Across Context")
plt.xlabel("Context Position Index")
plt.ylabel("Weight Value")
plt.show()
```

虽然第二幅图的绿色曲线在第 $250$ 列上可能显示出某种跨位置的规律，但这并不意味着第 $250$ 列就编码了完整的位置关系。<br>
它只是那个表达跨全部 $1024$ 个嵌入位置关系的 $768$ 维函数的一个部分编码。

位置编码矩阵里单独一列的确切贡献，最多也只是含糊地被理解。<br>
它是 $768$ 件乐器组成的管弦乐队里的一件，共同谱写出位置关系。

*然而*有趣的是，我们能从上面第二幅图看出这个模型有点欠训练。<br>
我们之所以能这么说，是因为图本身里有噪声。我们还没有完全收敛到正弦函数近似。<br>

**注意力机制里的参数长这样：**

```python
plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300, :300], cmap="gray");
```

**我们来玩一下从 HuggingFace 的 GPT-2 采样，看看事情*本该*怎么运作：**

```python
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
```
```
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
```
```
[{'generated_text': "Hello, I'm a language model, but what I'm really doing is making a human-readable document. There are other languages, but those are"},
 {'generated_text': "Hello, I'm a language model, not a syntax model. That's why I like it. I've done a lot of programming projects.\n"},
 {'generated_text': "Hello, I'm a language model, and I'll do it in no time!\n\nOne of the things we learned from talking to my friend"},
 {'generated_text': "Hello, I'm a language model, not a command line tool.\n\nIf my code is simple enough:\n\nif (use (string"},
 {'generated_text': "Hello, I'm a language model, I've been using Language in all my work. Just a small example, let's see a simplified example."}]
```

### 注意力就是全部所需吗？

目前而言，是的，差不多是。<br><br>
我们不会去深入讲那篇开山之作：["Attention Is All You Need" \[Vaswani et al. 2017\]](https://arxiv.org/abs/1706.03762)。<br><br>
![](./img/transformer_gpt.png)

**对于 GPT-2，我们可以把红框以外的部分全部丢弃**，也就是编码器以及它与解码器的连接（通过所谓的交叉注意力）。

> **GPT-2 是一个仅解码器（decoder-only）模型。**

我们会对默认解码器做一些小改动。<br>
首先，层归一化被移到了带掩码的多头注意力之前。<br>
除了在前馈网络之前也放一个层归一化外，在最后一个 Transformer 块之后还放了另一个层归一化。

总之，我们的目标是这种结构，完整的 GPT-2 结构：<br>
![](./img/gpt2_layout_shuffle_shapes.png)

**好，我们来造这个。**<br>
我会在这里逐个类地讲解实现，然后把所有东西拼到一起，形成一个能跑的实现。

### 配置

首先，我们创建一个数据类 `GPTConfig`，用来存放我们配置 GPT 模型的各项取值。<br>
这个数据类长这样：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

各参数含义如下：
- `block_size`：我们一次要处理的序列长度（上下文长度）
- `vocab_size`：可能的输入 token 数
- `n_layer`：GPT 内部堆叠的 transformer 块数量
- `n_head`：每个多头注意力机制的头数（每个有 `n_layer` 层）
- `n_embd`：嵌入的维度，即隐藏大小

### GPT 类

我们对 GPT-2 复现的结构，在 `GPT` 类里实现：

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding Layer
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding Layer
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # The Actual (12) Transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer Norm before the output
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # The Prediction Head, Linear Layer but without Bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

我们后面希望能把 HuggingFace 版 GPT-2 的权重加载进我们的模型。<br>
因此，我们要把这个类在风格上做得尽可能贴近 HuggingFace 的参数布局。<br><br>
为了让实现尽可能像 HuggingFace 的，我们建了一个字典结构 `transformer` 作为主容器，装下除预测头 `lm_head` 以外的所有东西。<br>
`transformer` 这样组织，就能像 HuggingFace 的 `transformer.[wpe].weight` 那样用 `key:value` 索引。

**具体来说，我们用了：**
- Token 嵌入层 `wte`，把每个 token 映射为一个大小为 `n_embd` 的嵌入向量，
- 位置嵌入层 `wpe`，把上下文大小里的每个位置映射为一个大小为 `n_embd` 的嵌入向量，
- 由 `n_layer` 个 Transformer 块组成的模块列表 `h`，我们现在对它做数值索引，如同 HuggingFace 的 `transformer.h.[0].ln_1.weight`
- 最后一个 Transformer 块之后、输出之前的层归一化 `ln_f`，如同 HuggingFace 的 `transformer.ln_f.weight`
- 作为预测头的线性层 `lm_head`，把归一化后的 Transformer 输出映射回词表大小，如同 HuggingFace 的 `lm_head.weight`

### Block 类

我们看到 `GPT` 类里 `h` 中的 Transformer 块由 `Block` 对象组成。<br>
`Block` 类概括了解码器 Transformer 块：

```python
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

我们做层归一化，把它喂给自注意力机制，<br>
再喂给另一个层归一化，最终过一个 `MLP`，也就是前馈网络。

注意 `forward()` 里残差连接在这里并没有被归一化。<br>
这样更好，残差越不被改动，流过它们的梯度就越不受阻碍。<br>
但要注意，我们这样也就偏离了[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)。

### MLP 类

`MLP` 在 Transformer 块里用来构建其前馈网络部分。

```python
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```

我们把两个线性层连起来，它们共同形成一个大小为 `n_embd * 4` 的隐藏层，在它们之间施加<br>
`GELU` 激活函数，然后把输出映射回原始嵌入大小 `n_embd`。

和原始 GPT-2 实现一样，我们使用 [GELU 激活 \[Hendrycks et al. 2016\]](https://arxiv.org/abs/1606.08415)：<br>
![](https://pytorch.org/docs/stable/_images/GELU.png)

与此相关，我们在 [N004 - Makemore 3](../N004%20-%20Makemore%203%20-%20Activations,%20BatchNorm/N004%20-%20Makemore_3.ipynb) 里深入讨论过死神经元问题。<br>
如果 ReLU 激活的神经元持有负值，这些值会被 ReLU 函数干脆地映射成零。<br>
该神经元随之会收到零梯度，得不到更新。

作为对策，**GeLU 始终至少提供一点小梯度，从而不会突然阻碍学习。**

### CausalSelfAttention 类

现在还缺的结构部分是每个 Transformer 块里的注意力机制。<br>
我们通过 `CausalSelfAttention` 类来实现它：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```


虽然可能不那么一目了然，但这个实现和我们在 [N007 - GPT](../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb) 里搭建的实现是对应的。<br>
那时我们用了两个类：`Head` 和 `MultiHeadAttention`。<br>
现在被重构成了一个类，但工作方式基本一样。

**那它具体怎么工作？**

假设我们有 `context_size` 个嵌入向量。每个向量被送过线性层 `c_attn`，该层对每个输入向量输出三个大小与输入向量相同的向量：
- `q` $\rightarrow$ **Query（查询）**，我们想为之计算注意力的词/token 的表示，用一个独特的权重矩阵 $W_q$ 作用于输入嵌入学习得到，
- `k` $\rightarrow$ **Key（键）**，上下文中每个词/token 的表示，我们由其计算查询的注意力，用一个独特的权重矩阵 $W_k$ 作用于输入嵌入学习得到，
- `v` $\rightarrow$ **Value（值）**，上下文中每个词/token 内容的表示，我们想把它纳入注意力计算，同样用一个独特的权重矩阵 $W_v$ 作用于输入嵌入学习得到。

**Query** 和 **Key** 接着通过点积相乘。这给了我们上下文中每个词/token 的注意力分数。<br><br>
然后我们对注意力矩阵对角线以下的部分做掩码，使得每个词/token 只能注意到上下文中之前的词/token，<br>
而绝不会从（假设性的）未来的词那里获取信息。<br>
结果做 softmax 得到一个类概率分布的输出，再与 **Value** 向量 `v` 相乘。<br>
加过权的值向量随后被求和。换句话说，我们实际上形成了一个由注意力加权的值的和。<br>
这个和就是**注意力输出**。

长话短说，上面这段代码等价于有一个显式的 `Head` 类、并把每个 `Head` 的输出拼接起来构成 `MultiHead` 输入送入它最后那个做重塑的线性层。

现在复现 GPT-2 只差 `GPT` 类里的 `forward` 函数了：

```python
    def forward(self, idx):
        # idx has shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape is (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # additively combine
        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer-norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
```

输入是 token 索引。我们创建位置索引、位置嵌入和 token 嵌入。<br>
把它们相加（这会在批量维度上引发广播），然后送过 Transformer 块。<br>
最后，我们对输出做归一化并送过预测头。<br><br>
返回的 logits 现在离形成下一个 token 的概率只差一个 softmax 激活了。

```python
from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024  # Length of input sequence (context length)
    vocab_size: int = 50257 # Number of possible input tokens (50,000 BPE tokens + 256 bytes tokens + 1 <|endoftext|>)
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of heads for multi-head attention
    n_embd: int = 768       # Dimension of embeddings

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # (batch size, sequence length, embedding dimensionality) (latter aka n_embd)
        # calculate query, key, values for all head in batch and move head forward to be the batch dim
        # nh is "number of heads",
        # hs is "head size",
        # C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        # GPT-specific Layer-Norm placement before attention
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Self-Attention Block -> Performing iterative map-reducing
        # [!] Notice how residual connections aren't normalized here
        # This is better, also deviating from original Transformer paper
        # The more untouched residuals, the better the gradient flow
        x = x + self.attn(self.ln_1(x)) # Puts x in context to sequence
        # Feed-Forward Block
        x = x + self.mlp(self.ln_2(x)) # Views normalized score individually
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config
        
        # Main container for all GPT building blocks is called 'transformer'
        # ModuleDict to reflect key:value indexing as in 'transformer.[wpe].weight'
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding Layer
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding Layer
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # The Actual (12) Transformer blocks
            # ModuleList to reflect numeric indexing as in 'transformer.h.[0].ln_1.weight'
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer Norm before the output
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # The Prediction Head, Linear Layer but without Bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

**到这一步，我开始把代码抄到 [`train_gpt2_1.py`](./train_gpt2_1.py) 里。<br>那个文件是从这里开始我所实现内容的快照。**

注意该文件还多了一个构造函数 `from_pretrained`，用于把 HuggingFace 模型的权重加载进我们自己的模型架构。<br>不过我们不会在这里细讲这个导入器。<br><br>
文件末尾，下面这段"高度精密"的代码用来检查实例化模型时是否会出错：

```python
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")
```

### 分词

我们复现了 GPT-2 模型架构并把原始权重加载了进去。但现在，我们怎么提示它生成文本呢？
我们可以用 `tiktoken` 库来替我们承担输入分词的大部分重活。`tiktoken` 当初也用于 GPT-2。
不过理论上，我们可以参考 [N008 - Tokenization](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) 自己造一个分词器。

首先，在 [`train_gpt2_1.py`](./train_gpt2_1.py) 里，我们把

```python
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")
```

替换成把模型推入 CUDA 显存的代码（如果可用）：

```python
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

然后，我们加上这段代码：

```python
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)
```

如果这些看着不眼熟，别担心，参考 [N008 - Tokenization](../N008%20-%20GPT%20Tokenizer/N008%20-%20Tokenization.ipynb) 复习一下 LLM 中分词如何以及为何这样工作。

现在要让模型真正做推理，剩下的就是这段迭代地从输入逐步构建<br>
文本序列、并把每个新序列再次送回模型、直到达到 `max_length` 的逻辑。<br>

我们也把这段追加到 [`train_gpt2_1.py`](./train_gpt2_1.py) 末尾并打印结果。

它长这样：

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits only at the last position (works but wasteful)
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (8, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (8, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # (8, T+1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
```

有了这段代码，我们实际上就为我们的 GPT-2 克隆复现了 HuggingFace `pipeline` 的基本行为。

运行这段代码得到：

```cmd
> Hello, I'm a language model, not a program.

So this morning I started studying for the interview in the lab. This was not
> Hello, I'm a language model, and one of the main things that bothers me when they create languages is how easy it becomes to create something that
> Hello, I'm a language model, and I wrote it off on the grounds that a language model would make me more fluent. But I'm not
> Hello, I'm a language model, I really like languages. I like languages because like, they're good. And the way we talk about languages
> Hello, I'm a language model, a language model I'm using for data modelling. All I did was test the results and then I wrote some
```

---

<br>
<br>
我们现在已经把所有权重移植过来，在自己的架构里跑它们，还能从模型采样生成文本。<br>
但现在，如何能从零初始化这个结构呢？

到目前为止，我们走近原始 GPT-2 实现，看了它的架构。<br>
然后我们复制了它的模型结构，达到能直接把原始模型权重加载进我们的模型并跑推理的程度。

但这只能带我们走到这儿了。**充其量，我们也就是追平 GPT-2。**<br>
我们并没真正用我们的实现做更多事。而且，恕我直言，GPT-2 也算是古董了。

接下来，我们的目标是在 GPT-2 复现的基础上继续构建，增强它的能力。

**来从零训练我们自己的 GPT-2 克隆吧！但我们的目标是要做得更好、更快、更强。**
<br>
<br>

---

## 重置权重

**到这一步，我把 [`train_gpt2_1.py`](./train_gpt2_1.py) 的代码抄了过来，在 [`train_gpt2_2.py`](./train_gpt2_2.py) 里继续。**

如果我们想生成跟 GPT-2 一样好甚至更好的序列，首先得把前面建好的模型设成随机初始化的权重以供训练。<br>
**我们不想再依赖导入 GPT-2 的原始权重了。**

让随机权重初始化跑起来其实挺简单，多亏了 PyTorch。<br>
我们只需把 `model = GPT.from_pretrained('gpt2')` 换成 `model = GPT(GPTConfig())`。

随机初始化后，现在可能得到类似这样的结果：

```cmd
> Hello, I'm a language model,itimate synthes primUrbanHO natives clearer)(� unwittingaelwk Warn demolitionthebit Chevron Inspection puraledCold SSH
> Hello, I'm a language model, CAT Wra appropriated atroc remarkablestanighedThor released Lazarus Unt 354rots tattweekly thorough Image unsu Bald hijasksAMD
> Hello, I'm a language model, ACE Wra comprehens plannedbird restraintourseseph exploredwriteRomanDonaldTrump akaherty close streetcar inmatesQaedaEmergency stabbed reign offender
> Hello, I'm a language model,went leapt FACE ≤ posts568demeredith burn Virt COVERner negotiators budgetary warhyp Kyleemortbuilt2010763 fortun
> Hello, I'm a language model, venue vo"}," heresy retail rabbit goofy modulation reactor 82 questions KN Corinthians Socialist vig blade real Pit cube padsmorph renal
```

我们当然没提升性能，但为训练清好了舞台。

## 构造分词后的训练输入

训练需要**大量**数据。<br>
为了训练我们的 GPT-2 克隆，我们会用一个熟悉的数据集：[`Tiny-Shakespeare`](../tiny-shakespeare.txt)。

```python
with open('../tiny-shakespeare.txt', 'r') as f:
    text = f.read()
print(text[:100]) # print the first 100 characters of the dataset

print('\nWord Count:', len(text.split(' ')))
print('Character Count:', len(text))
print('Line Count:', text.count('\n'))
```
```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You

Word Count: 169893
Character Count: 1115394
Line Count: 40000
```

如[前面](#分词)所述，我们可以用 `tiktoken` 库来替我们做分词。<br>
我们只会使用并编码数据集的前 $1000$ 个字符。

$1000$ 个字符的数据集绝对是不够用的，但这是故意的。<br>
我们先用这个更易管理的数据集打好基础，之后再把结论应用到更大的 `Tiny Shakespeare` 数据集上。

```python
import tiktoken

data = text[:1000]

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
print('First 24 tokens:', tokens[:24])
```
```
First 24 tokens: [5962, 22307, 25, 198, 8421, 356, 5120, 597, 2252, 11, 3285, 502, 2740, 13, 198, 198, 3237, 25, 198, 5248, 461, 11, 2740, 13]
```

分词搞定（外包给 `tiktoken`，但能跑）之后，我们现在需要找到一种办法，把 token 集合转换成我们 Transformer 网络的输入 `idx`。<br>
输入 `idx` 的形状必须是 `(batch_size, context_size)`，我们之前把它记作 `(B, T)`。

现在我们有一个持有全部 token 的一维张量。我们需要把它整理成长度不超过 `context_size` 的若干批次。<br>
对于我们这个玩具数据集例子，我们设 `B = batch_size = 4`、`T = context_size = 6`，于是 `B * T = 24`。

```python
import torch
buf = torch.tensor(tokens[:24]) # (B * T,)
x = buf.view(4, 6)  # (B, T)
print(x)
```
```
tensor([[ 5962, 22307,    25,   198,  8421,   356],
        [ 5120,   597,  2252,    11,  3285,   502],
        [ 2740,    13,   198,   198,  3237,    25],
        [  198,  5248,   461,    11,  2740,    13]])
```

如果我们有一个 $24$ 个 token 的张量，我们可以把它的视图改成 `(4, 6)`（比 `.reshape` 更省内存）。<br>
这样我们就有了 $4$ 个批次，每批 $6$ 个 token。<br>
每一行对应一个序列、一个 token 批次，一个接一个，本质上是把原张量堆叠起来。

**然而，上面的做法仍有缺陷：**<br>
以第一批为例，对 token `[25]` 来说，我们会输入它前面的序列 `[5962, 22307, 25]`。<br>
基于此，我们想学习预测该批次里的下一个 token `[198]`。<br>
这看似简单，直到我们意识到在同一批次里，序列 `[198, 8421, 356]` 啥也预测不了，因为我们在这一批里没给它提供标签。

我们可以很利落地解决：

```python
buf = torch.tensor(tokens[:24 + 1]) # (B * T) + 1 to provide very last token with label
x = buf[:-1].view(4, 6) # (B, T) - all but the last token (omitting the +1 here)
y = buf[1:].view(4, 6)  # (B, T) - all but the first token (right shift by 1 token)
print(x)
print(y)
```
```
tensor([[ 5962, 22307,    25,   198,  8421,   356],
        [ 5120,   597,  2252,    11,  3285,   502],
        [ 2740,    13,   198,   198,  3237,    25],
        [  198,  5248,   461,    11,  2740,    13]])
tensor([[22307,    25,   198,  8421,   356,  5120],
        [  597,  2252,    11,  3285,   502,  2740],
        [   13,   198,   198,  3237,    25,   198],
        [ 5248,   461,    11,  2740,    13,   198]])
```

我们对所有批次里的值做了一次右移。<br>
这样，当现在处理第一批里的序列 `[5962, 22307, 25]` 时，<br>
我们会在 `y` 中、在输入序列最后一个 token（`[25]`）的*同一索引*处找到对应的标签。

## 计算损失

我们把这个批处理做法用到 [`train_gpt2_2.py`](./train_gpt2_2.py) 里，构建一个数据加载器，为我们提供 `idx` 格式的输入和标签。<br>
一旦这部分跑通，我们就能继续计算损失并据此训练模型。

我们会更新代码文件里的分词部分，让它同时组装出 `x` 和 `y` 两个批次张量。<br>
完整实现见 [`train_gpt2_2.py`](./train_gpt2_2.py)，相关部分的片段也列在这里。

```python
# Tokenizer as used for GPT-2 originally
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]  # Limit to 1000 for Debugging purposes
tokens = enc.encode(text)
B, T = 4, 32 # Set like this for Debugging purposes
buf = torch.tensor(tokens[:B*T + 1]) # (B * T) + 1 to provide very last token with label
x = buf[:-1].view(B, T) # (4, 32)
y = buf[1:].view(B, T)  # (4, 32), right-shifted by 1

model = GPT(GPTConfig()) # random weight initialization
model.to(device)
logits = model(x)

print(logits.shape) # (4, 32, 50257) -> (B, T, vocab_size)
```

如果我们现在想基于 `x` 和 `y` 之间的差异来训练，我们就还得改造 `GPT` 里的 `forward` 函数，让它累加并返回损失。<br>
换句话说，我们现在想要的是 `logits, loss = model(x, y)`，而不是 `logits = model(x)`。

新的前向传播现在长这样：

```python
def forward(self, idx, targets=None):
        # idx, targets both of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

真正重要的只有 `loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))` 这一句。<br>
这里施加的是[交叉熵损失](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)。把我们的下一个词预测想成一个分类问题，我们要在 `vocab_size` 个类别里预测出正确的类别，也就是下一个 token。<br>
交叉熵损失是多类分类问题的常见选择。<br>
它在这里看着有点吓人，是因为 `F.cross_entropy` 不接受 `(B, T, vocab_size)` 形状。<br>
我们转而把 logits 和标签展平成 `(B * T, vocab_size)` 来让它跑通（标签 `y` 就变成了 `(B*T,)`）。

运行改过的脚本会得到类似这样的结果：`tensor(10.9276, grad_fn=<NllLossBackward0>)`。

事实上，我们可以对这个值做个合理性检查。理想情况下，在初始化时，权重应设置成使词表里每个 token 作为下一个 token 被采样的机会均等。这意味着每个 token 的概率应为 $\frac{1}{50257}$。<br>
交叉熵损失本质上计算的是正确 token 的负对数似然。如果是理想的均匀分布，负对数似然为 $-\log(\frac{1}{50257}) = \log(50257) \approx 10.824$。我们非常接近这个值，是个好兆头。可以开始优化了。

## 优化

### 合理性检查

我们现在有了输入、标签和损失计算。<br>
可以像这样来优化模型：

[`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 是我们要用的优化器，用来累加梯度并更新模型权重。<br>
你可以把 `AdamW` 想成 RMSprop 和带动量的随机梯度下降的结合体，也就是一个更高级的 SGD 版本。<br>
`AdamW` 是 `Adam` 的一个版本，对权重衰减有更好的实现。多数情况下你直接用 `AdamW` 替代 `Adam` 即可。

总之，出于这个例子的目的，我们就直接用它，把它当个黑箱看待。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Optimization loop
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"epoch {i}, loss: {loss.item()}")
```

**恭喜！你刚刚在单个批次 `x` 上把模型严重过拟合了！**<br>
不管怎样，我们能看到训练从 `10.927579879760742` 进展到 `0.0028732302598655224`，<br>
说明模型在学习，甚至完全记住了。

我们可以接着创建从数据集加载新批次、并在其上训练模型的逻辑。

### 数据加载

要解决只在单个批次上训练的问题，我们需要按批次从数据集加载数据。<br>
下面这段代码就是干这个的：

```python
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B # batch size
        self.T = T # context size

        # at init load tokens from disk and store them in memory
        with open('../tiny-shakespeare.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text) # encode full text into tokens
        self.tokens = torch.tensor(tokens) # wrap with tensor
        self.tok_count = len(self.tokens)
        # Just some stats for us nerds
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # token-level start position for each next batch
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # grab a chunk of tokens of size B * T + 1 (we explained this before)
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # input tensor of size (B * T)
        y = buf[1:].view(B, T)  # target tensor of size (B * T), right-shifted 1 position
        # advance position in data tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) >= len(self.tokens):
            # reset position to beginning of data
            self.current_position = 0
        return x, y
```

快速批处理的秘诀是以 `B * T` 个 token 为步长在数据集上移动。<br><br>
这么做时，我们每步实际要加载 `B * T + 1` 个 token，因为我们要给批次最后一个 token 提供标签。<br>
然后我们用前 `B * T` 个 token 作输入，把右移一位的那 `B * T` 个 token 作标签。<br>
为让这个函数真正在数据集上移动，我们用一个 `current_position`，从它开始组装批次。<br>
做完之后，我们把它加 `B * T` 个 token，或者让它回到数据集开头。

我们可以在训练脚本里像这样初始化 `DataLoaderLite`：`train_loader = DataLoaderLite(B=4, T=32)`。<br>
更新后的训练循环长这样：

```python
for i in range(50):
    # This is new
    x, y = train_loader.next_batch()
    # This is necessary now
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"epoch {i}, loss: {loss.item()}")
```

有了这套设置，我们的损失现在下降得慢得多、也"晃"得多，因为我们现在跨多个批次训练了。<br>
起始损失是：`10.927579879760742`<br>
$50$ 个批次后的损失是：`6.575399398803711`

**不错！** 但还有一些可以改进的地方。

## 修 Bug

### Token 嵌入层和线性输出层竟然是同一个？！

在[前面的小节](#gpt-2-架构概览)里，我们看了 GPT-2 架构。<br>
我们看到模型有一个 token 嵌入层和一个用于预测下一个 token 的线性输出层。<br><br>
但这俩层之间有件可能不那么一目了然的事。

```python
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M parameters, for "gpt2-xl" the gold model
sd_hf = model_hf.state_dict() # Dictionary of the model's weight/bias tensors

# print layer names and shapes
for identifier, tensor in sd_hf.items():
    print(identifier, tensor.shape)
```
```
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])
transformer.h.0.ln_1.weight torch.Size([768])
transformer.h.0.ln_1.bias torch.Size([768])
transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.0.attn.c_attn.bias torch.Size([2304])
transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias torch.Size([768])
transformer.h.0.ln_2.weight torch.Size([768])
transformer.h.0.ln_2.bias torch.Size([768])
transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.0.mlp.c_fc.bias torch.Size([3072])
transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.0.mlp.c_proj.bias torch.Size([768])
transformer.h.1.ln_1.weight torch.Size([768])
transformer.h.1.ln_1.bias torch.Size([768])
transformer.h.1.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.1.attn.c_attn.bias torch.Size([2304])
transformer.h.1.attn.c_proj.weight torch.Size([768, 768])
transformer.h.1.attn.c_proj.bias torch.Size([768])
transformer.h.1.ln_2.weight torch.Size([768])
transformer.h.1.ln_2.bias torch.Size([768])
transformer.h.1.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.1.mlp.c_fc.bias torch.Size([3072])
transformer.h.1.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.1.mlp.c_proj.bias torch.Size([768])
transformer.h.2.ln_1.weight torch.Size([768])
transformer.h.2.ln_1.bias torch.Size([768])
transformer.h.2.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.2.attn.c_attn.bias torch.Size([2304])
transformer.h.2.attn.c_proj.weight torch.Size([768, 768])
transformer.h.2.attn.c_proj.bias torch.Size([768])
transformer.h.2.ln_2.weight torch.Size([768])
transformer.h.2.ln_2.bias torch.Size([768])
transformer.h.2.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.2.mlp.c_fc.bias torch.Size([3072])
transformer.h.2.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.2.mlp.c_proj.bias torch.Size([768])
transformer.h.3.ln_1.weight torch.Size([768])
transformer.h.3.ln_1.bias torch.Size([768])
transformer.h.3.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.3.attn.c_attn.bias torch.Size([2304])
transformer.h.3.attn.c_proj.weight torch.Size([768, 768])
transformer.h.3.attn.c_proj.bias torch.Size([768])
transformer.h.3.ln_2.weight torch.Size([768])
transformer.h.3.ln_2.bias torch.Size([768])
transformer.h.3.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.3.mlp.c_fc.bias torch.Size([3072])
transformer.h.3.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.3.mlp.c_proj.bias torch.Size([768])
transformer.h.4.ln_1.weight torch.Size([768])
transformer.h.4.ln_1.bias torch.Size([768])
transformer.h.4.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.4.attn.c_attn.bias torch.Size([2304])
transformer.h.4.attn.c_proj.weight torch.Size([768, 768])
transformer.h.4.attn.c_proj.bias torch.Size([768])
transformer.h.4.ln_2.weight torch.Size([768])
transformer.h.4.ln_2.bias torch.Size([768])
transformer.h.4.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.4.mlp.c_fc.bias torch.Size([3072])
transformer.h.4.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.4.mlp.c_proj.bias torch.Size([768])
transformer.h.5.ln_1.weight torch.Size([768])
transformer.h.5.ln_1.bias torch.Size([768])
transformer.h.5.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.5.attn.c_attn.bias torch.Size([2304])
transformer.h.5.attn.c_proj.weight torch.Size([768, 768])
transformer.h.5.attn.c_proj.bias torch.Size([768])
transformer.h.5.ln_2.weight torch.Size([768])
transformer.h.5.ln_2.bias torch.Size([768])
transformer.h.5.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.5.mlp.c_fc.bias torch.Size([3072])
transformer.h.5.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.5.mlp.c_proj.bias torch.Size([768])
transformer.h.6.ln_1.weight torch.Size([768])
transformer.h.6.ln_1.bias torch.Size([768])
transformer.h.6.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.6.attn.c_attn.bias torch.Size([2304])
transformer.h.6.attn.c_proj.weight torch.Size([768, 768])
transformer.h.6.attn.c_proj.bias torch.Size([768])
transformer.h.6.ln_2.weight torch.Size([768])
transformer.h.6.ln_2.bias torch.Size([768])
transformer.h.6.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.6.mlp.c_fc.bias torch.Size([3072])
transformer.h.6.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.6.mlp.c_proj.bias torch.Size([768])
transformer.h.7.ln_1.weight torch.Size([768])
transformer.h.7.ln_1.bias torch.Size([768])
transformer.h.7.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.7.attn.c_attn.bias torch.Size([2304])
transformer.h.7.attn.c_proj.weight torch.Size([768, 768])
transformer.h.7.attn.c_proj.bias torch.Size([768])
transformer.h.7.ln_2.weight torch.Size([768])
transformer.h.7.ln_2.bias torch.Size([768])
transformer.h.7.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.7.mlp.c_fc.bias torch.Size([3072])
transformer.h.7.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.7.mlp.c_proj.bias torch.Size([768])
transformer.h.8.ln_1.weight torch.Size([768])
transformer.h.8.ln_1.bias torch.Size([768])
transformer.h.8.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.8.attn.c_attn.bias torch.Size([2304])
transformer.h.8.attn.c_proj.weight torch.Size([768, 768])
transformer.h.8.attn.c_proj.bias torch.Size([768])
transformer.h.8.ln_2.weight torch.Size([768])
transformer.h.8.ln_2.bias torch.Size([768])
transformer.h.8.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.8.mlp.c_fc.bias torch.Size([3072])
transformer.h.8.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.8.mlp.c_proj.bias torch.Size([768])
transformer.h.9.ln_1.weight torch.Size([768])
transformer.h.9.ln_1.bias torch.Size([768])
transformer.h.9.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.9.attn.c_attn.bias torch.Size([2304])
transformer.h.9.attn.c_proj.weight torch.Size([768, 768])
transformer.h.9.attn.c_proj.bias torch.Size([768])
transformer.h.9.ln_2.weight torch.Size([768])
transformer.h.9.ln_2.bias torch.Size([768])
transformer.h.9.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.9.mlp.c_fc.bias torch.Size([3072])
transformer.h.9.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.9.mlp.c_proj.bias torch.Size([768])
transformer.h.10.ln_1.weight torch.Size([768])
transformer.h.10.ln_1.bias torch.Size([768])
transformer.h.10.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.10.attn.c_attn.bias torch.Size([2304])
transformer.h.10.attn.c_proj.weight torch.Size([768, 768])
transformer.h.10.attn.c_proj.bias torch.Size([768])
transformer.h.10.ln_2.weight torch.Size([768])
transformer.h.10.ln_2.bias torch.Size([768])
transformer.h.10.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.10.mlp.c_fc.bias torch.Size([3072])
transformer.h.10.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.10.mlp.c_proj.bias torch.Size([768])
transformer.h.11.ln_1.weight torch.Size([768])
transformer.h.11.ln_1.bias torch.Size([768])
transformer.h.11.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.11.attn.c_attn.bias torch.Size([2304])
transformer.h.11.attn.c_proj.weight torch.Size([768, 768])
transformer.h.11.attn.c_proj.bias torch.Size([768])
transformer.h.11.ln_2.weight torch.Size([768])
transformer.h.11.ln_2.bias torch.Size([768])
transformer.h.11.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.11.mlp.c_fc.bias torch.Size([3072])
transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.11.mlp.c_proj.bias torch.Size([768])
transformer.ln_f.weight torch.Size([768])
transformer.ln_f.bias torch.Size([768])
lm_head.weight torch.Size([50257, 768])

Final Linear Output Layer Shape: torch.Size([50257, 768])
Word Token Embedding Layer Shape: torch.Size([50257, 768])
```

```python
print(f"Final Linear Output Layer Shape: {sd_hf['lm_head.weight'].shape}") # (50257, 768) - 50257 tokens, 768 features
print(f"Word Token Embedding Layer Shape: {sd_hf['transformer.wte.weight'].shape}\n") # (50257, 768) - 50257 tokens, 768 features

print(f"All tensor values are the same:\t\t\t{(sd_hf['lm_head.weight'] == sd_hf['transformer.wte.weight']).all()}")
print(f"The memory address of the layers is the same:\t{sd_hf['lm_head.weight'].data_ptr() == sd_hf['transformer.wte.weight'].data_ptr()}")
```
```
Final Linear Output Layer Shape: torch.Size([50257, 768])
Word Token Embedding Layer Shape: torch.Size([50257, 768])

All tensor values are the same:			True
The memory address of the layers is the same:	True
```

我们看到，token 嵌入层和线性输出层事实上就是同一层！<br>
其实，["Attention is All You Need" \[Vaswani et al. 2017\]](https://arxiv.org/abs/1706.03762) 的第 3.4 节就指出，token 嵌入层和线性输出层就是像这样绑在一起的。

**但为什么？**

对嵌入层和输出层用同一套权重，是个**省内存和算力**的好技巧。<br>
除此之外，**我们其实也希望这两层以相似方式工作，只是方向不同**（进入或离开嵌入空间）。<br>
从应用角度看，这意味着在嵌入空间里相似的两个 token，在输出空间里也应当相似，反之亦然。

更详细讨论这一点的论文是 [\[Press \& Wolf 2016\]](https://arxiv.org/abs/1608.05859)。尤其见引言第二段。

在我们的模型里修这个非常直接。<br>
在 `GPT` 类默认构造函数里，我们只需追加这句：

```python
self.transformer.wte.weight = self.lm_head.weight
```

光这一句就足以让两层（在内存里）完全相同。<br>
注意，我们实际上刚把模型参数量减少了约 $30\%$，而性能没有明显损失。<br><br>
（我现在得到的损失是 `6.793056964874268`）

### 权重初始化

遗憾的是，[GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)和 [GPT-3 论文](https://arxiv.org/abs/2005.14165)都没对权重*如何*初始化给出更深入的信息。<br>
幸运的是，我们可以看 GPT-2 的[实现](https://github.com/openai/gpt-2/blob/master/src/model.py)来了解是怎么做的。

**具体来说，我们观察到：**
- 嵌入层的权重用正态分布、标准差 $0.02$ 初始化，
- 偏置初始化为 $0$，
- token 嵌入也用正态分布、标准差 $0.02$ 初始化，
- 位置嵌入用标准差 $0.01$ 初始化

要在复现里照搬这一点，我们可以把这段代码集成进模型：

```python
def _init_weights(self, module):
    # Linear weights distributed with mean 0 and std 0.02
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            # Biases explicitly initialized to 0
            torch.nn.init.zeros_(module.bias)
    # Embedding weights distributed with mean 0 and std 0.02
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

我们模型里唯一其它带权重的就是 `LayerNorm` 层。<br>
幸运的是，PyTorch 默认就已经按 GPT-2 想要的方式初始化它们了，也就是缩放为 $1$、偏移为 $0$。<br>
（我现在得到的损失是 `6.847793102264404`。）

**注意，这种硬编码的权重初始化未必对所有模型都是最佳选择。**<br>
总体而言，它已不再被视为 SOTA，因为你希望权重的初始化能照顾到整体模型规模和架构。<br>
现在这大多用 [Xavier 初始化 \[Glorot \& Bengio 2010\]](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 或 [He 初始化 \[He et al. 2015\]](https://arxiv.org/abs/1502.01852) 来做，视上下文和激活函数而定。

**那么，权重初始化这就完事了？**<br>
*还不算。*

GPT-2 的 transformer 架构采用了绕过权重层的跳跃连接。<br>
绕过这些"实际学习"的层所带来的影响，会随网络深度而增大。<br>
为抵消这一点，[GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)建议把残差连接的权重按 $\frac{1}{\sqrt{N}}$ 缩放，其中 $N$ 是残差层的数量。<br>

---

**Q:** **这是什么意思？**

**A:** 实际上，按残差连接所绕过的总层数的比例，去初始化被残差连接绕过的那些层的权重，能让训练时权重的方差更小，使学习层和残差层能更均衡地施加影响。

**好长一句话。但说真的，这么想：** 这里提出的初始权重缩放是必要的，因为当我们把残差输出与学习层输出融合（通过相加，见 `x = x + self.attn(self.ln_1(x))`）时，我们必然要考虑新 `x` 中值的方差变大这件事。<br>
如果我们之后一遍遍施加残差连接，我们在模型里传递的值的方差可能变得非常高。而高方差意味着内部传递值之间分布上的表达力变低。这会让学习更难。

反过来，如果我们按残差层数的比例去缩放被残差连接"绕过"的那些层的权重，我们就能在相当程度上抵消这种熵的膨胀，因为我们缩小了流过可学习层的值的范围，从而把方差控制住。

> 按 $\frac{1}{\sqrt{N}}$ 缩放让我们能控制流过模型的值的方差增长，从而帮助我们把模型保持得更稳定、更可学习。

---

**Q:** **这似乎反直觉。我们为什么要把学习层的权重缩小？难道不该缩小跳跃连接的贡献吗？**

**A:** 我们用跳跃连接的原因是，它们是一条穿过模型的、未被改动的路径，利于梯度流动。缩放它们会干扰这条信息流。<br>
通过缩放学习层的输出，我们确保它们引入的改动不会"淹没"来自跳跃连接的未触碰信号。<br>
所以，真正会随着模型变深（即残差层越多）通过对残差值的更新而把我们的权重乃至梯度晃得更厉害的，其实是学习层。

---

```python
# Toy Example: Effects of scaled vs unscaled residuals on std dev
p = torch.zeros(768)
q = p.clone()

n = 100
for i in range(n):
    update = torch.randn(768)
    p += update
    q += n ** -0.5 * update

print(f'Unscaled Residual Std Dev: {p.std()}')
print(f'Scaled Residual Std Dev: {q.std()} (the lower the better/more expressive)')
```
```
Unscaled Residual Std Dev: 9.67028522491455
Scaled Residual Std Dev: 0.9670284986495972 (the lower the better/more expressive)
```

**好，那我们在模型里怎么实现？**<br>
首先，回忆一下残差连接是怎样集成进 transformer 解码器的：

![](./img/transformer_gpt.png)

我们可以给 `CausalSelfAttention`（即多头注意力）和 `MLP`（即前馈层）的实例加一个标志属性 `NANOGPT_SCALE_INIT`，把它们标记为残差层。然后在权重初始化时，我们检查这个标志是否存在，并相应地缩放权重。

```python
def _init_weights(self, module):
        # Linear weights distributed with mean 0 and std 0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 1 / sqrt(N) scaling for the std dev
                # 2 * n_layer because we have attention *and* mlp per block (blocks counted by n_layer)
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # Biases explicitly initialized to 0
                torch.nn.init.zeros_(module.bias)
        # Embedding weights distributed with mean 0 and std 0.02
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

（我现在得到的损失是 `6.799217700958252`。）

好，现在我们搭好了架构、数据加载、损失计算和权重初始化。<br>
下一步是优化训练流水线，扩大批大小和序列长度，以学到更有价值的相互依赖关系。

> **我们到此会冻结 [`train_gpt2_2.py`](./train_gpt2_2.py) 的开发。这是我们的第二个里程碑完成。<br>训练时间优化的第一部分将在 [`train_gpt2_3.py`](./train_gpt2_3.py) 里实现。**

## 优化模型训练

一般而言，想高效地训练模型，你需要了解你训练所在设备的能力、以及你具体在训练什么，有时甚至要到比特层面。<br>

如果你在 [`train_gpt2_2.py`](./train_gpt2_2.py) 脚本的 `logits, loss = model(x, y)` 之后插入 `import code; code.interact(local=locals())`，然后在那里请求 `logits.dtype`，你会看到 logits 内部是 `float32` 类型。

如果我们愿意牺牲一点精度，可以把 logits 转成 `float16` 来省内存，并把计算时间缩短**很多**。<br>
例如，在 NVIDIA A100 GPU 上，从 `float32` 换成 `float16` 意味着约 16 倍的计算时间下降。<br><br>
有意思的是，[NVIDIA 的 A100 产品规格](https://www.nvidia.com/en-us/data-center/a100/#:~:text=624%20TOPS%20%7C%201248%20TOPS*)提到有一种中间数据格式，**Tensor Float 32 (TF32)**。<br>
对于 A100 GPU，NVIDIA 表示约 $\sim 8 \times$ 的计算时间下降。<br>
这看起来是个不错的精度折中，值得先研究一下。

不过我们再把视角拉远一点。

### NVIDIA Tensor Core 与 TensorFloat32

很多 NVIDIA GPU 集成了所谓的 Tensor Core。<br>Tensor Core 本质上提供了一个专门用于 $4 \times 4$ 矩阵乘法的芯片组：
![](https://leimao.github.io/images/blog/2023-05-18-NVIDIA-Tensor-Core-Programming/turing-tensor-core-math.png)<br>
来源：[https://leimao.github.io](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)

你会注意到上面这个操作很像我们从神经网络里熟知的仿射变换。**这并非巧合。**

如果我们想通过这样的 GPU 来跑某个模型，这些 Tensor Core 是通过所谓的 CUDA 来控制的。CUDA 是一个*非常*强大的并行计算平台。对我们这个抽象例子而言，CUDA 管理着我们的值通过这些 $4\times 4$ 矩阵乘法单元的处理。这可能意味着要把值切分开好让它塞进 tensor core，从而产生开销。尽管如此，CUDA 仍把那开销控制得很小，使支持 CUDA 的 GPU 成为 ML 训练的理想之选。

但这对于我们用 `TensorFloat32` 替代 `Float32` 的想法意味着什么呢？<br>
看下面这张图，我们就能追溯 `TensorFloat32` 究竟是如何带来这种算力提升的思路：

![](./img/a100_wp_tf32_dtype.png)<br>
来源：[NVIDIA - A100 GPU 白皮书（第 27 页）](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

和几乎所有计算系统一样，在这块 A100 GPU 里，浮点值通过指数和尾数来表示。<br>
浮点值的取值通过：$\text{Value} = \text{Mantissa} \times \text{Base}^\text{Exponent}$<br>
指数移动小数点，尾数持有"特征性"的数字。

> 尾数越小，我们在给定可表示数范围内能直接表达的位置就越少。

`TF32` 的思路是通过截掉 $13$ 位来降低尾数的分辨率。<br>
GPU 随后用这些 $19$ 位浮点值来计算。最后，结果再被补回到 $32$ 位。

> NVIDIA 的 `TF32` 数据类型是计算成本和精度损失之间一个可接受的折中。对我们此处的目的来说，这点小小的"精度损失"完全能接受。

我们可以在实现中测量改变内部表示数据类型所带来的影响。

我把这条以及接下来的优化应用在脚本的一个副本 [`train_gpt2_3.py`](./train_gpt2_3.py) 里。<br>
此外，我用这个脚本去折磨一张 3060，每条优化各跑 1 个 epoch。<br>
对于换成 `TF32` 这条，我得到 `dt: 147900.74ms, tok/sec: 110.78`，挺搞笑的。

> 训练时，把每批的条目数拉满以最高效地训练。<br>
> 给 GPU 的批次尽可能用 $2$ 的高倍数。

就在模型初始化（`model = GPT(GPTConfig())`）之前，我们写上 [`torch.set_float32_matmul_precision('high')`](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html) 来强制使用 `TF32` 而非 `Float32`（默认设为 `'highest'`）。

即便从一张 3060 上也能看到明显的性能提升，降到仍然离谱的值：`dt: 91319.48ms, tok/sec: 179.41`。<br>
这是因为主要瓶颈是显存内的搬运。它仍然在相当程度上拖住处理速度。不过 3060 还是跟得上，挺逗的。

### 减少显存搬运

我们看到，即便在视频里，承诺的约 8 倍处理速率提升最终也只兑现到约 3 倍。<br>
这是因为我们不停地通过显存搬运大量数据。看看能不能减少它。
我们现在会改用 `BFloat16` 作为内部处理数据类型。

我们能在 PyTorch 文档里找到[一段很好的说明](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast)，讲如何使用混合精度以及如何把 `BFloat16` 集成进我们的模型。<br>归根结底，它就等于在 [`train_gpt2_3.py`](./train_gpt2_3.py) 里加这样一处改动：

```python
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

注意，这额外的计算速度提升现在意味着张量值会发生变化。<br>
这是我在 3060 上得到的结果：`dt: 70393.32ms, tok/sec: 232.75`。<br>
仍然糟糕，但我们基本把净处理时间砍掉了一半。

### PyTorch 编译

在 Linux 系统上，PyTorch 提供了把模型[编译](https://pytorch.org/tutorials//intermediate/torch_compile_tutorial.html)成更高效表示的能力。<br>
它极为强大、对部署等场景很有用，我们可以用一行代码把它集成进 [`train_gpt2_3.py`](./train_gpt2_3.py)：

```python
model = torch.compile(model)
```

这会在执行前先花一次编译时间，然后显著加快执行本身。

---

**Q:** **但 Python 不是解释型语言吗？我们怎么能编译它？**

**A:** 没错，Python 是解释型的。但 PyTorch 是包裹在与 Python 接口的 C++ 代码之上的。这些 C++ 代码可以被编译成机器码。<br>
当我们调用 `torch.compile` 时，PyTorch 会先在内部对模型做一次"预览"，然后（在 C++ 意义上）把它编译成机器码。

我们通过一个例子来看。在 `MLP` 类（我们用它作注意力块里的前馈层）里，我们用了带 `tanh` 近似的 `GELU` 激活。<br>
我们已经讲过它，但来看看这个功能等价、更冗长的 `tanh` 近似 `GELU` 版本：

```python
class TanhGELU(nn.Module):
    def forward(self, input):
        # Formula from https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        # This works identically to nn.GELU(approximate='tanh')
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
```

如果我们不编译、直接解释运行这段代码，我们要等执行到那一行才会遇到例如 `torch.pow`。只有那时我们才开始部署一个 kernel 来计算输入张量的幂。解释执行中这种"突如其来的意外"会产生开销。

---

GPU 和 CPU 一样，有能并行执行多条指令的核心。但 GPU 比 CPU 核多得多，专门为并行执行而优化。<br>
GPU 核在架构上也更简单，这使它在跨多个数据点执行同一条指令时更快。<br>
如果我们事先就知道 `TanhGELU` 里有 `torch.pow` 这个操作，我们就能把模型编译成让 GPU 并行执行这些简单操作块。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2021/09/GPU-memory-oversubscription.png)<br>
来源：[Improving GPU Memory Oversubscription Performance](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)

这张图里缺的是缓存。CPU 和 GPU 都有分级缓存，比高带宽显存（HBM）或主存（RAM）快得多，但也小得多。<br>
因此，在正确的时刻把正确的数据加载进缓存，对这里的性能至关重要。<br>你能在这里看到各内存类型之间的速度差异：

![](https://miro.medium.com/v2/resize:fit:500/format:webp/1*Diit8xd9fe27lZNM9jMW1Q.png)<br>
来源：[ahmdtaha.medium.com](https://ahmdtaha.medium.com/flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness-2a0aec52ed3d)

放大看 A100 GPU，我们实际上能看到芯片上 L1 和 L2 缓存的位置：

![](https://miro.medium.com/v2/resize:fit:1500/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg)<br>
来源：[NVIDIA - A100 GPU 白皮书（第 20、22 页）](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) 经 [jonathan-hui.medium.com](https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e)

具体到我们的例子，我们本可以事先把 `torch.pow` 所需的数据传到 GPU 缓存里，让它准时待在缓存中、方便 tensor core 访问。<br>
关键在于，现在我们可以预加载它，*并*让它在那儿待到不再需要为止，因为在模型执行过程中我们之后会多次用到它，<br>
比如和标量 `0.044715` 相乘。这种对模型的**前瞻式执行优化**就是 PyTorch 的 `torch.compile` 替我们做的事。

> `torch.compile` 减少了我们否则会因 Python 是解释型语言而遇到的开销。它让我们能在执行*之前*把操作以最合适的方式放进内存。不再是"突如其来的意外"，而是"事后之明"的好处。

### FlashAttention 及编译的一个坑

有些操作并不被 PyTorch 的编译器原样支持。<br>
一个例子是 [Flash Attention \[Dao et al. 2022\]](https://arxiv.org/abs/2205.14135)，它是注意力机制的一项基础性能优化。

![](./img/flashattention_fused_kernel.png)<br>
来源：[\[Dao et al. 2022\]](https://arxiv.org/abs/2205.14135)

在 [`train_gpt2_1.py`](./train_gpt2_1.py) 和 [`train_gpt2_2.py`](./train_gpt2_2.py) 里，我们在 `CaustalSelfAttention` 的前向传播中用了这个写法：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
```

Flash Attention 论证说，这种堆叠的操作（先矩阵乘、再掩码、再 softmax、再矩阵乘）可以在算法上改写成单个融合操作，从而快得多。<br>
但我们的 `torch.compile()` 并不会自己做这个优化，因为它需要一些算法上的调整。

> Flash Attention 的核心思想是避免对那个大的 `att` 矩阵（形状为 `(B, nh, T, T)`）做 GPU 写入 HBM 的操作。<br>
> 这是通过实现 **online softmax trick** [\[Milakov \& Gimelshein 2018\]](https://arxiv.org/abs/1805.02867) 来实现的。<br>
> 单靠 `torch.compile()` 找不到这种程度的、针对内存用量的深远优化。

现在，在当前的 [`train_gpt2_3.py`](./train_gpt2_3.py) 里，我们可以像这样手动实现这一优化：
    
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

不用 `torch.compile()`，仅靠 Flash Attention，我在 3060 上得到 `dt: 40637.73ms, tok/sec: 403.17`，<br>
这确实开始让这张 GPU 看起来有点能用了。

### 去掉"丑数"

模型训练中存在"漂亮数"和"丑数"。<br>
例如，一个批次里的序列数应是 $2$ 的倍数，以充分利用 GPU 的并行处理能力。<br>
相反，奇数或质数则会在某种程度上妨碍下游的并行化努力。

> 我们现在的优化启发式，就是从模型训练流水线里去掉"丑数"。

我邀请你比较一下未优化的 [`train_gpt2_2.py`](./train_gpt2_2.py) 和现已优化的 [`train_gpt2_3.py`](./train_gpt2_3.py) 里的数字。

一个很糟糕的丑数案例，是我们模型里按官方 GPT-2 实现的 `vocab_size` `50257`。<br>
这个数又大又是奇数。好在它不是质数。<br>
一个更漂亮的替代是 `50304`。它能被 $8,\ 16,\ 32$ 和 $128$ 整除。

要更新这个值，我们只需把 [`train_gpt2_3.py`](./train_gpt2_3.py) 里的 `GPTConfig` 数据类改成这样：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
```

一张不带 `torch.compile()` 的 3060 现在得到 `dt: 12998.15ms, tok/sec: 1260.49`。<br>
对比最初的 `dt: 147900.74ms, tok/sec: 110.78`。<br>

不过，我们刚刚也引入了更多我们明知永远不会被用到的 token，因为它们是我们人为加进去的。<br>
它们不可能属于词表。

但在我们的情况下这没问题，因为用我们小小的 `Tiny-Shakespeare` 数据集，词表里大多数 token 反正也用不到。<br>
这个数据集按设计就不会耗尽哪怕 `50257` 个 token 所提供的词表空间。<br>
所以，把 `vocab_size` 改成"漂亮数"并不会给我们带来什么真正的阻力。

另外注意，`torch.compile()` 也不会替我们做这个优化。<br>
我们需要根据应用场景，按自己判断手动来做。

### 算法层面的适配

视你的配置而定，到这一步，按现在的样子运行 [`train_gpt2_3.py`](./train_gpt2_3.py)，相比 [`train_gpt2_2.py`](./train_gpt2_2.py)（在那里设 `train_loader = DataLoaderLite(B=16, T=1024)`），你会体验到约 15 倍的计算时间下降。（我用完全未优化的起始配置得到 `dt: 195892.37ms, tok/sec: 83.64`）<br>

**我到此会停止在 [`train_gpt2_3.py`](./train_gpt2_3.py) 上的工作，让它描述我们优化过程中一个扎实的中间步骤，也就是我们到目前为止所做的事。**<br><br>
**不过还有更多优化可做，尤其是算法层面的改动。这些从现在起会在 [`train_gpt2_4.py`](./train_gpt2_4.py) 里讲。**

[GPT-2 论文 \[Radford et al. 2018\]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)和 [GPT-2 仓库](https://github.com/openai/gpt-2/blob/master/src/model.py)在详细超参数配置方面都没什么真正的信息量。<br>
幸运的是，[GPT-3 论文 \[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165)讲得更深入。

#### Adam / AdamW 优化器配置

第一条规范是使用 `Adam` 优化器，取 $\beta_1 = 0.9$、$\beta_2 = 0.95$、$\epsilon = 10^{-8}$。我们在这里集成它：<br>
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

#### 全局范数裁剪

接下来，我们把梯度的全局范数裁剪在 $1.0$，作为一个"最大范数"。<br>
这意味着训练时，如果梯度的全局范数（所有梯度平方和的平方根）超过 $1.0$，梯度会被缩小，<br>
从而有效避免梯度爆炸或个别的大更新（例如由个别"倒霉"批次引起），整体上稳定训练。

和之前的许多优化一样，这归结为模型里多加一行代码：
```python
# Clipping the global norm at 1.0
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

第一个 epoch 现在得到 `step    0 | loss: 10.947495 | norm: 28.5703 | dt: 13934.98ms | tok/sec: 1175.75`。<br>
训练过程中，norm 参数会趋向 `0`。

#### 学习率衰减

GPT-2 和 GPT-3 并不像我们现在这样用固定学习率。<br>
相反，它们采用余弦衰减的学习率调度，使得在迭代过特定数量的 token（2600 亿 token）之后，学习率衰减到只有起始值的 $10\%$。同样地，前 3.75 亿个 token 期间有学习率的线性预热/爬升。

下面是余弦衰减方法下学习率随时间调整的方式：

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*BJCssPOCn4u__NoAZs392w.png)<br>
来源：[scorrea92.medium.com](https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b)

实现大概长这样：
```python
max_lr = 6e-4 # According to GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        # 1) Linear warmup region for warmup_iters steps
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        # 2) if it > lr_decay_iters, flat out return the min_lr
        return min_lr
    # 3) In between warmup and max_steps, cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to zero
    return min_lr + coeff * (max_lr - min_lr)
```

我们以一种很有意思的方式把这个衰减的学习率引入训练循环，因为我们相当于覆盖掉旧设置并把它注入进去：
```python
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping the global norm at 1.0
    # Determine and set the learning rate for this iteration step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish above scheduled workload
    t1 = time.time()
    dt = (t1 - t0) # millisecond time difference
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
```

我的第一步现在长这样：`step    0 | loss: 10.947495 | lr: 6.0000e-05 | norm: 28.5703 | dt: 12595.26ms | tok/sec: 1300.81`。

#### 动态增大批大小

在 [\[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165) 的附录 `B - Details of Model Training` 里，我们可以看到批大小会随训练推进而增大。这么做是因为在训练早期阶段，我们大致是在学习"用上某些 token、忽略当前上下文里的另一些"。这种"要用/不用"对 token 关系的影响，在这个早期阶段几乎对批次里每个样本都存在，使得它们的梯度一开始看起来有点相似。可以把它想成输入促使模型对"哪些内部表示需要、哪些不需要"做一次粗略的初步分拣。

由于训练早期就是这种情形，我们暂时还并不真的需要很大的批大小。<br>
我们的模型其实不需要这个，因为它只在数据集大得多时才真正有意义。我们跳过它。

#### 数据加载

[GPT-3 论文 \[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165) 还指出这一点：
> "Data are sampled without replacement during training (until an epoch boundary is reached) to minimize overfitting."

这意味着我们从训练数据里取出并组装一个批次，让模型接触这个批次，但之后在这个 epoch 内不再复用这个批次的内容。<br>
幸运的是，我们已经这么做了，正如我们的 `DataLoaderLite` 用滑动窗口一步步在训练集上移动、无放回地组装批次所见。

#### 权重衰减与 Fused AdamW

最后，所有模型都使用 $0.1$ 的权重衰减。<br>
我们会把这个概念注入到 `AdamW` 优化器的创建中，方式是把创建过程包进一个单独的函数 `configure_optimizers`：

```python
import inspect

def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, but all biases and layer-norms don't.
        # Effectively splitting the parameters into two groups: decayable and non-decayable
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # 1D tensors (layer norms, biases) are not decayed
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # Essentially checking here if the option 'fused' is available in the AdamW optimizer (depends on PyTorch version)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = torch.cuda.is_available() and fused_available and device.startswith('cuda')
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
```

把所有权重都往下拉，能促使（所有）权重被均匀地使用，这对泛化有好处。<br>
`fused` 参数挺有意思：它不再让优化器逐个遍历参数、一个个更新，而是部署一个单独的 kernel 一次性更新所有参数。<br>
这是一个显著的加速，尤其对大模型而言。

我们通过把原生的 `AdamW` 初始化替换为 `configure_optimizers` 函数调用，把它注入到实际训练过程中：

```python
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
```

这会返回：<br>
`num decayed parameter tensors: 50, with 124354560 parameters`<br> 
`num non-decayed parameter tensors: 98, with 121344 parameters`

`fused` 只在某些 CUDA 设备上工作。能用就用。

#### 用梯度累积模拟大批大小

我们再来看一下 GPT-3（和 GPT-2）的规模、架构和学习超参数：

![](./img/gpt3_train_table.png)<br>
来源：[\[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165)

我们这里无法复现的一件事，是原始的 $500,000$ 个 token 的批大小。<br>
我们目前是每批 $16$ 个序列 $\times 1024$ 个 token。<br>
如果想保持这个上下文大小，我们需要每批 $\frac{500,000}{1024}\approx 488$ 个序列。<br>
我不知道你的配置如何，但我的 3060 要真敢这么干，会提前原地解体。

问题是，这 $500,000$ 个 token 的批大小跟学习率、层数、嵌入大小等是相关的。<br>
听起来再疯狂，我们也确实想用上这个巨大的批大小，因为它（至少按 OpenAI 的设置）能与我们已在用的其他超参数配合，把模型训练发挥到最好。

我们可以用一个技巧，叫**梯度累积**。

通过梯度累积，我们可以在多个批次上累加恢复出的梯度，然后再真正更新模型权重。<br>
这让我们能"模拟"更大的批次，并更高效地利用 GPU 显存进行训练。

把这个引入 [`train_gpt2_4.py`](./train_gpt2_4.py) 需要我们重构 `train_loader = DataLoaderLite(B=16, T=1024)`。

作为第一处改动，我们把 `grad_accum_steps` 参数引入训练设置：

```python
total_batch_size = 524288 # 2**19, ~0.5M tokens, as per GPT-3 paper
# Doing 16384 tokens per micro-batch 
# -> 32 micro-batches to accumulate into a full batch
B = 16 # micro-batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "Batch size must be divisible by (micro-batch size * sequence length)."
# Accumulate gradients over grad_accum_steps instead of backpropagating each step
grad_accum_steps = total_batch_size // (B * T) # 32
print(f"Total desired batch size: {total_batch_size}")
print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")

# DataLoaderLite for micro-batched training data
train_loader = DataLoaderLite(B=T, T=B)
```

注意我们仍用 `DataLoaderLite`，但现在用于 $16 \times 1024$ token 的微批次。<br>
我们在 $32$ 个微批次上累积梯度，实际上是在更新模型权重之前，把跨 $524288$ 个 token 的梯度累加起来。<br>
不过这部分我们还没实现。它在训练循环本身里完成：

```python
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # Reset gradients per macro-batch
    # Inner loop across micro-batches
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # This accumulates now, because we don't call zero_grad() in the inner loop
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping the global norm at 1.0
    # Determine and set the learning rate for this iteration step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish above scheduled workload
    t1 = time.time()
    dt = (t1 - t0) # millisecond time difference
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")  
```
<br>

**看着不错？嗯，它有 bug！**<br>

<br>
我们的问题很微妙，但很要紧。<br>
现在的损失计算基于各微批次损失之和。<br>
这不是我们想要的，因为每个微批次损失都会被该微批次里的 token 数所缩放。<br><br>

> 如果我们只是把这些"按微批次求过均值"的损失相加，得到的和并不能代表在完整批次上算出的正确损失。

我们用个例子来看：

```python
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 1)
)

torch.random.manual_seed(42)

x = torch.randn(4, 16)  # 4 samples, 16 features
y = torch.randn(4, 1)   # 4 samples, 1 target
net.zero_grad()
y_hat = net(x)
loss = torch.nn.functional.mse_loss(y_hat, y)
loss.backward()
# Show the first 10 gradients as example
grads_macro = net[0].weight.grad.view(-1)[:10]
print(grads_macro)
```
```
tensor([-0.0150,  0.0011,  0.0042, -0.0040,  0.0059, -0.0080, -0.0078, -0.0138,
        -0.0103, -0.0134])
```

实际上，上面这段代码执行的是这个 $\text{MSE}$ 损失计算：
$$\text{Loss} = \frac{1}{4}\times \left[ (y_0 - \hat{y}_0)^2 + (y_1 - \hat{y}_1)^2 + (y_2 - \hat{y}_2)^2 + (y_3 - \hat{y}_3)^2 \right]$$

下一段代码展示了我们上面实际上实现了什么：

```python
net.zero_grad()
for i in range(x.shape[0]):
    # Forward ONLY one sample micro-batch at a time
    y_hat = net(x[i])
    # Compute loss per each micro-batch
    loss = torch.nn.functional.mse_loss(y_hat, y[i])
    loss.backward()

# Show the first 10 gradients as example
grads_micro = net[0].weight.grad.view(-1)[:10]
print(grads_micro)
print(f"\nGradients are identical: {torch.allclose(grads_macro, grads_micro)}")
```
```
tensor([-0.0598,  0.0042,  0.0167, -0.0161,  0.0235, -0.0320, -0.0311, -0.0550,
        -0.0410, -0.0536])

Gradients are identical: False
```

问题在于，这段代码实际上是这样计算损失的：
$$\text{Loss} = \frac{1}{1}\times \left[ (y_0 - \hat{y}_0)^2 \right] + \frac{1}{1}\times \left[ (y_1 - \hat{y}_1)^2 \right] + \frac{1}{1}\times \left[ (y_2 - \hat{y}_2)^2 \right] + \frac{1}{1}\times \left[ (y_3 - \hat{y}_3)^2 \right]$$

> 默认情况下，单纯把微批次损失相加，并不等同于在完整批次上算出的损失。

我们可以这样解决这个行为：

```python
net.zero_grad()

for i in range(x.shape[0]):
    # Forward ONLY one sample micro-batch at a time
    y_hat = net(x[i])
    # Compute loss per each micro-batch
    loss = torch.nn.functional.mse_loss(y_hat, y[i])
    
    # NEW: Scale the loss down, 
    # normalize loss by relation of micro-batch size [this is an incomplete explanation, see below]
    loss = loss / x.shape[0]

    loss.backward()

# Show the first 10 gradients as example
grads_micro = net[0].weight.grad.view(-1)[:10]
print(grads_micro)
print(f"\nGradients are identical: {torch.allclose(grads_macro, grads_micro)}")
```
```
tensor([-0.0150,  0.0011,  0.0042, -0.0040,  0.0059, -0.0080, -0.0078, -0.0138,
        -0.0103, -0.0134])

Gradients are identical: True
```

我们发现：
> 微批次要求我们按该微批次对完整批次的贡献来缩放损失。<br>
> 这通过用 $\frac{\text{macro-batch size}}{\text{micro-batch size}}$ 这一项对微批次损失做归一化来完成。

上面的解释可能让我们对此仍有点摸不着头脑。<br>
当我们看另一个例子时，微批次和宏批次之间的关系就变明显了，<br>
这次批大小是 $2$ 而不是 $1$：

```python
net.zero_grad()

micro_batch_size = 2

# Run with old configuration
for i in range(0, x.shape[0], micro_batch_size):
    # Forward pass for each micro-batch
    y_hat = net(x[i:i + micro_batch_size])
    # Compute loss for the micro-batch
    loss = torch.nn.functional.mse_loss(y_hat, y[i:i + micro_batch_size])
    # Normalize loss by micro-batch size [this would be identical to the above approach, but its incomplete]
    loss = loss / (x.shape[0]) # Just normalizing by full batch size will not suffice
    # Backward pass
    loss.backward()

# Show the first 10 gradients as example
grads_micro = net[0].weight.grad.view(-1)[:10]
print(grads_micro)
print(f"\nGradients are identical: {torch.allclose(grads_macro, grads_micro)}\n\n")

net.zero_grad()

# Run with new configuration
for i in range(0, x.shape[0], micro_batch_size):
    # Forward pass for each micro-batch
    y_hat = net(x[i:i + micro_batch_size])
    # Compute loss for the micro-batch
    loss = torch.nn.functional.mse_loss(y_hat, y[i:i + micro_batch_size])
    # Normalize loss by relation of full batch size to micro-batch size [!!!]
    loss = loss / (x.shape[0] / micro_batch_size)
    # Backward pass
    loss.backward()

# Show the first 10 gradients as example
grads_micro = net[0].weight.grad.view(-1)[:10]
print(grads_micro)
print(f"\nGradients are identical: {torch.allclose(grads_macro, grads_micro)}")
```
```
tensor([-0.0075,  0.0005,  0.0021, -0.0020,  0.0029, -0.0040, -0.0039, -0.0069,
        -0.0051, -0.0067])

Gradients are identical: False


tensor([-0.0150,  0.0011,  0.0042, -0.0040,  0.0059, -0.0080, -0.0078, -0.0138,
        -0.0103, -0.0134])

Gradients are identical: True
```

> 我们需要谨慎地按完整批大小与微批大小的比例关系，对微批次损失做归一化。

如果我们把这个实现进 `GPT` 类，优化循环的内层循环就会有这样的改动：

```python
# Optimization loop
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # Reset gradients per macro-batch
    loss_accum = 0.0 # Loss accumulator over the micro-batches
    # Inner loop across micro-batches
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # Normalize micro-batch loss by relation of full batch size to micro-batch size
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # This accumulates now, because we don't call zero_grad() in the inner loop
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping the global norm at 1.0
    # Determine and set the learning rate for this iteration step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish above scheduled workload
    t1 = time.time()
    dt = (t1 - t0) # millisecond time difference
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps # Tokens processed per step
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}") 
```
<br>

---

**Q:** **为什么我们只用 `grad_accum_steps` 来归一化，而不是用像上面那样的某种表达式 `(total_batch_size / mini_batch_size)`？**
<br>
<br>

**A:** 实际上，我们正是那么做的。`grad_accum_steps` 已充分表达了这一关系：
$$\frac{\text{grad\_accum\_steps} \times \text{B} \times \text{T}}{\text{B} \times \text{T}} = \text{grad\_accum\_steps}$$

**还有一件事：** 上面我抄了整个优化循环，因为 `tokens_processed` 的计算也有改动。<br>
现在我们在每步处理 token 数的计算里额外把 `grad_accum_steps` 算进去，实际上把我们使用微批次这一事实纳入了计算。

---

**现在我们可以训练模型了！**<br>
如果你想试着训练，但只有一个小型 CUDA 设备，你可以把微批大小减小到比如 $8$ 或 $4$ 个序列。<br>
只需在初始化时改一下这个参数即可，其余全自动处理。

我用 NVIDIA 3060 跑了一次测试，设 `B = 8`，这让 $50$ 步训练花了约 $\sim 1$ 小时。<br>
第 $05$ 步我得到：`step   05 | loss: 8.609741 | lr: 3.6000e-04 | norm: 3.0927 | dt: 56625.14ms | tok/sec: 144.67`。<br>
第 $49$ 步我得到：`step   49 | loss: 5.558484 | lr: 6.0832e-05 | norm: 0.2166 | dt: 50348.39ms | tok/sec: 156.45`

## 多 GPU 训练

我们已经为 GPT-2 克隆的代码做了大量优化。具体来说，我们通过 PyTorch 设置、通过算法优化（如学习率衰减、权重衰减和梯度范数裁剪、`AdamW` 优化器配置以及带梯度累积的 `DataLoaderLite` 实现）做了优化。<br>

**改动不少。** 我们确实让代码更高效更快了。<br>
正因为如此，我们现在可以考虑在多个 GPU 上训练。<br>
如果没有已经到位的优化，这么做会太贵。

让我们拿出重武器，用 `DistributedDataParallel` 搭建多 GPU 训练。

**这一版训练脚本演进的所有代码都在 [`train_gpt2_5.py`](./train_gpt2_5.py) 里。**

### 引入 DistributedDataParallel

`DistributedDataParallel` 背后的想法是，把模型分到与 GPU 数量一样多的训练进程上，让每个 GPU 处理模型的一部分。<br>
然后，梯度在所有 GPU 间被收集并求平均，模型权重据此更新。<br>
这个训练任务会通过 `torchrun` 启动。

[`train_gpt2_5.py`](./train_gpt2_5.py) 脚本里新增的部分是这段设置代码：
```python
# Set up the DDP (distributed data parallel) environment
ddp = int(os.environ.get('RANK', -1)) != -1 # check if we are in a DDP environment/run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "DDP requires CUDA for now - please run on a GPU node."
    init_process_group(backend='nccl')  # initialize distributed backend
    ddp_rank = int(os.environ['RANK'])  # rank of current process
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # local rank/GPU index of current process
    # (ddp_local_rank means within the node, ddp_rank is global, meaning across all nodes; node: machine with multiple GPUs (just saying))
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # number of processes in the DDP environment
    device = f"cuda:{ddp_local_rank}"  # map the device according to the local rank (indicates which GPU to use on the node)
    torch.cuda.set_device(device)      # set the device to the local rank
    master_process = ddp_rank == 0     # check if current process is master (master does logging, checkpointing etc.)
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # Find the best available device to train on
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_initialized():
        device = "mps" # Apple Silicon
    print(f"Using device: {device}")
```

对我们来说这看着可能有点怪，因为这（目前）真的是脚本里唯一的改动。但用 `torchrun` 运行时，这个脚本会按这套设置在多个 GPU 上执行。<br>
`LOCAL_RANK` 指节点上的 GPU 索引，`RANK` 是跨所有节点的全局排名，`WORLD_SIZE` 是我们要运行的 DDP 环境里进程的总数。<br>

如果你（像通常那样）有 `8` 个 GPU 可用，这段脚本会被这 `8` 个进程各自读取，每个进程会被分配一个从 `0` 到 `7` 的不同 `LOCAL_RANK`。<br>
这段代码过后，其余训练代码也会被每个进程读取。<br>
但我们并不希望每个进程都执行完整的训练代码。<br><br>
因此，**我们现在得改造训练代码，使其能（尽可能均匀地）分发到多个 GPU 上。**

### 适配梯度累积

前几处改动是针对微批处理的：

```python
total_batch_size = 524288 # still the same
B = 16 # still the same
T = 1024 # still the same
assert total_batch_size % (B * T * ddp_world_size) == 0, "Batch size must be divisible by (micro-batch size * sequence length * ddp_world_size)."
# Accumulate gradients over grad_accum_steps instead of backpropagating each step
# Each process does B * T, and there are ddp_world_size many processes
# E.g. on 8 GPUs, we will now learn on 16 * 1024 * [8] = 131072 tokens per step before backpropagating
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
```

如果我们想要并行，我们实际上是要并发处理"$n$ 个独立的工作块"，以便在 $\text{timeframe}$ 内得到所有这些块的结果，而不是只在 $n \times \text{timeframe}$ 内顺序得到。我们想多处理的倍数就是我们可用的 GPU 数，在我们的情形里就是 `ddp_world_size`。据此，我们调整 `grad_accum_steps`，让它累积我们 `8` 个 GPU 能并行处理的量，也就是每 GPU `B * T` 个 token。

> 因为我们并行化了，所以能在同样时间内处理更多。这就是为什么我们能增加每步处理的 token 数，这实际上也降低了覆盖一个完整批次所需的微批数。

### 适配 DataLoader

接下来，我们需要让 `DataLoaderLite` 感知到分布式设置。具体来说，每个进程都不应加载相同的数据。<br>
每个进程应工作在它自己的数据子集上。**对此的解决方案挺利落：**

```python
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B # batch size
        self.T = T # context size
        self.process_rank = process_rank # rank of the current process
        self.num_processes = num_processes # total number of processes

        # at init load tokens from disk and store them in memory
        with open('../tiny-shakespeare.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text) # encode full text into tokens
        self.tokens = torch.tensor(tokens) # wrap with tensor
        self.tok_count = len(self.tokens)
        # Just some stats for us nerds
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # token-level start position for each next batch
        # NEW: This changes in relation to the process rank now, as to stride out processes across the data
        self.current_position = self.B * self.T * self.process_rank # Process 0 starts at 0, Process 1 at (next batch) B*T, etc.

    def next_batch(self):
        B, T = self.B, self.T
        # grab a chunk of tokens of size B * T + 1 (we explained this before)
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # input tensor of size (B * T)
        y = buf[1:].view(B, T)  # target tensor of size (B * T), right-shifted 1 position
        # advance position in data tensor
        # NEW: Jump to the next batch, taking into account the number of processes
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        # NEW: Check if our parallel-process-aware jump would be out of bounds
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            # reset position to beginning of data, w.r.t. process rank
            self.current_position = self.B * self.T * self.process_rank
        return x, y
```
（见标注 `NEW` 的部分）

我们现在额外把调用方的 `process_rank` 和 `num_processes` 传给 `DataLoaderLite` 构造函数，以了解进程版图。<br>
对每个进程，我们随后以一个由 `process_rank` 决定的批次偏移开始遍历数据集，即 `self.current_position = self.B * self.T * self.process_rank`。<br>
此外，每个进程现在在数据集里前进的方式也不同。我们在这里的解决办法是让每个进程"跳"到下一个批次时步进 `self.current_position += B * T * self.num_processes`。<br>
各进程现在并行地在数据集上跳到各自被专门分配的子集。

### 完整接入 DistributedDataParallel

还缺少的一点是：通过进程间通信来对梯度求平均的能力。<br>
我们只需把模型用 `DistributedDataParallel` 包起来就能很快做到：

```python
model = GPT(GPTConfig()) # random weight initialization
model.to(device)

# Check if we run on Linux:
if os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # compile model to TorchScript -> speed + memory savings

if ddp:
    # Once each GPUs backward is over, DDP will average the gradients across all GPUs
    model = DDP(model, device_ids=[ddp_local_rank])
```

如果处于分布式设置中，我们就把模型用 `DDP` 包起来。`device_ids` 参数设为该进程的 `ddp_local_rank`，即节点上的 GPU 索引。<br>
这是我们要让模型训练设置变成分布式所需做的唯一改动。进程间通信由 `DDP` 在底层处理。

### 分发训练循环

目前，由于微批处理，我们对每个微批次都做一次按比例缩放的 `loss.backward()` 调用。<br>
这没问题，但我们不想在每个微批次之后就在所有 GPU 间同步梯度。<br>
相反，我们想在完整批次处理完之后、即组成这个完整批次的最后一个微批次处理完之后，再同步梯度。

这是更新后的完整训练循环。*系好安全带：*

```python
# Optimization loop
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # Reset gradients per macro-batch
    loss_accum = 0.0 # Loss accumulator over the micro-batches
    # Inner loop across micro-batches
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # Normalize micro-batch loss by relation of full batch size to micro-batch size
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # DDP will take care of averaging the loss across all GPUs
            # DDP will do so only after the last micro-batch in each accumulation cycle
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # This accumulates now, because we don't call zero_grad() in the inner loop
        loss.backward()
    if ddp:
        # Average the macro-batch loss across all GPUs, 
        # after having done that, synchronize this average loss to be the same value set across all GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping the global norm at 1.0
    # Determine and set the learning rate for this iteration step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for GPU to finish above scheduled workload
    t1 = time.time()
    dt = (t1 - t0) # millisecond time difference
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # Tokens processed per step across all GPUs
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")      

if ddp:
    destroy_process_group() # DDP cleanup
```

通过调用 `model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)`，我们本质上是在等到处理完累积周期里的最后一个微批次后，才跨所有 GPU 同步（构成宏批次梯度的）那些梯度。<br>
然后，为对分布式模型的性能有个概念，我们用 `dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)` 跨所有 GPU 对累积损失 `loss_accum`（即完整批次损失的指标）求平均。<br>
然后我们通过主进程把这个平均值作为性能指标显示出来。<br>
最后，为稳妥起见，我们在训练结束后清理 DDP 环境。

> 如果你现在想看这套设置实跑，不要用 `python train_gpt2_5.py` 运行，而要用 `torchrun --standalone --nproc_per_node=8 train_gpt2_5.py`（如果你有 `8` 个 GPU 的话）。<br>按你的 GPU 数调整 `--nproc_per_node`。

## 我们确实需要更多数据

既然我们的训练设置已经颇具威力了，我们确实应该考虑在更多数据上训练。<br>
更重要的是，我们真的应该把数据集扩大到我们那点 Shakespeare 文本之外，以避免过拟合并让模型更通用。

据 [GPT-2 论文 \[Radford et al. 2018\]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)，GPT-2 模型是在从 Reddit 外链爬取的数据上训练的，且链接须获得至少 $3$ 个点赞。<br>
这形成了 $40$ GB 的文本数据集。

[GPT-3 论文 \[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165) 指出，GPT-3 模型是在 $570$ GB 文本数据上训练的，这些数据用 Common Crawl（一个网络爬虫）组装而成。

更具体地说，GPT-3 训练数据集由以下来源组成：

![GPT-3 Training Data Sources](./img/gpt3_training_data_sources.png)<br>
来源：[\[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165)

注意，*这个数据集从未公开发布*。<br>但有公开可用的替代品，如 [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)、[FineWeb](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html) 或 [FineWeb-EDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。

我们将用 `FineWeb-EDU` 数据集来训练。<br>
具体来说，我们用完整数据集的 `sample-10BT` 子集。

我们会通过 [`fineweb.py`](./fineweb.py) 脚本下载数据集并为训练做预处理。<br>
为此我们需要一个 `./edu_fineweb10B` 文件夹。numpy 数据分片会创建在那里。<br>
第一个分片 `edufineweb_val_000000.npy` 持有验证集。<br>
其余分片会有类似 `edufineweb_train_000001.npy` 这样的名字。

总共你会得到 $100$ 个数据分片。

### DataLoader 调整

我们的 `DataLoaderLite` 需要调整以加载这些数据分片，而不是 Shakespeare 文本。<br>
我不会把调整贴在这里，因为从逻辑上讲处理过程并没有真正改变。<br>
只是把它适配成了基于分片的数据加载。<br><br>
你当然仍能在 [`train_gpt2-5.py`](./train_gpt2_5.py) 里找到它。

## 评估、日志与可视化

现在我们的脚本已经工作得很好了，但我们还没做任何日志记录、评估或可视化。<br>
让我们先从在验证集上做评估开始。

为了评估，我们创建一个单独的 `DataLoaderLite` 类 `val_loader` 实例，它会加载验证数据分片。<br>
然后每 $100$ 个宏批次，我们在验证集上跑若干步来评估模型，并打印累积的验证损失。

这是我们要注入训练循环的内容：

```python
# Optimization loop
for step in range(max_steps):
    t0 = time.time()

    # once in a while, check on the validation set
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.6f}")
    # [...]
```

同样，除了最开始那一次，每 $100$ 步我们*终于*会展示一些模型在该训练时刻生成的样本文本。<br>

```python
    # [the ... from above]
    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # seed the number generator for reproducibility
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    # [...]
```

这本质上和我们在 [`train_gpt2_1.py`](./train_gpt2_1.py) 里用的逻辑完全一样。<br>
唯一的小区别是我们现在一次生成多个样本，并用了一个带种子的随机数生成器以便复现。

### HellaSwag 评估

我们不仅能在验证集上评估、还能在所谓的 HellaSwag 评估 [\[Zellers et al. 2019\]](https://arxiv.org/abs/1905.07830) 上对模型进行评估，从而获得对模型性能更有意义的洞察。

HellaSwag 是一个句子补全数据集，要求模型运用其获得的世界知识和推理能力来补全句子。

![](./img/hellaswag_question.png)<br>
来源：[\[Zellers et al. 2019\]](https://arxiv.org/abs/1905.07830)


这个评估其实也用于 [GPT-3](https://arxiv.org/abs/2005.14165)。而且它相当好，因为它为我们提供了关于模型性能的所谓"早期信号"。<br>
即便训练早期，我们也能看出模型在这个任务上做得如何、以及它如何随时间改进。

虽然这些都听起来不错、也合理——这个评估能用来测试模型的性能和泛化能力——但目前还不清楚如何在我们训练脚本里实现它。<br>
我们怎么让我们的文本生成模型回答多选题呢？

我们不能假设我们仍然相当小的模型已经掌握了回答多选题的概念。<br>
相反，我们可以把问答对——比如把同一个问题分别与四个候选答案各拼接一次——<br>
以一批问答序列的形式提供。<br>
我们用填充 token 把较短的序列填到一样长，长度取批次中最长序列的长度。<br>
批次中只有一条序列会包含对该问题的正确答案。

> 我们会说，平均 token 概率最高的那条序列就是模型所预测的包含答案的那条。<br>
> 其背后的直觉是，语言模型认为这条序列是紧跟该问题之后整体最可能的那条。

注意，这套设置既特定于模型、也特定于 HellaSwag 评估。其他评估可能需要不同设置。<br>
例如，在我们的设置里，我们不让模型一次看到所有候选答案，而是一次只看一个。<br>
这可根据我们对模型的要求来调整，但作为这里的起点已经不错。

这种用 `HellaSwag` 评估模型的实现，可以在 [`hellaswag.py`](./hellaswag.py) 脚本里找到。<br>
（`evaluate` 函数是理解该实现的一个好入口。）

如果这个 [`hellaswag.py`](./hellaswag.py) 脚本对 `gpt2 (124M)` 运行，它会以 `0.2955` 的分数评估模型。<br>
作为参考，随机猜对的概率是 `0.25`，所以 `gpt2` 只比随机略好一点。

评估 `gpt2-xl (1558M)` 时能看到一个真正的跃升，它得分 `0.4893`。<br>
这是相比 `gpt2` 模型的显著提升，但仍然不完美。

**在我们的设置下，我们的目标是打败 `gpt2 (124M)` 模型的基准。**

为纳入这个评估，我们得改造 [`train_gpt2_5.py`](./train_gpt2_5.py) 脚本。<br>
这可以在主训练循环里这样完成：

```python
# once in a while, evaluate the hellaswag dataset
if (step % 250 == 0 or last_step) and (not use_compile):
    model.eval()
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        # Only process examples where i % ddep_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # Render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # Get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(tokens, mask)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag acc_norm: {num_correct_norm}/{num_total}={acc_norm:.6f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hellaswag {acc_norm:.6f}\n")
```

我们把 HellaSwag 评估集成进 [`train_gpt2_5.py`](./train_gpt2_5.py) 的主训练循环，让评估每 $250$ 步运行一次。<br>
它向模型注入评估批次，并据正确标签判定模型预测的最可能序列。<br><br>
所有这些也都针对分布式数据并行做了优化，所以我们能跨多个 GPU 有效运行同步评估。<br>
为跟踪模型性能，结果会随时间记录在 `log.txt` 里。

### 可视化训练进度

有了损失计算和评估，训练一旦完成，我们就能可视化训练进度。<br><br>
例如，可以用下面的代码来做：

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

sz = "124M"

loss_baseline = {
    "124M": 3.2924,
}[sz]

# HellaSwag for GPT-2
hella2_baseline = {
    "124M": 0.294463,
    "350M": 0.375224,
    "774M": 0.431986,
    "1558M": 0.488946,
}[sz]

# EleutherAI for GPT-3
hella3_baseline = {
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.510,
    "1558M": 0.547,
}[sz]

# load the log file
with open("log124M/log.txt", 'r') as f:
    lines = f.readlines()

# parse the individual lines, group by stream (train, val, hella)
streams = {}
for line in lines:
    step, stream, val = line.strip().split()
    if stream not in streams:
        streams[stream] = []
    streams[stream][int(step)] = float(val)

# convert each stream from {step: val} to {steps[], vals[]}
# so it's easier for plotting
streams_xy = {}
for k, v in streams.items():
    # get all (step, val) pairs, sort by step
    xy = sorted(list(v.items()))
    # unpack the list of tuples into tuple of lists
    streams_xy[k] = list(zip(*xy))

# Create Figure
plt.figure(figsize=(16, 6))

# Panel 1: losses - both train and val
plt.subplot(1, 2, 1)
xs, ys = streams_xy["train"] # train losses
plt.plot(xs, ys, label=f"NanoGPT ({sz}) - Train Loss")
print("Min Train Loss:", min(ys))
xs, ys = streams_xy["val"] # val losses
plt.plot(xs, ys, label=f"NanoGPT ({sz}) - Val Loss")

# Horizontal line at GPT-2 baseline
if loss_baseline is not None:
    plt.axhline(y=loss_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) Loss Baseline")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.ylim(0.0, 4.0)
plt.legend()
plt.title("Loss")
print("Min Validation Loss:", min(ys))

# Panel 2: HellaSwag Eval
plt.subplot(1, 2, 2)
xs, ys = streams_xy["hella"] # HellaSwag scores
ys = np.array(ys)
plt.plot(xs, ys, label=f"NanoGPT ({sz}) - HellaSwag")
# Horizontal line at GPT-2 baseline
if hella2_baseline is not None:
    plt.axhline(y=hella2_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) HellaSwag Baseline")
if hella3_baseline is not None:
    plt.axhline(y=hella3_baseline, color='g', linestyle='--', label=f"OpenAI GPT-3 ({sz}) HellaSwag Baseline")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.title("HellaSwag Eval Accuracy")
print("Max HellaSwag:", max(ys))
```

作为参考，它可能长这样：

![](./img/plot_loss_hellaswag.png)

你可以看到，如果跑得足够久，我们 [`train_gpt2_5.py`](./train_gpt2_5.py) 脚本在 `FineWeb EDU` 数据集上的验证损失<br>
会显著下降到 `GPT-2` 验证损失基准之下。

同时，HellaSwag 评估准确率会随时间上升，并超越 `GPT-2` 模型基准，几乎达到 `GPT-3` 水平。<br>
这有点出奇，因为我们训练用的数据集比 `GPT-2` 模型训练用的要小得多。<br>
但举例来说，可能是原始数据集范围更广、因此在某种程度上噪声更多。<br>
另外，`FineWeb EDU` 全是英文，且不含太多数学或代码。

当然，这多少有点没有可比性，因为我们没有用同一数据集跑同一模型。<br>
但它仍是个不错的指标，因为我们至少把东西做得贴近原版。

## 进一步优化与增强的空间

在未来的改造中，可以改进 `DataLoaderLite`，在组装各 GPU 数据子集时引入更多置换。<br>
这能更好/更均衡地把完整数据集的情况告知各训练子进程，并改善泛化。

另外，事实证明原始的 `max_lr` 学习率相当保守。<br>
我们其实可以把它提高到甚至 `2e-4` 来更快训练。

如果想更接近 `GPT-3` 性能，我们也可以增大模型规模，并<br>
如 [GPT-3 论文 \[Brown et al. 2020\]](https://arxiv.org/abs/2005.14165) 所述，把上下文大小翻倍到 `2048`、把批大小减半到 `32`。

你也可以去尝试其他评估方法，比如 [Eleuther Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)，或在其他数据集上训练，比如 [the Pile](https://pile.eleuther.ai/)。

在 [LLM.c](https://github.com/karpathy/llm.c) 里也藏着我们这版 GPT 的一个实现，那也是 Andrej 做的。它本质上一样，但*明显更快*、且*内存效率高得多*。

<br>
<br>
<br>
<br>

---

<center><br>如果你现在把 <code>train_gpt2-5.py</code> 跑得足够久，你会得到一个在某些方面跟 OpenAI 的 GPT-2 模型一样好、<i>甚至更好</i>的模型。<br>
你要做的就是运行 <code>torchrun --standalone --nproc_per_node=[GPU_COUNT] train_gpt2_5.py</code>，并指定你可用 GPU 的数量。<br><br>
<b>这个脚本（配合正确的数据）在 2018 年会是 SOTA！</b><br><br></center>

---

<center>笔记本由 <a href="https://github.com/mk2112" target="_blank">mk2112</a> 编写。</center>
