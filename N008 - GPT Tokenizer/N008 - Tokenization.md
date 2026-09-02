# 分词

[视频](https://www.youtube.com/watch?v=zduSFxRajkE)
[代码仓库](https://github.com/karpathy/minbpe)
[Eureka Labs Discord](https://discord.com/invite/3zy8kqD9Cp)

## 目录

- [回顾：字符级分词](#回顾字符级分词)
- [分词的陷阱](#分词的陷阱)
- [分词器的思路](#分词器的思路)
  - [Unicode](#unicode)
- [字节对编码](#字节对编码)
  - [解码](#解码)
  - [编码](#编码)
- [走向 SOTA：实战中的分词器](#走向-sota实战中的分词器)
  - [GPT 分词器](#gpt-分词器)
  - [OpenAI TikToken](#openai-tiktoken)
  - [特殊 token](#特殊-token)
  - [Sentencepiece](#sentencepiece)
- [练习时间](#练习时间)
- [回过头来：vocab_size](#回过头来vocab_size)
- [结语](#结语)
- [参考文献](#参考文献)

要理解大语言模型（LLM）的内部工作原理，需要深入到生成过程的最初一步：**分词**。

**分词描述的是把字符序列转换成一串以数值方式表示的 token 的过程。**
**token 是语言模型的基本意义单位。**

LLM 是*纯数学*模型。它们无法直接处理原始文本，而是*只处理数字*。
这个任务看似简单：找到一种最合适的方式，把输入文本映射成某种可被处理的数值表示。

如果这种映射做得不对，这种转换往往会成为诸多下游问题的*根本*原因。

## 回顾：字符级分词

在[上一讲](<../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb>)中，我们已经为 GPT 模型实现了一种简化形式的分词。
这里快速回顾一下那个过程：

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import regex as re
import tiktoken
import os
import json
```

首先，我们把 `tiny-shakespeare.txt` 文本文件里的样本数据集读入内存：

```python
# 读入 txt 文件以查看其内容
with open('../tiny-shakespeare.txt', 'r') as f:
    text = f.read()

# 打印一段文本样本（前 100 个字符）
print("Length of Dataset:", len(text), "\n")
print(text[:100])
```

```
Length of Dataset: 1115394 

First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You
```

接着，我们确定数据集中出现的所有不重复字符：

```python
chars = sorted(list(set(text))) # 找出文本中所有不重复字符
vocab_size = len(chars)         # 词表的长度（含空格字符和换行符）
print('Unique Characters:', ''.join(chars))
print(f'\nVocabulary size: {vocab_size}')
```

```
Unique Characters: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

Vocabulary size: 65
```

给定文本文件中找到的这些不重复字符，我们接着把每一个映射到各自的整数，
从而以纯数值方式表示这些字符、乃至整个数据集：

```python
stoi = { ch:i for i,ch in enumerate(chars) }     # 字符到索引的映射
itos = { i:ch for i,ch in enumerate(chars) }     # 索引到字符的映射

# 这是编码器；把字符串编码成一个整数列表
encode = lambda s: [stoi[c] for c in s]
# 这是解码器；把整数列表解码成字符串
decode = lambda l: ''.join([itos[i] for i in l])

msg = "hii there"
token_list = encode(msg)
print(token_list)
print(decode(token_list))
```

```
[46, 47, 47, 1, 58, 46, 43, 56, 43]
hii there
```

> 上面这种做法叫**字符级分词**。

对输入文本文件中不重复字符的总数做编码，会得到同等数量的 token 供我们的 GPT 处理。
但我们并没有就此止步。
相反，我们接着建了一张查找表，把每个数值 token 映射到一个向量表示。

**为什么要多走这一步？**
用 $65$ 个向量（每个 token 一个）、每个为 $65$ 维（同样每个 token 一维）来表示词表，使每个 token 都获得了一个更深、固定分辨率的表示。
随后我们把这事交给 `BigramLM`，让它逐个 token 地优化该 token 向量表示里的值。

**总而言之，用可学习向量把 token 表示成统一大小，让 `BigramLM` 更好地捕捉字符之间的组合关系。**

**可以这样想：** 语义或功能上相近的字符，现在可以获得相近的向量表示。
因此，元音或辅音可能在向量空间里聚到一起，这让 `BigramLM` 更容易捕捉关联中的相似性。
向量的表现力远强于离散的、基于整数的 token 表示。

实现这一切的嵌入层 `BigramLM` 长这样：

```python
torch.manual_seed(1337) # 用于可复现性

# 此阶段还算不上是个语言模型，但我们终会走到那一步……
class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # 给词表做嵌入
        # vocab_size 个 token 中每一个都用一个大小为 vocab_size 的向量表示
        self.embed = nn.Embedding(vocab_size, vocab_size) # 65 个不重复的 65 维向量

    def forward(self, idx, targets):
        # idx 形状为 (batch_size, block_size)
        # targets 形状为 (batch_size, block_size)
        logits = self.embed(idx)
        return logits # 嵌入输入索引，形状变为 (batch_size, block_size, vocab_size) (B, T, C)
```

即便有上面描述的 $65 \times 65$ 嵌入矩阵，事实证明我们至多只是实现了一种低效的分词。
现实中，token 词表的构造方式要复杂得多，远不止字符级以及"字符→数值→向量"这种直接的表示映射。

> 文本其实并不是在字符级上分词的，而是在所谓的*块级*（chunk-level）上分词的。

**我们的目标是找到并探索一种在块级而非字符级上给文本分词的方法。**

## 分词的陷阱

让我们继续坚持 token 是信息基本单位这一基本观念。
目标是把文本字符串转换成对 LLM 而言最具信息量、最易解读的表示。

分词若做得不对，可能成为 LLM *大量*下游问题的根源。
一些与分词相关的最常见问题有：

- 为什么 LLM 不会拼写？
- 为什么 LLM 做不了像反转字符串这样简单的文本操作？
- 为什么 LLM 在非英语语言上可能表现更差？
- 为什么 LLM 不能正确做简单的算术运算？
- 为什么输入像 `<|endoftext|>` 这样的特殊字符会让生成停住？
- 为什么 LLM 会因"尾部空格"出毛病？
- 为什么 LLM 遇到单词里的大写字母会崩溃？
- 为什么用 YAML 而不是 JSON 来配合 LLM 工作可能更好？
- 为什么 'LLM' 其实并不等于 '端到端语言建模'？

> 一切罪恶的根源是什么？**分词！**

让我们到 [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app) 上，用 [GPT-2 分词器](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf) 做个实际例子：

```md
Tokenization is at the heart of much weirdness of LLMs. Do not brush it off.

127 + 677 = 804
1275 + 6773 = 8041

Egg.
I have an Egg.
egg.
EGG.

만나서 반가워요. 저는 OpenAI에서 개발한 대규모 언어 모델인 ChatGPT입니다. 궁금한 것이 있으시면 무엇이든 물어보세요.

for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

![tiktokenizer.vercel.app 上 GPT-2 分词器对示例文本的分词](./img/Tiktoken_Vercel_1.png)

每种颜色代表一个不同的 token。显然，GPT-2 的分词器并不是在字符级上运作的。

这在处理纯文本时影响可能不大，但看看算术运算的分词。
除 $127$ 外，每个数值都被切成了多个 token。
这种不合逻辑的 token 切分可能有问题，因为它可能导致难以保留任何数值或算术表达的语义。

奇怪的是，"Egg" 这个词按大小写和前导空格的不同，被分成了四种不同的 token。
我们还能看到，韩语文本的分词比英语文本更细碎。
这很大程度上归因于训练数据中的表示不平衡。分词器可能只是对英文文本优化得更好了。
正因如此，说韩语的终端用户用 LLM 完成同样的任务，可能要比说英语的用户付更高的费用，就因为更差/更细碎的分词更快填满了上下文窗口的 token 计数。
此外，用更多 token 表示同样多的数据，会导致下游注意力缓冲更快被填满。
这必然导致 LLM 整体性能下降。

在那段 Python 代码示例里，每个空白符都被单独分词。这降低了代码对 LLM 的可解读性，因为上下文窗口会被空白 token 迅速塞满。
和韩语文本一样，这会对模型性能产生负面影响。

把同样的输入给 GPT-4 分词器 `cl100k_base`，token 数基本上减半。
这说明该分词器的词表大约是 GPT-2 分词器的两倍大：

![tiktokenizer.vercel.app 上 GPT-4 分词器 cl100k_base 对相同样本的分词](./img/cl100k_base_1.png)

> 单凭词表增大这一项，就让 GPT-4 相比 GPT-2 几乎能把上下文窗口翻一倍。

## 分词器的思路

> 分词器是独立于 LLM 的一个实体。它可以单独在特定文本上调优或训练，而 LLM 则基于不同文本做优化。然而关键的是，虽然分词器不依赖 LLM，它却充当 LLM 与文本之间的接口。由此，LLM 得以在一个纯分词的世界里运作。

![分词器作为 LLM 与文本之间的接口](./img/Tokenizer_Schema.png)

我们现在想造一个超越字符级分词的分词器，用某个标识值来表示某些块。
由此出发，我们可以接着建一张查找表，把每个代表块的 token 映射到一个向量表示。这一步和之前一样。

哦，还有，别忘了，这次分词器还得能处理不同的文字系统。还有 Emoji，*我们需要 Emoji 支持！*

### Unicode

Python 开箱即用，就能处理不同的文字系统：

```python
some_text = "안녕하세요 👋 (hello in Korean!)"
print(some_text)
```

```
안녕하세요 👋 (hello in Korean!)
```

[文档](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)说 Python 的 `str` 类型是 **[Unicode](https://en.wikipedia.org/wiki/Unicode) 码点序列**。
Unicode 是一种字符编码标准，把每个字符映射到一个唯一的整数值。

我们可以用 `ord()` 函数读出这个值：

```python
print([ord(x) for x in some_text])
```

```
[50504, 45397, 54616, 49464, 50836, 32, 128075, 32, 40, 104, 101, 108, 108, 111, 32, 105, 110, 32, 75, 111, 114, 101, 97, 110, 33, 41]
```

那么到此为止，我们的编码问题算解决了……对吧？
**并非如此。**

Unicode 虽然全面，但它一直在变化、演进，而且已经相当庞大。**Unicode 并不稳定。**
但 Unicode 定义了三种*稳定*的编码形式，可用来表示 Unicode 字符的一个子集：[UTF-8](https://en.wikipedia.org/wiki/UTF-8)、[UTF-16](https://en.wikipedia.org/wiki/UTF-16) 和 [UTF-32](https://en.wikipedia.org/wiki/UTF-32)。
通过这些编码，字符用字节序列来表示。

> **为什么从上面的代码看，Unicode 用整数表示字符，而 UTF-8、UTF-16、UTF-32 用字节？**
> UTF-8、UTF-16、UTF-32 是编码方案，定义 Unicode 如何以二进制形式（字节序列）表示，用于存储和传输。
> 例如，UTF-8 专门设计来应对 Unicode 标准的演进特性，而自身保持稳定。
> Unicode 给字符赋整数值，而 UTF-8、UTF-16、UTF-32 是底层编码方案，描述这些码点如何转换成字节序列以供存储和传输。

用 UTF-8，你可以用 $1$ 到 $4$ 个字节的序列确定性地表示 Unicode 字符。**这是固定的。**
更多细节可参考 [Nathan Reed 的 Unicode 博文](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)，尤其是 [UTF-8 Everywhere 宣言](https://utf8everywhere.org/)。

下面这组原始字节就是我们的字符串按 UTF-8 编码后的表示：

```python
list(some_text.encode('utf-8'))
```

```
[236,
 149,
 136,
 235,
 133,
 149,
 237,
 149,
 152,
 236,
 132,
 184,
 236,
 154,
 148,
 32,
 240,
 159,
 145,
 139,
 32,
 40,
 104,
 101,
 108,
 108,
 111,
 32,
 105,
 110,
 32,
 75,
 111,
 114,
 101,
 97,
 110,
 33,
 41]
```

话虽如此，我们并不想把原始字节直接当 token。**但为什么不行？**

从上面可以看到，我们只区分 $256$ 种不同表示，因为我们用的是 $8$ 位表示/字节。
虽说我们确实可以用这种逐字节的方式表示字节数组，但这会是一种很低效的编码文本方式。
我们本质上是在放大文本、在很细的粒度上区分它的各个部分（字母编码的字节部分），从而不必要地拉长了 token 数。
而这反过来会塞满 LLM 的注意力缓冲，限制下游能力。

> 我们想支持一个大的词表，以 UTF-8 作为分词基础，但**我们不想直接用原始字节当 token，因为那太低效了**。

（应当指出，我们接下来做的任何事都会在 UTF-8 之上创造某种开销或抽象。在完美世界里，这应当避免。有趣的是，像 [\[Yu et al., 2023\]](https://arxiv.org/abs/2305.07185) 这样的工作正是想做这件事：避免分词开销，更接近直接用原始字节当 token。不过证明尚待时日。）

## 字节对编码

[维基百科上关于字节对编码（BPE）的文章](https://en.wikipedia.org/wiki/Byte_pair_encoding)很实用，强烈推荐。

> 对给定文本，BPE 迭代地把最常见的一对相邻 token 合并成一个单一新 token，然后把它加入我们的词表。我们重复这一过程，直到达到预设的词表大小。

可以看到这如何建立起 token 层级：最常见的一对先合并，然后是次常见的一对，依此类推。

**来看看实践中怎么做：**

```python
# 文本取自 https://www.reedbeta.com/blog/programmers-intro-to-unicode/ 的第一段
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = text.encode('utf-8')   # 字节数组
tokens = list(map(int, tokens)) # 转成整数列表以便可视化

print(text, "\n")
print("Original Text Length:", len(text), "\n\n")
print(tokens, "\n")
print("Length of Token List:", len(tokens)) # 简单字符映射到一个字节，但如 emoji 映射到 4 个字节
```

```
Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception. 

Original Text Length: 533 


[239, 188, 181, 239, 189, 142, 239, 189, 137, 239, 189, 131, 239, 189, 143, 239, 189, 132, 239, 189, 133, 33, 32, 240, 159, 133, 164, 240, 159, 133, 157, 240, 159, 133, 152, 240, 159, 133, 146, 240, 159, 133, 158, 240, 159, 133, 147, 240, 159, 133, 148, 226, 128, 189, 32, 240, 159, 135, 186, 226, 128, 140, 240, 159, 135, 179, 226, 128, 140, 240, 159, 135, 174, 226, 128, 140, 240, 159, 135, 168, 226, 128, 140, 240, 159, 135, 180, 226, 128, 140, 240, 159, 135, 169, 226, 128, 140, 240, 159, 135, 170, 33, 32, 240, 159, 152, 132, 32, 84, 104, 101, 32, 118, 101, 114, 121, 32, 110, 97, 109, 101, 32, 115, 116, 114, 105, 107, 101, 115, 32, 102, 101, 97, 114, 32, 97, 110, 100, 32, 97, 119, 101, 32, 105, 110, 116, 111, 32, 116, 104, 101, 32, 104, 101, 97, 114, 116, 115, 32, 111, 102, 32, 112, 114, 111, 103, 114, 97, 109, 109, 101, 114, 115, 32, 119, 111, 114, 108, 100, 119, 105, 100, 101, 46, 32, 87, 101, 32, 97, 108, 108, 32, 107, 110, 111, 119, 32, 119, 101, 32, 111, 117, 103, 104, 116, 32, 116, 111, 32, 226, 128, 156, 115, 117, 112, 112, 111, 114, 116, 32, 85, 110, 105, 99, 111, 100, 101, 226, 128, 157, 32, 105, 110, 32, 111, 117, 114, 32, 115, 111, 102, 116, 119, 97, 114, 101, 32, 40, 119, 104, 97, 116, 101, 118, 101, 114, 32, 116, 104, 97, 116, 32, 109, 101, 97, 110, 115, 226, 128, 148, 108, 105, 107, 101, 32, 117, 115, 105, 110, 103, 32, 119, 99, 104, 97, 114, 95, 116, 32, 102, 111, 114, 32, 97, 108, 108, 32, 116, 104, 101, 32, 115, 116, 114, 105, 110, 103, 115, 44, 32, 114, 105, 103, 104, 116, 63, 41, 46, 32, 66, 117, 116, 32, 85, 110, 105, 99, 111, 100, 101, 32, 99, 97, 110, 32, 98, 101, 32, 97, 98, 115, 116, 114, 117, 115, 101, 44, 32, 97, 110, 100, 32, 100, 105, 118, 105, 110, 103, 32, 105, 110, 116, 111, 32, 116, 104, 101, 32, 116, 104, 111, 117, 115, 97, 110, 100, 45, 112, 97, 103, 101, 32, 85, 110, 105, 99, 111, 100, 101, 32, 83, 116, 97, 110, 100, 97, 114, 100, 32, 112, 108, 117, 115, 32, 105, 116, 115, 32, 100, 111, 122, 101, 110, 115, 32, 111, 102, 32, 115, 117, 112, 112, 108, 101, 109, 101, 110, 116, 97, 114, 121, 32, 97, 110, 110, 101, 120, 101, 115, 44, 32, 114, 101, 112, 111, 114, 116, 115, 44, 32, 97, 110, 100, 32, 110, 111, 116, 101, 115, 32, 99, 97, 110, 32, 98, 101, 32, 109, 111, 114, 101, 32, 116, 104, 97, 110, 32, 97, 32, 108, 105, 116, 116, 108, 101, 32, 105, 110, 116, 105, 109, 105, 100, 97, 116, 105, 110, 103, 46, 32, 73, 32, 100, 111, 110, 226, 128, 153, 116, 32, 98, 108, 97, 109, 101, 32, 112, 114, 111, 103, 114, 97, 109, 109, 101, 114, 115, 32, 102, 111, 114, 32, 115, 116, 105, 108, 108, 32, 102, 105, 110, 100, 105, 110, 103, 32, 116, 104, 101, 32, 119, 104, 111, 108, 101, 32, 116, 104, 105, 110, 103, 32, 109, 121, 115, 116, 101, 114, 105, 111, 117, 115, 44, 32, 101, 118, 101, 110, 32, 51, 48, 32, 121, 101, 97, 114, 115, 32, 97, 102, 116, 101, 114, 32, 85, 110, 105, 99, 111, 100, 101, 226, 128, 153, 115, 32, 105, 110, 99, 101, 112, 116, 105, 111, 110, 46] 

Length of Token List: 616
```

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # 跨 token 的大小为 2 的滑动窗口
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(sorted(((v,k) for k,v in stats.items()), reverse=True)) # 反转键值关系并按键（计数）排序
print(sorted(((v,(chr(k[0]), chr(k[1]))) for k,v in stats.items()), reverse=True)[:5]) # 只为好玩，写出 5 个最常见的二元组

top_pair = max(stats, key=stats.get) # 取回最常见的二元组
print(top_pair)
```

```
[(20, (101, 32)), (15, (240, 159)), (12, (226, 128)), (12, (105, 110)), (10, (115, 32)), (10, (97, 110)), (10, (32, 97)), (9, (32, 116)), (8, (116, 104)), (7, (159, 135)), (7, (159, 133)), (7, (97, 114)), (6, (239, 189)), (6, (140, 240)), (6, (128, 140)), (6, (116, 32)), (6, (114, 32)), (6, (111, 114)), (6, (110, 103)), (6, (110, 100)), (6, (109, 101)), (6, (104, 101)), (6, (101, 114)), (6, (32, 105)), (5, (117, 115)), (5, (115, 116)), (5, (110, 32)), (5, (100, 101)), (5, (44, 32)), (5, (32, 115)), (4, (116, 105)), (4, (116, 101)), (4, (115, 44)), (4, (114, 105)), (4, (111, 117)), (4, (111, 100)), (4, (110, 116)), (4, (110, 105)), (4, (105, 99)), (4, (104, 97)), (4, (103, 32)), (4, (101, 97)), (4, (100, 32)), (4, (99, 111)), (4, (97, 109)), (4, (85, 110)), (4, (32, 119)), (4, (32, 111)), (4, (32, 102)), (4, (32, 85)), (3, (118, 101)), (3, (116, 115)), (3, (116, 114)), (3, (116, 111)), (3, (114, 116)), (3, (114, 115)), (3, (114, 101)), (3, (111, 102)), (3, (111, 32)), (3, (108, 108)), (3, (108, 101)), (3, (108, 32)), (3, (101, 115)), (3, (101, 110)), (3, (97, 116)), (3, (46, 32)), (3, (32, 240)), (3, (32, 112)), (3, (32, 109)), (3, (32, 100)), (3, (32, 98)), (2, (128, 153)), (2, (121, 32)), (2, (119, 104)), (2, (119, 101)), (2, (117, 112)), (2, (116, 97)), (2, (115, 117)), (2, (114, 121)), (2, (114, 111)), (2, (114, 97)), (2, (112, 114)), (2, (112, 112)), (2, (112, 111)), (2, (112, 108)), (2, (111, 110)), (2, (111, 103)), (2, (110, 115)), (2, (110, 111)), (2, (109, 109)), (2, (108, 105)), (2, (107, 101)), (2, (105, 116)), (2, (105, 111)), (2, (105, 107)), (2, (105, 100)), (2, (104, 116)), (2, (104, 111)), (2, (103, 114)), (2, (103, 104)), (2, (102, 116)), (2, (102, 111)), (2, (102, 32)), (2, (101, 226)), (2, (101, 118)), (2, (101, 112)), (2, (100, 111)), (2, (100, 105)), (2, (100, 97)), (2, (99, 97)), (2, (98, 101)), (2, (97, 108)), (2, (33, 32)), (2, (32, 114)), (2, (32, 110)), (2, (32, 99)), (1, (239, 188)), (1, (189, 143)), (1, (189, 142)), (1, (189, 137)), (1, (189, 133)), (1, (189, 132)), (1, (189, 131)), (1, (189, 32)), (1, (188, 181)), (1, (186, 226)), (1, (181, 239)), (1, (180, 226)), (1, (179, 226)), (1, (174, 226)), (1, (170, 33)), (1, (169, 226)), (1, (168, 226)), (1, (164, 240)), (1, (159, 152)), (1, (158, 240)), (1, (157, 240)), (1, (157, 32)), (1, (156, 115)), (1, (153, 116)), (1, (153, 115)), (1, (152, 240)), (1, (152, 132)), (1, (148, 226)), (1, (148, 108)), (1, (147, 240)), (1, (146, 240)), (1, (143, 239)), (1, (142, 239)), (1, (137, 239)), (1, (135, 186)), (1, (135, 180)), (1, (135, 179)), (1, (135, 174)), (1, (135, 170)), (1, (135, 169)), (1, (135, 168)), (1, (133, 164)), (1, (133, 158)), (1, (133, 157)), (1, (133, 152)), (1, (133, 148)), (1, (133, 147)), (1, (133, 146)), (1, (133, 33)), (1, (132, 239)), (1, (132, 32)), (1, (131, 239)), (1, (128, 189)), (1, (128, 157)), (1, (128, 156)), (1, (128, 148)), (1, (122, 101)), (1, (121, 115)), (1, (121, 101)), (1, (120, 101)), (1, (119, 111)), (1, (119, 105)), (1, (119, 99)), (1, (119, 97)), (1, (119, 32)), (1, (118, 105)), (1, (117, 116)), (1, (117, 114)), (1, (117, 103)), (1, (116, 119)), (1, (116, 116)), (1, (116, 108)), (1, (116, 63)), (1, (115, 226)), (1, (115, 111)), (1, (115, 105)), (1, (115, 101)), (1, (115, 97)), (1, (114, 117)), (1, (114, 108)), (1, (114, 100)), (1, (114, 95)), (1, (112, 116)), (1, (112, 97)), (1, (111, 122)), (1, (111, 119)), (1, (111, 116)), (1, (111, 108)), (1, (110, 226)), (1, (110, 110)), (1, (110, 101)), (1, (110, 99)), (1, (110, 97)), (1, (110, 46)), (1, (109, 121)), (1, (109, 111)), (1, (109, 105)), (1, (108, 117)), (1, (108, 100)), (1, (108, 97)), (1, (107, 110)), (1, (105, 118)), (1, (105, 109)), (1, (105, 108)), (1, (105, 103)), (1, (104, 105)), (1, (103, 115)), (1, (103, 101)), (1, (103, 46)), (1, (102, 105)), (1, (102, 101)), (1, (101, 120)), (1, (101, 109)), (1, (101, 46)), (1, (101, 44)), (1, (100, 119)), (1, (100, 45)), (1, (99, 104)), (1, (99, 101)), (1, (98, 115)), (1, (98, 108)), (1, (97, 119)), (1, (97, 103)), (1, (97, 102)), (1, (97, 98)), (1, (97, 32)), (1, (95, 116)), (1, (87, 101)), (1, (84, 104)), (1, (83, 116)), (1, (73, 32)), (1, (66, 117)), (1, (63, 41)), (1, (51, 48)), (1, (48, 32)), (1, (45, 112)), (1, (41, 46)), (1, (40, 119)), (1, (32, 226)), (1, (32, 121)), (1, (32, 118)), (1, (32, 117)), (1, (32, 108)), (1, (32, 107)), (1, (32, 104)), (1, (32, 101)), (1, (32, 87)), (1, (32, 84)), (1, (32, 83)), (1, (32, 73)), (1, (32, 66)), (1, (32, 51)), (1, (32, 40))]
[(20, ('e', ' ')), (15, ('ð', '\x9f')), (12, ('â', '\x80')), (12, ('i', 'n')), (10, ('s', ' '))]
(101, 32)
```

此刻 token 词表覆盖 $0$ 到 $255$，因为我们用的是 $8$ 位表示/字节。
现在我们可以创建一个新的、第 $256$ 个 token，来表示最常见的那一对 `('e', ' ')`：

```python
def merge(ids, pair, idx):
    # 遍历 ids，若找到 (pair)，则用值 idx 替换
    newids = []
    i = 0
    while i < len(ids):
        # 若并非处于最末位置 且 pair 匹配，则替换
        if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
            newids.append(idx)
            i += 2 # 跳过已替换的那一对
        else:
            newids.append(ids[i])
            i += 1
    return newids

# 合理性检查
print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))
```

```
[5, 6, 99, 9, 1]
```

```python
tokens2 = merge(tokens, top_pair, 256)

print(tokens2, "\n")
print("Length of Token List:", len(tokens2))
print("All Occurrences Removed" if (True if top_pair not in zip(tokens2, tokens2[1:]) else False) else "Still Some Occurrences Left")
```

```
[239, 188, 181, 239, 189, 142, 239, 189, 137, 239, 189, 131, 239, 189, 143, 239, 189, 132, 239, 189, 133, 33, 32, 240, 159, 133, 164, 240, 159, 133, 157, 240, 159, 133, 152, 240, 159, 133, 146, 240, 159, 133, 158, 240, 159, 133, 147, 240, 159, 133, 148, 226, 128, 189, 32, 240, 159, 135, 186, 226, 128, 140, 240, 159, 135, 179, 226, 128, 140, 240, 159, 135, 174, 226, 128, 140, 240, 159, 135, 168, 226, 128, 140, 240, 159, 135, 180, 226, 128, 140, 240, 159, 135, 169, 226, 128, 140, 240, 159, 135, 170, 33, 32, 240, 159, 152, 132, 32, 84, 104, 256, 118, 101, 114, 121, 32, 110, 97, 109, 256, 115, 116, 114, 105, 107, 101, 115, 32, 102, 101, 97, 114, 32, 97, 110, 100, 32, 97, 119, 256, 105, 110, 116, 111, 32, 116, 104, 256, 104, 101, 97, 114, 116, 115, 32, 111, 102, 32, 112, 114, 111, 103, 114, 97, 109, 109, 101, 114, 115, 32, 119, 111, 114, 108, 100, 119, 105, 100, 101, 46, 32, 87, 256, 97, 108, 108, 32, 107, 110, 111, 119, 32, 119, 256, 111, 117, 103, 104, 116, 32, 116, 111, 32, 226, 128, 156, 115, 117, 112, 112, 111, 114, 116, 32, 85, 110, 105, 99, 111, 100, 101, 226, 128, 157, 32, 105, 110, 32, 111, 117, 114, 32, 115, 111, 102, 116, 119, 97, 114, 256, 40, 119, 104, 97, 116, 101, 118, 101, 114, 32, 116, 104, 97, 116, 32, 109, 101, 97, 110, 115, 226, 128, 148, 108, 105, 107, 256, 117, 115, 105, 110, 103, 32, 119, 99, 104, 97, 114, 95, 116, 32, 102, 111, 114, 32, 97, 108, 108, 32, 116, 104, 256, 115, 116, 114, 105, 110, 103, 115, 44, 32, 114, 105, 103, 104, 116, 63, 41, 46, 32, 66, 117, 116, 32, 85, 110, 105, 99, 111, 100, 256, 99, 97, 110, 32, 98, 256, 97, 98, 115, 116, 114, 117, 115, 101, 44, 32, 97, 110, 100, 32, 100, 105, 118, 105, 110, 103, 32, 105, 110, 116, 111, 32, 116, 104, 256, 116, 104, 111, 117, 115, 97, 110, 100, 45, 112, 97, 103, 256, 85, 110, 105, 99, 111, 100, 256, 83, 116, 97, 110, 100, 97, 114, 100, 32, 112, 108, 117, 115, 32, 105, 116, 115, 32, 100, 111, 122, 101, 110, 115, 32, 111, 102, 32, 115, 117, 112, 112, 108, 101, 109, 101, 110, 116, 97, 114, 121, 32, 97, 110, 110, 101, 120, 101, 115, 44, 32, 114, 101, 112, 111, 114, 116, 115, 44, 32, 97, 110, 100, 32, 110, 111, 116, 101, 115, 32, 99, 97, 110, 32, 98, 256, 109, 111, 114, 256, 116, 104, 97, 110, 32, 97, 32, 108, 105, 116, 116, 108, 256, 105, 110, 116, 105, 109, 105, 100, 97, 116, 105, 110, 103, 46, 32, 73, 32, 100, 111, 110, 226, 128, 153, 116, 32, 98, 108, 97, 109, 256, 112, 114, 111, 103, 114, 97, 109, 109, 101, 114, 115, 32, 102, 111, 114, 32, 115, 116, 105, 108, 108, 32, 102, 105, 110, 100, 105, 110, 103, 32, 116, 104, 256, 119, 104, 111, 108, 256, 116, 104, 105, 110, 103, 32, 109, 121, 115, 116, 101, 114, 105, 111, 117, 115, 44, 32, 101, 118, 101, 110, 32, 51, 48, 32, 121, 101, 97, 114, 115, 32, 97, 102, 116, 101, 114, 32, 85, 110, 105, 99, 111, 100, 101, 226, 128, 153, 115, 32, 105, 110, 99, 101, 112, 116, 105, 111, 110, 46] 

Length of Token List: 596
All Occurrences Removed
```

我们实际上把文本映射到 UTF-8，找出了最常见的一对相邻字节值（$8$ 位取值范围 $0$ 到 $255$），并用一个代表该对的新 token（$256$）替换了其所有出现。**以上就是我们目前所做之事。**
不过我们还没有把自己的动作记录下来。

既然确认了基本实现可行，我们就可以随心所欲地对 token 循环很多遍，真正建起一个词表。
你可以把这扩充词表的过程，想成从叶子自底向根迭代地建一棵二叉树。

我们将用整篇博文来训练：

```python
text = """A Programmer’s Introduction to Unicode March 3, 2017 · Coding · 22 Comments  Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.  A few months ago, I got interested in Unicode and decided to spend some time learning more about it in detail. In this article, I’ll give an introduction to it from a programmer’s point of view.  I’m going to focus on the character set and what’s involved in working with strings and files of Unicode text. However, in this article I’m not going to talk about fonts, text layout/shaping/rendering, or localization in detail—those are separate issues, beyond my scope (and knowledge) here.  Diversity and Inherent Complexity The Unicode Codespace Codespace Allocation Scripts Usage Frequency Encodings UTF-8 UTF-16 Combining Marks Canonical Equivalence Normalization Forms Grapheme Clusters And More… Diversity and Inherent Complexity As soon as you start to study Unicode, it becomes clear that it represents a large jump in complexity over character sets like ASCII that you may be more familiar with. It’s not just that Unicode contains a much larger number of characters, although that’s part of it. Unicode also has a great deal of internal structure, features, and special cases, making it much more than what one might expect a mere “character set” to be. We’ll see some of that later in this article.  When confronting all this complexity, especially as an engineer, it’s hard not to find oneself asking, “Why do we need all this? Is this really necessary? Couldn’t it be simplified?”  However, Unicode aims to faithfully represent the entire world’s writing systems. The Unicode Consortium’s stated goal is “enabling people around the world to use computers in any language”. And as you might imagine, the diversity of written languages is immense! To date, Unicode supports 135 different scripts, covering some 1100 languages, and there’s still a long tail of over 100 unsupported scripts, both modern and historical, which people are still working to add.  Given this enormous diversity, it’s inevitable that representing it is a complicated project. Unicode embraces that diversity, and accepts the complexity inherent in its mission to include all human writing systems. It doesn’t make a lot of trade-offs in the name of simplification, and it makes exceptions to its own rules where necessary to further its mission.  Moreover, Unicode is committed not just to supporting texts in any single language, but also to letting multiple languages coexist within one text—which introduces even more complexity.  Most programming languages have libraries available to handle the gory low-level details of text manipulation, but as a programmer, you’ll still need to know about certain Unicode features in order to know when and how to apply them. It may take some time to wrap your head around it all, but don’t be discouraged—think about the billions of people for whom your software will be more accessible through supporting text in their language. Embrace the complexity!  The Unicode Codespace Let’s start with some general orientation. The basic elements of Unicode—its “characters”, although that term isn’t quite right—are called code points. Code points are identified by number, customarily written in hexadecimal with the prefix “U+”, such as U+0041 “A” latin capital letter a or U+03B8 “θ” greek small letter theta. Each code point also has a short name, and quite a few other properties, specified in the Unicode Character Database.  The set of all possible code points is called the codespace. The Unicode codespace consists of 1,114,112 code points. However, only 128,237 of them—about 12% of the codespace—are actually assigned, to date. There’s plenty of room for growth! Unicode also reserves an additional 137,468 code points as “private use” areas, which have no standardized meaning and are available for individual applications to define for their own purposes.  Codespace Allocation To get a feel for how the codespace is laid out, it’s helpful to visualize it. Below is a map of the entire codespace, with one pixel per code point. It’s arranged in tiles for visual coherence; each small square is 16×16 = 256 code points, and each large square is a “plane” of 65,536 code points. There are 17 planes altogether.  Map of the Unicode codespace (click to zoom)  White represents unassigned space. Blue is assigned code points, green is private-use areas, and the small red area is surrogates (more about those later). As you can see, the assigned code points are distributed somewhat sparsely, but concentrated in the first three planes.  Plane 0 is also known as the “Basic Multilingual Plane”, or BMP. The BMP contains essentially all the characters needed for modern text in any script, including Latin, Cyrillic, Greek, Han (Chinese), Japanese, Korean, Arabic, Hebrew, Devanagari (Indian), and many more.  (In the past, the codespace was just the BMP and no more—Unicode was originally conceived as a straightforward 16-bit encoding, with only 65,536 code points. It was expanded to its current size in 1996. However, the vast majority of code points in modern text belong to the BMP.)  Plane 1 contains historical scripts, such as Sumerian cuneiform and Egyptian hieroglyphs, as well as emoji and various other symbols. Plane 2 contains a large block of less-common and historical Han characters. The remaining planes are empty, except for a small number of rarely-used formatting characters in Plane 14; planes 15–16 are reserved entirely for private use.  Scripts Let’s zoom in on the first three planes, since that’s where the action is:  Map of scripts in Unicode planes 0–2 (click to zoom)  This map color-codes the 135 different scripts in Unicode. You can see how Han () and Korean () take up most of the range of the BMP (the left large square). By contrast, all of the European, Middle Eastern, and South Asian scripts fit into the first row of the BMP in this diagram.  Many areas of the codespace are adapted or copied from earlier encodings. For example, the first 128 code points of Unicode are just a copy of ASCII. This has clear benefits for compatibility—it’s easy to losslessly convert texts from smaller encodings into Unicode (and the other direction too, as long as no characters outside the smaller encoding are used).  Usage Frequency One more interesting way to visualize the codespace is to look at the distribution of usage—in other words, how often each code point is actually used in real-world texts. Below is a heat map of planes 0–2 based on a large sample of text from Wikipedia and Twitter (all languages). Frequency increases from black (never seen) through red and yellow to white.  Heat map of code point usage frequency in Unicode planes 0–2 (click to zoom)  You can see that the vast majority of this text sample lies in the BMP, with only scattered usage of code points from planes 1–2. The biggest exception is emoji, which show up here as the several bright squares in the bottom row of plane 1.  Encodings We’ve seen that Unicode code points are abstractly identified by their index in the codespace, ranging from U+0000 to U+10FFFF. But how do code points get represented as bytes, in memory or in a file?  The most convenient, computer-friendliest (and programmer-friendliest) thing to do would be to just store the code point index as a 32-bit integer. This works, but it consumes 4 bytes per code point, which is sort of a lot. Using 32-bit ints for Unicode will cost you a bunch of extra storage, memory, and performance in bandwidth-bound scenarios, if you work with a lot of text.  Consequently, there are several more-compact encodings for Unicode. The 32-bit integer encoding is officially called UTF-32 (UTF = “Unicode Transformation Format”), but it’s rarely used for storage. At most, it comes up sometimes as a temporary internal representation, for examining or operating on the code points in a string.  Much more commonly, you’ll see Unicode text encoded as either UTF-8 or UTF-16. These are both variable-length encodings, made up of 8-bit or 16-bit units, respectively. In these schemes, code points with smaller index values take up fewer bytes, which saves a lot of memory for typical texts. The trade-off is that processing UTF-8/16 texts is more programmatically involved, and likely slower.  UTF-8 In UTF-8, each code point is stored using 1 to 4 bytes, based on its index value.  UTF-8 uses a system of binary prefixes, in which the high bits of each byte mark whether it’s a single byte, the beginning of a multi-byte sequence, or a continuation byte; the remaining bits, concatenated, give the code point index. This table shows how it works:  UTF-8 (binary)\tCode point (binary)\tRange 0xxxxxxx\txxxxxxx\tU+0000–U+007F 110xxxxx 10yyyyyy\txxxxxyyyyyy\tU+0080–U+07FF 1110xxxx 10yyyyyy 10zzzzzz\txxxxyyyyyyzzzzzz\tU+0800–U+FFFF 11110xxx 10yyyyyy 10zzzzzz 10wwwwww\txxxyyyyyyzzzzzzwwwwww\tU+10000–U+10FFFF A handy property of UTF-8 is that code points below 128 (ASCII characters) are encoded as single bytes, and all non-ASCII code points are encoded using sequences of bytes 128–255. This has a couple of nice consequences. First, any strings or files out there that are already in ASCII can also be interpreted as UTF-8 without any conversion. Second, lots of widely-used string programming idioms—such as null termination, or delimiters (newlines, tabs, commas, slashes, etc.)—will just work on UTF-8 strings. ASCII bytes never occur inside the encoding of non-ASCII code points, so searching byte-wise for a null terminator or a delimiter will do the right thing.  Thanks to this convenience, it’s relatively simple to extend legacy ASCII programs and APIs to handle UTF-8 strings. UTF-8 is very widely used in the Unix/Linux and Web worlds, and many programmers argue UTF-8 should be the default encoding everywhere.  However, UTF-8 isn’t a drop-in replacement for ASCII strings in all respects. For instance, code that iterates over the “characters” in a string will need to decode UTF-8 and iterate over code points (or maybe grapheme clusters—more about those later), not bytes. When you measure the “length” of a string, you’ll need to think about whether you want the length in bytes, the length in code points, the width of the text when rendered, or something else.  UTF-16 The other encoding that you’re likely to encounter is UTF-16. It uses 16-bit words, with each code point stored as either 1 or 2 words.  Like UTF-8, we can express the UTF-16 encoding rules in the form of binary prefixes:  UTF-16 (binary)\tCode point (binary)\tRange xxxxxxxxxxxxxxxx\txxxxxxxxxxxxxxxx\tU+0000–U+FFFF 110110xxxxxxxxxx 110111yyyyyyyyyy\txxxxxxxxxxyyyyyyyyyy + 0x10000\tU+10000–U+10FFFF A more common way that people talk about UTF-16 encoding, though, is in terms of code points called “surrogates”. All the code points in the range U+D800–U+DFFF—or in other words, the code points that match the binary prefixes 110110 and 110111 in the table above—are reserved specifically for UTF-16 encoding, and don’t represent any valid characters on their own. They’re only meant to occur in the 2-word encoding pattern above, which is called a “surrogate pair”. Surrogate code points are illegal in any other context! They’re not allowed in UTF-8 or UTF-32 at all.  Historically, UTF-16 is a descendant of the original, pre-1996 versions of Unicode, in which there were only 65,536 code points. The original intention was that there would be no different “encodings”; Unicode was supposed to be a straightforward 16-bit character set. Later, the codespace was expanded to make room for a long tail of less-common (but still important) Han characters, which the Unicode designers didn’t originally plan for. Surrogates were then introduced, as—to put it bluntly—a kludge, allowing 16-bit encodings to access the new code points.  Today, Javascript uses UTF-16 as its standard string representation: if you ask for the length of a string, or iterate over it, etc., the result will be in UTF-16 words, with any code points outside the BMP expressed as surrogate pairs. UTF-16 is also used by the Microsoft Win32 APIs; though Win32 supports either 8-bit or 16-bit strings, the 8-bit version unaccountably still doesn’t support UTF-8—only legacy code-page encodings, like ANSI. This leaves UTF-16 as the only way to get proper Unicode support in Windows. (Update: in Win10 version 1903, they finally added UTF-8 support to the 8-bit APIs! 😊)  By the way, UTF-16’s words can be stored either little-endian or big-endian. Unicode has no opinion on that issue, though it does encourage the convention of putting U+FEFF zero width no-break space at the top of a UTF-16 file as a byte-order mark, to disambiguate the endianness. (If the file doesn’t match the system’s endianness, the BOM will be decoded as U+FFFE, which isn’t a valid code point.)  Combining Marks In the story so far, we’ve been focusing on code points. But in Unicode, a “character” can be more complicated than just an individual code point!  Unicode includes a system for dynamically composing characters, by combining multiple code points together. This is used in various ways to gain flexibility without causing a huge combinatorial explosion in the number of code points.  In European languages, for example, this shows up in the application of diacritics to letters. Unicode supports a wide range of diacritics, including acute and grave accents, umlauts, cedillas, and many more. All these diacritics can be applied to any letter of any alphabet—and in fact, multiple diacritics can be used on a single letter.  If Unicode tried to assign a distinct code point to every possible combination of letter and diacritics, things would rapidly get out of hand. Instead, the dynamic composition system enables you to construct the character you want, by starting with a base code point (the letter) and appending additional code points, called “combining marks”, to specify the diacritics. When a text renderer sees a sequence like this in a string, it automatically stacks the diacritics over or under the base letter to create a composed character.  For example, the accented character “Á” can be expressed as a string of two code points: U+0041 “A” latin capital letter a plus U+0301 “◌́” combining acute accent. This string automatically gets rendered as a single character: “Á”.  Now, Unicode does also include many “precomposed” code points, each representing a letter with some combination of diacritics already applied, such as U+00C1 “Á” latin capital letter a with acute or U+1EC7 “ệ” latin small letter e with circumflex and dot below. I suspect these are mostly inherited from older encodings that were assimilated into Unicode, and kept around for compatibility. In practice, there are precomposed code points for most of the common letter-with-diacritic combinations in European-script languages, so they don’t use dynamic composition that much in typical text.  Still, the system of combining marks does allow for an arbitrary number of diacritics to be stacked on any base character. The reductio-ad-absurdum of this is Zalgo text, which works by ͖͟ͅr͞aṋ̫̠̖͈̗d͖̻̹óm̪͙͕̗̝ļ͇̰͓̳̫ý͓̥̟͍ ̕s̫t̫̱͕̗̰̼̘͜a̼̩͖͇̠͈̣͝c̙͍k̖̱̹͍͘i̢n̨̺̝͇͇̟͙ģ̫̮͎̻̟ͅ ̕n̼̺͈͞u̮͙m̺̭̟̗͞e̞͓̰̤͓̫r̵o̖ṷs҉̪͍̭̬̝̤ ̮͉̝̞̗̟͠d̴̟̜̱͕͚i͇̫̼̯̭̜͡ḁ͙̻̼c̲̲̹r̨̠̹̣̰̦i̱t̤̻̤͍͙̘̕i̵̜̭̤̱͎c̵s ͘o̱̲͈̙͖͇̲͢n͘ ̜͈e̬̲̠̩ac͕̺̠͉h̷̪ ̺̣͖̱ḻ̫̬̝̹ḙ̙̺͙̭͓̲t̞̞͇̲͉͍t̷͔̪͉̲̻̠͙e̦̻͈͉͇r͇̭̭̬͖,̖́ ̜͙͓̣̭s̘̘͈o̱̰̤̲ͅ ̛̬̜̙t̼̦͕̱̹͕̥h̳̲͈͝ͅa̦t̻̲ ̻̟̭̦̖t̛̰̩h̠͕̳̝̫͕e͈̤̘͖̞͘y҉̝͙ ̷͉͔̰̠o̞̰v͈͈̳̘͜er̶f̰͈͔ḻ͕̘̫̺̲o̲̭͙͠ͅw̱̳̺ ͜t̸h͇̭͕̳͍e̖̯̟̠ ͍̞̜͔̩̪͜ļ͎̪̲͚i̝̲̹̙̩̹n̨̦̩̖ḙ̼̲̼͢ͅ ̬͝s̼͚̘̞͝p͙̘̻a̙c҉͉̜̤͈̯̖i̥͡n̦̠̱͟g̸̗̻̦̭̮̟ͅ ̳̪̠͖̳̯̕a̫͜n͝d͡ ̣̦̙ͅc̪̗r̴͙̮̦̹̳e͇͚̞͔̹̫͟a̙̺̙ț͔͎̘̹ͅe̥̩͍ a͖̪̜̮͙̹n̢͉̝ ͇͉͓̦̼́a̳͖̪̤̱p̖͔͔̟͇͎͠p̱͍̺ę̲͎͈̰̲̤̫a̯͜r̨̮̫̣̘a̩̯͖n̹̦̰͎̣̞̞c̨̦̱͔͎͍͖e̬͓͘ ̤̰̩͙̤̬͙o̵̼̻̬̻͇̮̪f̴ ̡̙̭͓͖̪̤“̸͙̠̼c̳̗͜o͏̼͙͔̮r̞̫̺̞̥̬ru̺̻̯͉̭̻̯p̰̥͓̣̫̙̤͢t̳͍̳̖ͅi̶͈̝͙̼̙̹o̡͔n̙̺̹̖̩͝ͅ”̨̗͖͚̩.̯͓  A few other places where dynamic character composition shows up in Unicode:  Vowel-pointing notation in Arabic and Hebrew. In these languages, words are normally spelled with some of their vowels left out. They then have diacritic notation to indicate the vowels (used in dictionaries, language-teaching materials, children’s books, and such). These diacritics are expressed with combining marks.  A Hebrew example, with niqqud:\tאֶת דַלְתִּי הֵזִיז הֵנִיעַ, קֶטֶב לִשְׁכַּתִּי יָשׁוֹד Normal writing (no niqqud):\tאת דלתי הזיז הניע, קטב לשכתי ישוד Devanagari, the script used to write Hindi, Sanskrit, and many other South Asian languages, expresses certain vowels as combining marks attached to consonant letters. For example, “ह” + “​ि” = “हि” (“h” + “i” = “hi”). Korean characters stand for syllables, but they are composed of letters called jamo that stand for the vowels and consonants in the syllable. While there are code points for precomposed Korean syllables, it’s also possible to dynamically compose them by concatenating their jamo. For example, “ᄒ” + “ᅡ” + “ᆫ” = “한” (“h” + “a” + “n” = “han”). Canonical Equivalence In Unicode, precomposed characters exist alongside the dynamic composition system. A consequence of this is that there are multiple ways to express “the same” string—different sequences of code points that result in the same user-perceived characters. For example, as we saw earlier, we can express the character “Á” either as the single code point U+00C1, or as the string of two code points U+0041 U+0301.  Another source of ambiguity is the ordering of multiple diacritics in a single character. Diacritic order matters visually when two diacritics apply to the same side of the base character, e.g. both above: “ǡ” (dot, then macron) is different from “ā̇” (macron, then dot). However, when diacritics apply to different sides of the character, e.g. one above and one below, then the order doesn’t affect rendering. Moreover, a character with multiple diacritics might have one of the diacritics precomposed and others expressed as combining marks.  For example, the Vietnamese letter “ệ” can be expressed in five different ways:  Fully precomposed: U+1EC7 “ệ” Partially precomposed: U+1EB9 “ẹ” + U+0302 “◌̂” Partially precomposed: U+00EA “ê” + U+0323 “◌̣” Fully decomposed: U+0065 “e” + U+0323 “◌̣” + U+0302 “◌̂” Fully decomposed: U+0065 “e” + U+0302 “◌̂” + U+0323 “◌̣” Unicode refers to set of strings like this as “canonically equivalent”. Canonically equivalent strings are supposed to be treated as identical for purposes of searching, sorting, rendering, text selection, and so on. This has implications for how you implement operations on text. For example, if an app has a “find in file” operation and the user searches for “ệ”, it should, by default, find occurrences of any of the five versions of “ệ” above!  Normalization Forms To address the problem of “how to handle canonically equivalent strings”, Unicode defines several normalization forms: ways of converting strings into a canonical form so that they can be compared code-point-by-code-point (or byte-by-byte).  The “NFD” normalization form fully decomposes every character down to its component base and combining marks, taking apart any precomposed code points in the string. It also sorts the combining marks in each character according to their rendered position, so e.g. diacritics that go below the character come before the ones that go above the character. (It doesn’t reorder diacritics in the same rendered position, since their order matters visually, as previously mentioned.)  The “NFC” form, conversely, puts things back together into precomposed code points as much as possible. If an unusual combination of diacritics is called for, there may not be any precomposed code point for it, in which case NFC still precomposes what it can and leaves any remaining combining marks in place (again ordered by rendered position, as in NFD).  There are also forms called NFKD and NFKC. The “K” here refers to compatibility decompositions, which cover characters that are “similar” in some sense but not visually identical. However, I’m not going to cover that here.  Grapheme Clusters As we’ve seen, Unicode contains various cases where a thing that a user thinks of as a single “character” might actually be made up of multiple code points under the hood. Unicode formalizes this using the notion of a grapheme cluster: a string of one or more code points that constitute a single “user-perceived character”.  UAX #29 defines the rules for what, precisely, qualifies as a grapheme cluster. It’s approximately “a base code point followed by any number of combining marks”, but the actual definition is a bit more complicated; it accounts for things like Korean jamo, and emoji ZWJ sequences.  The main thing grapheme clusters are used for is text editing: they’re often the most sensible unit for cursor placement and text selection boundaries. Using grapheme clusters for these purposes ensures that you can’t accidentally chop off some diacritics when you copy-and-paste text, that left/right arrow keys always move the cursor by one visible character, and so on.  Another place where grapheme clusters are useful is in enforcing a string length limit—say, on a database field. While the true, underlying limit might be something like the byte length of the string in UTF-8, you wouldn’t want to enforce that by just truncating bytes. At a minimum, you’d want to “round down” to the nearest code point boundary; but even better, round down to the nearest grapheme cluster boundary. Otherwise, you might be corrupting the last character by cutting off a diacritic, or interrupting a jamo sequence or ZWJ sequence.  And More… There’s much more that could be said about Unicode from a programmer’s perspective! I haven’t gotten into such fun topics as case mapping, collation, compatibility decompositions and confusables, Unicode-aware regexes, or bidirectional text. Nor have I said anything yet about implementation issues—how to efficiently store and look-up data about the sparsely-assigned code points, or how to optimize UTF-8 decoding, string comparison, or NFC normalization. Perhaps I’ll return to some of those things in future posts.  Unicode is a fascinating and complex system. It has a many-to-one mapping between bytes and code points, and on top of that a many-to-one (or, under some circumstances, many-to-many) mapping between code points and “characters”. It has oddball special cases in every corner. But no one ever claimed that representing all written languages was going to be easy, and it’s clear that we’re never going back to the bad old days of a patchwork of incompatible encodings.  Further reading:  The Unicode Standard UTF-8 Everywhere Manifesto Dark corners of Unicode by Eevee ICU (International Components for Unicode)—C/C++/Java libraries implementing many Unicode algorithms and related things Python 3 Unicode Howto Google Noto Fonts—set of fonts intended to cover all assigned code points"""
tokens = text.encode("utf-8") # 原始字节
tokens = list(map(int, tokens)) # 转成 0..255 范围的整数列表，便于处理
```

现在我们可以把词表存成一个简单的字典 `merges`，以 pair 为键、以它所代表的 token 为值：

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # 跨 token 的大小为 2 的滑动窗口
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    # 遍历 ids，若找到 (pair)，则用值 idx 替换
    newids = []
    i = 0
    while i < len(ids):
        # 若并非处于最末位置 且 pair 匹配，则替换
        if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
            newids.append(idx)
            i += 2 # 跳过已替换的那一对
        else:
            newids.append(ids[i])
            i += 1
    return newids


vocab_size = 276    # 我们（任意选定的）目标词表大小
num_merges = vocab_size - 256
ids = list(tokens)  # 从这里起我们就在 tokens 的副本上工作（list() 对列表做深拷贝）
merges = {}         # (int, int) -> int，合并字典；可看作 key: (child1, child2)，value: parent/新 token


for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get) # 取回最常见的二元组
    idx = 256 + i
    print(f"Merging {pair}\tinto new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```

```
Merging (101, 32)	into new token 256
Merging (105, 110)	into new token 257
Merging (115, 32)	into new token 258
Merging (116, 104)	into new token 259
Merging (101, 114)	into new token 260
Merging (99, 111)	into new token 261
Merging (116, 32)	into new token 262
Merging (226, 128)	into new token 263
Merging (44, 32)	into new token 264
Merging (97, 110)	into new token 265
Merging (111, 114)	into new token 266
Merging (100, 32)	into new token 267
Merging (97, 114)	into new token 268
Merging (101, 110)	into new token 269
Merging (257, 103)	into new token 270
Merging (261, 100)	into new token 271
Merging (121, 32)	into new token 272
Merging (46, 32)	into new token 273
Merging (97, 108)	into new token 274
Merging (259, 256)	into new token 275
```

把扩充后的词表和原来的比一下，可以看到：同样多的数据现在需要的 token 更少了，因为我们能用单个 token 表示更复杂的文本块：

```python
print("Old Length of Token List:", len(tokens))
print("New Length of Token List:", len(ids))
print(f"Compression Ratio: {len(tokens) / len(ids):.2f}x")
```

```
Old Length of Token List: 24597
New Length of Token List: 19438
Compression Ratio: 1.27x
```

只需 $20$ 次合并与新 token 创建，就实现了 $1.27$ 倍的压缩。

多亏了 `merges`，编码和解码我们都能做。分词器成了 LLM 与可读文本之间的翻译层。

但话说回来，token 深度并非唯一值得优化的参数。我们从中推导出字节对的那个数据集，应当能很好地代表你想分词的全部文本范围（至少对你预期的用例而言）。而我们在 GPT-2 与 GPT-4 分词器对比中已经见过这种训练数据选择的影响了。

### 解码

编码做好了，给定一串范围在 $[0;\ \text{vocab\_size}]$ 内的整数 token，对应的文本字符串是什么？

```python
# 解码器预处理：从 token-id 映射到该 token 对应的 bytes 对象
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():   # 需按我们往 merges 插入的顺序运行（用 Python >= 3.7）
    vocab[idx] = vocab[p0] + vocab[p1] # 在 idx（父整数）处填入子 p0、p1 的 bytes 对象的拼接，格式为 {idx: b'p0p1'}

def decode(ids):
    # 给定 ids（整数列表），返回 Python 字符串
    tokens = b"".join(vocab[idx] for idx in ids) # 把每个新 token-id 对应的 bytes 对象拼接起来
    text = tokens.decode("utf-8")                # 把字节解码成 Python 字符串
    return text
```

**好，我们刚才干了啥？**

字典 `vocab` 第一步被设成：以 $0$ 到 $255$ 这些整数为键，以各自对应的 bytes 对象为值。
在这个 `int -> byte(int)` 的"基础映射"之外，我们现在还要把 `new_token` 存为键、`(token1, token2)` 存为值。
你可以把它想成把 `merges` 里 `(token1, token2) -> new_token` 的键值关系反过来，在 `vocab` 里变成 `new_token -> (token1, token2)`。
但 `vocab` 不把 `(token1, token2)` 作为元组存，而是存两个 bytes 对象的拼接，即 `vocab[p0] + vocab[p1]` 或 `byte(token1) + byte(token2)`。

> `vocab` 现在持有**每个** `token`（不只是 `new_token`，而是所有 token）整数到其映射：要么是单个字节，要么是合并创建 `new_token` 时那两个 bytes 对象的拼接。

`decode` 函数随后接收一个 token 整数列表，逐一用各 token 整数去索引 `vocab`，一步步把 bytes 对象拼起来形成字符串。
最后，用 UTF-8 解码把 bytes 对象转成更可读的形式，理想情况下接近原文。

不过你可能已经注意到，这还留着一个实现问题：

```python
print(decode([97]))  # 结果是 "a"
print(decode([128])) # 结果是 ……有缺陷？！
```

```
a
```

```
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
... in decode(ids)
      8     tokens = b"".join(vocab[idx] for idx in ids)
----> 9     text = tokens.decode("utf-8")
     10     return text

UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

把 $128_{10}$ 映射成二进制，得到 $1000\ 0000_{2}$，根据 [维基百科](https://en.wikipedia.org/wiki/UTF-8#Encoding)，这不是一个合法的 UTF-8 起始字节。
我们给的是 $1000\ 0000_{2}$，但只有 $0\text{XXX\ XXXX}_{2}$ 或 $110\text{X\ XXXX}_{2}$ 才能作为合法 UTF-8 字节表示的起始字节。

**好在，这很容易修：**

```python
# 解码器预处理：从 token-id 映射到该 token 对应的 bytes 对象
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():   # 需按我们往 merges 插入的顺序运行（用 Python >= 3.7）
    vocab[idx] = vocab[p0] + vocab[p1] # 在 idx（父整数）处填入子 p0、p1 的 bytes 对象的拼接

def decode(ids):
    # 给定 ids（整数列表），返回 Python 字符串
    tokens = b"".join(vocab[idx] for idx in ids)     # 把每个新 token-id 对应的 bytes 对象拼接起来
    text = tokens.decode("utf-8", errors="replace")  # 把字节解码成 Python 字符串
    return text
```

```python
print(decode([97]))  # 仍然可行
print(decode([128])) # 现在可行了
```

```
a
�
```

**为什么 `errors=replace` 有效？**

见 [Python 文档](https://docs.python.org/3/library/codecs.html#error-handlers)：若解码器遇到无法解码成合法 Unicode 的字节序列，该序列会被替换成特殊的 Unicode 替换字符（U+FFFD）。我们基本上是优雅地跨过这个非法字节序列，喊一声"我不知道这是啥"，然后继续。

**如果我们现在只是跨过这个问题，那它一开始又怎么算个问题呢？我们不是只往编码器里输入了 UTF-8 吗？怎么还会返回别的东西？**

是的，我们往编码器里输入了 UTF-8。但现实中，我们并不解码刚刚分词的东西。解码的是 LLM 的输出。
而在那里，如果你遇到值为 $128$ 的 LLM 输出符号，说明 LLM 那边出了问题，值得进一步训练来避免。

**感觉这个 `decode` 实现太浅了。**
**这个设置如何处理 `merges` 里有些元组本身由 `new_token` 组成的情况？**
**那不会报错吗？**

这个实现只是乍看之下很浅。其实并不浅。我们严格按 `new_token` 加入 `merges` 的顺序逐步构建 `vocab`。这样，`new_token` 所表示的元组就保证已经存在于 `vocab` 中、随时可被引用。而如果你引用一个已在 `vocab` 中的 `new_token`，你会得到合并创建 `new_token` 时那两个 bytes 对象的拼接。bytes 对象在 `vocab` 里是递归建起来的。所以，若一个 `new_token` 被发现表示一个由 `new_token` 组成的元组，通过递归索引 `vocab`，自动就能找到最底层那些 bytes 对象。这让围绕 `vocab[idx] = vocab[p0] + vocab[p1]` 的循环成了一种很优雅的方案，用以找到代表每个 `new_token` 的真实 bytes 对象。

### 编码

给定一个文本字符串，对应的 $0$ 到 $\text{vocab\_size} - 1$ 整数 token 序列是什么？

```python
def encode(text):
    tokens = list(text.encode("utf-8")) # 原始字节，格式化为整数列表
    # 现在，查 merges（按从顶到底的历史顺序），递归地把 pair 替换成新 token-id
    while True:
        stats = get_stats(tokens) # 统计每对出现多少次，格式 (int, int) -> int（出现计数）
        pair = min(stats, key=lambda p: merges.get(p, float('inf'))) # 此处遍历 stats 的键；取回合并索引最低的那对
        if pair not in merges: # 可能因为 merges 不含 tokens 中出现的任何一对
            break # 可以停了，没有更多可合并的
        idx = merges[pair] # 取回可合并 pair 的新 token 表示
        tokens = merge(tokens, pair, idx) # 每个该 pair 都被替换成 idx，与我们之前做的一样
    return tokens
```

在 `get_stats` 里，奇怪的是我们并不真正在乎出现计数。更确切地说，`get_stats` 里的键构成了一张"值得考虑合并的 token 对"的列表。
从 `merges` 的视角，我们想先把在 `merges` 中列得最早的那些合并做掉。
然后，对于 `stats` 里的每个 `pair`，我们查 `merges` 字典。更具体地说，我们看 `pair` 会创建的那个 `new_token`。我们这样做的目的是找到值最小的那个 `new_token`。

例如，假设 `merges` 的第一个条目是 `(1, 2) -> 256`。
如果这个组合出现在 `tokens` 中、且被 `get_stats` 认为值得分词，我们就伸手进 `merges`。
在那里我们寻找能赋给某对的最小可能值（此处是 `(1, 2)` 的 `256`），把它作为 `pair` 的值取回。
正因如此，我们判定 `(1, 2)` 既确实出现在待编码文本中、又最该被分词，因为它的 `new_token` 最小。

实现好了，来测一下：

```python
print(encode("hello world!"))
print(encode("h"))
```

```
[104, 101, 108, 108, 111, 32, 119, 266, 108, 100, 33]
```

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
... in encode(text)
      5     stats = get_stats(tokens)
----> 6     pair = min(stats, key=lambda p: merges.get(p, float('inf')))
      7     if pair not in merges:

ValueError: min() arg is an empty sequence
```

只有一个字符或空字符串时，我们没法运行 `stats = get_stats(tokens)`，因为找不到任何对。

*来修一下：*

```python
def encode(text):
    tokens = list(text.encode("utf-8")) # 原始字节，格式化为整数列表
    # 现在，查 merges（按从顶到底的历史顺序），递归地把 pair 替换成新 token-id
    while len(tokens) > 1:
        stats = get_stats(tokens) # 统计每对出现多少次，格式 (int, int) -> int（出现计数）
        pair = min(stats, key=lambda p: merges.get(p, float('inf'))) # 此处遍历 stats 的键；取回合并索引最低的那对
        if pair not in merges: # 可能因为 merges 不含 tokens 中出现的任何一对
            break # 可以停了，没有更多可合并的
        idx = merges[pair] # 取回可合并 pair 的新 token 表示
        tokens = merge(tokens, pair, idx) # 每个该 pair 都被替换成 idx，与我们之前做的一样
    return tokens

print(encode("hello world!"))
print(encode("h"))
print(encode(""))
```

```
[104, 101, 108, 108, 111, 32, 119, 266, 108, 100, 33]
[104]
[]
```

最后，来解码一些编码后的内容：

```python
print(decode(encode("hello world!"))) # 可行

# 这段文本我们已经认识，分词器是在它上面训练的
text2 = decode(encode(text))
print('Training Text is Reconstructable: ', text2 == text) # 可行

# 这段文本我们不认识，不在训练数据里
valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
valtext2 = decode(encode(valtext))
print('Unseen Text is Reconstructable:   ', valtext2 == valtext) # 可行
```

```
hello world!
Training Text is Reconstructable:  True
Unseen Text is Reconstructable:    True
```

**这个 `decode(encode(x))` 对每个 `x` 都能还原成 `x` 吗？**

不，并非总是。处理纯 UTF-8 文本时我们能达到这种编码质量。但处理混合文本时，我们不能保证 $1:1$ 还原。
但无论如何，我们向着比字符级分词更高效的分词迈出了一大步。

## 走向 SOTA：实战中的分词器

### GPT 分词器

据 [\[Radford et al., 2019\]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)，GPT-2 是首个推动用字节对编码 [\[Sennrich et al., 2015\]](https://arxiv.org/abs/1508.07909) 来分词的 GPT 版本。
有趣的是，GPT-2 实现 BPE 的大部分做法和我们几乎一样。但论文随后就拐弯了。
想象一个在数据集里出现频率极高的词。每次它出现，周围可能是不同的其他词/字符。
取决于分词器如何处理，可能发生几种情况：

- 若这些部分经常共现，分词器可能决定把这个词和它周围上下文的部分一起切分或合并。
- 若该词的某些部分随上下文变化，分词器可能为同一个词的不同部分创建多个 token。
- 分词器甚至可能开始把该词的某些部分与上下文中其他常见的前导或尾随字符关联起来。

如论文所指，这种行为是次优的。它产生反映统计频率而非真正上下文含义的 token，导致看起来丰富但实则语义无关的簇。
为此，强制某些类型的字符永远不被合并。
实际分词器实现见 [GitHub - OpenAI/gpt-2/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)。
具体看第 $53$ 行。那里有一个正则模式，用来阻止某些字符被合并到一起。

```python
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") # |：或运算符

# 's、't、're、've、'm、'll、'd 都是缩写（奇怪地区分大小写），要单独挑出来
# " ?\p{L}+" 匹配任意字母序列，带可选前导空格
# " ?\p{N}+" 匹配任意数字序列，带可选前导空格
# " ?[^\s\p{L}\p{N}]+" 匹配任意非空白、非字母、非数字字符序列，带可选前导空格
# "\s+(?!\S)" 匹配任意其后不跟非空白字符的空白字符序列
# "\s+" 匹配任意空白字符序列

print(re.findall(gpt2pat, "Hello world how are you? I've heard you're      4.543 billion years old??!")) # 可行，""" ?\p{L}+""" 匹配 "Hello"，""" ?\p{L}+""" 匹配 "world" 等
```

```
['Hello', ' world', ' how', ' are', ' you', '?', ' I', "'ve", ' heard', ' you', "'re", '     ', ' 4', '.', '543', ' billion', ' years', ' old', '??!']
```

GPT-2 不是直接把输入原样编码，而是先把它切成一列符合正则的输入文本块。
列表条目各自单独处理，然后 token 表示被拼接到一起。

> 你实际上受到限制：只能在符合正则的块的边界内寻找 token，不能跨块。

有意思的是，`r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""` 其实挺次优的。
想想全大写的 `"SHOULD'VE TESTED THAT"`。
上面关于 `'ve` 的规则不匹配，因为它们区分大小写。

```python
print(re.findall(gpt2pat, "SHOULD'VE TESTED THAT"))
print()

example = """for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
"""
print(re.findall(gpt2pat, example))
```

```
['SHOULD', "'", 'VE', ' TESTED', ' THAT']

['for', ' i', ' in', ' range', '(', '1', ',', ' 101', '):', '\n   ', ' if', ' i', ' %', ' 3', ' ==', ' 0', ' and', ' i', ' %', ' 5', ' ==', ' 0', ':', '\n       ', ' print', '("', 'FizzBuzz', '")', '\n   ', ' elif', ' i', ' %', ' 3', ' ==', ' 0', ':', '\n       ', ' print', '("', 'Fizz', '")', '\n   ', ' elif', ' i', ' %', ' 5', ' ==', ' 0', ':', '\n       ', ' print', '("', 'Buzz', '")', '\n   ', ' else', ':', '\n       ', ' print', '(', 'i', ')', '\n']
```

有人可能以为 BPE 现在可以直接在这些输入文本块上跑了，但事实并非如此。
但奇怪的是，看看那段代码示例经正则切分后的输出，再对比 [TikTokenizer](https://tiktokenizer.vercel.app) 的输出：

![tiktokenizer.vercel.app 上 GPT-2 分词器对示例文本的分词](./img/Tiktoken_Vercel_1.png)

**有点对不上。**
OpenAI 似乎在符合正则的块和准备送去做 BPE 分词的文本片段之间，还插了一步额外的切分和切片。
这是为什么？我们此刻只能推测。

### OpenAI TikToken

TikToken 是 [OpenAI 官方分词器实现](https://github.com/openai/tiktoken)。
它为 GPT-2、GPT-3、GPT-4 和 GPT-5 所用的分词器提供接口：

```python
# GPT-2 分词器
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("    Hello World?!!")) # 空格未被合并，如在 TikTokenizer.vercel.app 上所见

# GPT-4 分词器
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("    Hello World?!!"))
```

```
[220, 220, 220, 18435, 2159, 30, 3228]
[262, 22691, 4435, 30, 3001]
```

[见 OpenAI TikToken 仓库中的这个文件](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)。

对 GPT-2 分词，可以看到 `r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""` 在功能上等价于 [GitHub - OpenAI/gpt-2/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)。

对 GPT-4 分词，还能看到这个模式被改进成了 `r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""`。

下面是这个正则模式的逐项说明：

```python
gpt4pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""") 

# | 是逻辑或运算符

# (?i:[sdmt]|ll|ve|re)       匹配 's、'd、'm、't、'll、've、're，现在*不区分大小写*
# [^\r\n\p{L}\p{N}]?+\p{L}+  匹配字母序列，带可选的前导非字母、非数字字符
# \p{N}{1,3}                 匹配长度 1 到 3 的数字序列（有意思！防止过长的数字 token）
#  ?[^\s\p{L}\p{N}]++[\r\n]* 匹配可选空格后跟一个或多个非空白、非字母、非数字字符，然后允许任意多个换行
# \s*[\r\n]                  匹配零或多个空白字符后跟单个换行字符
# \s+(?!\S)                  匹配一个或多个其后不跟非空白字符的空白
# \s+                        匹配一个或多个空白字符（无负向先行断言）

print(re.findall(gpt4pat, "P. Sherman, 42 Wallaby Way, Sydney")) # 可行，""" ?\p{L}+""" 匹配 "Hello"，""" ?\p{L}+""" 匹配 "world" 等
```

```
['P', '.', ' Sherman', ',', ' ', '42', ' Wallaby', ' Way', ',', ' Sydney']
```

让我们真的回到 [GitHub - OpenAI/gpt-2/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)。
可以看到分词器是从 `encoder.json` 文件和 `vocab.bpe` 文件取来的。

- `encoder.json` 持有 token 到整数的映射。
- `vocab.bpe` 持有字节对编码。

```python
# 像 TikToken 那样取回 GPT-2 分词器
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
!wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json
```

```python
# 这等价于我们的 `vocab`，格式相同，只是反过来：我们的是 {idx: b'p0p1'}，它的是 {b'p0p1': idx}
with open('encoder.json', 'r') as f:
    encoder = json.load(f)

# 这等价于我们的 `merges`，bpe_merges 格式相同：{(child1, child2): idx}
with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()

bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
```

```python
print("Vocab Format is {b'p0p1': idx}:           ", list(encoder.items())[:5])
print("Merges Format is {(child1, child2): idx}: ", bpe_merges[:5])
```

```
Vocab Format is {b'p0p1': idx}:            [('!', 0), ('"', 1), ('#', 2), ('$', 3), ('%', 4)]
Merges Format is {(child1, child2): idx}:  [('Ġ', 't'), ('Ġ', 'a'), ('h', 'e'), ('i', 'n'), ('r', 'e')]
```

简言之，我们实现了一个可训练的 BPE 分词器，OpenAI 也实现了一个。
但他们只分享训练结果，不分享我们自己实现的那套训练代码——有趣的是，那套代码与 OpenAI 产出的文件格式兼容。
然而，我们得到的分词结果与 OpenAI 的不同，因为存在一个架构决策：在伪公开的 BPE 分词器之上又加了第二层封闭源码、不公开的编码器。

### 特殊 token

在通过正则切分来引导 token 推导的基础上，我们还可以引入要加入词表的特殊 token。

```python
print(len(encoder)) # 256 个原始字节 token + 50,000 次合并 + --> 1 个特殊 token <--
print(list(encoder.items())[len(encoder)-1])
```

```
50257
('<|endoftext|>', 50256)
```

> 特殊的 `<|endoftext|>` token 被插入训练集，用来标记一篇（上下文相连的）文档的结束和另一篇的开始。模型应学会插入一个概念性的"断点"，以免把连续但可能无关的文档内容混到一起。

![tiktokenizer.vercel.app 上的特殊 token](./img/Tiktoken_Vercel_2.png)

把特殊 token 想成"注入"的 token，不属于 BPE 所称的词表。
特殊 token 可以被注入然后再训练。例如，某个特殊 token 的出现可能触发一次网络搜索例程来丰富 LLM 的上下文。
但这对于微调也是关键架构，比如用于内心独白的 `<|im_start|>`，
或据 [\[Bavarian et al., 2022\]](https://arxiv.org/abs/2207.14255) 的 `<|fim_prefix|>`、`<|fim_middle|>`、`<|fim_suffix|>` *（这其实用在 GPT-4 里）*。
你大可直接去 fork Tiktoken 库，用（注意索引要一致）自定义特殊 token 来扩展它。Tiktoken 会替你处理剩下的事。

### Sentencepiece

开源的 [Sentencepiece](https://github.com/google/sentencepiece) 库既能做 BPE 分词的训练也能做推理（此外还有其他算法）。

> Llama 和 Mistral 模型依赖 Sentencepiece 做分词。

用 Tiktoken 时，我们拿字符串数据，解码成 UTF-8，然后从字节层面开始分词。
Sentencepiece 直接在 Unicode 码点上跑 BPE，而不是在 UTF-8 字节表示上跑。
它有个 `character_coverage` 选项，处理出现次数少的码点：要么映射到一个 UNK token，要么在激活 `byte_fallback` 时，用 UTF-8 编码后改为对原始字节分词。

**Tiktoken 概念上更干净，而 Sentencepiece 更高效。**

```python
import sentencepiece as spm
```

```python
# Sentencepiece 喜欢和文件打交道，所以把训练数据写到文件：
with open("toy.txt", "w", encoding="utf-8") as f:
    # 取自 SentencePiece README（https://github.com/google/sentencepiece）
    f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing. This is not an official Google product.")
```

```python
options = dict(
  # 输入
  input="toy.txt",
  input_format="text",
  # 输出
  model_prefix="tok400", # 输出文件名前缀
  # 算法规格
  model_type="bpe", # BPE 算法
  vocab_size=400,
  # 归一化
  normalization_rule_name="identity", # 呃，关掉归一化
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # 训练句子的最大数量
  max_sentence_length=4192, # 每句的最大字节数
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # 稀有词处理
  character_coverage=0.99995,
  byte_fallback=True,
  # 合并规则（这是 tiktoken 通过正则做的事的另一种方式）
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # 硬编码特殊 token
  unk_id=0, # UNK token 必须存在
  bos_id=1, # 其余可选，设为 -1 关闭
  eos_id=2,
  pad_id=-1,
  # 系统
  num_threads=os.cpu_count(), # 用上~全部系统资源
)

spm.SentencePieceTrainer.train(**options)
```

```python
sp = spm.SentencePieceProcessor()
sp.load("tok400.model")

# 从模型文件检查词表
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
vocab
```

```
[['<unk>', 0],
 ['<s>', 1],
 ['</s>', 2],
 ['<0x00>', 3],
 ['<0x01>', 4],
 ['<0x02>', 5],
 ...（<0x03> 到 <0xFB>，索引 6–258，逐字对应每个字节值）…
 ['<0xFC>', 256],
 ['<0xFD>', 257],
 ['<0xFE>', 258],
 ['en', 259],
 ['▁t', 260],
 ['ce', 261],
 ['in', 262],
 ['ra', 263],
 ['▁a', 264],
 ['de', 265],
 ['er', 266],
 ['is', 267],
 ['pr', 268],
 ['▁s', 269],
 ['ent', 270],
 ['or', 271],
 ['▁m', 272],
 ['▁u', 273],
 ['ing', 274],
 ['▁an', 275],
 ['▁pr', 276],
 ['▁th', 277],
 ['ence', 278],
 ['entence', 279],
 ['Pi', 280],
 ['ed', 281],
 ['em', 282],
 ['ex', 283],
 ['ic', 284],
 ['iz', 285],
 ['la', 286],
 ['on', 287],
 ['st', 288],
 ['▁S', 289],
 ['▁n', 290],
 ['Pie', 291],
 ['end', 292],
 ['ext', 293],
 ['▁is', 294],
 ['▁to', 295],
 ['▁un', 296],
 ['▁the', 297],
 ['Piece', 298],
 ['▁Sentence', 299],
 ['▁SentencePiece', 300],
 ['.]', 301],
 ['Ne', 302],
 ['ag', 303],
 ['ct', 304],
 ['do', 305],
 ['gu', 306],
 ['ir', 307],
 ['it', 308],
 ['ly', 309],
 ['od', 310],
 ['of', 311],
 ['ot', 312],
 ['to', 313],
 ['▁(', 314],
 ['▁[', 315],
 ['▁f', 316],
 ['▁w', 317],
 ['.])', 318],
 ['age', 319],
 ['del', 320],
 ['fic', 321],
 ['ion', 322],
 ['ken', 323],
 ['lan', 324],
 ['ral', 325],
 ['wor', 326],
 ['yst', 327],
 ['▁Ne', 328],
 ['▁al', 329],
 ['▁de', 330],
 ['▁ma', 331],
 ['▁mo', 332],
 ['▁of', 333],
 ['izer', 334],
 ['rain', 335],
 ['ural', 336],
 ['▁and', 337],
 ['▁lan', 338],
 ['▁not', 339],
 ['▁pre', 340],
 ['guage', 341],
 ['ystem', 342],
 ['▁text', 343],
 ['▁model', 344],
 ['▁train', 345],
 ['kenizer', 346],
 ['▁system', 347],
 ['▁language', 348],
 ['▁training', 349],
 ['.,', 350],
 ['BP', 351],
 ['Go', 352],
 ['Ku', 353],
 ['Th', 354],
 ['ab', 355],
 ['al', 356],
 ['as', 357],
 ['at', 358],
 ['▁', 359],
 ['e', 360],
 ['n', 361],
 ['t', 362],
 ['i', 363],
 ['o', 364],
 ['r', 365],
 ['a', 366],
 ['s', 367],
 ['d', 368],
 ['c', 369],
 ['l', 370],
 ['u', 371],
 ['g', 372],
 ['p', 373],
 ['m', 374],
 ['.', 375],
 ['h', 376],
 ['-', 377],
 ['f', 378],
 ['w', 379],
 ['y', 380],
 ['P', 381],
 ['S', 382],
 ['b', 383],
 ['k', 384],
 [')', 385],
 ['x', 386],
 ['z', 387],
 ['(', 388],
 ['N', 389],
 ['[', 390],
 [']', 391],
 ['v', 392],
 [',', 393],
 ['/', 394],
 ['B', 395],
 ['E', 396],
 ['G', 397],
 ['K', 398],
 ['T', 399]]
```

```python
# 现在，词表理顺了，我们可以对文本编码和解码
ids = sp.encode("hello 안녕하세요")
print(ids) # 这是该文本的分词表示
```

```
[359, 376, 360, 370, 370, 364, 359, 239, 152, 139, 238, 136, 152, 240, 152, 155, 239, 135, 187, 239, 157, 151]
```

```python
print([sp.id_to_piece(idx) for idx in ids]) # 这是各 token id 所指代内容的表示
```

```
['▁', 'h', 'e', 'l', 'l', 'o', '▁', '<0xEC>', '<0x95>', '<0x88>', '<0xEB>', '<0x85>', '<0x95>', '<0xED>', '<0x95>', '<0x98>', '<0xEC>', '<0x84>', '<0xB8>', '<0xEC>', '<0x9A>', '<0x94>']
```

在上面的例子里，Sentencepiece 遇到了 `vocab` 没覆盖的输入。
若不是 `byte_fallback=True`，这就成问题了。
设了这个标志后，Sentencepiece 会把输入编码成 UTF-8，然后回退到对原始字节分词。

> 若不是 `byte_fallback=True`，我们会得到 `['_', 'h', 'e', 'l', 'l', 'o', '_', '<unk>']`，因为我们从没在训练数据里真正见过这些韩语字符、没把它们映射成 token，于是它们会被无情地替换成 `<unk>` token。

顺便说说，为什么前面多了一个 `'_'` token？
这是 `add_dummy_prefix=True` 造成的，目的是让句中的 `_hello` 和句首的 `hello` 编码一致。

我们刚才走完的这套设置，和 Llama 训练时用的非常接近。
其实，想要 Llama 的*确切*设置，可参考 [这个 issue](https://github.com/google/sentencepiece/issues/121)。

---

## 练习时间

你可以在随课程的[仓库](https://github.com/karpathy/minbpe/blob/master/exercise.md)里找到一个实现 BPE 分词器的练习。
把它放在这里解答并不适合本笔记本。

---

## 回过头来：vocab_size

本章开头我们已经谈过一些，但在[上一章](<../N007%20-%20GPT%20From%20Scratch/N007%20-%20GPT.ipynb>)里，我们用 PyTorch 实现了 [`gpt.py`](../N007%20-%20GPT%20From%20Scratch/gpt.py)。
我们的 `vocab_size` 是 $65$。我们用 $65$ 维的向量表示每个 token。但这仍是*字符级分词*。
显然，随着 `vocab_size` 增大，这种做法扩展性很差，因为在 $65$ 个向量表示的每一个里，
我们都记下了 $65$ 个 token 各自紧跟当前 token 的可能性。

> 这个分布变得越大，每个向量在"下一个位置该偏向哪个具体 token"上就越缺乏表现力。

```python
torch.manual_seed(1337) # 用于可复现性

# 此阶段还算不上是个语言模型，但我们终会走到那一步……
class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # 给词表做嵌入
        # vocab_size 个 token 中每一个都用一个大小为 vocab_size 的向量表示
        self.embed = nn.Embedding(vocab_size, vocab_size) # 65 个不重复的 65 维向量

    def forward(self, idx, targets):
        # idx 形状为 (batch_size, block_size)
        # targets 形状为 (batch_size, block_size)
        logits = self.embed(idx)
        return logits # 嵌入输入索引，形状变为 (batch_size, block_size, vocab_size) (B, T, C)
```

如果想用一个现有模型并扩展它的词表大小呢？加一个特殊 token 行不行？
两者都能做，但模型的嵌入矩阵必须重新训练。不过这事可以做得相当好。

## 结语

说实话，建模、重塑和微调 token 词表是个很有意思的研究领域，见 [\[Mu et al., 2023\]](https://arxiv.org/abs/2304.08467)。
除此之外，词表与分词的问题越来越脱离"专门为建模语言的工具"这一范畴。
比如 [\[Esser et al., 2020\]](https://arxiv.org/abs/2012.09841) 研究如何把图像作为额外模态喂给 LLM。

> 这个领域似乎正经历一种收敛：transformer 不动，但分词要扩展。"权当这图像也是文本。"如此说来。
> 这就是 OpenAI SORA 做的事：把图像输入高效切块成 token，下游用 transformer 架构处理。

好，收尾，来试着给前面喊出的那些问题找答案：

- **为什么 LLM 不会拼写？**
    - 字符被切成大小不一的 token；提示 `How many letters 'l' are in ".DefaultCellStyle"` 会得到各种错误的答案。
- **为什么 LLM 做不了像反转字符串这样简单的文本操作？**
    - 同样，字符被切成大小不一的 token；这让 ChatGPT 栽跟头，但反转 `.D e f a u l t C e l l S t y l e` 却因为这个原因而行得通。
- **为什么 LLM 在非英语语言上可能更差？**
    - 训练数据中表示更少，导致非英文字符的分词更少（甚至没有）。这撑爆注意力缓冲。糟糕。
- **为什么 LLM 不能正确做简单算术运算？**
    - 数字分词是根因。或者更贴切地说，[整数分词很离谱](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/)
- **为什么输入像 `<|endoftext|>` 这样的特殊字符会让生成停住？**
    - 它是那些特殊 token 之一。正因此它会被非常特定地解读。别把它当 token，而当命令。
- **为什么 LLM 会因"尾部空格"出毛病？**
    - token（例如 GPT 的）常常遵循 `(空格)(某物)` 的模式，所以加个空格会让 GPT 试图为这个已存在的空格也配出点合适的东西，而这很少见，因此"污染"了上下文。
- **为什么 LLM 遇到单词里的大写字母会崩溃？**
    - LLM 从没见过这个；它犯糊涂，很快转向预测词尾 token。
- **为什么用 YAML 配合 LLM 可能比 JSON 好？**
    - YAML 含更少特殊字符，所以更不容易绊到分词器。
- **为什么 LLM 其实不等于端到端语言建模？**
    - 我们需要分词，以形成统一、可扩展的文本表示基础。
- **[`SolidGoldMagikarp`](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) 到底是怎么回事？**
    - 这东西让 ChatGPT 之类的 LLM 彻底崩掉，本质上等于把它越狱了。似乎 reddit 用户 [SolidGoldMagikarp](https://reddit.com/u/SolidGoldMagikarp) 发帖太频繁，以至于分词器在训练中认识了它，但 LLM 不认识。

> 能消灭分词的人，永垂不朽。

## 参考文献

卡帕西讲座本身之外的参考文献。所链接的论文、博文及其他资源列于此。

- [**回顾：字符级分词**](#回顾字符级分词)
- [**分词的陷阱**](#分词的陷阱)
  - [Language Models are Unsupervised Multitask Learners \[Radford et al., 2019\]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)
  - [Tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/)
- [**分词器的思路**](#分词器的思路)
  - [**Unicode**](#unicode)
    - [Nathan Reed - Programmer's Intro to Unicode](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)
    - [UTF-8 Everywhere Manifesto](https://utf8everywhere.org/)
    - [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers \[Yu et al., 2023\]](https://arxiv.org/abs/2305.07185)
- [**字节对编码**](#字节对编码)
  - [**解码**](#解码)
  - [**编码**](#编码)
- [**走向 SOTA：实战中的分词器**](#走向-sota实战中的分词器)
  - [**GPT 分词器**](#gpt-分词器)
    - [Language Models are Unsupervised Multitask Learners \[Radford et al., 2019\]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)
    - [Neural Machine Translation of Rare Words with Subword Units \[Sennrich et al., 2015\]](https://arxiv.org/abs/1508.07909)
    - [GitHub.com/openai/GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
  - [**OpenAI TikToken**](#openai-tiktoken)
    - [GitHub.com/openai/tiktoken](https://github.com/openai/tiktoken)
  - [**特殊 token**](#特殊-token)
    - [Efficient Training of Language Models to Fill in the Middle \[Bavarian et al., 2022\]](https://arxiv.org/abs/2207.14255)
  - [**Sentencepiece**](#sentencepiece)
    - [GitHub.com/google/sentencepiece](https://github.com/google/sentencepiece)
    - [GitHub.com/google/sentencepiece Issue #121](https://github.com/google/sentencepiece/issues/121)
- [**练习时间**](#练习时间)
    - [GitHub.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- [**回过头来：vocab_size**](#回过头来vocab_size)
- [**结语**](#结语)
    - [Learning to Compress Prompts with Gist Tokens \[Mu et al., 2023\]](https://arxiv.org/abs/2304.08467)
    - [Taming Transformers for High-Resolution Image Synthesis \[Esser et al., 2020\]](https://arxiv.org/abs/2012.09841)
    - [Beren Millidge: Integer tokenization is insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/)
    - [Jessica Rumbelow: SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)

<center>Notebook by <a href="https://github.com/mk2112" target="_blank">mk2112</a>.</center>
