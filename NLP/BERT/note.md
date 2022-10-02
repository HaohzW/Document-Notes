# BERT : Pre-training of Deep Bidirectional Transformers for Language Understanding阅读笔记

[原文](https://arxiv.org/abs/1810.04805 )

## Abstract

**BERT**：一个新的语言表示模型，Bidirectional Encoder Representations from Transformers，在未加标签的文本中做预训练，只需要加一个输出层就能被应用到其他任务中。

与前人工作关系：

- [ELMo](https://arxiv.org/pdf/1802.05365)：ELMo使用RNN，应用到下游任务时候需要对架构做调整，BERT使用transformer，因此不需要做调整
- GPT：GPT只考虑单向的，用左边上下文信息预测未来，BERT联合了左右的上下文信息，是双向的

## Introduction

预训练语言模型能提升自然语言处理任务的效果（例如词嵌入，GPT等）

有两类利用预训练语言模型的策略：

1. 基于特征（feature-based）：例如ELMo，对每一个下游的任务，将预训练的表示作为额外的特征和输入一起作为模型的输入
2. 基于微调（fine-tuning）：例如GPT，将预训练的模型训练好后，不需要改变太多，只需要做微调

现有的预训练模型的局限性是：使用标准的语言模型，是单向的

​	BERT采用**基于掩码的语言模型（masked language model，MLM）**【MLM随机遮挡输入的某些token，目标是基于上下文预测该位置的原始token】，这样便结合了**双向的信息**；此外，BERT还做了**“预测下一个句子”**的任务，能让模型学习**句子层面**的信息；

- 这篇文章强调了双向预训练语言表示模型的重要性；
- BERT是一个一个基于微调的模型，对于很多sentence-level和token-level的特定任务均适用，在特定任务中，不需要做很繁重的模型调整和训练，且能表现地很好。

