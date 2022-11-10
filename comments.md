# Comments

> 一些执行过程中的评论和想法

## 10.12

完成了基于 mnist 这个任务的学习
- 图片比较好规范化
- mnist 是一个比较确定的**十分类**问题

迁移到NLP任务
- 文本不太好规范化(统一长度)
  - word2vec 可以将 word 转化为一个向量
  - 但是 sentence 不好做统一长度
- 不太确定这算是什么问题(也许算是一个**多分类**问题)

## 10.06

有点离谱 cs224n 看着头大 书也是从分词开始讲一些东西
> 缺少一个**自顶而下**的介绍来告诉我这些东西对于完成一个任务的作用

换一个实操性的视频 先把理论放一下 实现快速实操

## 10.03

从 cs224n 开始看 直接进 Deep Learning 有问题再补基础知识

cs224n 最后选择 2021winter B站翻译质量好一些 同时从 TensorFlow 换成了 PyTorch

同时选择书籍 《自然语言处理入门》

以任务为驱动 尽快完成一个给定的任务(预计5个工作日)

## 11.08

data process
1. (:label text)
2. cut(jieba.cut)
3. embedding(before torch.nn.Embedding)
   1. one-hot
   2. BoW(TF)
   3. [batch_size, len = maxInTrainData, dim = sqrt(vocab_size)]
4. model
   1. LSTM
   2. fc
   3. softmax