from functools import reduce
import vocab
import jieba as jb
import torch
import utils

VOCAB = vocab.get_vocab()

LABELS = utils.init_label_list(len(VOCAB) - 1)

def vocab_size(): return len(VOCAB) - 1

def word_dim_size(): return VOCAB["mslen"]

def word_max_count():
    return reduce(lambda _max,p: max(_max,p[1]), VOCAB.items(), 0)

def sen2vec(s: str, mlen:int = VOCAB["mslen"], useOneHot = False) -> list:
    """[batch, len, embed_dim = 50]"""
    word2int = onehot if useOneHot else BoW
    raw = list(map(lambda w:word2int(VOCAB, w), jb.cut(s)))
    # print(raw)
    result = []
    if len(raw) > mlen: result = raw[:mlen]
    else: result = raw + [0 for _ in range(mlen-len(raw))]
    # print(vocab_size, word_dim_size)
    embed = torch.nn.Embedding(vocab_size(), 50)
    return embed(torch.LongTensor(result)) # .to(config.device)

def word2vec(embedder, word:int) -> torch.LongTensor:
    return embedder(word)

def BoW(vocab:dict, word:str) -> int:
    return 0 if word not in vocab else vocab[word]

def onehot(vocab:dict, word:str) -> int:
    if word not in vocab:
        return 0
    return list(vocab.keys()).index(word)