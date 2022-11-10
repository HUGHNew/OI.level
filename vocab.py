import jieba as jb
import pickle
from utils import file_exists
from config import vocab_file


def get_vocab(file = "OI-dataset/assem_train.txt", refresh:bool = False) -> dict:
    if file_exists(vocab_file) and not refresh:
        with open(vocab_file,"rb") as fd:
            return pickle.load(fd)
    vocab = {} # "mslen" for max sentence length
    with open(file) as fd:
        content = fd.readlines()
    mslen = 0
    for line in content:
        line = line.split(' ', 2)[1]
        mslen = max(mslen, len(line))
        for word in jb.cut(line):
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    vocab["mslen"] = mslen
    with open(vocab_file, "wb+") as fd:
        pickle.dump(vocab, fd)
    return vocab
