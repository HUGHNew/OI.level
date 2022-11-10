import loader
from model import TaskNet
import utils
import ws
import vocab
import torch

def forward_test():
    output = TaskNet().forward(torch.randint(0,9,(4,193,50))/10)
    print(output, output.shape)

def init_label_test():
    lst = utils.init_label_list(5000)
    print(len(lst), lst[0].shape)

def loader_test():
    for i in loader.get_loader(embed=True):
        print(i)
        print(len(i), i[0].shape, i[1].shape)
        break

def vocab_test():
    print(ws.vocab_size(), ws.word_dim_size())
    voc = vocab.get_vocab()
    m = 0
    for key,val in voc.items():
        m = max(val, m)
    print("voc max:", m)
    print(ws.word_max_count())

def apply_test():
    lst = [(0,"我的经椎骨第六骨关节损伤太严重能算工伤吗？"), (5,"工伤五级")]
    utils.apply(lst, lambda x:(x[0], ws.sen2vec(x[1], useOneHot=True)), False)
    # lst[0] = (-1, sen2vec(lst[0][1],20))
    print(lst, lst[0][1].shape)