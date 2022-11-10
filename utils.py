from __future__ import annotations
import os
import torch

def dist_summary():
    """show data distribution
    -1 :3302/3996
    1  :6/3996
    2  :17/3996
    3  :11/3996
    4  :25/3996
    5  :32/3996
    6  :44/3996
    7  :48/3996
    8  :79/3996
    9  :180/3996
    10 :252/3996
    """
    with open("OI-dataset/assem.txt") as ds:
        content = ds.readlines()
    counter = [0] * 11
    for sentence in content:
        v = int(sentence.split(' ', 2)[0])
        if v == -1:
            v = 0
        counter[v] += 1
    for idx,val in enumerate(counter):
        print(f"{idx}:{val}/{len(content)}")

def file_exists(file: str) -> bool:
    return os.path.exists(file) and os.path.isfile(file)

def apply(_l: list, f, isMutable:bool = True):
    for idx in range(len(_l)):
        if isMutable: f(_l[idx])
        else: _l[idx] = f(_l[idx])

def num2onehot(num:int, len:int)->torch.LongTensor:
    x = torch.zeros(len, dtype=torch.long)
    x[num] = 1
    return x

def init_label_list(num_embedding)->list[torch.LongTensor]:
    embed = torch.nn.Embedding(num_embedding, 50)
    return [embed(num2onehot(i, 11)) for i in range(11)]