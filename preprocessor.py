from __future__ import annotations
from typing import Callable
from utils import file_exists
import os


def assemble_data(text_dict: dict[str, str], cols: list[int],
                  assm: str = "assem.txt", datasetpath: str = "OI-dataset",
                  *, combine: Callable[[str, str], str] = lambda l, t: l + "" + t):
    """storage data in format: col,col,..,col text

    Args:
        text_dict (dict[str, str]): dict of pair (text, label)
        cols (list[int]): in need cols in 15 cols
        assm (str, optional): storage file. Defaults to "assem".
        datasetpath (str, optional). Defaults to "OI-dataset".
    """
    if cols == []:
        return
    for i in range(len(cols)):
        if cols[i] < 0:
            cols[i] += 15
        if cols[i] > 14:
            raise "arg error! col should less than 15"
    assembles: list[str] = []
    for text, label in text_dict.items():
        if file_exists(text) and file_exists(label):
            print(f"process file: {text} & {label}")
            with open(text) as txt, open(label) as lbl:
                texts, labels = txt.readlines(), lbl.readlines()
                if len(texts) != len(labels):
                    raise "text len NOT equals label len"
                for i in range(len(texts)):
                    assembles.append(combine(
                        ",".join(
                            [
                                val 
                                for idx, val in enumerate(labels[i].split(' ')) 
                                if idx in cols
                            ]
                        ),
                        texts[i]
                    )
                    )
    if not os.path.exists(datasetpath):
        os.mkdir(datasetpath)
    if assembles != []:
        with open(os.path.join(datasetpath, assm), "w+") as save:
            save.writelines(assembles)


def process(prefix: str = "OI-dataset", cols: list[int] = [-3], *, force: bool = False):
    if file_exists(f"{prefix}/assem.txt") and not force:
        return
    tls = {}
    def file(x): return f"{prefix}/{x}.txt"
    for f in ["test", "train", "valid"]:
        tls[file(f)] = file(f+"_label")
    assemble_data(tls, cols)

import jieba
def process_fasttext(prefix: str = "OI-dataset", cols: list[int] = [-3], *, force: bool = False):
    def file(x): return f"{prefix}/{x}.txt"
    if file_exists(file("fast")) and not force:
        return
    tls = {}
    for f in ["test", "train", "valid"]:
        tls[file(f)] = file(f+"_label")
    assemble_data(tls, cols, assm="fast.txt", combine=lambda l,
                  t: "__label__"+l[0]+"\t"+" ".join(jieba.lcut(t)))
