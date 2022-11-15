from __future__ import annotations
from random import shuffle
import sys
import traceback
from utils import file_exists, apply
from ws import sen2vec, LABELS
from torch.utils.data import Dataset, DataLoader
import torch
import os
from config import dataset_path, assem_file

train_file = "assem_train.txt"
test_file = "assem_test.txt"


class TaskDataset(Dataset):
    def __init__(self, dataPath: str = dataset_path, dataFile: str = assem_file, *,
                 train: bool = True, trainRate: float = 0.8,
                 repartition: bool = False, betterPart: bool = False):
        self.data: list[tuple[int, str]] = []
        self.isTrain = train
        self.path = dataPath
        self.rawdata = os.path.join(self.path, dataFile)
        self._load_data(repartition, trainRate, betterPart)

    def _load_data(self, repart: bool, rate: float, betterPart: bool = False):
        if repart:
            if file_exists(self.rawdata):
                self.data = self.__read_from_file(self.rawdata)
            self.data = self.__partition(rate, betterPart)
        else:
            file = os.path.join(
                self.path, train_file if self.isTrain else test_file)
            if not file_exists(file):
                self._load_data(True, rate, betterPart)
            else:
                self.data = self.__read_from_file(file)

    def __read_from_file(self, file: str) -> list[tuple[int, str]]:
        if not file_exists(file):
            return []

        def l2t(l: tuple[str, str]): return (int(l[0]), l[1])
        with open(file, "r") as fd:
            result = [l2t(s.strip().split(' ', 2))
                      for s in fd.readlines() if s != "\n"]  # skip blank line
        return result
    #region partition
    def __part_base(self, rate: float) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        shuffle(self.data)
        edge: int = int(len(self.data) * rate)
        return self.data[:edge], self.data[edge:]

    def __part_better(self, rate: float) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        """keep the number unclassified equals others"""
        label_data:list[list[tuple[int, str]]] = [[] for _ in range(11)]
        other_count = 0
        for idx, text in self.data:
            label_data[idx].append((idx, text))
            if idx != 0: other_count += 1
        label_data[0] = label_data[0][:other_count]
        tr, ts = [], []
        for labels in label_data:
            tr.extend(labels[:int(len(labels)*rate)])
            ts.extend(labels[int(len(labels)*rate):])
        return tr, ts

    def __partition(self, rate: float, better: bool = False) -> list[tuple[int, str]]:
        """split data to train and test

        Args:
            rate (float): ratio for split

        Returns:
            list[tuple[int, str]]: return the in need part
        """
        tr, ts = self.__part_better(rate) if better else self.__part_base(rate)
        self.__write_or_cancel(os.path.join(self.path, train_file), tr)
        self.__write_or_cancel(os.path.join(self.path, test_file), ts)
        return tr if self.isTrain else ts
    #endregion partition

    def __write_or_cancel(self, file: str, data: list[tuple[int, str]]):
        try:
            with open(file, "w") as train:
                train.writelines(map(lambda x: f"{x[0]} {x[1]}\n", data))
        except Exception as e:
            print(f"fail to write data to {file}", file=sys.stderr)
            traceback.print_exc()
            os.remove(file)
    
    def __getitem__(self, index) -> tuple[int, str]:
        # raw = self.data[index]
        # return (LABELS[raw[0]], raw[1])
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def get_loader(batch: int = 4, train: bool = True, partBetter: bool = False) -> DataLoader:
    return DataLoader(TaskDataset(train=train, betterPart=partBetter), batch_size=batch, shuffle=True, drop_last=True)
