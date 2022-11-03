from __future__ import annotations
from random import shuffle
from utils import file_exists
from torch.utils.data import Dataset, DataLoader
import torch
import os

train_file = "assem_train.txt"
test_file = "assem_test.txt"


class TaskDataset(Dataset):
    def __init__(self, dataPath: str = "OI-dataset", dataFile: str = "assem.txt", *,
                 train: bool = True, embed: bool = False, trainRate: float = 0.8, repartition: bool = False):
        self.data: list[tuple[int, str | torch.Tensor]] = []
        self.embedding = embed
        self.isTrain = train
        self.path = dataPath
        self.rawdata = os.path.join(self.path, dataFile)
        self._load_data(repartition, trainRate)
        if self.embedding:
            # apply(self.data, lambda x: (x[0], word2vec(x[1])))
            pass

    def _load_data(self, repart: bool, rate:float):
        if repart:
            if file_exists(self.rawdata):
                self.data = self.__read_from_file(self.rawdata)
            self.data = self.__partition(rate)
        else:
            self.data = self.__read_from_file(os.path.join(
                self.path, train_file if self.isTrain else test_file))

    def __read_from_file(self, file: str) -> list[tuple[int, str]]:
        if not file_exists(file):
            return []

        def l2t(l): return (int(l[0]), l[1])
        with open(file, "r") as fd:
            result = [l2t(s.split(' ')) for s in fd.readlines()]
        return result

    def __partition(self, rate:float)-> list[tuple[int, str]]:
        """split data to train and test

        Args:
            rate (float): ratio for split

        Returns:
            list[tuple[int, str]]: return the in need part
        """
        shuffle(self.data)
        edge: int = int(len(self.data) * rate)
        tr, ts = self.data[:edge], self.data[edge:]
        with open(os.path.join(self.path, train_file), "w") as train:
            train.writelines(tr)
        with open(os.path.join(self.path, test_file), "w") as test:
            test.writelines(ts)
        return tr if self.isTrain else ts

    def __getitem__(self, index) -> tuple[int, str | torch.Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

def get_loader(batch: int = 4, train:bool = True) -> DataLoader:
    return DataLoader(TaskDataset(train=train), batch_size=batch, shuffle=True)