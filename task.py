from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import torch
from preprocessor import process
from __future__ import annotations
from config import dict_model_file, dict_optim_file, device
from utils import file_exists
import torch.nn.functional as F
import numpy as np


class LevelTask:
    def __init__(self, model: Module, optim: Optimizer,
                 model_save: str = dict_model_file, optim_save: str = dict_optim_file):
        self._model = model
        self._model_file = model_save
        self.__load_params(self._model, model_save)
        self._model.to(device)
        self._optim = optim
        self._optim_file = optim_save
        self.__load_params(self._optim, optim_save)
        self._model.to(device)

        process()

    def __load_params(self, params, file):
        if file_exists(file):
            params.load_state_dict(torch.load(file))
        params.to()

    def train(self, epoch: int > 0, loader: DataLoader):
        for epc in range(epoch):
            for idx, (target, input) in enumerate(loader): # -1, text
                input.to(device)
                target.to(device)
                self._optim.zero_grad()
                output = self._model(input)
                loss = F.nll_loss(output, target)
                loss.backward()
                self._optim.step()
                if (idx & 128) == 0:
                    print(f"epoch:{epc}, idx:{idx}, loss:{loss.item()}")
                if (idx & 1024) == 0:
                    torch.save(self._model.state_dict(), self._model_file)
                    torch.save(self._optim.state_dict(), self._optim_file)
            torch.save(self._model.state_dict(), self._model_file)
            torch.save(self._optim.state_dict(), self._optim_file)

    def test(self, loader: DataLoader):
        losses = []
        accs = []
        for idx, (input, target) in enumerate(loader):
            with torch.no_grad():
                output = self._model(input)
                loss = F.nll_loss(output, target)
                losses.append(loss)
                # accurate
                pred = output.max(dim=-1)[-1]
                acc = pred.eq(target).float().mean()
                accs.append(acc)
            print(f"acc:{np.mean(accs)}, loss:{np.mean(loss)}")
