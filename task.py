from __future__ import annotations
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import torch
import torch.nn.functional as F
import numpy as np
from preprocessor import process
from config import dict_model_file, dict_optim_file, device, model_path
from utils import file_exists, apply
import os

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
        # self._optim.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()

        process()

    def __load_params(self, params, file):
        if file_exists(file):
            params.load_state_dict(torch.load(file))
        # params.to(device)

    def train(self, epoch: int > 0, loader: DataLoader):
        for epc in range(epoch):
            for idx, (target, input) in enumerate(loader): # -1, text
                input = input.to(device)
                target = target.to(device)
                self._optim.zero_grad()
                output = self._model(input)
                # loss = self.criterion(output, target)
                loss = F.nll_loss(output, target)
                # print(loss, loss.shape)
                loss.backward()
                self._optim.step()
                if (idx % 128) == 0:
                    print(f"epoch:{epc}, idx:{idx}, loss:{loss.item()}")
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(self._model.state_dict(), self._model_file)
            torch.save(self._optim.state_dict(), self._optim_file)

    def test(self, loader: DataLoader):
        losses = []
        accs = []
        for idx, (target, input) in enumerate(loader):
            with torch.no_grad():
                if input.shape != torch.Size([4, 193, 50]):
                    print("test", idx, input.shape)
                input = input.to(device)
                target = target.to(device)
                output = self._model(input)
                loss = F.nll_loss(output, target)
                losses.append(loss)
                # accurate
                pred = output.max(dim=-1)[-1]
                acc = pred.eq(target).float().mean()
                accs.append(acc)
        acc = torch.mean(torch.tensor(accs)).to("cpu").item()
        loss = torch.mean(torch.tensor(losses)).to("cpu").item()
        print(f"acc:{acc}, loss:{loss}")
