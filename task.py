from __future__ import annotations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import torch
import torch.nn.functional as F
from preprocessor import process
from config import device, model_path
from utils import file_exists, apply
import os

class LevelTask:
    def __init__(self, model: Module, optim: Optimizer, label:str = ".",
                 model_save: str = "model.pt", optim_save: str = "optim.pt"):
        self.save_path = os.path.join(model_path, label)
        self._model = model
        self._model_file = os.path.join(self.save_path, model_save)
        self._model.to(device)
        self._optim = optim
        self._optim_file = os.path.join(self.save_path, optim_save)
        # self._optim.to(device)
        self.label = label
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss = [] # average loss of each epoch
        self.writer = SummaryWriter()

        process() # preprocess

    def train(self, epoch: int > 0, loader: DataLoader, testLoader: DataLoader = None):
        sum_loss = 0
        count = len(loader) * loader.batch_size
        acc = 0
        for epc in range(epoch):
            for idx, (target, input) in enumerate(loader): # -1, text
                # input = input.to(device)
                target = target.to(device)
                self._optim.zero_grad()
                output = self._model(input)
                # loss = self.criterion(output, target)
                loss = F.nll_loss(output, target)
                loss.backward()
                self._optim.step()
                sum_loss += loss.item()
                # print((output.max(dim=-1)[-1]>0).eq(target>0).int())
                # raise RuntimeError()
                # true_list = output.max(dim=-1)[-1].eq(target).int()
                acc += output.max(dim=-1)[-1].eq(target).int().sum().item()
                # acc += (output.max(dim=-1)[-1]>0).eq(target>0).int().sum().item()
            if (epc % 16) == 0:
                if testLoader != None:
                    ac, loss = self.test(testLoader, False)
                    self.writer.add_scalar(f"test/acc_{self.label}", ac, epc)
                    self._model.train()
            if (epc % 128) == 0:
                self.save(str(epc))
            # summary graph
            self.writer.add_scalar(f"train/loss_{self.label}", sum_loss/count, epc)
            self.writer.add_scalar(f"train/acc_{self.label}", acc/count, epc)
            print(epc,"loss", sum_loss/count,"acc", acc/count)
            sum_loss = 0
            acc = 0
        self.writer.close()
        self.save()


    def test(self, loader: DataLoader, verbose: bool = True) -> tuple[float, float]:
        """
        Args:
            loader (DataLoader): test dataloader
            verbose (bool, optional): print acc loss. Defaults to True.

        Returns:
            tuple[float, float]: (acc, loss)
        """
        losses = []
        accs = []
        self._model.eval()
        for idx, (target, input) in enumerate(loader):
            with torch.no_grad():
                # input = input.to(device)
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
        # torch.save(self._model, "models/debug_model.pt")
        # torch.save(self._optim, "models/debug_optim.pt")
        if verbose:
            print(f"acc:{acc}, loss:{loss}")
        return acc, loss

    def save(self, suffix:str = ""):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self._model.state_dict(), self._model_file+suffix)
        torch.save(self._optim.state_dict(), self._optim_file+suffix)
    
    def load(self, model_file, optim_file):
        if file_exists(model_file):
            self._model.load_state_dict(torch.load(model_file))
        if file_exists(optim_file):
            self._optim.load_state_dict(torch.load(optim_file))