from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from preprocessor import process

from utils import file_exists

class LevelTask:
    def __init__(self, model: Module, optim: Optimizer, loader: DataLoader):
        self._model = model
        self._optim = optim
        self._loader = loader

        process()

    def train(self, model_file: str, optim_file: str): pass
    def test(self,): pass
