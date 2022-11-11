from loader import get_loader
from task import LevelTask
import model
import torch

def main(epoch:int):
    _model = model.TaskNet()
    _optim = torch.optim.Adam(_model.parameters())
    task = LevelTask(_model, _optim)
    task.train(epoch, get_loader(embed=True))
    task.test(get_loader(train=False, embed=True))

if __name__=="__main__":
    main(1)