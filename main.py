from loader import get_loader
from task import LevelTask
import model
import torch

def main(epoch:int):
    _model = model.TaskNet()
    _optim = torch.optim.Adam(_model.parameters())
    task = LevelTask(_model, _optim)
    task.train(epoch, get_loader(embed=True))
    task.test(get_loader(train=False))


import local_test
if __name__=="__main__":
    # local_test.apply_test()
    # local_test.loader_test()
    # local_test.vocab_test()
    # local_test.forward_test()
    # local_test.init_label_test()
    main(1)