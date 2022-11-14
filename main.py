from loader import get_loader
from task import LevelTask
import model
import torch
import config

def main(epoch:int):
    _model = model.TaskNet()
    _optim = torch.optim.Adam(_model.parameters())
    task = LevelTask(_model, _optim)
    tl = get_loader(train=False, embed=True)
    task.train(epoch, get_loader(embed=True, partBetter=True), tl, label="100")
    task.test(tl)

def test():
    m = model.TaskNet()
    m.load_state_dict(torch.load(config.dict_model_file))
    o = torch.optim.Adam(m.parameters())
    o.load_state_dict(torch.load(config.dict_optim_file))
    # m = torch.load("models/debug_model.pt")
    # o = torch.load("models/debug_optim.pt")
    LevelTask(m, o).test(get_loader(train=False, embed=True))

if __name__=="__main__":
    main(100)
    # test()