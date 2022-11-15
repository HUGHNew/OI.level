from loader import get_loader
from task import LevelTask
import model
import torch
import config
import ws

def main(epoch:int):
    _model = model.TaskNet(ws.vocab_size(), lambda word: ws.onehot(ws.VOCAB, word))
    _optim = torch.optim.Adam(_model.parameters())
    task = LevelTask(_model, _optim)
    tl = get_loader(train=False)
    task.train(epoch, get_loader(partBetter=True), tl, label="|mini10k")
    # task.test(tl)

def test():
    m = model.TaskNet()
    m.load_state_dict(torch.load(config.dict_model_file))
    o = torch.optim.Adam(m.parameters())
    o.load_state_dict(torch.load(config.dict_optim_file))
    # m = torch.load("models/debug_model.pt")
    # o = torch.load("models/debug_optim.pt")
    LevelTask(m, o).test(get_loader(train=False))

if __name__=="__main__":
    main(10000)
    # test()