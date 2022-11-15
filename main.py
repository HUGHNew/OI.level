from loader import get_loader
from task import LevelTask
import model
import torch
import config
import ws

def main(epoch:int):
    _model = model.TaskNet(ws.vocab_size(), lambda word: ws.onehot(ws.VOCAB, word), max_len=50)
    _optim = torch.optim.Adam(_model.parameters())
    task = LevelTask(_model, _optim, label="mini100")
    testloader = get_loader(train=False)
    task.load("./models/mini10k/model.pt", "./models/mini10k/optim.pt")
    task.train(epoch, get_loader(partBetter=True))
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
    main(100)
    # test()