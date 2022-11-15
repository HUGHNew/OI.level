from loader import get_loader
from task import LevelTask
import model
import torch
import config
import ws

def main(epoch:int, label:str, loadPath:str = None):
    _model = model.TaskNet(ws.vocab_size(), lambda word: ws.onehot(ws.VOCAB, word), max_len=70)
    _optim = torch.optim.Adam(_model.parameters(), lr=1e-2)
    task = LevelTask(_model, _optim, label=label)
    testloader = get_loader(train=False)
    if not loadPath == None:
        task.load(f"./models/{loadPath}/model.pt", f"./models/{loadPath}/optim.pt")
    task.train(epoch, get_loader(partBetter=True), testloader)
    task.test(testloader)

def test():
    m = model.TaskNet(ws.vocab_size(), lambda word: ws.onehot(ws.VOCAB, word), max_len=50)
    o = torch.optim.Adam(m.parameters())
    m.load_state_dict(torch.load(config.dict_model_file))
    o.load_state_dict(torch.load(config.dict_optim_file))
    # m = torch.load("models/debug_model.pt")
    # o = torch.load("models/debug_optim.pt")
    LevelTask(m, o).test(get_loader(train=False))

if __name__=="__main__":
    main(1000, "1k_t1")
    # test()