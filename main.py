from loader import get_loader
from task import LevelTask

def main(epoch:int):
    task = LevelTask()
    task.train(epoch, get_loader())
    task.test(get_loader(train=False))

if __name__=="__main__":
    main()