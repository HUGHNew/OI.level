from loader import get_loader
from task import LevelTask

def main():
    task = LevelTask()
    task.train()
    task.test()

if __name__=="__main__":
    # main()
    for idx,(label, text) in enumerate(get_loader()):
        print(label)
        print(text)
        break