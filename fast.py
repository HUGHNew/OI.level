import fasttext
from utils import file_exists
from config import file_fast_relat, model_fast_relat, dataset_path, model_path
from preprocessor import process_fasttext
import os

train_file = os.path.join(dataset_path, "fast_train.txt")
valid_file = os.path.join(dataset_path, "fast_valid.txt")

def partition(rate: float):
    if file_exists(train_file): return
    if not file_exists(file_fast_relat): raise RuntimeError("dataset has not prepared!")
    with open(file_fast_relat) as fastfd:
        content = fastfd.readlines()
    idx = int(len(content) * rate)
    with open(train_file, "w") as tr: tr.writelines(content[:idx])
    with open(valid_file, "w") as tr: tr.writelines(content[idx:])

def prepare(rate: float= 0.8):
    if not file_exists(file_fast_relat):
        process_fasttext()
    partition(rate)

def train(file, model_): # 训练模型
    prepare()
    model = fasttext.train_supervised(file, lr=0.1, dim=50,
             epoch=5, word_ngrams=2, loss='softmax')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model.save_model(model_)

def valid(file, model): # 预测
    classifier = fasttext.load_model(model)
    result = classifier.test(file)
    print("准确率:", result)
    with open(file, encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            print(line, classifier.predict([line])[0][0][0])

def runDirect(useModel = False):
    prepare()
    if useModel:
        model = fasttext.load_model(model_fast_relat)
    else:
        model = fasttext.train_supervised(train_file, lr=0.05, dim=100,
                epoch=5000, word_ngrams=2, loss='softmax')
        model.save_model(model_fast_relat)
    result = model.test(valid_file)
    pos_acc = 0
    pos_count = 0
    with open(valid_file) as valid:
        for line in valid.readlines():
            line = line.strip()
            r = line.split("\t")[0]
            p = model.predict([line])[0][0][0]
            if not r == "__label__-":
                pos_count += 1
                if r == p:
                    pos_acc += 1
    print("准确率:", result[1], "确定正类", pos_acc / pos_count)

if __name__ == '__main__':
    # train(train_file, model_fast_relat)
    # valid(valid_file, model_fast_relat)
    runDirect(True)