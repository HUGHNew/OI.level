import torch
from torch.nn import Module, Linear, LSTM
import torch.nn.functional as F
import jieba
import config


class TaskNet(Module):
    def __init__(self, input_dim: int, word2int, max_len=150, num_dim=50, hidden_size=128):
        super(TaskNet, self).__init__()
        self.w2i = word2int
        self.sentence_fit = lambda s: s[:max_len] \
            if len(s) > max_len \
            else s + [0 for _ in range(max_len-len(s))]
        # self.dropout = torch.nn.Dropout()
        self.embedding = torch.nn.Embedding(input_dim, num_dim)
        self.lstm = LSTM(input_size=num_dim,
                         hidden_size=hidden_size, batch_first=True)
        self.fc_out = Linear(hidden_size, 11)

    def forward(self, raw: str):
        """
        :param raw: tuple[str,..] len(raw) === batch_size
        """
        input = [self.sentence_fit(list(map(self.w2i, jieba.cut(s))))
                 for s in raw]  # [batch_size, max_len, dim_len]
        input = torch.LongTensor(input).to(config.device)
        # input = self.dropout(self.embedding(input))
        input = self.embedding(input)
        output, (hidden, cell) = self.lstm(input)
        # [batch_size, max_len, dim_len]
        # output = output[:,-1,:]
        x = self.fc_out(F.relu(output[:, -1, :].clone().detach()))
        return F.log_softmax(x, dim=-1)
