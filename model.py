import torch
from torch.nn import Module, Linear, LSTM, ReLU
import torch.nn.functional as F

exp_shape = torch.Size([4, 193, 50])
class TaskNet(Module):
    def __init__(self):
        super(TaskNet,self).__init__()
        self.lstm = LSTM(input_size=50, hidden_size=50,batch_first=True)
        self.fc0 = Linear(50, 11)
        # self.layers = torch.nn.Sequential(
        #     LSTM(input_size=50, hidden_size=50,batch_first=True), # [batch_size, seq_len, embed_dim]
        #     ReLU(),
        #     Linear(50, 11)
        # )
    
    def forward(self, input: torch.LongTensor):
        """
        :param input: [batch_size, max_len, dim_len]
        """
        # x = self.layers(input)
        if input.shape != exp_shape:
            print("unexpected shape:",input.shape, input)
        output, (h,c) = self.lstm(input)
        # [batch_size, max_len, dim_len]
        output = output[:,-1,:]
        x = self.fc0(F.relu(output))
        # return F.softmax(x, dim=-1)
        return F.log_softmax(x , dim=-1)
