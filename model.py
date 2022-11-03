import torch
from torch.nn import Module, Linear, CrossEntropyLoss
import torch.nn.functional as F

class TaskNet(Module):
    def __init__(self):
        self.ly1 = Linear(1*28*28, 28)
        self.ly2 = Linear(28, 10)
    
    def forward(self, input: torch.Tensor):
        """
        :param input: [batch_size, 1, 28, 28]
        """
        x = input.view([input.size(0), 1*28*28])
        # input.view(-1, 1*28*28)
        
        # 全链接
        x = self.ly1(x)
        # activate function
        x = F.relu(x)
        # output
        x = self.ly2(x)
        return F.log_softmax(x , dim=-1)
