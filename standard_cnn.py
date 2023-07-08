from typing import List
import torch
from torch import nn

#evaluate correctness
def get_prediction(tensor):
    return tensor.argmax(dim=-1)

#convolutional neural network class for loading
class ConvoNet(nn.Module):
    def __init__(self, inputs: int, outputs: int, features: List[int], pooling: List[int], taps: List[int]):
        super().__init__()
        self.convo_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        for i, t in enumerate(taps):
            dim = 2 * t + 1
            self.convo_list.append(nn.Conv2d(features[i], features[i + 1], dim, padding='same'))
            self.pool_list.append(nn.AvgPool2d(pooling[i]))
        
        self.readout = nn.Linear(features[-1] * 7*7, outputs)
        self.softmax = nn.Softmax(dim = 1)

        self.active = nn.ReLU()
        
    def forward(self, x):
        for convo, pool in zip(self.convo_list, self.pool_list):
            x = pool(self.active(convo(x)))
        x = torch.flatten(x, start_dim = 1)
        x = self.readout(x)
        x = self.softmax(x)
        return x