import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.module import Module

class Conv1d(Module):
    def __init__(self, input_dim, class_dim, hidden_dim, dropout, kernel_size, layers=1):
        super(Conv1d, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.BatchNorm1d(input_dim))

        self.convs.append(nn.Conv1d(in_channels = input_dim, out_channels= hidden_dim, kernel_size =kernel_size ))
        self.convs.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(nn.MaxPool1d(kernel_size = 2, stride = 2))
        self.convs.append(nn.ReLU())
        if dropout !=0:
            self.convs.append(nn.Dropout(p=dropout))
        for i in range(layers-1):
            self.convs.append(nn.Conv1d(in_channels = hidden_dim, out_channels= hidden_dim, kernel_size =kernel_size ))
            self.convs.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(nn.MaxPool1d(kernel_size = 2, stride = 2))
            self.convs.append(nn.ReLU())
            if dropout !=0:
                self.convs.append(nn.Dropout(p=dropout))
        
        self.dense = nn.Linear(hidden_dim, class_dim)
        self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
        x = torch.flatten(x, start_dim=1, end_dim=- 1)
        x = self.dense(x)
        x = self.output(x)

    def compute_features(self, x):
        for _, conv in enumerate(self.convs):
            x = conv(x)
        return x

    def compute_probability_via_feature(self, feat):
        feat = self.dense(feat)
        return self.output(feat)