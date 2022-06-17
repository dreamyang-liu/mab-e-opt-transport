import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.module import Module
class Conv1d(Module):
    # from benchmark
    # input_dim = ((past_frames + future_frames + 1), flat_dim,)
    def __init__(self, input_dim, class_dim, hidden_dim, dropout, kernel_size, layers=1):
        super(Conv1d, self).__init__()
        self.convs = torch.nn.ModuleList()
        if layers == 1:
            self.convs.append(nn.BatchNorm1d(input_dim[-2]))
            self.convs.append(nn.Conv1d(in_channels = input_dim[-1], out_channels= hidden_dim, kernel_size =kernel_size ))
            self.convs.append(nn.BatchNorm1d(input_dim[-2]-kernel_size+1))
            self.convs.append(nn.MaxPool1d(kernel_size = 2, stride = 2))
            self.convs.append(nn.ReLU())
            if dropout !=0:
                self.convs.append(nn.Dropout(p=dropout))
            self.convs.append(nn.Flatten(start_dim=-2))
            self.dense = nn.Linear(int((input_dim[-2]-kernel_size+1)*hidden_dim/2), class_dim)
        else:
            assert layers == len(hidden_dim), f"number of layers not aligned with hidden_dim list"
            self.convs.append(nn.BatchNorm1d(input_dim[-2]))
            self.convs.append(nn.Conv1d(in_channels = input_dim[-1], out_channels= hidden_dim[0], kernel_size =kernel_size ))
            runnning_dim_1 = input_dim[-2]-kernel_size+1
            self.convs.append(nn.BatchNorm1d(runnning_dim_1))
            self.convs.append(nn.ReLU())
            if dropout !=0:
                self.convs.append(nn.Dropout(p=dropout))

            for i in range(layers-1):
                self.convs.append(nn.Conv1d(in_channels = hidden_dim[i], out_channels= hidden_dim[i+1], kernel_size =kernel_size ))
                runnning_dim_1 = runnning_dim_1-kernel_size+1
                self.convs.append(nn.BatchNorm1d(runnning_dim_1))
                self.convs.append(nn.MaxPool1d(kernel_size = 2, stride = 2))
                runnning_dim_1 = int(runnning_dim_1//2)
                self.convs.append(nn.ReLU())
                if dropout !=0:
                    self.convs.append(nn.Dropout(p=dropout))
            self.convs.append(nn.Flatten(start_dim=1))
            self.dense = nn.Linear(int(runnning_dim_1*hidden_dim[-1]), class_dim)
        
        self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            if isinstance(conv, nn.Conv1d) or isinstance(conv, nn.MaxPool1d):
                x = conv(x.transpose(-2,-1)).transpose(-2,-1)
            else:
                x = conv(x)
        x = torch.flatten(x, start_dim=1, end_dim=- 1)
        x = self.dense(x)
        x = self.output(x)
        return x

    def compute_features(self, x):
        for _, conv in enumerate(self.convs):
            if isinstance(conv, nn.Conv1d) or isinstance(conv, nn.MaxPool1d):
                x = conv(x.transpose(-2,-1)).transpose(-2,-1)
            else:
                x = conv(x)
        return x

    def compute_probability_via_feature(self, feat):
        feat = self.dense(feat)
        return self.output(feat)