from builtins import breakpoint
import torch.nn as nn
import torch.nn.functional as F


class FullConnected(nn.Module):

    def __init__(self, input_dim, class_dim, layers, drop=0.5, head=1):
        super(FullConnected, self).__init__()
        self.backbone = nn.ModuleList()
        if not isinstance(input_dim, int):
            layers.insert(0, input_dim[0])
        for i in range(len(layers) - 1):
            self.add_bn_dense_act_layer(layers[i], layers[i + 1], drop)
        self.dense = nn.Linear(layers[-1], class_dim)
        self.output = nn.Softmax(dim=-1)
    
    def add_bn_dense_act_layer(self, input_dim, hidden_dim, drop):
        self.backbone.append(nn.BatchNorm1d(input_dim))
        self.backbone.append(nn.Linear(input_dim, hidden_dim, bias=True))
        self.backbone.append(nn.ReLU())
        if drop > 0:
            self.backbone.append(nn.Dropout(p=drop))
    
    def forward(self, x):
        x = self.compute_feature(x)
        x = self.compute_probability_via_feature(x)
        return x
    
    def compute_feature(self, x):
        for l in self.backbone:
            x = l(x)
        return x
    
    def compute_probability_via_feature(self, feat):
        feat = self.dense(feat)
        return self.output(feat)


class FullConnected(nn.Module):

    def __init__(self, input_dim, class_dim, layers, drop=0.5, head=1):
        super(FullConnected, self).__init__()
        self.backbone = nn.ModuleList()
        if not isinstance(input_dim, int):
            layers.insert(0, input_dim[0])
        for i in range(len(layers) - 1):
            self.add_bn_dense_act_layer(layers[i], layers[i + 1], drop)
        self.dense = nn.Linear(layers[-1], class_dim)
        self.output = nn.Softmax(dim=-1)
    
    def add_bn_dense_act_layer(self, input_dim, hidden_dim, drop):
        self.backbone.append(nn.BatchNorm1d(input_dim))
        self.backbone.append(nn.Linear(input_dim, hidden_dim, bias=True))
        self.backbone.append(nn.ReLU())
        if drop > 0:
            self.backbone.append(nn.Dropout(p=drop))
    
    def forward(self, x):
        x = self.compute_feature(x)
        x = self.compute_probability_via_feature(x)
        return x
    
    def compute_feature(self, x):
        for l in self.backbone:
            x = l(x)
        return x
    
    def compute_probability_via_feature(self, feat):
        feat = self.dense(feat)
        return self.output(feat)
