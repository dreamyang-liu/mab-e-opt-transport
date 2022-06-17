from mse.models.conv1d import Conv1d
from mse.label_optimizer.sinkhorn import PseudoOptimizer, SinkhornLabelOptimizer
from mse.trainers.trainer import HybridTrainer, FullSupervisedTrainer, Conv1dTrainer
from mse.data_generator.data_generator import DataUtils, ContrasiveLearningDataset, NonTemporalDataset, TemporalDataset
from mse.utils.argparser import args
from mse.eval.eval import Eval

import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
import numpy as np
print(args)

class TemporalDataset_Simple():
    def __init__(self, feat, label, args):
        self.args = args
        self.feat, self.label = feat, label
        self.idxs = torch.arange(self.feat.shape[0])
        self.prepare_batch_idxs()
    def prepare_batch_idxs(self):
        self.batch_idxs = []
        for i in range(0, self.idxs.shape[0], self.args.batch_size):
            self.batch_idxs.append([i, min(i + self.args.batch_size, self.idxs.shape[0])])
        self.batch_idxs = torch.tensor(self.batch_idxs)
    def __getitem__(self, idx):
        idx = self.idxs[self.batch_idxs[idx][0]:self.batch_idxs[idx][1]]
        feat = self.feat[idx]
        label = self.label[idx]
        return feat, label

contrasive_data = ContrasiveLearningDataset('./data/train.npy', args)
noncontrasive_data_train = TemporalDataset('./task1_npy/calms21_task1_train.npy', args)
noncontrasive_data_train.randomize()
noncontrasive_data_test = TemporalDataset('./task1_npy/calms21_task1_test.npy', args)

# Split test from train
# train_idx, test_idx= train_test_split(
#     np.arange(noncontrasive_data.feat.shape[0]), test_size=0.2, random_state=66, shuffle=True, stratify=noncontrasive_data.raw_label)
# print (len(train_idx), len(test_idx))
# noncontrasive_data_train = TemporalDataset_Simple(noncontrasive_data.feat[train_idx], noncontrasive_data.label[train_idx], args)
# noncontrasive_data_test = TemporalDataset_Simple(noncontrasive_data.feat[test_idx], noncontrasive_data.label[test_idx], args)

# train_test_split_idx = int(noncontrasive_data.feat.shape[0]*0.8)
# noncontrasive_data_train = TemporalDataset_Simple(noncontrasive_data.feat[:train_test_split_idx], noncontrasive_data.label[:train_test_split_idx], args)
# noncontrasive_data_test = TemporalDataset_Simple(noncontrasive_data.feat[train_test_split_idx:], noncontrasive_data.label[:train_test_split_idx:], args)

# input_dim = contrasive_data.get_input_dim()
input_dim = noncontrasive_data_train.get_input_dim()
print (input_dim)
class_dim = args.num_clusters
model = Conv1d(input_dim, class_dim, [128, 64, 32], 0.5, 5, 3).to(args.device)
# model = Conv1d(input_dim, class_dim, 64, 0.5, 2, 1).to(args.device)
optimizer = optim.Adam(lr = args.lr, params=model.parameters())
for feat, label in noncontrasive_data_train:
    print(feat.shape, label.shape)
    break

label_optimizer = None
trainer = Conv1dTrainer(model, optimizer, contrasive_data=None, noncontrasive_data = noncontrasive_data_train, label_optimizer = label_optimizer, args = args)
trainer.train(noncontrasive_data_test)
print("Trainer finished")
# x, y = trainer.prepare_data_for_eval()
# evaluator = Eval(args)
# evaluator.eval(x, y)
# trainer.eval(noncontrasive_data_test)
