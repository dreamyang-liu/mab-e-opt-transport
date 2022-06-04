from mse.models.conv1d import Conv1d
from mse.label_optimizer.sinkhorn import PseudoOptimizer, SinkhornLabelOptimizer
from mse.trainers.trainer import HybridTrainer, FullSupervisedTrainer
from mse.data_generator.data_generator import DataUtils, ContrasiveLearningDataset, NonTemporalDataset, TemporalDataset
from mse.utils.argparser import args
from mse.eval.eval import Eval

import torch.optim as optim
print(args)

contrasive_data = ContrasiveLearningDataset('./data/train.npy', args)
noncontrasive_data = TemporalDataset('./data/train.npy', args)

input_dim = contrasive_data.get_input_dim()
class_dim = args.num_clusters

model = Conv1d(input_dim, class_dim, [128, 256, 64], 0.5, 3).to(args.device)
# optimizer = optim.Adam(lr = args.lr, params=model.parameters())
for feat, label in noncontrasive_data:
    print(feat.shape, label.shape)
    break
# trainer = FullSupervisedTrainer(model, optimizer, contrasive_data, noncontrasive_data, label_optimizer, args)
# trainer.train()
# print("Trainer finished, start evaluating...")
# x, y = trainer.prepare_data_for_eval()
# evaluator = Eval(args)
# evaluator.eval(x, y)
