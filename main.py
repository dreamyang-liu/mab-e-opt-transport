from mse.models.full_connected import FullConnected
from mse.label_optimizer.sinkhorn import PseudoOptimizer, SinkhornLabelOptimizer
from mse.trainers.trainer import HybridTrainer
from mse.data_generator.data_generator import DataUtils, ContrasiveLearningDataset, NonTemporalDataset
from mse.utils.argparser import args
from mse.eval.eval import Eval

import torch.optim as optim
print(args)

contrasive_data = ContrasiveLearningDataset('./data/train.npy', args)
noncontrasive_data = NonTemporalDataset('./data/train.npy', args)

input_dim = contrasive_data.get_input_dim()
class_dim = args.num_clusters

model = FullConnected(input_dim, class_dim, [128, 512, 256, 64]).to(args.device)
optimizer = optim.Adam(lr = args.lr, params=model.parameters())
label_optimizer = PseudoOptimizer(args)
label_optimizer = SinkhornLabelOptimizer(args)
trainer = HybridTrainer(model, optimizer, contrasive_data, noncontrasive_data, label_optimizer, args)
trainer.train()
print("Trainer finished, start evaluating...")
x, y = trainer.prepare_data_for_eval()
evaluator = Eval(args)
evaluator.eval(x, y)