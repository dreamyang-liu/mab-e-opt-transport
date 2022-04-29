from mse.models.full_connected import FullConnected
from mse.trainers.trainer import ContrasiveTrainer
from mse.data_generator.data_generator import DataUtils, ContrasiveLearningDataset
from mse.utils.argparser import args

import torch.optim as optim
print(args)
train = DataUtils.read_npy('./data/train.npy', flatten=True)
train = DataUtils.build_contrasive_learning(train)
dataset_train = ContrasiveLearningDataset(*train, args)

input_dim = dataset_train.get_input_dim()
class_dim = dataset_train.get_class_dim()

model = FullConnected(input_dim, class_dim, [128, 512, 256, 64]).to(args.device)
optimizer = optim.Adam(lr = args.lr, params=model.parameters())

trainer = ContrasiveTrainer(model, optimizer, dataset_train, None, None, args)

trainer.train()