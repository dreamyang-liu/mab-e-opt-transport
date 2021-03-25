from easydict import EasyDict
from configs.task1_baseline import task1_baseline_config
from copy import deepcopy

# Task2 uses pretrained model with linear probe and then further fine tuning
task2_baseline_config = deepcopy(task1_baseline_config)
task2_baseline_config.split_videos = True
task2_baseline_config.architecture = "conv_1D"
task2_baseline_config.layer_channels = (128, 64, 32)
task2_baseline_config.augment = True
task2_baseline_config.linear_probe_epochs = 5
task2_baseline_config.epochs = 10
task2_baseline_config = EasyDict(task2_baseline_config)