from easydict import EasyDict
from configs.task1_baseline import task1_baseline_config
from copy import deepcopy

class_weights = {"behavior-0": [1, 20],
                 "behavior-1": [1, 50],
                 "behavior-2": [1, 5],
                 "behavior-3": [1, 3],
                 "behavior-4": [1, 100],
                 "behavior-5": [1, 20],
                 "behavior-6": [1, 10],
                 }

# Task3 uses pretrained model from task1 and replaces the top layer
task3_baseline_config = deepcopy(task1_baseline_config)
task3_baseline_config.architecture = "conv_1D"
task3_baseline_config.layer_channels = (128, 64, 32)
task3_baseline_config.augment = True
task3_baseline_config.linear_probe_lr = 1e-3
task3_baseline_config.linear_probe_epochs = 30
task3_baseline_config.learning_rate = 5e-5
task3_baseline_config.epochs = 20
task3_baseline_config.class_weights = class_weights
task3_baseline_config = EasyDict(task3_baseline_config)
