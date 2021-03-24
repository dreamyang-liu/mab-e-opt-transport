from easydict import EasyDict
from configs.task1_baseline import task1_baseline_config
from copy import deepcopy

# Augmented conv1D config, uses multiple fully connected layers
task1_augment_config = deepcopy(task1_baseline_config)
task1_augment_config.architecture = "conv_1D"
task1_augment_config.layer_channels = (128, 64, 32)
task1_augment_config.augment = True
task1_augment_config = EasyDict(task1_augment_config)

# LSTM config, uses one lstm layer and fully connected layers afer that
task1_lstm_config = deepcopy(task1_baseline_config)
task1_lstm_config.architecture = "lstm"
task1_lstm_config.architecture_parameters = EasyDict({"lstm_size": 256})
task1_lstm_config.layer_channels = (256, 128)
task1_lstm_config = EasyDict(task1_lstm_config)

# Attention config, uses multiple layers of self attention
task1_attention_config = deepcopy(task1_baseline_config)
task1_attention_config.architecture = "attention"
task1_attention_config.layer_channels = (128, 64, 32)
task1_attention_config = EasyDict(task1_attention_config)

# Fully connected config, uses multiple fully connected layers
task1_fc_config = deepcopy(task1_baseline_config)
task1_fc_config.architecture = "fully_connected"
task1_fc_config.layer_channels = (512, 256, 128)
task1_fc_config.learning_rate = 3e-4
task1_fc_config = EasyDict(task1_fc_config)
