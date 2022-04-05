import argparse
import uuid
from easydict import EasyDict

from train_task1 import train_task1

from configs.task1_baseline import task1_baseline_config

task1_train_data_path = 'task1_npy/calms21_task1_train.npy'
task1_test_data_path = 'task1_npy/calms21_task1_test.npy'

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Seed value')

parser.add_argument('--val_size', type=float, default=0.2)

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')

parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')

parser.add_argument('--gpu_id', type=str, default='0')

parser.add_argument('--opt_label_period', type=int, default=1)

parser.add_argument('--label_method', type=str, default='kmeans')

hparams = parser.parse_args()

# fmod = hparams.filename_modifier or str(uuid.uuid4())

# Task 1 Baseline - 1D CNN
results_dir = 'results/task1_hparams/'
config = task1_baseline_config

config.seed = hparams.seed
# config.architecture_parameters.conv_size = hparams.conv_size
config.learning_rate = hparams.lr
# config.layer_channels = tuple(hparams.channels)
# config.past_frames = hparams.past_frames
# config.future_frames = hparams.future_frames
# config.frame_gap = hparams.frame_gap
# config.filename_modifier = fmod
config.val_size = hparams.val_size
config.dropout_rate = hparams.dropout_rate
config.epochs = hparams.epochs
config.gpu_id = hparams.gpu_id
config.opt_label_period = hparams.opt_label_period
config.label_method = hparams.label_method
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)
