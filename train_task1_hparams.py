import argparse
import uuid

from train_task1 import train_task1

from configs.task1_baseline import task1_baseline_config

task1_train_data_path = 'data/calms21_task1_train.npy'
task1_test_data_path = 'data/calms21_task1_test.npy'

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Seed value')

parser.add_argument('--conv-size', type=int, default=5,
                    help='1D Convolution Kernel Size')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')

parser.add_argument('--channels', nargs='+', type=int,
                    default=[128, 64, 32],
                    help='Channels in each layer')

parser.add_argument('--past-frames', type=int, default=100,
                    help='Number of past frames')
parser.add_argument('--future-frames', type=int, default=100,
                    help='Number of future frames')
parser.add_argument('--frame-gap', type=int, default=2,
                    help='Gap Between frames - 1 means no gap')

parser.add_argument('--filename-modifier',
                    help='Name to be added to the saved files')

hparams = parser.parse_args()

fmod = hparams.filename_modifier or str(uuid.uuid4())

# Task 1 Baseline - 1D CNN
results_dir = 'results/task1_hparams/'
config = task1_baseline_config

config.seed = hparams.seed
config.architecture_parameters.conv_size = hparams.conv_size
config.learning_rate = hparams.lr
config.layer_channels = tuple(hparams.channels)
config.past_frames = hparams.past_frames
config.future_frames = hparams.future_frames
config.frame_gap = hparams.frame_gap
config.filename_modifier = fmod

train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)
