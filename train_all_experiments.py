import argparse

from train_task1 import train_task1
from train_task2 import train_task2
from train_task3 import train_task3

from configs.task1_baseline import task1_baseline_config
from configs.experiments import (task1_augmented_config,
                                 task1_lstm_config,
                                 task1_attention_config,
                                 task1_fc_config,
                                 task1_singleframe_config,
                                 task1_causal_config)
from configs.task2_baseline import task2_baseline_config
from configs.task3_baseline import task3_baseline_config

task1_train_data_path = 'data/calms21_task1_train.npy'
task1_test_data_path = 'data/calms21_task1_test.npy'

task2_train_data_path = 'data/calms21_task2_train.npy'
task2_test_data_path = 'data/calms21_task2_test.npy'

task3_train_data_path = 'data/calms21_task3_train.npy'
task3_test_data_path = 'data/calms21_task3_test.npy'

# Parse for seed value
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Seed value')

seed = parser.parse_args().seed

# Task 1 Baseline with 100% train data - 1D CNN
results_dir = 'results/task1_train_full_data'
config = task1_baseline_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)

# # Task 1 Augmented with 100% train data
results_dir = 'results/task1_augmented_train_full_data'
config = task1_augmented_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)

# Task 2 Baseline with 100% train data
results_dir = 'results/task2_train_full_data'
pretrained_model_path = f'results/task1_augmented_train_full_data/task1_seed_{seed}_model.h5'
config = task2_baseline_config
config.seed = seed
config.val_size = 0.0
train_task2(task2_train_data_path, results_dir, config,
            pretrained_model_path, task2_test_data_path)

# Task 3 Baseline with 100% train data
results_dir = 'results/task3_train_full_data'
pretrained_model_path = f'results/task1_augmented_train_full_data/task1_seed_{seed}_model.h5'
config = task3_baseline_config
config.seed = seed
config.val_size = 0.0
train_task3(task3_train_data_path, results_dir, config,
            pretrained_model_path, task3_test_data_path)

# Task 1 Fully connected with 100% train data
results_dir = 'results/task1_fc_train_full_data'
config = task1_fc_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)

# Task 1 Attention with 100% train data
results_dir = 'results/task1_attention_train_full_data'
config = task1_attention_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)

# Task 1 LSTM with 100% train data
results_dir = 'results/task1_lstm_train_full_data'
config = task1_lstm_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)

# Task 1 Single Frame model with 100% train data
results_dir = 'results/task1_singleframe_train_full_data'
config = task1_singleframe_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)

# Task 1 Causal Model with 100% train data
results_dir = 'results/task1_causal_train_full_data'
config = task1_causal_config
config.seed = seed
config.val_size = 0.0
train_task1(task1_train_data_path, results_dir, config, task1_test_data_path)
