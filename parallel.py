from nlutils_mini.TrainingManager import TrainingManager
from nlutils_mini.ParameterWatcher import merge_files
# Baseline config - Convolution 1D

# def gpu_generator(scope):
#     idx = 0
#     while True:
#         yield scope[idx]
#         idx += 1
#         if idx >= len(scope):
#             idx = 0
# gpu_list = [0, 1, 2, 3, 4, 5]
# gpu_idx = gpu_generator(gpu_list)
# method = 'sinkhorn'
# base_cmd = "bash shell.sh "
# epochs = 200
# configs = []
# for label_period in [1, 2, 4, 8]:
#     for lr_rate in [1e-4, 5e-4, 1e-3]:
#         for dr in [0.1, 0.2, 0.5, 0.7]:
#             configs.append(f"{base_cmd} {lr_rate} {dr} {epochs} {next(gpu_idx)} {label_period} {method}")
# task_batch = 12
# batch_number = len(configs) // task_batch
# manager = TrainingManager()
# for i in range(batch_number):
#     manager.start_shell_training(configs[i*task_batch:(i+1)*task_batch])

merge_files()