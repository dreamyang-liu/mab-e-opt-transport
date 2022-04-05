from easydict import EasyDict

# Baseline config - Convolution 1D
task1_baseline_config = {"seed": 42,
                         "val_size": 0.2,
                         "split_videos": False,
                         "normalize": True,
                         "past_frames": 100,
                         "future_frames": 100,
                         "frame_gap": 2,
                         "architecture": "conv_1D",
                         "architecture_parameters": EasyDict({"conv_size": 5}),
                         "batch_size": 128,
                         "learning_rate": 2e-4,
                         "dropout_rate": 0.5,
                         "layer_channels": (128, 64, 32),
                         "epochs": 800,
                         "augment": False,
                         "opt_label_period": 1,
                         "gpu_id": "4"}

task1_baseline_config = EasyDict(task1_baseline_config)
