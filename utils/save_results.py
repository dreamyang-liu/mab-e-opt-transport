import numpy as np


def save_results(results_dir, model, config, train_metrics, val_metrics):
    model.save(f'{results_dir}/task1_model.h5')
    np.save(f"{results_dir}/task1_config", config)
    val_metrics.to_csv(f"{results_dir}/task1_metrics_val.csv", index=False)
    train_metrics.to_csv(f"{results_dir}/task1_metrics_train.csv", index=False)
