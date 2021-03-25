import numpy as np


def save_results(results_dir, taskname, model, config, 
                 train_metrics, val_metrics, test_metrics=None):
    model.save(f'{results_dir}/{taskname}_model.h5')
    np.save(f"{results_dir}/{taskname}_config", config)
    val_metrics.to_csv(f"{results_dir}/{taskname}_metrics_val.csv", index=False)
    train_metrics.to_csv(f"{results_dir}/{taskname}_metrics_train.csv", index=False)
    if test_metrics is not None:
        test_metrics.to_csv(f"{results_dir}/{taskname}_metrics_test.csv", index=False)
