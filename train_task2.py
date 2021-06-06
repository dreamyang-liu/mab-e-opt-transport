import os
import argparse
from copy import deepcopy
import tensorflow as tf

from utils.load_data import load_mabe_data_task2
from utils.dirs import create_dirs
from utils.preprocessing import normalize_data, transpose_last_axis
from utils.split_data import split_data
from utils.seeding import seed_everything
from trainers.mab_e_trainer import Trainer
from data_generator.mab_e_data_generator import mabe_generator
from data_generator.mab_e_data_generator import calculate_input_dim
from utils.save_results import save_results
from utils.model_utils import freeze_model_except_last_layer
from utils.model_utils import unfreeze_model_except_last_layer


def train_task2(train_data_path, results_dir, config,
                pretrained_model_path, test_data_path):

    # Load the data
    train_dataset, vocabulary = load_mabe_data_task2(train_data_path)
    test_dataset, _ = load_mabe_data_task2(test_data_path)

    # Create directories if not present
    create_dirs([results_dir])

    # The dataset has multiple annotator ids
    # We train for each annotator separately
    for annotator_id in train_dataset:
        # Get individual annotator data
        dataset = train_dataset[annotator_id]
        test_data = test_dataset[annotator_id]

        # Seed for reproducibilty
        seed_everything(config.seed)

        # Transpose last axis, used for augmentation and normalization
        dataset = transpose_last_axis(deepcopy(dataset))
        test_data = transpose_last_axis(deepcopy(test_data))
        feature_dim = (2, 7, 2)

        # Normalize the x y coordinates
        if config.normalize:
            dataset = normalize_data(dataset)
            test_data = normalize_data(test_data)

        # Split the data
        train_data, val_data = split_data(dataset,
                                          seed=config.seed,
                                          vocabulary=vocabulary,
                                          test_size=config.val_size,
                                          split_videos=config.split_videos)
        num_classes = len(vocabulary)

        # Calculate the input dimension based on past and future frames
        # Also flattens the channels as required by the architecture
        input_dim = calculate_input_dim(feature_dim,
                                        config.architecture,
                                        config.past_frames,
                                        config.future_frames)

        # Initialize data generators
        common_kwargs = {"batch_size": config.batch_size,
                         "input_dimensions": input_dim,
                         "past_frames": config.past_frames,
                         "future_frames": config.future_frames,
                         "class_to_number": vocabulary,
                         "frame_skip": config.frame_gap}
        train_generator = mabe_generator(train_data,
                                         augment=config.augment,
                                         shuffle=True,
                                         kwargs=common_kwargs)
        val_generator = mabe_generator(val_data,
                                       augment=False,
                                       shuffle=False,
                                       kwargs=common_kwargs)

        test_generator = mabe_generator(test_data,
                                        augment=False,
                                        shuffle=False,
                                        kwargs=common_kwargs)

        trainer = Trainer(train_generator=train_generator,
                          val_generator=val_generator,
                          test_generator=test_generator,
                          input_dim=input_dim,
                          class_to_number=vocabulary,
                          num_classes=num_classes,
                          architecture=config.architecture,
                          arch_params=config.architecture_parameters)

        # Model initialization
        trainer.initialize_model(layer_channels=config.layer_channels,
                                 dropout_rate=config.dropout_rate,
                                 learning_rate=config.learning_rate)

        # Print the model
        trainer.model.summary()

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # Make trainer model as the pretrained model
            trainer.model = tf.keras.models.load_model(pretrained_model_path)

            # Get zero shot trainer metrics
            seed = config.seed
            taskname = f'task2_annotator{annotator_id}_seed_{seed}'
            test_metrics_zs = trainer.get_metrics(mode='test')
            test_metrics_zs.to_csv(f'{results_dir}/{taskname}_test_results_zeroshot.csv',
                                   index=False)
            # Freeze all layers except last layer
            freeze_model_except_last_layer(trainer.model)

            # Train linear probe
            trainer.train(epochs=config.linear_probe_epochs)

            # Unfreeze all layers
            unfreeze_model_except_last_layer(trainer.model)

        # Train model
        trainer.train(epochs=config.epochs)

        # Get metrics
        train_metrics = trainer.get_metrics(mode='train')
        val_metrics = trainer.get_metrics(mode='validation')
        test_metrics = trainer.get_metrics(mode='test')

        # Save the results
        save_results(results_dir, f'task2_annotator{annotator_id}',
                     trainer.model, config,
                     train_metrics, val_metrics, test_metrics)


if __name__ == '__main__':
    train_data_path = 'data/task2_train_data.npy'
    results_dir = 'results/task2_baseline'
    test_data_path = 'data/task2_test_ground_truth.npy'

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed value')
    parser.add_argument('--full-train-set', action="store_true",
                        help='Train on 100% of the training set')

    seed = parser.parse_args().seed

    model_path = f'{results_dir}/task1_seed_{seed}_model.h5'
    pretrained_model_path = model_path if os.path.exists(model_path) else None

    from configs.task2_baseline import task2_baseline_config
    config = task2_baseline_config
    config.seed = seed
    train_task2(train_data_path, results_dir, config,
                pretrained_model_path, test_data_path)
