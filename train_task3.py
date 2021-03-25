import os
from copy import deepcopy
from tensorflow import keras

from utils.load_data import load_mabe_data_task3
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
from utils.model_utils import copy_model_weights_except_last_layer


def train_task3(train_data_path, results_dir, config, pretrained_model_path):

    # Load the data
    train_dataset = load_mabe_data_task3(train_data_path)

    # Create directories if not present
    create_dirs([results_dir])

    for behavior in train_dataset:

        # Get individual behavior data
        dataset = train_dataset[behavior]['sequences']
        vocabulary = train_dataset[behavior]["vocabulary"]

        # Seed for reproducibilty
        seed_everything(config.seed)

        # Transpose last axis, used for augmentation and normalization
        dataset = transpose_last_axis(deepcopy(dataset))
        feature_dim = (2, 7, 2)

        # Normalize the x y coordinates
        if config.normalize:
            dataset = normalize_data(dataset)

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

        trainer = Trainer(train_generator=train_generator,
                          val_generator=val_generator,
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

        # We use class weights for different behaviors in Task 3
        class_weight = config.class_weights[behavior]
        class_weight = {i: v for i, v in enumerate(class_weight)}

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # Make trainer model as the pretrained model
            pretrained_model = keras.models.load_model(pretrained_model_path)

            # Copy weights upto last layer
            copy_model_weights_except_last_layer(trainer.model, pretrained_model)

            # Set linear probe training learning rate
            trainer.model.optimizer.learning_rate.assign(config.linear_probe_lr)

            # Freeze all layers except last layer
            freeze_model_except_last_layer(trainer.model)

            # Train linear probe
            trainer.train(epochs=config.linear_probe_epochs,
                          class_weight=class_weight)

            # Unfreeze all layers
            unfreeze_model_except_last_layer(trainer.model)

            # Set post training learning rate
            trainer.model.optimizer.learning_rate.assign(config.learning_rate)

        # Train model
        trainer.train(epochs=config.epochs, class_weight=class_weight)

        # Get metrics
        train_metrics = trainer.get_metrics(mode='train')
        val_metrics = trainer.get_metrics(mode='validation')

        # Save the results
        save_results(results_dir, f'task2_{behavior}',
                     trainer.model, config,
                     train_metrics, val_metrics)


if __name__ == '__main__':
    train_data_path = 'data/task3_train_data.npy'
    results_dir = 'results/task3_baseline'
    pretrained_model_path = 'results/task1_augmented/task1_model.h5'
    from configs.task3_baseline import task3_baseline_config
    config = task3_baseline_config
    train_task3(train_data_path, results_dir, config, pretrained_model_path)
