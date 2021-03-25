import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
import sklearn
import pandas as pd

from utils.model_utils import add_layer


class Trainer:
    def __init__(self, *,
                 train_generator,
                 val_generator,
                 input_dim,
                 num_classes,
                 class_to_number=None,
                 architecture="conv_1d",
                 test_generator=None,
                 arch_params={}):

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.class_to_number = class_to_number

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator

        self.architecture = architecture
        self.arch_params = arch_params

    def delete_model(self):
        self.model = None

    def initialize_model(self, layer_channels=(512, 256), dropout_rate=0.,
                         learning_rate=1e-3, conv_size=5):

        inputs = layers.Input(self.input_dim)
        x = layers.BatchNormalization()(inputs)

        if self.architecture == 'lstm':
            lstm_size = self.arch_params.lstm_size
            x = layers.LSTM(lstm_size, activation='tanh')(x)

        for ch in layer_channels:
            x = add_layer(x, ch, drop=dropout_rate,
                          architecture=self.architecture,
                          arch_params=self.arch_params)
        x = layers.Flatten()(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)

        metrics = [tfa.metrics.F1Score(num_classes=self.num_classes)]
        optimizer = keras.optimizers.Adam(lr=learning_rate)

        model = Model(inputs, x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=metrics)

        self.model = model

    def _set_model(self, model):
        """ Set an external, provide initialized and compiled keras model """
        self.model = model

    def train(self, epochs=20, class_weight=None, callbacks=[]):
        if self.model is None:
            print("Please Call trainer.initialize_model first")
            return

        self.model.fit(self.train_generator,
                       validation_data=self.val_generator,
                       epochs=epochs,
                       class_weight=class_weight,
                       callbacks=callbacks)

    def get_generator_by_mode(self, mode='validation'):
        if mode == 'validation':
            return self.val_generator
        elif mode == 'train':
            return self.train_generator
        elif mode == 'test':
            return self.test_generator
        else:
            raise NotImplementedError

    def get_labels(self, generator):
        y_val = []
        for _, y in generator:
            y_val.extend(list(y))
        y_val = np.argmax(np.array(y_val), axis=-1)
        return y_val

    def get_predictions(self, generator):
        y_val_pred = self.model.predict(generator)
        y_val_pred = np.argmax(y_val_pred, axis=-1)
        return y_val_pred

    def get_metrics(self, mode='validation'):
        generator = self.get_generator_by_mode(mode)
        labels = self.get_labels(generator)
        predictions = self.get_predictions(generator)

        f1_scores = sklearn.metrics.f1_score(labels, predictions, average=None)
        rec_scores = sklearn.metrics.precision_score(
            labels, predictions, average=None)
        prec_scores = sklearn.metrics.recall_score(
            labels, predictions, average=None)
        classes = list(self.class_to_number.keys())
        metrics = pd.DataFrame({"Class": classes, "F1": f1_scores,
                               "Precision": prec_scores, "Recall": rec_scores})
        return metrics
