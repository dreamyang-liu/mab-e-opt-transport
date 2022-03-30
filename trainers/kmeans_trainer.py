from .mab_e_trainer import Trainer as BaseTrainer
from models.Models import Conv1d_Model
from nlutils_mini.Log import default_logger
from tqdm import tqdm, trange
from .pseudo_label_generator.kmeans import KMeansLabelGenerator
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import sklearn.metrics as skm
import itertools

class KMeansTrainer(BaseTrainer):


    def initialize_model(self, layer_channels=(512, 256), dropout_rate=0., learning_rate=1e-3, conv_size=5):
        self.log_time = time.time()
        arch_params = self.arch_params
        arch_params.conv_channels = layer_channels
        arch_params.drop_ratio = dropout_rate
        model = Conv1d_Model(self.input_dim, self.num_classes, self.arch_params)


        metrics = [tfa.metrics.F1Score(num_classes=self.num_classes)]
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        model.build(input_shape=(None, self.input_dim[0], self.input_dim[1]))
        model.compile(optimizer=self.optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=metrics)
        self.model = model

    def optimize_label_assignment(self):
        default_logger.warn(f"Optimize label start...")
        feats = []
        for x, _ in self.train_generator:
            feat = self.model.compute_features(x)
            feats.append(feat)
        feats =  tf.concat(feats, axis=0).cpu()
        clf = KMeansLabelGenerator(self.num_classes, use_pca=True, n_components=256)
        labels = clf.fit(feats)
        self.train_generator.update_y(labels)
        default_logger.warn(f"Optimize label finished...")
    
    def generate_label_maps(self, pred):
        permutations = list(itertools.permutations([0, 1, 2, 3], 4))
        preds = [np.zeros_like(pred) for _ in permutations]
        for idx, perm in enumerate(permutations):
            for pseudo, true in enumerate(perm):
                # import pdb; pdb.set_trace()
                preds[idx][pred == pseudo] = true
        return preds
    
    def validate_on_train(self):
        """ Validate the model on the training set """
        if self.model is None:
            print("Please Call trainer.initialize_model first")
            return
        preds = []
        for x, _ in self.train_generator:
            pred = self.model(x)
            pred = tf.argmax(pred, axis=-1)
            preds.append(pred)
        preds = tf.concat(preds, axis=0).cpu().numpy()
        preds = self.generate_label_maps(preds)
        true_labels = self.train_generator.get_true_labels()
        accs = []
        f1_scores = []
        for pred in preds:
            accs.append(skm.accuracy_score(true_labels, pred))
            f1_scores.append(skm.f1_score(true_labels, pred, average='macro'))
        with open(f"./validate_on_train.txt_{self.log_time}", "a") as f:
            idx = np.argmax(f1_scores)
            self.optimal_permuation = list(itertools.permutations([0, 1, 2, 3], 4))[idx]
            f.write(f"{accs[idx]} {f1_scores[idx]}\n")
    
    def map_labels(self, permutation, prediction):
        """ Map the labels to the permutation """
        mapped_pred = np.zeros_like(prediction)
        for pseudo, true in enumerate(permutation):
            mapped_pred[prediction == pseudo] = true
        return mapped_pred
    
    def validate_on_test(self):
        """ Validate the model on the training set """
        if self.model is None:
            print("Please Call trainer.initialize_model first")
            return
        default_logger.warn(f"Validate on test start...")
        preds = []
        for x, _ in self.test_generator:
            pred = self.model(x)
            pred = tf.argmax(pred, axis=-1)
            preds.append(pred)
        preds = tf.concat(preds, axis=0).cpu().numpy()
        preds = self.map_labels(self.optimal_permuation, preds)
        true_labels = self.test_generator.y
        accs = []
        f1_scores = []
        accs.append(skm.accuracy_score(true_labels, preds))
        f1_scores.append(skm.f1_score(true_labels, preds, average='macro'))
        with open(f"./validate_on_test.txt_{self.log_time}", "a") as f:
            idx = np.argmax(f1_scores)
            f.write(f"{accs[idx]} {f1_scores[idx]}\n")
        default_logger.warn(f"Validate on test finished...")
        

    def train(self, epochs=20, class_weight=None, callbacks=[]):
        """ Train the model for given epochs """
        if self.model is None:
            print("Please Call trainer.initialize_model first")
            return
        
        for ep in range(epochs):
            if (ep + 1) % self.opt_label_period == 0:
                self.optimize_label_assignment()
            self.model.fit(self.train_generator, epochs=1)
            self.validate_on_train()
            self.validate_on_test()