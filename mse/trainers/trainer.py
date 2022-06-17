from tqdm import tqdm, trange
from mse.utils.loss import intra_feature_loss, intra_prob_loss, inter_feature_loss, prob_loss
from abc import ABC, abstractmethod
import torch
import os
import shutil
from sklearn.metrics import f1_score, accuracy_score
class Trainer(ABC):

    def __init__(self, model, optimizer, contrasive_data=None, noncontrasive_data=None, label_optimizer=None, args=None):
        self.model = model
        self.contrasive_data = contrasive_data
        self.noncontrasive_data = noncontrasive_data
        self.label_optimizer = label_optimizer
        self.args = args
        self.optimizer = optimizer
    
    def save(self, epoch=None):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        else:
            shutil.rmtree(self.args.save_dir)
            os.makedirs(self.args.save_dir)
        if epoch is None:
            torch.save(self.model.state_dict(), f"{self.args.save_dir}/final.pth")
        else:
            torch.save(self.model.state_dict(), f"{self.args.save_dir}/model_{epoch}.pth")

    def train(self):
        self.model.train()
        with trange(self.args.epoch) as progress:
            for ep in progress:
                self.train_epoch(ep, progress)
                if self.args.save_checkpoint:
                    self.save(ep)

    def prepare_data_for_eval(self):
        # prepare feature extracted from backbone
        feats = []
        labels = []
        with torch.no_grad():
            self.noncontrasive_data.reset()
            for (feat, label) in self.noncontrasive_data:
                feat, label = feat.to(self.args.device), label.to(self.args.device)
                emb = self.model.compute_features(feat)
                feats.append(emb)
                labels.append(label)
        feats = torch.cat(feats, 0).cpu().numpy()
        labels = torch.cat(labels, 0).cpu().numpy()
        return feats, labels

    @abstractmethod
    def train_epoch(self, ep):
        # TODO: should be implemented in the child class
        pass
            
    @abstractmethod
    def eval(self):
        pass
class ContrasiveTrainer(Trainer):

    def train_epoch_contrasive(self):
        assert self.contrasive_data is not None, 'contrasive trainer requires contrasive data'
        loss = 0
        self.optimizer.zero_grad()
        for (feat, label), (feat_shadow, label_shadow) in self.contrasive_data: 
            feat, label = feat.to(self.args.device), label.to(self.args.device)
            feat_shadow, label_shadow = feat_shadow.to(self.args.device), label_shadow.to(self.args.device)
            breakpoint()
            emb = self.model.compute_features(feat)
            emb_shadow = self.model.compute_features(feat_shadow)
            loss_feat_intra = intra_feature_loss(emb, emb_shadow)
            loss_feat_inter = inter_feature_loss(emb, emb_shadow)
            prob = self.model.compute_probability_via_feature(emb)
            prob_shadow = self.model.compute_probability_via_feature(emb_shadow)
            loss_prob_intra = intra_prob_loss(prob, prob_shadow)
            loss += loss_feat_inter + loss_prob_intra + loss_feat_intra
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def eval(self):
        raise NotImplementedError("ContrasiveTrainer does not support eval")

class SinkhornTrainer(Trainer):

    def get_ps(self):
        probs = []
        with torch.no_grad():
            self.noncontrasive_data.reset()
            for (feat, label) in self.noncontrasive_data:
                feat, label = feat.to(self.args.device), label.to(self.args.device)
                emb = self.model.compute_feature(feat)
                prob = self.model.compute_probability_via_feature(emb)
                probs.append(prob)
        probs = torch.cat(probs, 0)
        return probs


    def train_epoch_sinkhorn(self):
        assert self.label_optimizer is not None, 'sinkhorn trainer requires label optimizer'
        self.noncontrasive_data.optimize(self.label_optimizer, self.get_ps())
        loss = 0
        self.optimizer.zero_grad()
        for (feat, label) in self.noncontrasive_data:
            feat, label = feat.to(self.args.device), label.to(self.args.device)
            emb = self.model.compute_feature(feat)
            prob = self.model.compute_probability_via_feature(emb)
            loss += prob_loss(prob, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_epoch(self):
        assert self.label_optimizer is not None, 'sinkhorn trainer requires label optimizer'
        return self.train_epoch_sinkhorn()
    
    def eval(self):
        raise NotImplementedError("ContrasiveTrainer does not support eval")


class HybridTrainer(ContrasiveTrainer, SinkhornTrainer):
    
    def train_epoch(self, ep, progress):
        if ep < self.args.warmup_epoch:
            progress.set_description(f"warmup epoch")
            self.train_epoch_contrasive()
        else:
            if ep % 2 == 0:
                progress.set_description(f"contrasive epoch")
                self.train_epoch_contrasive()
            else:
                progress.set_description(f"sinkhorn epoch {ep}")
        self.train_epoch_sinkhorn()
        
    
    def eval(self):
        raise NotImplementedError("HybridTrainer does not support eval")

class FullSupervisedTrainer(Trainer):
    def train_epoch(self, ep, progress):
        loss = 0
        self.optimizer.zero_grad()
        for (feat, label) in self.noncontrasive_data:
            feat, label = feat.to(self.args.device), label.to(self.args.device)
            emb = self.model.compute_features(feat)
            prob = self.model.compute_probability_via_feature(emb)
            loss += prob_loss(prob, label)
        loss.backward()
        self.optimizer.step()
        progress.set_description(f"full supervised epoch {ep}: {loss.item()}")
        return loss.item()
    
    def prepare_data_for_eval(self):
        # prepare feature extracted from backbone
        feats = []
        labels = []
        with torch.no_grad():
            self.noncontrasive_data.reset()
            for (feat, label) in self.noncontrasive_data:
                feat, label = feat.to(self.args.device), label.to(self.args.device)
                emb = self.model.compute_feature(feat)
                feats.append(emb)
                labels.append(label)
        feats = torch.cat(feats, 0).cpu().numpy()
        labels = torch.cat(labels, 0).cpu().numpy()
        return feats, labels
        
    def eval(self):
        raise NotImplementedError("FullSupervisedTrainer does not support eval")

class Conv1dTrainer(Trainer):
    def train_epoch(self, ep, progress):
        loss = 0
        self.model.train()
        for (feat, label) in self.noncontrasive_data:
            self.optimizer.zero_grad()
            feat, label = feat.to(self.args.device), label.to(self.args.device)
            emb = self.model.compute_features(feat)
            prob = self.model.compute_probability_via_feature(emb)
            loss = prob_loss(prob, label)
            loss.backward()
            self.optimizer.step()
        progress.set_description(f"conv1d epoch {ep}: {loss.item()}")
        return loss.item()

    def train(self, noncontrasive_data_test):
        self.model.train()
        with trange(self.args.epoch) as progress:
            for ep in progress:
                self.train_epoch(ep, progress)
                if ep % 3 ==0:
                    self.eval(noncontrasive_data_test)
                if self.args.save_checkpoint:
                    self.save(ep)

    def eval(self, noncontrasive_data_test):
        acc_list = []
        f1_list = []

        self.model.eval()
        with torch.no_grad():
            for (feat, label) in noncontrasive_data_test:
                feat, label = feat.to(self.args.device), label.to(self.args.device)
                y_pred = self.model.forward(feat)
                label_pred = torch.argmax(y_pred, 1)
                acc_list.append(accuracy_score(label.cpu().numpy(), label_pred.cpu().numpy()))
                f1_list.append(f1_score(label.cpu().numpy(), label_pred.cpu().numpy(), average='macro'))

        print ( 'Average Accuracy: {:.5f}'.format(sum(acc_list)/len(acc_list)))
        print ( 'Average F1: {:.5f}'.format(sum(f1_list)/len(f1_list)))
        # print("*" * 30)
        # print("Accuracy: {}".format(max_acc))
        # print("F1: {}".format(max_f1))
        # print("*" * 30)



class Conv1dContrasiveTrainer(Conv1dTrainer, ContrasiveTrainer):
    def train_epoch(self, ep, progress):
        loss = 0
        self.model.train()
        for (feat, label) in self.noncontrasive_data:
            self.optimizer.zero_grad()
            feat, label = feat.to(self.args.device), label.to(self.args.device)
            emb = self.model.compute_features(feat)
            prob = self.model.compute_probability_via_feature(emb)
            loss = prob_loss(prob, label)
            loss.backward()
            self.optimizer.step()
        progress.set_description(f"conv1d epoch {ep}: {loss.item()}")
        return loss.item()

    def train(self, noncontrasive_data_test):
        self.model.train()
        with trange(self.args.epoch) as progress:
            for ep in progress:
                self.train_epoch_contrasive()
                self.train_epoch(ep, progress)
                if ep % 3 ==0:
                    self.eval(noncontrasive_data_test)
                if self.args.save_checkpoint:
                    self.save(ep)

    def eval(self, noncontrasive_data_test):
        acc_list = []
        f1_list = []

        self.model.eval()
        with torch.no_grad():
            for (feat, label) in noncontrasive_data_test:
                feat, label = feat.to(self.args.device), label.to(self.args.device)
                y_pred = self.model.forward(feat)
                label_pred = torch.argmax(y_pred, 1)
                acc_list.append(accuracy_score(label.cpu().numpy(), label_pred.cpu().numpy()))
                f1_list.append(f1_score(label.cpu().numpy(), label_pred.cpu().numpy(), average='macro'))

        print ( 'Average Accuracy: {:.5f}'.format(sum(acc_list)/len(acc_list)))
        print ( 'Average F1: {:.5f}'.format(sum(f1_list)/len(f1_list)))
        # print("*" * 30)
        # print("Accuracy: {}".format(max_acc))
        # print("F1: {}".format(max_f1))
        # print("*" * 30)