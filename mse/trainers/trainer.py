from tqdm import tqdm, trange
import torch
from mse.utils.loss import intra_feature_loss, intra_prob_loss, inter_feature_loss, prob_loss
from abc import ABC, abstractmethod

class Trainer(ABC):

    def __init__(self, model, optimizer, contrasive_data, noncontrasive_data, label_optimizer, args):
        self.model = model
        self.contrasive_data = contrasive_data
        self.noncontrasive_data = noncontrasive_data
        self.label_optimizer = label_optimizer
        self.args = args
        self.optimizer = optimizer
    
    def save(self, epoch=None):
        if epoch is None:
            torch.save(self.model.state_dict(), f"{self.args.save_dir}/final.pth")
        else:
            torch.save(self.model.state_dict(), f"{self.args.save_dir}/model_{epoch}.pth")

    def train(self):
        self.model.train()
        for ep in trange(self.args.epoch):
            self.train_epoch(ep)
            if self.args.save_checkpoint:
                self.save(ep)
    @abstractmethod
    def train_epoch(self, ep):
        # TODO: should be implemented in the child class
        pass

            
    @abstractmethod
    def eval(self):
        pass
class ContrasiveTrainer(Trainer):

    def train_epoch_contrasive(self):
        loss = 0
        self.optimizer.zero_grad()
        for (feat, label), (feat_shadow, label_shadow) in self.contrasive_data:
            feat, label = feat.to(self.args.device), label.to(self.args.device)
            feat_shadow, label_shadow = feat_shadow.to(self.args.device), label_shadow.to(self.args.device)
            emb = self.model.compute_feature(feat)
            emb_shadow = self.model.compute_feature(feat_shadow)
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

    def train_epoch_sinkhorn(self):
        self.noncontrasive_data.optimize(self.label_optimizer)
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
    
    def eval(self):
        raise NotImplementedError("ContrasiveTrainer does not support eval")


class HybridTrainer(ContrasiveTrainer, SinkhornTrainer):
    
    def train_epoch(self, ep):
        if ep < self.args.warmup_epoch:
            self.train_epoch_contrasive()
        else:
            if ep % 2 == 0:
                self.train_epoch_contrasive()
            else:
                self.train_epoch_sinkhorn()
    