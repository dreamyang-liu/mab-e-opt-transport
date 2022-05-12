from tqdm import tqdm, trange
from mse.utils.loss import intra_feature_loss, intra_prob_loss, inter_feature_loss, prob_loss
from abc import ABC, abstractmethod
import torch
import os
import shutil
class Trainer(ABC):

    def __init__(self, model, optimizer, contrasive_data, noncontrasive_data, label_optimizer, args):
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
                progress.set_description(f"sinkhorn epoch")
                self.train_epoch_sinkhorn()
    
    def eval(self):
        raise NotImplementedError("HybridTrainer does not support eval")