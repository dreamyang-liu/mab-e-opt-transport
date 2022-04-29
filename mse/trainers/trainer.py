from tqdm import tqdm, trange
import torch
from mse.utils.loss import intra_feature_loss, intra_prob_loss, inter_feature_loss
from abc import ABC, abstractmethod

class Trainer(ABC):

    def __init__(self, model, optimizer, train_data_generator, eval_data_generator, label_optimizer, args):
        self.model = model
        self.train_data_generator = train_data_generator
        self.eval_data_generator = eval_data_generator
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
            self.train_data_generator.optimize(self.label_optimizer)
            self.train_epoch()
            if self.args.save_checkpoint:
                self.save(ep)
            # TODO shuftle the data
    @abstractmethod
    def train_epoch(self):
        # TODO: should be implemented in the child class
        pass

            
    @abstractmethod
    def eval(self):
        pass

class ContrasiveTrainer(Trainer):

    def train_epoch(self):
        loss = 0
        self.optimizer.zero_grad()
        for (feat, label), (feat_shadow, label_shadow) in self.train_data_generator:
            self.model.zero_grad()
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