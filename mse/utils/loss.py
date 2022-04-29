import torch
import torch.nn.functional as F


def intra_feature_loss(emb_1, emb_2):
    """
    Calculate the intra-class feature loss
    :param emb_1: embedding of class 1
    :param emb_2: embedding of class 2
    :return: loss
    """
    # Calculate the distance between the two embeddings
    dist = torch.dist(emb_1, emb_2)
    # Calculate the loss
    loss = torch.mean(dist)
    return loss

def inter_feature_loss(emb_1, emb_2):
    """
    Calculate the inter-class feature loss
    :param emb_1: embedding of class 1
    :param emb_2: embedding of class 2
    :return: loss
    """
    # Calculate the distance between the two embeddings
    dist = torch.dist(emb_1, emb_2)
    # Calculate the loss
    loss = -torch.mean(dist)
    return loss

def intra_prob_loss(pred_1, pred_2):
    """
    Calculate the intra-class probability loss
    :param pred_1: prediction of class 1
    :param pred_2: prediction of class 2
    :return: loss
    """
    # Calculate the distance between the two predictions
    loss = -F.cross_entropy(pred_1, pred_2)
    return loss

def inter_prob_loss(pred_1, pred_2):
    """
    Calculate the inter-class probability loss
    :param pred_1: prediction of class 1
    :param pred_2: prediction of class 2
    :return: loss
    """
    # Calculate the distance between the two predictions
    loss = F.cross_entropy(pred_1, pred_2)
    return loss

def prob_loss(pred, target):
    """
    Calculate the probability loss
    :param pred: prediction
    :param target: target
    :return: loss
    """
    # Calculate the distance between the two predictions
    loss = F.cross_entropy(pred, target)
    return loss