import itertools
import torch
from sklearn.metrics import f1_score, accuracy_score
from mse.eval.eval_utils import DimReducer, ClusterCls
class Eval(object):
    
    def __init__(self, args):
        self.args = args
        self.dim_reducer = DimReducer(64, 'pca')
        self.cluster_cls = ClusterCls(self.args.eval_num_clusters)
    
    @classmethod
    def label_mapping(cls, label_num, prediction):
        """
        Map the labels to the permutation
        :param label_num: number of label classes
        :param prediction: prediction
        """
        permutations = list(itertools.permutations(range(label_num)))
        for permutation in permutations:
            mapped_pred = torch.zeros_like(prediction)
            for pseudo, true in enumerate(permutation):
                mapped_pred[prediction == pseudo] = true
            yield mapped_pred
    
    def eval(self, x, y):
        """
        Evaluate the model
        :param x: input data
        :param y: target data
        :return: loss
        """
        # Reduce the dimensionality of the data
        reduced_x = self.dim_reducer.fit(x)
        # Cluster the data
        y_ = self.cluster_cls.fit(reduced_x)
        # Evaluate the model
        y_permutations = self.label_mapping(self.args.eval_num_clusters, torch.tensor(y_, dtype=torch.long))
        max_f1 = 0
        max_acc = 0
        for y_permutation in y_permutations:
            acc = accuracy_score(y, y_permutation.numpy())
            f1 = f1_score(y, y_permutation.numpy(), average='macro')
            max_f1 = max(max_f1, f1)
            max_acc = max(max_acc, acc)
        print("*" * 30)
        print("Accuracy: {}".format(max_acc))
        print("F1: {}".format(max_f1))
        print("*" * 30)
        return max_f1, max_acc

if __name__ == "__main__":
    prediction = torch.tensor([0, 1, 1, 2])
    p = Eval.label_mapping(4, prediction)
    for i in p:
        print(i)