import tensorflow as tf
import torch
import numpy as np
import time

class PseudoOptimizer(object):

    def __init__(self, args):
        self.args = args
    
    def optimize(self, payload):
        return payload
class SinkhornLabelOptimizer(object):

    def __init__(self, args):
        self.args = args
    
    def optimize_single(self, PS):
        assert len(PS.shape) == 2, "PS should be 2D"
        PS = PS.to(self.args.sinkhorn_device)
        PS = PS.double()
        N, K = PS.shape
        if self.args.dist == 'gauss':
            _K_dist = (torch.randn(size=(K, 1), dtype=torch.float64, device=self.args.sinkhorn_device) * self.args.gauss_sd + 1) * N / K
        elif self.args.dist == 'uniform':
            _K_dist = torch.ones(size=(K, 1), dtype=torch.float64, device=self.args.sinkhorn_device) * N / K
        else:
            raise NotImplementedError(f"Distribution {self.args.dist} not implemented")
        beta = torch.ones((N, 1), dtype=torch.float64, device=self.args.sinkhorn_device) / N
        PS.pow_(0.5 * self.args.lamb)
        r = 1./_K_dist
        r /= r.sum()

        c = 1./N
        err = 1e6
        _counter = 0

        ones = torch.ones(N, device=self.args.sinkhorn_device, dtype=torch.float64)
        while (err > 1e-2) and (_counter < 2000):
            alpha = r / torch.matmul(beta.t(), PS).t()
            beta_new = c / torch.matmul(PS, alpha)
            if _counter % 10 == 0:
                err = torch.sum(torch.abs((beta.squeeze() / beta_new.squeeze()) - ones)).cpu().item()
            beta = beta_new
            _counter += 1

        # inplace calculations
        torch.mul(PS, beta, out=PS)
        torch.mul(alpha.t(), PS, out=PS)
        newL = torch.argmax(PS, 1).to(self.args.device)
        return newL
    
    def optimize(self, PS_list):
        if isinstance(PS_list, list):
            L = []
            for PS in PS_list:
                L.append(self.optimize_single(PS))
        else:
            L = self.optimize_single(PS_list)
        return L

def optimize_L_sk_gpu(args, PS, hc):
    print('doing optimization now',flush=True)

    # create L
    N = PS.size(0)
    K = PS.size(1)
    tt = time.time()
    _K_dist = torch.ones((K, 1), dtype=torch.float64, device='cuda') # / K
    if args.distribution != 'default':
        marginals_argsort = torch.argsort(PS.sum(0))
        if (args.dist is None) or args.diff_dist_every:
            if args.distribution == 'gauss':
                if args.diff_dist_per_head:
                    _K_dists = [(torch.randn(size=(K, 1), dtype=torch.float64, device='cuda')*args.gauss_sd + 1) * N / K
                                for _ in range(args.headcount)]
                    args.dist = _K_dists
                    _K_dist = _K_dists[hc]
                else:
                    _K_dist = (torch.randn(size=(K, 1), dtype=torch.float64, device='cuda')*args.gauss_sd + 1) * N / K
                    _K_dist = torch.clamp(_K_dist, min=1)
                    args.dist = _K_dist
        else:
            if args.diff_dist_per_head:
                _K_dist = args.dist[hc]
            else:
                _K_dist = args.dist
        _K_dist[marginals_argsort] = torch.sort(_K_dist)[0]

    beta = torch.ones((N, 1), dtype=torch.float64, device='cuda') / N
    PS.pow_(0.5*args.lamb)
    r = 1./_K_dist
    r /= r.sum()

    c = 1./N
    err = 1e6
    _counter = 0

    ones = torch.ones(N, device='cuda:0', dtype=torch.float64)
    while (err > 1e-1) and (_counter < 2000):
        alpha = r / torch.matmul(beta.t(), PS).t()
        beta_new = c / torch.matmul(PS, alpha)
        if _counter % 10 == 0:
            err = torch.sum(torch.abs((beta.squeeze() / beta_new.squeeze()) - ones)).cpu().item()
        beta = beta_new
        _counter += 1

    # inplace calculations
    torch.mul(PS, beta, out=PS)
    torch.mul(alpha.t(), PS, out=PS)
    newL = torch.argmax(PS, 1).cuda()

    # return back to obtain cost (optional)
    torch.mul((1./alpha).t(), PS, out=PS)
    torch.mul(PS, 1./beta, out=PS)
    sol = np.nansum(torch.log(PS[torch.arange(0, len(newL)).long(), newL]).cpu().numpy())
    cost = -(1. / args.lamb) * sol / N
    return cost, newL

########################################################################################################################
# Below is the deprecated code

def modified_optimize_L_sk(PS_all):
    for PS in PS_all:
        N, K = PS.shape
        marginals_argsort = np.argsort(PS.sum(0))
        _K_dist = np.array([0.627, 0.289, 0.056, 0.028])
        beta = np.ones((N, 1), dtype=np.float64) / N
        PS **= LAMBDA
        _K_dist[marginals_argsort] = np.sort(_K_dist)
        r = _K_dist.reshape(-1, 1)

        c = 1./N
        err = 1e6
        _counter = 0

        ones = np.ones(N, dtype=np.float64)
        while (err > 1e-2) and (_counter < 2000):
            alpha = r / np.dot(beta.T, PS).reshape(-1, 1)
            beta_new = c / np.dot(PS, alpha)
            if _counter % 10 == 0:
                err = np.sum(np.abs((beta.squeeze() / beta_new.squeeze()) - ones))
            beta = beta_new
            _counter += 1
        PS *= beta
        PS = alpha.T * PS
        newL = np.argmax(PS, 1)
        yield newL


def balanced_optimize_L_sk(PS):
    """ Optimize psuedo label with Sinkhorn Algorithm (CPU)
    :param PS: The prediction matrix, N x K
    """
    N, K = PS.shape
    tt = time.time()
    PS = PS.T  # now it is K x N
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    PS **= LAMBDA  # K x N
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2:
        r = inv_K / (PS @ c)  # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    PS *= np.squeeze(c)
    PS = PS.T
    PS *= np.squeeze(r)
    PS = PS.T
    argmaxes = np.nanargmax(PS, 0)  # size N
    selflabels = argmaxes
    return selflabels

class SinkhornLabelGenerator:

    def __init__(self):
        pass
    
    def fit(self, prediction):
        """ Fit the model to the prediction matrix
        :param prediction: The prediction matrix, N x K
        """
        prediction = prediction.cpu().numpy()
        # self.labels = optimize_L_sk(prediction)
        self.labels = modified_optimize_L_sk(prediction)
        return self.labels
    