import tensorflow as tf
import torch
import numpy as np
import time

LAMBDA = 25

def optimize_L_sk_gpu(args, PS):
    print('doing optimization now',flush=True)
    args.dist = torch.tensor([0.627, 0.289, 0.056, 0.028])
    args.diff_dist_every = False

    # create L
    N = PS.size(0)
    K = PS.size(1)
    _K_dist = torch.ones((K, 1), dtype=torch.float64, device='cuda') # / K
    if args.distribution != 'default':
        marginals_argsort = torch.argsort(PS.sum(0))
        if (args.dist is None) or args.diff_dist_every:
            if args.distribution == 'gauss':
                _K_dist = (torch.randn(size=(K, 1), dtype=torch.float64, device='cuda')*args.gauss_sd + 1) * N / K
                _K_dist = torch.clamp(_K_dist, min=1)
                args.dist = _K_dist
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
    newL = torch.argmax(PS, 1)

    # return back to obtain cost (optional)
    torch.mul((1./alpha).t(), PS, out=PS)
    torch.mul(PS, 1./beta, out=PS)
    return newL


def modified_optimize_L_sk(PS):
    N, K = PS.shape
    _K_dist = np.array([0.627, 0.289, 0.056, 0.028]).reshape(-1, 1)
    beta = np.ones((N, 1), dtype=np.float64) / N
    PS **= (0.5 * LAMBDA)
    # r = 1./_K_dist
    # r /= r.sum()
    r = _K_dist

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

    # inplace calculations
    # import pdb; pdb.set_trace()
    PS *= beta
    PS = alpha.T * PS
    # np.matmul(PS, beta, out=PS)
    # np.matmul(alpha.T, PS, out=PS)
    newL = np.argmax(PS, 1)

    return newL


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
    