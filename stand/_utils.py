import dgl
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn import metrics


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)


def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    '''relu based hard shrinkage function'''
    x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
    return x


def interpolate(real_data, fake_data, cuda):
    shapes = [1 if i != 0 else real_data.size(i) for i in range(real_data.dim())]
    eta = torch.FloatTensor(*shapes).uniform_(0, 1)
    if cuda:
        eta = eta.cuda()
    else:
        eta = eta

    interpolated = eta * real_data + ((1 - eta) * fake_data)

    if cuda:
        interpolated = interpolated.cuda()
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    return interpolated


def calculate_gradient_penalty(D, real_g, fake_g, real_p=None, fake_p=None):
    '''calculate gradient penalty for training discriminator'''
    cuda = True if torch.cuda.is_available() else False
    inter_g = interpolate(real_g, fake_g, cuda)
    if real_p is None or fake_p is None:
        inter_p = None
        inputs = (inter_g) 
    else:
        inter_p = interpolate(real_p, fake_p, cuda)
        inputs = (inter_g, inter_p)

    # calculate probability of interpolated examples
    prob_interpolated = D(inter_g, inter_p)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=inputs,
                              grad_outputs=torch.ones(prob_interpolated.size()).cuda()
                              if cuda else torch.ones(prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


def evaluate(y_true, y_score):
    """calculate evaluation metrics"""
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)
    
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)
    
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thres = np.percentile(y_score, ratio)
    y_pred = (y_score >= thres).astype(int)
    y_true = y_true.astype(int)
    _, _, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f1