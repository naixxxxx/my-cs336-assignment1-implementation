import torch
from cs336_basics.attention import softmax

def Cross_entropy(x,target):

    logsumexp = torch.logsumexp(x, -1)
    p = x.gather(1, target.unsqueeze(-1)).squeeze(-1)
    loss = logsumexp - p
    return loss.mean()

"""
    x = softmax(x,-1)  
    p = x.gather(1, target.unsqueeze(-1)).squeeze(-1)
    loss = -torch.log(p)
    loss = loss.mean()
    return loss

"""