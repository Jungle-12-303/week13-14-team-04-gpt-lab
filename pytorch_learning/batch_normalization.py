import torch
import torch.nn as nn

class BatchNormalization(nn.Module):
  def __init__(self,emb_dim):
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
  def forward(self,x):
    mu = x.mean(dim = -1,keepdim = True)
    omega = x.var(dim = -1, keepdim = True, unbiased = False)
    norm_x = (x-mu) / torch.sqrt(omega + self.eps)
    return self.scale * norm_x + self.shift
    