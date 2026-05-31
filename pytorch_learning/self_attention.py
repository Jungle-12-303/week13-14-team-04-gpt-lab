import torch.nn  as nn
import torch
class SelfAttention_v1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.W_query = nn.Parameter(torch.rand(d_in, d_out))
    self.W_key = nn.Parameter(torch.rand(d_in, d_out))
    self.W_value = nn.Parameter(torch.rand(d_in, d_out))
  
  def forward(self, x):
    key = x @ self.W_key
    query = x @ self.W_query
    values = x @ self.W_value
    attn_scores = query @ key.T
    attn_weight = torch.softmax(attn_scores / (key.shape[-1] ** 0.5), dim=1)
    context_vec = attn_weight @ values
    return context_vec

class SelfAttention_v1(nn.Module):
  def __init__(self, d_in, d_out,qkv_bias = False):
    super().__init__()
    self.W_query = nn.Linear(d_in, d_out,bias = qkv_bias)
    self.W_key = nn.Linear(d_in, d_out,bias = qkv_bias)
    self.W_value = nn.Linear(d_in, d_out,bias = qkv_bias)
  
  def forward(self, x):
    key = self.W_key(x)
    query = self.W_query(x)
    values = self.W_value(x)
    attn_scores = query @ key.T
    attn_weight = torch.softmax(attn_scores / (key.shape[-1] ** 0.5), dim=1)
    context_vec = attn_weight @ values
    return context_vec

class CausalAttention(nn.Module):
  def __init__(self, d_in, d_out,qkv_bias=False):
    self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)

  def forward(self,x):
    query = self.W_query(x)
    key = self.W_key(x)
    value = self.W_value(x)
    attn_scores = query @ key.T
    attn_length = key.shape[0]
    attn_scores_mask = torch.tril(torch.ones(attn_length, attn_length,device=x.device)).bool()
    attn_scores = attn_scores.masked_fill(~attn_scores_mask,-torch.inf)
    attn_weight = torch.softmax(attn_scores  / (key.shape[-1] ** 0.5), dim=-1)
    context_vec = attn_weight @ value
    return context_vec

class SelfAttention_v1(nn.Module):
  def __init__(self, d_in, d_out,qkv_bias = False):
    super().__init__()
    self.W_query = nn.Linear(d_in, d_out,bias = qkv_bias)
    self.W_key = nn.Linear(d_in, d_out,bias = qkv_bias)
    self.W_value = nn.Linear(d_in, d_out,bias = qkv_bias)
  
  def forward(self, x):
    key = self.W_key(x)
    query = self.W_query(x)
    values = self.W_value(x)
    attn_scores = query @ key.T
    attn_weight = torch.softmax(attn_scores / (key.shape[-1] ** 0.5), dim=1)
    context_vec = attn_weight @ values
    return context_vec

class CausalAttention(nn.Module):
  def __init__(self, d_in, d_out,qkv_bias=False):
    self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)

  def forward(self,x):
    query = self.W_query(x)
    key = self.W_key(x)
    value = self.W_value(x)
    attn_scores = (query @ key.T)
    attn_length = key.shape[-1]
    attn_mask = torch.tril(torch.ones(attn_length,attn_length))
    attn_scores[attn_mask]
    return context_norm







d_in = 6
d_out = 6
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
sa_v2 = SelfAttention_v2(d_in, d_out)
1
# print(sa_v1(input))