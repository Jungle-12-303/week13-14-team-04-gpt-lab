import torch
import torch.nn as nn

GPT_CONFIG_124M = {
  "vocab_size" : 50257,
  "context_length" : 1024,
  "emb_dim" : 768,
  "n_heads" : 12,
  "n_layers" : 12,
  "drop_rate" : 0.1,
  "qkv_bias" : False,
}

class BatchNormalization(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))
  def forward(self,x):
    mu = x.mean(dim = -1,keepdim = True)
    omega = x.var(dim = -1, keepdim = True, unbiased = False)
    norm_x = (x-mu) / torch.sqrt(omega + self.eps)
    return self.scale * norm_x + self.shift

class GELU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    data = 0.5 * x *(1 + torch.tanh((torch.sqrt(torch.tensor(2.0 / torch.pi))) * (x + 0.044715 * torch.pow(x,3))))
    return data
  
# import matplotlib.pyplot as plt

# gelu, relu = GELU(), nn.ReLU()

# x = torch.linspace(-3, 3, 100)
# y_gelu,y_relu = gelu(x),relu(x)
# plt.figure(figsize = (8,3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu],["GELU","RELU"]),1):
#   plt.subplot(1,2,i)
#   plt.plot(x,y)
#   plt.title(f"{label} activation function")
#   plt.xlabel("x")
#   plt.ylabel(f"{label}(x)")
#   plt.grid(True)
# plt.tight_layout()
# plt.savefig("./gelu.png")

class FeedForward(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
      GELU(),
      nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
    )
  def forward(self,x):
    return self.layers(x)

class ExampleDeepNeuralNetwork(nn.Module):
  def __init__ (self, layer_sizes, use_shortcut):
    super().__init__()
    self.use_shortcut = use_shortcut
    self.layers = nn.ModuleList([
      nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]),GELU()),
      nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]),GELU()),
      nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]),GELU()),
      nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]),GELU()),
      nn.Sequential(nn.Linear(layer_sizes[4],layer_sizes[5]),GELU()),
    ])
  def forward(self, x):
    for layer in self.layers:
      layer_output = layer(x)
      if self.use_shortcut and x.shape == layer_output.shape:
        x = x + layer_output
      else:
        x = layer_output
    return x

def print_gradients(model, x):
  output = model(x)
  target = torch.tensor([[0.]])

  loss = nn.MSELoss()
  loss = loss(output, target)

  loss.backward()
  for name, param in model.named_parameters():
    if 'weight' in name:
      print(f"{name}의 평균 그레디언트는 {param.grad.abs().mean().item()}입니다.")

# layer_sizes = [3,3,3,3,3,1]

# sample_input = torch.tensor([[1.,0.,-1.,]])
# torch.manual_seed(123)
# model_without_shortcut = ExampleDeepNeuralNetwork(
#   layer_sizes, use_shortcut=False
# )

# print_gradients(model_without_shortcut, sample_input)
# torch.manual_seed(123)
# model_with_shortcut = ExampleDeepNeuralNetwork(
#   layer_sizes, use_shortcut=True
# )
# print("-----------------------------")
# print_gradients(model_with_shortcut, sample_input)

from cousal_attention_com import MultiHeadAttention

class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = MultiHeadAttention(
      d_in = cfg["emb_dim"],
      d_out = cfg["emb_dim"],
      context_length=cfg["context_length"],
      num_heads=cfg["n_heads"],
      dropout = cfg["drop_rate"],
      qkv_bias = cfg["qkv_bias"],
    )
    self.ff = FeedForward(cfg)
    self.norm1 = BatchNormalization(cfg["emb_dim"])
    self.norm2 = BatchNormalization(cfg["emb_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
  
  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x) + shortcut
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x) + shortcut
    return x

torch.manual_seed(123)
x = torch.rand(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("입력크기:", x.shape)
print("출력크기:",output.shape)

class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.trf_block = nn.Sequential(
    [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = BatchNormalization(cfg["emb_dim"])
    self.out_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
  
  def forward(self,in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(
      torch.arange(in_idx.shape, device = in_idx.device)
    )
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_block(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

