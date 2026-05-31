import torch
import torch.nn as nn
import torch.optim as optim

class Layers(nn.Module):
  def __init__(self,input_dim, output_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_dim, input_dim),
      nn.Sigmoid(),
      nn.Linear(input_dim, output_dim)
    )
  
  def forward(self,x):
    return self.layers(x)

paint = torch.tensor([
  [0,0,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0,0],
]).float()

model = Layers(paint.shape[0] * paint.shape[1],1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)

for epoch in range(1000):
    pred = model(paint.view(-1))
    loss = criterion(pred, torch.tensor([1.0]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}, pred = {pred} loss = {loss.item():.4f}")
1