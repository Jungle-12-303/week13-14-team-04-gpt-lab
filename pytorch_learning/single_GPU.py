import torch

X_train = torch.tensor([
  [-1.2, 3.1],
  [-0.9, 2.9],
  [-0.5, 2.6],
  [2.3, -1.1],
  [2.3, -1.5],
])
y_train = torch.tensor([0,0,0,1,1])

X_test = torch.tensor([
  [-0.8, 2.8],
  [2.6, -1.6],
])
y_test = torch.tensor([0,1])

from torch.utils.data import Dataset

class ToyDataset(Dataset):
  def __init__(self, X, y):
    self.features = X
    self.labels = y
  
  def __getitem__(self, index):
    one_x = self.features[index]
    one_y = self.features[index]
    return one_x, one_y

  def __len__(self):
    return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
  dataset = train_ds,
  batch_size = 2,
  shuffle = True,
  num_workers = 0,
  drop_last = True,
)

test_loader = DataLoader(
  dataset = test_ds,
  batch_size = 2,
  shuffle = False, 
  num_workers = 0,
)

import torch
from forward_model_example import NeuralNetwork
import torch.nn.functional as F


torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2,num_outputs=2)

device = torch.device("cuda")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

  model.train()
  for batch_idx, (features, labels) in enumerate(train_loader):
    features, labels = features.to(device), labels.to(device)
    logits = model(features)
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"에포크 : {epoch+1:03d}/{num_epochs:03d}"
          f" | 배치 {batch_idx:03d}/{len(train_loader):03d}"
          f" | 훈련 손실 : {loss:.2f}"
          )
model.eval()