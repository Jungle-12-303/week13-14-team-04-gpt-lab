import torch
import os,sys
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from forward_model_example import NeuralNetwork
import torch.nn.functional as F

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

def compute_accuracy(model, dataloader,device="cpu"):
  model = model.eval()
  correct = 0.0
  total_examples = 0

  for idx, (features, labels) in enumerate(dataloader):
    features = features.to(device)
    labels = labels.to(device)
    with torch.no_grad():
      logits = model(features)
    
    predictions = torch.argmax(logits, dim=1)
    compare = labels == predictions
    correct += torch.sum(compare)
    total_examples += len(compare)
  return (correct / total_examples).item()

def ddp_setup(rank, world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12345"
  init_process_group(
    backend="nccl",
    rank = rank,
    world_size = world_size
  )
  torch.cuda.set_device(rank)

def prepare_dataset():
  train_loader = DataLoader(
    dataset = train_ds,
    batch_size = 2,
    shuffle = False,
    pin_memory=True,
    drop_last = True,
    sampler = DistributedSampler(train_ds)
  )
  test_loader = DataLoader(
    dataset = test_ds,
    batch_size = 2,
    shuffle = False, 
    num_workers = 0,
  )
  return train_loader, test_loader

def main(rank, world_size, num_epochs):
  ddp_setup(rank, world_size)
  train_loader, test_loader = prepare_dataset()
  model = NeuralNetwork(num_inputs = 2, num_outputs=2)
  model.to(rank)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
  model = DDP(model, device_ids = [rank])
  for epoch in range(num_epochs):
    train_loader.sampler.set_epoch(epoch)
    model.train()
    for features, labels in train_loader:
      features, labels = features.to(rank), labels.to(rank)
      # 모델의 예측과 역전파 코드를 추가합니다.
      #start
      logits = model(features)
      loss = F.cross_entropy(logits, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #end
      print(f"[GPU{rank}] 에포크: {epoch+1:03d} / {num_epochs:03d}"
            f" | 배치 크기 {labels.shape[0]:03d}"
            f" | 훈련 손실: {loss:.2f}"
            )
  model.eval()
  train_acc = compute_accuracy(model, train_loader, device = rank)
  print(f"[GPU{rank}] 훈련 정확도",train_acc)
  test_acc = compute_accuracy(model, test_loader, device = rank)
  print(f"[GPU{rank}] 테스트 정확도",test_acc)
  destroy_process_group()

if __name__ == "__main__":
  print("사용 가능한 GPU 개수:", torch.cuda.device_count())
  torch.manual_seed(123)
  num_epochs = 3
  world_size = torch.cuda.device_count()
  mp.spawn(main, args = (world_size, num_epochs), nprocs = world_size)