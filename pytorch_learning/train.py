import torch
import torch.nn as nn

def calc_loss_batch(input_batch, target_batch, model, device):
  input_batch = input_batch.to(device)
  target_batch = target_batch.to(device)
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(
    logits.flatten(0,1), target_batch.flatten()
  )
  return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
  total_loss = 0
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))
  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      loss = calc_loss_batch(
        input_batch, target_batch, model, device
      )
    else:
      break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from ..src.model import GPTModel

GPT_CONFIG_124M = {
  "vocab_size": 50257,
  "context_length": 256,
  "emb_dim": 768,
  "n_heads": 12,
  "n_layers": 12,
  "drop_rate": 0.1,
  "qkv_bias": False,
}

file_path = "the-verdict.txt"
with open(file_path, "r",encoding = "utf-8") as file:
  text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encoding)


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
with torch.no_grad():
  train_loss = calc_loss_loader(train_loader, model. device)



def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                       num_epochs,eval_freq, eval_iter, start_context, tokenizer):
  train_losses, val_losses, track_tokens_seen = [], [], []
  token_seen, global_step = 0, -1

  for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      loss = calc_loss_batch(
        input_batch, target_batch, model, device
      )
      loss.backward()
      optimizer.step()
      tokens_seen += input_batch.numel()
      global_step += 1

      if global_step %eval_freq == 0:
        train_loss, val_loss = evaluate_model(
          model, train_loader, val_loader, device, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"에포크 {epoch + 1} (Step {global_step:06d}): "
              f"훈련 손실 {train_loss:.3f}, "
              f"검증 손실 {val_loss:.3f}"
        )
      generate_and_print_sample(
        model, tokenizer, device, start_context
      )
  return train_losses, val_losses, track_tokens_seen

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
  train_losses, val_losses, track_tokens_seen = [], [], []
  token_seen, global_step = 0, -1
