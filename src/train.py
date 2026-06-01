# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
import os,sys

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    x = input_batch.to(device)
    t = target_batch.to(device)
    loss, logit = model(x,t)
    return loss
    raise NotImplementedError("calc_loss_batch를 구현하세요.")


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0
    if data_loader == None:
        return float("nan")
    elif num_batches == None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for idx, (input_batch, target_batch) in enumerate(data_loader):
        if (num_batches > idx):
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches
    raise NotImplementedError("calc_loss_loader를 구현하세요.")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    torch.save({
        "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "epoch":epoch,
        "global_step":global_step,
    },
    path)
    # raise NotImplementedError("save_checkpoint를 구현하세요.")


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (checkpoint["epoch"], checkpoint["global_step"])
    raise NotImplementedError("load_checkpoint를 구현하세요.")

def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:,-1,:]
        temp_logits = logits
        if top_k is not None:
            top_k = min(top_k, temp_logits.shape[-1])
            top_value,_ = torch.topk(temp_logits,top_k, dim=-1)
            min_value = top_value[:,-1].unsqueeze(-1)
            temp_logits = temp_logits.masked_fill(temp_logits < min_value, -torch.inf)

        if temperature > 0.0:
            temp_logits = temp_logits / temperature
            probs = torch.softmax(temp_logits,dim=-1)
            last_word = torch.multinomial(probs,num_samples=1)
        else:
            last_word = torch.argmax(temp_logits,dim=-1,keepdim=True)
        if eos_id is not None and last_word == eos_id:
            break
        idx = torch.cat((idx,last_word),dim=1)
    return idx
    raise NotImplementedError("generate를 구현하세요.")

def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    model.eval()
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    token = tokenizer.encode(start_context)
    token = torch.tensor(token,dtype=torch.long,device = device).unsqueeze(0)
    with torch.no_grad():
        token = generate(model, token, max_new_tokens, context_size, temperature, top_k)

    word = tokenizer.decode(token[0].tolist())
    print(word)
    model.train()
    # raise NotImplementedError("generate_and_print_sample을 구현하세요.")


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    if eval_freq <= 0:
        raise ValueError("eval_freq는 0보다 커야합니다.")
    for epoch in range(start_epoch, num_epochs + start_epoch):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(
                        train_loader, model, device, eval_iter
                    )
                    val_loss = calc_loss_loader(
                        val_loader, model, device, eval_iter
                    )
                model.train()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"에포크 {epoch+1} (step {global_step:06d}): "
                      f"훈련 손실 {train_loss:.3f},"
                      f"검증 손실 {val_loss:.3f}"
                )
            if ckpt_freq is not None and global_step % ckpt_freq == 0:
                os.makedirs("checkpoints/",exist_ok=True)
                save_checkpoint(
                    model, optimizer, epoch, global_step, f"checkpoints/ckpt_step{global_step}.pt"
                    )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses
    raise NotImplementedError("train_model을 구현하세요.")


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
