# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import os

import matplotlib.pyplot as plt
import torch

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
    """ 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    # 모델과 입력 텐서가 같은 장치에 있어야 계산할 수 있음
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 모델은 loss와 logits를 함께 돌려주므로 학습에는 loss만 사용
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """ data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    # 검증 뒤 원래 학습 모드로 되돌리기 위해 현재 상태를 기억
    model_was_training = model.training

    # 검증 loss는 dropout 없이 고정된 상태에서 계산
    model.eval()
    losses = []

    # 검증에서는 gradient가 필요 없으므로 계산 그래프를 만들지 않음
    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            losses.append(loss.item())

    # 함수 호출 전 학습 모드였다면 다시 학습 모드로 복구
    if model_was_training:
        model.train()

    # 비어 있는 loader가 들어오면 계산할 평균이 없으므로 nan 반환
    return float(sum(losses) / len(losses)) if losses else float("nan")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """ model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # 저장 경로에 폴더가 포함되어 있으면 먼저 만들어 둠
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # 이어 학습하려면 모델 가중치와 optimizer 상태와 진행 위치가 모두 필요
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """ torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # 저장 당시 장치와 달라도 현재 device 기준으로 읽을 수 있게 함
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # optimizer가 주어진 경우에만 이어 학습 상태를 복원
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 없는 값은 0으로 두어 새 학습처럼 이어갈 수 있게 함
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """ temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    # 생성이 끝난 뒤 원래 모드로 되돌리기 위해 현재 상태를 기억
    model_was_training = model.training

    # 생성 중에는 dropout 없이 같은 입력에 안정적인 출력을 쓰기 위해 eval 모드 사용
    model.eval()

    # 생성은 추론이므로 gradient 계산이 필요 없음
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 모델의 context 길이를 넘지 않도록 가장 최근 토큰만 사용
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)

            # 다음 토큰 하나만 고르면 되므로 마지막 위치의 logits만 사용
            logits = logits[:, -1, :]

            if top_k is not None:
                # 확률 후보를 상위 k개로 제한해 너무 낮은 후보를 제외
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(
                    logits < top_values[:, [-1]],
                    torch.full_like(logits, float("-inf")),
                    logits,
                )
            if temperature == 0:
                # temperature가 0이면 가장 점수가 높은 토큰을 그대로 선택
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # temperature가 있으면 확률 분포를 만들고 그 분포에서 샘플링
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # 새로 고른 토큰을 기존 문맥 뒤에 붙임
            idx = torch.cat((idx, idx_next), dim=1)

            # 모든 배치가 EOS를 만들면 더 생성할 필요가 없음
            if eos_id is not None and torch.all(idx_next == eos_id):
                break

    if model_was_training:
        model.train()

    return idx


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
    """start_context를 encode하고 generate 후 decode하여 출력합니다."""
    # 문자열 문맥을 모델이 읽을 수 있는 token id로 변환
    encoded = tokenizer.encode(start_context, add_bos_eos=False)

    # 모델 입력 형식인 batch 차원을 앞에 추가
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    # 현재 문맥 뒤에 새 토큰을 이어 붙여 생성
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id(),
    )

    # batch의 첫 번째 결과를 다시 문자열로 복원해 확인
    decoded = tokenizer.decode(out[0].tolist(), skip_special=True)
    print(decoded)


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
    """ 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    # 모델과 batch가 같은 장치에서 계산되도록 모델을 먼저 이동
    model.to(device)

    # epoch마다 평균 train loss를 저장
    train_losses = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # 학습 중에는 dropout 같은 학습용 동작이 켜져야 함
        model.train()

        # 현재 epoch 안에서 batch별 loss를 모음
        epoch_losses = []

        for input_batch, target_batch in train_loader:
            # PyTorch는 gradient를 누적하므로 batch마다 먼저 비움
            optimizer.zero_grad()

            # 현재 batch의 다음 토큰 예측 loss 계산
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # loss를 줄이기 위한 각 파라미터의 gradient 계산
            loss.backward()

            # 계산된 gradient를 사용해 실제 파라미터 업데이트
            optimizer.step()

            # 평균 loss 계산을 위해 tensor loss를 숫자로 저장
            epoch_losses.append(loss.item())
            global_step += 1

            # 정해진 step마다 검증 loss를 확인
            if eval_freq and global_step % eval_freq == 0 and val_loader is not None:
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                print(f"Step {global_step}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")

            # 정해진 step마다 이어 학습용 checkpoint 저장
            if ckpt_freq and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    f"checkpoint_step_{global_step}.pt",
                )

        # 한 epoch이 끝나면 batch loss 평균을 기록
        train_losses.append(
            float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
        )

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()
