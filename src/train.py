# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
    # 1. input_batch, target_batch를 device로 옮김
    # 모델이 GPU에 있으면 입력 텐서도 같은 GPU에 있어야 함
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 2. 모델에 input_batch를 넣어서 logits를 얻음
    # GPT 모델 출력은 보통 shape -> [batch_size, sequence_length, vocab_size] = [B, T, V]
    # -> 각 배치의 각 위치마다 다음 토큰이 vocab 중 무엇일지에 대한 점수를 냄 
    logits = model(input_batch)

    # reshape를 위해 vocab_size 구하기 
    vocab_size = logits.shape[-1]

    # 3. target_batch: 정답 토큰 ID
    # shape -> [batch_size, sequence_length] = [B, T]
    # -> 각 위치에서 맞혀야 하는 실제 다음 토큰 ID가 들어있음

    # 4. cross_entropy에 넣기 위해 reshape 
    # Cross Entropy Loss가 원하는 입력 -> 
    # prediction: [N, vocab_size]
    # target:     [N]
    # -> logits: 3차원, target: 2차원
    # => batch와 sequence 차원을 하나로 펼쳐야 함
    # => logits:  [B, T, V] -> [B*T, V]
    # => target:  [B, T]    -> [B*T]
    re_logits = logits.reshape(-1, vocab_size) # 마지막 차원을 vocab_size 개씩 묶고, 앞 차원 개수는 알아서 계산 
    re_target = target_batch.reshape(-1) # 그냥 모든 차원을 하나로 합치면 됨 

    # 5. 펼친 logits와 target으로 loss 계산 -> 모든 토큰 위치에 대한 평균 loss 나옴 
    loss = F.cross_entropy(re_logits, re_target)

    return loss 
    # raise NotImplementedError("calc_loss_batch를 구현하세요.")


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    # 1. 모델을 평가 모드로 바꾸기
    # eval: 모델 자체의 상태를 바꾸는 메서드 
    # 학습할 땐 model.train()
    model.eval()

    # 2. loss 누적 변수를 만들기
    total_loss = 0

    # total_loss 평균 구할 때 사용할 실제 처리한 배치 수 
    processed_batches = 0


    # 3. 몇 배치를 볼지 정하기
    # input batch 따로, target batch 따로 ? 반복문 돌면서 뭘 뽑아야 하는데 ?
    if num_batches is None:
        batches_len = len(data_loader)
        # # 전체 data_loader를 다 보기
        # # for문을 data_loader 끝까지 돌라는 거? data_loader는 값이 뭔데 텐서? 튜플? 어케 순회함 
        # for input_batch, target_batch in data_loader:
        
        # # 실제 반복 개수를 정해두기
    else:
        batches_len = min(num_batches, len(data_loader))
        # # num_batches 개수만큼 data_loader를 보기
        # # num_batches가 실제 data_loader 길이보다 클 수도 있음
        # # -> 실제 반복 횟수 = data_loader 길이와 num_batches 중 작은 값 사용 

        # # 반복하다가 i가 원하는 개수에 도달하면 멈추기 
        # for i, batch in enumerate(data_loader):
        #     if i >= num_batches:
        #         break

        # # 실제 반복 개수를 정해두기

    batch_loss = 0
    
    with torch.no_grad():
        for i in range(0, len):
            # data_loader에서 input_batch, target_batch를 하나씩 꺼냄
            # -> 어떻게 함 

            # 너무 많이 돌았으면 멈추기 
            #     if i >= num_batches:
            #     break

            batch_loss = calc_loss_batch()

            total_loss += batch_loss.item()

            processed_batches += 1 

    return total_loss / processed_batches 
    # # 4. gradient 계산을 끄기
    # # 평균 loss만 계산할 거니까 역전파용 그래프 없이 torch.no_grad() 안에서 반복
    # # 어케 끄라고 torch.no_grad() 호출하라고?
    # with torch.no_grad():
    #     # 이 안에서는 gradient 계산 안 함
    #     # loss를 계산하긴 하지만, 역전파용 계산 그래프는 만들지 않음 

    # 5. data_loader를 돌면서 batch loss 계산 
    # -> input_batch, target_batch
    # -> 각 배치마다 이전에 만든 calc_loss_batch()를 호출하여 loss를 구하고 누적
        # for문을 data_loader 끝까지 돌라는 거? data_loader는 값이 뭔데 텐서? 튜플? 어케 순회함 
    # batch_loss = 0

    # total_loss += batch_loss.item()

    # 6. loss = tensor -> 숫자로 누적
    # calc_loss_batch 결과 = torch.Tensor
    # 평균을 float로 반환해야 함 
    # 누적: 파이썬 숫자로 꺼내는 방식 
    # loss 텐서에서 숫자 값만 꺼내기
    # 어케하는데?
    # loss.item()


    # processed_batches += 1 

    # 7. 평균 .loss 반환
    # -> total_loss / 실제 반복한 배치 수(processed_batches)
    # 그러니까 total_loss가 실제 반복한 배치 수라는 거?
    # return total_loss / processed_batches 
    
    # raise NotImplementedError("calc_loss_loader를 구현하세요.")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    raise NotImplementedError("save_checkpoint를 구현하세요.")


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
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
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    raise NotImplementedError("generate_and_print_sample을 구현하세요.")


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
