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
    # 나중에 모델 모드 되돌리기를 위해 초기 모드가 뭔지 파악하기 
    was_training = model.training

    # 처음 모델이 학습 모드였다면
    if was_training:
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
    # 실제 반복 개수를 정하기 
    if num_batches is None:
        # 전체 data_loader를 다 보기
        batches_len = len(data_loader)

        # for input_batch, target_batch in data_loader:
    else:
        # num_batches 개수만큼 data_loader를 보기
        # num_batches가 실제 data_loader 길이보다 클 수도 있음
        # -> 실제 반복 횟수 = data_loader 길이와 num_batches 중 작 은 값 사용 
        batches_len = min(num_batches, len(data_loader))

    # 4. gradient 계산을 끄기
    # 평균 loss만 계산할 거니까 역전파용 그래프 없이 torch.no_grad() 안에서 반복
    # 이 안에서는 gradient 계산 안 함
    # -> loss를 계산하긴 하지만, 역전파용 계산 그래프는 만들지 않음
    with torch.no_grad():
        # data_loader를 순회하며 input_batch, target_batch 얻기 
        # batch를 하나씩 꺼냄
        # -> 각 batch는 input_batch, target_batch의 쌍
        # -> 몇 번째 batch인지도 같이 알아야 해서 enumerate 사용 
        # => for문 한 문장으로 알아서 data_loader 순회하면서 한 batch씩 꺼내고 그 batch를 input_batch, target_batch로 분리까지 해줌 
        for i, (input_batch, target_batch) in enumerate(data_loader):
            # i: for문 돈 횟수
            # -> 너무 많이 돌았으면 멈춤 
            if i >= batches_len:
                break
            
            # 5. data_loader를 돌면서 batch loss 계산 
            # -> 각 배치마다 이전에 만든 calc_loss_batch()를 호출하여 loss를 구하고 누적
            batch_loss = calc_loss_batch(input_batch, target_batch, model=model, device=device)

            # 6. loss = tensor -> 숫자로 누적
            # calc_loss_batch 결과 = torch.Tensor
            # batch_loss.item(): loss 텐서에서 숫자 값만 꺼내기
            total_loss += batch_loss.item()

            # 실제 반복한 배치 수 누적 
            processed_batches += 1 

    # 모델을 다시 원래 모드로 되돌림 
    if was_training:
        model.train()

    # processed_batches 가 0이면 total_loss / processed_batches 계산에서 에러
    if processed_batches == 0:
        # 평균을 float로 반환해야 함 
        # -> nan 값으로 처리 
        return float("nan")

    # 7. 평균 loss 반환
    # -> total_loss / 실제 반복한 배치 수(processed_batches)
    return total_loss / processed_batches 
     
    # raise NotImplementedError("calc_loss_loader를 구현하세요.")

def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # torch.save는 json과 다르게 문자열 경로로 path를 받을 수 있음
    # -> 전달받은 path가 str이라 path = Path(path) 따로 안 해도 됨
    
    # model/optimizer 상태: model/optimizer가 학습 중 들고 있는 숫자 값  
    # model.state_dict(): 토큰 임베딩 가중치, attention layer 가중치, linear layer weight/bias, layer norm 값... 
    # => model 파라미터 
    # optimizer.state_dict(): learning rate 같은 설정, momentum/variance 누적값, 각 파라미터별 step 정보
    # => optimizer가 학습을 이어가기 위해 필요한 내부 값 

    save_dict = {}

    save_dict["model"] = model.state_dict() 
    save_dict["optimizer"] = optimizer.state_dict() 
    save_dict["epoch"] = epoch
    save_dict["global_step"] = global_step

    # 저장 
    torch.save(save_dict, path)

    # raise NotImplementedError("save_checkpoint를 구현하세요.")

def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # save에서 저장된 텐서가 GPU에서 저장됐는데 load는 CPU에서 하고 싶을 수 있음
    # -> load를 어느 장치에서 할지 명시(저장된 텐서를 device로 옮겨서 load)
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint["epoch"] 
    global_step = checkpoint["global_step"]

    return (epoch, global_step)
    # raise NotImplementedError("load_checkpoint를 구현하세요.")


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
    # 생성 중에는 gradient를 계산하지 않으므로, 현재 모델이 학습 모드였는지 먼저 저장한다.
    was_training = model.training

    # Dropout 같은 학습 전용 동작이 생성 결과를 흔들지 않도록 평가 모드로 바꾼다.
    model.eval()

    # max_new_tokens 개수만큼 새 토큰을 하나씩 뒤에 붙인다.
    for _ in range(max_new_tokens):
        # 모델의 position embedding 길이를 넘지 않도록 마지막 context_size 토큰만 잘라서 사용한다.
        idx_cond = idx[:, -context_size:]

        # 생성에서는 역전파가 필요 없으므로 계산 그래프를 만들지 않고 logits만 얻는다.
        with torch.no_grad():
            # GPTModel 출력 shape는 [batch, seq, vocab_size]이다.
            logits = model(idx_cond)

        # 혹시 모델이 (loss, logits) 형태를 반환하는 구현이어도 logits만 꺼내 쓸 수 있게 처리한다.
        if isinstance(logits, tuple):
            logits = logits[1]

        # 다음 토큰은 마지막 위치의 vocab 점수만 보고 고르면 된다.
        next_token_logits = logits[:, -1, :]

        # top_k가 양수로 지정되면 점수가 높은 k개 토큰만 후보로 남긴다.
        if top_k is not None and top_k > 0:
            # vocab 크기보다 큰 top_k가 들어와도 에러가 나지 않도록 실제 vocab 크기로 제한한다.
            top_k = min(top_k, next_token_logits.shape[-1])

            # 각 batch마다 상위 k개 점수와 그 중 가장 낮은 점수 경계값을 구한다.
            top_values, _ = torch.topk(next_token_logits, top_k)

            # 상위 k개 안에 들지 못한 토큰 위치를 True로 표시한다.
            remove_mask = next_token_logits < top_values[:, [-1]]

            # 후보 밖 토큰은 softmax 후 확률이 0이 되도록 -inf 점수로 바꾼다.
            next_token_logits = next_token_logits.masked_fill(remove_mask, -torch.inf)

        # temperature가 0 이하이면 샘플링 없이 가장 높은 점수 토큰을 고르는 greedy 방식으로 처리한다.
        if temperature <= 0:
            # argmax 결과 shape는 [batch]이므로 뒤에서 cat하기 위해 [batch, 1]로 바꾼다.
            next_idx = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # temperature가 작을수록 확률 분포가 뾰족해지고, 클수록 더 다양한 토큰을 뽑게 된다.
            scaled_logits = next_token_logits / temperature

            # logits 점수를 확률 분포로 바꾼다.
            probs = torch.softmax(scaled_logits, dim=-1)

            # 확률 분포에서 batch마다 다음 토큰 id 하나를 샘플링한다.
            next_idx = torch.multinomial(probs, num_samples=1)

        # 새로 뽑은 토큰을 기존 토큰 시퀀스 뒤에 붙인다.
        idx = torch.cat((idx, next_idx), dim=1)

        # eos_id가 지정되어 있고 모든 batch가 eos를 뽑았다면 더 생성하지 않고 멈춘다.
        if eos_id is not None and torch.all(next_idx == eos_id):
            break

    # 원래 학습 모드였던 모델이면 호출 전 상태를 유지하도록 다시 train 모드로 돌린다.
    if was_training:
        model.train()

    # 시작 토큰 뒤에 생성된 토큰이 이어 붙은 전체 시퀀스를 반환한다.
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
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    # 사람이 입력한 시작 문자열을 tokenizer가 이해하는 token id 목록으로 바꾼다.
    encoded = tokenizer.encode(start_context)

    # generate 함수는 [batch, seq] 형태를 기대하므로 batch 차원 1개를 추가한다.
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    # 모델도 같은 device에 있어야 입력 텐서와 연산 위치가 맞는다.
    model.to(device)

    # 지정한 샘플링 옵션으로 새 토큰을 생성한다.
    generated = generate(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
    )

    # decode에는 batch 차원이 필요 없으므로 첫 번째 batch의 token id만 리스트로 꺼낸다.
    generated_ids = generated[0].tolist()

    # token id 목록을 다시 사람이 읽을 수 있는 문자열로 바꾼다.
    generated_text = tokenizer.decode(generated_ids)

    # 학습 중간 샘플을 바로 확인할 수 있도록 출력한다.
    print(generated_text)


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
    # epoch마다 평균 train loss를 저장해서 마지막에 반환할 리스트를 만든다.
    train_losses = []

    # 학습에 사용할 장치로 모델 파라미터를 옮긴다.
    model.to(device)

    # start_epoch부터 시작하면 checkpoint에서 이어 학습할 때 이전 epoch를 건너뛸 수 있다.
    for epoch in range(start_epoch, num_epochs):
        # 파라미터 업데이트가 일어나야 하므로 모델을 학습 모드로 둔다.
        model.train()

        # 이번 epoch의 batch loss들을 평균 내기 위해 누적합을 준비한다.
        total_train_loss = 0.0

        # 실제로 처리한 batch 수를 세서 마지막 평균 계산에 사용한다.
        processed_batches = 0

        # train_loader에서 input과 target batch를 하나씩 꺼내 학습한다.
        for input_batch, target_batch in train_loader:
            # 이전 batch에서 남아 있는 gradient를 지운다.
            optimizer.zero_grad()

            # 한 batch의 next-token prediction loss를 계산한다.
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # loss를 기준으로 각 파라미터의 gradient를 계산한다.
            loss.backward()

            # optimizer가 gradient를 보고 모델 파라미터를 한 번 업데이트한다.
            optimizer.step()

            # 현재 batch loss를 Python 숫자로 꺼내 epoch 누적 loss에 더한다.
            total_train_loss += loss.item()

            # 평균 계산을 위해 처리한 batch 수를 1 늘린다.
            processed_batches += 1

            # 전체 학습 진행 상황을 나타내는 step을 1 늘린다.
            global_step += 1

            # eval_freq가 양수이고 해당 step에 도달하면 train/val loss를 간단히 출력한다.
            if eval_freq > 0 and global_step % eval_freq == 0:
                # 작은 eval_iter만 평가하면 학습 중간 점검을 빠르게 할 수 있다.
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)

                # validation loader가 없으면 val loss는 계산하지 않는다.
                val_loss = None

                # validation loader가 있으면 같은 방식으로 평균 validation loss를 계산한다.
                if val_loader is not None:
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

                # 사용자가 학습 진행을 볼 수 있도록 step 기준 loss를 출력한다.
                if val_loss is None:
                    print(f"Ep {epoch + 1} Step {global_step}: train loss {train_loss:.3f}")
                else:
                    print(f"Ep {epoch + 1} Step {global_step}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

            # ckpt_freq가 지정되어 있고 해당 step에 도달하면 checkpoint를 저장한다.
            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                # 별도 경로 인자가 없으므로 step 번호가 들어간 기본 파일명으로 저장한다.
                ckpt_path = f"checkpoint_step_{global_step}.pt"

                # 이어 학습에 필요한 model/optimizer/epoch/step 정보를 저장한다.
                save_checkpoint(model, optimizer, epoch=epoch, global_step=global_step, path=ckpt_path)

        # batch가 하나도 없으면 평균을 낼 수 없으므로 nan으로 기록한다.
        if processed_batches == 0:
            epoch_train_loss = float("nan")
        else:
            # 이번 epoch에서 처리한 모든 batch loss의 평균을 계산한다.
            epoch_train_loss = total_train_loss / processed_batches

        # epoch별 train loss 리스트에 이번 epoch 평균 loss를 추가한다.
        train_losses.append(epoch_train_loss)

        # epoch가 끝날 때마다 시작 문맥으로 생성 샘플을 출력해 학습 상태를 눈으로 확인한다.
        if tokenizer is not None and start_context:
            generate_and_print_sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                start_context=start_context,
                context_size=getattr(model, "context_length", 256),
            )

    # num_epochs 동안 모은 epoch별 train loss를 반환한다.
    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    # 테스트처럼 화면이 없는 Agg 백엔드에서도 독립적인 figure를 만들도록 새 figure를 연다.
    plt.figure()

    # epoch별 train loss를 선 그래프로 그린다.
    plt.plot(train_losses, label="Train")

    # validation loss가 전달되었을 때만 함께 그린다.
    if val_losses is not None:
        plt.plot(val_losses, label="Val")

    # x축은 epoch 번호를 의미한다.
    plt.xlabel("Epoch")

    # y축은 loss 값을 의미한다.
    plt.ylabel("Loss")

    # Train/Val 선을 구분할 수 있도록 범례를 표시한다.
    plt.legend()

    # 그래프 제목을 붙인다.
    plt.title("Training / Validation Loss")

    # 테스트 환경에서는 화면에 띄우지 않고 figure를 닫아 Agg 백엔드 경고를 막는다.
    plt.close()
