# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    # TSV 파일을 읽어서 {"text": ..., "label": ...} 리스트로 바꾸는 내부 helper를 만든다.
    def read_nsmc_tsv(path: str | Path | None) -> list[dict]:
        # test 파일 경로가 None이면 빈 리스트를 반환해서 이후 로직이 단순해지게 한다.
        if path is None:
            return []

        # 문자열 경로도 Path 객체처럼 다룰 수 있도록 변환한다.
        path = Path(path)

        # 변환된 감성 분류 샘플들을 담을 리스트를 만든다.
        rows = []

        # NSMC 원본은 tab으로 구분된 TSV이므로 newline과 encoding을 명시해서 연다.
        with open(path, "r", encoding="utf-8", newline="") as f:
            # 첫 줄 header(id, document, label)를 이용해 dict 형태로 한 줄씩 읽는다.
            reader = csv.DictReader(f, delimiter="\t")

            # TSV의 각 row를 순회한다.
            for row in reader:
                # document 컬럼에서 리뷰 텍스트를 꺼내고, None이면 빈 문자열로 처리한다.
                text = row.get("document") or ""

                # 앞뒤 공백만 있는 리뷰는 빈 리뷰로 취급하기 위해 strip한다.
                text = text.strip()

                # label 컬럼도 꺼내고, 없으면 이 row는 사용할 수 없다.
                label = row.get("label")

                # 빈 리뷰이거나 label이 없으면 학습 샘플에서 제외한다.
                if not text or label is None:
                    continue

                # label이 정수로 바뀌지 않는 이상한 row는 안전하게 건너뛴다.
                try:
                    label_id = int(label)
                except ValueError:
                    continue

                # NSMC 감성 label은 0/1이어야 하므로 그 외 값은 제외한다.
                if label_id not in (0, 1):
                    continue

                # 모델이 사용할 표준 형식으로 변환해서 리스트에 추가한다.
                rows.append({"text": text, "label": label_id})

        # 정제된 샘플 리스트를 반환한다.
        return rows

    # train TSV에서 전체 train 후보 데이터를 읽는다.
    all_train_rows = read_nsmc_tsv(train_tsv_path)

    # test TSV가 있으면 별도 test 데이터로 읽고, 없으면 빈 리스트로 둔다.
    test_data = read_nsmc_tsv(test_tsv_path)

    # 같은 seed를 쓰면 매번 train/val split이 같아지도록 random generator를 만든다.
    rng = random.Random(seed)

    # 원본 리스트를 직접 섞지 않도록 복사본을 만든다.
    shuffled_rows = list(all_train_rows)

    # train/validation split을 위해 샘플 순서를 섞는다.
    rng.shuffle(shuffled_rows)

    # val_ratio가 0~1 범위를 벗어나도 너무 이상하게 동작하지 않도록 범위 안으로 제한한다.
    val_ratio = max(0.0, min(1.0, val_ratio))

    # validation 샘플 개수를 계산한다.
    val_size = int(len(shuffled_rows) * val_ratio)

    # 데이터가 있고 val_ratio도 양수인데 int 계산으로 0이 되면 최소 1개는 validation으로 둔다.
    if len(shuffled_rows) > 0 and val_ratio > 0 and val_size == 0:
        val_size = 1

    # 섞인 데이터 앞쪽 val_size개를 validation 데이터로 사용한다.
    val_data = shuffled_rows[:val_size]

    # 나머지 데이터를 train 데이터로 사용한다.
    train_data = shuffled_rows[val_size:]

    # output_dir이 주어지면 나중에 재사용할 수 있도록 JSONL 파일로 저장한다.
    if output_dir is not None:
        # 문자열 경로도 Path 객체로 바꾼다.
        output_dir = Path(output_dir)

        # 출력 폴더가 없으면 만든다.
        output_dir.mkdir(parents=True, exist_ok=True)

        # split 이름과 데이터를 묶어 반복 저장하기 쉽게 만든다.
        split_map = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

        # train/val/test 각각을 jsonl 파일로 저장한다.
        for split_name, split_data in split_map.items():
            # 저장할 파일 경로를 만든다.
            output_path = output_dir / f"nsmc_sentiment_{split_name}.jsonl"

            # JSONL은 한 줄에 JSON 객체 하나씩 저장하는 형식이다.
            with open(output_path, "w", encoding="utf-8") as f:
                # split 데이터의 각 샘플을 한 줄씩 저장한다.
                for item in split_data:
                    # ensure_ascii=False로 한글을 깨지지 않게 저장한다.
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # train, validation, test 데이터를 tuple로 반환한다.
    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        # idx번째 감성 분류 샘플을 가져온다.
        item = self.data[idx]

        # 샘플 dict에서 리뷰 텍스트를 꺼낸다.
        text = item["text"]

        # 샘플 dict에서 정답 label을 꺼내 int로 맞춘다.
        label = int(item["label"])

        # tokenizer로 문자열을 token id 리스트로 바꾼다.
        # add_bos_eos=True를 지원하는 tokenizer라면 문장 시작/끝 토큰도 함께 붙인다.
        token_ids = self.tokenizer.encode(text, add_bos_eos=True)

        # token 길이가 max_length보다 길면 뒤쪽을 잘라서 고정 길이에 맞춘다.
        token_ids = token_ids[: self.max_length]

        # token 길이가 max_length보다 짧으면 pad_id를 뒤에 채워 고정 길이에 맞춘다.
        if len(token_ids) < self.max_length:
            # 부족한 길이만큼 padding token id를 만든다.
            padding = [self.pad_id] * (self.max_length - len(token_ids))

            # 원래 token 뒤에 padding을 붙인다.
            token_ids = token_ids + padding

        # 모델 입력은 torch.long dtype의 1차원 tensor여야 한다.
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # DataLoader가 label을 자동으로 tensor로 묶을 수 있게 Python int label을 함께 반환한다.
        return input_ids, label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        # GPT hidden state의 마지막 차원 크기, 즉 token embedding 차원을 가져온다.
        hidden_size = gpt_model.config["emb_dim"]

        # 분류 head에 들어가기 전 overfitting을 줄이기 위한 dropout layer를 만든다.
        self.dropout = nn.Dropout(drop_rate)

        # 문장 대표 hidden vector를 num_labels 개의 분류 logit으로 바꾸는 Linear layer를 만든다.
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # GPTModel.forward()는 lm_head까지 통과한 vocab logits를 반환하므로,
        # 분류에서는 lm_head 전 hidden state가 필요해서 backbone 내부 layer를 직접 통과시킨다.

        # 입력 길이가 GPT position embedding의 context_length보다 길면 마지막 context_length 토큰만 사용한다.
        input_ids = input_ids[:, -self.gpt.context_length:]

        # token id를 token embedding + position embedding으로 바꾼다.
        hidden_states = self.gpt.embedding(input_ids)

        # GPT의 TransformerBlock들을 순서대로 통과시켜 문맥이 반영된 hidden state를 만든다.
        for block in self.gpt.blocks:
            # 각 block은 causal self-attention과 feed-forward를 적용한다.
            hidden_states = block(hidden_states)

        # 마지막 layer normalization을 적용해 GPT backbone의 최종 hidden state를 얻는다.
        hidden_states = self.gpt.final_layernorm(hidden_states)

        # 문장 전체를 대표할 vector로 마지막 token 위치의 hidden state를 사용한다.
        pooled_output = hidden_states[:, -1, :]

        # 분류 head 앞에서 dropout을 적용한다.
        pooled_output = self.dropout(pooled_output)

        # pooled vector를 class별 점수 logits로 변환한다.
        logits = self.classifier(pooled_output)

        # labels가 주어지지 않은 추론 상황이면 logits만 반환한다.
        if labels is None:
            return logits

        # labels가 Python list 등으로 들어올 수도 있으니 tensor가 아니면 tensor로 바꾼다.
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)

        # labels가 tensor라면 모델 입력과 같은 device로 옮기고 dtype을 long으로 맞춘다.
        labels = labels.to(input_ids.device, dtype=torch.long)

        # 다중 class 분류 loss인 cross entropy를 계산한다.
        loss = F.cross_entropy(logits, labels)

        # 학습 루프에서 loss와 logits를 모두 쓸 수 있도록 tuple로 반환한다.
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    # 모델 파라미터가 입력 tensor와 같은 device에 있도록 옮긴다.
    model.to(device)

    # dropout 등이 학습 모드로 동작하도록 설정한다.
    model.train()

    # epoch 전체 loss 합계를 저장할 변수를 만든다.
    total_loss = 0.0

    # 정확도 계산을 위해 맞힌 개수를 저장할 변수를 만든다.
    correct = 0

    # 정확도 계산을 위해 전체 샘플 개수를 저장할 변수를 만든다.
    total = 0

    # 평균 loss 계산을 위해 처리한 batch 수를 저장한다.
    processed_batches = 0

    # train_loader에서 batch를 하나씩 꺼내 학습한다.
    for input_ids, labels in train_loader:
        # input token id tensor를 학습 device로 옮긴다.
        input_ids = input_ids.to(device)

        # label tensor를 학습 device로 옮기고 long dtype으로 맞춘다.
        labels = labels.to(device, dtype=torch.long)

        # 이전 batch에서 남아 있는 gradient를 지운다.
        optimizer.zero_grad()

        # 모델 forward로 loss와 class logits를 얻는다.
        loss, logits = model(input_ids, labels=labels)

        # loss를 기준으로 gradient를 계산한다.
        loss.backward()

        # optimizer가 gradient를 이용해 모델 파라미터를 업데이트한다.
        optimizer.step()

        # 현재 batch loss를 Python 숫자로 꺼내 누적한다.
        total_loss += loss.item()

        # 평균 계산을 위해 처리 batch 수를 1 늘린다.
        processed_batches += 1

        # 가장 큰 logit을 가진 class를 예측 label로 고른다.
        preds = torch.argmax(logits, dim=-1)

        # 예측과 정답이 같은 샘플 개수를 누적한다.
        correct += (preds == labels).sum().item()

        # 전체 샘플 수를 누적한다.
        total += labels.numel()

    # batch가 없으면 평균 loss와 accuracy를 계산할 수 없으므로 nan과 0.0을 반환한다.
    if processed_batches == 0:
        return float("nan"), 0.0

    # batch별 loss 평균을 계산한다.
    avg_loss = total_loss / processed_batches

    # 전체 샘플 중 맞힌 비율을 accuracy로 계산한다.
    accuracy = correct / total if total > 0 else 0.0

    # 평균 loss와 accuracy를 반환한다.
    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    # 모델 파라미터가 입력 tensor와 같은 device에 있도록 옮긴다.
    model.to(device)

    # dropout 등이 꺼지도록 평가 모드로 설정한다.
    model.eval()

    # 전체 loss 합계를 저장할 변수를 만든다.
    total_loss = 0.0

    # 맞힌 샘플 개수를 저장할 변수를 만든다.
    correct = 0

    # 전체 샘플 개수를 저장할 변수를 만든다.
    total = 0

    # 평균 loss 계산을 위해 처리한 batch 수를 저장한다.
    processed_batches = 0

    # 평가는 gradient가 필요 없으므로 no_grad 안에서 실행한다.
    with torch.no_grad():
        # data_loader에서 batch를 하나씩 꺼내 평가한다.
        for input_ids, labels in data_loader:
            # input token id tensor를 평가 device로 옮긴다.
            input_ids = input_ids.to(device)

            # label tensor를 평가 device로 옮기고 long dtype으로 맞춘다.
            labels = labels.to(device, dtype=torch.long)

            # 모델 forward로 loss와 class logits를 얻는다.
            loss, logits = model(input_ids, labels=labels)

            # 현재 batch loss를 Python 숫자로 꺼내 누적한다.
            total_loss += loss.item()

            # 평균 계산을 위해 처리 batch 수를 1 늘린다.
            processed_batches += 1

            # 가장 큰 logit을 가진 class를 예측 label로 고른다.
            preds = torch.argmax(logits, dim=-1)

            # 예측과 정답이 같은 샘플 개수를 누적한다.
            correct += (preds == labels).sum().item()

            # 전체 샘플 수를 누적한다.
            total += labels.numel()

    # batch가 없으면 평균 loss와 accuracy를 계산할 수 없으므로 nan과 0.0을 반환한다.
    if processed_batches == 0:
        return float("nan"), 0.0

    # batch별 loss 평균을 계산한다.
    avg_loss = total_loss / processed_batches

    # 전체 샘플 중 맞힌 비율을 accuracy로 계산한다.
    accuracy = correct / total if total > 0 else 0.0

    # 평균 loss와 accuracy를 반환한다.
    return avg_loss, accuracy
