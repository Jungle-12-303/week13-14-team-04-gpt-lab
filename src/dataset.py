# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        # 만들 수 있는 학습 샘플 개수를 self._length에 저장하세요.
        # 학습 샘플 = 현재 토큰 구간을 input으로, 한 칸 뒤로 민 같은 길이 구간을 target으로 쓰는 한 쌍 
        # input, target 길이 = context_length 

        # 첫 start 인덱스 
        start = 0
        
        # 학습 샘플 개수 = input, target 쌍 수
        # 학습 샘플 1개 = input, target 한 쌍 
        samples = 0

        # for문으로 범위를 계속 돌면 나중에 start가 리스트 길이를 넘어가도 계속 검사
        # sample 생성 가능할 때까지만 반복하도록 하기
        #for i in range(0, len(token_ids)):
        while(len(self.token_ids) != 0):
            # start + self.context_length에서 target을 만들 수 있으면 input도 만들 수 있다
            # target은 start + 1에서 시작해서 context_length개 -> 마지막 인덱스는 길이 - 1 값!
            if start + self.context_length < len(token_ids):
                # sample 생성 가능
                samples += 1

                # 이후 start 인덱스 
                start += self.stride
            else:
                break

        self._length = samples

        # raise NotImplementedError("GPTDataset.__init__에서 self._length를 구현하세요.")

    def __len__(self) -> int:
        """전체 샘플 개수를 반환합니다."""

        return self._length
        # raise NotImplementedError("GPTDataset.__len__을 구현하세요.")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 input_ids와 target_ids를 LongTensor로 반환합니다.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        # idx번째 학습 샘플을 구하기 위한 start 인덱스 계산 
        start = idx * self.stride
    
        # input을 저장할 리스트
        input = []
        
        # target을 저장할 리스트
        target = []
        
        # start 인덱스부터 시작, context_length 까지 반복
        for i in range(start, start + self.context_length):
            # 현재 인덱스에서 샘플 만들 수 있는지 검사
            if start + self.context_length < len(self.token_ids):
                # start 인덱스부터 context_length까지 있는 요소들을 하나씩 리스트에 넣어줌 
                input.append(self.token_ids[i])
                target.append(self.token_ids[i + 1])
            else:
                break
        
        # 리스트를 LongTensor로 변환
        input_ids = torch.tensor(input, dtype=torch.long)
        target_ids = torch.tensor(target, dtype=torch.long)

        # 두 LongTensor를 tuple로 묶어서 반환 
        return tuple((input_ids, target_ids))
    
        # raise NotImplementedError("GPTDataset.__getitem__을 구현하세요.")


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """TODO: GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다."""
    raise NotImplementedError("create_dataloader를 구현하세요.")
