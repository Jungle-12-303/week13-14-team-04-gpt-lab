# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length

        # token_embedding, position_embedding, dropout을 정의하세요.

        # 토큰/위치 임베딩 레이어 만들기 = 임베딩 레이어가 하는 일은 nn.Embedding 내부에서 알아서 처리
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

        # raise NotImplementedError("InputEmbedding.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs
            batch_size: 한 번에 처리하는 문장 개수
            seq_len: 각 문장 안의 토큰 개수 
            -> 출력 shape 정보 

            x[문장 idx, 토큰 idx] = token id

        Returns:
            (batch_size, seq_len, emb_dim)
            emb_dim: 임베딩 차원, 단어 하나를 몇 차원의 밀집 벡터로 표현할 것인지 
        """
        # 토큰 id를 토큰 임베딩 레이어에 넣기 
        # -> 토큰 id는 x 안에 들어있음 => x 자체가 token id들의 2차원 텐서 
        # => nn.Embedding이 (batch_size, seq_len) 모양의 정수 텐서를 한 번에 받아서 각 id를 전부 벡터로 바꿔줌 
        result_token = self.token_embedding(x)

        # seq_len 길이만큼의 위치 id 목록을 만들어서 위치 임베딩 레이어에 넣기 
        # -> 위치 임베딩 레이어가 원하는 값: [0, 1, 2, 3, ..., seq_len - 1]
        # => seq_len가 5라면 위치 id는 [0, 1, 2, 3, 4] -> 이걸 위치 임베딩 레이어에 넣으면 각 위치마다 벡터가 나옴 
        # 0부터 seq_len - 1까지 위치 번호를 넣을 리스트 
        positions = []

        # 입력 x에서 현재 문장 길이 seq_len를 알아냄  
        # 튜플 언패킹 X -> 텐서 x의 모양을 알아냄 
        batch_size, seq_len = x.shape

        for i in range(0, seq_len):
            if i < seq_len:
                positions.append(i)
            else:
                break
        
        # 위치 번호 리스트를 torch longtensor(int64)로 변환 
        position_ids = torch.tensor(positions, dtype=torch.long)

        # 위치 번호를 위치 임베딩 레이어에 넣기 
        result_position = self.position_embedding(position_ids)

        # 토큰 임베딩 결과와 위치 임베딩 결과를 더한다
        embedding_result = result_token + result_position

        # 마지막에 dropout을 통과시켜서 반환
        result_dropout = self.dropout(embedding_result)

        return result_dropout
        # raise NotImplementedError("InputEmbedding.forward를 구현하세요.")
