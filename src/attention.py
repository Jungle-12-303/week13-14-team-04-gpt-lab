# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int, # == emb_dim -> 토큰 하나를 표현하는 벡터 길이
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.drop_rate = drop_rate

        # qkv projection, output projection, dropout을 정의하세요.
        # qkv_projection: Q, K, V를 만들기 위한 선형 레이어
        # Liner(self.d_model, 3 * d_model): 토큰 하나 당 d_model 길이의 벡터가 들어옴, 
        # -> 3 * d_model: Q(Query), K(Key), V(Value) 세 개를 한 번에 만들기 위해 출력 자원을 3 * d_model로 설정
        self.qkv_projection = nn.Linear(self.d_model, 3 * self.d_model)

        # attention 계산이 끝난 뒤 결과를 다시 모델 차원으로 정리하는 선형 레이어 
        # 입력 d_model, 출력 d_model -> 중간에 head를 나눴다가 마지막에 다시 합침
        # 합친 후 shape는 다시 d_model이라 출력 d_model 
        self.output_projection = nn.Linear(self.d_model, self.d_model)

        # attention weight나 최종 projection 결과에 적용할 dropout 레이어
        self.dropout = nn.Dropout(self.drop_rate)

        #raise NotImplementedError("MultiHeadAttention.__init__을 구현하세요.")

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")
