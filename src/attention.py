# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn
import math

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
        d_model: int,
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
        # qkv projection, output projection, dropout을 정의하세요.

        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)
        """
        q_proj: 입력 x에서 Query를 만들 준비
        k_proj: 입력 x에서 Key를 만들 준비
        v_proj: 입력 x에서 Value를 만들 준비
        out_proj: 여러 head 결과를 합친 뒤 다시 정리할 준비
        attn_dropout: attention 비율에 dropout을 걸 준비
        resid_dropout: 최종 출력에 dropout을 걸 준비
        """

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(out))
        if return_attention_weights:
            return out, weights
        return out