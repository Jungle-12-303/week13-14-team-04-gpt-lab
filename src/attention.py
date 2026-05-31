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
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        self.W_queries = nn.Linear(d_model, d_model,bias = qkv_bias) 
        self.W_keys = nn.Linear(d_model, d_model,bias = qkv_bias)
        self.W_values = nn.Linear(d_model, d_model,bias = qkv_bias)
        self.output_projection = nn.Linear(d_model,d_model, bias = qkv_bias)
        self.dropout = nn.Dropout(drop_rate)
        # raise NotImplementedError("MultiHeadAttention.__init__을 구현하세요.")

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
        b, token, d_model = x.shape

        queries = self.W_queries(x)
        keys = self.W_keys(x)
        values = self.W_values(x)

        queries = queries.view(b,token,self.n_heads,self.head_dim)
        keys = keys.view(b,token,self.n_heads,self.head_dim)
        values = values.view(b,token,self.n_heads,self.head_dim)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        attn_mask = torch.triu(torch.ones([token, token],device = x.device),diagonal = 1).bool()
        if (causal_mask):
            attn_scores[:,:,attn_mask] = -torch.inf
        attn_weight = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim = -1)
        attn_weight = self.dropout(attn_weight)
        context_vec = (attn_weight @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, token, self.d_model)
        if return_attention_weights:
            return (self.output_projection(context_vec), attn_weight)
        else:
            return self.output_projection(context_vec)
        raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")
