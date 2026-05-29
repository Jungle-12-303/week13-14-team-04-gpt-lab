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
        self.qkv_bias = qkv_bias

        # qkv projection, output projection, dropout을 정의하세요.
        # qkv_projection: Q, K, V를 만들기 위한 선형 레이어
        # Liner(self.d_model, 3 * d_model): 토큰 하나 당 d_model 길이의 벡터가 들어옴, 
        # -> 3 * d_model: Q(Query), K(Key), V(Value) 세 개를 한 번에 만들기 위해 출력 자원을 3 * d_model로 설정
        self.qkv_projection = nn.Linear(self.d_model, 3 * self.d_model, bias = self.qkv_bias)

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
            => x: (B, T, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환

            미래 위치: 현재 토큰보다 뒤에 있는 토큰 위치
            미래 위치를 mask 처리: 단어 예측 모델이 예측할 단어를 미리 알지 못하게 함 
        """
        # self가 아닌 입력 x에서 꺼낸 값을 사용해야 함
        # _ -> 받긴 받는데 안 쓸 값이라고 무시
        # -> x.shape 값이 3개라서 3개 중 두 개만 사용한다면 하나는 무시 처리 
        batch_size, seq_len, _ = x.shape

        # __init__에서 q, k, v를 한번에 만들기 때문에 일단 입력 x를 넣어줌 
        qkv_projection = self.qkv_projection(x)

        # chunk(나눌 텐서, 몇 조각으로 나눌지, 어느 차원을 기준으로 나눌지): 텐서를 특정 차원을 기준으로 n개 조각으로 나눔 
        # 양수 인덱스: dim = 0 -> B, dim = 1 -> T, dim = 2 -> 3 * d_model(마지막 차원)
        # 음수 인덱스: dim = -3 -> B, dim = -2 -> T, dim = -1 -> 3 * d_model(마지막 차원)
        # => 3 * d_model을 d_model 3개로 나눔 
        q, k, v = torch.chunk(qkv_projection, 3, dim = -1)

        # d_model을 head로 나누기 -> 차원을 쪼갬 
        # d_model = head_dim * n_heads
        # head_dim = d_model // n_heads
        # (B, T, d_model) -> reshape: (B, T, n_heads, head_dim) -> transpose: (B, n_heads, T, head_dim) 
        # 텐서의 형상을 바꿀 땐 view, reshape 둘 중 하나 사용
        # -> transpose로 차원 순서를 바꿔줘야 하기 때문에 유연한 reshape 사용해야 함(permute 사용해도 ㄱㅊ)
        # transpose: 두 개의 차원만 서로 위치를 바꿈, permute: 여러 차원의 순서를 한 번에 원하는 대로 재배치 
        # => 순서까지 바꾸는 이유: attention 계산할 때 편하게 하기 위해 -> attention score를 head 별로 계산하기 때문 
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim) 
        q = q.transpose(1, 2) # 내가 무엇을 찾고 싶은지
        k = k.transpose(1, 2) # 나는 어떤 특징으로 검색될 수 있는지 
        v = v.transpose(1, 2) # 실제로 가져갈 정보 

        # attention score 계산할 때만 k의 원소 순서를 바꿈 -> k^t
        # k^T: (B, n_heads, T, head_dim) -> (B, heads, head_dim, T)
        k_t = k.transpose(-2, -1)

        # attention score 계산
        # sqrt -> 제곱근 
        # attention score shape: (B, heads, T, T)
        attention_score = (q @ k_t) / math.sqrt(self.head_dim)

        # attention 행렬을 먼저 만들고 mask 씌워주기 
            # 미래 위치를 볼 수 없게 mask 처리 
            # 입력 벡터 x가 아닌 나중에 만드는 attention score 행렬에 mask를 씌워야 함 
            # (B, heads, T, T)에서 마지막 두 차원 (T, T)에서 미래 위치를 막아야 함 
        if causal_mask:
            # torch.ones(행 크기, 열 크기, device = 어느 장치인지)
            # -> seq_len * seq_len 크기의 ones 텐서(전부 1로 채운 텐서)를 x와 같은 device에 만든다 
            # => 값 중요 X, 그냥 미래 위치를 설정하기 위한 틀 => 1로 채운 텐서 사용 
            # device는 반드시 device = x.device 형태의 키워드 인자로 넘겨야 함 
            input_tensor = torch.ones(seq_len, seq_len, device = x.device)

            # torch.triu(input, diagonal=0): upper triangular, 미래 위치 차단용
            # input: 2차원 텐서, diagonal=0: 주 대각선 포함, diagonal=1: 주 대각선 바로 위부터 선택, diagonal=1: 미래 위치만 선택(자기 자신 제외)
            # -> 대각선 위쪽만 남김(미래 위치)
            # mask를 만들기 위한 기존 tensor에 사용하기 때문에 torch.triu로 사용 
            future_mask = torch.triu(input_tensor, diagonal=1)

            # masked_fill에 넣기 위해 bool 타입으로 변환
            # 텐서 전체를 bool 하나로 만드는 게 X -> 텐서의 dtype을 bool tensor로 바꿔줘야 함 
            # => 텐서의 각 원소를 bool 타입으로 변환 
            # 텐서.dtype: 텐서의 타입 확인
            # 텐서.bool() or 텐서.to(torch.bool): bool 타입으로 변환
            future_mask = future_mask.bool()

            # masked_fill(bool mask, 채울 값): mask 위치를 아주 작은 값 또는 -inf로 채움
            # attention_score에 적용할 내용이기 때문에 torch 함수로 사용하기 보다 score 텐서의 메서드로 사용하는 게 좋음 
            # inf: 무한대
            attention_score = attention_score.masked_fill(future_mask, float("-inf"))
            
        # softmax로 attention weight 만들기
        # attention_weights: 각 토큰들이 어떤 토큰들을 얼마나 참고할지에 대한 수치 -> softmax를 통해 참고 비율을 정함 
        # attention_score에 적용할 내용이기 때문에 torch 함수로 사용하기 보다 score 텐서의 메서드로 사용하는 게 좋음 
        # 만들기는 무조건 진행!!! 반환 할지 말지는 return_attention_weights에 따라 달라짐
        # torch 함수로 쓸 때: 적용할 점수.softmax(적용할 점수, 확률화할 기준 차원이 어디인지)
        # attention_score 메소드로 쓸 때: 적용할 텐서.softmax(확률화할 기준 차원이 어디인지)
        attention_weights = attention_score.softmax(dim = -1)

        # attention_weights dropout 통과시키기 
        # dropout: 과적합 방지, 학습할 때 일부 값을 랜덤하게 꺼버리는 기법 -> 단순히 0으로 만드는 게 아니라 남은 값들을 스케일링 해서 전체 기대값 유지되도록 처리 
        # 왜 사용?: Transformer가 파라미터도 많고 표현력이 강함 -> 데이터가 작거나 학습이 길어지면 쉽게 외울 수 있어서 dropout을 통해 노이즈 주고 학습 제대로 할 수 있게 함 
        attention_weights = self.dropout(attention_weights)

        # 실제로 전달될 내용 정보 v에 가중치 곱하기 
        # -> softmax로 정해진 참고 비율대로 v를 섞어서 결과 만듦 -> attention output 또는 context vector
        # shape: (B, heads, T, T) @ (B, heads, T, head_dim) = (B, heads, T, head_dim)
        # -> (T, T) @ (T, head_dim) = (T, head_dim) => 각 head마다 토큰별 context vector가 만들어짐 
        attention_output = attention_weights @ v

        # :94부터 나눈 head 다시 합치기 
        # 1. 차원 순서를 바꾸기
        # (B, heads, T, head_dim) -> (B, T, heads, head_dim)
        attention_output = attention_output.transpose(1, 2)

        # 2. 차원을 합치기
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)

        # 차원을 합친 결과를 다시 선형 변환
        output_projection = self.output_projection(attention_output)
        
        # TODO: Dropout을 또 쓸지? 말지?
        # 이미 dropout 레이어를 attention score에 썼다면 projection 뒤에선 안 쓸 수도 있음 

        # torch.Tensor | tuple[torch.Tensor, torch.Tensor] -> torch.Tensor 하나 반환 or torch.Tensor 두 개가 들어있는 tuple 반환
        if return_attention_weights:
            # softmax까지 끝난 attention weight를 tuple로 함께 반환 
            return (output_projection, attention_weights)
        else:
            return output_projection

        # raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")
