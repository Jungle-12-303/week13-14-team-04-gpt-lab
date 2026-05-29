# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""
    # batch 전체를 섞어서 보는 게 아니라 각 토큰 벡터 하나하나를 독립적으로 정규화 
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # x[batch][seq]: batch, seq로 2차원 배열 구성 
        # hidden: 각 배열 안에 들어가있는 벡터 -> x[batch][seq]에 벡터가 들어있고, 벡터는 내부에 hidden개의 요소를 가지고 있음
        # LayerNorm: 입력 x에 존재하는 모든 벡터를 순회하면서 각 벡터 내부 요소들로 평균과 분산을 구해야 함
        # -> 개념적으론 x의 벡터들을 순회해야 하지만, PyTorch 가 제공하는 텐서 연산 함수 사용하면 간단히 구할 수 있음
        # => 텐서 연산할 때 마지막 차원을 기준으로 연산을 시키면(dim = -1) 알아서 게산해줌 

        # dim: 어느 차원 기준으로 계산할지
        # keepdim: 계산한 차원을 남길지 말지 - Fasle: 해당 차원 사라짐, True: 해당 차원 크기 1로 남음 
        # unbiased: 분산을 구할 때 나누는 값을 강제 -> False: N으로 나눔, True: N - 1로 나눔(표본 분산을 구할 때 사용 => 전체 집단을 다 보는 게 아니라 일부 샘플만 보고 전체 분산을 추정하려고 할 때 보정하는 방식)
        # x.mean(dim, keepdim): 평균 계산, dim =  -1, keepdim = True 
        mean = x.mean(dim = -1, keepdim = True)

        # x.var(dim, keepdim, unbaised): 분산 계산, dim =  -1, keepdim = True, unbiased = False
        # unbiased = False: 표본으로 전체를 추정하는 것이 아닌, 현재 벡터 자체를 정규화하기 때문에 False 
        # 분산: 각 값이 평균에서 얼마나 떨어져 있는지 제곱해서 평균낸 값 
        var = x.var(dim = -1, keepdim = True, unbiased = False)

        # 표준 편차 = 분산의 제곱근 = torch.sqrt(var)
        # -> 분산을 구할 때 차이를 제곱 -> 원래 단위로 되돌리기 위해 제곱근 사용 
        # var가 그냥 숫자가 아니라 torch tensor이기 때문에 math.sqrt가 아닌 torch.sqrt 사용 
        std_deviation = torch.sqrt(var + self.eps)

        # 정규화된 입력 x = (x - mean) / std_deviation
        # -> std_deviation로 나누기 때문에 std_deviation가 0이거나 너무 작은 값이 되면 안 됨 -> var에 eps를 더한 상태로 제곱근 사용
        x_norm = (x - mean) / std_deviation

        # -> 정규화 과정을 거치면 마지막 차원 기준으로 평균 = 대략 0, 분산 = 대략 1로 맞춰짐 
        # 대략인 이유?: eps를 더하기 때문 

        # 정규화된 값에 학습 가능한 스케일 gamma 곱하고, 이동값 beta 더함
        # -> 정규화만 하면 표현력이 제할될 수 있으니, 모델이 필요하면 다시 크기와 위치를 조절할 수 있게 해 주는 단계 
        out_norm = x_norm * self.gamma + self.beta

        return out_norm 
        # raise NotImplementedError("LayerNorm.forward를 구현하세요.")


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        raise NotImplementedError("GELU.forward를 구현하세요.")


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        raise NotImplementedError("FeedForward.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        raise NotImplementedError("FeedForward.forward를 구현하세요.")


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        raise NotImplementedError("GPTModel.__init__을 구현하세요.")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        raise NotImplementedError("GPTModel.forward를 구현하세요.")


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    raise NotImplementedError("generate_text_simple을 구현하세요.")
