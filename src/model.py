# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import math
import torch
import torch.nn as nn
# torch.nn.functional.gelu 사용을 위한 functional API import 
# import torch.nn.functional as F
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
        """ 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
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
    # Relu와의 차이점: Relu는 음수는 버리고 양수만 살리는 활성화 함수 -> GELU는 버리지 않고 입력값을 부드럽게 조금씩 통과시키는 활성화 함수
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        # 근사식: 복잡한 현상이나 계산하기 어려운 함수의 값을 다루기 쉽고 직관적인 다항식으로 대체하여 어림하는 식 
        
        # tanh 근사식 -----------------
        # 1. 입력 x를 받는다
        x_origin = x

        # 2. x의 세제곱 항을 만든다 
        x_cubed = x_origin ** 3

        # 3. x와 x^3을 어떤 상수들과 조합함 -> x_origin에 0.044715 * x_cubed를 더하고 그 전체에 sqrt(2 / pi)를 곱한다
        # -> 직선적인 x만 쓰면 원래 GELU 곡선을 잘 따라가기 어렵기 때문에 x_cubed 항을 조금 섞어서 곡선을 더 비슷하게 만든다
        # tanh 근사식에서 사용하는 대표 상수: sqrt(2 / pi) and 0.044715
        # sqrt(2 / pi): tanh 안쪽 전체에 곱해지는 스케일 상수 -> 입력값의 크기를 조절해서 tanh 곡선이 GELU 모양에 가깝게 나오도록 해줌 
        # 0.044715: x_cubed 앞에 붙는 보정 계수 -> 곡률 보정용 계수 -> x^3의 영향이 너무 커지지 않게 작게 조절(x^3 항의 세기를 조절)하면서, 그래도 GELU 모양에 가깝게 만들도록 도와줌 
        # 곡선을 따라간다 = 진짜 GELU 함수가 만드는 출력값과 tanh 근사식이 만드는 출력값이 최대한 비슷하게 나오게 한다 
        # -> GELU는 어려운 함수라 매번 정확한 계산을 하면 비용이 더 들어가기 때문에 근사식 사용 
        formula = math.sqrt(2 / math.pi) * (x_origin + 0.044715 * x_cubed)
        
        # 4. 그 결과를 tanh 안에 넣고 1을 더함(tanh 적용)
        gate = torch.tanh(formula) + 1

        # 5. 마지막으로 x와 0.5를 곱해 스케일을 맞춰서 최종 GELU 값을 만듦  
        # -> x * 부드러운 gate(tanh로 만들어지는 값)
        # => 입력이 매우 크면 거의 1에 가까워 x 잘 통과
        # => 입력이 매우 작거나 음수면 덜 통과
        GELU_out = 0.5 * x * gate

        # +) 수식 한 줄 정리 
        # GELU_out = 0.5 * x * (torch.tanh(math.sqrt(2 / math.pi) * (x_origin + 0.044715 * x_cubed)) + 1)

        # torch 내장 연산 ----------------
        # Pytorch에서 제공하는 GELU 계산 기능 사용
        # torch.nn.GELU or torch.nn.functional.gelu 사용
        # torch.nn.GELU: 모듈 클래스, 보통 __init__에서 만들어두고 forward에서 호출하는 스타일
        # torch.nn.functional.gelu: 함수 형태, forward 안에서 입력 x를 바로 넣어서 계산
        # -> 이번 프로젝트는 torch.nn.functional.gelu 사용이 조금 더 자연스러움 
        # -> functional API를 쓰려면 보통 torch.nn.functional을 어떤 이름으로 가져오거나, 전체 경로로 접근해야 함
        # GELU_out = F.gelu(x)
        
        return GELU_out
        # raise NotImplementedError("GELU.forward를 구현하세요.")


class FeedForward(nn.Module):
    """ Transformer FFN: Linear -> GELU -> Linear -> Dropout."""
    # 입력 벡터 크기가 d_model이면, 중간에서 잠깐 더 큰 차원으로 확장했다가 다시 d_model로 줄이는 작은 MLP
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.

        # Linear(입력 feature 수, 출력 feature 수)

        # d_model -> mult*d_model
        self.first_Linear = nn.Linear(d_model, mult*d_model)

        # GELU -> __init__엔 실제 입력 x 없어서 그냥 인자 안 받음 
        self.gelu = GELU()

        # mult*d_model -> d_model
        self.second_Linear = nn.Linear(mult*d_model, d_model) 

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # raise NotImplementedError("FeedForward.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ FeedForward 네트워크를 통과시킵니다."""

        # 인자 x를 통째로 넣어줌 -> 알아서 각 토큰의 hidden 벡터 변환함 
        first_out  = self.first_Linear(x)

        gelu_out = self.gelu(first_out)

        second_out = self.second_Linear(gelu_out)

        result = self.dropout(second_out)

        return result
        # raise NotImplementedError("FeedForward.forward를 구현하세요.")


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

        # 첫 번째 
        self.first_layernorm = LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias)
        
        # 첫 번째 결과 dropout
        self.dropout = nn.Dropout(drop_rate)

        # 두 번째 
        self.second_layernorm = LayerNorm(d_model)
        
        # 두 번째 결과 
        self.ffn = FeedForward(d_model, drop_rate)

        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # Residual connection: 블록이 계산한 결과에 원래 입력을 다시 더해주는 연결
        # 입력 x -> 연산 -> 연산 결과
        # 연산 결과 + 입력 x -> output
        # 깊은 모델에서 정보가 너무 많이 바뀌거나 학습이 어려워지는 걸 막기 위해 수행
        # -> 모델이 새로 계산한 변화량만 더하도록 하면 필요할 땐 입력 정보를 거의 그대로 유지할 수 있음  

        before_norm = self.first_layernorm(x)

        attention_result = self.attention(before_norm, causal_mask)

        attention_result = self.dropout(attention_result)

        # attention 이후 Residual connection
        x = x + attention_result

        after_norm = self.second_layernorm(x)

        ffn_result = self.ffn(after_norm)

        # FFN 이후 Residual connection
        x = x + ffn_result

        return x
        # raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # embedding, blocks, final layernorm, lm_head를 정의하세요.

        # GPT_CONFIG_SMALL = {
        #     "vocab_size": 1000,
        #     "context_length": 64,
        #     "emb_dim": 64,
        #     "n_heads": 4,
        #     "n_layers": 2,
        #     "drop_rate": 0.1,
        #     "qkv_bias": False,
        # }

        # config에서 각 요소들 꺼내오기 
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.d_model = config["emb_dim"] # emb_dim == d_model
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.drop_rate = config["drop_rate"]
        self.qkv_bias = config["qkv_bias"]

        # 입력: config->vocab_size/emb_dim/context_length/drop_rate, 출력:[batch, seq, d_model]
        self.embedding = InputEmbedding(self.vocab_size, self.d_model, self.context_length, self.drop_rate)

        # block 하나씩 담을 리스트 
        blocks = []

        # TransformerBlock를 한 개가 아니라 layer 개수만큼 쌓기 
        for i in range(0, self.n_layers):
            # 출력: [batch, seq, d_model]
            block = TransformerBlock(self.d_model, self.n_heads, self.drop_rate, self.qkv_bias)

            # block 하나 만들 때마다 blocks에 넣기 
            blocks.append(block)
        
        # -> 모델 레이어로 등록되는 리스트 구조로 TransformerBlock 여러 개 저장
        # -> nn.ModuleList or nn.Sequential 사용
        # => block 담긴 리스트를 TransformerBlock()로 감싸기
        self.blocks = nn.ModuleList(blocks)
        
        # 모든 TransformerBlock을 통과 후 마지막으로 LayerNorm 한 번 더 적용 
        # LayerNorm에 들어가야하는 d_model 값은 config에 있음  
        self.final_layernorm = LayerNorm(self.d_model)

        # 마지막 hidden 벡터를 vocab 크기만큼의 logits로 바꾸는 Linear 
        # 출력: [batch, seq, vocab_size]
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        # raise NotImplementedError("GPTModel.__init__을 구현하세요.")

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
