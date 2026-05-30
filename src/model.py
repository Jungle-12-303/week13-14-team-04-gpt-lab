# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import math
import torch
import torch.nn as nn
# torch.nn.functional.gelu와 torch.nn.functional.cross_entropy 사용을 위한 functional API import 
import torch.nn.functional as F
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
    # -> 표현력을 늘릴 수 있다
    # -> attention만 있으면 토큰들 사이에서 정보를 섞을 순 있지만 그 섞인 정보를 복잡하게 변환하는 능력 부족
    # -> FeedForward로 각 토큰 벡터에 작은 MLP를 적용해 중요한 특징은 더 키우고, 덜 중요한 특징은 줄이고, 여러 feature 조합을 비선형적으로 바꿈 
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.

        # Linear(입력 feature 수, 출력 feature 수)

        # d_model -> mult*d_model
        # 넓은 공간에서 정보를 조합해서 더 다양한 패턴 표현 
        self.first_Linear = nn.Linear(d_model, mult*d_model)

        # GELU -> __init__엔 실제 입력 x 없어서 그냥 인자 안 받음 
        self.gelu = GELU()

        # mult*d_model -> d_model
        # 여러 표현 조합을 만들고 다시 원래 크기로 압축 
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
    # 입력 토큰 벡터를 받아 1. attention, 2. FeedForward 수행 
    # 1. attention: 각 토큰이 앞에 있는 다른 토큰들을 참고해서 문맥 정보를 섞음 
    # 2. FeedForward: attention으로 섞인 각 토큰 벡터를 한 번 더 가공함 
    # -> 두 작업을 묶어서, 문맥을 반영한 더 좋은 토큰 표현을 만든다 

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
        # 이 단계에서 하는 일: 입력 숫자를 벡터 표현으로 바꾼 뒤 그 표현을 계속 더 많은 정보 표현으로 갱신
        # -> 토큰 id를 벡터로 바꾼 뒤 문맥을 반영하도록 계속 표현을 업데이트한다 

        # 각 단게별 상태
        # 1. 토큰 번호
        # 2. 토큰별 기본 벡터 + 위치 정보 -> embedding
        # 3.앞 토큰들을 참고한 문맥 반영 벡터 -> 첫 block
        # 4. 더 복잡한 문맥/패턴이 반영된 벡터 -> 이후 blocks
        # 5. 각 위치에서 다음 토큰 후보별 점수 -> lm_head 

        # idx: [batch, seq] -> token id => 벡터 X, 그냥 의미 없는 단어 번호
        # token id를 토큰/위치 임베딩 더한 벡터로 변환 
        # vocab_size, emb_dim, context_length, drop_rate 이런건 init에서 레이어를 만들 때 사용함
        # -> 레이어를 실행할 때 넣는 건 실제 입력 데이터 idx 
        # 출력: [batch, seq, d_model] 텐서 = block에 들어갈 첫 번째 hidden tensor 
        # -> 임베딩 과정을 거쳐 토큰 아이디를 벡터로 바꾸고 위치 정보를 더함 
        # => hidden tensor: 각 토큰 위치 단계에서 d_model 개의 숫자 벡터를 저장한 텐서 
        # [batch, seq, d_model] 텐서는 모델 내부에서 계속 같은 shape를 유지함 -> 안에 있는 숫자들만 바뀜 
        x = self.embedding(idx)

        # block이 담겨있는 리스트 blocks를 순회하며 설정값이 아닌 이전 단계에서 나온 hidden tensor를 넣어줌
        # 처음 hidden tensor: 토큰 의미 + 위치 벡터 
        # block 이후: 문맥을 반영한 토큰 벡터 
        for block in self.blocks:
            # 이전 단계에서 나온 hidden tensor: 첫 block -> 임베딩 함수 출력 값, 이후 blocks -> 이전 block의 출력 
            # -> 연쇄적으로 들어감 -> x = block(x) 
            # -> x: 원래 입력값 x가 아님!! 현재 단계의 표현을 담은 다른 변수임 
            x = block(x)
            # TransformerBlock 안에서 일어나는 일 1. Attention, 2. FeedForward
            # 1. Attention: 토큰들 사이 정보를 섞음 -> 각 토큰 벡터가 앞 토큰들의 벡터를 참고해서 업데이트 됨 = 문맥 반영
            # 2. FeedForward: 토큰끼리 섞지 않음, 각 토큰 위치의 벡터를 따로따로 MLP로 변환함 = 벡터 내부 가공 
            # -> 각 벡터 내부의 feature들을 비선형적으로 조합
            # => TransformerBlock 전체를 지나면 attention으로 토큰 간 문맥을 섞고, feedforward로 각 토큰 벡터 자체를 더 가공한 hidden tensor가 됨

        # 모든 TransformerBlock을 통과 후 마지막으로 LayerNorm 한 번 더 적용 
        # -> 각 토큰 벡터를 안정적인 스케일로 맞춤 
        # -> 입력: [batch, seq, d_model] 출력: [batch, seq, d_model] => shape 안 바뀌고, 각 토큰의 d_model 벡터 값만 정규화 됨 
        x = self.final_layernorm(x)

        # 마지막 hidden tensor를 vocab 크기만큼의 logits로 바꾸는 Linear 
        # -> 마지막 hidden tensor를 vocab 점수로 바꾸는 Linear
        # 출력: [batch, seq, vocab_size]
        # vocab: 모델이 알고 있는 토큰 사전
        # vocab_size: vocab에 들어있는 토큰(종류) 개수
        # GPT 모델에서 사용되는 vocab는?
        # -> vocab: 입력된 문장 다음에 올 후보 토큰 목록
        # -> vocab 점수: 그 후보 각각이 다음 토큰일 가능성에 대한 점수 
        # -> 토큰 위치 하나마다 vocab 전체 후보에 대한 점수표가 생김 => shape 변화
        # 각 토큰 위치마다 d_model 길이 벡터 hidden tensor가 있음
        # 이걸 vocabulary 크기만큼의 점수로 바꿈
        # -> hidden tensor의 요소 개수를 d_model에서 vocab_size로 바꿈
        # 전체 shape 바뀜 [batch, seq, d_model] -> [batch, seq, vocab_size] 
        # => 이 결과가 logits = 점수(확률X)
        # 어떤 위치에서 다음 토큰 후보가 1000개라면, 각 후보마다 점수가 하나씩 나옴
        # logits[batch_idx][seq_idx]
        # => 해당 위치에서 vocab 전체 토큰에 대한 점수 벡터

        # -> 이 logits를 소프트맥스 함수에 넣어서 확률을 계산함
        logits = self.lm_head(x)

        # targets가 있으면 logits와 targets 이용해서 loss 계산
        # targets: 모델이 맞춰야 하는 정답 토큰 id(진짜 다음에 올 토큰 id)
        # -> 학습 샘플 튜플의 그 targets(dataset, dataloader -> input 값과 쌍)
        # 모델 출력 logits: 각 위치마다 vocab 전체 후보에 대한 점수 
        # 첫 번째 위치 logits = vocab 전체 토큰 점수
        # target 첫 번째 값 = 정답 토큰 id
        if targets is not None:
            # loss: 손실함수의 loss, 모델의 예측 점수와 정답 target을 비교해서 얼마나 틀렸는지 숫자로 나타낸 값(Cross Entropy Loss)
            # -> loss 값이 작을 수록 모델이 정답에 가까운 예측을 했다는 의미
            # -> 학습할 땐 이 loss를 줄이는 방향으로 파라미터를 업데이트한다 
            
            # logits shape: [batch, seq, vocab_size]
            # targets shape: [batch, seq]
            # Cross Entropy가 기대하는 형태
            # -> 입력 logits: [N, vocab_size] -> N:예측해야 하는 전체 위치 개수
            # -> 정답 targets: [N]
            # -> batch, seq 총 2개의 차원을 하나로 합친다 -> N = batch * seq -> reshape(-1)
            # => 모든 배치의 모든 토큰 위치를 한 줄의 예측 문제 목록으로 만든다
            re_logits = logits.reshape(-1, self.vocab_size)
            
            # targets는 그냥 모든 차원을 하나로 합치면 됨 
            targets = targets.reshape(-1)

            loss = F.cross_entropy(re_logits, targets)

            return (loss, logits) 
        # targets가 없으면 여기서 logits만 반환
        else: 
            return logits

        #  => GPT Model 순전파는 학습에도 쓰이고 추론에도 쓰이는 공통 함수임 
        # 학습 단계(정답 있음)
        # model(idx, targets)로 호출
        # -> logits 만들고 targets와 비교=> loss 계산
        # 2. 추론/생성 단계
        # model(idx)or model(idx, targets=None)로 호출 => targets가 없음
        # -> 정답 없이 그냥 다음 토큰 점수만 보고 싶은 상황임 -> 그냥 logits만 반환 
        # -> 반환된 logits 안에서 다음 토큰을 고름 ex) 가장 점수가 높은 토큰 선택 or 확률적으로 샘플링 

        # raise NotImplementedError("GPTModel.forward를 구현하세요.")

def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    # 이미 있는 모델로 다음 토큰을 하나씩 생성하는 함수
    # idx: 현재까지의 토큰 id -> shape: [batch, current_seq_len]

    # max_new_tokens번 반복 -> 다음 토큰 하나를 예측 -> idx 뒤에 붙임 
    # 현재 idx에서 모델이 볼 수 있는 길이만 남김
    # GPT 최대 context 길이 때문에 너무 길어지면 마지막 뒤에서부터 context_size개 토큰만 사용
    # 토큰 생성 중엔 idx가 계속 길어지기 때문에 모델에 전체를 다 넣지 않고 마지막 context_size개만 넣음 
    # 뒤에서부터 쓰는 이유: 다음 토큰 예측엔 가장 최근 문맥이 중요, GPT의 위치 임베딩, 어텐션이 처리할 수 있는 최대 길이가 정해져있기 때문 
    # -> idx[:, -context_size:]
    # 모든 batch는 유지, 각 batch 문장의 마지막 context_size개 토큰만 사용 

    # 아래 과정을 max_new_tokens번 반복 
    # 잘라낸 입력을 모델에 넣음
    # 생성 단계 -> targets는 넣지 않음 
    # 모델 출력: logits
    # shape: [batch, seq, vocab_size]
    for i in range(max_new_tokens):
        # 마지막 context_size개 토큰만 가져오기(슬라이싱)
        idx_cond = idx[:, -context_size:]

        # 모델 출력 저장 -> [batch, seq, vocab_size]
        logits = model(idx_cond)

        # 마지막 위치의 logits만 확인 
        # 다음 토큰을 예측하려면 전체 seq 중 마지막 토큰 위치의 출력을 봐야 함
        # shape: [batch, vocab_size]
        # 마지막 위치 logits 가져오기 -> 마지막 seq 위치 -> [batch, vocab_size] 
        last_logits = logits[:, -1, :]

        # greedy 방식으로 가장 점수가 높은 토큰 id 고름
        # greedy: 확률적으로 뽑는 게 아니라 점수가 가장 큰 토큰 선택
        # torch 최대값의 인덱스를 구하는 연산 사용 
        # 가장 점수가 높은 vocab index 고름 -> [batch]  
        next_id = torch.argmax(last_logits, dim=-1)

        # idx를 붙이기 위해 차원을 하나 추가 -> [batch, 1]   
        next_id = next_id.unsqueeze(1)

        # 고른 토큰 id를 기존 idx 뒤에 붙임 
        # 기존 idx: [batch, current_seq_len]
        # 새 토큰: [batch, 1]
        # 붙인 결과: [batch, current_seq_len + 1]
        # 기존 idx 뒤에 새 토큰을 붙임
        # -> dim=1(=seq) 차원 방향으로 붙인다
        idx = torch.cat((idx, next_id), dim = 1)
         
    return idx
    # raise NotImplementedError("generate_text_simple을 구현하세요.")
