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

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var)
        output = self.gamma * x_hat + self.beta
        return output
        raise NotImplementedError("LayerNorm.forward를 구현하세요.")


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        # 근사식
        output = 0.5 * x *(1 + torch.tanh(((2.0 / torch.pi) ** 0.5) * (x + 0.044715 * torch.pow(x,3))))
        # torch 연산
        # gelu = nn.GELU()
        # y = gelu(x)

        return output
        raise NotImplementedError("GELU.forward를 구현하세요.")


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            GELU(),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout)
        )
        # raise NotImplementedError("FeedForward.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.layers(x)
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
        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        self.dropout = nn.Dropout(drop_rate)

        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, drop_rate)

        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x,causal_mask)
        x = shortcut + self.dropout(x)

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        logits = shortcut + x
        return logits 
        raise NotImplementedError("TransformerBlock.forward를 구현하세요.")

class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        self.embed = InputEmbedding(config["vocab_size"], config["emb_dim"], config["context_length"], config["drop_rate"])
        self.tfb = nn.Sequential(*[TransformerBlock(config["emb_dim"],config["n_heads"],config["drop_rate"],config["qkv_bias"]) for _ in range(config["n_layers"])])
        self.norm = LayerNorm(config["emb_dim"])
        self.linear = nn.Linear(config["emb_dim"],config["vocab_size"])
        self.ce = nn.CrossEntropyLoss()

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
        idx = self.embed(idx)
        idx = self.tfb(idx)
        idx = self.norm(idx)
        logits = self.linear(idx)
        if targets != None:
            B, T, C = logits.shape
            ce = self.ce(logits.view(B*T,C), targets.view(B*T))
            return (ce,logits)
        else:
            return logits
        raise NotImplementedError("GPTModel.forward를 구현하세요.")


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:,-1,:]
        probs = torch.softmax(logits,dim=-1)
        last_word = torch.argmax(probs,dim=-1,keepdim=True)
        idx = torch.cat((idx,last_word),dim=1)
    return idx
    # raise NotImplementedError("generate_text_simple을 구현하세요.")