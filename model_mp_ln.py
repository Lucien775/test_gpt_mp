import torch
import torch.nn as nn
import mptorch.quant as qpt
import mptorch.quant.functional as Q

from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class ModelLNMPConfig:
    """
    Configuration for the adaptive LayerNorm mixed-precision experiment.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        n_embd: Embedding dimension.
        block_size: Maximum context length.
        n_head: Number of attention heads.
        dropout: Dropout probability.
        n_layer: Number of transformer blocks.
        layer_format: Precision used by linear layers and matmuls.
        softmax_format: Precision used by softmax.
        LN_format: Low-precision format used by LayerNorm.
        LN_high_format: Higher-precision format used for selected values.
        proximity_threshold: Threshold for deciding whether a value uses the high-precision path.
        name: Experiment name.
    """
    vocab_size: int
    n_embd: int
    block_size: int
    n_head: int
    dropout: float
    n_layer: int
    layer_format: qpt.QAffineFormats
    softmax_format: qpt.QSoftmaxFormats
    LN_format: qpt.QLayerNormFormats
    LN_high_format: qpt.QLayerNormFormats
    proximity_threshold: float
    name: str


class MPLayerNorm(qpt.QLayerNorm):
    """
    Adaptive mixed-precision LayerNorm.

    The layer uses a low-precision path by default and switches to a higher-
    precision normalization path for values close to the mean.
    """

    def __init__(
        self,
        normalized_shape,
        formats,
        formats_high,
        proximity_treshold,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(
            normalized_shape,
            formats,
            eps,
            elementwise_affine,
            bias,
        )
        self.proximity_treshold = proximity_treshold
        self.formats_high = formats_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        deviation = torch.abs(x - mean)
        close_mask = deviation < self.proximity_treshold

        out_low = super().forward(x)

        if torch.any(close_mask):
            x_fp16 = x.to(torch.float16)
            mean_fp16 = x_fp16.mean(dim=-1, keepdim=True)
            var_fp16 = x_fp16.var(dim=-1, unbiased=False, keepdim=True)
            normalized_fp16 = (x_fp16 - mean_fp16) / torch.sqrt(var_fp16 + self.eps)

            weight_fp16 = self.weight.to(torch.float16)
            bias_fp16 = self.bias.to(torch.float16) if self.bias is not None else None

            out_fp16 = normalized_fp16 * weight_fp16
            if bias_fp16 is not None:
                out_fp16 = out_fp16 + bias_fp16

            out_fp16 = out_fp16.to(out_low.dtype)
            out = torch.where(close_mask, out_fp16, out_low)
        else:
            out = out_low

        return out

class Head(nn.Module):
    """Single attention head used by the adaptive LayerNorm experiment."""

    def __init__(self, config: ModelLNMPConfig, head_size: int):
        super().__init__()
        self.key = qpt.QLinear(config.n_embd, head_size, config.layer_format, False)
        self.query = qpt.QLinear(config.n_embd, head_size, config.layer_format, False)
        self.value = qpt.QLinear(config.n_embd, head_size, config.layer_format, False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.layer_format = config.layer_format
        self.softmax_format = config.softmax_format
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        k = self.key(x_flat).reshape(B, T, -1)
        q = self.query(x_flat).reshape(B, T, -1)
        v = self.value(x_flat).reshape(B, T, -1)

        wei = Q.qmatmul(q, k.transpose(-2, -1), self.layer_format) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = Q.qsoftmax(wei, dim=-1, formats=self.softmax_format)
        wei = self.dropout(wei)

        out = Q.qmatmul(wei, v, self.layer_format)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention block for the adaptive LayerNorm experiment."""
    def __init__(self, config: ModelLNMPConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = qpt.QLinear(head_size * config.n_head, config.n_embd, config.layer_format)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out2d = out.reshape(B * T, out.shape[-1])
        out = self.proj(out2d).reshape(B, T, -1)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward MLP used in the adaptive LayerNorm experiment."""
    def __init__(self, config: ModelLNMPConfig):
        super().__init__()
        self.net = nn.Sequential(
            qpt.QLinear(config.n_embd, 4 * config.n_embd, config.layer_format),
            nn.ReLU(),
            qpt.QLinear(4 * config.n_embd, config.n_embd, config.layer_format),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        out = self.net(x_flat).reshape(B, T, -1)
        return out


class Block(nn.Module):
    """Transformer block using adaptive LayerNorm."""
    def __init__(self, config: ModelLNMPConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = MPLayerNorm(
            config.n_embd,
            config.LN_format,
            config.LN_high_format,
            config.proximity_threshold,
        )
        self.ln2 = MPLayerNorm(
            config.n_embd,
            config.LN_format,
            config.LN_high_format,
            config.proximity_threshold,
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModelLNMP(nn.Module):
    """GPT model with adaptive LayerNorm mixed precision."""
    def __init__(self, config: ModelLNMPConfig):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = MPLayerNorm(
            config.n_embd,
            config.LN_format,
            config.LN_high_format,
            config.proximity_threshold,
        )
        self.lm_head = qpt.QLinear(config.n_embd, config.vocab_size, config.layer_format)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        logits = self.lm_head(x_flat).reshape(B, T, -1)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
