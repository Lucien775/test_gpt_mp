import torch
import torch.nn as nn
import mptorch.quant as qpt
import mptorch.quant.functional as Q

from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class ModelBlockMPConfig:
    """
    Configuration for a mixed-precision GPT model with separate attention/feed-forward/LN formats.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        n_embd: Embedding dimension.
        block_size: Maximum context length.
        n_head: Number of attention heads.
        dropout: Dropout probability.
        n_layer: Number of transformer blocks.
        attn_layer_format: Precision format used by attention linear projections and matmuls.
        attn_softmax_format: Precision format used by attention softmax.
        ffwd_layer_format: Precision format used by the feed-forward MLP.
        ln_format: Precision format used by LayerNorm layers.
        name: Experiment name.
    """
    vocab_size: int
    n_embd: int
    block_size: int
    n_head: int
    dropout: float
    n_layer: int
    attn_layer_format: qpt.QAffineFormats
    attn_softmax_format: qpt.QSoftmaxFormats
    ffwd_layer_format: qpt.QAffineFormats
    ln_format: qpt.QLayerNormFormats
    name: str


class Head(nn.Module):
    """Single attention head for the mixed-precision block experiment."""
    def __init__(self, config: ModelBlockMPConfig, head_size: int):
        super().__init__()
        self.key = qpt.QLinear(config.n_embd, head_size, config.attn_layer_format, False)
        self.query = qpt.QLinear(config.n_embd, head_size, config.attn_layer_format, False)
        self.value = qpt.QLinear(config.n_embd, head_size, config.attn_layer_format, False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.attn_layer_format = config.attn_layer_format
        self.attn_softmax_format = config.attn_softmax_format
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        x_flat = x.reshape(B*T, C)
        k = self.key(x_flat).reshape(B, T, -1)
        q = self.query(x_flat).reshape(B, T, -1)
        v = self.value(x_flat).reshape(B, T, -1)

        wei = Q.qmatmul(q, k.transpose(-2, -1), self.attn_layer_format) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0,float("-inf"))
        wei = Q.qsoftmax(wei, dim=-1,formats=self.attn_softmax_format)
        wei = self.dropout(wei)

        out = Q.qmatmul(wei, v, self.attn_layer_format)  # (B, T, hs)
        return out
    


class MultiHeadAttention(nn.Module):
    """Multi-head attention block used in the mixed-precision model."""
    def __init__(self, config: ModelBlockMPConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )
        self.proj = qpt.QLinear(head_size * config.n_head, config.n_embd, config.attn_layer_format)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out2d =  out.reshape(B * T, out.shape[-1])
        out = self.proj(out2d).reshape(B, T, -1)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward MLP whose precision can differ from attention."""
    def __init__(self, config: ModelBlockMPConfig):
        super().__init__()
        self.net = nn.Sequential(
            qpt.QLinear(config.n_embd, 4 * config.n_embd, config.ffwd_layer_format),
            nn.ReLU(),
            qpt.QLinear(4 * config.n_embd, config.n_embd, config.ffwd_layer_format),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
        out = self.net(x_flat).reshape(B, T, -1)
        return out


class Block(nn.Module):
    """Transformer block for the mixed-precision GPT model."""
    def __init__(self, config: ModelBlockMPConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa   = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1  = qpt.QLayerNorm(config.n_embd, config.ln_format)
        self.ln2  = qpt.QLayerNorm(config.n_embd, config.ln_format)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModelBlockMP(nn.Module):
    """GPT model whose attention, feed-forward and LayerNorm use separate precisions."""

    def __init__(self, config: ModelBlockMPConfig):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f   = qpt.QLayerNorm(config.n_embd, config.ln_format)
        self.lm_head = qpt.QLinear(config.n_embd, config.vocab_size, config.attn_layer_format)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
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
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx