import torch
import torch.nn as nn
import mptorch.quant as qpt
import mptorch.quant.functional as Q

from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelSimilarityMPConfig:
    """Configuration for the similarity mixed-precision attention experiment.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        n_embd: Embedding dimension.
        block_size: Maximum context length.
        n_head: Number of attention heads.
        dropout: Dropout probability.
        n_layer: Number of transformer blocks.
        layer_format: Default precision for the output projection.
        LN_format: Precision used by LayerNorm layers.
        matmul_format_low: Low-precision format used for the initial attention scores.
        matmul_format_high: Higher-precision format used for selected scores.
        softmax_format_low: Softmax precision for the low-precision path.
        softmax_format_high: Softmax precision for the high-precision path.
        tau: Threshold above which the high-precision attention path is used.
        name: Experiment name.
        weighted: Whether the similarity-based selection uses weighted scores.
    """
    vocab_size: int
    n_embd: int
    block_size: int
    n_head: int
    dropout: float
    n_layer: int
    layer_format: qpt.QAffineFormats
    LN_format: qpt.QLayerNormFormats
    matmul_format_low: qpt.QAffineFormats
    matmul_format_high: qpt.QAffineFormats
    softmax_format_low: qpt.QSoftmaxFormats
    softmax_format_high: qpt.QSoftmaxFormats
    tau: float
    name: str
    weighted: bool = True



class Head(nn.Module):
    """One attention head using a similarity-based mixed-precision policy."""

    def __init__(self, config: ModelSimilarityMPConfig, head_size: int):
        super().__init__()
        self.key = qpt.QLinear(config.n_embd, head_size, config.layer_format, False)
        self.query = qpt.QLinear(config.n_embd, head_size, config.layer_format, False)
        self.value = qpt.QLinear(config.n_embd, head_size, config.layer_format, False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layer_format = config.layer_format
        self.matmul_format_low = config.matmul_format_low
        self.matmul_format_high = config.matmul_format_high
        self.softmax_format_high = config.softmax_format_high
        self.softmax_format_low = config.softmax_format_low
        self.tau = config.tau
        self.weighted = config.weighted

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
        k = self.key(x_flat).reshape(B, T, -1)
        q = self.query(x_flat).reshape(B, T, -1)
        v = self.value(x_flat).reshape(B, T, -1)

        causal_mask = self.tril[:T, :T]

        scores = Q.qmatmul(q, k.transpose(-2, -1), self.matmul_format_low) * k.shape[-1] ** -0.5
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        att = Q.qsoftmax(scores, dim=-1, formats=self.softmax_format_low)

        kappa = 2 * att * (1 - att)
        if self.weighted:
            kappa = kappa * torch.abs(scores)

        kappa = torch.nan_to_num(kappa, nan=0)
        wei = att.clone()
        mask = kappa > self.tau

        if torch.any(mask):
            scores_hp = Q.qmatmul(q, k.transpose(-2,-1), self.matmul_format_high)
            scores_hp = scores_hp.masked_fill(causal_mask == 0, float("-inf"))
            att_hp = Q.qsoftmax(scores_hp, dim=-1, formats=self.softmax_format_high)
            wei[mask] = att_hp[mask]
            wei = self.layer_format.output_quant(wei)

        wei = self.dropout(wei)

        out = Q.qmatmul(wei, v, self.layer_format)
        return out
                       


class MultiHeadAttention(nn.Module):
    """Parallel multi-head attention block for the similarity experiment."""

    def __init__(self, config: ModelSimilarityMPConfig, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )
        self.proj = qpt.QLinear(head_size * config.n_head, config.n_embd, config.layer_format)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out2d =  out.reshape(B * T, out.shape[-1])
        out = self.proj(out2d).reshape(B, T, -1)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config: ModelSimilarityMPConfig):
        super().__init__()
        self.net = nn.Sequential(
            qpt.QLinear(config.n_embd, 4 * config.n_embd, config.layer_format),
            nn.ReLU(),
            qpt.QLinear(4*config.n_embd, config.n_embd, config.layer_format),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)
        out = self.net(x_flat).reshape(B, T, -1)
        return out

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: ModelSimilarityMPConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = qpt.QLayerNorm(config.n_embd, config.LN_format)
        self.ln2 = qpt.QLayerNorm(config.n_embd, config.LN_format)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTModelSimilarityMP(nn.Module):
    """GPT model using a similarity mixed-precision attention strategy."""

    def __init__(self, config: ModelSimilarityMPConfig):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = qpt.QLayerNorm(config.n_embd, config.LN_format)
        self.lm_head = qpt.QLinear(config.n_embd, config.vocab_size, config.layer_format)

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

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



    

        