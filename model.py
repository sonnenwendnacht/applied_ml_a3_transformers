"""Configurable version of the causal Transformer in transformer.ipynb.

Module names are retained so the original experiment state dictionaries load
without conversion. The notebook remains the historical experiment record.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 500
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % 2 or self.n_embd % self.n_head:
            raise ValueError("n_embd must be even and divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        _, length, _ = x.shape
        key, query = self.key(x), self.query(x)
        weights = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:length, :length] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(
            self.proj(torch.cat([head(x) for head in self.heads], dim=-1))
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        precise = x.float()
        normalized = precise * torch.rsqrt(
            precise.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normalized.type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.rmsnorm1 = RMSNorm(config.n_embd)
        self.rmsnorm2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.rmsnorm1(x))
        return x + self.ffwd(self.rmsnorm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        pe = torch.zeros(cfg.block_size, cfg.n_embd)
        position = torch.arange(cfg.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, cfg.n_embd, 2).float() * (-math.log(10000.0) / cfg.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.rmsnorm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError(
                "input must have shape [batch, time] with 1 <= time <= block_size"
            )
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as the input")
        x = self.token_embedding_table(idx) + self.pe[: idx.shape[1]]
        logits = self.lm_head(self.rmsnorm_f(self.blocks(x)))
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        )
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens=32, temperature=1.0):
        if max_new_tokens < 0 or temperature <= 0 or not math.isfinite(temperature):
            raise ValueError(
                "token count must be non-negative and temperature finite and positive"
            )
        if idx.ndim != 2 or idx.shape[1] < 1:
            raise ValueError("generation needs a non-empty [batch, time] prompt")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.config.block_size :])
                probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx = torch.cat(
                    (idx, torch.multinomial(probabilities, num_samples=1)), dim=1
                )
            return idx
        finally:
            self.train(was_training)
