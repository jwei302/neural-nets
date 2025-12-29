from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# ------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    A causal multi-head self-attention layer.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 #  tokens per head = n_embd / n_head
        # combined key, query, value projection matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

        # the lower triangular mask that allows for gradual accumulation via weighted averages
        self.register_buffer("bias", torch.tril(torch.ones((config.context_size, config.context_size)))
                                                        .view(1, 1, config.context_size, config.context_size))
    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value; hs = head size, nh = number of heads
        qkv = self.c_attn(x) # (B, T, C) -> (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, 3*C) -> (B, T, C) for q, k, v
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, C) --> (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, C) --> (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, C) --> (B, T, nh, hs)
        # move head dimension to the second dimension to emulate the multi-head attention mechanism
        q = q.transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        scale = q.shape[-1] ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale # (B, nh, T, T)
        y = att @ v #(B, nh, T, T) x (B, nh, T, hs) --> (B, nh, T, hs)
        # transpose to and bring back head dimension to third dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) --> (B, T, C)
        # output projection 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    A feed forward network with a GELU activation function.
    """
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # can use approximate or precise
        # later models use swiglu and other activations 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    A transformer block with a causal multi-head self-attention layer and a feed forward network.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd) # note layer norms go before
        self.mlp = MLP(config) # same as feed forward network 

        # attention allows communication between tokens
        # mlp allows communication to the next layer 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_1(x)) 
        return x


# use dataclass to automatically generate a __init__, __repr__, __eq__ methods. 
@dataclass
class GPTConfig:
    """
    Configuration for a GPT-2 model.
    """
    context_size: int = 1024 # number of tokens in context
    vocab_size: int = 50257 # number of tokens in vocabulary: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of transformer block layers
    n_head: int = 12 # number of attention heads in multi-head attention
    n_embd: int = 768 # number of embedding dimension


class GPT(nn.Module):
    """
    A GPT-2 model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # vocab embedding
            wpe = nn.Embedding(config.context_size, config.n_embd), # position embedding 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # n_layer of transformer blocks 
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size) # language model head, project back to logit
    
