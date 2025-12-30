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
        self.c_proj.FLAG = 1
        self.n_head = config.n_head

        # the lower triangular mask that allows for gradual accumulation via weighted averages
        self.register_buffer("bias", torch.tril(torch.ones((config.context_size, config.context_size)))
                                                        .view(1, 1, config.context_size, config.context_size))
    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value; hs = head size, nh = number of heads
        qkv = self.c_attn(x) # (B, T, C) -> (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2) # (B, T, 3*C) -> (B, T, C) for q, k, v
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, C) --> (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, C) --> (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, C) --> (B, T, nh, hs)
        # move head dimension to the second dimension to emulate the multi-head attention mechanism
        q = q.transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, T, nh, hs) --> (B, nh, T, hs)
        scale = q.shape[-1] ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # apply causal mask
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
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
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # can use approximate or precise
        # later models use swiglu and other activations 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.FLAG = 1

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
        x = x + self.mlp(self.ln_2(x)) 
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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language model head, project back to logit

        # weight sharing between wte and lm_head to save memory and improve performance
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)

    # initialize weights for better optimization following OpenAI codebase
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'FLAG'):
                std *= (2 * self.config.n_layer) ** -0.5
            # consistent with Xavier initialization which sets initialization as 1 / sqrt(n_incoming_features)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None): 
        B, T = idx.shape #(batch size, sequence length)
        assert T <= self.config.context_size, f"Cannot forward sequence of length {T} when context size is {self.config.context_size}"
        # token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    # use class method to generate a model given the model_type from Hugging Face
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load a pretrained GPT-2 model weights from Hugging Face.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt-2: {model_type}")

        # n_layer, n_head, n_embd are determined by model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1.558B parameters
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50,257 for GPT model
        config_args['context_size'] = 1024 # always 1024 for GPT model

        config = GPTConfig(**config_args)
        model = GPT(config)

        # state_dict() returns a dictionary of the model's parameters, specific to nn.Module
        sd = model.state_dict() 
        sd_keys = sd.keys()
        # the bias is not used in the model, it is just used for the autogressive mask
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # remove the bias from the state dictionary

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # hf implementation uses tensorflow, need to transpose the weights to match the pytorch convention
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose these weights 
                # ensure the transposed shape matches the original shape
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"mismatch: {sd_hf[k].shape[::-1]} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t()) # copy over the weights and use .t() to transpose
            else: 
                # can directly transfer weights over
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

# tokenize text using tiktoken from OpenAI
import tiktoken

# create a data loader to initialize and return batch of data
class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('../data/shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            print(f'Loaded {len(text)} words')
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        print(f'Loaded {len(self.tokens)} tokens') # this should be around 1/3 of the total words.
        print(f'1 Epoch is {self.tokens.size(0) // (B * T)} batches')

        self.current_position = 0
    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position:self.current_position + B * T+1]
        x = buffer[:-1].view(B, T) # all but last token
        y = buffer[1:].view(B, T) # all but first token
        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ------------------------------------------------------------
def main():
    # choose device 
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")

    model = GPT(GPTConfig())
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_loader = DataLoader(B=4, T=8)
    for i in range(5): 
        # get batch from dataloader
        x, y = train_loader.next_batch()
        # move batch to device
        x, y = x.to(device), y.to(device)
        # set to none to avoid memory leaks by setting to zero, default = True
        optimizer.zero_grad(set_to_none=True)
        # forward pass, note we use optional targets to compute loss now
        logits, loss = model(x, y)
        # backward pass 
        loss.backward()
        optimizer.step()
        print(f'Step {i+1}: loss={loss.item():.4f}')
    # num_return_sequences = 5
    # max_length = 30

    # model = GPT.from_pretrained('gpt2')
    # # for model can do .to(device) to move the model to the device
    # # but for tensors, we need to do x = x.to(device) to move the tensor to the device
    # model.to(device)

    # enc = tiktoken.get_encoding('gpt2')
    # text = "Yale University"
    # tokens = enc.encode(text)
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device) # repeat num_return_sequences time 

    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)

    # while x.size(1) < max_length:
    #     with torch.no_grad():
    #         logits, loss = model(x)
    #         logits = logits[:, -1, :] # (B, vocab_size)
    #         # we only need last token to determine the next token
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, vocab_size)
    #         # do top-k sampling with k=50
    #         # top-k sampling: keep the top k probabilities and set the rest to 0
    #         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
    #         # sample from the top k probabilities
    #         ix = torch.multinomial(topk_probs, 1) # (B, 1)
    #         # use torch.gather to get the indices corresponding to the sampled tokens
    #         xnew = topk_indices.gather(-1, ix) # (B, 1)
    #         # concatenate new token to existing tokens
    #         x = torch.cat((x, xnew), dim=1) # (B, T+1)
    
    # for i in range(num_return_sequences):
    #     tokens = x[i].tolist()
    #     decoded = enc.decode(tokens)
    #     print('>' + decoded)

if __name__ == "__main__":
    main()