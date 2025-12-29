"""
Full implementation of a decoder only Generative Pretrained Transformer (GPT). 

Following papers are important references. 
- Attention is All You Need (Vaswani et al)
- Improving Language Understanding by Generative Pre-Training (Radford et al)
- Layer Normalization (Ba et al)
- Deep Residual Learning for Image Recognition (He et al)
- Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al)

Also, you need to run this on a GPU. The current parameters should take around 45 minutes on an A100. 
"""
import torch
import torch.nn as nn
from torch.nn import LayerNorm, functional as F

# hyperparameters
batch_size = 64 # number of seequences processed in parallel
context_size = 256 # maximum context length for prediction
max_iters = 15000
# when estimating the loss
eval_interval = 1000
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2
n_layer = 6
n_head = 6
n_embd = 384

torch.manual_seed(0)

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# encode the text
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# get a batch
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

        self.dropout = nn.Dropout(dropout) # add dropout to prevent overfitting
    def forward(self, x):
        # input of size (batch, time, channels)
        # output of size (batch, time-step, head-size)
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # normalize by sqrt(head_size)

        # compute attention scores, affinities between keys and queries
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, hs) @ (B, hs, T) --> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        v = self.value(x) # (B, T, hs)
        out = weights @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple Heads of self-attention in parallel, followed by concatenating the results"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over the channel dimension
        # need to project for residual connections
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearty"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # growing layer in ffwd, following paper. 
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd), # projection for residual
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: communication followed by computation. """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads for multi-head attention
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # add layer norm to again help with optimization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # add residual connections
        # implement pre norm
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # add a positional embedding to each token 
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # apply several blocks of multi-head attention
        x = self.ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, new_token_length):
        for _ in range(new_token_length):
            # crop the length of the context to be within context_size
            idx_cond = idx[:, -context_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every eval_interval, estimate the loss on the validation set
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, new_token_length=500)[0].tolist()))