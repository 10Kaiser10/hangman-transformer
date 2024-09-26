import torch
import torch.nn as nn
from torch.nn import functional as F
import math
torch.manual_seed(42)

class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B,T,C = x.shape   #B,T,C

        k = self.key(x)   #B,T,H   (H is head size)
        q = self.query(x)   #B,T,H   (H is head size)

        w = q @ k.transpose(-2,-1) #B,T,H @ B,H,T -> B,T,T
        w = self.softmax(w * (C**-0.5))
        w = self.dropout(w)

        v = self.value(x)  #B,T,H
        out = w @ v   #B,T,T @ B,T,H -> B,T,H
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, embed_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embed_size, head_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),  #B,T,4C
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),  #B,T,C
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed_forward(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, n_head, dropout):
        super().__init__()
        head_size = embed_size // n_head
        self.mh_attention = MultiHeadAttention(n_head, head_size, embed_size, dropout)
        #self.mh_attention = nn.MultiheadAttention(embed_size, n_head, dropout, batch_first=True)
        self.feed_fwd = FeedForward(embed_size, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        #normed = self.layer_norm1(x)
        #x = x + self.mh_attention(normed,normed,normed, need_weights=False)[0]  # B,T,C
        x = x + self.mh_attention(self.layer_norm1(x)) #B,T,C
        x = x + self.feed_fwd(self.layer_norm2(x))  # B,T,C
        return x

class HangmanTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, n_head, n_layer, output_size, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_size, n_head, dropout) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(embed_size)
        self.final_linear_layer = nn.Linear(embed_size, output_size+1)
        self.output_func = nn.Softmax(dim=-1)

        self.device = device
        self.output_size = output_size

    def forward(self, x):
        B, T = x.shape         #batch, time
        
        mask = x.ge(27)
        mask = ~mask.unsqueeze(-1).repeat(1, 1, self.output_size+1)

        tok_embd = self.token_embedding_table(x)  #B,T,C
        pos_embd = self.position_embedding_table(torch.arange(T, device=self.device))  #T,C
        x = tok_embd + pos_embd  #B,T,C
        x = self.transformer_blocks(x)  #B,T,C
        x = self.final_layer_norm(x)   #B,T,C
        logits = self.final_linear_layer(x)  #logits for each position, B,T,output_size
        probs = self.output_func(logits)  #probabilities for each position,  B,T,output_size

        probs = probs.masked_fill(mask, 0)

        final_probs = 1 - torch.prod(1-probs, dim=1)  #logits for each token, B,output_size

        return final_probs[:, :-1]
    

class HangmanTransformerDirect(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, n_head, n_layer, output_size, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.sinosoidal_positional_embedding = self.positionalencoding1d(embed_size, block_size).to(device)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_size, n_head, dropout) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(embed_size)
        self.final_linear_layer = nn.Linear(embed_size*block_size, output_size)
        self.output_func = nn.Softmax(dim=-1)

        self.device = device

    def positionalencoding1d(self, embed_size, block_size):
        if embed_size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(embed_size))
        
        pe = torch.zeros(block_size, embed_size, requires_grad=False)
        position = torch.arange(0, block_size).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_size, 2, dtype=torch.float) * -(math.log(10000.0) / embed_size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def forward(self, x):
        B, T = x.shape         #batch, time

        tok_embd = self.token_embedding_table(x)  #B,T,C
        #pos_embd = self.position_embedding_table(torch.arange(T, device=self.device))  #T,C
        pos_embd = self.sinosoidal_positional_embedding  #T,C
        x = tok_embd + pos_embd  #B,T,C
        x = self.transformer_blocks(x)  #B,T,C
        x = self.final_layer_norm(x)   #B,T,C
        x = torch.flatten(x, start_dim=1) #B,T*C
        logits = self.final_linear_layer(x)  #logits for each position, B,output_size

        return logits