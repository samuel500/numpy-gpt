import math

import numpy as np

from nptransformer.tensor import Tensor, Linear, SequentialModel, Model, Embedding, Sigmoid, Tanh, ReLU, NewGELU, Softmax


class MultiHeadSelfAttention(Model):

    def __init__(self, n_embd, n_heads = 8):
        super().__init__()
        assert n_embd % n_heads == 0

        self.n_embd = n_embd
        self.n_heads = n_heads

        self.attn = Linear(n_embd, 3*n_embd)

        self.out_proj = Linear(n_embd, n_embd)

        # self.attn_dropout = Dropout(attn_pdrop)
        # self.resid_dropout = Dropout(resid_pdrop)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        k, q, v = self.attn(x).split(3, dim=2)

        k = k.reshape(B, T, self.n_embd, C / self.n_heads).transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??
        q = q.reshape(B, T, self.n_embd, C / self.n_heads).transpose(0,2,1,3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_embd, C / self.n_heads).transpose(0,2,1,3)  # (B, nh, T, hs)

        att = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.n_embd)

        # apply attention mask

        att = Softmax(att)

        # att dropout

        out = att @ v

        out = self.out_proj(out)

        # resid dropout

        return out


class Block(Model):

    def __init__(self, n_embd, n_heads = 8):
        super().__init__()
        # LayerNorm 1 & 2

        self.attn = MultiHeadSelfAttention(n_embd=n_embd, n_heads=n_heads)

        self.mlp = SequentialModel(
            layers = [
                Linear(n_embd, 4 * n_embd, nonlin=NewGELU),
                Linear(4 * n_embd, n_embd),
            ]
        )

    def forward(self, x: 'Tensor'):
        x = x + self.attn(x)  # Missing LayerNorm
        x = x + self.mlp(x)   # Missing LayerNorm
        return x


class GPT(Model):


    def __init__(self, n_layers, n_heads, n_embd, vocab_size, block_size):
        super().__init__()

        # embeddings init
        self.word_embedding = Embedding(vocab_size, n_embd)
        self.positional_embedding = Embedding(block_size, n_embd)

        # dropout 

        self.blocks = [Block(n_embd=n_embd, n_heads=n_heads) for _ in n_layers]

        # layernorm

        self.lm_head = Linear(n_embd, vocab_size)

    def forward(self, idx: np.array):
        
        assert idx.dtype is np.int8  # embeddings

        b, t = idx.shape  # batch, sequence length

        # embeddings

        tok_emb = self.word_embedding(idx) # batch, seq, embedding size

        pos = np.expand_dims(np.arange(t), axis=0)
        pos_emb = self.positional_embedding(pos)

        x = tok_emb + pos_emb

        # dropout

        for block in self.blocks:
            x = block(x)

        # layer norm

        x = self.lm_head(x)

        return x

    def generate(self, x, max_new_tokens):
        pass








