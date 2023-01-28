import math

import numpy as np

from nptransformer.tensor import Tensor, Linear, SequentialModel, Model, Embedding, Sigmoid, Tanh, ReLU, NewGELU, Softmax


class MultiHeadSelfAttention(Model):

    def __init__(self, n_embd, block_size, n_heads=8, name=''):
        super().__init__()
        assert n_embd % n_heads == 0

        self.n_embd = n_embd
        self.n_heads = n_heads

        self.block_size = block_size

        self.attn = Linear(n_embd, 3*n_embd, name=f"{name}-attn")

        self.bias = (1-np.tril(np.ones(shape=(1, 1, self.block_size, self.block_size)))) #.astype(np.int8)
        self.bias[self.bias==1.] = -np.inf

        self.out_proj = Linear(n_embd, n_embd, name="{name}-out_proj")

        # self.attn_dropout = Dropout(attn_pdrop)
        # self.resid_dropout = Dropout(resid_pdrop)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        k, q, v = self.attn(x).split(3, axis=2)

        k = k.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??
        q = q.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0,2,1,3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0,2,1,3)  # (B, nh, T, hs)

        att = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.n_embd)

        # apply forward attention mask
        att += self.bias[:,:,:T,:T]

        att = Softmax(att)

        # att dropout

        out = att @ v  # batch, n_heads, seq_len, n_embd // n_heads

        out = out.transpose(0,2,1,3).reshape(B,T,C)  # B, T, C  # reshape directly to correct shape??

        out = self.out_proj(out)  

        # resid dropout

        return out


class Block(Model):

    def __init__(self, n_embd, block_size, n_heads = 8, name=''):
        super().__init__()
        # LayerNorm 1 & 2

        self.attn = MultiHeadSelfAttention(n_embd=n_embd, block_size=block_size, n_heads=n_heads, name=f"{name}-MHSA")

        self.mlp = SequentialModel(
            layers = [
                Linear(n_embd, 4 * n_embd, nonlin=NewGELU, name=f"{name}-mlp0"),
                Linear(4 * n_embd, n_embd, name=f"{name}-mlp1"),
            ]
        )

    def forward(self, x: 'Tensor'):
        x = x + self.attn(x)  # Missing LayerNorm
        x = x + self.mlp(x)   # Missing LayerNorm
        return x


class GPT(Model):


    def __init__(self, n_layers, n_heads, n_embd, vocab_size, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        # embeddings init
        self.word_embedding = Embedding(vocab_size, n_embd)
        self.positional_embedding = Embedding(block_size, n_embd)

        # dropout 

        self.blocks = [Block(n_embd=n_embd, block_size=block_size, n_heads=n_heads, name=f"Block[{i}]") for i in range(n_layers)]

        # layernorm

        self.lm_head = Linear(n_embd, vocab_size, name="GPT_lm_head")

    def __call__(self, x: 'Tensor', y = None):
        return self.forward(x, y)

    def forward(self, idx: np.array, y = None):
        idx = np.array(idx, dtype=np.int8)
        if y is not None:
            y = np.array(y, dtype=np.int8)
        # logg
        # assert idx.dtype is np.int8  # embeddings

        b, t = idx.shape  # batch, sequence length

        # embeddings

        tok_emb = self.word_embedding(idx) # batch, seq, embedding size

        # pos = np.expand_dims(np.arange(t), axis=0)
        pos = np.arange(t)
        pos_emb = self.positional_embedding(pos)

        x = tok_emb + pos_emb

        # dropout

        for block in self.blocks:
            x = block(x)

        # layer norm

        logits = self.lm_head(x)

        loss = None
        if y is not None:
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            mask = np.array(y)
            mask[mask!=-3] = 1
            mask[mask==-3] = 0
            y = Tensor(np.eye(self.vocab_size)[y], nograd=True) 
            softm = logits.softmax() 
            loss = -(softm.log()*y + (1-y)*(1-softm).log()) #/ 64  #  64.0 #0.0
            # mask = Tensor(mask, nograd=True)
            # loss = (loss.T*mask).T  # ignore -1 labels
            loss.grad_mask = mask

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx = np.array(idx, dtype=np.int8)
        for _ in range(max_new_tokens):

            logits, _ = self(idx)

            logits = logits.data[:,-1,:]
            exp_self = np.exp(logits)
            probs = exp_self / np.sum(exp_self, axis=-1, keepdims=True)

            x_next = np.argmax(probs, axis=-1)
            x_next = np.expand_dims(x_next, axis=1)
            idx = np.concatenate((idx, x_next), axis=1) 
            
        return idx







