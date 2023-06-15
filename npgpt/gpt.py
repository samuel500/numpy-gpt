# Resources:
# https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# https://huggingface.co/transformers/v4.8.2/_modules/transformers/models/gpt_neo/modeling_gpt_neo.html

import math

import numpy as np

from npgpt.tensor import Tensor, Linear, SequentialModel, Model, Embedding, Dropout, Sigmoid, Tanh, ReLU, NewGELU, Softmax, cross_entropy


class MultiHeadSelfAttention(Model):

    def __init__(self, n_embd, block_size, n_heads=8, name='', attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_heads == 0

        self.n_embd = n_embd
        self.n_heads = n_heads

        self.block_size = block_size

        self.attn = Linear(n_embd, 3*n_embd, use_bias=False, name=f"{name}-attn")

        self.bias = (np.tril(np.ones(shape=(1, 1, self.block_size, self.block_size)))) #.astype(np.int8)
        # self.bias[self.bias==1.] = -np.inf

        self.out_proj = Linear(n_embd, n_embd, use_bias=True, name=f"{name}-out_proj")

        self.attn_dropout = Dropout(attn_pdrop)
        self.proj_dropout = Dropout(resid_pdrop)

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        attn_out = self.attn(x)
        k0, q0, v0 = attn_out.split(3, axis=2)

        k = (1*k0.reshape(B, T, self.n_heads, C // self.n_heads)).transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??
        # k = (1*k0).reshape(B, self.n_heads, T, C // self.n_heads) #.transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??
       
        q = (1*q0.reshape(B, T, self.n_heads, C // self.n_heads)).transpose(0,2,1,3)  # (B, nh, T, hs)
        # q = (1*q0).reshape(B, self.n_heads, T, C // self.n_heads) #.transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??

        v = (1*v0.reshape(B, T, self.n_heads, C // self.n_heads)).transpose(0,2,1,3)  # (B, nh, T, hs)
        # v = (1*v0).reshape(B, self.n_heads, T, C // self.n_heads) #.transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??


        att = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.n_embd) #, nograd=True)

        # apply forward attention mask
        att.data *= self.bias[:,:,:T,:T]  # += self.bias[:,:,:T,:T]
        att.data[att.data==0.] = -np.inf
        soft_att = Softmax(att)

        # att dropout
        soft_att = self.attn_dropout(soft_att)

        out = soft_att @ v  # batch, n_heads, seq_len, n_embd // n_heads

        trans_out = 1*out.transpose(0,2,1,3)
        
        reshape_out = trans_out.reshape(B,T,C)  # B, T, C  # reshape directly to correct shape??

        # reshape_out = out.reshape(B,T,C)  # B, T, C  # reshape directly to correct shape??


        proj_out = self.out_proj(reshape_out)  

        proj_out = self.proj_dropout(proj_out)

        return proj_out


class Block(Model):

    def __init__(self, n_embd, block_size, n_heads = 8, name='', mlp_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        # LayerNorm 1 & 2

        self.attn = MultiHeadSelfAttention(
            n_embd=n_embd, 
            block_size=block_size, 
            n_heads=n_heads, 
            name=f"{name}-MHSA", 
            attn_pdrop=attn_pdrop, 
            resid_pdrop=resid_pdrop
        )

        self.mlp = SequentialModel(
            layers = [
                Linear(n_embd, 4 * n_embd, nonlin=NewGELU, name=f"{name}-mlp0"),
                Linear(4 * n_embd, n_embd, name=f"{name}-mlp1"),
                Dropout(mlp_pdrop)
            ]
        )

    def forward(self, x: 'Tensor'):
        x = x + self.attn(x)  # Missing LayerNorm
        x = x + self.mlp(x)   # Missing LayerNorm
        return x


class GPT(Model):


    def __init__(self, n_layers, n_heads, n_embd, vocab_size, block_size, pdrop=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        # embeddings init
        self.word_embedding = Embedding(vocab_size, n_embd)
        self.positional_embedding = Embedding(block_size, n_embd)

        self.drop = Dropout(pdrop)

        self.blocks = [Block(n_embd=n_embd, block_size=block_size, n_heads=n_heads, name=f"Block[{i}]", mlp_pdrop=pdrop, attn_pdrop=pdrop, resid_pdrop=pdrop) for i in range(n_layers)]

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
        pos = np.atleast_2d(np.arange(t)).repeat(idx.shape[0], axis=0)
        pos_emb = self.positional_embedding(pos)

        x = tok_emb + pos_emb

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        # layer norm

        logits = self.lm_head(x)

        loss = None
        if y is not None:
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            mask = np.array(y)
            mask[mask!=-1] = 1
            mask[mask==-1] = 0
            y = Tensor(np.array(np.eye(self.vocab_size)[y], dtype=np.float64), nograd=False) 
            # log_softm = logits.log_softmax() 
            # loss = -log_softm*y
            # loss /= (loss.shape[0]/2)  #12288
            loss = cross_entropy(logits=logits, target=y)
            loss.data = ((loss.data.T*mask).T)*(mask.size/(mask!=0).sum())  # ignore -1 labels
            loss.grad_mask = mask

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx = np.array(idx, dtype=np.int8)
        for _ in range(max_new_tokens):

            logits, _ = self(idx)

            logits = logits.data[:,-1,:]
            # exp_self = np.exp(logits)
            probs = Tensor(logits).softmax().data #

            # probs = exp_self / np.sum(exp_self, axis=-1, keepdims=True)

            x_next = np.argmax(probs, axis=-1)
            x_next = np.expand_dims(x_next, axis=1)
            idx = np.concatenate((idx, x_next), axis=1) 
            
        return idx






