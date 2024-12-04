"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

# fix the top_k_top_p_filtering import error bug, refer to https://github.com/huggingface/trl/issues/1409
# from transformers import top_k_top_p_filtering
from transformers.generation.utils import top_k_top_p_filtering

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present   # TODO: check that this does not break anything


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


##Modified from https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size) # 1001
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        labels = labels.squeeze(-1) # [Batch]
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, token_factorization=False,
                 weight_tying=True, class_num=1000, token_drop=0.1, cls_token_number=1):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, token_factorization=token_factorization, class_num=class_num,
                           token_drop=token_drop, cls_token_number=cls_token_number)
        # input embedding stem
        if token_factorization:
            self.pre_emb = nn.Embedding(config.vocab_size, config.n_embd) # 2**9 codebook size
            self.post_emb = nn.Embedding(config.vocab_size, config.n_embd) #2**9 codebook size
            self.class_emb = LabelEmbedder(config.class_num, config.n_embd)
        else:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd) # not doing any factorization
            self.class_emb = LabelEmbedder(config.class_num, config.n_embd) #for class conditional

        self.token_drop = nn.Dropout(config.token_drop)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        ### whether using token Factorization
        self.token_factorization = token_factorization
        self.weight_tying = weight_tying
        self.cls_token_number = cls_token_number
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if token_factorization:
            if not weight_tying: #weight typing using the shared input embedidng and 
                self.head_pre = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
                self.head_post = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            if not weight_tying:
                self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        if self.token_factorization:
            idx_pre, idx_post, idx_cls = idx[0], idx[1], idx[2] #idx
            token_embeddings_pre = self.pre_emb(idx_pre)
            token_embeddings_post = self.post_emb(idx_post)
            cls_token_embeddings = self.class_emb(idx_cls, train=self.training)[:, :self.cls_token_number]
            token_embeddings = token_embeddings_pre + token_embeddings_post ## summation
            token_embeddings = torch.concat([cls_token_embeddings, token_embeddings], dim=1)
            token_embeddings = self.token_drop(token_embeddings)
        else: #not using token factorization
            idx, idx_cls = idx[0], idx[1]
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            cls_token_embeddings = self.class_emb(idx_cls, train=self.training)[:, :self.cls_token_number]
            token_embeddings = torch.concat([cls_token_embeddings, token_embeddings], dim=1)
            token_embeddings = self.token_drop(token_embeddings)
        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        if self.token_factorization:
            if self.weight_tying:
                logits_pre = x @ self.pre_emb.weight.T
                logits_post = x @ self.post_emb.weight.T
            else:
                logits_pre = self.head_pre(x)
                logits_post = self.head_post(x)
            logits = (logits_pre, logits_post)
        else:
            if self.weight_tying:
                logits = x @ self.tok_emb.weight.T
            else:
                logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None, first_step=False):
        # inference only
        assert not self.training
        if self.token_factorization:
            if first_step:
                token_embeddings = self.class_emb(idx, train=self.training)
            else: ## the next token input
                idx_pre, idx_post = idx[0], idx[1]
                token_embedding_pre = self.pre_emb(idx_pre)
                token_embedding_post = self.post_emb(idx_post)
                token_embeddings = token_embedding_pre + token_embedding_post ##token summation for factorization
        else:
            if first_step:
                token_embeddings = self.class_emb(idx, train=self.training)
            else: #
                token_embeddings = self.tok_emb(idx)    # each index maps to a (learnable) vector
        
        if embeddings is not None:              # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)   # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            if self.token_factorization:
                expected_shape = [self.config.n_layer, 2, idx[0].shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            else:
                expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        if self.token_factorization:
            if self.weight_tying:
                logits_pre = x @ self.pre_emb.weight.T
                logits_post = x @ self.post_emb.weight.T
            else:
                logits_pre = self.head_pre(x)
                logits_post = self.head_post(x)
            logits = (logits_pre, logits_post)
        else:
            if self.weight_tying:
                logits = x @ self.tok_emb.weight.T
            else:
                logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)  # _, _, n_layer, 2, b, nh, 1, dim_head


class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


class CodeGPT(nn.Module):
    """Takes in semi-embeddings"""
    def __init__(self, vocab_size, block_size, in_channels, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        self.tok_emb = nn.Linear(in_channels, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector

        if embeddings is not None: # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.taming_cinln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss



#### sampling utils

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None, token_factorization=False, cfg_scale=1.0):
    # x is conditioning 
    bs, _ = x.shape
    if token_factorization:
        if cfg_scale > 1.0: ## two batchsize
            cond_token, uncond_token = torch.split(x, bs // 2, dim=0)
            sample_pre, sample_post = cond_token, cond_token
        else:
            sample_pre, sample_post = x, x
    else:
        assert x is not None
        if cfg_scale > 1.0:
            cond_token, uncond_token = torch.split(x, bs // 2, dim=0)
            sample = cond_token
        else:
            sample = x
    
    cond_len = x.shape[1]
    past = None #past is shared, it is not in token factorization
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1), first_step=(n==0))
        if past is None:
            past = [present]
        else:
            past.append(present)
        if token_factorization:
            logits_pre, logits_post = logits[0], logits[1]
            if cfg_scale > 1.0: #0611
                cond_logits_pre, uncond_logits_pre = torch.split(logits_pre, bs // 2, dim=0)
                logits_pre = uncond_logits_pre + (cond_logits_pre - uncond_logits_pre) * cfg_scale
                cond_logits_post, uncond_logits_post = torch.split(logits_post, bs // 2, dim=0)
                logits_post = uncond_logits_post + (cond_logits_post - uncond_logits_post) * cfg_scale
            ### start to sample
            logits_pre = logits_pre[:, -1, :] / temperature
            logits_post = logits_post[:, -1, :] / temperature
            logits = (logits_pre, logits_post) # always keeping the format of logits
        else:
            if cfg_scale > 1.0:
                cond_logits, uncond_logits = torch.split(logits, bs // 2, dim=0)
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            if token_factorization:
                logits_pre, logits_post = logits[0], logits[1]
                logits_pre = top_k_top_p_filtering(logits_pre, top_k=top_k, top_p=top_p)
                logits_post = top_k_top_p_filtering(logits_post, top_k=top_k, top_p=top_p)
                probs_pre = F.softmax(logits_pre, dim=-1)
                probs_post = F.softmax(logits_post, dim=-1)

                if not sample_logits:
                    _, x_pre = top_k(probs_pre, k=1, dim=-1)
                    _, x_post = top_k(probs_post, k=1, dim=-1)
                else:
                    x_pre = torch.multinomial(probs_pre, num_samples=1)
                    x_post = torch.multinomial(probs_post, num_samples=1)
                
                if cfg_scale >  1.0:
                    x = (torch.concat([x_pre, x_pre]), torch.concat([x_post, x_post]))
                else:
                    x = (x_pre, x_post) #for next embedding
                ## for decoding
                sample_pre = torch.cat((sample_pre, x_pre), dim=1)
                sample_post = torch.cat((sample_post, x_post), dim=1)
            else:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

                probs = F.softmax(logits, dim=-1)

                if not sample_logits:
                    _, x = torch.topk(probs, k=1, dim=-1)
                else:
                    x = torch.multinomial(probs, num_samples=1)
                # append to the sequence and continue
                if cfg_scale > 1.0:
                    x = torch.concat([x, x])
                else:
                    x = x
                sample = torch.cat((sample, x), dim=1)
    
    if token_factorization:
        sample_pre = sample_pre[:, cond_len:]
        sample_post = sample_post[:, cond_len:]
        sample = (sample_pre, sample_post)
    else:
        sample = sample[:, cond_len:]  # cut conditioning off
    del past
    return sample


#### clustering utils

class KMeans(nn.Module):
    def __init__(self, ncluster=512, nc=3, niter=10):
        super().__init__()
        self.ncluster = ncluster
        self.nc = nc
        self.niter = niter
        self.shape = (3,32,32)
        self.register_buffer("C", torch.zeros(self.ncluster,nc))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def is_initialized(self):
        return self.initialized.item() == 1

    @torch.no_grad()
    def initialize(self, x):
        N, D = x.shape
        assert D == self.nc, D
        c = x[torch.randperm(N)[:self.ncluster]] # init clusters at random
        for i in range(self.niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a==k].mean(0) for k in range(self.ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, self.niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

        self.C.copy_(c)
        self.initialized.fill_(1)


    def forward(self, x, reverse=False, shape=None):
        if not reverse:
            # flatten
            bs,c,h,w = x.shape
            assert c == self.nc
            x = x.reshape(bs,c,h*w,1)
            C = self.C.permute(1,0)
            C = C.reshape(1,c,1,self.ncluster)
            a = ((x-C)**2).sum(1).argmin(-1) # bs, h*w indices
            return a
        else:
            # flatten
            bs, HW = x.shape
            """
            c = self.C.reshape( 1, self.nc,  1, self.ncluster)
            c = c[bs*[0],:,:,:]
            c = c[:,:,HW*[0],:]
            x =      x.reshape(bs,       1, HW,             1)
            x = x[:,3*[0],:,:]
            x = torch.gather(c, dim=3, index=x)
            """
            x = self.C[x]
            x = x.permute(0,2,1)
            shape = shape if shape is not None else self.shape
            x = x.reshape(bs, *shape)

            return x