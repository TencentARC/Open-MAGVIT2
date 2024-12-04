"""
Refer to 
https://github.com/FoundationVision/LlamaGen
https://github.com/kakaobrain/rq-vae-transformer
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
import math

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

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

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

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
        # labels = torch.where(drop_ids, torch.tensor(self.num_classes, dtype=labels.dtype, device=labels.device), labels)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        labels = labels.squeeze(-1) # [Batch]
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    if not x.is_contiguous():
        x = x.contiguous()
    xshaped = x.float().view(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    freqs_cis = freqs_cis.to(xshaped.dtype)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    # return x_out2.type_as(x)
    return x_out2

class Block(nn.Module):
    def __init__(self, config, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, spatial_n_layer=12, n_head=8, dim=256, cond_dim=256, factorized_n_layer=2,
                 embd_pdrop=0., resid_dropout_p=0., attn_dropout_p=0., ffn_dropout_p=0.1, drop_path_rate=0.0, 
                 n_unmasked=0, token_factorization=False, max_batch_size=32, max_seq_len=2048,
                 class_num=1000, token_drop=0.1, cls_token_num=1, rope_base=10000,
                 norm_eps=1e-5, ffn_dim_multiplier=None, initalizer_range=0.02, multiple_of=256, n_kv_head=None, 
                 factorized_k=2, factorized_bits=[9, 9]):
        super().__init__()

        self.config = GPTConfig(vocab_size=vocab_size, block_size=block_size, cond_dim = cond_dim, 
                    embd_pdrop=embd_pdrop, resid_dropout_p=resid_dropout_p, attn_dropout_p=attn_dropout_p, 
                    spatial_n_layer=spatial_n_layer, factorized_n_layer=factorized_n_layer, n_head=n_head, dim=dim, 
                    ffn_dropout_p=ffn_dropout_p, drop_path_rate=drop_path_rate, n_unmasked=n_unmasked, token_factorization=token_factorization, 
                    class_num=class_num, token_drop=token_drop, cls_token_num=cls_token_num, rope_base=rope_base, norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, initializer_range=initalizer_range, multiple_of=multiple_of,
                    max_batch_size=max_batch_size, max_seq_len=max_seq_len, n_kv_head=n_kv_head, factorized_k=factorized_k,
                    factorized_bits=factorized_bits)

        ## Embedding Layer
        # input embedding stem
        if token_factorization:
            self.pre_emb = nn.Embedding(2 ** factorized_bits[0], self.config.dim)  #2**k codebook size
            self.post_emb = nn.Embedding(2 ** factorized_bits[1], self.config.dim) #2**(h-k) codebook size
            self.class_emb = LabelEmbedder(self.config.class_num, self.config.dim)
        else:
            self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.dim) # not doing any factorization
            self.class_emb = LabelEmbedder(self.config.class_num, self.config.dim) #for class conditional
        
        self.token_drop = nn.Dropout(self.config.token_drop)
        spatial_dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, self.config.spatial_n_layer)]

        factorized_dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, self.config.factorized_n_layer)]
        
        # transformer
        self.spatial_blocks = nn.ModuleList()
        for idx in range(self.config.spatial_n_layer):
            self.spatial_blocks.append(Block(self.config, spatial_dpr[idx]))

        self.factorized_blocks = nn.ModuleList()
        for idx in range(self.config.factorized_n_layer):
            self.factorized_blocks.append(Block(self.config, factorized_dpr[idx]))

        # output layer
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.token_factorization = token_factorization

        assert token_factorization is True
        self.head = nn.ModuleList([nn.Linear(self.config.dim, 2 ** self.config.factorized_bits[_], bias=False) for _ in range(factorized_k)])

        # 2d rotary pos embedding
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.config.cls_token_num)

        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initalize_weights() ## initalize the weight

    def initalize_weights(self):
        ## initalize the weight of linear and embedding
        self.apply(self._init_weights)
        
        ### Zero-out output layer
        if self.token_factorization:
            for i in range(self.config.factorized_k):
                nn.init.constant_(self.head[i].weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.spatial_blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.config.cls_token_num)
    
    def setup_factorized_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        self.max_batch_size = max_batch_size
        max_seq_length = find_multiple(max_seq_length, 8)
        for b in self.factorized_blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

    def forward(
        self, idx, input_pos=None, mask=None, targets=None, 
    ):
        if self.token_factorization:
            idx_pre, idx_post, idx_cls = idx[0], idx[1], idx[2] #idx
            token_embeddings_pre = self.pre_emb(idx_pre)
            token_embeddings_post = self.post_emb(idx_post)
            cls_token_embeddings = self.class_emb(idx_cls, train=self.training)[:, :self.config.cls_token_num]
            token_embeddings = token_embeddings_pre + token_embeddings_post ## summation
            token_embeddings = torch.concat([cls_token_embeddings, token_embeddings[:, :-1, :]], dim=1)
            h = self.token_drop(token_embeddings)
        else:
            idx, idx_cls = idx[0], idx[1]
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            cls_token_embeddings = self.class_emb(idx_cls, train=self.training)[:, :self.config.cls_token_num]
            token_embeddings = torch.concat([cls_token_embeddings, token_embeddings], dim=1)
            h = self.token_drop(token_embeddings)
        
        B, N, D = h.shape

        # for training and evaluation
        freqs_cis = self.freqs_cis[:token_embeddings.shape[1]] # seq_len
        freqs_cis = freqs_cis.to(h.device)
        for block in self.spatial_blocks:
            h = block(h, freqs_cis, input_pos, mask) #[B N C] 

        ### Intra Transformer
        assert self.token_factorization ## should use token factorization
        token_embeddings_pre = self.pre_emb(idx_pre)
        token_embeddings_post = self.post_emb(idx_post)
        factorized_ctx = torch.stack([h, token_embeddings_pre], dim=-2)
        if not factorized_ctx.is_contiguous():
            factorized_ctx = factorized_ctx.contiguous()
        factorized_ctx = factorized_ctx.view(B*N, -1, D)
        factorized_ctx_freqs_cis = freqs_cis[:factorized_ctx.shape[1]]
        for block in self.factorized_blocks:
            factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis, mask)

        if not factorized_ctx.is_contiguous():
            factorized_ctx = factorized_ctx.contiguous()
        h = factorized_ctx.view(B, N, -1, D)

        h = self.norm(h)
        logits = [self.head[i](h[:, :, i, :]) for i in range(self.config.factorized_k)]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate_context(self, idx, input_pos=None, targets=None, first_step=False):
        """
        Generate context token for Inference
        """
        assert not self.training
        if first_step:
            token_embeddings = self.class_emb(idx, train=self.training)
        else: ## the next token input
            idx_pre, idx_post = idx[0], idx[1]
            token_embedding_pre = self.pre_emb(idx_pre)
            token_embedding_post = self.post_emb(idx_post)
            token_embeddings = token_embedding_pre + token_embedding_post ##token summation for factorization

        bs, N, D = token_embeddings.shape
        
        mask = self.causal_mask[:bs, None, input_pos]
        h = self.token_drop(token_embeddings)
        freq_cis = self.freqs_cis[input_pos] #(cls_token_num+grid_size**2, head_dim // 2, 2) shape
        freq_cis = freq_cis.to(h.device)
        for block in self.spatial_blocks:
            h = block(h, freq_cis, input_pos, mask)
        
        return h
    
    def decode_subtoken(self, h, x, input_pos=None, first_step=False):
        """
        Auto-Regressive generate subtoken
        """
        B, N, D = h.shape
        if not h.is_contiguous():
            h = h.contiguous()
        if first_step: ## only context
            factorized_ctx = h.reshape(B*N, -1, D)
        else: ## subtoken
            idx = x[0]
            token_embedding = self.pre_emb(idx)
            factorized_ctx = token_embedding.reshape(B*N, -1, D)
        
        mask = self.causal_mask[:B, None, input_pos]
        factorized_ctx_freqs_cis = self.freqs_cis[input_pos]
        factorized_ctx_freqs_cis = factorized_ctx_freqs_cis.to(h.device)

        for block in self.factorized_blocks:
            factorized_ctx = block(factorized_ctx, factorized_ctx_freqs_cis, start_pos=input_pos, mask=mask)
        
        h = factorized_ctx.reshape(B, N, -1, D)
        h = self.norm(h)
        
        logits = self.head[0](h[:, :, 0, :]) if first_step else self.head[1](h[:, :, 0, :])

        return logits 


@torch.no_grad()
def sample_Open_MAGVIT2(x, model, steps, temperature=1.0, sample_logits=True, 
           top_k=None, top_p=None, callback=None, token_factorization=True, cfg_scale=1.0):
    assert token_factorization is True ### using factorization should be true
    k = 2
    bs, _ = x.shape
    device = x.device
    if cfg_scale[0] > 1.0:
        cond_token, uncond_token = torch.split(x, bs // 2, dim=0)
        sample_pre, sample_post = cond_token, cond_token
    else:
        cond_token = x
        sample_pre, sample_post = cond_token, cond_token
    
    cond_len = x.shape[1]
    if cfg_scale[0] > 1.0:
        max_batch_size = x.shape[0] // 2
    else:
        max_batch_size = x.shape[0]

    max_seq_length = cond_len + steps
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale[0] >= 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.class_emb.embedding_table.weight.dtype)
    
    for n in range(steps): ## start to sample
        if n == 0: #prefill operation
            input_pos = torch.arange(0, cond_len, device=device) #C
        elif n == 1:
            input_pos = torch.tensor([cond_len], device=device)
        else:
            input_pos = input_pos + 1
        
        h = model.generate_context(x, input_pos=input_pos, first_step=(n==0))

        x = []
        with torch.device(device): ##setup factorization 
            max_batch_size_cfg = max_batch_size * 2 if cfg_scale[0] >= 1.0 else max_batch_size
            model.setup_factorized_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.class_emb.embedding_table.weight.dtype)
        for i in range(k):
            if i == 0:## only CLS Token
                if cfg_scale[i] > 1.0:
                    factor_x = torch.concat([cond_token, uncond_token])
                else:
                    factor_x = cond_token
            factor_input_pos = torch.tensor([i], device=device)
            logits = model.decode_subtoken(h, factor_x, factor_input_pos, first_step=(i==0))

            if cfg_scale[i] > 1.0: #0611
                cond_logits, uncond_logits = torch.split(logits, bs // 2, dim=0)
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale[i]

            factor_x = sample_from_logits(logits, temperature[i], top_k[i], top_p[i])

            if i == 0:
                sample_pre = torch.cat((sample_pre, factor_x), dim=1)
            else:
                sample_post = torch.cat((sample_post, factor_x), dim=1)
            
            if cfg_scale[i] > 1.0:
                cfg_x = torch.concat([factor_x, factor_x])
                factor_x = [cfg_x, torch.concat([cond_token, uncond_token])]
                x.append(cfg_x)
            else:
                non_cfg_x = factor_x
                factor_x = (non_cfg_x, cond_token)
                x.append(non_cfg_x)
        if cfg_scale[0] > 1.0:
            x.append(torch.concat([cond_token, uncond_token]))
        else:
            x.append(cond_token)
    
    sample_pre = sample_pre[:, cond_len:]
    sample_post = sample_post[:, cond_len:]
    sample = (sample_pre, sample_post)

    return sample  

def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits[:, -1, :] / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x