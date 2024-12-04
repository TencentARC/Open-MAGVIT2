import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

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
        for k, v in kwargs.items():
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
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


##Modified from https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)  # 1001
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
        labels = labels.squeeze(-1)  # [Batch]
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


class MLP_with_bias(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

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
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.dim = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.n_embd, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

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
            is_causal=True if mask is None else False,  # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)],
                             dim=-1)  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+grid_size**2, head_dim // 2, 2)
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
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.shared_aln = config.shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, config.n_embd) / config.n_embd ** 0.5)
        else:
            lin = nn.Linear(config.cond_dim, 6 * config.n_embd)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.C = config.n_embd

    def forward(
            self, x: torch.Tensor, cond_BD, freqs_cis: torch.Tensor, start_pos: int,
            mask: Optional[torch.Tensor] = None):
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(
                2)  # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        h = x + self.drop_path(
            self.attention(self.attention_norm(x).mul(scale1.add(1)).add_(shift1), freqs_cis, start_pos, mask).mul_(
                gamma1))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h).mul(scale2.add(1)).add_(shift2)).mul(gamma2))
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


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.rmsnorm = RMSNorm(C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.rmsnorm(x_BLC).mul(scale.add(1)).add_(shift)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256, cond_dim=256,
                 embd_pdrop=0., resid_dropout_p=0., attn_dropout_p=0., ffn_dropout_p=0.1, drop_path_rate=0.0,
                 n_unmasked=0, max_batch_size=32, max_seq_len=2048, class_num=1000, token_drop=0.1, cls_token_num=1,
                 rope_base=10000, norm_eps=1e-5, ffn_dim_multiplier=None,
                 initalizer_range=0.02, multiple_of=256, n_kv_head=None, shared_aln=False, alng=1e-3,
                 use_pretrained_codebook=False, codebook_ckpt_path=None, n_codebook_embd=256):
        super().__init__()

        self.config = GPTConfig(vocab_size=vocab_size, block_size=block_size, cond_dim=cond_dim,
                                embd_pdrop=embd_pdrop, resid_dropout_p=resid_dropout_p, attn_dropout_p=attn_dropout_p,
                                n_layer=n_layer, n_head=n_head, n_embd=n_embd, ffn_dropout_p=ffn_dropout_p,
                                drop_path_rate=drop_path_rate,
                                n_unmasked=n_unmasked, class_num=class_num,
                                token_drop=token_drop, cls_token_num=cls_token_num, rope_base=rope_base,
                                norm_eps=norm_eps,
                                ffn_dim_multiplier=ffn_dim_multiplier, initializer_range=initalizer_range,
                                multiple_of=multiple_of,
                                max_batch_size=max_batch_size, max_seq_len=max_seq_len, n_kv_head=n_kv_head,
                                shared_aln=shared_aln,
                                use_pretrained_codebook=use_pretrained_codebook,
                                codebook_ckpt_path=codebook_ckpt_path, n_codebook_embd=n_codebook_embd
                                )

        ## Embedding Layer
        # input embedding stem
        self.use_pretrained_codebook = use_pretrained_codebook
        if self.use_pretrained_codebook:
            self.tok_emb = nn.Embedding(self.config.vocab_size, n_codebook_embd)
            self.load_pretrained_codebook(codebook_ckpt_path)
            # self.embedding_projection = nn.Linear(n_codebook_embd, n_embd)
            self.embedding_projection = MLP_with_bias(n_codebook_embd, n_embd, n_embd)
        else:
            self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.class_emb = LabelEmbedder(self.config.class_num, self.config.n_embd)  # for class conditional

        self.token_drop = nn.Dropout(self.config.token_drop)
        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, self.config.n_layer)]

        # transformer
        self.blocks = nn.ModuleList()
        for idx in range(self.config.n_layer):
            self.blocks.append(Block(self.config, dpr[idx]))

        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.config.cond_dim,
                                                                                 6 * self.config.n_embd)) if shared_aln else nn.Identity()
        # output layer
        self.head_nm = AdaLNBeforeHead(self.config.n_embd, self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.n_embd // self.config.n_head,
                                                 self.config.rope_base, self.config.cls_token_num)

        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initalize_weights(init_adaln_gamma=alng)  ## initalize the weight


    def load_pretrained_codebook(self, ckpt_path):
        self.tok_emb.weight.data = torch.load(ckpt_path, map_location="cpu")["state_dict"]["quantize.embedding.weight"]
        self.tok_emb.weight.data = self.tok_emb.weight.data.float()
        self.tok_emb.weight.required_grad = False
        print(f"Transformer Embedding initialized from {ckpt_path}")

    def initalize_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02):
        ## initalize the weight of linear and embedding
        self.apply(self._init_weights)

        for block in self.blocks:  ## specific for AdaLN
            if hasattr(block, "ada_lin"):
                block.ada_lin[-1].weight.data[2 * self.config.n_embd:].mul_(init_adaln)
                block.ada_lin[-1].weight.data[:2 * self.config.n_embd].mul_(init_adaln_gamma)
                if hasattr(block.ada_lin[-1], "bias") and block.ada_lin[-1].bias is not None:
                    block.ada_lin[-1].bias.data.zero_()
            elif hasattr(block, "ada_gss"):
                block.ada_gss.data[:, :, 2:].mul_(init_adaln)
                block.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

        ## Adaln Head
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        ### Zero-out output layer
        nn.init.constant_(self.head.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if not self.use_pretrained_codebook:
                module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.n_embd // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blocks:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.config.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.n_embd // self.config.n_head,
                                                 self.config.rope_base, self.config.cls_token_num)

    def forward(
            self, idx, input_pos=None, mask=None, targets=None,
    ):
        idx, idx_cls = idx[0], idx[1]
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        if self.use_pretrained_codebook:
            token_embeddings = self.embedding_projection(token_embeddings)
        cls_token_embeddings = self.class_emb(idx_cls, train=self.training)[:, :self.config.cls_token_num]
        token_embeddings = torch.concat([cls_token_embeddings, token_embeddings], dim=1)
        h = self.token_drop(token_embeddings)

        cond_BD = self.shared_ada_lin(cls_token_embeddings)
        # for training and evaluation
        freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]  # seq_len
        freqs_cis = freqs_cis.to(h.device)
        for block in self.blocks:
            h = block(h, cond_BD, freqs_cis, input_pos, mask)
        h = self.head_nm(h, cond_BD)
        logits = self.head(h)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def decode_tokens(self, idx, input_pos=None, targets=None, first_step=False, ):
        """
        Inference Only
        """
        assert not self.training
        if first_step:
            token_embeddings = cond_BD = self.class_emb(idx, train=self.training)
        else:  #
            idx, cls_idx = idx[0], idx[1]
            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
            if self.use_pretrained_codebook:
                token_embeddings = self.embedding_projection(token_embeddings)
            cond_BD = self.class_emb(cls_idx, train=self.training)

        ### KV cache
        bs = token_embeddings.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = self.token_drop(token_embeddings)

        freq_cis = self.freqs_cis[input_pos]  # (cls_token_num+grid_size**2, head_dim // 2, 2) shape
        freq_cis = freq_cis.to(h.device)
        for block in self.blocks:
            h = block(h, cond_BD, freq_cis, input_pos, mask)

        h = self.head_nm(h, cond_BD)
        logits = self.head(h)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


@torch.no_grad()
def sample_IBQ(x, model, steps, temperature=1., sample_logits=True,
           top_k=None, top_p=None, callback=None, cfg_scale=1.0, token_factorization=False):
    # x is conditioning
    bs, _ = x.shape
    device = x.device
    assert x is not None
    if cfg_scale > 1.0:
        cond_token, uncond_token = torch.split(x, bs // 2, dim=0)
        sample = cond_token
    else:
        sample = x

    cond_len = x.shape[1]
    if cfg_scale > 1.0:
        max_batch_size = x.shape[0] // 2
    else:
        max_batch_size = x.shape[0]

    max_seq_length = cond_len + steps
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length,
                           dtype=model.class_emb.embedding_table.weight.dtype)

    for n in range(steps):
        if n == 0:  # prefill operation
            input_pos = torch.arange(0, cond_len, device=device)  # C
        elif n == 1:
            input_pos = torch.tensor([cond_len], device=device)
        else:
            input_pos = input_pos + 1

        logits, _ = model.decode_tokens(x, input_pos=input_pos, first_step=(n == 0))

        if cfg_scale > 1.0:
            cond_logits, uncond_logits = torch.split(logits, bs // 2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            if top_k > 0 or top_p < 1.0:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)

            if not sample_logits:
                _, x = torch.topk(probs, k=1, dim=-1)
            else:
                x = torch.multinomial(probs, num_samples=1)
            # append to the sequence and continue
            sample = torch.cat((sample, x), dim=1)
            if cfg_scale > 1.0:
                x = (torch.concat([x, x]),  torch.concat([cond_token, uncond_token]))
            else:
                x = (x, cond_token)
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample











