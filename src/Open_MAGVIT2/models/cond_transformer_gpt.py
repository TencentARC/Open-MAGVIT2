"""
Refer to 
https://github.com/FoundationVision/LlamaGen
https://github.com/FoundationVision/VAR
"""

import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L
import inspect

from main import instantiate_from_config
from src.Open_MAGVIT2.modules.util import SOSProvider


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Net2NetTransformer(L.LightningModule):
    def __init__(self,
                transformer_config,
                first_stage_config,
                cond_stage_config,
                permuter_config=None,
                ckpt_path=None,
                ignore_keys=[],
                first_stage_key="image",
                cond_stage_key="depth",
                downsample_cond_size=-1,
                pkeep=1.0,
                sos_token=0,
                unconditional=False,
                learning_rate=None,
                token_factorization=False,
                weight_decay=1e-2,
                resume_lr = None,
                wp = 0,
                wp0 = 0.005, #initial lr ratio at the begging of lr warm up
                wpe = 0.01, #final lr ratio at the end of training
                twde = 0,
                 ):
        super().__init__()

        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "src.Open_MAGVIT2.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.token_factorization = token_factorization
        self.resume_lr = resume_lr

        ## for scheduler
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        self.twde = twde or weight_decay

    def state_dict(self, *kwargs, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*kwargs, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        if self.token_factorization: ## must be use
            a_indices_pre, a_indices_post = a_indices[0], a_indices[1]
            cz_indices = (a_indices_pre, a_indices_post, c_indices) ## AR inside

            target_pre = a_indices_pre
            target_post = a_indices_post
            
            logits, _ = self.transformer(cz_indices) #[B N 2 D]

            logits_pre, logits_post = logits[0], logits[1]

            logits = (logits_pre, logits_post)
            target = (target_pre, target_post)

        else:
            # target includes all sequence elements (no need to handle first one
            # differently because we are conditioning)
            target = z_indices
            # make the prediction
            cz_indices = (a_indices[:, :-1], c_indices) #not token factorization
            logits, _ = self.transformer(cz_indices)
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, indices, _ = self.first_stage_model.encode(x)
        if isinstance(indices, tuple):
            indices_pre, indices_post = indices[0], indices[1]
            indices_pre = indices_pre.view(quant_z.shape[0], -1)
            indices_post = indices_post.view(quant_z.shape[0], -1)
            indices = (indices_pre, indices_post)
        else:
            indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        if self.token_factorization: #using token factorization
            index_pre, index_post = index[0], index[1]
            # index_pre = self.permuter(index_pre, reverse=True)
            # index_post = self.permuter(index_post, reverse=True) #
            bhwc_pre = (zshape[0], zshape[2], zshape[3], self.transformer.config.factorized_bits[0])
            bhwc_post = (zshape[0], zshape[2], zshape[3], self.transformer.config.factorized_bits[1])
            # index_post -= 2**(zshape[1] // 2) ## no need since the seperate head
            # index_post[index_post < 0] = 0 #in case the overflow of index but usually not happended
            quant_pre = self.first_stage_model.quantize.get_codebook_entry(index_pre, bhwc_pre, order="pre")
            quant_post = self.first_stage_model.quantize.get_codebook_entry(index_post, bhwc_post, order="post")
            quant_z = torch.concat([quant_pre, quant_post], dim=1) # concate in the final dimension 
            x = self.first_stage_model.decode(quant_z)
        else:
            # index = self.permuter(index, reverse=True)
            bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
            quant_z = self.first_stage_model.quantize.get_codebook_entry(
                index, shape=bhwc)
            x = self.first_stage_model.decode(quant_z)
        return x

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        if self.token_factorization:
            logits_pre, target_pre = logits[0], target[0]
            logits_post, target_post = logits[1], target[1]
            loss_pre = F.cross_entropy(logits_pre.reshape(-1, logits_pre.size(-1)), target_pre.reshape(-1))
            loss_post = F.cross_entropy(logits_post.reshape(-1, logits_post.size(-1)), target_post.reshape(-1))
            loss = loss_pre + loss_post
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss, (loss_pre, loss_post)

    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt = self.optimizers()
            for opt_param_group in opt.param_groups:
                opt_param_group["lr"] = self.resume_lr

    def training_step(self, batch, batch_idx):
        ## control lr and wd
        ## --------------------------------------------
        iters_train = len(self.trainer.train_dataloader) ## get the total iterations in a epoch
        g_it = self.trainer.global_step
        max_it = self.trainer.max_epochs * iters_train
        wp_it = self.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = self.lr_wd_annealing(self.learning_rate, self.weight_decay, self.twde, g_it, wp_it, max_it, wp0=self.wp0, wpe=self.wpe)
        ## --------------------------------------------
        loss, (loss_pre, loss_post) = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_pre", loss_pre, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_post", loss_post, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, (loss_pre, loss_post) = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_pre", loss_pre, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss_post", loss_post, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following NanoGPT, since we adopt the Llama-Like framework for AutoRegressive Visual Generation
        """
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused_available = False
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95), **extra_args)

        return optimizer
    
    def lr_wd_annealing(self, peak_lr, wd, wd_end, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
        """
        Modified from VAR
        """
        wp_it = round(wp_it)
        if cur_it < wp_it:
            cur_lr = wp0 + (1-wp0) * cur_it / wp_it
        else:
            pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
            rest = 1 - pasd     # [1, 0]
            ## using linear decay by default
            T = 0.05; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else: cur_lr = wpe + (1-wpe) * rest / max_rest

        cur_lr *= peak_lr
        pasd = cur_it / (max_it-1)
        cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))
    
        inf = 1e6
        min_lr, max_lr = inf, -1
        min_wd, max_wd = inf, -1
        for param_group in self.optimizers().param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
            max_lr = max(max_lr, param_group['lr'])
            min_lr = min(min_lr, param_group['lr'])
            
            param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)
            max_wd = max(max_wd, param_group['weight_decay'])
            if param_group['weight_decay'] > 0:
                min_wd = min(min_wd, param_group['weight_decay'])

        if min_lr == inf: min_lr = -1
        if min_wd == inf: min_wd = -1
        return min_lr, max_lr, min_wd, max_wd