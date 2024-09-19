import torch
import torch.nn as nn
import torch.nn.functional as F

from open_magvit2.taming.modules.losses.lpips import LPIPS
from open_magvit2.taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def _sigmoid_cross_entropy_with_logits(labels, logits):
    """
    non-saturating loss 
    """
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = (logits >= zeros)
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))

def non_saturate_gen_loss(logits_fake):
    """
    logits_fake: [B 1 H W]
    """
    B, _, _, _ = logits_fake.shape
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = torch.mean(logits_fake, dim=-1)
    gen_loss = torch.mean(_sigmoid_cross_entropy_with_logits(
        labels = torch.ones_like(logits_fake), logits=logits_fake
    ))
    
    return gen_loss

def non_saturate_discriminator_loss(logits_real, logits_fake):
    B, _, _, _ = logits_fake.shape
    logits_real = logits_real.reshape(B, -1)
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = logits_fake.mean(dim=-1)
    logits_real = logits_real.mean(dim=-1)

    real_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.ones_like(logits_real), logits=logits_real)

    fake_loss = _sigmoid_cross_entropy_with_logits(
        labels= torch.zeros_like(logits_fake), logits=logits_fake
    )

    discr_loss = real_loss.mean() + fake_loss.mean()
    return discr_loss

class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay
    
    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1- self.decay) 
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)
    
def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
            torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 commit_weight = 0.25, codebook_enlarge_ratio=3, codebook_enlarge_steps=2000,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", gen_loss_weight=None, lecam_loss_weight=None):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "non_saturate"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.commit_weight = commit_weight
        self.codebook_enlarge_ratio = codebook_enlarge_ratio
        self.codebook_enlarge_steps = codebook_enlarge_steps
        self.gen_loss_weight = gen_loss_weight
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()
        
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non_saturate":
            self.disc_loss = non_saturate_discriminator_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, loss_break, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss.clone()
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            nll_loss = nll_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            g_loss = non_saturate_gen_loss(logits_fake)
            if self.gen_loss_weight is None:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(self.gen_loss_weight)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if not self.training:
                real_g_loss = disc_factor * g_loss
            g_loss = d_weight * disc_factor * g_loss
            
            scale_codebook_loss = self.codebook_weight * codebook_loss #entropy_loss
            if self.codebook_enlarge_ratio > 0:
                scale_codebook_loss = self.codebook_enlarge_ratio * (max(0, 1 - global_step / self.codebook_enlarge_steps)) * scale_codebook_loss + scale_codebook_loss

            loss = nll_loss + g_loss + scale_codebook_loss + loss_break.commitment * self.commit_weight
            if disc_factor == 0:
                log = {"{}/total_loss".format(split): loss.clone().detach(),
                       "{}/per_sample_entropy".format(split): loss_break.per_sample_entropy.detach(),
                       "{}/codebook_entropy".format(split): loss_break.codebook_entropy.detach(),
                       "{}/commit_loss".format(split): loss_break.commitment.detach(),
                       "{}/nll_loss".format(split): nll_loss.detach(),
                       "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                       "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                       "{}/d_weight".format(split): torch.tensor(0.0),
                       "{}/disc_factor".format(split): torch.tensor(0.0),
                       "{}/g_loss".format(split): torch.tensor(0.0),
                       }
            else:
                if self.training:
                    log = {"{}/total_loss".format(split): loss.clone().detach(),
                           "{}/per_sample_entropy".format(split): loss_break.per_sample_entropy.detach(),
                           "{}/codebook_entropy".format(split): loss_break.codebook_entropy.detach(),
                           "{}/commit_loss".format(split): loss_break.commitment.detach(),
                           "{}/entropy_loss".format(split): codebook_loss.detach(),
                           "{}/nll_loss".format(split): nll_loss.detach(),
                           "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                           "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                           "{}/d_weight".format(split): d_weight,
                           "{}/disc_factor".format(split): torch.tensor(disc_factor),
                           "{}/g_loss".format(split): g_loss.detach(),
                           }
                else:
                    # validation only monitor the reconstruct_loss and p_loss
                    log = {
                           "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                           "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                           "{}/g_loss".format(split): real_g_loss.detach(),
                           }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            #---------------------------------------------------------------------------------------
            # Non-Saturate Loss is the Format of GAN Training, for D Loss, We still adopt Hinge Loss
            #---------------------------------------------------------------------------------------
            if self.lecam_loss_weight is not None:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_loss = disc_factor * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_loss = disc_factor * non_saturate_d_loss

            # d_loss = disc_factor * 
            if disc_factor == 0:
                log = {"{}/disc_loss".format(split): torch.tensor(0.0),
                       "{}/logits_real".format(split): torch.tensor(0.0),
                       "{}/logits_fake".format(split): torch.tensor(0.0),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/lecam_loss".format(split): lecam_loss.detach(),
                       "{}/non_saturated_d_loss".format(split): non_saturate_d_loss.detach(),
                       }
            else:
                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                       "{}/logits_real".format(split): logits_real.detach().mean(),
                       "{}/logits_fake".format(split): logits_fake.detach().mean(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/lecam_loss".format(split): lecam_loss.detach(),
                       "{}/non_saturated_d_loss".format(split): non_saturate_d_loss.detach(),
                       }
            return d_loss, log