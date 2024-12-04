import torch
import torch.nn as nn
import torch.nn.functional as F

from src.IBQ.modules.losses.lpips import LPIPS
from src.IBQ.modules.discriminator.model import NLayerDiscriminator, weights_init


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
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", gen_loss_weight=None, lecam_loss_weight=None,
                 quant_loss_weight=1.0, entropy_loss_weight=1.0):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
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
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.quant_loss_weight = quant_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

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

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss.clone()
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            nll_loss = nll_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = torch.mean(nll_loss)

        use_entropy_loss = isinstance(codebook_loss, tuple)

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

            # pytorch lightning global_step bug when useing multple optimizers
            # disc_factor = adopt_weight(self.disc_factor, int(global_step / 2), threshold=self.discriminator_iter_start)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            real_g_loss = disc_factor * g_loss
            g_loss = d_weight * disc_factor * g_loss
            if not use_entropy_loss:
                codebook_loss = self.codebook_weight * codebook_loss
            else:
                quant_loss, sample_entropy_loss, avg_entropy_loss, entropy_loss = codebook_loss
                codebook_loss = self.quant_loss_weight * quant_loss + self.entropy_loss_weight * entropy_loss

            loss = nll_loss + g_loss + codebook_loss

            if not use_entropy_loss:
                if disc_factor == 0:
                    log = {"{}/total_loss".format(split): loss.clone().detach(),
                           "{}/codebook_quant_loss".format(split): codebook_loss.detach(),
                           "{}/nll_loss".format(split): nll_loss.detach(),
                           "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                           "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                           "{}/d_weight".format(split): torch.tensor(0.0),
                           "{}/disc_factor".format(split): torch.tensor(0.0),
                           "{}/g_loss".format(split): torch.tensor(0.0),
                           "{}/unsacled_g_loss".format(split): torch.tensor(0.0),
                           }
                else:
                    if self.training:
                        log = {"{}/total_loss".format(split): loss.clone().detach(),
                               "{}/codebook_quant_loss".format(split): codebook_loss.detach(),
                               "{}/nll_loss".format(split): nll_loss.detach(),
                               "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                               "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                               "{}/d_weight".format(split): d_weight.detach(),
                               "{}/disc_factor".format(split): torch.tensor(disc_factor),
                               "{}/g_loss".format(split): g_loss.detach(),
                               "{}/unsacled_g_loss".format(split): real_g_loss.detach(),
                               }
                    else:
                        log = {"{}/total_loss".format(split): loss.clone().detach(),
                               "{}/codebook_quant_loss".format(split): codebook_loss.detach(),
                               "{}/nll_loss".format(split): nll_loss.detach(),
                               "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                               "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                               "{}/d_weight".format(split): d_weight.detach(),
                               "{}/disc_factor".format(split): torch.tensor(disc_factor),
                               "{}/g_loss".format(split): real_g_loss.detach(),
                               }
            else:
                if disc_factor == 0:
                    log = {"{}/total_loss".format(split): loss.clone().detach(),
                           "{}/codebook_quant_loss".format(split): quant_loss.detach(),
                           "{}/codebook_entropy_loss".format(split): entropy_loss.detach(),
                           "{}/codebook_sample_entropy_loss".format(split): sample_entropy_loss.detach(),
                           "{}/codebook_avg_entropy_loss".format(split): avg_entropy_loss.detach(),
                           "{}/codebook_loss".format(split): codebook_loss.detach(),
                           "{}/nll_loss".format(split): nll_loss.detach(),
                           "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                           "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                           "{}/d_weight".format(split): torch.tensor(0.0),
                           "{}/disc_factor".format(split): torch.tensor(0.0),
                           "{}/g_loss".format(split): torch.tensor(0.0),
                           "{}/unsacled_g_loss".format(split): torch.tensor(0.0),
                           }
                else:
                    if self.training:
                        log = {"{}/total_loss".format(split): loss.clone().detach(),
                               "{}/codebook_quant_loss".format(split): quant_loss.detach(),
                               "{}/codebook_entropy_loss".format(split): entropy_loss.detach(),
                               "{}/codebook_sample_entropy_loss".format(split): sample_entropy_loss.detach(),
                               "{}/codebook_avg_entropy_loss".format(split): avg_entropy_loss.detach(),
                               "{}/codebook_loss".format(split): codebook_loss.detach(),
                               "{}/nll_loss".format(split): nll_loss.detach(),
                               "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                               "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                               "{}/d_weight".format(split): d_weight.detach(),
                               "{}/disc_factor".format(split): torch.tensor(disc_factor),
                               "{}/g_loss".format(split): g_loss.detach(),
                               "{}/unsacled_g_loss".format(split): real_g_loss.detach(),
                               }
                    else:
                        log = {"{}/total_loss".format(split): loss.clone().detach(),
                               "{}/codebook_quant_loss".format(split): quant_loss.detach(),
                               "{}/codebook_entropy_loss".format(split): entropy_loss.detach(),
                               "{}/codebook_sample_entropy_loss".format(split): sample_entropy_loss.detach(),
                               "{}/codebook_avg_entropy_loss".format(split): avg_entropy_loss.detach(),
                               "{}/codebook_loss".format(split): codebook_loss.detach(),
                               "{}/nll_loss".format(split): nll_loss.detach(),
                               "{}/reconstruct_loss".format(split): rec_loss.detach().mean(),
                               "{}/perceptual_loss".format(split): p_loss.detach().mean(),
                               "{}/d_weight".format(split): d_weight.detach(),
                               "{}/disc_factor".format(split): torch.tensor(disc_factor),
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

            # disc_factor = adopt_weight(self.disc_factor, int(global_step / 2), threshold=self.discriminator_iter_start)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            if self.lecam_loss_weight is not None and disc_factor > 0: ## update only when disc_factor > 0
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_loss = disc_factor * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_loss = disc_factor * non_saturate_d_loss

            if disc_factor == 0:
                log = {"{}/disc_loss".format(split): torch.tensor(0.0),
                       "{}/logits_real".format(split): torch.tensor(0.0),
                       "{}/logits_fake".format(split): torch.tensor(0.0),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/lecam_loss".format(split): torch.tensor(0.0),
                       }
            else:
                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                       "{}/logits_real".format(split): logits_real.detach().mean(),
                       "{}/logits_fake".format(split): logits_fake.detach().mean(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/lecam_loss".format(split): lecam_loss.detach(),
                       }
            return d_loss, log
