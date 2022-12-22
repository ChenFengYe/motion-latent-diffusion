import torch
import torch.nn as nn
from torchmetrics import Metric


class TemosLosses(Metric):
    """
    Loss
    Modify loss
    refer to temos loss
    add loss like deep-motion-editing
    'gen_loss_total': l_total,
    'gen_loss_adv': l_adv,
    'gen_loss_recon_all': l_rec,
    'gen_loss_recon_r': l_r_rec,
    'gen_loss_recon_s': l_s_rec,
    'gen_loss_feature_all': l_ft,
    'gen_loss_feature_r': l_ft_r,
    'gen_loss_feature_s': l_ft_s,
    'gen_loss_feature_t': l_ft_t,
    'gen_loss_quaternion': l_qt,
    'gen_loss_twist': l_tw,
    'gen_loss_triplet': l_triplet,
    'gen_loss_joint': l_joint,
           
    """

    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        self.vae = vae
        self.mode = mode

        loss_on_both = False
        force_loss_on_jfeats = True
        ablation_no_kl_combine = False
        ablation_no_kl_gaussian = False
        ablation_no_motionencoder = False

        self.loss_on_both = loss_on_both
        self.ablation_no_kl_combine = ablation_no_kl_combine
        self.ablation_no_kl_gaussian = ablation_no_kl_gaussian
        self.ablation_no_motionencoder = ablation_no_motionencoder

        losses = []
        if mode == "xyz" or force_loss_on_jfeats:
            if not ablation_no_motionencoder:
                losses.append("recons_jfeats2jfeats")
            losses.append("recons_text2jfeats")
        if mode == "smpl":
            if not ablation_no_motionencoder:
                losses.append("recons_rfeats2rfeats")
            losses.append("recons_text2rfeats")
        else:
            ValueError("This mode is not recognized.")

        if vae or loss_on_both:
            kl_losses = []
            if not ablation_no_kl_combine and not ablation_no_motionencoder:
                kl_losses.extend(["kl_text2motion", "kl_motion2text"])
            if not ablation_no_kl_gaussian:
                if ablation_no_motionencoder:
                    kl_losses.extend(["kl_text"])
                else:
                    kl_losses.extend(["kl_text", "kl_motion"])
            losses.extend(kl_losses)

        if not self.vae or loss_on_both:
            if not ablation_no_motionencoder:
                losses.append("latent_manifold")
        losses.append("total")

        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0))
        #     self.register_buffer(loss, default=torch.tensor(0.0), dist_reduce_fx="sum")
        # self.register_buffer("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        # Instantiate loss functions
        # self._losses_func = {loss: hydra.utils.instantiate(kwargs[loss + "_func"])
        #                      for loss in losses if loss != "total"}
        self._losses_func = {}
        self._params = {}

        for loss in losses:
            if loss != 'total':
                if loss.split('_')[0] == 'kl':
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
                elif loss.split('_')[0] == 'recons':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(
                        reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_REC
                elif loss.split('_')[0] == 'latent':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(
                        reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_LATENT
                elif loss.split('_')[0] == 'cycle':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(
                        reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_CYCLE
                else:
                    ValueError("This loss is not recognized.")

    def update(self,
               f_text=None,
               f_motion=None,
               f_ref=None,
               lat_text=None,
               lat_motion=None,
               dis_text=None,
               dis_motion=None,
               dis_ref=None):
        total: float = 0.0

        if self.mode == "xyz" or self.force_loss_on_jfeats:
            if not self.ablation_no_motionencoder:
                total += self._update_loss("recons_jfeats2jfeats", f_motion,
                                           f_ref)
            total += self._update_loss("recons_text2jfeats", f_text, f_ref)

        if self.mode == "smpl":
            if not self.ablation_no_motionencoder:
                total += self._update_loss("recons_rfeats2rfeats",
                                           f_motion.rfeats, f_ref.rfeats)
            total += self._update_loss("recons_text2rfeats", f_text.rfeats,
                                       f_ref.rfeats)

        if self.vae or self.loss_on_both:
            if not self.ablation_no_kl_combine and not self.ablation_no_motionencoder:
                total += self._update_loss("kl_text2motion", dis_text,
                                           dis_motion)
                total += self._update_loss("kl_motion2text", dis_motion,
                                           dis_text)
            if not self.ablation_no_kl_gaussian:
                total += self._update_loss("kl_text", dis_text, dis_ref)
                if not self.ablation_no_motionencoder:
                    total += self._update_loss("kl_motion", dis_motion,
                                               dis_ref)
        if not self.vae or self.loss_on_both:
            if not self.ablation_no_motionencoder:
                total += self._update_loss("latent_manifold", lat_text,
                                           lat_motion)

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name


class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
