import torch
import torch.nn as nn
from torchmetrics import Metric

class ACTORLosses(Metric):
    """
    Loss
    Modify loss
           
    """
    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        self.vae = vae
        self.mode = mode

        losses = []
        losses.append("recons_feature")
        losses.append("recons_verts")
        losses.append("recons_joints")
        losses.append("recons_limb")

        # latent loss
        losses.append("latent_st2sm")

        # KL loss
        losses.append("kl_motion")
        losses.append("total")

        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0))
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss !='total':
                if loss.split('_')[0] == 'kl':
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
                elif loss.split('_')[0] == 'recons':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_REC
                elif loss.split('_')[0] == 'cross':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_CROSS
                elif loss.split('_')[0] =='latent':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_LATENT
                elif loss.split('_')[0] =='cycle':
                    self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                    self._params[loss] = cfg.LOSS.LAMBDA_CYCLE
                else:
                    ValueError("This loss is not recognized.")


    def update(self, rs_set, dist_ref):
        total: float = 0.0
        # Compute the losses
        # loss1 - reconstruction loss
        total += self._update_loss("recons_feature", rs_set['m_rst'], rs_set['m_ref'])
        # total += self._update_loss("recons_verts", rs_set['verts_rs'], rs_set['verts_ref'])
        # total += self._update_loss("recons_joints", rs_set['joints_rs'], rs_set['joints_ref'])
        # total += self._update_loss("recons_limb", rs_set['rs_base'], rs_set['m1'])

        # loss - text motion latent loss
        total += self._update_loss("kl_motion", rs_set['dist_m'], dist_ref)

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss)/count for loss in self.losses}

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
        return sum([self.klloss(q, p)
                    for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
