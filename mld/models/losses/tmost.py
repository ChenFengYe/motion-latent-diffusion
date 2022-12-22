import torch
import torch.nn as nn
from torchmetrics import Metric

class TmostLosses(Metric):
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

        # Loss notes
        # c:content s:style
        # m:motion  t:text
        # e.g. cmsm: content motion + content style

        # data mode => xyz?
        # recons loss
        losses = []
        losses.append("recons_mm2m")
        losses.append("recons_t2m")

        losses.append("cross_mt2m")
        losses.append("cross_tm2m")

        # cycle consistency loss
        losses.append("cycle_cmsm2mContent")
        losses.append("cycle_cmsm2mStyle")

        # latent loss
        losses.append("latent_ct2cm")
        losses.append("latent_st2sm")

        # KL loss
        losses.append("kl_motion")
        losses.append("kl_text")
        losses.append("kl_ct2cm")
        losses.append("kl_cm2ct")

        losses.append("total")

        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0))
        self.losses = losses

        self.ablation_cycle = cfg.TRAIN.ABLATION.CYCLE

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
        """
        loss list
        - triplet loss
            - anchor style1
            - pos    style2
            - neg    diff_style
                anchor = s_xa
                pos = s_xpos
                neg = self.gen.enc_style(co_data[diff_style], diff_style[-2:])
                l_triplet = self.triplet_loss(anchor, pos, neg)
        - 
        """

        # ToDo
        # may only need half compute

        # ToDo
        # ref to XYZDatastruct or SMPLDatastruct
        # /apdcephfs/share_1227775/shingxchen/AIMotion/TMOST/tmost/transforms/xyz.py

        # loss1 - reconstruction loss
        #       - from one motion-text pair
        total += self._update_loss("recons_mm2m", rs_set['rs_cm1sm1'], rs_set['m1'])
        total += self._update_loss("recons_t2m", rs_set['rs_ct1st1'], rs_set['m1'])

        # loss - cross reconstruction loss
        total += self._update_loss("cross_mt2m", rs_set['rs_cm1st1'], rs_set['m1'])
        total += self._update_loss("cross_tm2m", rs_set['rs_ct1sm1'], rs_set['m1'])

        # total += self._update_loss("recons_tm2m", rs_set['rs_c1st1'], rs_set['m1'])
        # total += self._update_loss("recons_m2m", rs_set['rs_cm2sm2'], m2)
        # total += self._update_loss("recons_mt2m", rs_set['rs_cm2st2'], m2)
        # total += self._update_loss("recons_cmst2m", ds_motion.cmst_jfeats, ds_ref.jfeats)

        # loss2 - cycle cotent/style consistency loss
        #       - from cross motion-motion pair
        if self.ablation_cycle:
            total += self._update_loss("cycle_cmsm2mContent", rs_set['cyc_rs_cm1sm1'], rs_set['m1'])
            total += self._update_loss("cycle_cmsm2mStyle", rs_set['cyc_rs_cm2sm2'], rs_set['m2'])

        # [to-do] loss for labeled style motions
        # refer to deep-motion-editing datasets
        # xxx

        # [to-do] loss for text style words / content words
        # check KIT dataset and HumanML dataset
        # xxx

        # loss3 - text motion latent loss
        total += self._update_loss("latent_ct2cm", rs_set['lat_ct1'], rs_set['lat_cm1'])
        total += self._update_loss("latent_st2sm", rs_set['lat_st1'], rs_set['lat_sm1'])

        # loss4 - content loss!!!

        # loss5 - text encoder/decoder loss!!!

        # loss - kl loss
        total += self._update_loss("kl_motion", rs_set['dist_cm1'], dist_ref)
        # total += self._update_loss("kl_motion", rs_set['dist_sm1'], dist_ref)

        total += self._update_loss("kl_text", rs_set['dist_ct1'], dist_ref)
        # total += self._update_loss("kl_text", rs_set['dist_st1'], dist_ref)

        total += self._update_loss("kl_ct2cm", rs_set['dist_ct1'], rs_set['dist_cm1'])
        total += self._update_loss("kl_cm2ct", rs_set['dist_cm1'], rs_set['dist_ct1'])

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
