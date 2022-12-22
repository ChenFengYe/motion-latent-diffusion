# todo remove this file
import torch

# todo remove 
# ---
def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
    """
    Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def compute_2d_loss(model, batch):
    '''
    keypoints loss
    '''
    gt = batch["kp_2d"]
    out = batch["pred_2d"]
    mask = batch["mask"]
  
    gtmasked = gt[mask]
    outmasked = out[mask]    
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_limb_loss(model, batch):
    # limb position loss
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]

    # remove glob translation
    # [bs njoint nfeats lenghs] = > [bs lengths njoints nfeats]
    rootindex = JOINTSTYPE_ROOT[model.jointstype]
    gt = x - x[:,:,[rootindex],:]
    out = output - output[:,:,[rootindex],:]    
    
    limbndex = JOINTSTYPE_LIMB[model.jointstype]
    gtmasked = gt[:,:,limbndex,:][mask]
    outmasked = out[:,:,limbndex,:][mask]
    
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_glob_loss(model, batch):
    # glob rotation for the first (root) joint
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]

    # [bs njoint nfeats lenghs] = > [bs lengths njoints nfeats]    
    rootindex = JOINTSTYPE_ROOT[model.jointstype]
    gtmasked = x[:,:,[rootindex],:][mask]
    outmasked = output[:,:,[rootindex],:][mask]
    
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_theta_loss(model, batch):
    x = batch['theta']
    output = batch["output_theta"]
    mask = batch["mask"]

    gtmasked = x[mask]
    outmasked = output[mask]
    
    # translation loss
    root_index = THETA_MAP['root']
    w_root = batch["w_root"][mask][:,None]
    gtmasked[:,root_index] *= w_root
    outmasked[:,root_index] *= w_root 

    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_rc_loss(model, batch):
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]

    gtmasked = x[mask]
    outmasked = output[mask]
    
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_rcxyz_loss(model, batch):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]

    # dummpy
    # ---ignore global output for no global dataset---
    root_index = THETA_MAP['root']
    w_root = batch["w_root"][mask][:,None,None]
    trans = batch['theta'][:,:,None,root_index,...][mask]
    output_trans = batch['output_theta'][:,:,None,root_index][mask]
    
    gtmasked = x[mask]
    outmasked = output[mask]
    
    gtmasked -= trans*(1-w_root)
    outmasked -= output_trans*(1-w_root)
    # -------------------------------------------------
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_rcverts_loss(model, batch):
    x = batch["x_vertices"]
    output = batch["output_vertices"]
    mask = batch["mask"]

    # dummy
    # ---ignore global output for no global dataset---
    root_index = THETA_MAP['root']
    w_root = batch["w_root"][mask][:,None,None]
    trans = batch['theta'][:,:,None,root_index,...][mask]
    output_trans = batch['output_theta'][:,:,None,root_index][mask]
    
    gtmasked = x[mask]
    outmasked = output[mask]
    
    gtmasked -= trans*(1-w_root)
    outmasked -= output_trans*(1-w_root)
    # -------------------------------------------------
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_vel_loss(model, batch):
    x = batch["x"]
    output = batch["output"]
    gtvel = (x[:,1:,...] - x[:, :-1,...])
    outputvel = (output[:,1:,...] - output[:,1:,...])

    mask = batch["mask"][:,1:]
    
    gtvelmasked = gtvel[mask]
    outvelmasked = outputvel[mask]
    
    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


def compute_velxyz_loss(model, batch):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    gtvel = (x[:,1:,...] - x[:,:-1,...])
    outputvel = (output[:,1:,...] - output[:,:-1,...])

    mask = batch["mask"][:, 1:]
    
    gtvelmasked = gtvel[mask]
    outvelmasked = outputvel[mask]
    
    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


def compute_hp_loss(model, batch):
    loss = hessian_penalty(model.return_latent, batch, seed=torch.random.seed())
    return loss


def compute_kl_loss(model, batch):
    mu, logvar = batch["mu"], batch["logvar"]
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss

_matching_ = {"rc": compute_rc_loss, "kl": compute_kl_loss, "hp": compute_hp_loss,
              "rcxyz": compute_rcxyz_loss,
              "vel": compute_vel_loss, "velxyz": compute_velxyz_loss, 
              "glob":compute_glob_loss, "limb":compute_limb_loss, "rcverts": compute_rcverts_loss,
              "theta": compute_theta_loss, "2d": compute_2d_loss}

def get_loss_function(ltype):
    return _matching_[ltype]


def get_loss_names():
    return list(_matching_.keys())
# ---
