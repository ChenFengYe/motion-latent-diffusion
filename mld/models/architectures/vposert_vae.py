from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder, SkipTransformerDecoder, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
'''
vae
skip connection encoder 
skip connection decoder
mem for each decoder layer
'''


class VPosert(nn.Module):

    def __init__(self, cfg, **kwargs) -> None:

        super(VPosert, self).__init__()

        num_neurons = 512
        self.latentD = 256

        # self.num_joints = 21
        n_features = 196 * 263

        self.encoder_net = nn.Sequential(
            BatchFlatten(), nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons), nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons), nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD))

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, n_features),
            ContinousRotReprDecoder(),
        )

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        q_z = self.encode(features)
        feats_rst = self.decode(q_z)
        return feats_rst, q_z

    def encode(self, pose_body, lengths: Optional[List[int]] = None):
        '''
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        q_z = self.encoder_net(pose_body)
        q_z_sample = q_z.rsample()
        return q_z_sample.unsqueeze(0), q_z

    def decode(self, Zin, lengths: Optional[List[int]] = None):
        bs = Zin.shape[0]
        Zin = Zin[0]

        prec = self.decoder_net(Zin)

        return prec

    # def forward(self, pose_body):
    #     '''
    #     :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
    #     :param input_type: matrot / aa for matrix rotations or axis angles
    #     :param output_type: matrot / aa
    #     :return:
    #     '''

    #     q_z = self.encode(pose_body)
    #     q_z_sample = q_z.rsample()
    #     decode_results = self.decode(q_z_sample)
    #     decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
    #     return decode_results

    # def sample_poses(self, num_poses, seed=None):
    #     np.random.seed(seed)

    #     some_weight = [a for a in self.parameters()][0]
    #     dtype = some_weight.dtype
    #     device = some_weight.device
    #     self.eval()
    #     with torch.no_grad():
    #         Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype, device=device)

    #     return self.decode(Zgen)


class BatchFlatten(nn.Module):

    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ContinousRotReprDecoder(nn.Module):

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 196, 263)

        # b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        # dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        # b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        # b3 = torch.cross(b1, b2, dim=1)

        # return torch.stack([b1, b2, b3], dim=-1)
        return reshaped_input


class NormalDistDecoder(nn.Module):

    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout),
                                                 F.softplus(self.logvar(Xout)))
