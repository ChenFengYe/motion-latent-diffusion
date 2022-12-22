from typing import Optional

import torch
from torch import Tensor
from einops import rearrange

from mld.utils.temos_utils import matrix_to, nfeats_of, to_matrix
import mld.utils.geometry as geometry

from .base import Rots2Rfeats


class SMPLVelP(Rots2Rfeats):

    def __init__(self,
                 path: Optional[str] = None,
                 normalization: bool = False,
                 pose_rep: str = "rot6d",
                 canonicalize: bool = False,
                 offset: bool = True,
                 **kwargs) -> None:
        super().__init__(path=path, normalization=normalization)
        self.canonicalize = canonicalize
        self.pose_rep = pose_rep
        self.nfeats = nfeats_of(pose_rep)
        self.offset = offset

    def forward(self, data) -> Tensor:
        matrix_poses, trans = data.rots, data.trans
        # matrix_poses: [nframes, 22, 3, 3]

        # extract the root gravity axis
        # for smpl it is the last coordinate
        root_y = trans[..., 2]
        trajectory = trans[..., [0, 1]]

        # Comoute the difference of trajectory (for X and Y axis)
        vel_trajectory = torch.diff(trajectory, dim=-2)
        # 0 for the first one => keep the dimentionality
        vel_trajectory = torch.cat(
            (0 * vel_trajectory[..., [0], :], vel_trajectory), dim=-2)

        # first normalize the data
        if self.canonicalize:
            global_orient = matrix_poses[..., 0, :, :]
            # remove the rotation
            rot2d = geometry.matrix_to_axis_angle(global_orient[..., 0, :, :])
            # Remove the fist rotation along the vertical axis
            # construct this by extract only the vertical component of the rotation
            rot2d[..., :2] = 0

            if self.offset:
                # add a bit more rotation
                rot2d[..., 2] += torch.pi / 2

            rot2d = geometry.axis_angle_to_matrix(rot2d)

            # turn with the same amount all the rotations
            global_orient = torch.einsum("...kj,...kl->...jl", rot2d,
                                         global_orient)

            matrix_poses = torch.cat(
                (global_orient[..., None, :, :], matrix_poses[..., 1:, :, :]),
                dim=-3)

            # Turn the trajectory as well
            vel_trajectory = torch.einsum("...kj,...lk->...lj",
                                          rot2d[..., :2, :2], vel_trajectory)

        poses = matrix_to(self.pose_rep, matrix_poses)
        features = torch.cat(
            (root_y[..., None], vel_trajectory,
             rearrange(poses, "... joints rot -> ... (joints rot)")),
            dim=-1)
        features = self.normalize(features)
        return features

    def extract(self, features):
        root_y = features[..., 0]
        vel_trajectory = features[..., 1:3]
        poses_features = features[..., 3:]
        poses = rearrange(poses_features,
                          "... (joints rot) -> ... joints rot",
                          rot=self.nfeats)
        return root_y, vel_trajectory, poses

    def inverse(self, features):
        features = self.unnormalize(features)
        root_y, vel_trajectory, poses = self.extract(features)

        # integrate the trajectory
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # Get back the translation
        trans = torch.cat([trajectory, root_y[..., None]], dim=-1)
        matrix_poses = to_matrix(self.pose_rep, poses)

        from temos.transforms.smpl import RotTransDatastruct
        return RotTransDatastruct(rots=matrix_poses, trans=trans)
