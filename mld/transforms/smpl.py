from typing import Optional
from torch import Tensor
import numpy as np
import torch
import contextlib
from .base import Datastruct, dataclass, Transform
import os
from .rots2rfeats import Rots2Rfeats
from .rots2joints import Rots2Joints
from .joints2jfeats import Joints2Jfeats


class SMPLTransform(Transform):

    def __init__(self, rots2rfeats: Rots2Rfeats, rots2joints: Rots2Joints,
                 joints2jfeats: Joints2Jfeats, **kwargs):
        self.rots2rfeats = rots2rfeats
        self.rots2joints = rots2joints
        self.joints2jfeats = joints2jfeats

    def Datastruct(self, **kwargs):
        return SMPLDatastruct(_rots2rfeats=self.rots2rfeats,
                              _rots2joints=self.rots2joints,
                              _joints2jfeats=self.joints2jfeats,
                              transforms=self,
                              **kwargs)

    def __repr__(self):
        return "SMPLTransform()"


class RotIdentityTransform(Transform):

    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return RotTransDatastruct(**kwargs)

    def __repr__(self):
        return "RotIdentityTransform()"


@dataclass
class RotTransDatastruct(Datastruct):
    rots: Tensor
    trans: Tensor

    transforms: RotIdentityTransform = RotIdentityTransform()

    def __post_init__(self):
        self.datakeys = ["rots", "trans"]

    def __len__(self):
        return len(self.rots)


@dataclass
class SMPLDatastruct(Datastruct):
    transforms: SMPLTransform
    _rots2rfeats: Rots2Rfeats
    _rots2joints: Rots2Joints
    _joints2jfeats: Joints2Jfeats

    features: Optional[Tensor] = None
    rots_: Optional[RotTransDatastruct] = None
    rfeats_: Optional[Tensor] = None
    joints_: Optional[Tensor] = None
    jfeats_: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features", "rots_", "rfeats_", "joints_", "jfeats_"]
        # starting point
        if self.features is not None and self.rfeats_ is None:
            self.rfeats_ = self.features

    @property
    def rots(self):
        # Cached value
        if self.rots_ is not None:
            return self.rots_

        # self.rfeats_ should be defined
        assert self.rfeats_ is not None

        self._rots2rfeats.to(self.rfeats.device)
        self.rots_ = self._rots2rfeats.inverse(self.rfeats)
        return self.rots_

    @property
    def rfeats(self):
        # Cached value
        if self.rfeats_ is not None:
            return self.rfeats_

        # self.rots_ should be defined
        assert self.rots_ is not None

        self._rots2rfeats.to(self.rots.device)
        self.rfeats_ = self._rots2rfeats(self.rots)
        return self.rfeats_

    @property
    def joints(self):
        # Cached value
        if self.joints_ is not None:
            return self.joints_

        self._rots2joints.to(self.rots.device)
        self.joints_ = self._rots2joints(self.rots)
        return self.joints_

    @property
    def jfeats(self):
        # Cached value
        if self.jfeats_ is not None:
            return self.jfeats_

        self._joints2jfeats.to(self.joints.device)
        self.jfeats_ = self._joints2jfeats(self.joints)
        return self.jfeats_

    def __len__(self):
        return len(self.rfeats)


# This code is based on https://github.com/Mathux/ACTOR.git
from smplx import SMPLLayer as _SMPLLayer
from smplx.lbs import vertices2joints

# action2motion_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 24, 38]
# change 0 and 8
action2motion_joints = [
    8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38
]

SMPL_DATA_PATH = 'deps/smpl'

JOINTSTYPE_ROOT = {
    "a2m": 0,  # action2motion
    "smpl": 0,
    "a2mpl": 0,  # set(smpl, a2m)
    "vibe": 8
}  # 0 is the 8 position: OP MidHip below

JOINT_MAP = {
    'OP Nose': 24,
    'OP Neck': 12,
    'OP RShoulder': 17,
    'OP RElbow': 19,
    'OP RWrist': 21,
    'OP LShoulder': 16,
    'OP LElbow': 18,
    'OP LWrist': 20,
    'OP MidHip': 0,
    'OP RHip': 2,
    'OP RKnee': 5,
    'OP RAnkle': 8,
    'OP LHip': 1,
    'OP LKnee': 4,
    'OP LAnkle': 7,
    'OP REye': 25,
    'OP LEye': 26,
    'OP REar': 27,
    'OP LEar': 28,
    'OP LBigToe': 29,
    'OP LSmallToe': 30,
    'OP LHeel': 31,
    'OP RBigToe': 32,
    'OP RSmallToe': 33,
    'OP RHeel': 34,
    'Right Ankle': 8,
    'Right Knee': 5,
    'Right Hip': 45,
    'Left Hip': 46,
    'Left Knee': 4,
    'Left Ankle': 7,
    'Right Wrist': 21,
    'Right Elbow': 19,
    'Right Shoulder': 17,
    'Left Shoulder': 16,
    'Left Elbow': 18,
    'Left Wrist': 20,
    'Neck (LSP)': 47,
    'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49,
    'Thorax (MPII)': 50,
    'Spine (H36M)': 51,
    'Jaw (H36M)': 52,
    'Head (H36M)': 53,
    'Nose': 24,
    'Left Eye': 26,
    'Right Eye': 25,
    'Left Ear': 28,
    'Right Ear': 27
}

JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist',
    'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip',
    'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye',
    'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel',
    'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee',
    'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist',
    'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow',
    'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)',
    'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose',
    'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear'
]


# adapted from VIBE/SPIN to output smpl_joints, vibe joints and action2motion joints
class SMPL(_SMPLLayer):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, smpl_path=SMPL_DATA_PATH, **kwargs):
        model_path = os.path.join(smpl_path, "SMPL_NEUTRAL.pkl")
        J_path = os.path.join(smpl_path, 'J_regressor_extra.npy')
        kwargs["model_path"] = model_path

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)

        J_regressor_extra = np.load(J_path)
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        vibe_indexes = np.array([JOINT_MAP[i] for i in JOINT_NAMES])
        a2m_indexes = vibe_indexes[action2motion_joints]
        smpl_indexes = np.arange(24)
        a2mpl_indexes = np.unique(np.r_[smpl_indexes, a2m_indexes])

        self.maps = {
            "vibe": vibe_indexes,
            "a2m": a2m_indexes,
            "smpl": smpl_indexes,
            "a2mpl": a2mpl_indexes
        }

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        extra_joints = vertices2joints(self.J_regressor_extra,
                                       smpl_output.vertices)
        all_joints = torch.cat([smpl_output.joints, extra_joints], dim=1)

        output = {"vertices": smpl_output.vertices}

        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]

        return output
