# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
from torch.nn import functional as F


def matrix_of_angles(cos, sin, inv=False, dim=2):
    assert dim in [2, 3]
    sin = -sin if inv else sin
    if dim == 2:
        row1 = torch.stack((cos, -sin), axis=-1)
        row2 = torch.stack((sin, cos), axis=-1)
        return torch.stack((row1, row2), axis=-2)
    elif dim == 3:
        row1 = torch.stack((cos, -sin, 0 * cos), axis=-1)
        row2 = torch.stack((sin, cos, 0 * cos), axis=-1)
        row3 = torch.stack((0 * sin, 0 * cos, 1 + 0 * cos), axis=-1)
        return torch.stack((row1, row2, row3), axis=-2)


def matrot2axisangle(matrots):
    # This function is borrowed from https://github.com/davrempe/humor/utils/transforms.py
    # axisang N x 3
    """
    :param matrots: N*num_joints*9
    :return: N*num_joints*3
    """
    import cv2

    batch_size = matrots.shape[0]
    matrots = matrots.reshape([batch_size, -1, 9])
    out_axisangle = []
    for mIdx in range(matrots.shape[0]):
        cur_axisangle = []
        for jIdx in range(matrots.shape[1]):
            a = cv2.Rodrigues(matrots[mIdx, jIdx : jIdx + 1, :].reshape(3, 3))[
                0
            ].reshape((1, 3))
            cur_axisangle.append(a)

        out_axisangle.append(np.array(cur_axisangle).reshape([1, -1, 3]))
    return np.vstack(out_axisangle)


def axisangle2matrots(axisangle):
    # This function is borrowed from https://github.com/davrempe/humor/utils/transforms.py
    # axisang N x 3
    """
    :param axisangle: N*num_joints*3
    :return: N*num_joints*9
    """
    import cv2

    batch_size = axisangle.shape[0]
    axisangle = axisangle.reshape([batch_size, -1, 3])
    out_matrot = []
    for mIdx in range(axisangle.shape[0]):
        cur_axisangle = []
        for jIdx in range(axisangle.shape[1]):
            a = cv2.Rodrigues(axisangle[mIdx, jIdx : jIdx + 1, :].reshape(1, 3))[0]
            cur_axisangle.append(a)

        out_matrot.append(np.array(cur_axisangle).reshape([1, -1, 9]))
    return np.vstack(out_matrot)


def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)

    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = (
            torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(rot_mat.shape[0], -1, -1)
        )
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def estimate_translation_np(
    S, joints_2d, joints_conf, focal_length=5000.0, img_size=224.0
):
    """
    This function is borrowed from https://github.com/nkolot/SPIN/utils/geometry.py

    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length, focal_length])
    # optical center
    center = np.array([img_size / 2.0, img_size / 2.0])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000.0, img_size=224.0):
    """
    This function is borrowed from https://github.com/nkolot/SPIN/utils/geometry.py

    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, 25:, :].cpu().numpy()
    joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float6432)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size
        )
    return torch.from_numpy(trans).to(device)


def rot6d_to_rotmat_spin(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)

    # inp = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    # denom = inp.pow(2).sum(dim=1).sqrt().unsqueeze(-1) + 1e-8
    # b2 = inp / denom

    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


import mld.utils.rotation_conversions as rotation_conversions


def rot6d(x_rotations, pose_rep):
    time, njoints, feats = x_rotations.shape

    # Compute rotations (convert only masked sequences output)
    if pose_rep == "rotvec":
        rotations = rotation_conversions.axis_angle_to_matrix(x_rotations)
    elif pose_rep == "rotmat":
        rotations = x_rotations.view(njoints, 3, 3)
    elif pose_rep == "rotquat":
        rotations = rotation_conversions.quaternion_to_matrix(x_rotations)
    elif pose_rep == "rot6d":
        rotations = rotation_conversions.rotation_6d_to_matrix(x_rotations)
    else:
        raise NotImplementedError("No geometry for this one.")

    rotations_6d = rotation_conversions.matrix_to_rotation_6d(rotations)
    return rotations_6d


def rot6d_batch(x_rotations, pose_rep):
    nsamples, time, njoints, feats = x_rotations.shape

    # Compute rotations (convert only masked sequences output)
    if pose_rep == "rotvec":
        rotations = rotation_conversions.axis_angle_to_matrix(x_rotations)
    elif pose_rep == "rotmat":
        rotations = x_rotations.view(-1, njoints, 3, 3)
    elif pose_rep == "rotquat":
        rotations = rotation_conversions.quaternion_to_matrix(x_rotations)
    elif pose_rep == "rot6d":
        rotations = rotation_conversions.rotation_6d_to_matrix(x_rotations)
    else:
        raise NotImplementedError("No geometry for this one.")

    rotations_6d = rotation_conversions.matrix_to_rotation_6d(rotations)
    return rotations_6d


def rot6d_to_rotvec_batch(pose):
    # nsamples, time, njoints, feats = rot6d.shape
    bs, nfeats = pose.shape
    rot6d = pose.reshape(bs, 24, 6)
    rotations = rotation_conversions.rotation_6d_to_matrix(rot6d)
    rotvec = rotation_conversions.matrix_to_axis_angle(rotations)
    return rotvec.reshape(bs, 24 * 3)
