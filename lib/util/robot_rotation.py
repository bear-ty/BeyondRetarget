"""Rotation conversions used by RGB2Robo inference."""

import torch
import torch.nn.functional as functional

from lib.util.geometry import rotation_matrix_to_quaternion


def rowwise_rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    """Decode the row-wise 6D root rotation representation used by this project."""
    flat = rot6d.reshape(-1, 6)
    row_1 = functional.normalize(flat[:, 0:3], dim=1, eps=1e-6)
    row_2_raw = flat[:, 3:6]
    projection = torch.sum(row_1 * row_2_raw, dim=1, keepdim=True)
    row_2 = functional.normalize(row_2_raw - projection * row_1, dim=1, eps=1e-6)
    row_3 = torch.cross(row_1, row_2, dim=1)
    return torch.stack([row_1, row_2, row_3], dim=1).reshape(*rot6d.shape[:-1], 3, 3)


def rot6d_to_quat_wxyz(rot6d: torch.Tensor) -> torch.Tensor:
    rotmat = rowwise_rot6d_to_rotmat(rot6d).reshape(-1, 3, 3)
    homogeneous_column = torch.zeros(rotmat.shape[0], 3, 1, dtype=rotmat.dtype, device=rotmat.device)
    homogeneous_column[:, 2, 0] = 1.0
    quat_wxyz = rotation_matrix_to_quaternion(torch.cat([rotmat, homogeneous_column], dim=-1))
    return quat_wxyz.view(*rot6d.shape[:-1], 4)
