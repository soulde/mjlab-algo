"""Shared online and expert AMP feature layout."""

import torch


def quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternion.dtype).eps)
    x, y, z, w = quaternion.unbind(dim=-1)
    column_0 = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + z * w),
            2 * (x * z - y * w),
        ),
        dim=-1,
    )
    column_1 = torch.stack(
        (
            2 * (x * y - z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z + x * w),
        ),
        dim=-1,
    )
    return torch.cat((column_0, column_1), dim=-1)


def build_amp_frame(
    root_height: torch.Tensor,
    root_quaternion: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    anchor_link_positions: torch.Tensor,
) -> torch.Tensor:
    """Build one AMP frame using the amp_go2 feature order."""
    return torch.cat(
        (
            root_height,
            quaternion_to_rotation_6d(root_quaternion),
            root_linear_velocity,
            root_angular_velocity,
            joint_position,
            joint_velocity,
            anchor_link_positions.flatten(start_dim=1),
        ),
        dim=-1,
    )
