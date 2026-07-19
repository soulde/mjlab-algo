"""Expert motion transition loader adapted from amp_go2-main."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmrl.amp.features import build_amp_frame
from mmrl.config import get_config_value, require_config_value
from mmrl.memories import AMPTransitionBatch


@dataclass(frozen=True)
class GMRMotion:
    root_pos: np.ndarray
    root_rot: np.ndarray
    joint_pos: np.ndarray
    root_linear_vel: np.ndarray
    root_angular_vel: np.ndarray
    joint_vel: np.ndarray
    frame_duration: float


def load_gmr_motion(path: str | Path, joint_count: int) -> GMRMotion:
    """Load and validate one official-GMR pickle motion."""
    path = Path(path)
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"GMR motion {path}: expected a .pkl file.")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"GMR motion {path}: payload must be a mapping.")
    fps = float(payload.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"GMR motion {path}: fps must be positive and finite.")
    root_pos = _motion_array(payload, "root_pos", path, 3)
    root_rot = _motion_array(payload, "root_rot", path, 4, len(root_pos))
    joint_pos = _motion_array(
        payload, "dof_pos", path, joint_count, len(root_pos)
    )
    norms = np.linalg.norm(root_rot, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"GMR motion {path}: root_rot contains zero quaternion.")
    root_rot = root_rot / norms[:, None]
    for index in range(1, len(root_rot)):
        if np.dot(root_rot[index - 1], root_rot[index]) < 0.0:
            root_rot[index] *= -1.0
    frame_duration = 1.0 / fps
    root_velocity = np.zeros_like(root_pos)
    root_velocity[1:] = np.diff(root_pos, axis=0) / frame_duration
    joint_velocity = np.zeros_like(joint_pos)
    joint_velocity[1:] = np.diff(joint_pos, axis=0) / frame_duration
    return GMRMotion(
        root_pos,
        root_rot,
        joint_pos,
        root_velocity,
        _angular_velocity(root_rot, frame_duration),
        joint_velocity,
        frame_duration,
    )


class MuJoCoForwardKinematics:
    """Compute configured link positions from an AMP URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: Sequence[str],
        anchor_base: str,
        anchor_links: Sequence[str],
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "AMP motion loading requires `pip install mmrl[amp]`."
            ) from exc
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
        self.data = mujoco.MjData(self.model)
        self.joint_addresses = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"URDF has no joint named {name!r}.")
            self.joint_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self.anchor_id = self._body_id(anchor_base)
        self.link_ids = [self._body_id(name) for name in anchor_links]

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"URDF has no body named {name!r}.")
        return body_id

    def link_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        result = np.empty(
            (len(joint_positions), len(self.link_ids), 3), dtype=np.float64
        )
        for frame_index, positions in enumerate(joint_positions):
            self.data.qpos[self.joint_addresses] = positions
            self._mujoco.mj_forward(self.model, self.data)
            anchor_pos = self.data.xpos[self.anchor_id]
            anchor_rot = self.data.xmat[self.anchor_id].reshape(3, 3)
            links = self.data.xpos[self.link_ids]
            result[frame_index] = (links - anchor_pos) @ anchor_rot
        return result


class AMPLoader:
    """Weighted expert transition source built from environment AMP config."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        time_between_frames: float,
        motion_files: Sequence[str | Path],
        motion_weights: Mapping[str, float] | None,
        joint_names: Sequence[str],
        anchor_base: str,
        anchor_links: Sequence[str],
        urdf_path: str | Path,
        preload_transitions: bool = True,
        num_preload_transitions: int = 1_000_000,
        forward_kinematics_factory=MuJoCoForwardKinematics,
    ) -> None:
        self.device = torch.device(device)
        self.time_between_frames = float(time_between_frames)
        self.motion_files = tuple(Path(path) for path in motion_files)
        if not self.motion_files:
            raise ValueError("cfg.amp.motion_files cannot be empty.")
        self.weights = _motion_weights(self.motion_files, motion_weights)
        self.trajectories: list[torch.Tensor] = []
        self.frame_durations: list[float] = []
        self.kinematics = forward_kinematics_factory(
            urdf_path, joint_names, anchor_base, anchor_links
        )
        for path in self.motion_files:
            motion = load_gmr_motion(path, len(joint_names))
            links = self.kinematics.link_positions(motion.joint_pos)
            trajectory = build_amp_frame(
                torch.as_tensor(motion.root_pos[:, 2:3], dtype=torch.float32),
                torch.as_tensor(motion.root_rot, dtype=torch.float32),
                torch.as_tensor(motion.root_linear_vel, dtype=torch.float32),
                torch.as_tensor(motion.root_angular_vel, dtype=torch.float32),
                torch.as_tensor(motion.joint_pos, dtype=torch.float32),
                torch.as_tensor(motion.joint_vel, dtype=torch.float32),
                torch.as_tensor(links, dtype=torch.float32),
            ).to(self.device)
            self.trajectories.append(trajectory)
            self.frame_durations.append(motion.frame_duration)
        self._observation_dim = self.trajectories[0].shape[1]
        if any(item.shape[1] != self._observation_dim for item in self.trajectories):
            raise ValueError("All AMP motions must use the same feature layout.")
        self.preloaded: AMPTransitionBatch | None = None
        if preload_transitions:
            self.preloaded = self._sample_interpolated(num_preload_transitions)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        device: str | torch.device,
        forward_kinematics_factory=MuJoCoForwardKinematics,
    ) -> AMPLoader:
        """Construct from the environment-owned ``cfg.amp`` object."""
        return cls(
            device=device,
            time_between_frames=require_config_value(cfg, "time_between_frames"),
            motion_files=require_config_value(cfg, "motion_files"),
            motion_weights=get_config_value(cfg, "motion_weights", {}),
            joint_names=require_config_value(cfg, "joint_names"),
            anchor_base=require_config_value(cfg, "anchor_base"),
            anchor_links=require_config_value(cfg, "anchor_links"),
            urdf_path=require_config_value(cfg, "urdf_path"),
            preload_transitions=get_config_value(cfg, "preload_transitions", True),
            num_preload_transitions=get_config_value(
                cfg, "num_preload_transitions", 1_000_000
            ),
            forward_kinematics_factory=forward_kinematics_factory,
        )

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    def sample(
        self, batch_size: int, device: str | torch.device
    ) -> AMPTransitionBatch:
        if self.preloaded is None:
            batch = self._sample_interpolated(batch_size)
        else:
            indices = torch.randint(
                0, self.preloaded.state.shape[0], (batch_size,), device=self.device
            )
            batch = AMPTransitionBatch(
                self.preloaded.state[indices], self.preloaded.next_state[indices]
            )
        return AMPTransitionBatch(
            batch.state.to(device), batch.next_state.to(device)
        )

    def _sample_interpolated(self, batch_size: int) -> AMPTransitionBatch:
        trajectory_ids = np.random.choice(
            len(self.trajectories), size=batch_size, p=self.weights
        )
        states = torch.empty(
            batch_size, self.observation_dim, device=self.device
        )
        next_states = torch.empty_like(states)
        for trajectory_id in np.unique(trajectory_ids):
            indices = np.flatnonzero(trajectory_ids == trajectory_id)
            trajectory = self.trajectories[int(trajectory_id)]
            frame_dt = self.frame_durations[int(trajectory_id)]
            max_time = max(
                (len(trajectory) - 1) * frame_dt - self.time_between_frames,
                0.0,
            )
            times = np.random.uniform(0.0, max_time, size=len(indices))
            states[indices] = _interpolate(trajectory, times / frame_dt)
            next_states[indices] = _interpolate(
                trajectory, (times + self.time_between_frames) / frame_dt
            )
        return AMPTransitionBatch(states, next_states)


def _motion_array(payload, key, path, width, frame_count=None):
    if key not in payload:
        raise ValueError(f"GMR motion {path}: missing field {key!r}.")
    value = np.asarray(payload[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != width or len(value) < 2:
        raise ValueError(
            f"GMR motion {path}: invalid shape for {key!r}: {value.shape}."
        )
    if frame_count is not None and len(value) != frame_count:
        raise ValueError(f"GMR motion {path}: inconsistent frame count for {key!r}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"GMR motion {path}: {key!r} contains non-finite values.")
    return value


def _angular_velocity(quaternion, frame_duration):
    velocity = np.zeros((len(quaternion), 3), dtype=np.float64)
    inverse = quaternion[:-1].copy()
    inverse[:, :3] *= -1.0
    relative = _quaternion_multiply(quaternion[1:], inverse)
    relative[relative[:, 3] < 0.0] *= -1.0
    norm = np.linalg.norm(relative[:, :3], axis=1)
    angle = 2.0 * np.arctan2(norm, relative[:, 3])
    valid = norm > np.finfo(np.float64).eps
    velocity[1:][valid] = relative[valid, :3] * (
        angle[valid] / norm[valid] / frame_duration
    )[:, None]
    return velocity


def _quaternion_multiply(left, right):
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _motion_weights(paths, configured):
    configured = configured or {}
    names = [path.name for path in paths]
    unknown = set(configured) - set(names)
    if unknown:
        raise ValueError(f"Unknown AMP motion weight(s): {sorted(unknown)}.")
    weights = np.asarray([float(configured.get(name, 1.0)) for name in names])
    invalid = not np.all(np.isfinite(weights)) or np.any(weights < 0)
    if invalid or weights.sum() <= 0:
        raise ValueError(
            "AMP motion weights must be finite, non-negative, and nonzero."
        )
    return weights / weights.sum()


def _interpolate(trajectory: torch.Tensor, frame_positions) -> torch.Tensor:
    position = torch.as_tensor(
        frame_positions, dtype=torch.float32, device=trajectory.device
    ).clamp(0, len(trajectory) - 1)
    low = position.floor().long()
    high = position.ceil().long()
    blend = (position - low).unsqueeze(-1)
    return (1.0 - blend) * trajectory[low] + blend * trajectory[high]
