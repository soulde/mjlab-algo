"""Expert motion transition loader adapted from amp_go2-main."""

from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmrl.amp.features import build_amp_frame
from mmrl.config import require_config_value
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
        urdf_path = Path(urdf_path)
        urdf_root = ET.parse(urdf_path).getroot()
        for inertia in urdf_root.findall(".//inertial/inertia"):
            inertia.attrib.update(
                {
                    "ixx": "1",
                    "ixy": "0",
                    "ixz": "0",
                    "iyy": "1",
                    "iyz": "0",
                    "izz": "1",
                }
            )
        for mesh in urdf_root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.attrib["filename"] = str(
                    (urdf_path.parent / filename).resolve()
                )
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(urdf_root, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        self.joint_addresses = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"URDF has no joint named {name!r}.")
            self.joint_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        link_names = {node.attrib["name"] for node in urdf_root.findall("link")}
        child_names = {
            node.attrib["link"] for node in urdf_root.findall("joint/child")
        }
        root_names = link_names - child_names
        if len(root_names) != 1:
            raise ValueError("AMP URDF must contain exactly one root link.")
        self.root_link = next(iter(root_names))
        self.joint_by_child = {}
        for joint in urdf_root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is None or parent is None:
                raise ValueError("AMP URDF joint is missing parent or child.")
            origin = joint.find("origin")
            self.joint_by_child[child.attrib["link"]] = (
                parent.attrib["link"],
                joint.attrib["type"],
                _origin_matrix(origin),
            )
        self.anchor = self._resolve_link(anchor_base)
        self.links = [self._resolve_link(name) for name in anchor_links]

    def _resolve_link(self, name: str) -> tuple[int, np.ndarray]:
        body_name = "world" if name == self.root_link else name
        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id >= 0:
            return body_id, np.eye(4)
        if name not in self.joint_by_child:
            raise ValueError(f"URDF has no link named {name!r}.")
        parent, joint_type, parent_to_link = self.joint_by_child[name]
        if joint_type != "fixed":
            raise ValueError(f"Movable URDF link {name!r} is absent in MuJoCo.")
        parent_id, body_to_parent = self._resolve_link(parent)
        return parent_id, body_to_parent @ parent_to_link

    def _world_matrix(self, resolver: tuple[int, np.ndarray]) -> np.ndarray:
        body_id, body_to_link = resolver
        body_to_world = np.eye(4)
        body_to_world[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        body_to_world[:3, 3] = self.data.xpos[body_id]
        return body_to_world @ body_to_link

    def link_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        result = np.empty(
            (len(joint_positions), len(self.links), 3), dtype=np.float64
        )
        for frame_index, positions in enumerate(joint_positions):
            self.data.qpos[self.joint_addresses] = positions
            self._mujoco.mj_forward(self.model, self.data)
            world_to_anchor = np.linalg.inv(self._world_matrix(self.anchor))
            for link_index, resolver in enumerate(self.links):
                link_to_world = self._world_matrix(resolver)
                result[frame_index, link_index] = (
                    world_to_anchor @ link_to_world
                )[:3, 3]
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
            time_between_frames=require_config_value(cfg, "dt"),
            motion_files=require_config_value(cfg, "amp_motion_files"),
            motion_weights=require_config_value(cfg, "amp_motion_weights"),
            joint_names=require_config_value(cfg, "joint_names"),
            anchor_base=require_config_value(cfg, "amp_anchor_base"),
            anchor_links=require_config_value(cfg, "amp_anchor_links"),
            urdf_path=require_config_value(cfg, "urdf_path"),
            preload_transitions=require_config_value(
                cfg, "preload_transitions"
            ),
            num_preload_transitions=require_config_value(
                cfg, "amp_num_preload_transitions"
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


def _origin_matrix(origin) -> np.ndarray:
    if origin is None:
        return np.eye(4)
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(
        origin.attrib.get("rpy", "0 0 0"), sep=" "
    )
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    matrix[:3, 3] = xyz
    return matrix


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
