import pickle

import numpy as np
import pytest

from mmrl.amp import AMPLoader
from mmrl.amp.motion_loader import MuJoCoForwardKinematics


class _FakeKinematics:
    def __init__(self, urdf_path, joint_names, anchor_base, anchor_links):
        self.link_count = len(anchor_links)

    def link_positions(self, joint_positions):
        return np.zeros(
            (len(joint_positions), self.link_count, 3), dtype=np.float64
        )


def test_amp_loader_builds_and_samples_expert_transitions(tmp_path):
    motion_path = tmp_path / "walk.pkl"
    frames = 5
    with motion_path.open("wb") as stream:
        pickle.dump(
            {
                "fps": 10.0,
                "root_pos": np.arange(frames * 3).reshape(frames, 3),
                "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)),
                "dof_pos": np.arange(frames * 2).reshape(frames, 2),
            },
            stream,
        )

    class AMPCfg:
        dt = 0.1
        amp_motion_files = (motion_path,)
        amp_motion_weights = {"walk.pkl": 2.0}
        joint_names = ("joint_a", "joint_b")
        amp_anchor_base = "base"
        amp_anchor_links = ("foot",)
        urdf_path = tmp_path / "robot.urdf"
        preload_transitions = True
        amp_num_preload_transitions = 16

    loader = AMPLoader.from_config(
        AMPCfg(), "cpu", forward_kinematics_factory=_FakeKinematics
    )
    batch = loader.sample(4, "cpu")

    assert loader.observation_dim == 20
    assert batch.state.shape == batch.next_state.shape == (4, 20)


def test_mujoco_fk_resolves_fixed_amp_links(tmp_path):
    pytest.importorskip("mujoco")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="base"/>
  <link name="leg">
    <inertial>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0"
               iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="foot"/>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="leg"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="foot_fixed" type="fixed">
    <parent link="leg"/><child link="foot"/>
    <origin xyz="0 0 -1"/>
  </joint>
</robot>"""
    )
    kinematics = MuJoCoForwardKinematics(
        urdf, ("hip",), "base", ("foot",)
    )

    positions = kinematics.link_positions(np.zeros((2, 1)))

    assert positions.shape == (2, 1, 3)
    np.testing.assert_allclose(positions[:, 0], [[0, 0, -1], [0, 0, -1]])
