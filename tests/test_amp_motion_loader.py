import pickle

import numpy as np

from mmrl.amp import AMPLoader


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
        time_between_frames = 0.1
        motion_files = (motion_path,)
        motion_weights = {"walk.pkl": 2.0}
        joint_names = ("joint_a", "joint_b")
        anchor_base = "base"
        anchor_links = ("foot",)
        urdf_path = tmp_path / "robot.urdf"
        preload_transitions = True
        num_preload_transitions = 16

    loader = AMPLoader.from_config(
        AMPCfg(), "cpu", forward_kinematics_factory=_FakeKinematics
    )
    batch = loader.sample(4, "cpu")

    assert loader.observation_dim == 20
    assert batch.state.shape == batch.next_state.shape == (4, 20)
