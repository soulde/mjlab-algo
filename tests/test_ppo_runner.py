import torch

from mmrl.env_wrappers import EnvWrapper
from mmrl.ppo import PPOAlgorithmCfg, PPOMemoryCfg, PPORunner, PPORunnerCfg


class _PPOEnv(EnvWrapper):
    num_envs = 2
    obs_dim = 3
    action_dim = 2
    device = torch.device("cpu")
    unwrapped = None

    def __init__(self):
        self.steps = 0

    def rand_act(self):
        return torch.zeros(self.num_envs, self.action_dim)

    def reset(self):
        return torch.zeros(self.num_envs, self.obs_dim)

    def step(self, action):
        self.steps += 1
        obs = torch.full((self.num_envs, self.obs_dim), float(self.steps))
        reward = torch.ones(self.num_envs)
        done = torch.tensor([False, self.steps % 2 == 0])
        return obs, reward, done, {}

    def close(self):
        pass


class _AsymmetricPPOEnv(_PPOEnv):
    critic_obs_dim = 5

    def __init__(self):
        super().__init__()
        self.critic_obs = torch.zeros(self.num_envs, self.critic_obs_dim)

    def reset(self):
        self.critic_obs.zero_()
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.critic_obs.fill_(float(self.steps))
        return obs, reward, done, info

    def get_critic_observations(self):
        return self.critic_obs


def _runner_cfg() -> PPORunnerCfg:
    return PPORunnerCfg(
        device="cpu",
        max_iterations=1,
        save_interval=0,
        log_interval=0,
        algorithm=PPOAlgorithmCfg(
            num_learning_epochs=1,
            num_mini_batches=2,
            learning_rate=1e-3,
        ),
        memory=PPOMemoryCfg(num_steps_per_env=4),
    )


def test_ppo_runner_learns_saves_and_loads(tmp_path):
    runner = PPORunner(_PPOEnv(), _runner_cfg(), tmp_path)

    runner.learn()
    checkpoint = tmp_path / "models" / "final.pt"
    policy = runner.get_inference_policy()

    assert runner.current_iteration == 1
    assert checkpoint.exists()
    assert policy(torch.zeros(2, 3)).shape == (2, 2)

    restored = PPORunner(_PPOEnv(), _runner_cfg(), tmp_path / "restored")
    restored.load(checkpoint)
    assert restored.current_iteration == 1


def test_ppo_runner_uses_asymmetric_critic_observations(tmp_path):
    runner = PPORunner(_AsymmetricPPOEnv(), _runner_cfg(), tmp_path)

    runner.learn()

    assert runner.policy.critic.net[0].in_features == 5
    assert runner.memory.storage["critic_obs"].shape[-1] == 5


def test_ppo_runner_rejects_external_components(tmp_path):
    cfg = _runner_cfg()
    cfg.algorithm.class_name = "environment.PPO"

    try:
        PPORunner(_PPOEnv(), cfg, tmp_path)
    except ValueError as error:
        assert "Unsupported algorithm.class_name" in str(error)
    else:
        raise AssertionError("Runner accepted an external PPO implementation")


def test_ppo_runner_validates_minibatch_divisibility(tmp_path):
    cfg = _runner_cfg()
    cfg.memory.num_steps_per_env = 3
    cfg.algorithm.num_mini_batches = 4

    try:
        PPORunner(_PPOEnv(), cfg, tmp_path)
    except ValueError as error:
        assert "Rollout size 6" in str(error)
    else:
        raise AssertionError("Runner accepted an uneven mini-batch split")
