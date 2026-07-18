import torch

from mmrl.memories import OnPolicyRolloutMemory
from mmrl.ppo import ActorCritic, PPO, PPOActorCriticCfg, PPOAlgorithmCfg


def test_ppo_updates_policy_from_rollout():
    torch.manual_seed(1)
    policy = ActorCritic(PPOActorCriticCfg(), obs_dim=3, action_dim=2)
    algorithm_cfg = PPOAlgorithmCfg(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
    )
    algorithm = PPO(algorithm_cfg, policy, device="cpu")
    memory = OnPolicyRolloutMemory(4, 2, (3,), (2,), "cpu")

    for _ in range(4):
        obs = torch.randn(2, 3)
        with torch.no_grad():
            action, log_prob, value = algorithm.act(obs)
        memory.add(
            obs,
            action,
            reward=torch.randn(2),
            done=torch.zeros(2, dtype=torch.bool),
            log_prob=log_prob,
            value=value,
        )
    memory.compute_returns(torch.zeros(2), gamma=0.99, gae_lambda=0.95)
    before = policy.actor[0].weight.detach().clone()

    metrics = algorithm.update(memory)

    assert set(metrics) == {
        "value_loss",
        "surrogate_loss",
        "entropy",
        "kl",
        "learning_rate",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(before, policy.actor[0].weight)
