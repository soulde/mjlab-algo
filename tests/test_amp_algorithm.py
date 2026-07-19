import torch

from mmrl.amp import AMP, AMPAlgorithmCfg
from mmrl.memories import OnPolicyRolloutMemory, TensorAMPDataset
from mmrl.models import AMPDiscriminator
from mmrl.ppo import ActorCritic, PPOActorCriticCfg


def test_amp_updates_ppo_and_discriminator():
    algorithm_cfg = AMPAlgorithmCfg(
        num_learning_epochs=1,
        num_mini_batches=1,
        amp_replay_capacity=32,
    )
    policy_cfg = PPOActorCriticCfg(
        actor_hidden_dims=(8,), critic_hidden_dims=(8,)
    )
    policy = ActorCritic(policy_cfg, 4, 2)
    expert = TensorAMPDataset(torch.randn(16, 3), torch.randn(16, 3))
    amp = AMP(
        algorithm_cfg,
        policy,
        AMPDiscriminator(3, (8,)),
        expert,
        amp_observation_dim=3,
        device="cpu",
    )
    memory = OnPolicyRolloutMemory(2, 2, (4,), (2,), "cpu")
    for _ in range(2):
        obs = torch.randn(2, 4)
        action, log_prob, value = amp.act(obs)
        memory.add(
            obs,
            action,
            torch.randn(2),
            torch.zeros(2, dtype=torch.bool),
            log_prob,
            value,
        )
        state = torch.randn(2, 3)
        amp.add_amp_transition(state, torch.randn(2, 3))
    memory.compute_returns(torch.zeros(2, 1))

    metrics = amp.update(memory)

    assert "amp_loss" in metrics
    assert "amp_gradient_penalty" in metrics
    reward, style_reward = amp.combine_rewards(
        torch.randn(2, 3), torch.randn(2, 3), torch.ones(2), 0.2, 0.8
    )
    assert reward.shape == style_reward.shape == (2,)
