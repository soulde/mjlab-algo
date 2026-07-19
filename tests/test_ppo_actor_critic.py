import torch

from mmrl.ppo import ActorCritic, PPOActorCriticCfg, PPORunnerCfg


def test_ppo_config_is_layered():
    cfg = PPORunnerCfg()

    assert cfg.actor_critic.class_name == "ActorCritic"
    assert cfg.algorithm.class_name == "PPO"
    assert cfg.memory.class_name == "OnPolicyRolloutMemory"


def test_actor_critic_shapes_and_gradients():
    model = ActorCritic(PPOActorCriticCfg(), obs_dim=5, action_dim=2)
    obs = torch.randn(4, 5)

    action, log_prob, value = model.act(obs)
    new_log_prob, entropy, new_value = model.evaluate_actions(obs, action)
    loss = -(new_log_prob + 0.01 * entropy).mean() + new_value.mean()
    loss.backward()

    assert action.shape == (4, 2)
    assert log_prob.shape == (4, 1)
    assert value.shape == (4, 1)
    assert model.log_std.grad is not None


def test_actor_critic_supports_asymmetric_observations():
    model = ActorCritic(
        PPOActorCriticCfg(), obs_dim=3, action_dim=2, critic_obs_dim=5
    )
    actor_obs = torch.randn(4, 3)
    critic_obs = torch.randn(4, 5)

    action, log_prob, value = model.act(actor_obs, critic_obs)
    _, _, evaluated_value = model.evaluate_actions(
        actor_obs, action, critic_obs
    )

    assert action.shape == (4, 2)
    assert log_prob.shape == value.shape == evaluated_value.shape == (4, 1)
