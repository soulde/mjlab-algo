from mjlab_algo.logging import format_training_log


def test_format_training_log_includes_ppo_style_debug_fields():
    text = format_training_log(
        title="FastSAC iteration 1/10",
        total_steps=1024,
        steps_per_second=512.0,
        collection_time=0.25,
        learning_time=0.75,
        losses={"actor": 1.25, "critic": 2.5},
        mean_reward=12.0,
        mean_episode_length=48.0,
        extras={"Alpha": 0.2, "Replay buffer": 1024},
        iteration_time=1.0,
        elapsed_time=5.0,
        eta_seconds=45.0,
        log_dir="logs/test",
    )

    assert "FastSAC iteration 1/10" in text
    assert "Total steps:" in text
    assert "Steps per second:" in text
    assert "Collection time:" in text
    assert "Learning time:" in text
    assert "Mean actor loss:" in text
    assert "Mean critic loss:" in text
    assert "Mean reward:" in text
    assert "Mean episode length:" in text
    assert "Replay buffer:" in text
    assert "Log directory:" in text
    assert "ETA:" in text
