import pytest

from mmrl.logging import LoggerCfg, MetricLogger, format_training_log


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


def test_metric_logger_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="Unsupported logging backend"):
        MetricLogger(tmp_path, LoggerCfg(backends=("unknown",)))


def test_metric_logger_writes_tensorboard_events(tmp_path):
    pytest.importorskip("tensorboard")
    logger = MetricLogger(tmp_path, {"backends": ("tensorboard",)})

    logger.log({"loss": 1.25}, step=4, prefix="train")
    logger.close()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert event_files
    assert event_files[0].stat().st_size > 0
