"""Compatibility imports for FastSAC replay memory."""

from mmrl.memories.off_policy import OffPolicyBatch, OffPolicyReplayMemory

FastSACBatch = OffPolicyBatch
FastSACReplayBuffer = OffPolicyReplayMemory

