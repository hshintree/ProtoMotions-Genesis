#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Prefer CPU on macOS to avoid GPU-related segfaults
os.environ.setdefault("PROTOMOTIONS_GENESIS_FORCE_CPU", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extras.rollout_recorder import RolloutRecorder
from extras.save_video import save_video, collect_frames_from_env

# This is a minimal example; for typical use, run:
#   python extras/record_rollout.py checkpoint=path/to/last.ckpt


def main():
    # Import lazily to avoid heavy imports at module import time
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from lightning.fabric import Fabric

    CHECKPOINT_PATH = "results/smpl_COMPLETE/last.ckpt"
    OUTPUT_DIR = "results/rollouts"
    MAX_STEPS = 500

    checkpoint = Path(CHECKPOINT_PATH)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"

    config = OmegaConf.load(config_path)
    config.num_envs = 1
    config.headless = False

    # Force CPU fabric for safety
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()

    env = instantiate(config.env, device=fabric.device)

    from protomotions.agents.ppo.agent import PPO
    agent = PPO.load_from_checkpoint(
        str(CHECKPOINT_PATH), env=env, fabric=fabric, map_location=fabric.device
    )
    agent.eval()

    def policy(obs_dict, deterministic=True):
        with torch.no_grad():
            return agent.model.act(obs_dict)

    recorder = RolloutRecorder(env, max_steps=MAX_STEPS)
    rollout_file = recorder.run(policy, deterministic=True, output_dir=OUTPUT_DIR)

    print(f"Saved rollout: {rollout_file}")

    # Optionally collect frames and save a video
    # frames = collect_frames_from_env(env, policy, max_steps=MAX_STEPS)
    # save_video(frames, "results/videos/rollout.mp4", fps=30)


if __name__ == "__main__":
    main() 