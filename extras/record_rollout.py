#!/usr/bin/env python3
"""Simple script to record a rollout from a trained agent.

Usage:
    python extras/record_rollout.py checkpoint=path/to/last.ckpt \
        auto_reset_on_done=true ignore_done=false stop_on_done=false \
        ++env.config.max_episode_length=10000 ++env.config.enable_height_termination=false
"""

import os
import sys
from pathlib import Path

# Safe env settings to avoid OpenMP/MPS issues on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
if sys.platform == "darwin":
    os.environ.setdefault("PROTOMOTIONS_GENESIS_FORCE_CPU", "1")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

# Register OmegaConf resolvers used by configs
try:
    from protomotions.utils import config_utils  # noqa: F401
except Exception:
    # Fallback minimal resolvers if import fails
    import math
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("len", lambda x: len(x))

# Handle simulator imports
has_robot_arg = False
simulator = None
for arg in sys.argv:
    if "robot" in arg:
        has_robot_arg = True
    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401
            simulator = "isaacgym"
        elif "isaaclab" in arg.split("=")[-1]:
            from isaaclab.app import AppLauncher
            simulator = "isaaclab"
        elif "genesis" in arg.split("=")[-1]:
            simulator = "genesis"

from lightning.fabric import Fabric
from extras.rollout_recorder import RolloutRecorder


@hydra.main(config_path="../protomotions/config", config_name="base", version_base="1.1")
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is None:
        raise ValueError("Must provide checkpoint=path/to/checkpoint.ckpt")

    # Load config from checkpoint directory
    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            raise ValueError(f"Could not find config at {config_path}")

    print(f"Loading config from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Start from the training config to preserve all required keys
    config = train_config
    
    # Ensure checkpoint path is set on config for agent.load
    config.checkpoint = str(checkpoint)
    
    # Set eval mode settings
    config.num_envs = 1  # Record single environment
    config.headless = False  # Show visualization

    # ---- NEW: recorder behavior flags (default: auto-reset) ----
    # You can override at CLI: ignore_done=true or stop_on_done=true
    if "ignore_done" not in override_config:
        override_config.ignore_done = False
    if "stop_on_done" not in override_config:
        override_config.stop_on_done = False
    if "auto_reset_on_done" not in override_config:
        override_config.auto_reset_on_done = not (override_config.ignore_done or override_config.stop_on_done)
    # Deterministic policy flag (read with fallback)
    deterministic_flag = bool(getattr(override_config, "deterministic", True))

    # Apply nested env.config overrides if provided on CLI
    try:
        if hasattr(override_config, "env") and hasattr(override_config.env, "config"):
            for k in override_config.env.config.keys():
                try:
                    v = override_config.env.config[k]
                    # Create nested containers if missing
                    if not hasattr(config, "env"):
                        config.env = OmegaConf.create({})
                    if not hasattr(config.env, "config"):
                        config.env.config = OmegaConf.create({})
                    config.env.config[k] = v
                except Exception:
                    pass
    except Exception:
        pass

    # Force CPU Fabric to avoid macOS MPS-related segfaults
    try:
        if "fabric" in config and hasattr(config, "fabric"):
            config.fabric.accelerator = "cpu"
            config.fabric.devices = 1
            # Use auto strategy for single-process CPU eval
            if hasattr(config.fabric, "strategy"):
                config.fabric.strategy = "auto"
    except Exception:
        # Best-effort override; continue with defaults if structure differs
        pass

    # Create fabric and environment
    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if simulator == "isaaclab":
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher({"headless": config.headless})
        simulation_app = app_launcher.app
        env = instantiate(
            config.env, device=fabric.device, simulation_app=simulation_app
        )
    else:
        env = instantiate(config.env, device=fabric.device)

    # Load agent
    from protomotions.agents.ppo.agent import PPO
    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.load(config.checkpoint)
    agent.eval()

    # Create policy function
    def policy(obs_dict, deterministic=True):
        with torch.no_grad():
            return agent.model.act(obs_dict, mean=deterministic)

    # Record rollout
    max_steps = getattr(config, "max_eval_steps", 2000)
    output_dir = getattr(config, "rollout_output_dir", "results/rollouts")
    
    print(f"\n{'='*60}")
    print(f"Recording rollout (max {max_steps} steps)...")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    recorder = RolloutRecorder(
        env,
        max_steps=max_steps,
        ignore_done=bool(override_config.ignore_done),
        auto_reset_on_done=bool(override_config.auto_reset_on_done),
        stop_on_done=bool(override_config.stop_on_done),
    )
    rollout_file = recorder.run(policy, deterministic=deterministic_flag, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Rollout saved to: {rollout_file}")
    print(f"\nTo replay with Viser:")
    print(f"    python extras/viser_replay.py --npz {rollout_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main() 