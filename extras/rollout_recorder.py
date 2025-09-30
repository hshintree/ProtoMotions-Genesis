import time
import numpy as np
import torch
import os
from pathlib import Path


class RolloutRecorder:
    def __init__(self, env, max_steps=2000):
        self.env = env
        self.max_steps = max_steps
        # Link names in common ordering
        try:
            self.link_names = list(self.env.simulator.robot_config.body_names)
        except Exception:
            self.link_names = None

    def run(self, policy, deterministic=True, output_dir="results/rollouts"):
        """Run a rollout with the given policy and record all data.
        
        Args:
            policy: Callable that takes obs and returns actions
            deterministic: Whether to use deterministic actions
            output_dir: Directory to save rollout data
            
        Returns:
            Path to saved rollout file
        """
        # Reset environment (note: different from gym-style reset)
        obs_dict = self.env.get_obs()
        
        # Initialize buffers
        buf = {
            "obs": [],
            "act": [],
            "rew": [],
            "info": [],
            "base_pos": [],
            "base_quat": [],
            "link_names": self.link_names,
            "link_pos": [],
            "link_quat": [],
        }

        for step in range(self.max_steps):
            # Get action from policy
            with torch.no_grad():
                action = policy(obs_dict, deterministic=deterministic)
            
            # Step environment
            obs_next, reward, done, info = self.env.step(action)

            # Get robot state for base pose (use cross-backend API)
            try:
                root_state = self.env.simulator.get_root_state()
            except Exception:
                root_state = None

            # Get per-link world poses if available
            link_state = None
            try:
                link_state = self.env.simulator.get_bodies_state()
            except Exception:
                link_state = None

            # Collect data - handle both single env and multi-env cases
            if isinstance(obs_dict, dict):
                # Stack all obs tensors into a single array
                obs_array = torch.cat([
                    v.flatten(start_dim=1) if v.dim() > 1 else v.unsqueeze(1)
                    for v in obs_dict.values()
                ], dim=1)
                buf["obs"].append(obs_array.cpu().numpy())
            else:
                buf["obs"].append(obs_dict.cpu().numpy())
                
            buf["act"].append(action.cpu().numpy())
            buf["rew"].append(reward.cpu().numpy())
            
            # Extract reward terms from info if available
            info_to_save = {}
            if "to_log" in info:
                log_dict = info["to_log"]
                # Extract reward terms
                reward_terms = {}
                for key, val in log_dict.items():
                    if "rew" in key or "reward" in key:
                        if hasattr(val, "cpu"):
                            reward_terms[key] = val.cpu().numpy()
                        else:
                            reward_terms[key] = val
                if reward_terms:
                    info_to_save["reward_terms"] = reward_terms
            
            # Add any other info fields
            for k, v in info.items():
                if k not in ["to_log", "terminate"]:
                    if hasattr(v, "cpu"):
                        info_to_save[k] = v.cpu().numpy()
                    else:
                        info_to_save[k] = v
                        
            buf["info"].append(info_to_save)

            # Record base pose for visualization
            if root_state is not None and root_state.root_pos is not None:
                buf["base_pos"].append(root_state.root_pos.cpu().numpy())
            if root_state is not None and root_state.root_rot is not None:
                buf["base_quat"].append(root_state.root_rot.cpu().numpy())

            # Record per-link world poses
            if link_state is not None and link_state.rigid_body_pos is not None:
                buf["link_pos"].append(link_state.rigid_body_pos.cpu().numpy())
            if link_state is not None and link_state.rigid_body_rot is not None:
                buf["link_quat"].append(link_state.rigid_body_rot.cpu().numpy())

            obs_dict = obs_next
            
            # Check for termination (any env done)
            if torch.any(done):
                print(f"Episode terminated at step {step}")
                break

        # Save to file
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        out = f"{output_dir}/rollout_{ts}.npz"
        
        # Convert to numpy arrays, handling info specially
        save_dict = {}
        for k, v in buf.items():
            if k == "info":
                save_dict[k] = np.array(v, dtype=object)
            else:
                save_dict[k] = np.array(v)
        
        np.savez_compressed(out, **save_dict)
        print(f"Rollout saved to {out}")
        return out


def make_policy(ckpt_path, device="cuda"):
    """Create a policy function from a checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Callable policy function
    """
    import torch
    from protomotions.agents.ppo.agent import PPO
    
    # Load the agent from checkpoint
    agent = PPO.load_from_checkpoint(ckpt_path, map_location=device)
    agent.eval()
    
    def act(obs_dict, deterministic=True):
        with torch.no_grad():
            # Get action from the model
            actions = agent.model.act(obs_dict)
        return actions
    
    return act 