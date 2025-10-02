import time
import numpy as np
import torch
import os
from pathlib import Path

class RolloutRecorder:
    def __init__(self, env, max_steps=2000,
                 ignore_done=False, auto_reset_on_done=True, stop_on_done=False):
        self.env = env
        self.max_steps = max_steps
        self.ignore_done = ignore_done
        self.auto_reset_on_done = auto_reset_on_done
        self.stop_on_done = stop_on_done

        # Try to get link/body names from the same source we’ll read poses from
        self.link_names = None
        try:
            if hasattr(self.env.simulator, "get_bodies_state"):
                state = self.env.simulator.get_bodies_state()
                if hasattr(state, "names") and state.names is not None:
                    self.link_names = list(state.names)
        except Exception:
            pass
        if self.link_names is None:
            try:
                self.link_names = list(self.env.simulator.robot_config.body_names)
            except Exception:
                self.link_names = None

    def _squeeze1(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach()
            # Many sim APIs return [num_env, N, ...]; for num_env=1, strip the first dim
            if x.dim() >= 1 and x.size(0) == 1:
                x = x[0]
            return x.cpu().numpy()
        # numpy path
        x = np.asarray(x)
        if x.ndim >= 1 and x.shape[0] == 1:
            x = x[0]
        return x

    def _pack_obs(self, obs_dict):
        # envs in ProtoMotions return a dict of tensors
        if isinstance(obs_dict, dict):
            flat = []
            for v in obs_dict.values():
                if isinstance(v, torch.Tensor):
                    v = v.detach()
                    flat.append(v.flatten(start_dim=1) if v.dim() > 1 else v.unsqueeze(1))
            if flat:
                x = torch.cat(flat, dim=1)
                return x.cpu().numpy()
        # fallback
        if hasattr(obs_dict, "cpu"):
            return obs_dict.cpu().numpy()
        return np.asarray(obs_dict)

    def run(self, policy, deterministic=True, output_dir="results/rollouts"):
        # Always reset once to get a clean episode
        try:
            obs_dict, _ = self.env.reset()
        except Exception:
            obs_dict = self.env.reset()

        buf = {
            "obs": [], "act": [], "rew": [], "info": [],
            "base_pos": [], "base_quat": [],
            "link_names": self.link_names,
            "link_pos": [], "link_quat": [],
            "done": [], "truncated": [],
        }

        steps = 0
        while steps < self.max_steps:
            with torch.no_grad():
                action = policy(obs_dict, deterministic=deterministic)

            # ProtoMotions base env typically: (obs, rew, done, extras)
            out = self.env.step(action)
            if len(out) == 4:
                obs_next, reward, done, extras = out
                truncated = extras.get("truncated", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            elif len(out) == 5:
                # Some envs: (obs, rew, done, truncated, extras)
                obs_next, reward, done, truncated, extras = out
            else:
                raise RuntimeError("Unexpected env.step() return format")

            # Save observations/actions/rewards
            buf["obs"].append(self._pack_obs(obs_dict))
            buf["act"].append(action.detach().cpu().numpy())
            buf["rew"].append(self._squeeze1(reward))

            # Save termination flags
            buf["done"].append(self._squeeze1(done))
            buf["truncated"].append(self._squeeze1(truncated))

            # Save reward terms & misc info if present
            info_to_save = {}
            if isinstance(extras, dict):
                # reward terms
                if "to_log" in extras and isinstance(extras["to_log"], dict):
                    reward_terms = {}
                    for k, v in extras["to_log"].items():
                        if "rew" in k or "reward" in k:
                            if hasattr(v, "detach"):
                                v = v.detach()
                            reward_terms[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
                    if reward_terms:
                        info_to_save["reward_terms"] = reward_terms
                # termination reasons (common patterns)
                for k in ("terminate", "terminate_reason", "termination_reason"):
                    if k in extras:
                        val = extras[k]
                        if hasattr(val, "cpu"):
                            val = val.detach().cpu().numpy()
                        info_to_save[k] = val
            buf["info"].append(info_to_save)

            # Base + link world poses (squeezed to single-env)
            root_state = None
            try:
                root_state = self.env.simulator.get_root_state()
            except Exception:
                pass
            if root_state is not None:
                if hasattr(root_state, "root_pos") and root_state.root_pos is not None:
                    bp = self._squeeze1(root_state.root_pos)
                    if bp is not None: buf["base_pos"].append(bp)
                if hasattr(root_state, "root_rot") and root_state.root_rot is not None:
                    bq = self._squeeze1(root_state.root_rot)
                    if bq is not None: buf["base_quat"].append(bq)

            link_state = None
            try:
                link_state = self.env.simulator.get_bodies_state()
            except Exception:
                pass
            if link_state is not None:
                if hasattr(link_state, "rigid_body_pos") and link_state.rigid_body_pos is not None:
                    lp = self._squeeze1(link_state.rigid_body_pos)
                    if lp is not None: buf["link_pos"].append(lp)
                if hasattr(link_state, "rigid_body_rot") and link_state.rigid_body_rot is not None:
                    lq = self._squeeze1(link_state.rigid_body_rot)
                    if lq is not None: buf["link_quat"].append(lq)

            # Decide how to handle done
            # These are tensors scalars for num_envs=1 after squeeze
            is_done = bool(torch.any(done).item()) if isinstance(done, torch.Tensor) else bool(np.any(done))
            is_trunc = bool(torch.any(truncated).item()) if isinstance(truncated, torch.Tensor) else bool(np.any(truncated))
            episode_end = is_done or is_trunc

            steps += 1
            obs_dict = obs_next

            if episode_end:
                print(f"[recorder] episode ended at step {steps} (done={is_done}, truncated={is_trunc})")
                if self.stop_on_done:
                    break
                if self.auto_reset_on_done:
                    try:
                        obs_dict, _ = self.env.reset()
                    except Exception:
                        obs_dict = self.env.reset()
                    # continue recording new episode until max_steps
                    continue
                # ignore_done: keep stepping without reset

        # Save
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        out = f"{output_dir}/rollout_{ts}.npz"

        # Numpy-pack (keep 'info' as object)
        save = {}
        for k, v in buf.items():
            if k == "info":
                save[k] = np.array(v, dtype=object)
            else:
                try:
                    save[k] = np.array(v)
                except Exception:
                    save[k] = np.array(v, dtype=object)

        np.savez_compressed(out, **save)
        print(f"Rollout saved to {out}")
        return out 