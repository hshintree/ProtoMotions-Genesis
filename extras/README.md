# Rollout Recording and Replay Tools

This directory contains utilities for recording, saving, and replaying agent rollouts with rich visualization.

## Features

- **Rollout Recording**: Save complete rollouts including states, actions, rewards, and metadata
- **Video Export**: Record videos of agent behavior
- **Viser Replay**: Interactive 3D replay with timeline scrubbing, camera controls, and live reward visualization

## Quick Start

### 1. Record a Rollout

Use the integrated rollout recorder within your evaluation script:

```python
from extras.rollout_recorder import RolloutRecorder, make_policy

# Create environment and policy
env = ...  # Your environment
policy = make_policy("path/to/checkpoint.ckpt")

# Record rollout
recorder = RolloutRecorder(env, max_steps=2000)
rollout_file = recorder.run(policy, output_dir="results/rollouts")
```

### 2. Replay with Viser

Launch the interactive Viser viewer:

```bash
python extras/viser_replay.py --npz results/rollouts/rollout_20250930_123456.npz
```

Then open your browser to `http://localhost:8080`

### 3. Save Videos (Optional)

```python
from extras.save_video import save_video, collect_frames_from_env

# Collect frames during rollout
frames = collect_frames_from_env(env, policy, max_steps=2000)

# Save as video
save_video(frames, "results/videos/rollout.mp4", fps=30)
```

## Viser Viewer Controls

The Viser viewer provides:

- **Replay Control**
  - Play/Pause button
  - Frame slider for timeline scrubbing
  - Speed control (0.1x to 4x)
  - Loop toggle

- **Camera Presets**
  - Top View: Bird's eye view
  - Side View: Profile view
  - Front View: Front-facing view
  - Follow Robot: Camera tracks the robot base

- **Reward Terms**
  - Real-time display of all reward components
  - Total reward counter
  - Individual term values updated per frame

## File Structure

```
extras/
├── rollout_recorder.py      # Main rollout recording class
├── save_video.py             # Video saving utilities
├── viser_replay.py           # Interactive Viser viewer
├── record_and_replay_example.py  # Full example script
└── README.md                 # This file
```

## Rollout Data Format

Rollouts are saved as `.npz` files with the following structure:

```python
{
    'obs': ndarray,          # Observations [T, N, obs_dim]
    'act': ndarray,          # Actions [T, N, act_dim]
    'rew': ndarray,          # Rewards [T, N]
    'info': ndarray,         # Info dicts (object array) [T]
    'base_pos': ndarray,     # Base positions [T, N, 3]
    'base_quat': ndarray,    # Base quaternions [T, N, 4]
}
```

Where `T` is the number of timesteps and `N` is the number of parallel environments.

## Integrating with Your Code

### Adding Reward Terms to Environment

Ensure your environment exposes reward terms in the info dict:

```python
# In your environment's compute_reward() method
def compute_reward(self):
    # ... compute individual reward components ...
    
    reward_terms = {
        'pose_rew': pose_reward,
        'vel_rew': velocity_reward,
        'energy_rew': energy_reward,
        # ... other terms ...
    }
    
    # Add to log dict (already done in ProtoMotions environments)
    for name, val in reward_terms.items():
        self.log_dict[f"raw/{name}_mean"] = val.mean()
    
    # The rollout recorder will automatically extract these
```

### Keyboard Controls (Future Enhancement)

To add keyboard toggle for policy on/off:

```python
# In your viewer/simulator loop
def on_key_press(key):
    if key.lower() == 'l':
        policy_enabled = not policy_enabled
        
if policy_enabled:
    action = policy(obs)
else:
    action = torch.zeros_like(action)  # or use PD controller
```

## Dependencies

- `numpy`
- `torch`
- `viser` (for replay viewer)
- `imageio` (for video saving)

Install with:
```bash
pip install viser imageio
```

## Tips

1. **Multiple Environments**: The recorder handles both single and multi-environment setups. For visualization, the first environment is used.

2. **Large Rollouts**: For very long rollouts (>5000 steps), consider splitting into chunks or reducing the save frequency.

3. **Custom Visualizations**: Extend `viser_replay.py` to add robot meshes, trajectories, or other 3D elements.

4. **Offscreen Rendering**: If your simulator supports offscreen rendering, enable it for video capture without displaying a window.

## Examples

See `record_and_replay_example.py` for a complete example integrating all features. 