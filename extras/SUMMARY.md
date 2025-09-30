# Rollout Recording & Replay System - Implementation Summary

## ✅ What Was Created

### Core Files

1. **`rollout_recorder.py`** (4.6KB)
   - `RolloutRecorder` class for recording rollouts
   - `make_policy()` helper to load policies from checkpoints
   - Automatically captures states, actions, rewards, and base poses
   - Extracts reward terms from environment's `log_dict`

2. **`save_video.py`** (1.7KB)
   - `save_video()` function using imageio.v2
   - `collect_frames_from_env()` helper for capturing frames during rollout
   - Handles frame normalization and encoding

3. **`viser_replay.py`** (7.8KB) ⭐
   - Interactive 3D replay viewer using Viser
   - Timeline scrubbing with play/pause
   - Camera presets (Top, Side, Front, Follow)
   - Live reward term visualization
   - Loop mode and speed control

### Scripts

4. **`record_rollout.py`** (3.6KB)
   - Standalone script for easy rollout recording
   - Loads checkpoint and config automatically
   - Usage: `python extras/record_rollout.py checkpoint=path/to/last.ckpt`

5. **`record_and_replay_example.py`** (3.5KB)
   - Full example showing all features together
   - Demonstrates policy loading, recording, and video saving

### Documentation

6. **`README.md`** (4.4KB)
   - Comprehensive documentation
   - Feature overview, API reference, integration guide

7. **`USAGE.md`** (2.6KB)
   - Quick-start guide
   - Common commands and troubleshooting

8. **`__init__.py`** (289B)
   - Makes extras a proper Python package
   - Exports main classes and functions

## 🚀 Quick Start

### 1. Record a Rollout

```bash
python extras/record_rollout.py checkpoint=outputs/smpl_COMPLETE/last.ckpt
```

Output: `results/rollouts/rollout_TIMESTAMP.npz`

### 2. Replay with Viser

```bash
python extras/viser_replay.py --npz results/rollouts/rollout_20250930_123456.npz
```

Open browser to: `http://localhost:8080`

### 3. Programmatic Usage

```python
from extras import RolloutRecorder, make_policy

# Load and record
policy = make_policy("checkpoint.ckpt")
recorder = RolloutRecorder(env, max_steps=2000)
rollout_file = recorder.run(policy)
```

## 📦 Data Format

Each `.npz` file contains:

```python
{
    'obs': ndarray,          # [T, N, obs_dim] - Observations
    'act': ndarray,          # [T, N, act_dim] - Actions  
    'rew': ndarray,          # [T, N] - Rewards
    'info': ndarray,         # [T] - Info dicts (includes reward_terms)
    'base_pos': ndarray,     # [T, N, 3] - Base positions
    'base_quat': ndarray,    # [T, N, 4] - Base quaternions
}
```

## 🎮 Viser Controls

**Replay Control:**
- Play/Pause button
- Frame slider (timeline scrubbing)
- Speed control (0.1x - 4x)
- Loop toggle

**Camera Presets:**
- Top View - Bird's eye view
- Side View - Profile view
- Front View - Front-facing view
- Follow Robot - Tracks robot base

**Reward Visualization:**
- Total reward display
- Individual term sliders
- Auto-updates per frame

## 🔧 Integration Notes

### Reward Terms
The recorder automatically extracts reward terms from `env.extras["to_log"]`. 

ProtoMotions environments already populate this correctly:
```python
# In compute_reward()
for rew_name, rew in rew_dict.items():
    self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
    
# Base env adds this to extras["to_log"]
```

### Video Recording
If your environment supports `render(mode="rgb_array")`:
```python
from extras import collect_frames_from_env, save_video

frames = collect_frames_from_env(env, policy, max_steps=2000)
save_video(frames, "video.mp4", fps=30)
```

### Keyboard Toggle (Future Enhancement)
To add policy on/off toggle:
```python
# In your viewer loop
toggle = {"policy_on": True}

def on_key(key):
    if key.lower() == "l":
        toggle["policy_on"] = not toggle["policy_on"]

# During step
if toggle["policy_on"]:
    action = policy(obs)
else:
    action = torch.zeros_like(action)
```

## 📁 File Tree

```
extras/
├── __init__.py                      # Package exports
├── rollout_recorder.py              # Core recorder class
├── save_video.py                    # Video utilities
├── viser_replay.py                  # Interactive viewer ⭐
├── record_rollout.py                # Standalone recording script
├── record_and_replay_example.py    # Full example
├── README.md                        # Full documentation
├── USAGE.md                         # Quick reference
└── SUMMARY.md                       # This file
```

## ✨ Key Features Implemented

✅ Rollout recording with full state capture  
✅ Automatic reward term extraction  
✅ Interactive Viser replay viewer  
✅ Timeline scrubbing and playback controls  
✅ Multiple camera presets  
✅ Live reward visualization  
✅ Video export support  
✅ Multi-environment handling  
✅ Standalone scripts for easy use  
✅ Comprehensive documentation  

## 🎯 Next Steps

1. **Record your first rollout:**
   ```bash
   python extras/record_rollout.py checkpoint=outputs/smpl_COMPLETE/last.ckpt
   ```

2. **Replay it:**
   ```bash
   python extras/viser_replay.py --npz results/rollouts/rollout_*.npz
   ```

3. **Customize:**
   - Add robot meshes to Viser visualization
   - Implement keyboard controls for policy toggle
   - Add trajectory visualization
   - Export analysis plots from reward terms

## 📚 References

- Viser docs: https://viser.studio/latest/
- ImageIO docs: https://imageio.readthedocs.io/
- ProtoMotions environments: `protomotions/envs/`

---

**Status**: ✅ Complete and ready to use! 