import imageio.v2 as imageio
import numpy as np


def save_video(frames, path, fps=30):
    """Save a list of frames as a video file.
    
    Args:
        frames: List of numpy arrays (H, W, 3) representing RGB frames
        path: Output video file path (e.g., "video.mp4")
        fps: Frames per second
    """
    # Ensure frames are uint8
    frames_uint8 = []
    for frame in frames:
        if frame.dtype != np.uint8:
            # Normalize to 0-255 if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        frames_uint8.append(frame)
    
    # Save video
    imageio.mimsave(path, frames_uint8, fps=fps, quality=8)
    print(f"Video saved to {path}")


def collect_frames_from_env(env, policy, max_steps=2000):
    """Collect RGB frames from an environment during rollout.
    
    Args:
        env: Environment with render() method
        policy: Policy callable
        max_steps: Maximum number of steps
        
    Returns:
        List of RGB frames
    """
    import torch
    
    frames = []
    obs_dict = env.get_obs()
    
    for step in range(max_steps):
        # Render frame
        try:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
        except:
            # If render is not supported, skip frame collection
            pass
        
        # Step environment
        with torch.no_grad():
            action = policy(obs_dict, deterministic=True)
        
        obs_dict, reward, done, info = env.step(action)
        
        if torch.any(done):
            break
    
    return frames 