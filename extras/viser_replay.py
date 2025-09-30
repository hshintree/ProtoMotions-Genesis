#!/usr/bin/env python3
"""Viser replay viewer for recorded rollouts.

Usage:
    python extras/viser_replay.py --npz path/to/rollout.npz
"""

import argparse
import time
import numpy as np
import viser
from pathlib import Path


def load_rollout(path):
    """Load rollout data from npz file."""
    D = np.load(path, allow_pickle=True)
    return {k: D[k] for k in D.files}


def extract_reward_terms(info_array):
    """Extract reward terms from info array."""
    reward_terms = []
    for info_dict in info_array:
        terms = {}
        if isinstance(info_dict, dict) and "reward_terms" in info_dict:
            terms = info_dict["reward_terms"]
        elif isinstance(info_dict, np.ndarray) and info_dict.size == 1:
            # Handle object arrays
            info_obj = info_dict.item()
            if isinstance(info_obj, dict) and "reward_terms" in info_obj:
                terms = info_obj["reward_terms"]
        reward_terms.append(terms)
    return reward_terms


def get_term_names(reward_terms):
    """Get all unique reward term names."""
    term_names = set()
    for terms in reward_terms:
        if isinstance(terms, dict):
            term_names.update(terms.keys())
    return sorted(term_names)


def main(path):
    ro = load_rollout(path)
    steps = len(ro["obs"])
    
    # Extract reward terms
    reward_terms = extract_reward_terms(ro["info"])
    term_names = get_term_names(reward_terms)

    # Start Viser server
    server = viser.ViserServer()
    
    # Flat GUI (no folder kwargs in this version)
    server.gui.add_text("Replay Control", initial_value="Use Play/Pause and Frame slider")
    play_button = server.gui.add_button("Play / Pause")
    slider = server.gui.add_slider(
        "Frame", min=0, max=max(steps-1, 1), step=1, initial_value=0
    )
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0
    )
    loop_toggle = server.gui.add_checkbox("Loop", initial_value=True)

    server.gui.add_text("Camera Presets", initial_value="Top/Side/Front/Follow")
    cam_top = server.gui.add_button("Top View")
    cam_side = server.gui.add_button("Side View")
    cam_front = server.gui.add_button("Front View")
    cam_follow = server.gui.add_button("Follow Robot")

    server.gui.add_text("Reward Terms", initial_value="Individual terms:")
    total_rew_display = server.gui.add_number(
        "Total Reward", initial_value=0.0, disabled=True
    )
    
    # Create sliders for each reward term
    rbars = {}
    for name in term_names:
        if "rew" in name.lower():
            min_val, max_val = -10.0, 10.0
        else:
            min_val, max_val = -1.0, 1.0
        rbars[name] = server.gui.add_slider(
            name,
            min=min_val,
            max=max_val,
            step=0.001,
            initial_value=0.0,
            disabled=True,
        )

    # State management
    state = {"playing": False}

    @play_button.on_click
    def _(_):
        state["playing"] = not state["playing"]

    @cam_top.on_click
    def _(_):
        server.scene.set_up_direction("+z")
        for client in server.get_clients().values():
            client.camera.position = (0.0, 0.0, 5.0)
            client.camera.look_at = (0.0, 0.0, 0.0)

    @cam_side.on_click
    def _(_):
        server.scene.set_up_direction("+z")
        for client in server.get_clients().values():
            client.camera.position = (4.0, 0.0, 1.5)
            client.camera.look_at = (0.0, 0.0, 1.0)

    @cam_front.on_click
    def _(_):
        server.scene.set_up_direction("+z")
        for client in server.get_clients().values():
            client.camera.position = (0.0, 4.0, 1.5)
            client.camera.look_at = (0.0, 0.0, 1.0)

    @cam_follow.on_click
    def _(_):
        i = int(slider.value)
        if len(ro["base_pos"]) > i:
            p = ro["base_pos"][i]
            # Handle both single env and multi-env cases
            if p.ndim > 1:
                p = p[0]  # Use first environment
            server.scene.set_up_direction("+z")
            for client in server.get_clients().values():
                client.camera.position = (float(p[0]) + 2.5, float(p[1]), float(p[2]) + 1.5)
                client.camera.look_at = (float(p[0]), float(p[1]), float(p[2]) + 0.9)

    # Create a frame to represent the robot base
    base_frame = server.scene.add_frame("/robot_base", wxyz=(1, 0, 0, 0), position=(0, 0, 0), show_axes=True, axes_length=0.3, axes_radius=0.01)

    def update_visualization(i):
        """Update the visualization for frame i."""
        # Update base pose if available
        if len(ro["base_pos"]) > i and len(ro["base_quat"]) > i:
            pos = ro["base_pos"][i]
            quat = ro["base_quat"][i]
            
            # Handle multi-env case (take first env)
            if pos.ndim > 1:
                pos = pos[0]
            if quat.ndim > 1:
                quat = quat[0]
            
            # Convert to wxyz format (Viser expects w-first)
            if len(quat) == 4:
                if abs(quat[3]) > abs(quat[0]):
                    wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
                else:
                    wxyz = tuple(float(q) for q in quat)
                
                base_frame.wxyz = wxyz
            base_frame.position = tuple(float(p) for p in pos)

        # Update reward terms
        terms = reward_terms[i] if i < len(reward_terms) else {}
        total = 0.0
        
        for name in term_names:
            val = 0.0
            if isinstance(terms, dict) and name in terms:
                term_val = terms[name]
                if isinstance(term_val, np.ndarray):
                    if term_val.size == 1:
                        val = float(term_val.item())
                    else:
                        val = float(term_val.mean())
                else:
                    val = float(term_val)
                total += val
            
            if name in rbars:
                rbars[name].value = val
        
        total_rew_display.value = total

    # Main loop
    last_time = time.time()
    print(f"\n{'='*50}")
    print(f"Viser server started!")
    print(f"Loaded rollout: {path}")
    print(f"Total frames: {steps}")
    print(f"Reward terms: {', '.join(term_names) if term_names else 'None'}")
    print(f"\nOpen the viewer at: http://localhost:8080")
    print(f"{'='*50}\n")

    while True:
        i = int(slider.value)
        update_visualization(i)
        
        if state["playing"]:
            dt = time.time() - last_time
            step_inc = max(1, int(dt * 60 * speed.value))
            new_frame = i + step_inc
            
            if new_frame >= steps:
                if loop_toggle.value:
                    slider.value = 0
                else:
                    state["playing"] = False
                    slider.value = steps - 1
            else:
                slider.value = new_frame
        
        last_time = time.time()
        time.sleep(1/60)  # 60 Hz update rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay recorded rollouts with Viser")
    parser.add_argument("--npz", required=True, help="Path to .npz rollout file")
    args = parser.parse_args()
    
    if not Path(args.npz).exists():
        print(f"Error: File not found: {args.npz}")
        exit(1)
    
    main(args.npz) 