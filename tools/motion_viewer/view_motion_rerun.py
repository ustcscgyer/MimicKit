#!/usr/bin/env python3
"""
Motion viewer using Rerun.io - Advanced visualization with multiple panels,
timeline control, and real-time analysis.
"""
import sys
import os
import argparse
import numpy as np

# Add mimickit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../mimickit'))

from anim.motion import load_motion
from anim.mjcf_char_model import MJCFCharModel
import util.torch_util as torch_util
import torch

try:
    import rerun as rr
except ImportError:
    print("Rerun not installed. Install with: pip install rerun-sdk")
    sys.exit(1)

def visualize_motion(motion_file, character_file, speed=1.0, save_file=None):
    """Visualize motion with Rerun.io."""
    
    # Load motion
    print(f"Loading motion: {motion_file}")
    motion = load_motion(motion_file)
    print(f"  Frames: {motion.frames.shape[0]}, FPS: {motion.fps}, Duration: {motion.get_length():.2f}s")
    
    # Load character model
    print(f"Loading character: {character_file}")
    char_model = MJCFCharModel(device='cpu')
    char_model.load(character_file)
    body_names = char_model.get_body_names()
    print(f"  Bodies: {len(body_names)}")
    
    # Initialize Rerun - save to file for remote viewing
    if save_file:
        rr.init("MimicKit Motion Viewer", recording_id=os.path.basename(motion_file))
        rr.save(save_file)
        print(f"\nSaving to: {save_file}")
        print(f"View with: rerun {save_file}")
    else:
        rr.init("MimicKit Motion Viewer", spawn=True)
        print("\nRerun viewer opened")
    
    # Log static data
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Log motion metadata
    rr.log("motion/info", rr.TextDocument(f"""
# Motion Info
- **File**: {os.path.basename(motion_file)}
- **Frames**: {motion.frames.shape[0]}
- **FPS**: {motion.fps}
- **Duration**: {motion.get_length():.2f}s
- **Loop Mode**: {'WRAP' if motion.loop_mode.value == 1 else 'CLAMP'}
- **Character**: {os.path.basename(character_file).replace('.xml', '')}
- **Bodies**: {len(body_names)}
"""), static=True)
    
    # Get parent indices for bones
    parent_indices = [char_model.get_parent_id(i) for i in range(len(body_names))]
    
    # Process each frame
    print("\nVisualizing motion...")
    for frame_idx, frame in enumerate(motion.frames):
        # Set timeline
        rr.set_time_sequence("frame", frame_idx)
        
        # Extract pose data
        root_pos = torch.tensor(frame[0:3], dtype=torch.float32)
        root_rot_exp = torch.tensor(frame[3:6], dtype=torch.float32)
        joint_dof = torch.tensor(frame[6:], dtype=torch.float32)
        
        # Convert to quaternions
        root_rot = torch_util.exp_map_to_quat(root_rot_exp.unsqueeze(0)).squeeze(0)
        joint_rot = char_model.dof_to_rot(joint_dof.unsqueeze(0)).squeeze(0)
        
        # Forward kinematics
        body_pos, body_rot = char_model.forward_kinematics(
            root_pos.unsqueeze(0),
            root_rot.unsqueeze(0),
            joint_rot.unsqueeze(0)
        )
        
        body_pos = body_pos.squeeze(0).cpu().numpy()
        body_rot = body_rot.squeeze(0).cpu().numpy()
        
        # Log joint positions as 3D points with labels
        joint_positions = body_pos
        joint_colors = [[255, 100, 100] if i == 0 else [100, 150, 255] for i in range(len(joint_positions))]
        rr.log("skeleton/joints", rr.Points3D(
            joint_positions,
            radii=0.05,
            colors=joint_colors,
            labels=body_names
        ))
        
        # Log bones as line segments
        bone_starts = []
        bone_ends = []
        for i in range(1, len(body_names)):
            parent_idx = parent_indices[i]
            if parent_idx >= 0:
                bone_starts.append(body_pos[parent_idx])
                bone_ends.append(body_pos[i])
        
        if bone_starts:
            rr.log("skeleton/bones", rr.LineStrips3D(
                [np.stack([bone_starts[i], bone_ends[i]]) for i in range(len(bone_starts))],
                colors=[150, 180, 255],
                radii=0.02
            ))
        
        # Log transforms for each body (shows coordinate frames)
        for i, (name, pos, rot) in enumerate(zip(body_names, body_pos, body_rot)):
            # Convert quaternion from [x,y,z,w] to [w,x,y,z] for Rerun
            rot_rerun = [rot[3], rot[0], rot[1], rot[2]]
            rr.log(f"skeleton/transforms/{name}", rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=rot_rerun),
                axis_length=0.1 if i == 0 else 0.05
            ))
        
        # Log root trajectory (path over time)
        if frame_idx > 0:
            prev_root_pos = torch.tensor(motion.frames[frame_idx-1][0:3], dtype=torch.float32).numpy()
            rr.log("trajectory/root", rr.LineStrips3D(
                [[prev_root_pos, root_pos.numpy()]],
                colors=[255, 200, 0],
                radii=0.015
            ))
        
        # Log velocity (as scalar time series)
        if frame_idx > 0:
            prev_frame = motion.frames[frame_idx - 1]
            prev_root_pos = torch.tensor(prev_frame[0:3], dtype=torch.float32)
            velocity = np.linalg.norm((root_pos - prev_root_pos).numpy()) * motion.fps
            rr.log("metrics/velocity", rr.Scalar(velocity))
        
        # Log height
        rr.log("metrics/height", rr.Scalar(root_pos[2].item()))
        
        # Log joint angles (first 3 for demo)
        for i in range(min(3, len(joint_dof))):
            rr.log(f"metrics/joint_{i}", rr.Scalar(joint_dof[i].item()))
    
    print(f"\nVisualized {len(motion.frames)} frames")
    print("Rerun viewer opened - use timeline to scrub through frames!")
    print("Features:")
    print("  - Click joints to see their transforms")
    print("  - Toggle layers on/off in the blueprint panel")
    print("  - Scrub timeline to see motion")
    print("  - View metrics in time series panel")
    
    # Keep the script running so Rerun stays open
    input("\nPress Enter to exit...")

def main():
    parser = argparse.ArgumentParser(description='Visualize MimicKit motion files with Rerun.io')
    parser.add_argument('--motion', required=True, help='Path to motion .pkl file')
    parser.add_argument('--character', help='Path to character .xml file (auto-detected if not provided)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--save', help='Save to .rrd file for viewing later (recommended for remote machines)')
    
    args = parser.parse_args()
    
    # Auto-detect character file
    if args.character is None:
        # Extract character name from motion path
        parts = args.motion.split(os.sep)
        if 'motions' in parts:
            idx = parts.index('motions')
            if idx + 1 < len(parts):
                character_name = parts[idx + 1]
                args.character = os.path.join('data', 'assets', character_name, f'{character_name}.xml')
    
    if not os.path.exists(args.motion):
        print(f"Error: Motion file not found: {args.motion}")
        return
    
    if not os.path.exists(args.character):
        print(f"Error: Character file not found: {args.character}")
        print(f"Please specify --character explicitly")
        return
    
    visualize_motion(args.motion, args.character, args.speed, save_file=args.save)

if __name__ == '__main__':
    main()
