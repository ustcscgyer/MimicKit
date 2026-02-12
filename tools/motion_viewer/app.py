#!/usr/bin/env python3
from flask import Flask, render_template, jsonify
import sys
import os
import numpy as np

import torch
import pickle

# Add both parent and mimickit directories to path
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'mimickit'))

from anim.motion import load_motion
from anim.mjcf_char_model import MJCFCharModel
import util.torch_util as torch_util
import xml.etree.ElementTree as ET

app = Flask(__name__)

MOTION_DIR = os.path.join(os.path.dirname(__file__), '../../data/motions')
ASSET_DIR = os.path.join(os.path.dirname(__file__), '../../data/assets')

# Cache for character models
_character_models = {}

def get_character_model(character_name):
    """Load or retrieve cached character model."""
    if character_name not in _character_models:
        xml_path = os.path.join(ASSET_DIR, character_name, f"{character_name}.xml")
        if os.path.exists(xml_path):
            model = MJCFCharModel(device='cpu')
            model.load(xml_path)
            _character_models[character_name] = model
        else:
            return None
    return _character_models[character_name]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/motions')
def list_motions():
    """List all available motion files."""
    motion_files = []
    
    for root, dirs, files in os.walk(MOTION_DIR):
        for file in files:
            if file.endswith('.pkl'):
                rel_path = os.path.relpath(os.path.join(root, file), MOTION_DIR)
                motion_files.append(rel_path)
    
    motion_files.sort()
    return jsonify(motion_files)

# NOTE: More specific routes must come before generic ones
@app.route('/api/motion/<path:filename>/skeleton')
def get_motion_skeleton(filename):
    """Get full skeleton data with forward kinematics for all frames."""
    filepath = os.path.join(MOTION_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        # Determine character type from filename
        parts = filename.split('/')
        character_name = parts[0] if len(parts) > 0 else 'humanoid'
        
        # Load character model
        char_model = get_character_model(character_name)
        if char_model is None:
            return jsonify({"error": f"Character model not found for {character_name}"}), 404
        
        # Load motion
        motion = load_motion(filepath)
        
        # Extract pose data and compute forward kinematics for all frames
        frames_skeleton = []
        for frame in motion.frames:
            # Extract root pos, root rot (exp map), joint dof
            root_pos = torch.tensor(frame[0:3], dtype=torch.float32)
            root_rot_exp = torch.tensor(frame[3:6], dtype=torch.float32)
            joint_dof = torch.tensor(frame[6:], dtype=torch.float32)
            
            # Convert exponential map to quaternion
            root_rot = torch_util.exp_map_to_quat(root_rot_exp.unsqueeze(0)).squeeze(0)
            
            # Convert joint DOF to rotations
            joint_rot = char_model.dof_to_rot(joint_dof.unsqueeze(0)).squeeze(0)
            
            # Forward kinematics
            body_pos, body_rot = char_model.forward_kinematics(
                root_pos.unsqueeze(0),
                root_rot.unsqueeze(0),
                joint_rot.unsqueeze(0)
            )
            
            # Convert to list (ensure Python native types)
            body_pos = body_pos.squeeze(0).cpu().numpy()
            frames_skeleton.append([[float(x) for x in pos] for pos in body_pos])
        
        # Get skeleton structure
        body_names = char_model.get_body_names()
        parent_indices = [int(char_model.get_parent_id(i)) for i in range(len(body_names))]
        
        return jsonify({
            "body_names": body_names,
            "parent_indices": parent_indices,
            "frames": frames_skeleton,
            "num_bodies": len(body_names)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/motion/<path:filename>/raw')
def get_motion_raw(filename):
    """Get raw unpickled data from motion file."""
    filepath = os.path.join(MOTION_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        with open(filepath, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in raw_data.items():
            if hasattr(value, 'tolist'):
                serializable_data[key] = {
                    'type': 'numpy.ndarray',
                    'shape': list(value.shape) if hasattr(value, 'shape') else None,
                    'dtype': str(value.dtype) if hasattr(value, 'dtype') else None,
                    'data': value.tolist()
                }
            else:
                serializable_data[key] = {
                    'type': type(value).__name__,
                    'value': value
                }
        
        return jsonify(serializable_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/motion/<path:filename>')
def get_motion(filename):
    """Get motion data for a specific file."""
    filepath = os.path.join(MOTION_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        motion = load_motion(filepath)
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            "fps": int(motion.fps),
            "loop_mode": int(motion.loop_mode.value),
            "frames": motion.frames.tolist(),
            "num_frames": motion.frames.shape[0],
            "duration": motion.get_length(),
            "file_size": file_size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/character/<character_name>')
def get_character(character_name):
    """Get character skeleton structure from XML file."""
    xml_path = os.path.join(ASSET_DIR, character_name, f"{character_name}.xml")
    
    if not os.path.exists(xml_path):
        return jsonify({"error": "Character XML not found"}), 404
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Parse joint hierarchy
        joints = []
        def parse_body(body_elem, parent_name=None, depth=0):
            body_name = body_elem.get('name')
            body_pos = body_elem.get('pos', '0 0 0')
            pos = [float(x) for x in body_pos.split()]
            
            # Get joints in this body
            for joint_elem in body_elem.findall('joint'):
                joint_name = joint_elem.get('name')
                joint_axis = joint_elem.get('axis', '1 0 0')
                joint_range = joint_elem.get('range', '-180 180')
                
                joints.append({
                    'name': joint_name,
                    'body': body_name,
                    'parent': parent_name,
                    'axis': [float(x) for x in joint_axis.split()],
                    'range': [float(x) for x in joint_range.split()],
                    'pos': pos
                })
            
            # Recursively process child bodies
            for child_body in body_elem.findall('body'):
                parse_body(child_body, body_name, depth + 1)
        
        # Find worldbody and parse
        worldbody = root.find('worldbody')
        if worldbody is not None:
            for body in worldbody.findall('body'):
                parse_body(body)
        
        return jsonify({
            "character": character_name,
            "joints": joints,
            "num_joints": len(joints)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Motion Viewer on http://localhost:5000")
    print(f"Motion directory: {MOTION_DIR}")
    print(f"Asset directory: {ASSET_DIR}")
    app.run(debug=True, host='0.0.0.0', port=5000)
