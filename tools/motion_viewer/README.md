# Motion Viewer

A web-based viewer for MimicKit motion files (.pkl) with 3D visualization.

## Installation

```bash
pip install flask
```

## Usage

From the MimicKit root directory:

```bash
cd tools/motion_viewer
python app.py
```

Then open your browser to http://localhost:5000

## Features

- Browse all motion files in `data/motions/`
- Play/pause/reset controls
- Frame-by-frame scrubbing with slider
- Keyboard controls:
  - `Space`: Play/Pause
  - `←/→`: Previous/Next frame
  - `Home`: First frame
  - `End`: Last frame
- Automatic motion grouping by character type
- Displays motion metadata (FPS, duration, loop mode)
- **Raw Data View**: Inspect unpickled file contents with data types and shapes

## Motion Format

Motion files are pickle files containing:
- `fps`: Frame rate
- `loop_mode`: 0 (CLAMP) or 1 (WRAP)
- `frames`: NumPy array of poses, each frame contains:
  - Root position (3D)
  - Root rotation (3D exponential map)
  - Joint rotations (format depends on character)

See [mimickit/anim/motion.py](../../mimickit/anim/motion.py) for details.
