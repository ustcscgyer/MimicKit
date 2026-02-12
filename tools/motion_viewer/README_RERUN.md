# Motion Viewer with Rerun.io

Advanced motion visualization with Rerun.io providing multiple views, timeline control, and real-time metrics.

## Installation

```bash
pip install rerun-sdk
```

## Usage

```bash
cd tools/motion_viewer

# Auto-detect character from motion path
python view_motion_rerun.py --motion ../../data/motions/humanoid/humanoid_backflip.pkl

# Explicit character file
python view_motion_rerun.py --motion ../../data/motions/g1/g1_cartwheel.pkl --character ../../data/assets/g1/g1.xml

# Adjust playback speed
python view_motion_rerun.py --motion ../../data/motions/humanoid/humanoid_walk.pkl --speed 0.5
```

## Advanced Features

### Multiple Synchronized Views
- **3D Scene**: Full skeleton with joints and bones
- **Time Series**: Velocity, height, joint angles
- **Text Log**: Motion metadata and statistics
- **Blueprint**: Layer visibility controls

### Interactive Timeline
- Scrub through frames with the timeline slider
- Play/pause with spacebar
- Jump to specific times
- See all metrics synchronized with the 3D view

### Rich Annotations
- Click any joint to see its transform (position + rotation)
- Toggle coordinate frames on/off
- View root trajectory path
- Color-coded joints (root=red, others=blue)

### Analysis Tools
- Real-time velocity tracking
- Height profile over time
- Individual joint angle plots
- Automatic metric correlation

### Data Export
- Export view states
- Save screenshots/recordings
- Export data to CSV

## Keyboard Shortcuts (in Rerun)
- `Space`: Play/Pause
- `←/→`: Previous/Next frame  
- `Home`: First frame
- `End`: Last frame
- `R`: Reset view
- `F`: Focus on selection
- `G`: Toggle grid
- `T`: Toggle time panel

## Comparison with Basic Viewer

| Feature | Flask Viewer | Rerun Viewer |
|---------|--------------|--------------|
| 3D Skeleton | ✓ | ✓✓ (Better quality) |
| Timeline | Basic slider | Advanced timeline |
| Metrics | None | Real-time plots |
| Multiple Views | No | Yes (split screen) |
| Annotations | No | Yes (clickable) |
| Export | No | Yes (screenshots, data) |
| Performance | Slow (large files) | Fast (streaming) |
| Coordinate Frames | No | Yes (per joint) |
| Trajectory | No | Yes (path over time) |

## Why Rerun is Better

1. **Purpose-built for robotics**: Designed for motion/trajectory data
2. **Streaming architecture**: Handles large datasets efficiently
3. **Multiple synchronized views**: See 3D + plots + logs simultaneously
4. **Production-ready**: Used by major robotics labs
5. **Better debugging**: Click-to-inspect, layer toggle, metric correlation
