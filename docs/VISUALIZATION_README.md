# Rerun Visualization System for Hand-to-Object Training

This document describes the modular Rerun visualization system integrated into the hand-to-object diffusion training pipeline.

## 🎯 Features

The visualization system provides comprehensive real-time visualization of:

- **Hand Trajectories**: Left (green) and right (blue) hand positions over time
- **Object Motion**: Ground truth (red) and predicted (orange) object trajectories
- **Hand Articulations**: Detailed hand meshes and skeletons using MANO models
- **Enhanced Scene**: Full 3D scene with hands, objects, and motion analysis
- **Training Progress**: Real-time visualization during training with configurable frequency

## 🚀 Quick Start

### 1. Enable Visualization in Training

```bash
python trainer_hand_to_object_diffusion_overfit.py \
    --use_rerun \
    --exp_name "my_experiment" \
    --save_dir "runs/my_experiment"
```

### 2. View Visualizations

```bash
# View training progress
rerun runs/my_experiment/rerun_visualizations/training_step_*.rrd --web-viewer --port 9877

# View final results
rerun runs/my_experiment/rerun_visualizations/full_trajectory_final.rrd --web-viewer --port 9877

# View enhanced scene
rerun runs/my_experiment/rerun_visualizations/*_enhanced_hands.rrd --web-viewer --port 9877
```

## 📁 File Structure

```
runs/my_experiment/
├── rerun_visualizations/
│   ├── training_step_1000.rrd      # Training progress snapshots
│   ├── training_step_2000.rrd
│   ├── full_trajectory_final.rrd   # Final trajectory comparison
│   └── demo_id_enhanced_hands.rrd  # Enhanced scene visualization
├── weights/                         # Model checkpoints
├── training_summary.txt             # Training summary
└── opt.yaml                        # Training configuration
```

## ⚙️ Configuration Options

### Basic Visualization

```bash
--use_rerun                          # Enable Rerun visualization
--exp_name "experiment_name"         # Experiment name for visualization
--save_dir "runs/experiment"         # Output directory
```

### Advanced Visualization

```bash
--mano_models_dir "data/mano_models"                    # MANO model directory
--hand_articulations_path "data/hand_articulations.pkl"  # Hand articulation data
--generation_data_path "data/generation.pkl"             # Generation data
--visualization_frequency 100                            # Visualization frequency (steps)
--enhanced_scene_frames 500                              # Frames for enhanced scene
```

## 🔧 Data Requirements

The visualization system requires the following data files:

### Required Files

1. **MANO Models** (`data/mano_models/`)
   - `MANO_LEFT.pkl` - Left hand MANO model
   - `MANO_RIGHT.pkl` - Right hand MANO model

2. **Hand Articulations** (`data/hand_articulations.pkl`)
   - Detailed hand mesh and skeleton data
   - 21 landmark positions and connectivity

3. **Generation Data** (`data/generation.pkl`)
   - Basic hand poses and object data
   - Used for enhanced scene visualization

### Data Format

```python
# Hand articulations format
hand_articulations = {
    "sequence_id": {
        "left_hand": [
            {
                "landmarks_21": np.array([21, 3]),      # 21 3D landmarks
                "mesh_vertices": np.array([N, 3]),      # Mesh vertices
                "mesh_faces": np.array([M, 3]),         # Triangle faces
            },
            # ... more frames
        ],
        "right_hand": [...],
        "landmark_connectivity": [[0, 1], [1, 2], ...]  # Bone connections
    }
}
```

## 🧪 Testing the System

Run the test script to verify visualization functionality:

```bash
python test_visualization.py
```

This will:
- Test basic visualization with sample data
- Test enhanced scene visualization (if data files exist)
- Generate test output files in `test_output/`

## 📊 Visualization Types

### 1. Training Progress Visualization

- **Real-time updates** during training
- **Hand trajectories** with color coding
- **Object motion** comparison (GT vs predicted)
- **Motion classification** (moving/stationary)
- **Velocity metrics** and sequence information

### 2. Full Trajectory Visualization

- **Complete trajectory** comparison
- **Statistical analysis** (length, velocity)
- **Side-by-side** ground truth vs prediction
- **Exportable** Rerun files for sharing

### 3. Enhanced Scene Visualization

- **3D hand meshes** using MANO models
- **Detailed hand articulations** with skeletons
- **Object poses** and transformations
- **Multi-frame** animation support

## 🎨 Customization

### Coordinate Transformations

Modify the `_apply_coordinate_transform` method in `RerunVisualizer`:

```python
def _apply_coordinate_transform(self, position):
    """Custom coordinate transformation"""
    # Example: 90-degree rotation around X-axis
    transform_matrix = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])
    return transform_matrix @ position
```

### Color Schemes

Customize colors in visualization methods:

```python
# Hand colors
left_hand_color = [0, 255, 0]      # Green
right_hand_color = [0, 0, 255]     # Blue

# Object colors
gt_color = [255, 0, 0]             # Red
pred_color = [255, 165, 0]         # Orange
```

## 🐛 Troubleshooting

### Common Issues

1. **MANO Models Not Loading**
   - Check `data/mano_models/` directory exists
   - Verify `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` files
   - Install SMPLX: `pip install smplx`

2. **Hand Articulations Missing**
   - Check `data/hand_articulations.pkl` exists
   - Verify data format matches expected structure

3. **Rerun Not Initializing**
   - Check Rerun installation: `pip install rerun-sdk`
   - Verify no other Rerun processes are running

4. **Visualization Too Slow**
   - Reduce `--visualization_frequency`
   - Reduce `--enhanced_scene_frames`
   - Use smaller data subsets for testing

### Performance Tips

- **Visualization frequency**: Use higher values (e.g., 500) for faster training
- **Frame limits**: Limit enhanced scene frames for quicker generation
- **Data caching**: The system automatically caches loaded data
- **Error handling**: Failed visualizations won't stop training

## 📚 API Reference

### RerunVisualizer Class

```python
class RerunVisualizer:
    def __init__(self, exp_name, save_dir, enable_visualization=True, 
                 mano_models_dir=None, hand_articulations_path=None, 
                 generation_data_path=None):
        """Initialize visualizer with configurable paths"""
    
    def visualize_training_frame(self, step, left_hand, right_hand, 
                               object_motion_gt, object_motion_pred=None, 
                               seq_len=None, is_moving=None, mean_velocity=None):
        """Visualize single training frame"""
    
    def visualize_full_trajectory(self, dataset, sampled_motion, step_name="final"):
        """Visualize complete trajectory comparison"""
    
    def visualize_enhanced_scene(self, dataset, sequence_key=None, num_frames=500):
        """Create enhanced 3D scene visualization"""
```

### VisualizationManager Class

```python
class VisualizationManager:
    def __init__(self, opt, dataset):
        """Initialize visualization manager for training"""
    
    def visualize_training_step(self, step, left_hand, right_hand, 
                              object_motion, **kwargs):
        """Visualize training step with error handling"""
    
    def visualize_best_model_prediction(self, **kwargs):
        """Visualize best model during evaluation"""
    
    def visualize_final_results(self, sampled_motion_full):
        """Visualize final training results"""
```

## 🔗 Integration Examples

### Basic Training Integration

```python
from rerun_visualizer import RerunVisualizer

# Initialize visualizer
visualizer = RerunVisualizer("my_exp", "runs/my_exp")

# Visualize training step
visualizer.visualize_training_frame(
    step=1000,
    left_hand=left_hand_data,
    right_hand=right_hand_data,
    object_motion_gt=object_gt,
    object_motion_pred=object_pred
)
```

### Advanced Training Integration

```python
from trainer_hand_to_object_diffusion_overfit import VisualizationManager

# Initialize visualization manager
viz_manager = VisualizationManager(opt, dataset)

# Use throughout training loop
viz_manager.visualize_training_step(step, left_hand, right_hand, object_motion)
viz_manager.visualize_best_model_prediction(...)
viz_manager.visualize_final_results(sampled_motion)
```

## 📈 Future Enhancements

- **Interactive controls** for visualization parameters
- **Export to video** formats (MP4, GIF)
- **Batch visualization** for multiple experiments
- **Web dashboard** for remote monitoring
- **Custom visualization** plugins

## 🤝 Contributing

To contribute to the visualization system:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Update** documentation
5. **Submit** a pull request

## 📄 License

This visualization system is part of the hand-to-object diffusion training pipeline.
