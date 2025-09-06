# Hand-to-Object Diffusion Training

A modular, production-quality system for training diffusion models to predict object motion from hand trajectories, with real-time 3D visualization using Rerun.

## 🎯 Project Overview

This project implements a **hand-to-object diffusion training pipeline** that:

- **Trains diffusion models** to predict object motion from hand trajectories
- **Provides real-time visualization** using Rerun with hand articulations and MANO models
- **Supports multiple data formats** (9D poses, 12D poses with velocity)
- **Offers modular architecture** for easy extension and maintenance
- **Includes comprehensive evaluation** and analysis tools

### Key Features

- ✅ **Real-time 3D visualization** with Rerun
- ✅ **Hand articulation visualization** using MANO models
- ✅ **Object trajectory prediction** and comparison
- ✅ **Motion classification** (moving/stationary)
- ✅ **Modular visualization system** with error handling
- ✅ **Comprehensive data processing** pipeline
- ✅ **Multiple training modes** (overfitting, full training)

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Pixi** (dependency manager)
- **CUDA-compatible GPU** (recommended for training)

### Installation

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd omomo_release
   ```

3. **Install dependencies with Pixi**:
   ```bash
   pixi install
   ```

4. **Activate the environment**:
   ```bash
   pixi shell
   ```

### Running Training

#### Basic Training (No Visualization)
```bash
pixi run python main.py \
  --data_path data/processed_data.pkl \
  --save_dir runs/basic_training \
  --num_steps 5000
```

#### Training with Rerun Visualization
```bash
pixi run python main.py \
  --data_path data/processed_data_with_velocity.pkl \
  --save_dir runs/visualized_training \
  --use_rerun \
  --exp_name "demo_with_visualization" \
  --num_steps 10000
```

#### Training with Advanced Options
```bash
pixi run python main.py \
  --data_path data/processed_data_with_velocity.pkl \
  --save_dir runs/advanced_training \
  --use_rerun \
  --use_velocity \
  --window 128 \
  --d_model 512 \
  --learning_rate 1e-4 \
  --num_steps 15000 \
  --visualization_frequency 200 \
  --enhanced_scene_frames 300
```

## 📁 Project Structure

```
omomo_release/
├── src/                                    # Source code package
│   ├── visualization/                      # Visualization modules
│   │   ├── rerun_visualizer.py            # Core Rerun visualization
│   │   ├── visualization_manager.py        # Visualization manager
│   │   └── visualization_config.py         # Configuration
│   ├── training/                           # Training modules
│   │   ├── trainer_hand_to_object_diffusion_overfit.py
│   │   ├── trainer_hand_to_object_diffusion.py
│   │   └── demo_training_with_rerun.py
│   ├── data_processing/                    # Data processing modules
│   │   ├── extract_hand_articulations.py  # Hand articulation extraction
│   │   ├── preprocess_data.py             # Data preprocessing
│   │   └── inspect_*.py                   # Data inspection tools
│   └── utils/                              # Utility modules
│       ├── evaluation_metrics.py          # Evaluation utilities
│       ├── setup_mano.py                  # MANO model setup
│       └── visualize_*.py                 # Visualization utilities
├── docs/                                   # Documentation
├── data/                                   # Data files
│   ├── mano_models/                        # MANO hand models
│   ├── hand_articulations.pkl              # Hand articulation data
│   └── processed_data*.pkl                 # Processed training data
├── main.py                                 # Main entry point
├── pixi.toml                               # Pixi configuration
└── README.md                               # This file
```

## 🔧 Configuration

### Command Line Options

#### Data Parameters
- `--data_path`: Path to processed data pickle file
- `--demo_id`: Specific demo ID to use
- `--target_object_id`: Specific object ID to track
- `--use_velocity`: Use 12D data with velocity

#### Model Parameters
- `--window`: Training window size (default: 128)
- `--d_model`: Transformer model dimension (default: 512)
- `--n_dec_layers`: Number of decoder layers (default: 6)
- `--n_head`: Number of attention heads (default: 8)

#### Training Parameters
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_steps`: Number of training steps (default: 10000)
- `--sampling_mode`: Window sampling mode (random/sequential)

#### Visualization Parameters
- `--use_rerun`: Enable Rerun visualization
- `--visualization_frequency`: How often to save visualization files
- `--enhanced_scene_frames`: Number of frames for enhanced scene
- `--mano_models_dir`: Directory containing MANO model files

## 🎨 Visualization

### Rerun Visualization

The system provides comprehensive 3D visualization:

- **Real-time training progress** with hand and object trajectories
- **Hand articulations** using detailed MANO models
- **Object motion comparison** (ground truth vs predicted)
- **Motion classification** and velocity metrics
- **Enhanced scene visualization** with full 3D environment

### Viewing Visualizations

```bash
# View training progress
rerun runs/your_experiment/rerun_visualizations/training_step_*.rrd --web-viewer --port 9877

# View final results
rerun runs/your_experiment/rerun_visualizations/full_trajectory_final.rrd --web-viewer --port 9877

# View enhanced scene
rerun runs/your_experiment/rerun_visualizations/*_enhanced_hands.rrd --web-viewer --port 9877
```

## 📊 Data Format

### Input Data Structure

The system expects processed data in the following format:

```python
{
    "demo_id": {
        "left_hand": {
            "poses_9d": np.array([T, 9]),      # 9D poses (translation + rotation)
            "poses_12d": np.array([T, 12])     # 12D poses (translation + velocity + rotation)
        },
        "right_hand": {
            "poses_9d": np.array([T, 9]),
            "poses_12d": np.array([T, 12])
        },
        "objects": {
            "object_id": {
                "poses_9d": np.array([T, 9]),
                "poses_12d": np.array([T, 12])
            }
        }
    }
}
```

### Hand Articulations

For enhanced visualization, the system can use detailed hand articulation data:

```python
{
    "sequence_id": {
        "left_hand": [
            {
                "landmarks_21": np.array([21, 3]),      # 21 3D landmarks
                "mesh_vertices": np.array([N, 3]),      # Mesh vertices
                "mesh_faces": np.array([M, 3]),         # Triangle faces
            }
        ],
        "right_hand": [...],
        "landmark_connectivity": [[0, 1], [1, 2], ...]  # Bone connections
    }
}
```

## 🧪 Development

### Using as a Package

```python
from src.visualization import VisualizationManager
from src.training import train_overfit, HandToObjectDataset
from src.data_processing import extract_hand_articulations

# Initialize visualization
viz_manager = VisualizationManager(opt, dataset)

# Run training
train_overfit(opt, device)
```

### Adding New Features

1. **Choose the right package** based on functionality
2. **Create the module** in the appropriate directory
3. **Update the package's `__init__.py`** to export the module
4. **Add tests** in the `tests/` directory
5. **Update documentation**

### Testing

```bash
# Run all tests
pixi run pytest tests/

# Run specific package tests
pixi run pytest tests/visualization/
pixi run pytest tests/training/
```

## 📚 Documentation

- **This README** - Project overview and quick start
- **PROJECT_STRUCTURE.md** - Detailed project structure
- **docs/VISUALIZATION_README.md** - Comprehensive visualization guide
- **Code Documentation** - Inline docstrings and type hints

## 🔍 Troubleshooting

### Common Issues

1. **MANO Models Not Loading**
   - Check `data/mano_models/` directory exists
   - Verify `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` files
   - Install SMPLX: `pixi add smplx`

2. **Rerun Not Initializing**
   - Check Rerun installation: `pixi add rerun-sdk`
   - Verify no other Rerun processes are running

3. **CUDA Issues**
   - Check CUDA installation and compatibility
   - Verify PyTorch CUDA version matches system CUDA

4. **Data Loading Errors**
   - Verify data file paths and formats
   - Check pickle file compatibility

### Performance Tips

- **Visualization frequency**: Use higher values (e.g., 500) for faster training
- **Frame limits**: Limit enhanced scene frames for quicker generation
- **Batch size**: Adjust based on GPU memory
- **Data caching**: The system automatically caches loaded data

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Update** documentation
5. **Submit** a pull request

## 📄 License

This project is part of the hand-to-object diffusion training pipeline.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review the documentation files
- Open an issue on the repository

---

**Happy Training! 🚀**

This modular system makes it easy to experiment with hand-to-object diffusion models while providing rich, real-time visualization of your training progress.
