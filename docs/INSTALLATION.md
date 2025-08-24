# Installation Guide - Using Pixi

This guide explains how to set up the hand-to-object diffusion training project using **Pixi** as the dependency manager.

## 🎯 Why Pixi?

**Pixi** is a modern, fast dependency manager that:
- ✅ **Manages Python environments** automatically
- ✅ **Handles complex dependencies** (PyTorch, CUDA, etc.)
- ✅ **Provides reproducible builds** across different systems
- ✅ **Integrates seamlessly** with conda-forge and PyPI
- ✅ **Faster than conda** and more reliable than pip

## 🚀 Installation Steps

### 1. Install Pixi

#### On Linux/macOS:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

#### On Windows:
```bash
# Using PowerShell
irm https://pixi.sh/install.ps1 | iex

# Using winget
winget install pixi
```

#### Verify Installation:
```bash
pixi --version
```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd omomo_release
```

### 3. Install Dependencies

```bash
# Install all dependencies defined in pixi.toml
pixi install
```

This will:
- Create a new Python environment
- Install PyTorch with CUDA support
- Install Rerun SDK for visualization
- Install all other required packages

### 4. Activate the Environment

```bash
# Activate the Pixi environment
pixi shell

# Your prompt should now show the environment name
(omomo-hot3d) user@machine:~/omomo_release$
```

## 🔧 Environment Management

### Activating the Environment

```bash
# Always activate before working on the project
pixi shell

# Or run commands directly with pixi run
pixi run python main.py --help
```

### Deactivating the Environment

```bash
# When you're done working
exit

# Or press Ctrl+D
```

### Updating Dependencies

```bash
# Update to latest compatible versions
pixi update

# Or update specific packages
pixi add package_name
```

## 📦 What Gets Installed

The `pixi.toml` file automatically installs:

### Core Dependencies
- **Python 3.10** - Programming language
- **PyTorch 2.1.2** - Deep learning framework with CUDA support
- **Rerun SDK** - 3D visualization framework
- **NumPy, SciPy** - Scientific computing

### Visualization Dependencies
- **Rerun SDK** - Real-time 3D visualization
- **Matplotlib** - 2D plotting
- **OpenCV** - Computer vision
- **Trimesh** - 3D mesh processing

### Deep Learning Dependencies
- **PyTorch** - Neural network framework
- **TorchVision** - Computer vision models
- **Einops** - Tensor operations
- **EMA PyTorch** - Exponential moving averages

### Data Processing Dependencies
- **Project Aria Tools** - AR data processing
- **VRS** - Video recording format
- **H5Py** - HDF5 file format
- **PyYAML** - YAML configuration files

## 🧪 Testing the Installation

### 1. Test Python Environment

```bash
pixi shell
python --version  # Should show Python 3.10.x
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import rerun; print('Rerun imported successfully')"
```

### 2. Test Project Imports

```bash
pixi run python -c "
from src.visualization import VisualizationManager
from src.training import train_overfit
print('✓ All imports successful')
"
```

### 3. Test Main Script

```bash
pixi run python main.py --help
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **Pixi Command Not Found**
```bash
# Restart your terminal after installation
# Or add to your shell profile:
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### 2. **CUDA Issues**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. **Permission Issues**
```bash
# If you get permission errors, try:
sudo chown -R $USER:$USER ~/.pixi
```

#### 4. **Package Conflicts**
```bash
# Clean environment and reinstall
pixi remove
pixi install
```

### Performance Issues

#### 1. **Slow Installation**
```bash
# Use faster conda channels
pixi install --channel conda-forge
```

#### 2. **Memory Issues**
```bash
# Reduce parallel downloads
export PIXI_MAX_PARALLEL_DOWNLOADS=2
pixi install
```

## 🔄 Development Workflow

### Daily Usage

```bash
# 1. Start work
cd omomo_release
pixi shell

# 2. Run training
python main.py --use_rerun --exp_name "my_experiment"

# 3. View results
rerun runs/my_experiment/rerun_visualizations/*.rrd --web-viewer

# 4. End work
exit
```

### Adding New Dependencies

```bash
# Add Python package
pixi add package_name

# Add specific version
pixi add "package_name>=1.0.0"

# Add development dependency
pixi add --group dev package_name
```

### Updating the Environment

```bash
# Update all packages
pixi update

# Update specific package
pixi add package_name@latest

# Lock file will be updated automatically
git add pixi.lock
git commit -m "Update dependencies"
```

## 📚 Additional Resources

- **Pixi Documentation**: https://pixi.sh/
- **Pixi GitHub**: https://github.com/prefix-dev/pixi
- **Conda Forge**: https://conda-forge.org/
- **PyTorch Installation**: https://pytorch.org/get-started/

## 🎉 You're Ready!

Once you've completed these steps, you can:

1. **Run training**: `pixi run python main.py --use_rerun`
2. **View visualizations**: `pixi run rerun --web-viewer`
3. **Process data**: `pixi run python src/data_processing/extract_hand_articulations.py`
4. **Develop features**: `pixi shell` then edit code

The Pixi environment ensures all dependencies are compatible and the project runs consistently across different systems!
