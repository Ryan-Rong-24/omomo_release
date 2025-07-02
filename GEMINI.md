# Gemini Workspace Context

This document outlines the context for the Gemini AI assistant to effectively collaborate on this project.

## User Profile & Goal

- **Role:** Researcher at Stanford AI Lab.
- **Primary Objective:** Develop and train a diffusion model to inpaint and denoise object motion trajectories.
- **Conditioning Data:** The model will be conditioned on the motion trajectories of the user's left and right hands.

## Project Overview

This project is a fork and adaptation of the `omomo` repository, which originally focused on denoising full-body human motion conditioned on hand and foot movements. We are repurposing its core components, specifically the diffusion transformer architecture, for our research task.

### Key Characteristics:

-   **Core Task:** Inpainting/denoising of object motion trajectories.
-   **Model:** Diffusion Transformer.
-   **Dataset:** HOT3D.
-   **Input Data:**
    -   Left Hand Trajectory: `T x 9` (3D translation + 6D rotation matrix).
    -   Right Hand Trajectory: `T x 9` (3D translation + 6D rotation matrix).
-   **Output/Target Data:**
    -   Object Trajectory: `T x 9` (3D translation + 6D rotation matrix).

## Current Status & Important Files

The project is in a work-in-progress state. The immediate focus is on validating the model architecture and training pipeline.

-   **Overfitting Test:** `trainer_hand_to_object_diffusion_overfit.py` is a key script created to test the model's ability to overfit to a single data sample from the HOT3D dataset. This serves as an initial sanity check for the adapted diffusion model.
-   **Data Loading:** Data loading logic, likely adapted from the original `omomo` structure, needs to be understood to handle the HOT3D format.
-   **Visualization:** Visualization scripts will be used to inspect the model's output and debug the denoising/in-painting process.

## Codebase Analysis

### `trainer_hand_to_object_diffusion_overfit.py`

- **Purpose:** This script is designed to overfit the diffusion model on a single demonstration from the HOT3D dataset. This is a crucial step to verify that the model and training pipeline are correctly implemented.
- **`HandToObjectDataset`:** A custom dataset class that loads a single demonstration from a pickle file. It extracts the left hand, right hand, and object trajectories. It uses a sliding window approach to create smaller training samples from the full trajectory.
- **`train_overfit` function:**
    - Initializes the `CondGaussianDiffusion` model.
    - Sets up the optimizer and wandb for logging.
    - The training loop samples windows from the `HandToObjectDataset` and feeds them to the model.
    - Periodically evaluates the model by sampling from the diffusion model and comparing the generated object trajectory to the ground truth.
    - Saves the best model based on the evaluation error.
    - After training, it generates and saves the full sampled trajectory, the input hand poses, and the ground truth object motion.

### `manip/model/transformer_hand_to_object_diffusion_model.py`

- **`CondGaussianDiffusion`:** This is the main class that orchestrates the diffusion process. It wraps the `TransformerDiffusionModel`.
- **`TransformerDiffusionModel`:** This is the core neural network. It's a Transformer-based model that takes the noisy object motion, the timestep embedding, and the conditioning (hand poses) as input, and predicts the denoised object motion.
- **Forward Pass:** The `forward` method of `TransformerDiffusionModel` takes the source (noisy object motion), noise timestep embedding, and condition (hand poses) as input. It passes these through the `motion_transformer` (a `Decoder` from `transformer_module.py`) and a final linear layer to produce the output.

### `manip/model/transformer_module.py`

- **`Decoder`:** This is a standard Transformer decoder architecture. It consists of a stack of `DecoderLayer` modules.
- **`DecoderLayer`:** Each decoder layer has a self-attention mechanism and a position-wise feed-forward network.
- **`MultiHeadAttention`:** A standard multi-head attention implementation.

## Gemini's Role

Your primary role is to assist in the research and development process. This includes:
-   Reading and understanding the existing codebase (`omomo` fork).
-   Modifying and writing new code for the diffusion model, data loaders, and training loops.
-   Implementing and adapting visualization tools.
-   Running experiments and analyzing results.
-   Adhering to the existing coding style and structure of the project.