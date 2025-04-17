# Half-UNet-FNO Optimization for Turbulent Radiative Layer Prediction

This project focuses on optimizing a Half-UNet Fourier Neural Operator (FNO) hybrid model to predict the next time step of a 2D turbulent radiative layer flow field, given the previous time step. The data used is the `turbulent_radiative_layer_2D` dataset from `the_well` repository. Hyperparameter optimization is performed using Optuna. 
*While the performance appears lacking, proper scaling of this architecture has not been investigated due to lack of available hardware.*

## Model Architecture: Half-UNet-FNO

### Half-UNet Benefits:

*   **Simplicity & Efficiency:** It simplifies the standard U-Net by removing the decoder's convolutional blocks and using direct fusion (summation) of upsampled encoder features. This reduces parameters and computational cost.
*   **Performance:** It often maintains strong performance, particularly when high-frequency details recovered by a complex decoder are less critical than capturing multi-scale features efficiently.

### Motivation for FNO Blocks (Replacing Standard Convolutions):

*   **Global Dependencies:** Standard convolutions use local kernels. Fourier Neural Operator (FNO) blocks operate in the frequency domain using `SpectralConv2d`. This allows them to capture global dependencies and long-range interactions across the entire input field much more effectively, which is crucial for modeling complex physical systems like fluid dynamics (turbulent flow).
*   **Physics Modeling:** FNOs are specifically designed for learning solutions to PDEs and are often better suited for capturing the underlying physics compared to standard CNNs. Replacing spatial convolutions with spectral convolutions aims to leverage this advantage for potentially better prediction accuracy in this physics-based task.

*(Note: While the original Half-UNet paper used Ghost Modules within its convolutions for efficiency, this specific `HalfUNetFNO` implementation replaces the standard convolutional blocks of the U-Net structure entirely with FNO blocks, rather than specifically replacing Ghost Modules within those blocks.)*

## Project Structure

```
project_root/
├── configs/
│   ├── config.py          # Configuration management (base_config.yaml)
├── data/                  # Data files (managed by the_well)
├── src/                   # Source code
│   ├── data_processing/   # Data loading and preprocessing (using the_well)
│   ├── models/            # Model definitions (HalfUNetFNO)
│   ├── training/          # Training logic, plotting
├── outputs/               # Generated outputs (models, Optuna DB, figures)
│   ├── models/            # Saved final model weights
│   ├── optuna/            # Optuna study database and trial models
│   └── figures/           # Saved loss curves and comparison plots
├── train.py               # Script for standard training with default/best hyperparameters
├── optimize.py            # Script for running Optuna hyperparameter optimization study
├── requirements.txt       # Project dependencies
└── README.md              # Project description
```

## Setup

1.  **Clone the repository and navigate to the project root.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure Data:** Make sure the `turbulent_radiative_layer_2D` dataset from `the_well` is accessible at the paths specified in `configs/base_config.yaml`.
4.  **Run Optimization (Optional):** To find the best hyperparameters using Optuna:
    ```bash
    python optimize.py
    ```
    This will run multiple training trials and save the study results and best trial models in the `outputs/optuna/` directory.
5.  **Run Standard Training:** To train a model using the default hyperparameters (or the best ones found by Optuna, after updating the config):
    ```bash
    python train.py
    ```
    This saves the final best model to the path specified in `configs/base_config.yaml`.

## Configuration

Hyperparameters, data paths, Optuna settings, and model paths are configured in `configs/base_config.yaml`. Key sections include:

*   `Data`: Paths to training/testing data and dataset name.
*   `Training`: Default settings for a standard training run (epochs, learning rate, patience, etc.).
*   `Model Architecture`: Default hyperparameters for the HalfUNetFNO (modes, width, levels, etc.). These are the defaults used in `train.py` and potentially overridden by Optuna during optimization.
*   `Optuna Study`: Settings specific to the `optimize.py` script (number of trials, epochs per trial, storage location, etc.).

## Hyperparameter Optimization with Optuna

The `optimize.py` script utilizes the Optuna framework to search for optimal hyperparameters for the Half-UNet-FNO model. It minimizes the validation loss over a defined search space specified within the script. The study progress and results are stored in an SQLite database (`outputs/optuna/half_unet_fno_study.db` by default).
It was found that the width of the input had a much more significant impact to the performance than the number of Fourier modes and the number of levels within the network. This is likely due to the greater degree of freedom in abstracting the data into a higher number of dimensions. The study was limited to a maximum width of 32 due to hardware constraints (running off a medium-spec laptop).

## Results

Training progress and results are visualized and saved in the `outputs/figures/` directory.


### Comparison Plots

Comparison plots showing the input field, the model's prediction, and the target field for a sample from the validation set are generated periodically during training.

<div align="center">
  <img src="figures\comparison.PNG" alt="Comparison after 15 epochs." width="800">
  <br>
  <em>Predictions from the Half-UNET-FNO after 15 epochs.</em>
</div>


### Citing
The Half-UNet architecture was introduced by
> Lu, Haoran, et al. "Half-UNet: A simplified U-Net architecture for medical image segmentation." Frontiers in Neuroinformatics 16 (2022).

The Fourier Neural Operator networks for PDEs was introduced by
> Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).

The data-set from which this model was trained was obtained from The Well collection.
> Ohana, Ruben, et al., "The well: a large-scale collection of diverse physics simulations for machine learning." dvances in Neural Information Processing Systems 37 (2024).