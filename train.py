import torch
import yaml
# import optuna # Removed optuna import
import os

from src.data_processing.loader import get_data_loaders
from src.models.neural_network import get_model
from src.training.trainer import train_model

# --- Configuration ---
CONFIG_PATH = "configs/base_config.yaml"
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Data ---
print("Loading data...")
train_loader, val_loader, num_input_channels, num_output_channels = get_data_loaders(
    batch_size=config["batch_size"],
    train_path=config["train_path"],
    test_path=config["test_path"],
    dataset_name=config["dataset_name"]
)
print("Data loaded.")

# Removed Optuna Objective Function

# --- Main Execution ---
if __name__ == "__main__":
    # This script now only runs standard training using config defaults
    print("\n--- Starting Standard Training Run ---")
    # Ensure model output directory exists
    os.makedirs(os.path.dirname(config["best_model_path"]), exist_ok=True)

    # --- Model Initialization (using defaults from config) ---
    model = get_model(
        device=device,
        in_channels=num_input_channels,
        out_channels=num_output_channels,
        modes=config["default_modes"],
        width=config["default_width"],
        levels=config["default_levels"],
        num_final_blocks=config["default_num_final_blocks"],
        activation_name=config["default_activation"]
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # --- Loss, Optimizer, Scheduler (using defaults from config) ---
    loss_fn = torch.nn.MSELoss(reduction='mean')
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=config["step_size"],
        gamma=config["gamma"]
    )

    # --- Training ---
    train_model(
        epochs=config["epochs"],
        model=model,
        loss_fn=loss_fn,
        opt=opt,
        scheduler=scheduler,
        train_dl=train_loader,
        valid_dl=val_loader,
        patience=config["patience"],
        best_model_path=config["best_model_path"]
    )
    print(f"Standard training finished. Best model saved to: {config['best_model_path']}")