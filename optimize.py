import torch
import yaml
import optuna
import os
import sys # Import sys for exiting if DB connection fails

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

# --- Load Data (Done ONCE before Optuna study) ---
print("Loading data...")
# These variables need to be accessible by the objective function
train_loader, val_loader, num_input_channels, num_output_channels = get_data_loaders(
    batch_size=config["batch_size"],
    train_path=config["train_path"],
    test_path=config["test_path"],
    dataset_name=config["dataset_name"]
)
print("Data loaded.")

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    """Optuna objective function to minimize validation loss."""
    print(f"\n--- Starting Trial {trial.number} ---")

    # --- Hyperparameter Suggestion (Reduced Search Space Example) ---
    # lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Fixed
    modes = trial.suggest_categorical("modes", [12, 16, 20]) # Reduced range
    width = trial.suggest_categorical("width", [16, 32]) # Kept original reduced range
    levels = trial.suggest_categorical("levels", [3, 4, 5]) # Reduced range
    # num_final_blocks = trial.suggest_int("num_final_blocks", 1, 3) # Fixed
    # activation_name = trial.suggest_categorical("activation", ["relu", "gelu"]) # Fixed

    # --- Use Fixed Values ---
    lr = 3e-4
    weight_decay = 1e-5
    num_final_blocks = 2
    activation_name = "gelu"

    print(f"  Trial {trial.number} Parameters:")
    print(f"    learning_rate: {lr:.6f}")
    print(f"    weight_decay: {weight_decay:.6f} (Fixed)")
    print(f"    modes: {modes}")
    print(f"    width: {width}")
    print(f"    levels: {levels}")
    print(f"    num_final_blocks: {num_final_blocks} (Fixed)")
    print(f"    activation: {activation_name} (Fixed)")

    # --- Model Initialization ---
    # Access num_input_channels and num_output_channels from the outer scope
    model = get_model(
        device=device,
        in_channels=num_input_channels,
        out_channels=num_output_channels,
        modes=modes,
        width=width,
        levels=levels,
        num_final_blocks=num_final_blocks,
        activation_name=activation_name
    )

    # --- Loss, Optimizer, Scheduler ---
    loss_fn = torch.nn.MSELoss(reduction='mean')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5) # Fixed scheduler for trials

    # --- Training for the Trial ---
    # Access config, train_loader, val_loader from the outer scope
    trial_best_model_path = config["optuna_best_model_path_template"].format(trial_number=trial.number)
    best_val_loss = train_model(
        epochs=config["optuna_epochs_per_trial"],
        model=model,
        loss_fn=loss_fn,
        opt=opt,
        scheduler=scheduler,
        train_dl=train_loader,
        valid_dl=val_loader,
        patience=config["optuna_patience_per_trial"],
        best_model_path=trial_best_model_path,
        trial=trial
    )

    print(f"--- Finished Trial {trial.number} | Best Validation Loss: {best_val_loss:.6f} ---")
    return best_val_loss

# --- Helper Function for Directory Creation ---
def ensure_parent_dir_exists(path_or_uri):
    """Extracts directory from path/URI and creates it if it doesn't exist."""
    dir_path = None
    if path_or_uri:
        # Handle potential SQLite URI prefix
        if path_or_uri.startswith("sqlite:///"):
            file_path = path_or_uri.replace("sqlite:///", "", 1)
            dir_path = os.path.dirname(file_path)
        else: # Assume it's a regular file path
            dir_path = os.path.dirname(path_or_uri)

        # Create directory if path is not empty (avoids issues with root)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Ensured directory exists: {dir_path}")

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Starting Optuna Hyperparameter Optimization Study ---")

    # --- Ensure Optuna output directories exist ---
    ensure_parent_dir_exists(config["optuna_storage"])
    ensure_parent_dir_exists(config["optuna_best_model_path_template"].format(trial_number=0))
    # --- End Directory Creation ---

    # --- Find Unique Study Name ---
    base_study_name = config["optuna_study_name"]
    storage_uri = config["optuna_storage"]
    study_name = base_study_name
    suffix = 0

    while True:
        try:
            # Try to load study to see if it exists
            optuna.load_study(study_name=study_name, storage=storage_uri)
            # If it exists, generate a new name with suffix
            print(f"Study '{study_name}' already exists.")
            study_name = f"{base_study_name}_{suffix}"
            suffix += 1
        except KeyError:
            # Study does not exist, we found a unique name
            print(f"Using study name: '{study_name}'")
            break
        except Exception as e:
            # Handle other potential errors during loading (e.g., DB connection)
            print(f"Error checking for study '{study_name}': {e}")
            print("Exiting.")
            sys.exit(1) # Exit if we can't verify study existence

    # --- Create Study ---
    # Now create the study with the unique name.
    # load_if_exists=True is safe here; it will only load if the loop above
    # somehow exited on an existing name (e.g., due to non-KeyError exception),
    # otherwise it creates the new one.
    study = optuna.create_study(
        study_name=study_name, # Use the determined unique name
        storage=storage_uri,
        direction="minimize",
        load_if_exists=True, # Keep True for resilience
        pruner=optuna.pruners.MedianPruner()
    )

    # Run the optimization
    study.optimize(
        objective, # Pass the objective function
        n_trials=config["optuna_n_trials"],
        # timeout=3600 * 8 # Optional timeout
    )

    # --- Print Study Results ---
    print("\n--- Optuna Study Finished ---")
    print(f"Study Name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"  Best Validation Loss: {best_trial.value:.6f}")
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- Train Final Model with Best Parameters ---
    print("\n--- Training Final Model with Best Parameters ---")
    best_params = best_trial.params
    final_model = get_model(
        device=device,
        in_channels=num_input_channels,
        out_channels=num_output_channels,
        modes=best_params["modes"],
        width=best_params["width"],
        levels=best_params["levels"],
        # num_final_blocks=best_params["num_final_blocks"],
        # activation_name=best_params["activation"]
    )
    final_loss_fn = torch.nn.MSELoss(reduction='mean')
    final_opt = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    final_scheduler = torch.optim.lr_scheduler.StepLR(
        final_opt,
        step_size=config["step_size"], # Use final step_size from config
        gamma=config["gamma"]         # Use final gamma from config
    )

    # Use final epochs and patience from config
    train_model(
        epochs=config["epochs"],
        model=final_model,
        loss_fn=final_loss_fn,
        opt=final_opt,
        scheduler=final_scheduler,
        train_dl=train_loader,
        valid_dl=val_loader,
        patience=config["patience"],
        best_model_path=config["best_model_path"], # Save final best model
        trial=None # No Optuna trial for the final run
    )
    print(f"Final best model saved to: {config['best_model_path']}")
