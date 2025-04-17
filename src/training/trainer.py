import torch
import copy
import os
import optuna # Import optuna
from .plotting import plot_loss, plot_comparison

# Modified to accept trial and return best_val_loss
def train_model(epochs, model, loss_fn, opt, scheduler, train_dl, valid_dl, patience, best_model_path, trial=None):
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    last_val_batch_for_plot = None # Renamed to avoid conflict

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0
        train_samples = 0

        # Simplified training loop print statement
        for batch_idx, (xb, yb) in enumerate(train_dl):
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_accum += loss.item() * xb.size(0)
            train_samples += xb.size(0)
            # Print training progress every 10 batches
            # if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_dl)} | Loss: {loss.item():.6f}")

        avg_train_loss = train_loss_accum / train_samples if train_samples > 0 else float('nan')
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss_accum = 0.0
        val_samples = 0
        # last_val_batch = None # Removed, use last_val_batch_for_plot

        with torch.no_grad():
            for i, (xb_val, yb_val) in enumerate(valid_dl):
                pred_val = model(xb_val)
                val_loss = loss_fn(pred_val, yb_val)
                val_loss_accum += val_loss.item() * xb_val.size(0)
                val_samples += xb_val.size(0)
                if i == len(valid_dl) - 1: # Store last batch for potential plotting
                    last_val_batch_for_plot = (xb_val, pred_val, yb_val)

        avg_val_loss = val_loss_accum / val_samples if val_samples > 0 else float('nan')
        history['val_loss'].append(avg_val_loss)

        scheduler.step() # Step the scheduler

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- Optuna Pruning Integration ---
        if trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print("Trial pruned by Optuna.")
                # Return best loss seen so far for this pruned trial
                return best_val_loss if best_val_loss != float('inf') else avg_val_loss
        # --- End Optuna Integration ---

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            # Save the best model state immediately for this trial
            # Note: In Optuna study, this path might be temporary or trial-specific
            torch.save(best_model_state, best_model_path)
            print(f"  New best validation loss: {best_val_loss:.6f}. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        plot_loss(history, output_dir="figures", filename=f"loss_curves_trial_{trial.number if trial else 'final'}.png")
        if last_val_batch_for_plot:
            plot_comparison(epoch + 1, *last_val_batch_for_plot, output_dir="figures", trial_num=trial.number if trial else None)

    print(f"Training finished for this trial/run. Best validation loss: {best_val_loss:.6f}")

    # Return the best validation loss achieved during this run
    return best_val_loss