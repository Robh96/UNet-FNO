import os
import matplotlib.pyplot as plt
import numpy as np
import math # Import math for log10 and floor

# Add trial_num parameter with a default value of None
def plot_comparison(epoch, xb, pred, yb, output_dir="./figures", trial_num=None):
    os.makedirs(output_dir, exist_ok=True)
    for channel_idx in range(xb.shape[1]): # Iterate through channels if needed
        # Select the first sample in the batch and the current channel
        input_sample = xb[0, channel_idx].detach().cpu().numpy()
        pred_sample = pred[0, channel_idx].detach().cpu().numpy()
        target_sample = yb[0, channel_idx].detach().cpu().numpy()

        # Determine consistent color limits based on input and target
        all_vals = np.concatenate([input_sample.flatten(), target_sample.flatten()])
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        titles = [f"Input (Ch {channel_idx})", f"Prediction (Ch {channel_idx})", f"Target (Ch {channel_idx})"]
        data_to_plot = [input_sample, pred_sample, target_sample]

        for i, ax in enumerate(axes):
            im = ax.imshow(data_to_plot[i], aspect='auto', cmap='turbo', vmin=vmin, vmax=vmax)
            # Include trial number in the title if available
            plot_title = f"Epoch {epoch}: {titles[i]}"
            if trial_num is not None:
                plot_title = f"Trial {trial_num} - {plot_title}"
            ax.set_title(plot_title)
            ax.set_xlabel("Y index")
            ax.set_ylabel("X index")
            fig.colorbar(im, ax=ax)

        plt.tight_layout()
        # Modify filename to include trial number if available
        base_filename = f"channel_{channel_idx}_epoch_{epoch:04d}"
        if trial_num is not None:
            filename = os.path.join(output_dir, f"trial_{trial_num}_{base_filename}.png")
        else: # Keep original filename format for non-Optuna runs
            filename = os.path.join(output_dir, f"{base_filename}.png")

        try:
            plt.savefig(filename)
        except Exception as e:
            print(f"  Error saving plot {filename}: {e}")
        finally:
            plt.close(fig) # Close the figure to free memory

def plot_loss(history, output_dir=r".\figures", filename="loss_curves.png"):
    """Plots training and validation loss curves on a log scale."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    epochs_ran = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_ran, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_ran, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log') # Set y-axis to logarithmic scale
    plt.legend()
    plt.xlim(left=1, right=len(history['train_loss'])) # Set x-axis limits to match epochs
    plt.grid(True, which="both", ls="--") # Grid lines for both major and minor ticks on log scale
    min_loss = min(min(history['train_loss']), min(history['val_loss']))
    ylim_min = 10**math.floor(math.log10(min_loss))
    plt.ylim(bottom=ylim_min)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()