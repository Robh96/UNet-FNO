from the_well.data import WellDataset
from torch.utils.data import DataLoader
import torch
import functools

def calculate_channel_min_max(dataset, field_name, num_channels):
    """Calculates the min and max value for each channel across the dataset."""
    print(f"Calculating min/max for '{field_name}'...")
    # Initialize min/max tensors
    # Assuming data is float32
    channel_min = torch.full((num_channels,), float('inf'), dtype=torch.float32)
    channel_max = torch.full((num_channels,), float('-inf'), dtype=torch.float32)

    # Use a simple DataLoader to iterate without shuffling or batching for stats
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_dict in temp_loader:
        # Shape: (1, T, X, Y, C)
        data = batch_dict[field_name]
        # Reduce over all dims except channel dim (last dim, index 4)
        batch_min = torch.amin(data.float(), dim=(0, 1, 2, 3)) # Shape: (C,)
        batch_max = torch.amax(data.float(), dim=(0, 1, 2, 3)) # Shape: (C,)

        channel_min = torch.minimum(channel_min, batch_min)
        channel_max = torch.maximum(channel_max, batch_max)

    print(f"Min values for '{field_name}': {channel_min.tolist()}")
    print(f"Max values for '{field_name}': {channel_max.tolist()}")
    # Reshape for broadcasting during scaling: (1, C, 1, 1)
    return channel_min.view(1, -1, 1, 1), channel_max.view(1, -1, 1, 1)

def get_data_loaders(batch_size, train_path, test_path, dataset_name):
    # --- 1. Load Training Dataset ---
    trainset = WellDataset(
        well_base_path=train_path,
        well_dataset_name=dataset_name,
        well_split_name="train"
    )

    # --- 2. Calculate Normalization Statistics from Training Set ---
    # Determine number of channels from the first sample
    # This assumes all samples have the same channel count
    first_sample = trainset[0]
    num_input_channels = first_sample['input_fields'].shape[-1]
    num_output_channels = first_sample['output_fields'].shape[-1]
    print(f"Determined input channels: {num_input_channels}, output channels: {num_output_channels}") # Added print

    input_min, input_max = calculate_channel_min_max(trainset, 'input_fields', num_input_channels)
    output_min, output_max = calculate_channel_min_max(trainset, 'output_fields', num_output_channels)

    # Move stats to the target device once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_min, input_max = input_min.to(device), input_max.to(device)
    output_min, output_max = output_min.to(device), output_max.to(device)

    # --- 3. Load Validation Dataset ---
    valset = WellDataset(
        well_base_path=test_path,
        well_dataset_name=dataset_name,
        well_split_name="test"
    )

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # Added num_workers and pin_memory
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False) # Added num_workers and pin_memory

    # --- 5. Create Preprocessing Function with Stats ---
    # Use functools.partial to create versions of the preprocess function
    # that already have the min/max stats baked in.
    preprocess_func_with_stats = functools.partial(
        preprocess_batch_conv2d,
        input_min=input_min,
        input_max=input_max,
        output_min=output_min,
        output_max=output_max
    )

    # --- 6. Wrap DataLoaders ---
    train_dl = WrappedDataLoader(train_loader, preprocess_func_with_stats)
    valid_dl = WrappedDataLoader(val_loader, preprocess_func_with_stats)

    # --- 7. Return loaders and channel info ---
    return train_dl, valid_dl, num_input_channels, num_output_channels


class WrappedDataLoader:
    """Wraps a DataLoader to apply a preprocessing function to each batch."""
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch_dict in self.dl:  # DataLoader collates dicts into a dict of batches
            # Apply the preprocessing function to the batch dictionary
            xb, yb = self.func(batch_dict)  # Ensure the function returns a tuple (xb, yb)
            yield xb, yb

def preprocess_batch_conv2d(batch_dict, input_min, input_max, output_min, output_max):
    """
    Preprocesses a batch dictionary for Conv2d input.
    Selects the single time step (t=0) and permutes dimensions.
    Assumes 'output_fields' corresponds to the same time step.
    """
    input_batch = batch_dict['input_fields']   # Shape: (batch_size, 1, x, y, c)
    target_batch = batch_dict['output_fields'] # Shape: (batch_size, 1, x, y, c)
    device = input_min.device # Get device from stats tensors

    # --- Input Processing ---
    # Select the single time step (index 0)
    # Shape: (batch_size, x, y, c)
    selected_input = input_batch[:, 0, :, :, :]
    processed_input = selected_input.permute(0, 3, 1, 2) # (batch_size, channels, height, width)
    processed_input = processed_input.to(device)


    # Min-Max Scaling (-1, 1) for Input
    input_range = input_max - input_min
    epsilon = 1e-8
    processed_input = 2 * (processed_input - input_min) / (input_range + epsilon) - 1


    # --- Target Processing ---
    # Select the single time step (index 0)
    # Shape: (batch_size, x, y, c)
    selected_target = target_batch[:, 0, :, :, :]
    processed_target = selected_target.permute(0, 3, 1, 2)
    processed_target = processed_target.to(device)
    
    # Min-Max Scaling (-1, 1) for Target
    output_range = output_max - output_min
    processed_target = 2 * (processed_target - output_min) / (output_range + epsilon) - 1

    return processed_input, processed_target