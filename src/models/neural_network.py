import torch.nn as nn
import torch
import torch.fft

# def get_model(device):
#     model = nn.Sequential(
#         nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1)
#     ).to(device)
#     return model

# class SpectralConv2d(nn.Module):
#     """ 2D Fourier layer (FFT -> Linear Transform -> IFFT) """
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1 # Number of Fourier modes to keep in first dimension
#         self.modes2 = modes2 # Number of Fourier modes to keep in second dimension

#         self.scale = (1 / (in_channels * out_channels))
#         # Weights for the positive and negative frequency components
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2 // 2 + 1, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2 // 2 + 1, dtype=torch.cfloat))

#     def compl_mul2d(self, input, weights):
#         # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]
#         # Compute Fourier coeffcients
#         x_ft = torch.fft.rfft2(x, norm='ortho')

#         # Filter modes and perform linear transform
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2//2+1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2//2+1], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2//2+1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2//2+1], self.weights2)

#         # Compute Inverse Fourier Transform
#         x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
#         return x

# class FNOBlock(nn.Module):
#     """ Combines Spectral Convolution and Spatial Convolution with Residual """
#     def __init__(self, in_channels, out_channels, modes1, modes2, activation=nn.GELU()):
#         super().__init__()
#         self.spec_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
#         self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
#         self.activation = activation

#     def forward(self, x):
#         x_res = self.residual_conv(x)
#         x_spec = self.spec_conv(x)
#         x_spat = self.spatial_conv(x)
#         x_out = self.activation(x_spec + x_spat) + x_res
#         return x_out

# # --- U-Net FNO Model ---
# class UNetFNO(nn.Module):
#     def __init__(self, in_channels, out_channels, modes, width, levels=3, activation=nn.GELU()):
#         super().__init__()
#         self.modes = modes
#         self.width = width
#         self.levels = levels
#         self.activation = activation

#         self.lifting = nn.Conv2d(in_channels, self.width, kernel_size=1)

#         self.encoder = nn.ModuleList()
#         self.downsamplers = nn.ModuleList()
#         current_width = self.width
#         for i in range(self.levels):
#             self.encoder.append(FNOBlock(current_width, current_width * 2, self.modes, self.modes, activation=self.activation))
#             self.downsamplers.append(nn.MaxPool2d(2))
#             current_width *= 2
#             # Reduce modes at deeper levels (optional, common practice)
#             # self.modes = max(self.modes // 2, 4)

#         self.bottleneck = FNOBlock(current_width, current_width, self.modes, self.modes, activation=self.activation)

#         self.decoder = nn.ModuleList()
#         self.upsamplers = nn.ModuleList()
#         for i in range(self.levels):
#             # self.modes = min(self.modes * 2, modes) # Increase modes back
#             self.upsamplers.append(nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                 nn.Conv2d(current_width, current_width // 2, kernel_size=1) # Halve channels after upsampling
#             ))
#             # Input to decoder block: upsampled channels + skip connection channels
#             decoder_in_channels = current_width + (current_width // 2)
#             self.decoder.append(FNOBlock(decoder_in_channels, current_width // 2, self.modes, self.modes, activation=self.activation))
#             current_width //= 2

#         self.projection = nn.Sequential(
#             nn.Conv2d(self.width, self.width * 4, kernel_size=1),
#             self.activation,
#             nn.Conv2d(self.width * 4, out_channels, kernel_size=1)
#         )

#     def forward(self, x):
#         # Expecting (B, C, H, W)
#         if x.dim() != 4:
#              raise ValueError(f"Input must be 4D (B, C, H, W). Got {x.shape}")
#         if x.shape[1] != self.lifting.in_channels:
#              raise ValueError(f"Input channel mismatch. Expected {self.lifting.in_channels}, got {x.shape[1]}")

#         x = self.lifting(x)

#         skips = []
#         for i in range(self.levels):
#             x = self.encoder[i](x)
#             skips.append(x)
#             x = self.downsamplers[i](x)

#         x = self.bottleneck(x)

#         for i in range(self.levels):
#             x = self.upsamplers[i](x)
#             skip = skips[self.levels - 1 - i]

#             # Handle potential size mismatch from pooling/upsampling
#             if x.shape[-2:] != skip.shape[-2:]:
#                 # Use interpolation instead of padding for potentially better results
#                 x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)

#             x = torch.cat((x, skip), dim=1) # Concatenate along channel dimension
#             x = self.decoder[i](x)

#         x = self.projection(x)
#         return x



import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building Blocks (Unchanged) ---

class SpectralConv2d(nn.Module):
    """ 2D Fourier layer (FFT -> Linear Transform -> IFFT) """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Store the maximum modes requested
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights with the maximum requested modes
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2 // 2 + 1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2 // 2 + 1, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        # Einsum is efficient but requires exact dimension matches.
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Get spatial dimensions H, W
        H, W = x.size(-2), x.size(-1)

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho')
        # x_ft shape: (batch, in_channel, H, W//2 + 1)

        # --- Determine Effective Modes based on Input Size ---
        modes1_eff = min(self.modes1, H)
        # modes2_eff corresponds to the index limit in the rfft output's last dim
        modes2_eff_half = min(self.modes2 // 2 + 1, x_ft.size(-1))

        # Allocate space for filtered Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, H, W//2 + 1, dtype=torch.cfloat, device=x.device)

        # --- Apply Weights using Effective Modes ---
        # Slice both input and weights to the effective modes before einsum
        # Top-left modes
        out_ft[:, :, :modes1_eff, :modes2_eff_half] = self.compl_mul2d(
            x_ft[:, :, :modes1_eff, :modes2_eff_half],
            self.weights1[:, :, :modes1_eff, :modes2_eff_half] # Slice weights
        )
        # Bottom-left modes (negative frequencies)
        out_ft[:, :, -modes1_eff:, :modes2_eff_half] = self.compl_mul2d(
            x_ft[:, :, -modes1_eff:, :modes2_eff_half],
            self.weights2[:, :, :modes1_eff, :modes2_eff_half] # Slice weights
        )
        # --- End Modification ---

        # Compute Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho') # Use original H, W
        return x

class FNOBlock(nn.Module):
    """ Combines Spectral Convolution and Spatial Convolution with Residual """
    def __init__(self, in_channels, out_channels, modes1, modes2, activation=nn.GELU()):
        super().__init__()
        # Spectral convolution path
        self.spec_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        # Spatial convolution path (pointwise)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Residual connection (identity or 1x1 conv if channels change)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        # Activation function
        self.activation = activation

    def forward(self, x):
        # Calculate residual connection first
        x_res = self.residual_conv(x)
        # Calculate spectral and spatial paths
        x_spec = self.spec_conv(x)
        x_spat = self.spatial_conv(x)
        # Combine spectral and spatial paths, apply activation, add residual
        x_out = self.activation(x_spec + x_spat) + x_res
        return x_out

# --- Half-UNet FNO Model ---
class HalfUNetFNO(nn.Module):
    """
    Half-UNet architecture combined with FNO blocks.

    Replaces the standard U-Net encoder-decoder with:
    - Encoder using FNOBlocks with constant channel width.
    - Simplified decoder using full-scale feature fusion (summation).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (int): Number of Fourier modes to keep in each dimension for FNOBlocks.
        width (int): The constant channel width used throughout the encoder and fusion.
        levels (int): Number of downsampling levels in the encoder (total encoder blocks = levels + 1).
        activation (nn.Module): Activation function to use in FNOBlocks and projection. Default: nn.GELU().
        num_final_blocks (int): Number of FNOBlocks to apply after feature fusion. Default: 2.
    """
    def __init__(self, in_channels, out_channels, modes, width, levels=4, activation=nn.GELU(), num_final_blocks=2):
        super().__init__()
        self.modes = modes
        self.width = width
        self.levels = levels # Number of pooling layers
        self.activation = activation
        self.num_final_blocks = num_final_blocks

        # Initial lifting layer to project input to the specified width
        self.lifting = nn.Conv2d(in_channels, self.width, kernel_size=1)

        # --- Encoder Path ---
        self.encoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        # Add encoder blocks and pooling layers
        # Total encoder blocks = levels + 1 (including bottleneck)
        for i in range(self.levels + 1):
            # All encoder blocks operate with constant channel width
            block_in_channels = self.width if i > 0 else self.width # Input is 'width' after lifting
            self.encoder_blocks.append(
                FNOBlock(block_in_channels, self.width, self.modes, self.modes, activation=self.activation)
            )
            # Add pooling layer for all levels except the last (bottleneck)
            if i < self.levels:
                self.pooling_layers.append(nn.MaxPool2d(2))

        # --- Decoder Path (Full-Scale Feature Fusion) ---
        # No separate decoder blocks needed; fusion happens in forward pass.

        # --- Final Processing Blocks ---
        # Apply FNO blocks after fusion, maintaining channel width
        self.final_fno_blocks = nn.Sequential(
            *[FNOBlock(self.width, self.width, self.modes, self.modes, activation=self.activation)
              for _ in range(self.num_final_blocks)]
        )

        # --- Final Projection Layer ---
        # Maps the features to the desired output channels
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, self.width * 4, kernel_size=1), # Expand features
            self.activation,
            nn.Conv2d(self.width * 4, out_channels, kernel_size=1) # Project to output channels
        )

    def forward(self, x):
        # Expecting (B, C, H, W)
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W). Got {x.shape}")
        if x.shape[1] != self.lifting.in_channels:
            raise ValueError(f"Input channel mismatch. Expected {self.lifting.in_channels}, got {x.shape[1]}")

        # 1. Lifting
        x = self.lifting(x)
        target_size = x.shape[2:] # Store original size (H, W) for upsampling

        # 2. Encoder Path
        encoder_outputs = []
        current_input = x
        for i in range(self.levels + 1): # Iterate through all encoder blocks
            # Apply FNO block
            enc_out = self.encoder_blocks[i](current_input)
            encoder_outputs.append(enc_out)

            # Apply pooling if not the last block (bottleneck)
            if i < self.levels:
                current_input = self.pooling_layers[i](enc_out)
            else:
                # This was the bottleneck block
                pass

        # 3. Full-Scale Feature Fusion (Decoder)
        fused = None
        for i, enc_out in enumerate(encoder_outputs):
            # Upsample each encoder output to the original spatial size
            upsampled_out = F.interpolate(enc_out, size=target_size, mode='bilinear', align_corners=False)
            # Sum the upsampled feature maps
            if fused is None:
                fused = upsampled_out
            else:
                fused += upsampled_out
        # 'fused' now contains the summed features at the original resolution with 'width' channels

        # 4. Final FNO Blocks
        x_out = self.final_fno_blocks(fused)

        # 5. Projection
        x_out = self.projection(x_out)

        return x_out

# # Example Usage (demonstration purposes)
# if __name__ == '__main__':
#     # Example parameters
#     in_ch = 3
#     out_ch = 1
#     modes = 12  # Fourier Modes
#     width = 32  # Constant channel width
#     levels = 4  # Number of downsampling steps

#     model = HalfUNetFNO(in_channels=in_ch, out_channels=out_ch, modes=modes, width=width, levels=levels)

#     # Create a dummy input tensor
#     batch_size = 2
#     height = 64
#     width_img = 64
#     input_tensor = torch.randn(batch_size, in_ch, height, width_img)

#     # Forward pass
#     try:
#         output_tensor = model(input_tensor)
#         print(f"Input Shape: {input_tensor.shape}")
#         print(f"Output Shape: {output_tensor.shape}") # Should be [batch_size, out_ch, height, width_img]

#         # Count parameters
#         total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"Total trainable parameters: {total_params:,}")

#     except ValueError as e:
#         print(f"Error during forward pass: {e}")
#     except Exception as e:
#          print(f"An unexpected error occurred: {e}")



def get_model(device,
              in_channels,
              out_channels,
              modes,
              width,
              levels,
              num_final_blocks, # Added
              activation_name): # Changed to name
    """
    Instantiates and returns the HalfUNetFNO model on the specified device
    with the given hyperparameters.
    """
    # Map activation name to actual function
    if activation_name.lower() == 'relu':
        activation = nn.ReLU()
    elif activation_name.lower() == 'gelu':
        activation = nn.GELU()
    # Add other activations as needed
    # elif activation_name.lower() == 'silu':
    #     activation = nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

    model = HalfUNetFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=modes,
        width=width,
        levels=levels,
        activation=activation, # Pass the instance
        num_final_blocks=num_final_blocks
    ).to(device)
    return model
