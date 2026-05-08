# %% [markdown]
# # Module 09: Convolutions - Processing Images with Convolutions
# 
# Welcome to Module 09! You'll implement spatial operations that transform machine learning from working with simple vectors to understanding images and spatial patterns.
# 
# ## 🔗 Prerequisites & Progress
# **You've Built**: Complete training pipeline with MLPs, optimizers, and data loaders
# **You'll Build**: Spatial operations - Conv2d, MaxPool2d, AvgPool2d for image processing
# **You'll Enable**: Convolutional Neural Networks (CNNs) for computer vision
# 
# **Connection Map**:
# ```
# Training Pipeline → Spatial Operations → CNN (Milestone 03)
#     (MLPs)            (Conv/Pool)        (Computer Vision)
# ```
# 
# ## 🎯 Learning Objectives
# By the end of this module, you will:
# 1. Implement Conv2d with explicit loops to understand O(N²M²K²) complexity
# 2. Build pooling operations (Max and Average) for spatial reduction
# 3. Understand receptive fields and spatial feature extraction
# 4. Analyze memory vs computation trade-offs in spatial operations
# 
# Let's get started!
# 
# ## 📦 Where This Code Lives in the Final Package
# 
# **Learning Side:** You work in `modules/09_convolutions/convolutions_dev.py`
# **Building Side:** Code exports to `tinytorch.core.spatial`
# 
# ```python
# # How to use this module:
# from tinytorch.core.spatial import Conv2d, MaxPool2d, AvgPool2d
# ```
# 
# **Why this matters:**
# - **Learning:** Complete spatial processing system in one focused module for deep understanding
# - **Production:** Proper organization like PyTorch's torch.nn.Conv2d with all spatial operations together
# - **Consistency:** All convolution and pooling operations in core.spatial
# - **Integration:** Works seamlessly with existing layers for complete CNN architectures

# %%
#| default_exp core.spatial
#| export

import numpy as np
rng = np.random.default_rng(42)
import time

from tinytorch.core.tensor import Tensor

# Enable autograd for gradient tracking (required for BatchNorm2d learnable parameters)
from tinytorch.core.autograd import enable_autograd, Function, ReLUBackward
enable_autograd()

# Constants for convolution defaults
DEFAULT_KERNEL_SIZE = 3  # Default kernel size for convolutions
DEFAULT_STRIDE = 1  # Default stride for convolutions
DEFAULT_PADDING = 0  # Default padding for convolutions

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion

# %% [markdown]
# ## 📋 Module Dependencies
# 
# **Prerequisites**: Modules 01-08 must be complete
# 
# **External Dependencies**:
# - `numpy` (for array operations and numerical computing)
# - `time` (for performance measurements)
# 
# **TinyTorch Dependencies**:
# - `tinytorch.core.tensor` (Tensor class from Module 01)
# - `tinytorch.core.autograd` (gradient tracking from Module 06)
# 
# **Important**: This module builds on the complete training pipeline.
# Spatial operations will integrate with your existing layers and training system.
# 
# **Dependency Flow**:
# ```
# Training Pipeline (Modules 01-08) → Spatial Operations (Module 09) → CNNs (Milestone 03)
#          ↓
#   Complete training system enables CNN development
# ```
# 
# Students completing this module will have built the spatial processing
# foundation that powers computer vision applications.

# %% [markdown]
# ## 💡 Introduction: What are Spatial Operations?
# 
# Spatial operations transform machine learning from working with simple vectors to understanding images and spatial patterns. When you look at a photo, your brain naturally processes spatial relationships, including edges, textures, and objects. Spatial operations give neural networks this same capability.
# 
# ### The Two Core Spatial Operations
# 
# **Convolution**: Detects local patterns by sliding filters across the input
# **Pooling**: Reduces spatial dimensions while preserving important features
# 
# ### Visual Example: How Convolution Works
# 
# ```
# Input Image (5×5):        Kernel (3×3):        Output (3×3):
# ┌─────────────────┐      ┌───────────┐       ┌─────────┐
# │ 1  2  3  4  5   │      │  1  0  -1 │       │ ?  ?  ? │
# │ 6  7  8  9  0   │  *   │  1  0  -1 │   =   │ ?  ?  ? │
# │ 1  2  3  4  5   │      │  1  0  -1 │       │ ?  ?  ? │
# │ 6  7  8  9  0   │      └───────────┘       └─────────┘
# │ 1  2  3  4  5   │
# └─────────────────┘
# 
# Sliding Window Process:
# Position (0,0): [1,2,3]   Position (0,1): [2,3,4]   Position (0,2): [3,4,5]
#                [6,7,8] *               [7,8,9] *               [8,9,0] *
#                [1,2,3]                 [2,3,4]                 [3,4,5]
#                = Output[0,0]           = Output[0,1]           = Output[0,2]
# ```
# 
# Each output pixel summarizes a local neighborhood, allowing the network to detect patterns like edges, corners, and textures.
# 
# ### Why Spatial Operations Transform ML
# 
# ```
# Without Convolution:                    With Convolution:
# 32×32×3 image = 3,072 inputs          32×32×3 → Conv → 32×32×16
# ↓                                      ↓                     ↓
# Dense(3072 → 1000) = 3M parameters    Shared 3×3 kernel = 432 parameters
# ↓                                      ↓                     ↓
# Memory explosion + no spatial awareness Efficient + preserves spatial structure
# ```
# 
# Convolution achieves dramatic parameter reduction (1000× fewer!) while preserving the spatial relationships that matter for visual understanding.

# %% [markdown]
# ## 📐 Mathematical Foundations
# 
# ### Understanding Convolution Step by Step
# 
# Convolution sounds complex, but it's just "sliding window multiplication and summation." Let's see exactly how it works:
# 
# ```
# Step 1: Position the kernel over input
# Input:          Kernel:
# ┌─────────┐     ┌─────┐
# │ 1 2 3 4 │     │ 1 0 │  ← Place kernel at position (0,0)
# │ 5 6 7 8 │  ×  │ 0 1 │
# │ 9 0 1 2 │     └─────┘
# └─────────┘
# 
# Step 2: Multiply corresponding elements
# Overlap:        Computation:
# ┌─────┐         1×1 + 2×0 + 5×0 + 6×1 = 1 + 0 + 0 + 6 = 7
# │ 1 2 │
# │ 5 6 │
# └─────┘
# 
# Step 3: Slide kernel and repeat
# Position (0,1):  Position (1,0):  Position (1,1):
# ┌─────┐         ┌─────┐          ┌─────┐
# │ 2 3 │         │ 5 6 │          │ 6 7 │
# │ 6 7 │         │ 9 0 │          │ 0 1 │
# └─────┘         └─────┘          └─────┘
# Result: 9       Result: 5        Result: 7
# 
# Final Output:  ┌─────┐
#                │ 7 9 │
#                │ 5 7 │
#                └─────┘
# ```
# 
# ### The Mathematical Formula
# 
# For 2D convolution, we slide kernel K across input I:
# ```
# O[i,j] = Σ Σ I[i+m, j+n] × K[m,n]
#          m n
# ```
# 
# This formula captures the "multiply and sum" operation for each kernel position.
# 
# ### Pooling: Spatial Summarization
# 
# ```
# Max Pooling Example (2×2 window):
# Input:             Output:
# ┌───────────────┐  ┌───────┐
# │ 1  3  2  4    │  │ 6   8 │  ← max([1,3,5,6])=6, max([2,4,7,8])=8
# │ 5  6  7  8    │  │ 9   9 │  ← max([2,9,0,1])=9, max([1,3,9,3])=9
# │ 2  9  1  3    │  └───────┘
# │ 0  1  9  3    │
# └───────────────┘
# 
# Average Pooling (same window):
# ┌─────────────┐
# │ 3.75   5.25 │  ← avg([1,3,5,6])=3.75, avg([2,4,7,8])=5.25
# │ 3.0    4.0  │  ← avg([2,9,0,1])=3.0, avg([1,3,9,3])=4.0
# └─────────────┘
# ```
# 
# ### Why This Complexity Matters
# 
# For convolution with input (1, 3, 224, 224) and kernel (64, 3, 3, 3):
# - **Operations**: 1 × 64 × 3 × 3 × 3 × 224 × 224 = 86.7 million multiply-adds
# - **Memory**: Input (600KB) + Weights (6.9KB) + Output (12.8MB) = ~13.4MB
# 
# This is why kernel size matters enormously - a 7×7 kernel would require 5.4× more computation!
# 
# ### Key Properties That Enable Deep Learning
# 
# **Translation Equivariance**: Move the cat → detection moves the same way
# **Parameter Sharing**: Same edge detector works everywhere in the image
# **Local Connectivity**: Each output only looks at nearby inputs (like human vision)
# **Hierarchical Features**: Early layers detect edges → later layers detect objects

# %% [markdown]
# ## 🏗️ Implementation: Building Spatial Operations
# 
# Now we'll implement convolution step by step, using explicit loops so you can see and feel the computational complexity. This helps you understand why modern optimizations matter!
# 
# ### Conv2d: Detecting Patterns with Sliding Windows
# 
# Convolution slides a small filter (kernel) across the entire input, computing weighted sums at each position. Think of it like using a template to find matching patterns everywhere in an image.
# 
# ```
# Convolution Visualization:
# Input (4×4):              Kernel (3×3):           Output (2×2):
# ┌─────────────┐          ┌─────────┐             ┌─────────┐
# │ a b c d │            │ k1 k2 k3│             │ o1  o2 │
# │ e f g h │     ×      │ k4 k5 k6│      =      │ o3  o4 │
# │ i j k l │            │ k7 k8 k9│             └─────────┘
# │ m n o p │            └─────────┘
# └─────────────┘
# 
# Computation Details:
# o1 = a×k1 + b×k2 + c×k3 + e×k4 + f×k5 + g×k6 + i×k7 + j×k8 + k×k9
# o2 = b×k1 + c×k2 + d×k3 + f×k4 + g×k5 + h×k6 + j×k7 + k×k8 + l×k9
# o3 = e×k1 + f×k2 + g×k3 + i×k4 + j×k5 + k×k6 + m×k7 + n×k8 + o×k9
# o4 = f×k1 + g×k2 + h×k3 + j×k4 + k×k5 + l×k6 + n×k7 + o×k8 + p×k9
# ```
# 
# ### The Six Nested Loops of Convolution
# 
# Our implementation will use explicit loops to show exactly where the computational cost comes from:
# 
# ```
# for batch in range(B):          # Loop 1: Process each sample
#     for out_ch in range(C_out):     # Loop 2: Generate each output channel
#         for out_h in range(H_out):      # Loop 3: Each output row
#             for out_w in range(W_out):      # Loop 4: Each output column
#                 for k_h in range(K_h):          # Loop 5: Each kernel row
#                     for k_w in range(K_w):          # Loop 6: Each kernel column
#                         for in_ch in range(C_in):       # Loop 7: Each input channel
#                             # The actual multiply-accumulate operation
#                             result += input[...] * kernel[...]
# ```
# 
# Total operations: B × C_out × H_out × W_out × K_h × K_w × C_in
# 
# For typical values (B=32, C_out=64, H_out=224, W_out=224, K_h=3, K_w=3, C_in=3):
# That's 32 × 64 × 224 × 224 × 3 × 3 × 3 = **2.8 billion operations** per forward pass!

# %% [markdown]
# ### Shared Input Validation
# 
# All spatial operations (Conv2d, MaxPool2d, AvgPool2d) require 4D inputs shaped
# as (batch, channels, height, width). Rather than duplicating this validation
# logic three times, we define it once here.
# 
# This is NOT a student task -- it is shared infrastructure.

# %%
#| export


def validate_4d_input(x, layer_name):
    """
    Validate that input tensor is 4D (batch, channels, height, width).

    Provides educational error messages that anticipate common student
    mistakes: forgetting the batch dimension, passing flattened data, etc.

    Args:
        x: Input Tensor to validate
        layer_name: Name of the calling layer (for error messages)

    Raises:
        ValueError: If input is not 4D, with specific guidance per case
    """
    if len(x.shape) == 4:
        return  # Valid input

    if len(x.shape) == 3:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got 3D: {x.shape}\n"
            f"  Missing batch dimension\n"
            f"  {layer_name} processes batches of images, not single images\n"
            f"  Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]}, {x.shape[2]})"
        )
    elif len(x.shape) == 2:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got 2D: {x.shape}\n"
            f"  Got a matrix, expected an image tensor\n"
            f"  {layer_name} needs spatial dimensions (height, width) plus batch and channels\n"
            f"  If this is a flattened image, reshape it: x.reshape(1, channels, height, width)"
        )
    else:
        raise ValueError(
            f"{layer_name} expected 4D input (batch, channels, height, width), got {len(x.shape)}D: {x.shape}\n"
            f"  Wrong number of dimensions\n"
            f"  {layer_name} expects: (batch_size, channels, height, width)\n"
            f"  Reshape your input to 4D with the correct dimensions"
        )

# %% [markdown]
# ### Conv2d Implementation - Building the Core of Computer Vision
# 
# Conv2d is the workhorse of computer vision. It slides learned filters across images to detect patterns like edges, textures, and eventually complex objects.
# 
# #### How Conv2d Transforms Machine Learning
# 
# ```
# Before Conv2d (Dense Only):         After Conv2d (Spatial Aware):
# Input: 32×32×3 = 3,072 values      Input: 32×32×3 structured as image
#          ↓                                   ↓
# Dense(3072→1000) = 3M params       Conv2d(3→16, 3×3) = 448 params
#          ↓                                   ↓
# No spatial awareness               Preserves spatial relationships
# Massive parameter count            Parameter sharing across space
# ```
# 
# #### Weight Initialization: He Initialization for ReLU Networks
# 
# Our Conv2d uses He initialization, specifically designed for ReLU activations:
# - **Problem**: Wrong initialization → vanishing/exploding gradients
# - **Solution**: std = sqrt(2 / fan_in) where fan_in = channels × kernel_height × kernel_width
# - **Why it works**: Maintains variance through ReLU nonlinearity
# 
# #### The 6-Loop Implementation Strategy
# 
# We'll implement convolution with explicit loops to show the true computational cost:
# 
# ```
# Nested Loop Structure:
# for batch:           ← Process each sample in parallel (in practice)
#   for out_channel:   ← Generate each output feature map
#     for out_h:       ← Each row of output
#       for out_w:     ← Each column of output
#         for k_h:     ← Each row of kernel
#           for k_w:   ← Each column of kernel
#             for in_ch: ← Accumulate across input channels
#               result += input[...] * weight[...]
# ```
# 
# This reveals why convolution is expensive: O(B×C_out×H×W×K_h×K_w×C_in) operations!
# 
# ### Implementation Strategy: Decomposed Helpers
# 
# We break Conv2d.forward into focused helpers so you can build and test each concept
# independently:
# 
# 1. **`_compute_output_shape(in_h, in_w)`** -- Apply the output dimension formula
# 2. **`_apply_padding(x_data)`** -- Zero-pad spatial dimensions
# 3. **`_convolve_loops(padded, batch, oh, ow)`** -- The sliding window dot products
# 4. **`forward(x)`** -- Compose all helpers together
# 
# Each helper teaches ONE concept. Build them in order, test each before moving on.

# %% [markdown]
# ### Step 1: Output Shape Formula
# 
# Before running convolution, you must know the output dimensions. The formula is:
# 
# ```
# out_height = (in_height + 2 * padding - kernel_h) // stride + 1
# out_width  = (in_width  + 2 * padding - kernel_w) // stride + 1
# ```
# 
# Examples with 32x32 input and 3x3 kernel:
# - padding=0, stride=1: (32 + 0 - 3) // 1 + 1 = 30 (shrinks)
# - padding=1, stride=1: (32 + 2 - 3) // 1 + 1 = 32 (same size)
# - padding=0, stride=2: (32 + 0 - 3) // 2 + 1 = 15 (downsamples)

# %% [markdown]
# ### Step 2: Zero Padding
# 
# Padding adds zeros around the spatial borders of the input, which controls
# whether convolution preserves the spatial size:
# 
# ```
# Before padding (3x3):      After padding=1 (5x5):
# ┌─────────┐                ┌───────────────┐
# │ 1  2  3 │                │ 0  0  0  0  0 │
# │ 4  5  6 │    -->         │ 0  1  2  3  0 │
# │ 7  8  9 │                │ 0  4  5  6  0 │
# └─────────┘                │ 0  7  8  9  0 │
#                            │ 0  0  0  0  0 │
#                            └───────────────┘
# ```
# 
# Only spatial dimensions (height, width) are padded. Batch and channel dims
# stay unchanged.

# %% [markdown]
# ### Step 3: The Convolution Loops
# 
# This is the core computation. For each output position, you extract a patch
# from the input and compute its dot product with the kernel:
# 
# ```
# Convolution = Sliding Window Dot Products:
# ┌──────────────────────────────────┐
# │ For EACH output position:        │
# │  Input patch     Kernel          │
# │  ┌───┬───┬───┐  ┌───┬───┬───┐  │
# │  │ a │ b │ c │  │ w₁│ w₂│ w₃│  │
# │  ├───┼───┼───┤  ├───┼───┼───┤  │
# │  │ d │ e │ f │ ×│ w₄│ w₅│ w₆│  │
# │  ├───┼───┼───┤  ├───┼───┼───┤  │
# │  │ g │ h │ i │  │ w₇│ w₈│ w₉│  │
# │  └───┴───┴───┘  └───┴───┴───┘  │
# │  output = a·w₁ + b·w₂ + ... + i·w₉  │
# └──────────────────────────────────┘
# ```
# 
# The 7 nested loops iterate over:
# - batch (each image in the batch)
# - output channel (each filter)
# - output height and width (each spatial position)
# - kernel height, kernel width, and input channel (the dot product)

# %%

#| export

class Conv2dBackward(Function):
    """
    Gradient computation for 2D convolution.

    Computes gradients for Conv2d backward pass:
    - grad_input: gradient w.r.t. input (for backprop to previous layer)
    - grad_weight: gradient w.r.t. filters (for weight updates)
    - grad_bias: gradient w.r.t. bias (for bias updates)

    This uses explicit loops to show the gradient computation, matching
    the educational approach of the forward pass.
    """

    def __init__(self, x, weight, bias, stride, padding, kernel_size, padded_shape):
        # Register all tensors that need gradients with autograd
        if bias is not None:
            super().__init__(x, weight, bias)
        else:
            super().__init__(x, weight)
        self.x = x
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.padded_shape = padded_shape

    def apply(self, grad_output):
        """
        Compute gradients for convolution inputs and parameters.

        Args:
            grad_output: Gradient flowing back from next layer
                        Shape: (batch_size, out_channels, out_height, out_width)

        Returns:
            Tuple of (grad_input, grad_weight, grad_bias)
        """
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = self.x.shape
        kernel_h, kernel_w = self.kernel_size

        # Apply padding to input if needed (for gradient computation)
        if self.padding > 0:
            padded_input = np.pad(self.x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = self.x.data

        # Initialize gradients
        grad_input_padded = np.zeros_like(padded_input)
        grad_weight = np.zeros_like(self.weight.data)
        grad_bias = None if self.bias is None else np.zeros_like(self.bias.data)

        # Compute gradients using explicit loops (educational approach)
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Position in input
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Gradient value flowing back to this position
                        grad_val = grad_output[b, out_ch, out_h, out_w]

                        # Distribute gradient to weight and input
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    # Input position
                                    in_h = in_h_start + k_h
                                    in_w = in_w_start + k_w

                                    # Gradient w.r.t. weight
                                    grad_weight[out_ch, in_ch, k_h, k_w] += (
                                        padded_input[b, in_ch, in_h, in_w] * grad_val
                                    )

                                    # Gradient w.r.t. input
                                    grad_input_padded[b, in_ch, in_h, in_w] += (
                                        self.weight.data[out_ch, in_ch, k_h, k_w] * grad_val
                                    )

        # Compute gradient w.r.t. bias (sum over batch and spatial dimensions)
        if grad_bias is not None:
            for out_ch in range(out_channels):
                grad_bias[out_ch] = grad_output[:, out_ch, :, :].sum()

        # Remove padding from input gradient
        if self.padding > 0:
            grad_input = grad_input_padded[:, :,
                                          self.padding:-self.padding,
                                          self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        # Return gradients as numpy arrays (autograd system handles storage)
        # Following TinyTorch protocol: return (grad_input, grad_weight, grad_bias)
        return grad_input, grad_weight, grad_bias

#| export

class Conv2d:
    """
    2D Convolution layer for spatial feature extraction.

    Implements convolution with explicit loops to demonstrate
    computational complexity and memory access patterns.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output feature maps
        kernel_size: Size of convolution kernel (int or tuple)
        stride: Stride of convolution (default: 1)
        padding: Zero-padding added to input (default: 0)
        bias: Whether to add learnable bias (default: True)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        Initialize Conv2d layer with proper weight initialization.

        TODO: Complete Conv2d initialization

        APPROACH:
        1. Store hyperparameters (channels, kernel_size, stride, padding)
        2. Initialize weights using He initialization for ReLU compatibility
        3. Initialize bias (if enabled) to zeros
        4. Use proper shapes: weight (out_channels, in_channels, kernel_h, kernel_w)

        WEIGHT INITIALIZATION:
        - He init: std = sqrt(2 / (in_channels * kernel_h * kernel_w))
        - This prevents vanishing/exploding gradients with ReLU

        HINT: Convert kernel_size to tuple if it's an integer
        """
        super().__init__()

        ### BEGIN SOLUTION
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError(f"kernel_size should be a integer or tuple, got {kernel_size}.")
        
        weight_std = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        self.weight = Tensor(
            rng.normal(0.0, weight_std, (out_channels, in_channels, *self.kernel_size))
        )

        self.bias = Tensor(np.zeros(out_channels)) if bias else None
        ### END SOLUTION

    def _compute_output_shape(self, in_h, in_w):
        """
        Calculate output spatial dimensions for convolution.

        TODO: Apply the convolution output size formula

        APPROACH:
        1. Apply the standard formula for each spatial dimension:
           out_dim = (in_dim + 2 * padding - kernel_size) // stride + 1
        2. Return (out_height, out_width) as a tuple

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        >>> oh, ow = conv._compute_output_shape(32, 32)
        >>> print(oh, ow)  # 32, 32 (same padding preserves size)

        >>> conv2 = Conv2d(3, 16, kernel_size=3, padding=0, stride=1)
        >>> oh, ow = conv2._compute_output_shape(32, 32)
        >>> print(oh, ow)  # 30, 30 (shrinks by kernel_size - 1)

        HINT: The formula is the same for height and width, just with
        different input dimensions.
        """
        ### BEGIN SOLUTION
        kernel_h, kernel_w = self.kernel_size
        
        out_h = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_h, out_w
        ### END SOLUTION

    def _apply_padding(self, x_data):
        """
        Zero-pad the spatial dimensions of the input numpy array.

        TODO: Add zero-padding around spatial dimensions (height, width)

        APPROACH:
        1. If self.padding > 0, use np.pad to add zeros around spatial dims
        2. Only pad dimensions 2 and 3 (height, width), not batch or channels
        3. If self.padding == 0, return the input unchanged

        EXAMPLE:
        >>> conv = Conv2d(1, 1, kernel_size=3, padding=1)
        >>> x = np.ones((1, 1, 3, 3))
        >>> padded = conv._apply_padding(x)
        >>> print(padded.shape)  # (1, 1, 5, 5) -- 3+2*1=5

        HINT: np.pad takes a tuple of (before, after) pairs per dimension.
        Use (0,0) for batch and channel dims, (padding, padding) for spatial.
        """
        ### BEGIN SOLUTION
        padding = self.padding
        
        if padding > 0:
            padded = np.pad(
                x_data, 
                ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                mode='constant', constant_values=0.0
            )
            return padded
        else:
            return x_data
        ### END SOLUTION

    def _convolve_loops(self, padded, batch_size, out_h, out_w):
        """
        The core convolution: sliding window dot products over the input.

        TODO: Implement the nested loop convolution

        APPROACH:
        1. Initialize output array of shape (batch_size, out_channels, out_h, out_w)
        2. Loop over: batch, output channel, output row, output column
        3. For each output position, accumulate the dot product over:
           kernel height, kernel width, and input channels
        4. Store the accumulated sum at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    conv_sum += padded[b, in_ch, ...] * weight[out_ch, in_ch, ...]
                        output[b, out_ch, oh, ow] = conv_sum

        HINT: The input position for kernel element (k_h, k_w) at output
        position (oh, ow) with stride s is: (oh * s + k_h, ow * s + k_w).
        """
        ### BEGIN SOLUTION
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_size
        s = self.stride
        
        weight = self.weight.data
        bias = None if self.bias is None else self.bias.data

        output = np.empty((batch_size, out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih_slice = slice(oh * s, oh * s + kernel_h)
                    iw_slice = slice(ow * s, ow * s + kernel_w)
                    
                    patch = padded[b, :, ih_slice, iw_slice] # shape: (in_channels, kernel_h, kernel_w)
                    conv_sums = np.sum(patch * weight, axis=(1, 2, 3))
                    # conv_sums = np.einsum('ihw,oihw->o', patch, weight)

                    if bias is not None:
                        conv_sums += bias
                    output[b, :, oh, ow] = conv_sums

        return output
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through Conv2d layer.

        This method composes four steps:
        1. Validate input is 4D (shared helper)
        2. Compute output spatial dimensions
        3. Pad input if needed
        4. Run the sliding window convolution loops
        5. Add bias and attach gradient tracking

        Each step is a separate helper you implement below.
        See the individual helper docstrings for details.

        EXAMPLE:
        >>> conv = Conv2d(3, 16, kernel_size=3, padding=1)
        >>> x = Tensor(rng.standard_normal((2, 3, 32, 32)))  # batch=2, RGB, 32x32
        >>> out = conv(x)
        >>> print(out.shape)  # Should be (2, 16, 32, 32)
        """
        ### BEGIN SOLUTION
        validate_4d_input(x, "Conv2d")
        
        batch_size, in_channels, in_h, in_w = x.shape
        out_h, out_w = self._compute_output_shape(in_h, in_w)
        
        padded = self._apply_padding(x.data)
        output = self._convolve_loops(padded, batch_size, out_h, out_w)
        return Tensor(output)
        ### END SOLUTION

    def parameters(self):
        """Return trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
# ### 🧪 Unit Test: Conv2d Output Shape Computation
# 
# This test validates that `_compute_output_shape` correctly applies the
# convolution output dimension formula.
# 
# ```
# Output size formula:
#   out = (in + 2 * padding - kernel) // stride + 1
# 
# Examples:
#   (32 + 2*1 - 3) // 1 + 1 = 32  (same padding)
#   (32 + 2*0 - 3) // 1 + 1 = 30  (no padding)
#   (32 + 2*0 - 3) // 2 + 1 = 15  (stride 2)
# ```
# 
# **What we're testing**: Output dimension formula for various configurations
# **Why it matters**: Wrong dimensions cause silent shape bugs in CNNs
# **Expected**: Matches hand-calculated values

# %%


def test_unit_conv2d_output_shape():
    """Test Conv2d._compute_output_shape for various configurations."""
    print("Testing Conv2d output shape computation...")

    # Same padding: output == input
    conv_same = Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
    oh, ow = conv_same._compute_output_shape(32, 32)
    assert (oh, ow) == (32, 32), f"Same padding: expected (32, 32), got ({oh}, {ow})"

    # No padding: output shrinks by (kernel - 1)
    conv_no_pad = Conv2d(3, 16, kernel_size=3, padding=0, stride=1)
    oh, ow = conv_no_pad._compute_output_shape(32, 32)
    assert (oh, ow) == (30, 30), f"No padding: expected (30, 30), got ({oh}, {ow})"

    # Stride 2: output roughly halves
    conv_stride = Conv2d(3, 16, kernel_size=3, padding=0, stride=2)
    oh, ow = conv_stride._compute_output_shape(32, 32)
    assert (oh, ow) == (15, 15), f"Stride 2: expected (15, 15), got ({oh}, {ow})"

    # Non-square input
    conv_rect = Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
    oh, ow = conv_rect._compute_output_shape(28, 14)
    assert (oh, ow) == (28, 14), f"Rectangular: expected (28, 14), got ({oh}, {ow})"

    # Larger kernel
    conv_5x5 = Conv2d(3, 16, kernel_size=5, padding=0, stride=1)
    oh, ow = conv_5x5._compute_output_shape(32, 32)
    assert (oh, ow) == (28, 28), f"5x5 kernel: expected (28, 28), got ({oh}, {ow})"

    print("Conv2d output shape computation works correctly!")

if __name__ == "__main__":
    test_unit_conv2d_output_shape()

# %% [markdown]
# ### 🧪 Unit Test: Conv2d Padding
# 
# This test validates that `_apply_padding` correctly zero-pads the spatial
# dimensions while leaving batch and channel dimensions untouched.
# 
# ```
# Before padding (1, 1, 3, 3):       After padding=1 (1, 1, 5, 5):
#                                     ┌─────────────────┐
# ┌─────────┐                         │ 0  0  0  0  0   │
# │ 1  2  3 │                         │ 0  1  2  3  0   │
# │ 4  5  6 │          -->            │ 0  4  5  6  0   │
# │ 7  8  9 │                         │ 0  7  8  9  0   │
# └─────────┘                         │ 0  0  0  0  0   │
#                                     └─────────────────┘
# ```
# 
# **What we're testing**: Zero-padding adds correct borders to spatial dims
# **Why it matters**: Padding controls whether convolution preserves spatial size
# **Expected**: Padded array has correct shape and zero borders

# %%


def test_unit_conv2d_padding():
    """Test Conv2d._apply_padding for zero-padding behavior."""
    print("Testing Conv2d padding...")

    # No padding: input unchanged
    conv_no_pad = Conv2d(1, 1, kernel_size=3, padding=0)
    x = np.ones((1, 1, 4, 4))
    result = conv_no_pad._apply_padding(x)
    assert result.shape == (1, 1, 4, 4), f"No-pad shape: expected (1,1,4,4), got {result.shape}"
    assert np.array_equal(result, x), "No-pad should return input unchanged"

    # Padding=1: adds 1 pixel border of zeros
    conv_pad1 = Conv2d(1, 1, kernel_size=3, padding=1)
    x = np.ones((1, 1, 3, 3))
    result = conv_pad1._apply_padding(x)
    assert result.shape == (1, 1, 5, 5), f"Pad-1 shape: expected (1,1,5,5), got {result.shape}"
    # Check that borders are zero
    assert np.all(result[:, :, 0, :] == 0), "Top border should be zeros"
    assert np.all(result[:, :, -1, :] == 0), "Bottom border should be zeros"
    assert np.all(result[:, :, :, 0] == 0), "Left border should be zeros"
    assert np.all(result[:, :, :, -1] == 0), "Right border should be zeros"
    # Check that center is preserved
    assert np.all(result[:, :, 1:4, 1:4] == 1), "Center should be preserved"

    # Padding=2: adds 2 pixel border
    conv_pad2 = Conv2d(1, 1, kernel_size=5, padding=2)
    x = np.ones((2, 3, 4, 4))
    result = conv_pad2._apply_padding(x)
    assert result.shape == (2, 3, 8, 8), f"Pad-2 shape: expected (2,3,8,8), got {result.shape}"
    # Batch and channel dims unchanged
    assert result.shape[0] == 2, "Batch dim should be unchanged"
    assert result.shape[1] == 3, "Channel dim should be unchanged"

    print("Conv2d padding works correctly!")

if __name__ == "__main__":
    test_unit_conv2d_padding()

# %% [markdown]
# ### 🧪 Unit Test: Conv2d Convolution Loops
# 
# This test validates the core sliding window computation in `_convolve_loops`.
# 
# ```
# Convolution = Sliding Window Dot Products:
# ┌──────────────────────────────────┐
# │ For EACH output position:        │
# │  Input patch     Kernel          │
# │  ┌───┬───┬───┐  ┌───┬───┬───┐  │
# │  │ a │ b │ c │  │ w₁│ w₂│ w₃│  │
# │  ├───┼───┼───┤  ├───┼───┼───┤  │
# │  │ d │ e │ f │ ×│ w₄│ w₅│ w₆│  │
# │  ├───┼───┼───┤  ├───┼───┼───┤  │
# │  │ g │ h │ i │  │ w₇│ w₈│ w₉│  │
# │  └───┴───┴───┘  └───┴───┴───┘  │
# │  output = a·w₁ + b·w₂ + ... + i·w₉  │
# └──────────────────────────────────┘
# ```
# 
# **What we're testing**: The 7-nested-loop convolution produces correct values
# **Why it matters**: This is THE core operation of computer vision
# **Expected**: Output matches hand-computed convolution results

# %%


def test_unit_conv2d_convolve_loops():
    """Test Conv2d._convolve_loops with known input/weight values."""
    print("Testing Conv2d convolution loops...")

    # Create a Conv2d with known weights (1 input channel, 1 output channel, 2x2 kernel)
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    # Set weights to known values: [[1, 0], [0, 1]] (identity-like kernel)
    conv.weight = Tensor(np.array([[[[1.0, 0.0],
                                      [0.0, 1.0]]]]), requires_grad=True)

    # Input: 1 batch, 1 channel, 3x3
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    padded = np.array([[[[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]]]])

    # Output should be 2x2 (no padding):
    # pos(0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
    # pos(0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
    # pos(1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
    # pos(1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
    output = conv._convolve_loops(padded, batch_size=1, out_h=2, out_w=2)

    expected = np.array([[[[6.0, 8.0],
                            [12.0, 14.0]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test with multiple output channels
    conv2 = Conv2d(in_channels=1, out_channels=2, kernel_size=2, bias=False)
    # Channel 0: all ones kernel, Channel 1: all twos kernel
    conv2.weight = Tensor(np.array([[[[1.0, 1.0], [1.0, 1.0]]],
                                     [[[2.0, 2.0], [2.0, 2.0]]]]), requires_grad=True)

    output2 = conv2._convolve_loops(padded, batch_size=1, out_h=2, out_w=2)

    # Channel 0 (all-ones kernel): sum of each 2x2 window
    # pos(0,0): 1+2+4+5=12, pos(0,1): 2+3+5+6=16
    # pos(1,0): 4+5+7+8=24, pos(1,1): 5+6+8+9=28
    expected_ch0 = np.array([[12.0, 16.0], [24.0, 28.0]])
    expected_ch1 = expected_ch0 * 2  # All-twos kernel = 2x all-ones
    assert np.allclose(output2[0, 0], expected_ch0), f"Channel 0 mismatch"
    assert np.allclose(output2[0, 1], expected_ch1), f"Channel 1 mismatch"

    print("Conv2d convolution loops work correctly!")

if __name__ == "__main__":
    test_unit_conv2d_convolve_loops()

# %% [markdown]
# ### 🧪 Unit Test: Conv2d Forward (Composition)
# 
# This test validates the complete `forward` method that composes all helpers
# together: validation, shape computation, padding, convolution loops, bias,
# and gradient tracking.
# 
# **What we're testing**: End-to-end Conv2d with shape preservation, padding, stride
# **Why it matters**: Convolution is the foundation of computer vision
# **Expected**: Correct output shapes and reasonable value ranges

# %%


def test_unit_conv2d():
    """Test Conv2d forward pass with multiple configurations."""
    print("Testing Conv2d...")

    # Test 1: Basic convolution without padding
    print("  Testing basic convolution...")
    conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    x1 = Tensor(rng.standard_normal((2, 3, 32, 32)))
    out1 = conv1(x1)

    expected_h = (32 - 3) + 1  # 30
    expected_w = (32 - 3) + 1  # 30
    assert out1.shape == (2, 16, expected_h, expected_w), f"Expected (2, 16, 30, 30), got {out1.shape}"

    # Test 2: Convolution with padding (same size)
    print("  Testing convolution with padding...")
    conv2 = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x2 = Tensor(rng.standard_normal((1, 3, 28, 28)))
    out2 = conv2(x2)

    # With padding=1, output should be same size as input
    assert out2.shape == (1, 8, 28, 28), f"Expected (1, 8, 28, 28), got {out2.shape}"

    # Test 3: Convolution with stride
    print("  Testing convolution with stride...")
    conv3 = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2)
    x3 = Tensor(rng.standard_normal((1, 1, 16, 16)))
    out3 = conv3(x3)

    expected_h = (16 - 3) // 2 + 1  # 7
    expected_w = (16 - 3) // 2 + 1  # 7
    assert out3.shape == (1, 4, expected_h, expected_w), f"Expected (1, 4, 7, 7), got {out3.shape}"

    # Test 4: Parameter counting
    print("  Testing parameter counting...")
    conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=True)
    params = conv4.parameters()

    # Weight: (128, 64, 3, 3) = 73,728 parameters
    # Bias: (128,) = 128 parameters
    # Total: 73,856 parameters
    weight_params = 128 * 64 * 3 * 3
    bias_params = 128
    total_params = weight_params + bias_params

    actual_weight_params = np.prod(conv4.weight.shape)
    actual_bias_params = np.prod(conv4.bias.shape) if conv4.bias is not None else 0
    actual_total = actual_weight_params + actual_bias_params

    assert actual_total == total_params, f"Expected {total_params} parameters, got {actual_total}"
    assert len(params) == 2, f"Expected 2 parameter tensors, got {len(params)}"

    # Test 5: No bias configuration
    print("  Testing no bias configuration...")
    conv5 = Conv2d(in_channels=3, out_channels=16, kernel_size=5, bias=False)
    params5 = conv5.parameters()
    assert len(params5) == 1, f"Expected 1 parameter tensor (no bias), got {len(params5)}"
    assert conv5.bias is None, "Bias should be None when bias=False"

    print("✅ Conv2d works correctly!")

if __name__ == "__main__":
    test_unit_conv2d()

# %% [markdown]
# ## 🏗️ Pooling Operations - Spatial Dimension Reduction
# 
# Pooling operations compress spatial information while keeping the most important features. Think of them as creating "thumbnail summaries" of local regions.
# 
# ### MaxPool2d: Keeping the Strongest Signals
# 
# Max pooling finds the strongest activation in each window, preserving sharp features like edges and corners.
# 
# ```
# MaxPool2d Example (2×2 kernel, stride=2):
# Input (4×4):              Windows:               Output (2×2):
# ┌─────────────┐          ┌─────┬─────┐          ┌───────┐
# │ 1  3 │ 2  8 │          │ 1 3 │ 2 8 │          │ 6   8 │
# │ 5  6 │ 7  4 │    →     │ 5 6 │ 7 4 │    →     │ 9   7 │
# ├──────┼──────┤          ├─────┼─────┤          └───────┘
# │ 2  9 │ 1  7 │          │ 2 9 │ 1 7 │
# │ 0  1 │ 3  6 │          │ 0 1 │ 3 6 │
# └─────────────┘          └─────┴─────┘
# 
# Window Computations:
# Top-left: max(1,3,5,6) = 6     Top-right: max(2,8,7,4) = 8
# Bottom-left: max(2,9,0,1) = 9  Bottom-right: max(1,7,3,6) = 7
# ```
# 
# ### AvgPool2d: Smoothing Local Features
# 
# Average pooling computes the mean of each window, creating smoother, more general features.
# 
# ```
# AvgPool2d Example (same 2×2 kernel, stride=2):
# Input (4×4):              Output (2×2):
# ┌─────────────┐          ┌─────────────┐
# │ 1  3 │ 2  8 │          │ 3.75   5.25 │
# │ 5  6 │ 7  4 │    →     │ 3.0    4.25 │
# ├──────┼──────┤          └─────────────┘
# │ 2  9 │ 1  7 │
# │ 0  1 │ 3  6 │
# └─────────────┘
# 
# Window Computations:
# Top-left: (1+3+5+6)/4 = 3.75    Top-right: (2+8+7+4)/4 = 5.25
# Bottom-left: (2+9+0+1)/4 = 3.0  Bottom-right: (1+7+3+6)/4 = 4.25
# ```
# 
# ### Why Pooling Matters for Computer Vision
# 
# ```
# Memory Impact:
# Input: 224×224×64 = 3.2M values    After 2×2 pooling: 112×112×64 = 0.8M values
# Memory reduction: 4× less!         Computation reduction: 4× less!
# 
# Information Trade-off:
# ✅ Preserves important features     ⚠️ Loses fine spatial detail
# ✅ Provides translation invariance  ⚠️ Reduces localization precision
# ✅ Reduces overfitting             ⚠️ May lose small objects
# ```
# 
# ### Sliding Window Pattern
# 
# Both pooling operations follow the same sliding window pattern:
# 
# ```
# Sliding 2×2 window with stride=2:
# Step 1:     Step 2:     Step 3:     Step 4:
# ┌──┐        ┌──┐
# │▓▓│        │▓▓│
# └──┘        └──┘                    ┌──┐        ┌──┐
#                                     │▓▓│        │▓▓│
#                                     └──┘        └──┘
# 
# Non-overlapping windows → Each input pixel used exactly once
# Stride=2 → Output dimensions halved in each direction
# ```
# 
# The key difference: MaxPool takes max(window), AvgPool takes mean(window).

# %% [markdown]
# ### MaxPool2d Implementation - Preserving Strong Features
# 
# MaxPool2d finds the strongest activation in each spatial window, creating a compressed representation that keeps the most important information.
# 
# #### Why Max Pooling Works for Computer Vision
# 
# ```
# Edge Detection Example:
# Input Window (2×2):         Max Pooling Result:
# ┌─────┬─────┐
# │ 0.1 │ 0.8 │ ←  Strong edge signal
# ├─────┼─────┤
# │ 0.2 │ 0.1 │              Output: 0.8 (preserves edge)
# └─────┴─────┘
# 
# Noise Reduction Example:
# Input Window (2×2):
# ┌─────┬─────┐
# │ 0.9 │ 0.1 │ ←  Feature + noise
# ├─────┼─────┤
# │ 0.2 │ 0.1 │              Output: 0.9 (removes noise)
# └─────┴─────┘
# ```
# 
# #### The Sliding Window Pattern
# 
# ```
# MaxPool with 2×2 kernel, stride=2:
# 
# Input (4×4):                Output (2×2):
# ┌───┬───┬───┬───┐          ┌───────┬───────┐
# │ a │ b │ c │ d │          │max(a,b│max(c,d│
# ├───┼───┼───┼───┤     →    │   e,f)│   g,h)│
# │ e │ f │ g │ h │          ├───────┼───────┤
# ├───┼───┼───┼───┤          │max(i,j│max(k,l│
# │ i │ j │ k │ l │          │   m,n)│   o,p)│
# ├───┼───┼───┼───┤          └───────┴───────┘
# │ m │ n │ o │ p │
# └───┴───┴───┴───┘
# 
# Benefits:
# ✓ Translation invariance (cat moved 1 pixel still detected)
# ✓ Computational efficiency (4× fewer values to process)
# ✓ Hierarchical feature building (next layer sees larger receptive field)
# ```
# 
# #### Memory and Computation Impact
# 
# For input (1, 64, 224, 224) with 2×2 pooling:
# - **Input memory**: 64 × 224 × 224 × 4 bytes = 12.8 MB
# - **Output memory**: 64 × 112 × 112 × 4 bytes = 3.2 MB
# - **Memory reduction**: 4× less memory needed
# - **Computation**: No parameters, minimal compute cost

# %%

#| export

class MaxPool2dBackward(Function):
    """
    Gradient computation for 2D max pooling.

    Max pooling gradients flow only to the positions that were selected
    as the maximum in the forward pass.
    """

    def __init__(self, x, output_shape, kernel_size, stride, padding):
        super().__init__(x)
        self.x = x
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Store max positions for gradient routing
        self.max_positions = {}

    def apply(self, grad_output):
        """
        Route gradients back to max positions.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient w.r.t. input
        """
        batch_size, channels, in_height, in_width = self.x.shape
        _, _, out_height, out_width = self.output_shape
        kernel_h, kernel_w = self.kernel_size

        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(self.x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=-np.inf)
            grad_input_padded = np.zeros_like(padded_input)
        else:
            padded_input = self.x.data
            grad_input_padded = np.zeros_like(self.x.data)

        # Route gradients to max positions
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Find max position in this window
                        max_val = -np.inf
                        max_h, max_w = 0, 0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                in_h = in_h_start + k_h
                                in_w = in_w_start + k_w
                                val = padded_input[b, c, in_h, in_w]
                                if val > max_val:
                                    max_val = val
                                    max_h, max_w = in_h, in_w

                        # Route gradient to max position
                        grad_input_padded[b, c, max_h, max_w] += grad_output[b, c, out_h, out_w]

        # Remove padding
        if self.padding > 0:
            grad_input = grad_input_padded[:, :,
                                          self.padding:-self.padding,
                                          self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        # Return as tuple (following Function protocol)
        return (grad_input,)

#| export

class MaxPool2d:
    """
    2D Max Pooling layer for spatial dimension reduction.

    Applies maximum operation over spatial windows, preserving
    the strongest activations while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize MaxPool2d layer.

        TODO: Store pooling parameters

        APPROACH:
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided (non-overlapping)
        3. Store padding parameter

        HINT: Default stride equals kernel_size for non-overlapping windows
        """
        super().__init__()

        ### BEGIN SOLUTION
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            self.stride = kernel_size if stride is None else stride
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
            self.stride = kernel_size[0] if stride is None else stride
        else:
            raise TypeError(f"kernel_size should be a integer or tuple, got {kernel_size}.")

        self.padding = padding
        ### END SOLUTION

    def _compute_pool_output_shape(self, in_h, in_w):
        """
        Calculate output spatial dimensions for pooling.

        TODO: Apply the pooling output size formula

        APPROACH:
        1. Apply the same formula as convolution:
           out_dim = (in_dim + 2 * padding - kernel_size) // stride + 1
        2. Return (out_height, out_width) as a tuple

        EXAMPLE:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> oh, ow = pool._compute_pool_output_shape(8, 8)
        >>> print(oh, ow)  # 4, 4 (halved)

        HINT: This formula is identical to convolution's output shape formula.
        """
        ### BEGIN SOLUTION
        kernel_h, kernel_w = self.kernel_size
        
        out_h = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_h, out_w
        ### END SOLUTION

    def _maxpool_loops(self, padded, batch_size, channels, out_h, out_w):
        """
        The core max pooling: find maximum value in each window.

        TODO: Implement the nested loop max pooling

        APPROACH:
        1. Initialize output array of shape (batch_size, channels, out_h, out_w)
        2. Loop over: batch, channel, output row, output column
        3. For each output position, scan the kernel window to find the maximum
        4. Store the maximum at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        max_val = -infinity
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                max_val = max(max_val, padded[b, c, ...])
                        output[b, c, oh, ow] = max_val

        HINT: Initialize max_val to -np.inf so any real value is larger.
        The input position is (oh * stride + k_h, ow * stride + k_w).
        """
        ### BEGIN SOLUTION
        kernel_h, kernel_w = self.kernel_size
        s = self.stride
        
        output = np.empty((batch_size, channels, out_h, out_w))
        
        for b in range(batch_size):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih_slice = slice(oh * s, oh * s + kernel_h)
                    iw_slice = slice(ow * s, ow * s + kernel_w)
                    
                    patch = padded[b, :, ih_slice, iw_slice] # shape: (channels, kernel_h, kernel_w)
                    max_vals = np.max(patch, axis=(1, 2))

                    output[b, :, oh, ow] = max_vals

        return output
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through MaxPool2d layer.

        This method composes the helpers:
        1. Validate input is 4D
        2. Compute output dimensions
        3. Pad input (with -inf for max pooling)
        4. Run the max pooling loops
        5. Attach gradient tracking

        EXAMPLE:
        >>> pool = MaxPool2d(kernel_size=2, stride=2)
        >>> x = Tensor(rng.standard_normal((1, 3, 8, 8)))
        >>> out = pool(x)
        >>> print(out.shape)  # Should be (1, 3, 4, 4)
        """
        ### BEGIN SOLUTION
        validate_4d_input(x, "MaxPool2d")
        
        batch_size, channels, in_h, in_w = x.shape
        out_h, out_w = self._compute_pool_output_shape(in_h, in_w)
        
        padding = self.padding
        if padding > 0:
            padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                mode='constant', constant_values=-np.inf
            )
        else:
            padded = x.data
        
        output = self._maxpool_loops(padded, batch_size, channels, out_h, out_w)
        return Tensor(output)
        ### END SOLUTION

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
# ### Unit Test: MaxPool2d Output Shape
# 
# This test validates that `_compute_pool_output_shape` correctly computes
# the spatial dimensions after max pooling.
# 
# ```
# Output size formula (same as convolution):
#   out = (in + 2 * padding - kernel) // stride + 1
# 
# Common case: kernel=2, stride=2, padding=0
#   (8 + 0 - 2) // 2 + 1 = 4  (spatial dimensions halved)
# ```
# 
# **What we're testing**: Pooling output dimension calculation
# **Why it matters**: Wrong dimensions break the CNN dimension chain
# **Expected**: Matches hand-calculated values for various configs

# %%


def test_unit_maxpool2d_output_shape():
    """Test MaxPool2d._compute_pool_output_shape."""
    print("Testing MaxPool2d output shape computation...")

    # Standard 2x2 pooling with stride 2: halves dimensions
    pool = MaxPool2d(kernel_size=2, stride=2)
    oh, ow = pool._compute_pool_output_shape(8, 8)
    assert (oh, ow) == (4, 4), f"2x2 stride 2: expected (4, 4), got ({oh}, {ow})"

    # Non-square input
    oh, ow = pool._compute_pool_output_shape(16, 8)
    assert (oh, ow) == (8, 4), f"Non-square: expected (8, 4), got ({oh}, {ow})"

    # Overlapping pooling: kernel=3, stride=1
    pool_overlap = MaxPool2d(kernel_size=3, stride=1)
    oh, ow = pool_overlap._compute_pool_output_shape(5, 5)
    assert (oh, ow) == (3, 3), f"Overlapping: expected (3, 3), got ({oh}, {ow})"

    # Large kernel
    pool_large = MaxPool2d(kernel_size=4, stride=4)
    oh, ow = pool_large._compute_pool_output_shape(16, 16)
    assert (oh, ow) == (4, 4), f"4x4 stride 4: expected (4, 4), got ({oh}, {ow})"

    print("MaxPool2d output shape computation works correctly!")

if __name__ == "__main__":
    test_unit_maxpool2d_output_shape()

# %% [markdown]
# ### Unit Test: MaxPool2d Loops
# 
# This test validates that `_maxpool_loops` correctly finds the maximum
# value in each pooling window.
# 
# ```
# MaxPool2d sliding window (2x2, stride 2):
# ┌─────┬─────┐    ┌─────┐
# │ 1 3 │ 2 8 │    │ 6 8 │
# │ 5 6 │ 7 4 │ -> │ 9 7 │
# ├─────┼─────┤    └─────┘
# │ 2 9 │ 1 7 │
# │ 0 1 │ 3 6 │
# └─────┴─────┘
# ```
# 
# **What we're testing**: The max-finding loops produce correct values
# **Why it matters**: Max pooling preserves the strongest activations
# **Expected**: Output matches hand-computed max values per window

# %%


def test_unit_maxpool2d_loops():
    """Test MaxPool2d._maxpool_loops with known values."""
    print("Testing MaxPool2d loops...")

    pool = MaxPool2d(kernel_size=2, stride=2)

    # Known 4x4 input
    padded = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]])

    output = pool._maxpool_loops(padded, batch_size=1, channels=1, out_h=2, out_w=2)

    # Window maxes:
    # top-left: max(1,2,5,6) = 6
    # top-right: max(3,4,7,8) = 8
    # bottom-left: max(9,10,13,14) = 14
    # bottom-right: max(11,12,15,16) = 16
    expected = np.array([[[[6.0, 8.0], [14.0, 16.0]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test with negative values
    padded_neg = np.array([[[[-5.0, -1.0],
                              [-3.0, -2.0]]]])
    pool_small = MaxPool2d(kernel_size=2, stride=2)
    output_neg = pool_small._maxpool_loops(padded_neg, 1, 1, 1, 1)
    assert output_neg[0, 0, 0, 0] == -1.0, f"Max of negatives: expected -1.0, got {output_neg[0,0,0,0]}"

    print("MaxPool2d loops work correctly!")

if __name__ == "__main__":
    test_unit_maxpool2d_loops()

# %% [markdown]
# ### AvgPool2d Implementation - Smoothing and Generalizing Features
# 
# AvgPool2d computes the average of each spatial window, creating smoother features that are less sensitive to noise and exact pixel positions.
# 
# #### MaxPool vs AvgPool: Different Philosophies
# 
# ```
# Same Input Window (2×2):    MaxPool Output:    AvgPool Output:
# ┌─────┬─────┐
# │ 0.1 │ 0.9 │               0.9              0.425
# ├─────┼─────┤              (max)             (mean)
# │ 0.3 │ 0.3 │
# └─────┴─────┘
# 
# Interpretation:
# MaxPool: "What's the strongest feature here?"
# AvgPool: "What's the general feature level here?"
# ```
# 
# #### When to Use Average Pooling
# 
# ```
# Use Cases:
# ✓ Global Average Pooling (GAP) for classification
# ✓ When you want smoother, less noisy features
# ✓ When exact feature location doesn't matter
# ✓ In shallower networks where sharp features aren't critical
# 
# Typical Pattern:
# Feature Maps → Global Average Pool → Dense → Classification
# (256×7×7)   →        (256×1×1)      → FC   →    (10)
#               Replaces flatten+dense with parameter reduction
# ```
# 
# #### Mathematical Implementation
# 
# ```
# Average Pooling Computation:
# Window: [a, b]    Result = (a + b + c + d) / 4
#         [c, d]
# 
# For efficiency, we:
# 1. Sum all values in window: window_sum = a + b + c + d
# 2. Divide by window area: result = window_sum / (kernel_h × kernel_w)
# 3. Store result at output position
# 
# Memory access pattern identical to MaxPool, just different aggregation!
# ```
# 
# #### Practical Considerations
# 
# - **Memory**: Same 4× reduction as MaxPool
# - **Computation**: Slightly more expensive (sum + divide vs max)
# - **Features**: Smoother, more generalized than MaxPool
# - **Use**: Often in final layers (Global Average Pooling) to reduce parameters

# %%

#| export

class AvgPool2d:
    """
    2D Average Pooling layer for spatial dimension reduction.

    Applies average operation over spatial windows, smoothing
    features while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initialize AvgPool2d layer.

        TODO: Store pooling parameters (same as MaxPool2d)

        APPROACH:
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided
        3. Store padding parameter
        """
        super().__init__()

        ### BEGIN SOLUTION
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            self.stride = kernel_size if stride is None else stride
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
            self.stride = kernel_size[0] if stride is None else stride
        else:
            raise TypeError(f"kernel_size should be a integer or tuple, got {kernel_size}.")

        self.padding = padding
        ### END SOLUTION

    def _compute_pool_output_shape(self, in_h, in_w):
        """
        Calculate output spatial dimensions for pooling.

        TODO: Apply the pooling output size formula

        APPROACH:
        1. Apply the standard formula for each spatial dimension:
           out_dim = (in_dim + 2 * padding - kernel_size) // stride + 1
        2. Return (out_height, out_width) as a tuple

        EXAMPLE:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> oh, ow = pool._compute_pool_output_shape(8, 8)
        >>> print(oh, ow)  # 4, 4 (halved)

        HINT: This formula is identical to MaxPool2d and Conv2d output shapes.
        """
        ### BEGIN SOLUTION
        kernel_h, kernel_w = self.kernel_size
        
        out_h = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_h, out_w
        ### END SOLUTION

    def _avgpool_loops(self, padded, batch_size, channels, out_h, out_w):
        """
        The core average pooling: compute mean of each window.

        TODO: Implement the nested loop average pooling

        APPROACH:
        1. Initialize output array of shape (batch_size, channels, out_h, out_w)
        2. Loop over: batch, channel, output row, output column
        3. For each output position, sum all values in the kernel window
        4. Divide by window area (kernel_h * kernel_w) to get the average
        5. Store the average at the output position

        LOOP STRUCTURE:
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                window_sum += padded[b, c, ...]
                        output[b, c, oh, ow] = window_sum / (kernel_h * kernel_w)

        HINT: Unlike max pooling, you accumulate a sum and then divide.
        The input position is (oh * stride + k_h, ow * stride + k_w).
        """
        ### BEGIN SOLUTION
        kernel_h, kernel_w = self.kernel_size
        s = self.stride
        
        output = np.empty((batch_size, channels, out_h, out_w))
        
        for b in range(batch_size):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih_slice = slice(oh * s, oh * s + kernel_h)
                    iw_slice = slice(ow * s, ow * s + kernel_w)
                    
                    patch = padded[b, :, ih_slice, iw_slice] # shape: (channels, kernel_h, kernel_w)
                    max_vals = np.mean(patch, axis=(1, 2))

                    output[b, :, oh, ow] = max_vals

        return output
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through AvgPool2d layer.

        This method composes the helpers:
        1. Validate input is 4D
        2. Compute output dimensions
        3. Pad input (with zeros for average pooling)
        4. Run the average pooling loops
        5. Return result with gradient tracking

        EXAMPLE:
        >>> pool = AvgPool2d(kernel_size=2, stride=2)
        >>> x = Tensor(rng.standard_normal((1, 3, 8, 8)))
        >>> out = pool(x)
        >>> print(out.shape)  # Should be (1, 3, 4, 4)
        """
        ### BEGIN SOLUTION
        validate_4d_input(x, "MaxPool2d")
        
        batch_size, channels, in_h, in_w = x.shape
        out_h, out_w = self._compute_pool_output_shape(in_h, in_w)
        
        padding = self.padding
        if padding > 0:
            padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                mode='constant', constant_values=0.0
            )
        else:
            padded = x.data
        
        output = self._avgpool_loops(padded, batch_size, channels, out_h, out_w)
        return Tensor(output)
        ### END SOLUTION

    def parameters(self):
        """Return empty list (pooling has no parameters)."""
        return []

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
# ### Unit Test: AvgPool2d Output Shape
# 
# This test validates that `_compute_pool_output_shape` correctly computes
# the spatial dimensions after average pooling.
# 
# **What we're testing**: Pooling output dimension calculation
# **Why it matters**: Must match MaxPool2d formula for interchangeability
# **Expected**: Same results as MaxPool2d for identical configurations

# %%


def test_unit_avgpool2d_output_shape():
    """Test AvgPool2d._compute_pool_output_shape."""
    print("Testing AvgPool2d output shape computation...")

    # Standard 2x2 pooling: halves dimensions
    pool = AvgPool2d(kernel_size=2, stride=2)
    oh, ow = pool._compute_pool_output_shape(8, 8)
    assert (oh, ow) == (4, 4), f"2x2 stride 2: expected (4, 4), got ({oh}, {ow})"

    # Non-square input
    oh, ow = pool._compute_pool_output_shape(16, 8)
    assert (oh, ow) == (8, 4), f"Non-square: expected (8, 4), got ({oh}, {ow})"

    # Overlapping pooling: kernel=3, stride=1
    pool_overlap = AvgPool2d(kernel_size=3, stride=1)
    oh, ow = pool_overlap._compute_pool_output_shape(5, 5)
    assert (oh, ow) == (3, 3), f"Overlapping: expected (3, 3), got ({oh}, {ow})"

    print("AvgPool2d output shape computation works correctly!")

if __name__ == "__main__":
    test_unit_avgpool2d_output_shape()

# %% [markdown]
# ### Unit Test: AvgPool2d Loops
# 
# This test validates that `_avgpool_loops` correctly computes the mean of
# each pooling window.
# 
# ```
# AvgPool2d sliding window (2x2, stride 2):
# ┌─────┬─────┐    ┌───────────┐
# │ 1 2 │ 3 4 │    │ 3.5   5.5 │
# │ 5 6 │ 7 8 │ -> │11.5  13.5 │
# ├─────┼─────┤    └───────────┘
# │ 9 10│11 12│
# │13 14│15 16│
# └─────┴─────┘
# 
# Top-left: (1+2+5+6)/4 = 3.5
# ```
# 
# **What we're testing**: The sum-and-divide loops produce correct averages
# **Why it matters**: Average pooling creates smoother features than max pooling
# **Expected**: Output matches hand-computed averages per window

# %%


def test_unit_avgpool2d_loops():
    """Test AvgPool2d._avgpool_loops with known values."""
    print("Testing AvgPool2d loops...")

    pool = AvgPool2d(kernel_size=2, stride=2)

    # Known 4x4 input
    padded = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]])

    output = pool._avgpool_loops(padded, batch_size=1, channels=1, out_h=2, out_w=2)

    # Window averages:
    # top-left: (1+2+5+6)/4 = 3.5
    # top-right: (3+4+7+8)/4 = 5.5
    # bottom-left: (9+10+13+14)/4 = 11.5
    # bottom-right: (11+12+15+16)/4 = 13.5
    expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test that avg is always <= max for same data
    pool_max = MaxPool2d(kernel_size=2, stride=2)
    max_output = pool_max._maxpool_loops(padded, 1, 1, 2, 2)
    assert np.all(output <= max_output), "Average should always be <= maximum"

    print("AvgPool2d loops work correctly!")

if __name__ == "__main__":
    test_unit_avgpool2d_loops()

# %% [markdown]
# ## 🏗️ Batch Normalization - Stabilizing Deep Network Training
# 
# Batch Normalization (BatchNorm) is one of the most important techniques for training deep networks. It normalizes activations across the batch dimension, dramatically improving training stability and speed.
# 
# ### Why BatchNorm Matters
# 
# ```
# Without BatchNorm:                  With BatchNorm:
# Layer outputs can have              Layer outputs are normalized
# wildly varying scales:              to consistent scale:
# 
# Layer 1: mean=0.5, std=0.3         Layer 1: mean≈0, std≈1
# Layer 5: mean=12.7, std=8.4   →    Layer 5: mean≈0, std≈1
# Layer 10: mean=0.001, std=0.0003   Layer 10: mean≈0, std≈1
# 
# Result: Unstable gradients         Result: Stable training
#         Slow convergence                   Fast convergence
#         Careful learning rate              Robust to hyperparameters
# ```
# 
# ### The BatchNorm Computation
# 
# For each channel c, BatchNorm computes:
# ```
# 1. Batch Statistics (during training):
#    μ_c = mean(x[:, c, :, :])     # Mean over batch and spatial dims
#    σ²_c = var(x[:, c, :, :])     # Variance over batch and spatial dims
# 
# 2. Normalize:
#    x̂_c = (x[:, c, :, :] - μ_c) / sqrt(σ²_c + ε)
# 
# 3. Scale and Shift (learnable parameters):
#    y_c = γ_c * x̂_c + β_c       # γ (gamma) and β (beta) are learned
# ```
# 
# ### Train vs Eval Mode
# 
# This is a critical systems concept:
# 
# ```
# Training Mode:                      Eval Mode:
# ┌────────────────────┐             ┌────────────────────┐
# │ Use batch stats    │             │ Use running stats  │
# │ Update running     │             │ (accumulated from  │
# │ mean/variance      │             │  training)         │
# └────────────────────┘             └────────────────────┘
#    ↓                                  ↓
# Computes μ, σ² from                Uses frozen μ, σ² for
# current batch                      consistent inference
# ```
# 
# **Why this matters**: During inference, you might process just 1 image. Batch statistics from 1 sample would be meaningless. Running statistics provide stable normalization.

# %%

#| export

class BatchNorm2d:
    """
    Batch Normalization for 2D spatial inputs (images).

    Normalizes activations across batch and spatial dimensions for each channel,
    then applies learnable scale (gamma) and shift (beta) parameters.

    Key behaviors:
    - Training: Uses batch statistics, updates running statistics
    - Eval: Uses frozen running statistics for consistent inference

    Args:
        num_features: Number of channels (C in NCHW format)
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics update (default: 0.1)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initialize BatchNorm2d layer.

        TODO: Initialize learnable and running parameters

        APPROACH:
        1. Store hyperparameters (num_features, eps, momentum)
        2. Initialize gamma (scale) to ones - identity at start
        3. Initialize beta (shift) to zeros - no shift at start
        4. Initialize running_mean to zeros
        5. Initialize running_var to ones
        6. Set training mode to True initially

        EXAMPLE:
        >>> bn = BatchNorm2d(64)  # For 64-channel feature maps
        >>> print(bn.gamma.shape)  # (64,)
        >>> print(bn.training)     # True
        """
        super().__init__()

        ### BEGIN SOLUTION
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True
        ### END SOLUTION

    def train(self):
        """Set layer to training mode."""
        self.training = True
        return self

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False
        return self

    def _validate_input(self, x):
        """
        Validate that input tensor has the correct shape for BatchNorm2d.

        TODO: Validate input is 4D with correct channel count

        APPROACH:
        1. Check input is 4D (batch, channels, height, width)
        2. Provide helpful error messages for common mistakes (3D, 2D)
        3. Verify channel dimension matches num_features

        HINTS:
        - Use len(x.shape) to check dimensionality
        - Use ❌ What → 💡 Why → 🔧 Fix error message format
        """
        ### BEGIN SOLUTION
        if len(x.shape) != 4:
            if len(x.shape) == 3:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got 3D: {x.shape}\n"
                    f"  ❌ Missing batch dimension\n"
                    f"  💡 BatchNorm2d computes statistics over the batch dimension\n"
                    f"  🔧 Add batch dim: x.reshape(1, {x.shape[0]}, {x.shape[1]}, {x.shape[2]})"
                )
            elif len(x.shape) == 2:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got 2D: {x.shape}\n"
                    f"  ❌ Got a matrix, expected an image tensor\n"
                    f"  💡 BatchNorm2d normalizes over spatial dimensions per channel\n"
                    f"  🔧 If this is a flattened image, reshape it: x.reshape(1, channels, height, width)"
                )
            else:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch, channels, height, width), got {len(x.shape)}D: {x.shape}\n"
                    f"  ❌ Wrong number of dimensions\n"
                    f"  💡 BatchNorm2d expects: (batch_size, channels, height, width)\n"
                    f"  🔧 Reshape your input to 4D with the correct dimensions"
                )

        batch_size, channels, height, width = x.shape

        if channels != self.num_features:
            raise ValueError(
                f"BatchNorm2d channel mismatch: expected {self.num_features} channels, got {channels}\n"
                f"  ❌ Input has {channels} channels but BatchNorm2d was created for {self.num_features}\n"
                f"  💡 BatchNorm2d(num_features) must match the channel dimension of your input\n"
                f"  🔧 Either fix your input shape or create BatchNorm2d({channels})"
            )
        ### END SOLUTION

    def _get_stats(self, x):
        """
        Get mean and variance for normalization (batch or running stats).

        TODO: Compute or retrieve normalization statistics based on mode

        APPROACH:
        1. If training: compute batch mean/var over axes (0, 2, 3),
           then update running statistics with momentum
        2. If eval: use frozen running_mean and running_var

        HINTS:
        - np.mean(x.data, axis=(0, 2, 3)) gives per-channel mean
        - Running update: running = (1 - momentum) * running + momentum * batch
        """
        ### BEGIN SOLUTION
        if self.training:
            batch_mean = np.mean(x.data, axis=(0, 2, 3))
            batch_var = np.var(x.data, axis=(0, 2, 3))
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            return batch_mean, batch_var
            
        else:
            return self.running_mean, self.running_var
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through BatchNorm2d.

        TODO: Compose _validate_input, _get_stats, and normalize+scale

        APPROACH:
        1. Validate input with self._validate_input(x)
        2. Get mean/var with self._get_stats(x)
        3. Normalize: (x - mean) / sqrt(var + eps) with proper broadcasting
        4. Scale by gamma and shift by beta

        EXAMPLE:
        >>> bn = BatchNorm2d(16)
        >>> x = Tensor(rng.standard_normal((2, 16, 8, 8)))
        >>> y = bn(x)
        >>> print(y.shape)  # (2, 16, 8, 8)

        HINTS:
        - Reshape mean/var/gamma/beta to (1, C, 1, 1) for broadcasting
        """
        ### BEGIN SOLUTION
        self._validate_input(x)
        
        mean, var = self._get_stats(x)
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        
        normalized = (x.data - mean) / np.sqrt(var + self.eps)

        gamma = self.gamma.data.reshape(1, -1, 1, 1)
        beta = self.beta.data.reshape(1, -1, 1, 1)
        
        output = gamma * normalized + beta
        
        return Tensor(
            output, 
            requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        )
        ### END SOLUTION

    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
# ### 🧪 Unit Test: BatchNorm2d._validate_input
# 
# **What we're testing**: Input shape validation catches common mistakes
# **Why it matters**: Clear errors save hours of debugging wrong tensor shapes
# **Expected**: Accepts valid 4D input, rejects 2D/3D/wrong channels with helpful messages

# %%
def test_unit_batchnorm2d_validate_input():
    """🧪 Test BatchNorm2d._validate_input implementation."""
    print("🧪 Unit Test: BatchNorm2d._validate_input...")

    bn = BatchNorm2d(num_features=16)

    # Valid 4D input should not raise
    x_valid = Tensor(rng.standard_normal((2, 16, 8, 8)))
    bn._validate_input(x_valid)  # Should pass silently

    # 3D input should raise
    try:
        bn._validate_input(Tensor(rng.standard_normal((16, 8, 8))))
        assert False, "Should have raised ValueError for 3D input"
    except ValueError as e:
        assert "3D" in str(e), f"Error should mention 3D, got: {e}"

    # Wrong channel count should raise
    try:
        bn._validate_input(Tensor(rng.standard_normal((2, 8, 4, 4))))
        assert False, "Should have raised ValueError for wrong channels"
    except ValueError as e:
        assert "mismatch" in str(e), f"Error should mention mismatch, got: {e}"

    print("✅ BatchNorm2d._validate_input works correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d_validate_input()

# %% [markdown]
# ### 🧪 Unit Test: BatchNorm2d._get_stats
# 
# **What we're testing**: Statistics computation in training vs eval mode
# **Why it matters**: Wrong statistics = wrong normalization = broken model
# **Expected**: Training mode computes batch stats and updates running stats; eval mode uses frozen stats

# %%
def test_unit_batchnorm2d_get_stats():
    """🧪 Test BatchNorm2d._get_stats implementation."""
    print("🧪 Unit Test: BatchNorm2d._get_stats...")

    bn = BatchNorm2d(num_features=4)
    x = Tensor(rng.standard_normal((8, 4, 6, 6)))

    # Training mode: should return batch stats and update running stats
    bn.train()
    running_mean_before = bn.running_mean.copy()
    mean, var = bn._get_stats(x)

    assert mean.shape == (4,), f"Expected per-channel mean shape (4,), got {mean.shape}"
    assert var.shape == (4,), f"Expected per-channel var shape (4,), got {var.shape}"
    assert not np.allclose(bn.running_mean, running_mean_before), \
        "Running mean should be updated in training mode"

    # Eval mode: should return running stats (frozen)
    bn.eval()
    running_mean_snapshot = bn.running_mean.copy()
    mean_eval, var_eval = bn._get_stats(x)

    assert np.allclose(mean_eval, running_mean_snapshot), \
        "Eval mode should return running mean"
    assert np.allclose(bn.running_mean, running_mean_snapshot), \
        "Running mean should not change in eval mode"

    print("✅ BatchNorm2d._get_stats works correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d_get_stats()

# %% [markdown]
# ### 🧪 Unit Test: BatchNorm2d
# 
# This test validates batch normalization implementation.
# 
# **What we're testing**: Normalization behavior, train/eval mode, running statistics
# **Why it matters**: BatchNorm is essential for training deep CNNs effectively
# **Expected**: Normalized outputs with proper mean/variance characteristics

# %%


def test_unit_batchnorm2d():
    """🧪 Test BatchNorm2d implementation."""
    print("🧪 Unit Test: BatchNorm2d...")

    # Test 1: Basic forward pass shape
    print("  Testing basic forward pass...")
    bn = BatchNorm2d(num_features=16)
    x = Tensor(rng.standard_normal((4, 16, 8, 8)))  # batch=4, channels=16, 8x8
    y = bn(x)

    assert y.shape == x.shape, f"Output shape should match input, got {y.shape}"

    # Test 2: Training mode normalization
    print("  Testing training mode normalization...")
    bn2 = BatchNorm2d(num_features=8)
    bn2.train()  # Ensure training mode

    # Create input with known statistics per channel
    x2 = Tensor(rng.standard_normal((32, 8, 4, 4)) * 10 + 5)  # Mean~5, std~10
    y2 = bn2(x2)

    # After normalization, each channel should have mean≈0, std≈1
    # (before gamma/beta are applied, since gamma=1, beta=0)
    for c in range(8):
        channel_mean = np.mean(y2.data[:, c, :, :])
        channel_std = np.std(y2.data[:, c, :, :])
        assert abs(channel_mean) < 0.1, f"Channel {c} mean should be ~0, got {channel_mean:.3f}"
        assert abs(channel_std - 1.0) < 0.1, f"Channel {c} std should be ~1, got {channel_std:.3f}"

    # Test 3: Running statistics update
    print("  Testing running statistics update...")
    initial_running_mean = bn2.running_mean.copy()

    # Forward pass updates running stats
    x3 = Tensor(rng.standard_normal((16, 8, 4, 4)) + 3)  # Offset mean
    _ = bn2(x3)

    # Running mean should have moved toward batch mean
    assert not np.allclose(bn2.running_mean, initial_running_mean), \
        "Running mean should update during training"

    # Test 4: Eval mode uses running statistics
    print("  Testing eval mode behavior...")
    bn3 = BatchNorm2d(num_features=4)

    # Train on some data to establish running stats
    for _ in range(10):
        x_train = Tensor(rng.standard_normal((8, 4, 4, 4)) * 2 + 1)
        _ = bn3(x_train)

    saved_running_mean = bn3.running_mean.copy()
    saved_running_var = bn3.running_var.copy()

    # Switch to eval mode
    bn3.eval()

    # Process different data - running stats should NOT change
    x_eval = Tensor(rng.standard_normal((2, 4, 4, 4)) * 5)  # Different distribution
    _ = bn3(x_eval)

    assert np.allclose(bn3.running_mean, saved_running_mean), \
        "Running mean should not change in eval mode"
    assert np.allclose(bn3.running_var, saved_running_var), \
        "Running var should not change in eval mode"

    # Test 5: Parameter counting
    print("  Testing parameter counting...")
    bn4 = BatchNorm2d(num_features=64)
    params = bn4.parameters()

    assert len(params) == 2, f"Should have 2 parameters (gamma, beta), got {len(params)}"
    assert params[0].shape == (64,), f"Gamma shape should be (64,), got {params[0].shape}"
    assert params[1].shape == (64,), f"Beta shape should be (64,), got {params[1].shape}"

    print("✅ BatchNorm2d works correctly!")

if __name__ == "__main__":
    test_unit_batchnorm2d()

# %% [markdown]
# ### 🧪 Unit Test: Pooling Operations
# 
# This test validates both max and average pooling implementations.
# 
# **What we're testing**: Dimension reduction, aggregation correctness
# **Why it matters**: Pooling is essential for computational efficiency in CNNs
# **Expected**: Correct output shapes and proper value aggregation

# %%


def test_unit_pooling():
    """🧪 Test MaxPool2d and AvgPool2d implementations."""
    print("🧪 Unit Test: Pooling Operations...")

    # Test 1: MaxPool2d basic functionality
    print("  Testing MaxPool2d...")
    maxpool = MaxPool2d(kernel_size=2, stride=2)
    x1 = Tensor(rng.standard_normal((1, 3, 8, 8)))
    out1 = maxpool(x1)

    expected_shape = (1, 3, 4, 4)  # 8/2 = 4
    assert out1.shape == expected_shape, f"MaxPool expected {expected_shape}, got {out1.shape}"

    # Test 2: AvgPool2d basic functionality
    print("  Testing AvgPool2d...")
    avgpool = AvgPool2d(kernel_size=2, stride=2)
    x2 = Tensor(rng.standard_normal((2, 16, 16, 16)))
    out2 = avgpool(x2)

    expected_shape = (2, 16, 8, 8)  # 16/2 = 8
    assert out2.shape == expected_shape, f"AvgPool expected {expected_shape}, got {out2.shape}"

    # Test 3: MaxPool vs AvgPool on known data
    print("  Testing max vs avg behavior...")
    # Create simple test case with known values
    test_data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)
    x3 = Tensor(test_data)

    maxpool_test = MaxPool2d(kernel_size=2, stride=2)
    avgpool_test = AvgPool2d(kernel_size=2, stride=2)

    max_out = maxpool_test(x3)
    avg_out = avgpool_test(x3)

    # For 2x2 windows:
    # Top-left: max([1,2,5,6]) = 6, avg = 3.5
    # Top-right: max([3,4,7,8]) = 8, avg = 5.5
    # Bottom-left: max([9,10,13,14]) = 14, avg = 11.5
    # Bottom-right: max([11,12,15,16]) = 16, avg = 13.5

    expected_max = np.array([[[[6, 8], [14, 16]]]])
    expected_avg = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])

    assert np.allclose(max_out.data, expected_max), f"MaxPool values incorrect: {max_out.data} vs {expected_max}"
    assert np.allclose(avg_out.data, expected_avg), f"AvgPool values incorrect: {avg_out.data} vs {expected_avg}"

    # Test 4: Overlapping pooling (stride < kernel_size)
    print("  Testing overlapping pooling...")
    overlap_pool = MaxPool2d(kernel_size=3, stride=1)
    x4 = Tensor(rng.standard_normal((1, 1, 5, 5)))
    out4 = overlap_pool(x4)

    # Output: (5-3)/1 + 1 = 3
    expected_shape = (1, 1, 3, 3)
    assert out4.shape == expected_shape, f"Overlapping pool expected {expected_shape}, got {out4.shape}"

    # Test 5: No parameters in pooling layers
    print("  Testing parameter counts...")
    assert len(maxpool.parameters()) == 0, "MaxPool should have no parameters"
    assert len(avgpool.parameters()) == 0, "AvgPool should have no parameters"

    print("✅ Pooling operations work correctly!")

if __name__ == "__main__":
    test_unit_pooling()

# %% [markdown]
# ## 📊 Systems Analysis: Spatial Operation Performance
# 
# Let's understand ONE key systems concept: **computational complexity and memory trade-offs in spatial operations**.
# 
# This single analysis reveals why certain design choices matter for real-world performance, and why modern CNNs use specific architectural patterns.

# %%


def analyze_convolution_complexity():
    """📊 Analyze convolution computational complexity across different configurations."""
    print("📊 Analyzing Convolution Complexity...")

    # Test configurations optimized for educational demonstration (smaller sizes)
    configs = [
        {"input": (1, 3, 16, 16), "conv": (8, 3, 3), "name": "Small (16×16)"},
        {"input": (1, 3, 24, 24), "conv": (12, 3, 3), "name": "Medium (24×24)"},
        {"input": (1, 3, 32, 32), "conv": (16, 3, 3), "name": "Large (32×32)"},
        {"input": (1, 3, 16, 16), "conv": (8, 3, 5), "name": "Large Kernel (5×5)"},
    ]

    print(f"{'Configuration':<20} {'FLOPs':<15} {'Memory (MB)':<12} {'Time (ms)':<10}")
    print("-" * 70)

    for config in configs:
        # Create convolution layer
        in_ch = config["input"][1]
        out_ch, k_size = config["conv"][0], config["conv"][2]
        conv = Conv2d(in_ch, out_ch, kernel_size=k_size, padding=k_size//2)

        # Create input tensor
        x = Tensor(rng.standard_normal(config["input"]))

        # Calculate theoretical FLOPs
        batch, in_channels, h, w = config["input"]
        out_channels, kernel_size = config["conv"][0], config["conv"][2]

        # Each output element requires in_channels * kernel_size² multiply-adds
        flops_per_output = in_channels * kernel_size * kernel_size * 2  # 2 for MAC
        total_outputs = batch * out_channels * h * w  # Assuming same size with padding
        total_flops = flops_per_output * total_outputs

        # Measure memory usage
        input_memory = np.prod(config["input"]) * 4  # float32 = 4 bytes
        weight_memory = out_channels * in_channels * kernel_size * kernel_size * 4
        output_memory = batch * out_channels * h * w * 4
        total_memory = (input_memory + weight_memory + output_memory) / (1024 * 1024)  # MB

        # Measure execution time
        start_time = time.time()
        _ = conv(x)
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000  # ms

        print(f"{config['name']:<20} {total_flops:<15,} {total_memory:<12.2f} {exec_time:<10.2f}")

    print("\n💡 Key Insights:")
    print("🔸 FLOPs scale as O(H×W×C_in×C_out×K²) - quadratic in spatial and kernel size")
    print("🔸 Memory scales linearly with spatial dimensions and channels")
    print("🔸 Large kernels dramatically increase computational cost")
    print("🚀 This motivates more efficient convolution variants that reduce computational cost")

# Run the systems analysis
if __name__ == "__main__":
    analyze_convolution_complexity()

# %%


def analyze_pooling_effects():
    """📊 Analyze pooling's impact on spatial dimensions and features."""
    print("\n📊 Analyzing Pooling Effects...")

    # Create sample input with spatial structure
    # Simple edge pattern that pooling should preserve differently
    pattern = np.zeros((1, 1, 8, 8))
    pattern[0, 0, :, 3:5] = 1.0  # Vertical edge
    pattern[0, 0, 3:5, :] = 1.0  # Horizontal edge
    x = Tensor(pattern)

    print("Original 8×8 pattern:")
    print(x.data[0, 0])

    # Test different pooling strategies
    pools = [
        (MaxPool2d(2, stride=2), "MaxPool 2×2"),
        (AvgPool2d(2, stride=2), "AvgPool 2×2"),
        (MaxPool2d(4, stride=4), "MaxPool 4×4"),
        (AvgPool2d(4, stride=4), "AvgPool 4×4"),
    ]

    print(f"\n{'Operation':<15} {'Output Shape':<15} {'Feature Preservation'}")
    print("-" * 60)

    for pool_op, name in pools:
        result = pool_op(x)
        # Measure how much of the original pattern is preserved
        preservation = np.sum(result.data > 0.1) / np.prod(result.shape)
        print(f"{name:<15} {str(result.shape):<15} {preservation:<.2%}")

        print(f"  Output:")
        print(f"  {result.data[0, 0]}")
        print()

    print("💡 Key Insights:")
    print("🔸 MaxPool preserves sharp features better (edge detection)")
    print("🔸 AvgPool smooths features (noise reduction)")
    print("🔸 Larger pooling windows lose more spatial detail")
    print("🚀 Choice depends on task: classification vs detection vs segmentation")

# Run the systems analysis
if __name__ == "__main__":
    analyze_pooling_effects()

# %% [markdown]
# ## 🔧 Integration - Building a Complete CNN
# 
# Now let's combine convolution and pooling into a complete CNN architecture. You'll see how spatial operations work together to transform raw pixels into meaningful features.
# 
# ### CNN Architecture: From Pixels to Predictions
# 
# A CNN processes images through alternating convolution and pooling layers, gradually extracting higher-level features:
# 
# ```
# Complete CNN Pipeline:
# 
# Input Image (32×32×3)     Raw RGB pixels
#        ↓
# Conv2d(3→16, 3×3)        Detect edges, textures
#        ↓
# ReLU Activation          Remove negative values
#        ↓
# MaxPool(2×2)             Reduce to (16×16×16)
#        ↓
# Conv2d(16→32, 3×3)       Detect shapes, patterns
#        ↓
# ReLU Activation          Remove negative values
#        ↓
# MaxPool(2×2)             Reduce to (8×8×32)
#        ↓
# Flatten                  Reshape to vector (2048,)
#        ↓
# Linear(2048→10)          Final classification
#        ↓
# Softmax                  Probability distribution
# ```
# 
# ### The Parameter Efficiency Story
# 
# ```
# CNN vs Dense Network Comparison:
# 
# CNN Approach:                     Dense Approach:
# ┌─────────────────┐               ┌─────────────────┐
# │ Conv1: 3→16     │               │ Input: 32×32×3  │
# │ Params: 448     │               │ = 3,072 values  │
# ├─────────────────┤               ├─────────────────┤
# │ Conv2: 16→32    │               │ Hidden: 1,000   │
# │ Params: 4,640   │               │ Params: 3M+     │
# ├─────────────────┤               ├─────────────────┤
# │ Linear: 2048→10 │               │ Output: 10      │
# │ Params: 20,490  │               │ Params: 10K     │
# └─────────────────┘               └─────────────────┘
# Total: ~25K params                Total: ~3M params
# 
# CNN wins with 120× fewer parameters!
# ```
# 
# ### Spatial Hierarchy: Why This Architecture Works
# 
# ```
# Layer-by-Layer Feature Evolution:
# 
# Layer 1 (Conv 3→16):              Layer 2 (Conv 16→32):
# ┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐ ┌──────┐ ┌───────┐
# │Edge │ │Edge │ │Edge │           │Shape│ │Corner│ │Texture│
# │ \\ /│ │  |  │ │ / \\│           │ ◇   │ │  L   │ │ ≈≈≈≈≈ │
# └─────┘ └─────┘ └─────┘           └─────┘ └──────┘ └───────┘
# Simple features                   Complex combinations
# 
# Why pooling between layers:
# ✓ Reduces computation for next layer
# ✓ Increases receptive field (each conv sees larger input area)
# ✓ Provides translation invariance (cat moved 1 pixel still detected)
# ```
# 
# This hierarchical approach mirrors human vision: we first detect edges, then shapes, then objects!

# %% [markdown]
# ### SimpleCNN Implementation - Putting It All Together
# 
# Now we'll build a complete CNN that demonstrates how convolution and pooling work together. This is your first step from processing individual tensors to understanding complete images!
# 
# #### The CNN Architecture Pattern
# 
# ```
# SimpleCNN Architecture Visualization:
# 
# Input: (batch, 3, 32, 32)     ← RGB images (CIFAR-10 size)
#          ↓
# ┌─────────────────────────┐
# │ Conv2d(3→16, 3×3, p=1)  │    ← Detect edges, textures
# │ ReLU()                  │    ← Remove negative values
# │ MaxPool(2×2)            │    ← Reduce to (batch, 16, 16, 16)
# └─────────────────────────┘
#          ↓
# ┌─────────────────────────┐
# │ Conv2d(16→32, 3×3, p=1) │   ← Detect shapes, patterns
# │ ReLU()                  │   ← Remove negative values
# │ MaxPool(2×2)            │   ← Reduce to (batch, 32, 8, 8)
# └─────────────────────────┘
#          ↓
# ┌─────────────────────────┐
# │ Flatten()               │   ← Reshape to (batch, 2048)
# │ Linear(2048→10)         │   ← Final classification
# └─────────────────────────┘
#          ↓
# Output: (batch, 10)           ← Class probabilities
# ```
# 
# #### Why This Architecture Works
# 
# ```
# Feature Hierarchy Development:
# 
# Layer 1 Features (3→16):     Layer 2 Features (16→32):
# ┌─────┬─────┬─────┬─────┐   ┌─────┬─────┬─────┬─────┐
# │Edge │Edge │Edge │Blob │   │Shape│Corner│Tex-│Pat- │
# │ \\  │  |  │ /   │  ○  │   │ ◇   │  L  │ture│tern  │
# └─────┴─────┴─────┴─────┘   └─────┴─────┴─────┴─────┘
# Simple features             Complex combinations
# 
# Spatial Dimension Reduction:
# 32×32 → 16×16 → 8×8
#  1024    256     64  (per channel)
# 
# Channel Expansion:
# 3 → 16 → 32
# More feature types at each level
# ```
# 
# #### Parameter Efficiency Demonstration
# 
# ```
# CNN vs Dense Comparison for 32×32×3 → 10 classes:
# 
# CNN Approach:                    Dense Approach:
# ┌────────────────────┐          ┌────────────────────┐
# │ Conv1: 3→16, 3×3   │          │ Input: 3072 values │
# │ Params: 448        │          │        ↓           │
# ├────────────────────┤          │ Dense: 3072→512    │
# │ Conv2: 16→32, 3×3  │          │ Params: 1.57M      │
# │ Params: 4,640      │          ├────────────────────┤
# ├────────────────────┤          │ Dense: 512→10      │
# │ Dense: 2048→10     │          │ Params: 5,120      │
# │ Params: 20,490     │          └────────────────────┘
# └────────────────────┘          Total: 1.58M params
# Total: 25,578 params
# 
# CNN has 62× fewer parameters while preserving spatial structure!
# ```
# 
# #### Receptive Field Growth
# 
# ```
# How each layer sees progressively larger input regions:
# 
# Layer 1 Conv (3×3):           Layer 2 Conv (3×3):
# Each output pixel sees        Each output pixel sees
# 3×3 = 9 input pixels         7×7 = 49 input pixels
#                              (due to pooling+conv)
# 
# Final Result: Layer 2 can detect complex patterns
# spanning 7×7 regions of original image!
# ```

# %%

#| export

class SimpleCNN:
    """
    Simple CNN demonstrating spatial operations integration.

    Architecture:
    - Conv2d(3→16, 3×3) + ReLU + MaxPool(2×2)
    - Conv2d(16→32, 3×3) + ReLU + MaxPool(2×2)
    - Flatten + Linear(features→num_classes)
    """

    def __init__(self, num_classes=10):
        """
        Initialize SimpleCNN.

        TODO: Build CNN architecture with spatial and dense layers

        APPROACH:
        1. Conv layer 1: 3 → 16 channels, 3×3 kernel, padding=1
        2. Pool layer 1: 2×2 max pooling
        3. Conv layer 2: 16 → 32 channels, 3×3 kernel, padding=1
        4. Pool layer 2: 2×2 max pooling
        5. Calculate flattened size and add final linear layer

        HINT: For 32×32 input → 32→16→8 spatial reduction
        Final feature size: 32 channels × 8×8 = 2048 features
        """
        super().__init__()

        ### BEGIN SOLUTION
        self.conv1 = Conv2d(3, 16, 3, padding=1)
        self.pool1 = MaxPool2d(2)
        
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.pool2 = MaxPool2d(2)

        self.flattened_size = 32 * 8 * 8
        self.num_classes = num_classes
        ### END SOLUTION

    def forward(self, x):
        """
        Forward pass through SimpleCNN.

        TODO: Implement CNN forward pass

        APPROACH:
        1. Apply conv1 → ReLU → pool1
        2. Apply conv2 → ReLU → pool2
        3. Flatten spatial dimensions
        4. Apply final linear layer (when available)

        For now, return features before final linear layer
        since we haven't imported Linear from layers module yet.
        """
        ### BEGIN SOLUTION
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = x.reshape(x.shape[0], -1)
        return x
        ### END SOLUTION

    def relu(self, x):
        """ReLU activation with gradient tracking for CNN."""
        result_data = np.maximum(0, x.data)
        result = Tensor(result_data)
        if x.requires_grad:
            result.requires_grad = True
            result._grad_fn = ReLUBackward(x)
        return result

    def parameters(self):
        """Return all trainable parameters."""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        # Linear layer parameters would be added here
        return params

    def __call__(self, x):
        """Enable model(x) syntax."""
        return self.forward(x)

# %% [markdown]
# ### 🧪 Unit Test: SimpleCNN Integration
# 
# This test validates that spatial operations work together in a complete CNN architecture.
# 
# **What we're testing**: End-to-end spatial processing pipeline
# **Why it matters**: Spatial operations must compose correctly for real CNNs
# **Expected**: Proper dimension reduction and feature extraction

# %%


def test_unit_simple_cnn():
    """🧪 Test SimpleCNN integration with spatial operations."""
    print("🧪 Unit Test: SimpleCNN Integration...")

    # Test 1: Forward pass with CIFAR-10 sized input
    print("  Testing forward pass...")
    model = SimpleCNN(num_classes=10)
    x = Tensor(rng.standard_normal((2, 3, 32, 32)))  # Batch of 2, RGB, 32×32

    features = model(x)

    # Expected: 2 samples, 32 channels × 8×8 spatial = 2048 features
    expected_shape = (2, 2048)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"

    # Test 2: Parameter counting
    print("  Testing parameter counting...")
    params = model.parameters()

    # Conv1: (16, 3, 3, 3) + bias (16,) = 432 + 16 = 448
    # Conv2: (32, 16, 3, 3) + bias (32,) = 4608 + 32 = 4640
    # Total: 448 + 4640 = 5088 parameters

    conv1_params = 16 * 3 * 3 * 3 + 16  # weights + bias
    conv2_params = 32 * 16 * 3 * 3 + 32  # weights + bias
    expected_total = conv1_params + conv2_params

    actual_total = sum(np.prod(p.shape) for p in params)
    assert actual_total == expected_total, f"Expected {expected_total} parameters, got {actual_total}"

    # Test 3: Different input sizes
    print("  Testing different input sizes...")

    # Test with different spatial dimensions
    x_small = Tensor(rng.standard_normal((1, 3, 16, 16)))
    features_small = model(x_small)

    # 16×16 → 8×8 → 4×4, so 32 × 4×4 = 512 features
    expected_small = (1, 512)
    assert features_small.shape == expected_small, f"Expected {expected_small}, got {features_small.shape}"

    # Test 4: Batch processing
    print("  Testing batch processing...")
    x_batch = Tensor(rng.standard_normal((8, 3, 32, 32)))
    features_batch = model(x_batch)

    expected_batch = (8, 2048)
    assert features_batch.shape == expected_batch, f"Expected {expected_batch}, got {features_batch.shape}"

    print("✅ SimpleCNN integration works correctly!")

if __name__ == "__main__":
    test_unit_simple_cnn()

# %% [markdown]
# ## 🧪 Module Integration Test
# 
# Final validation that everything works together correctly.

# %%


def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire spatial module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")

    # Conv2d helper tests
    test_unit_conv2d_output_shape()
    test_unit_conv2d_padding()
    test_unit_conv2d_convolve_loops()
    test_unit_conv2d()

    # BatchNorm2d helper tests
    test_unit_batchnorm2d_validate_input()
    test_unit_batchnorm2d_get_stats()

    # MaxPool2d helper tests
    test_unit_maxpool2d_output_shape()
    test_unit_maxpool2d_loops()

    # AvgPool2d helper tests
    test_unit_avgpool2d_output_shape()
    test_unit_avgpool2d_loops()

    # Remaining unit tests
    test_unit_batchnorm2d()
    test_unit_pooling()
    test_unit_simple_cnn()

    print("\nRunning integration scenarios...")

    # Test realistic CNN workflow with BatchNorm
    print("🧪 Integration Test: Complete CNN pipeline with BatchNorm...")

    # Create a mini CNN for CIFAR-10 with BatchNorm (modern architecture)
    conv1 = Conv2d(3, 8, kernel_size=3, padding=1)
    bn1 = BatchNorm2d(8)
    pool1 = MaxPool2d(2, stride=2)
    conv2 = Conv2d(8, 16, kernel_size=3, padding=1)
    bn2 = BatchNorm2d(16)
    pool2 = AvgPool2d(2, stride=2)

    # Process batch of images (training mode)
    batch_images = Tensor(rng.standard_normal((4, 3, 32, 32)))

    # Forward pass: Conv → BatchNorm → ReLU → Pool (modern pattern)
    x = conv1(batch_images)  # (4, 8, 32, 32)
    x = bn1(x)               # (4, 8, 32, 32) - normalized
    x = Tensor(np.maximum(0, x.data))  # ReLU
    x = pool1(x)             # (4, 8, 16, 16)

    x = conv2(x)             # (4, 16, 16, 16)
    x = bn2(x)               # (4, 16, 16, 16) - normalized
    x = Tensor(np.maximum(0, x.data))  # ReLU
    features = pool2(x)      # (4, 16, 8, 8)

    # Validate shapes at each step
    assert features.shape[0] == 4, f"Batch size should be preserved, got {features.shape[0]}"
    assert features.shape == (4, 16, 8, 8), f"Final features shape incorrect: {features.shape}"

    # Test parameter collection across all layers
    all_params = []
    all_params.extend(conv1.parameters())
    all_params.extend(bn1.parameters())
    all_params.extend(conv2.parameters())
    all_params.extend(bn2.parameters())

    # Pooling has no parameters
    assert len(pool1.parameters()) == 0
    assert len(pool2.parameters()) == 0

    # BatchNorm has 2 params each (gamma, beta)
    assert len(bn1.parameters()) == 2, f"BatchNorm should have 2 parameters, got {len(bn1.parameters())}"

    # Total: Conv1 (2) + BN1 (2) + Conv2 (2) + BN2 (2) = 8 parameters
    assert len(all_params) == 8, f"Expected 8 parameter tensors total, got {len(all_params)}"

    # Test train/eval mode switching
    print("🧪 Integration Test: Train/Eval mode switching...")
    bn1.eval()
    bn2.eval()

    # Run inference with single sample (would fail with batch stats)
    single_image = Tensor(rng.standard_normal((1, 3, 32, 32)))
    x = conv1(single_image)
    x = bn1(x)  # Uses running stats, not batch stats
    assert x.shape == (1, 8, 32, 32), f"Single sample inference should work in eval mode"

    print("✅ CNN pipeline with BatchNorm works correctly!")

    # Test memory efficiency comparison
    print("🧪 Integration Test: Memory efficiency analysis...")

    # Compare different pooling strategies (reduced size for faster execution)
    input_data = Tensor(rng.standard_normal((1, 16, 32, 32)))

    # No pooling: maintain spatial size
    conv_only = Conv2d(16, 32, kernel_size=3, padding=1)
    no_pool_out = conv_only(input_data)
    no_pool_size = np.prod(no_pool_out.shape) * 4  # float32 bytes

    # With pooling: reduce spatial size
    conv_with_pool = Conv2d(16, 32, kernel_size=3, padding=1)
    pool = MaxPool2d(2, stride=2)
    pool_out = pool(conv_with_pool(input_data))
    pool_size = np.prod(pool_out.shape) * 4  # float32 bytes

    memory_reduction = no_pool_size / pool_size
    assert memory_reduction == 4.0, f"2×2 pooling should give 4× memory reduction, got {memory_reduction:.1f}×"

    print(f"  Memory reduction with pooling: {memory_reduction:.1f}×")
    print("✅ Memory efficiency analysis complete!")

    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! Module ready for export.")
    print("Run: tito module complete 09")

# Run module test when this cell is executed
if __name__ == "__main__":
    test_module()



# %% [markdown]
# ## 🤔 ML Systems Reflection Questions
# 
# Answer these to deepen your understanding of spatial operations and their systems implications:
# 
# ### 1. Conv2d Memory Footprint
# A Conv2d layer with 64 filters (3×3) processes a (224×224×3) image.
# - Calculate the memory footprint during the forward pass
# - Consider: input activations, output activations, filter weights, and biases
# - What happens when batch size increases from 1 to 32?
# 
# **Think about**: Why do modern vision models use techniques like gradient checkpointing?
# 
# ---
# 
# ### 2. Spatial Locality and CPU Performance
# Why are CNNs faster on CPUs than fully-connected networks of similar parameter count?
# 
# **Consider**:
# - Cache locality in convolution operations
# - Data reuse patterns in sliding windows
# - Memory access patterns (sequential vs random)
# 
# **Hint**: Think about what happens when the same filter is applied across the image.
# 
# ---
# 
# ### 3. Im2col Trade-off
# The im2col algorithm transforms convolution into matrix multiplication, using more memory but speeding up computation.
# 
# **When is this trade-off worthwhile?**
# - Small vs large batch sizes
# - Small vs large images
# - Training vs inference
# - Mobile vs server deployment
# 
# **Think about**: Why don't mobile devices always use im2col?
# 
# ---
# 
# ### 4. Pooling's Systems Benefits
# MaxPool2d reduces spatial dimensions (e.g., 224×224 → 112×112).
# 
# **What's the systems benefit beyond reducing parameters?**
# - Memory bandwidth requirements
# - Computation in subsequent layers
# - Gradient memory during backpropagation
# - Cache efficiency in deeper layers
# 
# **Calculate**: If 5 layers each use 2x2 pooling, what's the total memory reduction?
# 
# ---
# 
# ### 5. Mobile ML Deployment
# Why do mobile ML models prefer depthwise-separable convolutions over standard Conv2d?
# 
# **Analyze the FLOPs**:
# - Standard 3×3 conv: C_in × C_out × H × W × 9
# - Depthwise + Pointwise: (C_in × H × W × 9) + (C_in × C_out × H × W)
# 
# **When does the trade-off favor depthwise separable?**
# - As number of channels increases
# - As spatial dimensions change
# - Energy consumption vs accuracy
# 
# **Real-world context**: This is why MobileNet and EfficientNet architectures exist.

# %% [markdown]
# ## ⭐ Aha Moment: Convolution Extracts Features
# 
# **What you built:** Convolutional layers that process spatial data like images.
# 
# **Why it matters:** Conv2d looks at local neighborhoods, detecting edges, textures, and patterns.
# Unlike Linear layers that see pixels independently, Conv2d understands that nearby pixels are
# related. This is why CNNs revolutionized computer vision!
# 
# In the milestones, you'll use these spatial operations to build a CNN that recognizes digits.

# %%
def demo_convolutions():
    """🎯 See Conv2d process spatial data."""
    print("🎯 AHA MOMENT: Convolution Extracts Features")
    print("=" * 45)

    # Create a simple 8x8 "image" with 1 channel
    image = Tensor(rng.standard_normal((1, 1, 8, 8)))

    # Conv2d: 1 input channel → 4 feature maps
    conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3)

    output = conv(image)

    print(f"Input:  {image.shape}  ← 1 image, 1 channel, 8×8")
    print(f"Output: {output.shape}  ← 1 image, 4 features, 6×6")
    print(f"\nConv kernel: 3×3 sliding window")
    print(f"Output smaller: 8 - 3 + 1 = 6 (no padding)")

    print("\n✨ Conv2d detects spatial patterns in images!")

# %%
if __name__ == "__main__":
    test_module()
    print("\n")
    demo_convolutions()

# %% [markdown]
# ## 🚀 MODULE SUMMARY: Spatial Operations
# 
# Congratulations! You've built the spatial processing foundation that powers computer vision!
# 
# ### Key Accomplishments
# - **Built Conv2d** with explicit loops showing O(N^2 M^2 K^2) complexity
# - **Implemented BatchNorm2d** with train/eval mode and running statistics
# - **Implemented MaxPool2d and AvgPool2d** for spatial dimension reduction
# - **Created SimpleCNN** demonstrating spatial operation integration
# - **Analyzed computational complexity** and memory trade-offs in spatial processing
# - **All tests pass** (validated by `test_module()`)
# 
# ### Systems Insights Discovered
# - **Convolution Complexity**: Quadratic scaling with spatial size; kernel size significantly impacts cost
# - **Batch Normalization**: Train vs eval mode is critical (batch stats during training, running stats during inference)
# - **Memory Patterns**: Pooling provides 4x memory reduction while preserving important features
# - **Architecture Design**: Strategic spatial reduction enables parameter-efficient feature extraction
# - **Cache Performance**: Spatial locality in convolution benefits from optimal memory access patterns
# 
# ### Ready for Next Steps
# Your spatial operations enable building complete CNNs for computer vision tasks!
# 
# **Next**: Milestone 03 will combine your spatial operations with training pipeline to build a CNN for CIFAR-10!
# 
# Export with: `tito module complete 09`


