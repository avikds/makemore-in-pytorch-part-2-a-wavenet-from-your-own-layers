"""
makemore in PyTorch, Part 2: A WaveNet From Your Own Layers

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_splits
import os
import random
import tempfile
import urllib.request
import torch

def load_splits(block_size=8, seed=42):
    """Return (itos, {'train': (X, Y), 'dev': (X, Y), 'test': (X, Y)})
    with block_size characters of context.
    """

    # Download the names dataset once and cache it in the temporary directory.
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    file_path = os.path.join(tempfile.gettempdir(), "makemore_names.txt")

    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        urllib.request.urlretrieve(url, file_path)

    # Read the names, keeping one name per line.
    with open(file_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    # Build the vocabulary: '.' receives ID 0, followed by sorted characters.
    chars = sorted(set("".join(names)) - {"."})

    itos = {0: "."}
    itos.update({i + 1: ch for i, ch in enumerate(chars)})

    # Create the character-to-ID lookup.
    stoi = {ch: i for i, ch in itos.items()}

    # Shuffle a copy of the names deterministically.
    shuffled_names = names.copy()
    random.Random(seed).shuffle(shuffled_names)

    # Split the names into training, development, and test sets.
    n = len(shuffled_names)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n)

    train_names = shuffled_names[:n_train]
    dev_names = shuffled_names[n_train:n_dev]
    test_names = shuffled_names[n_dev:]

    # Convert names into input contexts (X) and next-character targets (Y).
    def build_dataset(name_list):
        X, Y = [], []

        for name in name_list:
            # Start each name with a context of block_size padding tokens.
            context = [0] * block_size

            # Predict each character, followed by the final '.' token.
            for ch in name + ".":
                target = stoi[ch]

                X.append(context.copy())
                Y.append(target)

                # Slide the context window forward by one character.
                context = context[1:] + [target]

        # Convert the input contexts and targets into PyTorch tensors.
        X = torch.tensor(X, dtype=torch.long).reshape(-1, block_size)
        Y = torch.tensor(Y, dtype=torch.long)

        return X, Y

    # Build all three datasets.
    splits = {
        "train": build_dataset(train_names),
        "dev": build_dataset(dev_names),
        "test": build_dataset(test_names)
    }

    return itos, splits

# Step 2 - Linear
class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0, generator=None):
        """Kaiming-initialised weight (fan_in, fan_out) and an optional zero bias."""

        # Initialize weights using Kaiming initialization.
        self.weight = torch.randn(
            fan_in,
            fan_out,
            generator=generator
        ) * (gain / fan_in ** 0.5)

        # Initialize bias to zero if enabled.
        self.bias = torch.zeros(fan_out) if bias else None

        # Store the output for inspection.
        self.out = None

    def __call__(self, x):
        """x @ weight (+ bias), also stored in self.out."""

        # Apply the linear transformation.
        self.out = x @ self.weight

        # Add bias if it is enabled.
        if self.bias is not None:
            self.out = self.out + self.bias

        return self.out

    def parameters(self):
        """Return the trainable parameters of the layer."""

        # Include bias only when it exists.
        return [self.weight] + ([self.bias] if self.bias is not None else [])

# Step 3 - Tanh and activation_stats
class Tanh:
    def __init__(self):
        # Store the output for inspecting activations.
        self.out = None

    def __call__(self, x):
        """tanh(x), also stored in self.out."""

        # Apply the hyperbolic tangent activation function.
        self.out = torch.tanh(x)

        return self.out

    def parameters(self):
        """Tanh has no trainable parameters."""
        return []


def activation_stats(t, threshold=0.97):
    """Return activation statistics:
    mean, standard deviation, and percentage of saturated activations.
    """

    # Calculate the mean and standard deviation.
    mean = t.mean().item()
    std = t.std().item()

    # Calculate the percentage of values whose absolute magnitude
    # exceeds the specified saturation threshold.
    saturated = (t.abs() > threshold).float().mean().item() * 100

    return {
        "mean": mean,
        "std": std,
        "saturated": saturated
    }

# Step 4 - BatchNorm1d
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        """gamma, beta (parameters); running_mean, running_var (buffers); training flag."""

        # Hyperparameters.
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters: gain and bias.
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Running statistics used during evaluation.
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

        # Start in training mode.
        self.training = True

        # Store the output for inspection.
        self.out = None

    def __call__(self, x):
        """Normalize over every dimension but the last; batch stats in training,
        running stats in evaluation.
        """

        # BatchNorm1d supports 2-D and 3-D inputs.
        if x.ndim not in (2, 3):
            raise ValueError(
                f"Expected a 2-D or 3-D input, but received {x.ndim}-D input."
            )

        if self.training:
            # Compute statistics across every dimension except the last.
            # 2-D: (N, D)    -> reduce over dimension (0,)
            # 3-D: (N, L, D) -> reduce over dimensions (0, 1)
            reduce_dims = tuple(range(x.ndim - 1))

            batch_mean = x.mean(dim=reduce_dims)
            batch_var = x.var(dim=reduce_dims)

            # Use current batch statistics for normalization.
            mean = batch_mean
            var = batch_var

            # Update running statistics without tracking gradients.
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    batch_mean.detach(),
                    alpha=self.momentum
                )

                self.running_var.mul_(1 - self.momentum).add_(
                    batch_var.detach(),
                    alpha=self.momentum
                )

        else:
            # Use previously accumulated running statistics in evaluation mode.
            mean = self.running_mean
            var = self.running_var

        # Normalize the input using the selected statistics.
        xhat = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable gain (gamma) and bias (beta).
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        """Return the learnable parameters of BatchNorm1d."""

        # Running statistics are buffers, not trainable parameters.
        return [self.gamma, self.beta]

# Step 5 - Embedding
class Embedding:
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        """Initialize an embedding weight matrix of shape
        (num_embeddings, embedding_dim).
        """

        # Initialize the embedding matrix with random values.
        self.weight = torch.randn(
            num_embeddings,
            embedding_dim,
            generator=generator
        )

        # Store the output for inspection.
        self.out = None

    def __call__(self, IX):
        """Look up embeddings using integer indices and store the result in self.out."""

        # Retrieve the embedding vectors corresponding to the input indices.
        self.out = self.weight[IX]

        return self.out

    def parameters(self):
        """Return the trainable embedding weight matrix."""
        return [self.weight]

# Step 6 - FlattenConsecutive
class FlattenConsecutive:
    def __init__(self, n):
        """Initialize the layer with the number of consecutive vectors to flatten."""

        if n <= 0:
            raise ValueError("n must be a positive integer.")

        self.n = n

        # Store the output for inspection.
        self.out = None

    def __call__(self, x):
        """Transform (B, T, C) into (B, T // n, C * n).
        Remove the middle dimension if its size is 1.
        """

        # Extract the input dimensions.
        B, T, C = x.shape

        # Ensure the sequence length can be divided into equal groups.
        if T % self.n != 0:
            raise ValueError(
                f"Sequence length {T} must be divisible by n={self.n}."
            )

        # Group consecutive vectors and concatenate their features.
        self.out = x.reshape(B, T // self.n, C * self.n)

        # Remove the middle dimension when its size is 1.
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)

        return self.out

    def parameters(self):
        """FlattenConsecutive has no trainable parameters."""
        return []

# Step 7 - Sequential
class Sequential:
    def __init__(self, layers):
        """Initialize the Sequential container with a list of layers."""

        # Store the layers in the order they should be executed.
        self.layers = layers

        # Store the final output for inspection.
        self.out = None

    def __call__(self, x):
        """Pass x through every layer in order."""

        # Sequentially apply each layer to the output of the previous layer.
        for layer in self.layers:
            x = layer(x)

        # Store and return the final output.
        self.out = x

        return self.out

    def parameters(self):
        """Return every layer's parameters in one flat list."""

        # Collect trainable parameters from all layers.
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

# Step 8 - build_wavenet
def build_wavenet(vocab_size=27, n_embd=10, n_hidden=68, seed=42):
    """Build a hierarchical WaveNet for 8 characters of context.

    Architecture:
        Embedding
        Three hierarchical blocks:
            FlattenConsecutive(2)
            Linear
            BatchNorm1d
            Tanh
        Output Linear
    """

    # Create a seeded generator for reproducible initialization.
    generator = torch.Generator().manual_seed(seed)

    # Build the hierarchical WaveNet using the previously defined custom layers.
    model = Sequential([
        # Convert character indices into dense embedding vectors.
        Embedding(vocab_size, n_embd, generator=generator),

        # First block: combine 8 character embeddings into 4 representations.
        FlattenConsecutive(2),
        Linear(n_embd * 2, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden),
        Tanh(),

        # Second block: combine 4 representations into 2 representations.
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden),
        Tanh(),

        # Third block: combine 2 representations into a single representation.
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False, generator=generator),
        BatchNorm1d(n_hidden),
        Tanh(),

        # Final layer: predict the next character from the vocabulary.
        Linear(n_hidden, vocab_size, generator=generator)
    ])

    # Reduce the initial output weights to prevent overly confident predictions.
    with torch.no_grad():
        model.layers[-1].weight.mul_(0.1)

    # Enable gradient computation for every trainable parameter.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    return model

# Step 9 - train (not yet solved)
# TODO: implement

# Step 10 - smooth (not yet solved)
# TODO: implement

# Step 11 - evaluate (not yet solved)
# TODO: implement

# Step 12 - sample (not yet solved)
# TODO: implement

