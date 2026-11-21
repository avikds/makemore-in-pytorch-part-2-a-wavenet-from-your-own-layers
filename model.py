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

# Step 2 - Linear (not yet solved)
# TODO: implement

# Step 3 - Tanh and activation_stats (not yet solved)
# TODO: implement

# Step 4 - BatchNorm1d (not yet solved)
# TODO: implement

# Step 5 - Embedding (not yet solved)
# TODO: implement

# Step 6 - FlattenConsecutive (not yet solved)
# TODO: implement

# Step 7 - Sequential (not yet solved)
# TODO: implement

# Step 8 - build_wavenet (not yet solved)
# TODO: implement

# Step 9 - train (not yet solved)
# TODO: implement

# Step 10 - smooth (not yet solved)
# TODO: implement

# Step 11 - evaluate (not yet solved)
# TODO: implement

# Step 12 - sample (not yet solved)
# TODO: implement

