"""
makemore in PyTorch, Part 2: A WaveNet From Your Own Layers scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""makemore in PyTorch, Part 2: a WaveNet from your own layers.

Story: rebuild the names dataset with eight characters of context, check that a fresh WaveNet
starts healthy, train it while watching the update-to-data ratios, compare it with a flat MLP
of the same size, and sample names from it.
"""
import math
import torch


def main():
    itos, splits = load_splits(block_size=8, seed=42)
    Xtr, Ytr = splits["train"]
    Xdev, Ydev = splits["dev"]
    Xte, Yte = splits["test"]

    # ---- 1. A healthy start ----
    model = build_wavenet(len(itos), n_embd=10, n_hidden=68, seed=42)
    print("WaveNet parameters:", sum(p.nelement() for p in model.parameters()))
    print("initial dev loss:", round(evaluate(model, Xdev, Ydev), 4), "vs ln 27 =", round(math.log(27), 4))
    model(Xtr[:1000])
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Tanh):
            s = activation_stats(layer.out)
            print(f"  tanh layer {i}: std {s['std']:.2f}, saturated {s['saturated']:.1f}%")

    # ---- 2. Train and watch the updates ----
    losses, ud = train(model, Xtr, Ytr, steps=20000, batch_size=32, lr=0.1, seed=0)
    print("smoothed loss every 2,000 steps:", [round(v, 3) for v in smooth(losses, 2000)])
    print("final log10 update ratios:", [round(r, 2) for r in ud[-1]])
    print("WaveNet train loss:", round(evaluate(model, Xtr, Ytr), 4))
    print("WaveNet dev loss:  ", round(evaluate(model, Xdev, Ydev), 4))

    # ---- 3. The same budget, flat ----
    g = torch.Generator().manual_seed(42)
    flat = Sequential([Embedding(len(itos), 10, generator=g), FlattenConsecutive(8),
                       Linear(80, 200, bias=False, generator=g), BatchNorm1d(200), Tanh(),
                       Linear(200, len(itos), generator=g)])
    with torch.no_grad():
        flat.layers[-1].weight *= 0.1
    for p in flat.parameters():
        p.requires_grad = True
    print("flat MLP parameters:", sum(p.nelement() for p in flat.parameters()))
    train(flat, Xtr, Ytr, steps=20000, batch_size=32, lr=0.1, seed=0)
    print("flat MLP dev loss:  ", round(evaluate(flat, Xdev, Ydev), 4))

    print("WaveNet test loss:", round(evaluate(model, Xte, Yte), 4), "(checked once, at the end)")
    print("WaveNet samples:", sample(model, itos, n=8, seed=2147483647))


if __name__ == "__main__":
    main()

