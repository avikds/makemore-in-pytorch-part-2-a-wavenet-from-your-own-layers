# makemore in PyTorch, Part 2: A WaveNet From Your Own Layers

Continue Andrej Karpathy's makemore by writing the layers the later lectures build: a Linear layer with Kaiming initialization, a Tanh layer you can inspect for saturation, and a BatchNorm1d with learned gain and bias, running statistics and a training flag that handles 3-D inputs correctly. Add Embedding, FlattenConsecutive and a Sequential container, and assemble the hierarchical WaveNet that fuses eight characters two at a time. Train it while watching update-to-data ratios, smooth the loss curve, evaluate in evaluation mode, compare it with a flat MLP of the same size, and sample new names.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_splits
- [x] **2.** Linear
- [x] **3.** Tanh and activation_stats
- [x] **4.** BatchNorm1d
- [x] **5.** Embedding
- [x] **6.** FlattenConsecutive
- [x] **7.** Sequential
- [x] **8.** build_wavenet
- [x] **9.** train
- [x] **10.** smooth
- [x] **11.** evaluate
- [ ] **12.** sample

---

Built on Deep-ML.
