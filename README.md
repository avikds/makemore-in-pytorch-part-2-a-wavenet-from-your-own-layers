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
- [x] **12.** sample

## Results

```
WaveNet parameters: 22397
initial dev loss: 3.304 vs ln 27 = 3.2958
  tanh layer 4: std 0.61, saturated 5.4%
  tanh layer 8: std 0.61, saturated 4.7%
  tanh layer 12: std 0.63, saturated 3.3%
smoothed loss every 2,000 steps: [2.397, 2.255, 2.225, 2.2, 2.168, 2.161, 2.137, 2.136, 2.101, 2.083]
final log10 update ratios: [-3.95, -3.29, -3.17, -3.15, -2.87]
WaveNet train loss: 2.0678
WaveNet dev loss:   2.0957
flat MLP parameters: 22097
flat MLP dev loss:   2.1065
WaveNet test loss: 2.0947 (checked once, at the end)
WaveNet samples: ['dexze', 'moulier', 'ilazi', 'ahnell', 'imitta', 'nella', 'kaka', 'da']
```
