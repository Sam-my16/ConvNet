# CNN Forward Pass — NumPy from Scratch

Implements the **forward pass** (since backprop is not that hard but tedious) of a Convolutional Neural Network using only NumPy, to build intuition before using frameworks like TensorFlow (more on that soon to come).

## Structure

1. **Part 1 — Try It Yourself**: functions to implement + tests.
2. **Part 2 — Solutions**: reference implementations, plus optional backward-pass functions.

## Functions implemented

- **`zero_pad`** — pads images with zeros on height/width.
- **`conv_single_step`** — applies one filter to a single slice (multiply + sum + bias).
- **`conv_forward`** — full conv layer forward pass across all positions/filters.
- **`pool_forward`** — max/average pooling forward pass.


## Requirements

```
numpy
h5py
matplotlib
```

## Notes

Loop-based, not vectorized, clarity over performance. 