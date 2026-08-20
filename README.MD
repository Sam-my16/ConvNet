# CNN Forward Pass — NumPy from Scratch and GoogLeNet

This repository contains two related projects focused on understanding and reproducing Convolutional Neural Networks (CNNs), starting from a NumPy implementation and progressing toward a PyTorch implementation of GoogLeNet.

The goal is to build intuition about how CNNs work internally before relying on high-level deep learning frameworks.

## Projects

### Part 1 — CNN Forward Pass from Scratch

Implements the **forward pass of a Convolutional Neural Network using only NumPy**.

The implementation is deliberately loop-based rather than optimized or vectorized, prioritizing clarity and understanding over performance.

The project covers the fundamental operations behind convolutional networks:

* Zero padding
* Convolution
* Filter application
* Bias addition
* Max and average pooling
* Forward propagation through convolutional and pooling layers

Backpropagation is not implemented as part of the main exercise because, while conceptually straightforward, it adds considerable implementation complexity.

### Part 2 — GoogLeNet / Inception

Started reproducing the **GoogLeNet architecture** in PyTorch based on the original paper:

> Szegedy et al., *Going Deeper with Convolutions*
> https://arxiv.org/abs/1409.4842

At the current stage, the **Inception module** has been implemented.

The Inception module processes the same input through multiple branches with different operations and concatenates their outputs along the channel dimension:

```text
                    ┌── 1×1 → 3×3 ──┐
                    │                │
                    ├── 1×1 → 5×5 ──┤
Input ──────────────┼── Pool → 1×1 ──┼── Concatenate
                    │                │
                    └────── 1×1 ─────┘
```

The current implementation also includes:

* Global Average Pooling
* A flattened hidden representation
* Classification logits

For an input tensor of shape `(B, 3, 32, 32)`, the implemented Inception module produces a hidden representation suitable for reuse in later image-captioning experiments.

## Repository Structure

### Part 1 — Try It Yourself

Contains the functions to implement, together with tests designed to verify the forward-pass operations.

### Part 2 — Solutions

Contains reference implementations and optional backward-pass functions.

The GoogLeNet section contains the ongoing PyTorch reproduction of the architecture.

## Functions Implemented in NumPy

### `zero_pad`

Pads the height and width dimensions of an image with zeros.

### `conv_single_step`

Applies a single convolutional filter to one slice of the input by performing element-wise multiplication, summation, and bias addition.

### `conv_forward`

Implements the forward pass of a complete convolutional layer by applying multiple filters across all spatial positions and input channels.

### `pool_forward`

Implements max pooling and average pooling during the forward pass.

## Requirements

```text
numpy
h5py
matplotlib
torch
```

## Notes

The NumPy implementation uses explicit loops rather than vectorized operations. This is intentional: the purpose of the project is **to understand the mechanics of CNNs**, not to achieve computational efficiency.

The PyTorch implementation then builds on this intuition to reproduce a more complex real-world CNN architecture.

The project is developed incrementally, with emphasis on understanding the mathematical and architectural ideas behind each component before moving to higher-level implementations.
