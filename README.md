# Equivariant-Vision-Network-Demo

This project explores the use of **equivariant convolutional neural networks (E-CNNs)** to predict the number of planets in a planetary system using simulated astronomical imaging data. The model leverages the `e2cnn` library to handle rotational symmetries that are naturally present in telescope data.

![Screenshot 2025-04-05 174652](https://github.com/user-attachments/assets/1d90e6ad-1b8e-496c-b960-2a2b73f3b574)

---

## Project Goal

The architecture of planetary systems (number of planets, their configuration) offers key insights into how solar systems form and evolve. This project uses **symmetry-aware deep learning** to:
- Detect and count planets in noisy or rotated images
- Build a model that is robust to spatial orientation
- Demonstrate rotational equivariance using group theory

---

## Model Overview

The model is built using [`e2cnn`](https://github.com/QUVA-Lab/e2cnn) and includes:
- Cyclic group symmetry `C8` for 8-fold rotational invariance
- Multiple equivariant convolution layers
- Equivariant batch normalization
- Group pooling and a final fully connected layer for regression

---

## Dataset (Simulated)

Synthetic 64×64 grayscale images are generated using NumPy. Each "planet" is represented as a bright blob placed randomly in the image. The number of blobs = number of planets (1–5), forming the ground truth.

---

## Dependencies

```bash
pip install torch numpy matplotlib e2cnn
```
## How It Works
Generate synthetic astronomical images with random blobs

Normalize the number of planets to the range [0, 1]

Train an Equivariant CNN using e2cnn with group convolutions

Unnormalize predictions to get planet count

Visualize model performance using a scatter plot
