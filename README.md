# Neural Networks from Scratch - MNIST Digit Classification

A pure Python implementation of neural networks for MNIST digit classification, built from scratch to understand the fundamental concepts behind deep learning.

## Overview

This repository contains two Jupyter notebooks that implement neural networks without relying on high-level frameworks like TensorFlow or PyTorch. The implementations focus on educational value and understanding the mathematical foundations of neural networks.

### Files

- **`binary_classifier.ipynb`** - Binary classification between two handwritten MNIST digits
- **`multi_class_classifier.ipynb`** - Multi-class classification for all 10 MNIST digits (0-9)

## Purpose

These implementations are designed to solidify understanding of:

- Forward propagation through neural network layers
- Backpropagation algorithm and gradient computation
- Loss functions (binary cross-entropy vs categorical cross-entropy)
- Activation functions (sigmoid, ReLU, softmax)
- Weight initialization strategies
- Training loops and optimization
- Performance evaluation metrics

## Binary Classifier (`binary_classifier.ipynb`)

### What it does
Distinguishes between two specific MNIST digits (e.g., classifying whether a handwritten digit is a "1" or an "9").

### Key Learning Scenarios
- **Medical Diagnosis**: Similar to how a model might classify X-rays as "normal" vs "abnormal"
- **Email Filtering**: Like determining if an email is "spam" vs "not spam"
- **Quality Control**: Such as classifying manufactured parts as "defective" vs "acceptable"

### Implementation Details
- Uses sigmoid activation for binary output
- Implements binary cross-entropy loss function
- Single output neuron (probability between 0 and 1)
- Demonstrates basic gradient descent optimization

### Mathematical Concepts Covered
```
Forward Pass: z = W·x + b → a = σ(z)
Loss: L = -[y·log(a) + (1-y)·log(1-a)]
Backprop: dW = (a-y)·x, db = (a-y)
```

## Multi-Class Classifier (`multi_class_classifier.ipynb`)

### What it does
Classifies handwritten digits into all 10 possible classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

### Key Learning Scenarios
- **Image Recognition**: Like identifying different types of objects in photos
- **Document Classification**: Such as categorizing news articles by topic (sports, politics, technology, etc.)
- **Voice Recognition**: Similar to classifying spoken words or commands
- **Handwriting Recognition**: Exactly what this implementation does - recognizing written digits

### Implementation Details
- Uses softmax activation for multi-class output
- Implements categorical cross-entropy loss function
- 10 output neurons (one probability per digit class)
- Demonstrates one-hot encoding for target labels

### Mathematical Concepts Covered
```
Softmax: p_i = e^(z_i) / Σ(e^(z_j))
Loss: L = -Σ(y_i · log(p_i))
Backprop: dW = (p-y) ⊗ x (outer product)
```

## Core Concepts Implemented

### 1. Neural Network Architecture
- Input layer (784 neurons for 28×28 pixel images)
- Hidden layer(s) with ReLU activation
- Output layer with appropriate activation (sigmoid/softmax)

### 2. Forward Propagation
Hand-coded matrix operations for:
- Linear transformations: `Z = W·X + b`
- Activation functions: `A = f(Z)`
- Layer-by-layer computation

### 3. Backpropagation
Manual implementation of:
- Loss function derivatives
- Weight updates using gradient descent

### 4. Training Process
- Data preprocessing and normalization
- Mini-batch processing
- Iterative optimization
- Loss monitoring and convergence

## Real-World Applications

### Binary Classification Use Cases
- **Fraud Detection**: Credit card transaction (fraudulent vs legitimate)
- **Medical Screening**: Test result (positive vs negative)
- **A/B Testing**: User preference (version A vs version B)

### Multi-Class Classification Use Cases
- **Autonomous Vehicles**: Traffic sign recognition (stop, yield, speed limit, etc.)
- **Healthcare**: Disease classification from symptoms or scans
- **E-commerce**: Product categorization (electronics, clothing, books, etc.)
- **Natural Language Processing**: Sentiment analysis (very negative, negative, neutral, positive, very positive)

## Prerequisites

```python
from fastai.vision.all import *
```

## Getting Started

1. Clone this repository
2. Install required dependencies (fastai)
3. Open either notebook in Jupyter
4. Run cells sequentially to see the implementation in action

### For Binary Classification:
```bash
jupyter notebook binary_classifier.ipynb
```

### For Multi-Class Classification:
```bash
jupyter notebook multi_class_classifier.ipynb
```

## Learning Outcomes

After working through these notebooks, you will understand:

- How neural networks make predictions (forward pass)
- How neural networks learn from mistakes (backward pass)
- The mathematical relationship between different loss functions
- Why different activation functions are used in different scenarios
- How to implement gradient descent from scratch
- The difference between binary and multi-class classification approaches

## Implementation Philosophy

This code prioritizes:
- **Clarity over efficiency** - Every step is explicit and commented
- **Educational value over performance** - Focus on understanding concepts
- **Mathematical transparency** - All formulas are implemented manually
- **Minimal dependencies** - Using basic NumPy operations to see the math


## Contributing

Feel free to suggest improvements or additional educational examples that would help solidify neural network concepts!

---
*Built for learning, optimized for understanding* 🧠
