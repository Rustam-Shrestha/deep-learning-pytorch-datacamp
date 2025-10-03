# Day 5: Deep Learning with PyTorch

## 1. Introduction

**Instructor**: Jasmin, Senior Data Science Content Developer at DataCamp  
Today I explored the foundations of deep learning and how PyTorch supports building neural networks.

---

## 2. Why Deep Learning Matters

Deep learning powers many modern technologies:

- Language translation
- Self-driving cars
- Medical diagnostics
- Chatbots

These systems rely on neural networks to learn patterns from large datasets.

---

## 3. What Is Deep Learning?

Deep learning is a subset of machine learning. Its structure includes:

- Inputs
- Hidden layers (one or more)
- Outputs

Inspired by the human brain, these models use interconnected units called **neurons**.  
They require large datasets and high computational power to learn effectively.

---

## 4. PyTorch Overview

PyTorch is a popular deep learning framework:

- Originally developed by Meta AI
- Now maintained by the Linux Foundation
- Designed to be intuitive and similar to NumPy
- Supports dynamic computation graphs and GPU acceleration

---

## 5. Understanding Tensors

Tensors are the core data structure in PyTorch. They are:

- Multi-dimensional arrays (generalized matrices)
- Used to represent inputs, weights, and outputs
- Created from Python lists or NumPy arrays

---

## 6. Tensor Attributes

Key properties of tensors:

- `shape`: Dimensions of the tensor
- `dtype`: Data type (e.g., int64, float32)

These help ensure compatibility with models and assist in debugging.

---

## 7. Tensor Operations

Tensors support:

- **Addition and subtraction**: Requires matching shapes
- **Element-wise multiplication**: Multiplies corresponding elements
- **Matrix multiplication**: Combines rows and columns to produce new values

These operations are fundamental to how neural networks process data.

---

## 8. Practical Example: Temperature Adjustment

### What the `temperatures` tensor represents:

- A 2D grid of temperature values
- Each row could represent a day, each column a time or location

### What the `adjustment` tensor represents:

- A 2D grid of offsets to apply to the temperatures
- Same shape as `temperatures` for element-wise addition

### What the operation does:

- Adds each value in `adjustment` to the corresponding value in `temperatures`
- This is basic element-wise addition, resulting in adjusted temperature values

---

## 9. Tensor vs Matrix

- **Matrix**: A 2D array (rows × columns)
- **Tensor**: A general term for multi-dimensional arrays (0D to nD)

A matrix is just a 2D tensor. Tensors can represent more complex data like:

- Images (3D)
- Video frames (4D)
- Batches of inputs (nD)

---

## 10. Why Tensors Matter in Deep Learning

Tensors are preferred because:

- They are optimized for GPU computation
- They support automatic differentiation (for training)
- They can represent complex, high-dimensional data structures

---

## Summary

Today I learned how PyTorch uses tensors to build and train deep learning models.  
I explored tensor creation, attributes, operations, and how they differ from traditional matrices.  
This forms the foundation for building neural networks and understanding how data flows through them.


