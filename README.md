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

# Day 6 – Deep Learning Essentials (80/20 Focus)

## Neural Network Building Blocks

* **Tensors**: Fundamental data structure, generalization of scalars, vectors, and matrices.
* **Neurons**: Receive inputs, apply weighted sums and non-linear activation, produce outputs.
* **Layers**: Stacking neurons into input, hidden, and output layers creates the network.

## Forward Propagation

* Input data passes through weights and biases.
* Non-linear activations (ReLU, Sigmoid, Tanh) introduce complexity beyond linear relationships.
* Output layer provides predictions.

## Loss Functions

* Measure difference between predicted and actual values.
* Examples: Cross-entropy for classification, Mean Squared Error for regression.

## Backpropagation & Gradient Descent

* **Backpropagation**: Calculates gradients of loss with respect to weights.
* **Gradient Descent**: Iteratively updates weights in the opposite direction of the gradient to minimize loss.

## Key Concepts Driving Deep Learning Power

* **High-dimensional pattern recognition**: Neural nets excel at extracting features from complex data.
* **Representation learning**: Layers progressively transform raw inputs into abstract features.
* **Scalability**: Large datasets and GPUs enable training of deep and accurate models.

## Practical Takeaways

* Deep learning extends traditional machine learning by automatically learning features.
* Success depends on quality of data, choice of architecture, and effective optimization.
'



#day 7
# Using Derivatives to Update Model Parameters

## 1. Why We Need Derivatives

Once we calculate the loss, we need a way to reduce it.  
Derivatives (also called gradients in deep learning) help us figure out how to adjust the model’s weights and biases to make better predictions.  
They tell us the direction and steepness of change — like a slope on a hill.

---

## 2. Visual Analogy: Loss as a Valley

Imagine the loss function as a valley:

- The height of the valley represents the loss value
- The slope tells us how steep the curve is at a point
- A steep slope means the loss is changing quickly
- A flat slope means the loss is stable

We want to reach the **lowest point** in the valley — this is where the loss is minimal.


   ![image.png](attachment:dd9e84d1-a421-4d14-b2ef-2896c90995f2.png)

- Red arrows show steep slopes -> large gradient -> big update steps
- Green arrows show gentle slopes -> small gradient -> small update steps
- Blue arrow at the valley floor -> slope is zero -> gradient is zero -> we’ve reached the minimum

---

## 3. Convex vs Non-Convex Functions
![image.png](attachment:14d49cf8-aa8a-4c70-9bfb-d2ce9cdadb85.png)

- **Convex function**: has one global minimum. Easy to find and optimize.
- **Non-convex function**: has many local minima. These are low points, but not the lowest possible.

Loss functions in deep learning are usually **non-convex** because of complex layer interactions.  
Our goal is to get as close as possible to the **global minimum**, even if we pass through local dips along the way.

---

## 4. How Derivatives Connect to Training

During training:

![image.png](attachment:352342a7-8872-4e52-957c-b317903ade13.png)

- We run a **forward pass** to get predictions
- We calculate the **loss** by comparing predictions to actual labels
- We then run a **backward pass** to compute gradients

These gradients tell us how each weight and bias contributed to the error.  
We use them to adjust the parameters so the model improves over time.

---

## 5. What Are Gradients?

In deep learning, derivatives are called **gradients**.  
They measure how much the loss changes when we tweak a specific parameter.

- A large gradient means the parameter has a big impact on the loss
- A small gradient means the parameter has little effect

We use gradients to update weights and biases in the direction that reduces the loss.

---

## 6. Backpropagation: The Core Mechanism

Backpropagation is the process of computing gradients layer by layer, starting from the output and moving backward.

If we have a network with three layers:

![image.png](attachment:8bf3246a-7e4c-4da6-a39d-b1752cccf748.png)

- First, we compute gradients for the last layer (L2)
- Then for the middle layer (L1)
- Finally for the first layer (L0)

Each layer’s gradients depend on the layers that come after it.  
This chain of calculations is what makes deep learning powerful and efficient.

---

## 7. How Parameters Are Updated

Once we have gradients, we update each parameter like this:

- Multiply the gradient by a **learning rate**
- Subtract that value from the current weight or bias
![image.png](attachment:79889b85-59fe-42e6-8f7c-189d357dacfd.png)
This moves the parameter in the direction that reduces the loss.  
The learning rate controls how big each step is too big and we overshoot, too small and we move too slowly.

---

## 8. Gradient Descent: The Optimization Strategy

Gradient descent is the method we use to find the minimum of the loss function.

Steps:
- Calculate gradients
- Move parameters in the direction that reduces loss
- Repeat until the loss is low enough

There are many versions of gradient descent:
- **Batch gradient descent**: uses all data at once
- **Stochastic gradient descent (SGD)**: uses one sample at a time
- **Mini-batch gradient descent**: uses small groups of samples

Most deep learning frameworks use **SGD with mini-batches** for speed and stability.

---

## 9. Optimizers: Automating the Updates

Instead of manually updating parameters, we use **optimizers**.  
They handle gradient calculations and parameter updates for us.

Popular optimizers:
- **SGD**: basic and reliable
- **Adam**: adaptive learning rates, faster convergence
- **RMSprop**: good for noisy data

Optimizers use the gradients and learning rate to update all model parameters automatically.  
This is what makes training scalable and efficient.

---

## Summary

- Derivatives (gradients) tell us how to reduce loss
- Loss is like a valley — we want to reach the lowest point
- Backpropagation computes gradients layer by layer
- Gradients are used to update weights and biases
- Gradient descent is the strategy to minimize loss
- Optimizers automate the update process

This is the heart of how neural networks learn. Next step: training loops and learning rate tuning.



#day 8
# Deep Learning with PyTorch — Learning Summary

## Data Loading and Preparation

Efficient data loading ensures:
- Smooth training
- Faster computation
- Better generalization

---

## Animal Classification Dataset

**Features:**
- hair, feathers, eggs, milk, predator, legs, tail

**Target:**
- type (e.g., bird → 0, mammal → 1, reptile → 2)

**Ignored:**
- animal_name (not useful for prediction)

---

## Feature and Label Extraction

- Use `.iloc` to select relevant columns
- Convert to NumPy arrays for speed and compatibility
- Encode labels using `LabelEncoder`

---

## Tensor Conversion and Dataset Wrapping

- Convert features and labels to PyTorch tensors
- Wrap them using `TensorDataset`
- Access samples using indexing (e.g., `dataset[0]`)

---

## DataLoader for Batching and Shuffling

**Parameters:**
- `batch_size`: controls how many samples per training step
- `shuffle`: randomizes data order each epoch

**Why batching matters:**
- Speeds up training
- Reduces memory usage
- Stabilizes learning

**Why shuffling matters:**
- Prevents memorization of data order
- Improves generalization

---

## Salary Prediction with Deep Learning

### Preprocessing
- Encode categorical columns
- Impute missing values
- Scale features and target using `StandardScaler`

### Tensor Conversion
- Convert scaled arrays to PyTorch tensors

---

## Model Architecture (Regression)

- Input layer based on feature count
- Hidden layers with ReLU activation
- Output layer with one neuron for regression

**Example architecture:**
- 64 → 32 → 16 → 1 neurons

---

## Loss Function and Optimizer

- Use `MSELoss` for regression
- Use `SGD` or `Adam` for optimization

---

## Training Loop

- Iterate over epochs
- For each batch:
  - Predict
  - Compute loss
  - Backpropagate
  - Update weights
- Print loss after each epoch

---

## Key Learnings

- PyTorch training loop is manual and flexible
- Label encoding is essential for categorical features
- Scaling improves model stability and convergence
- MSE loss in scaled space ≈ small decimals → good performance
- Inverse transform predictions to get real-world salary values

---

## Real-World Evaluation

- Use inverse transform to recover actual salary predictions
- Compute metrics like Mean Absolute Error (MAE) in USD

---

## Reinforcement Insights

| Component       | Purpose                                      |
|----------------|----------------------------------------------|
| CSV file        | Raw data source                              |
| DataFrame       | Organize and manipulate data                 |
| LabelEncoder    | Convert categories to numeric codes          |
| StandardScaler  | Normalize features and targets               |
| TensorDataset   | Wraps features and labels for PyTorch        |
| DataLoader      | Manages batching and shuffling               |
| nn.Sequential   | Builds the neural network                    |
| MSELoss         | Measures prediction error                    |
| Optimizer       | Updates model weights                        |

---

## Summary

You built a full PyTorch regression pipeline:
- From raw CSV to scaled tensors
- From label encoding to deep learning
- From training loop to real-world salary predictions

You're officially deep learning fluent in PyTorch.






# day 9
## Quick Recall — October 6, 2025

### Activation Functions
- **ReLU**: Outputs input if positive, else zero. Fast, avoids vanishing gradients. Best for hidden layers.
- **Leaky ReLU**: Like ReLU but allows small negative outputs. Prevents dead neurons.
- **Sigmoid**: Maps input to (0, 1). Good for binary classification output. Saturates for large/small inputs.
- **Softmax**: Converts scores to probabilities summing to 1. Used in multi-class output layers. Also saturates.

### Optimization: Learning Rate and Momentum
- **Learning Rate**: Controls step size during weight updates.  
  - Too high → unstable, overshooting  
  - Too low → slow convergence  
  - Optimal → smooth descent
- **Momentum**: Adds inertia to updates.  
  - Helps escape local minima  
  - Smooths optimization  
  - Typical range: 0.85–0.99

### Loss Landscapes
- **Convex**: Single global minimum, easy to optimize.
- **Non-convex**: Multiple local minima and saddle points. Requires careful tuning and momentum to avoid getting stuck.

### Layer Initialization
- Initial weights should be small to prevent unstable outputs.
- PyTorch uses default ranges (e.g., -0.125 to +0.125).
- Custom initialization via `torch.nn.init` (uniform, normal, Xavier, He).
- Initialization method should match activation function.

### Transfer Learning
- Reuse pretrained model weights for a new task.
- Saves time, improves performance, especially with limited data.
- Use `torch.save` and `torch.load` to manage weights.

### Fine-Tuning
- Load pretrained weights and train further with a smaller learning rate.
- Freeze early layers (set `requires_grad = False`) and update later layers.
- Useful when new task is similar to original.

### Summary
Today’s focus was on understanding how neural networks learn and optimize:
- Activation functions shape signal flow and gradient behavior.
- Learning rate and momentum control convergence dynamics.
- Initialization, transfer learning, and fine-tuning improve training efficiency and reuse.

