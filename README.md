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


#day 10


# Model Evaluation and Overfitting Strategies in PyTorch

This guide consolidates key concepts for evaluating deep learning models and combating overfitting using PyTorch. It covers data splits, loss tracking, accuracy metrics, and regularization techniques such as dropout, weight decay, and data augmentation.

---

## 1. Data Splits: Train, Validation, Test

Effective model evaluation begins with proper data partitioning:

- **Training set**: Used to update model parameters (weights and biases) during learning.
- **Validation set**: Used to tune hyperparameters and monitor generalization performance during training.
- **Test set**: Used for final performance evaluation after training is complete.

Tracking performance across these splits helps detect overfitting and ensures the model generalizes well to unseen data.

---

## 2. Training and Validation Loss

Loss functions measure how far off the model's predictions are from the true labels.

- **Training loss** is computed during each epoch by summing the loss across all training batches. A decreasing training loss indicates the model is fitting the training data.
- **Validation loss** is computed after each epoch using the validation set. It reflects how well the model generalizes. Validation loss is calculated in evaluation mode to disable training-specific behaviors like dropout.

If training loss continues to decrease while validation loss increases, the model may be overfitting.

---

## 3. Detecting Overfitting

Overfitting occurs when a model performs well on training data but poorly on validation or test data. It memorizes patterns instead of learning generalizable features.

**Common signs of overfitting**:
- Training loss decreases steadily, but validation loss starts increasing.
- High training accuracy, low validation accuracy.
- Validation performance plateaus or worsens while training continues to improve.

**Underlying causes**:
- Small or insufficiently diverse dataset.
- Model with too many parameters relative to data size.
- Lack of regularization.
- Noisy or mislabeled data.
- Poor preprocessing (e.g., unnormalized inputs).

---

## 4. Accuracy and Evaluation Metrics

Accuracy measures how often the model's predictions match the true labels. It is especially useful for classification tasks.

For multi-class classification:
- The predicted class is selected using the highest probability score.
- Accuracy is calculated by comparing predicted classes with true labels.

However, accuracy alone may not be sufficient. Additional metrics provide deeper insight:

| Metric           | Description                                      | Use Case                                  |
|------------------|--------------------------------------------------|--------------------------------------------|
| Precision        | True positives / predicted positives             | Important when false positives are costly  |
| Recall           | True positives / actual positives                | Important when false negatives are costly  |
| F1 Score         | Harmonic mean of precision and recall            | Balanced view of precision and recall      |
| ROC-AUC          | Measures ranking quality for binary classifiers  | Useful for imbalanced datasets             |
| Confusion Matrix | Breakdown of prediction errors                   | Visual insight into misclassifications     |

---

## 5. Strategies to Fight Overfitting

Overfitting can be mitigated using several techniques that encourage generalization and reduce model reliance on specific patterns.

### A. Dropout Regularization

Dropout randomly deactivates a fraction of neurons during training. This prevents the model from relying too heavily on any single neuron and encourages it to learn redundant, distributed representations.

- Typically applied after activation functions.
- Common dropout rates range from 0.2 to 0.5.
- Dropout is active during training and disabled during evaluation.

Dropout acts like training an ensemble of sub-networks, improving robustness and generalization.

### B. Weight Decay (L2 Regularization)

Weight decay penalizes large weights by adding a regularization term to the loss function. This discourages the model from fitting noise or overly complex patterns.

- The penalty term is proportional to the square of the weights.
- Helps keep model parameters small and stable.
- Stronger weight decay leads to stronger regularization, but excessive decay may cause underfitting.

**Loss function with weight decay**:
Total Loss = Original Loss + lambda × sum of squared weights

Weight decay is especially effective in deep networks and complements other regularization techniques.

### C. Data Augmentation

Data augmentation artificially increases dataset size and diversity by applying transformations to existing samples. It is particularly effective for image and text data.

**For image data**:
- Rotation, flipping, scaling, cropping, brightness adjustment

**For text data**:
- Synonym replacement, back translation, sentence shuffling

**For tabular data**:
- SMOTE (Synthetic Minority Over-sampling Technique), noise injection, feature masking

Augmentation introduces variability that helps the model learn more robust features and reduces the risk of memorization.

---

## 6. Best Practices for Evaluation and Regularization

- Always switch between training (`model.train()`) and evaluation (`model.eval()`) modes appropriately.
- Use `torch.no_grad()` during validation to disable gradient tracking and improve efficiency.
- Track both loss and accuracy across training and validation sets.
- Use early stopping to halt training when validation loss begins to increase.
- Combine multiple regularization strategies for stronger generalization.
- Monitor additional metrics like precision, recall, and F1 score to understand model behavior beyond accuracy.

---

## 7. Summary Table

| Component         | Purpose                                      | Notes                                      |
|------------------|----------------------------------------------|--------------------------------------------|
| Training loss     | Tracks model fit on training data            | Should decrease over time                  |
| Validation loss   | Tracks generalization performance            | Should decrease, then plateau              |
| Accuracy          | Measures prediction correctness              | Useful for classification tasks            |
| Dropout           | Prevents neuron co-adaptation                | Applied during training only               |
| Weight Decay      | Penalizes large weights                      | Encourages smoother functions              |
| Data Augmentation | Expands dataset diversity                    | Applied only to training data              |

---

## 8. Final Takeaways

- Overfitting is a generalization failure, not a training success.
- Regularization techniques like dropout and weight decay help constrain model complexity.
- Data augmentation simulates a larger dataset and improves robustness.
- Evaluation is not just about metrics — it's about understanding how your model behaves across different data splits.
- A well-regularized model balances fit and generalization, making it reliable on unseen data.

# Performance Optimization Strategy for Deep Learning Models

This document outlines a three-step strategy to maximize model performance on image classification tasks. The approach begins with overfitting the training set to confirm solvability, followed by reducing overfitting to improve generalization, and concludes with hyperparameter fine-tuning.

---

## Step 1: Overfit the Training Set

### Purpose
- Confirm that the model can learn the task
- Catch implementation bugs early
- Establish a performance baseline

### Procedure
- Start with a **single data point** to verify solvability
- Modify the training loop to repeatedly train on that example
- Expect near-zero loss and 100% accuracy if the model is correctly configured
- Once validated, scale up to the **entire training set**
- Use a model architecture large enough to overfit
- Keep default hyperparameters (e.g., learning rate)

---

## Step 2: Reduce Overfitting

### Goal
- Improve generalization to unseen validation data

### Techniques
- **Dropout**: Randomly disables neurons during training
- **Data Augmentation**: Introduces variability in training samples
- **Weight Decay**: Penalizes large weights to encourage simpler models
- **Reduce Model Capacity**: Use fewer layers or parameters

### Trade-offs
- Regularization may reduce training accuracy
- Excessive regularization can harm both training and validation performance
- Balance is key: monitor metrics closely to find optimal configuration

---

## Step 3: Fine-Tune Hyperparameters

### Objective
- Refine model behavior for best validation performance

### Common Targets
- **Learning Rate**
- **Momentum**
- **Weight Decay**

### Search Strategies

#### Grid Search
- Tests fixed values across a defined range
- Example:
  - Momentum: 0.85 to 0.99
  - Learning Rate: \(10^{-2}\) to \(10^{-6}\)

#### Random Search
- Samples values randomly within a range
- Example:
  - `np.random.uniform(2, 6)` selects a value between 2 and 6
- Often more efficient than grid search
- Increases likelihood of discovering optimal settings

---

## Summary of Optimization Workflow

| Step                     | Purpose                                      | Techniques Used                          |
|--------------------------|----------------------------------------------|------------------------------------------|
| Overfit Training Set     | Confirm solvability and catch bugs           | Single sample training, large model      |
| Reduce Overfitting       | Improve generalization                       | Dropout, augmentation, weight decay      |
| Fine-Tune Hyperparameters| Maximize validation performance              | Grid search, random search               |

---

## Final Notes

Effective model training requires a balance between capacity and generalization. Begin by ensuring the model can learn, then apply regularization to reduce overfitting, and finally fine-tune hyperparameters to reach optimal performance. Each step builds on the previous, forming a robust and reproducible optimization pipeline.



# Day 15: Unified Deep Learning Workflow in PyTorch

This document summarizes a full-cycle implementation of a deep learning workflow using PyTorch. It integrates synthetic data generation, model design, training, evaluation, activation analysis, dropout behavior, and hyperparameter tuning. The session also includes a structured performance optimization strategy applicable to both classification and regression tasks.

---

## 1. Synthetic Data Setup

- Generated a dataset with 100 samples and 4 numerical features.
- Created two targets:
  - **Binary classification label** based on a rule involving the first two features.
  - **Continuous regression target** simulating a salary-like prediction.
- Standardized the regression target using `StandardScaler` for stable learning.
- Combined inputs and targets into a unified dataset and batched using `DataLoader`.

---

## 2. Model Architecture: SmartNet

Designed a dual-headed neural network with shared layers and two output branches:

### Shared Feature Extractor
- Dense layers with ReLU and LeakyReLU activations.
- Dropout applied to prevent overfitting.

### Classification Head
- Outputs logits for two classes (binary classification).

### Regression Head
- Outputs a single continuous value.

The model supports simultaneous learning of classification and regression tasks from shared representations.

---

## 3. Training Strategy

- Used **CrossEntropyLoss** for classification and **MSELoss** for regression.
- Combined both losses with a weighted sum to balance learning objectives.
- Trained for 15 epochs using the **Adam optimizer**.
- Tracked classification accuracy using `torchmetrics.Accuracy`.

Each epoch reported classification loss, regression loss, and accuracy.

---

## 4. Evaluation and Inference

- Switched to evaluation mode to disable dropout and gradient tracking.
- Performed inference on a sample input:
  - Predicted class label using softmax and argmax.
  - Predicted regression value with inverse scaling applied.

This confirmed the model’s ability to generalize to unseen data.

---

## 5. Activation Function Comparison

Visualized the behavior of common activation functions:

| Activation | Description |
|------------|-------------|
| ReLU       | Zeroes out negatives, fast and sparse |
| LeakyReLU  | Allows small gradient for negatives |
| Sigmoid    | Smooth curve, bounded between 0 and 1 |
| Tanh       | Centered at zero, saturates at extremes |

This comparison aids in selecting appropriate activations for different layers.

---

## 6. Dropout Behavior Analysis

Demonstrated the effect of dropout:

- In **training mode**, dropout randomly disables neurons.
- In **evaluation mode**, dropout is inactive, producing stable outputs.

This illustrates how dropout contributes to regularization during training.

---

## 7. Hyperparameter Search Visualization

Simulated a random search over learning rates and momentum values:

- Learning rates sampled logarithmically.
- Momentum values sampled uniformly.
- Visualized using a scatter plot to explore the search space.

This supports efficient tuning of optimizer settings.

---

## 8. Performance Optimization Strategy

Outlined a three-step strategy to maximize model performance:

### Step 1: Overfit the Training Set
- Train on a single data point to verify solvability.
- Scale up to full training set using a large-capacity model.

### Step 2: Reduce Overfitting
- Apply regularization techniques:
  - Dropout
  - Data augmentation
  - Weight decay
  - Reduced model capacity
- Monitor validation accuracy to balance generalization and learning.

### Step 3: Fine-Tune Hyperparameters
- Target learning rate, momentum, and weight decay.
- Use grid search or random search to explore parameter space.
- Random search often yields better results with fewer trials.

---

## 9. Summary of Concepts Covered

| Component               | Description                                             |
|-------------------------|---------------------------------------------------------|
| Synthetic Data          | Rule-based binary and continuous targets                |
| Dual-Headed Model       | Shared layers with classification and regression heads  |
| Training Loop           | Combined loss, optimizer, and metric tracking           |
| Evaluation              | Inference with softmax and inverse scaling              |
| Activation Analysis     | Visual comparison of ReLU, LeakyReLU, Sigmoid, Tanh     |
| Dropout Behavior        | Demonstrated difference between train and eval modes    |
| Hyperparameter Tuning   | Random search over optimizer settings                   |
| Optimization Strategy   | Overfit → Regularize → Fine-tune                        |

---

## 10. Final Notes

This session represents a complete synthesis of foundational deep learning concepts using PyTorch. By integrating classification and regression tasks, visual diagnostics, and performance tuning, the workflow demonstrates a robust and reproducible approach to model development. It serves as a wrap-up of the introductory deep learning curriculum and establishes a strong baseline for future experimentation and deployment.

