# Toy Automatic Differentiation Framework

## Overview

This project is a simple toy framework designed to illustrate the principles of automatic differentiation using Python. It provides a basic implementation of forward and backward propagation through a computational graph, allowing users to define mathematical expressions and compute their derivatives with ease. The framework uses a node class (`Node`) to represent variables and operations within expressions, as well as an operation class (`Op`) to define executable operations. 

**This project is primarily educational and serves as an introductory tool for understanding automatic differentiation.** 

## Documentation

We have provided a [instruction.md](./doc/instruction.md) in Chinese. Due to limited personal time, this document is not currently translated into English. If anyone is willing to contribute a translation, we welcome a pull request.

## Usage

This project is pure Python and just requires a `numpy` module for testing. 

Ensure your Python environment is version 3.x or later.

### Defining Variables and Operations

```python
# Create variable nodes
import numpy as np
x1 = Var(2., 'x1')
x2 = Var(3., 'x2')

# The variable can be a numpy array:
# x1 = Var(np.array([1,2,3]), 'x1')
# x2 = Var(np.array([2,2,6]), 'x2')

# Define operations (multiplication and addition)
y = x1 * x2 + x2 * 4.3 + 30
```

### Executing Backward Propagation

```python
# Calculate gradients
ex = Executor(y)
ex.gradient()
print("x1.grad=" , x1.grad)
print("x2.grad=" , x2.grad)
```

## Testing

The framework includes several test functions to demonstrate its functionality:

- `__test_topo_sort()`: Tests the topological sorting of the computational graph.
- `__test_FB_pass()`: Tests the forward and backward pass.
- `__test_ad()`: Tests the automatic differentiation with various examples.

These tests can be run individually to verify the correctness of the framework.

## Acknowledgments

This project was inspired by the work listed below. Our sincere appreciation goes to the authors of the following resources:

- **[CSE 599W: Systems for ML](https://dlsys.cs.washington.edu/)** at the University of Washington, where the course materials, including the sample code, provided a practical foundation for understanding the implementation of automatic differentiation. 
- **"[Understanding Deep Learning](https://udlbook.github.io/udlbook/)"** by Simon J.D. Prince, published by MIT Press, which offered profound insights into the theory and practice of deep learning. The Chapter 7 of this book were especially helpful in developing a clear understanding of the backward propagation algorithm.

### License

This project is licensed under the MIT License.

