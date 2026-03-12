# Educational Autograd

This repository is an educational project demonstrating how **automatic differentiation** works.

The goal of the project is to immerse the reader in the fundamentals of autograd systems through simple and clear examples, without the excessive complexity of industrial libraries.

## Core Concepts

Currently implemented:
- **Forward-mode Automatic Differentiation** using `Dual Numbers`.

## Structure

- `dual/`: Module containing the implementation of dual numbers and basic mathematical functions for them.
- `examples/`: Examples of using the library to calculate derivatives of complex functions.

## Usage

An example of calculating a derivative using dual numbers can be found in `examples/hard_function_derivative.py`.

```python
from dual.DualNumber import DualNumber
from dual.functions import sin

# Example of calculating the derivative of sin(x) at x = 2
x = DualNumber(2, 1) # 2 - value, 1 - 'seed' for the derivative
y = sin(x)
print(f"Value: {y.real}, Derivative: {y.dual}")
```
