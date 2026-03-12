# Forward-mode Automatic Differentiation

## Why we need it?

As you know, we often need to calculate derivatives of functions. For example, if we want to find minimum of function $f(x)$, we can use gradient descent algorithm, which requires us to calculate derivative of $f$ at each step. 

But why we can't just use numeric approximation of derivative? For example, we can use formula:

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h}
$$

But this formula is not very accurate:

If we pick very small $h$, then we will have problem with floating point precision, because $f(x + h)$ and $f(x)$ will be very close to each other, and we will lose significant digits. 

On the other hand, if we pick large $h$, then we will have problem with approximation error, because this formula is only an approximation of derivative with error of order $O(h)$.

Yes, we can use more complex formulas t oget error order like $O(h^2)$ and even $O(h^4)$, but they will require more function evaluations, and also they will still have problem with floating point precision if we pick very small $h$.

In this project, we will describe how to calculate derivatives of functions with machine precision, without any approximation error and without any problem with floating point precision.

## Complex numbers

Maybe some of you know what is complex numbers. If we have equation like this:

$$
a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0 = 0
$$

sometimes we can decompose it into factors like this:

$$
a_n(x - r_1)(x - r_2)...(x - r_n) = 0
$$

But if we have equation like this:

$$
x^2 + 1 = 0
$$

we can't decompose it into linear factors with real coefficients.

But what if we introduce new number $i$ such that $i^2 = -1$. Then we can at least solve previous equation:

$$
x^2 + 1 = (x - i)(x + i) = 0 \implies x = \pm i
$$

But the main point is that we can decompose any polynomial into linear factors if we allow complex coefficients. This is called the **Fundamental Theorem of Algebra**. I don't want to prove it there because it's too complex (in our linear algebra exam this proof was separated into 3 tickets), and this doesn't matter for us.

The main result from there is that we constructed linear space: 

$$
\mathbb{C} = \{a + bi | a, b \in \mathbb{R}\} = \mathbb{R} + i\mathbb{R}
$$

So we have basis $\{1, i\}$ and any complex number can be represented as linear combination of these basis vectors.

## Dual numbers

But the goal of this project is to describe how to calculate accurate derivatives of functions.

So instead if $\{1, i\}$ we will use other basis: $\{1, \varepsilon\}$, where $\varepsilon \neq 0$ and $\varepsilon^2 = 0$. If you stuck on this point, maybe this example will help you

$$
1 := \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} \quad \varepsilon := \begin{bmatrix}
0 & 1 \\
0 & 0
\end{bmatrix} \quad i := \begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

You can check that $\varepsilon^2 = 0$ and $i^2 = -1$, and also 1 commutes with both $\varepsilon$ and $i$, but $\varepsilon$ and $i$ don't commute (but we don't need this because we will not use them together). 

So we can construct linear space:

$$
\mathbb{D} = \{a + b\varepsilon \,\, | \,\, a, b \in \mathbb{R}\} = \mathbb{R} + \varepsilon\mathbb{R}
$$

But why we have selected this strange number $\varepsilon$? Well, let's recall Taylor series of function $f$ in point $x$:

$$
f(x + h) = f(x) + f'(x)h + \frac{f''(x)}{2}h^2 + ... + \frac{f^{(n)}(x)}{n!}h^n + ...
$$

But what if we want to calculate $f(A + \varepsilon B)$? Then,

$$
f(A + \varepsilon B) = f(A) + \varepsilon B f'(A) + \varepsilon^2 B^2 \frac{f''(x)}{2} + ... = f(A) + \varepsilon B f'(A) + \underbrace{\varepsilon^2}_{=0} (...) = f(A) + \varepsilon B f'(A)
$$

So, the main result is that 

$$
f(A + \varepsilon B) = f(A) + \varepsilon B f'(A)
$$

If we set $B = 1$, then we can easily calculate $f'(A)$:

$$
f(A + \varepsilon) = f(A) + \varepsilon f'(A)
$$

And this will help us to calculate derivatives of any functions, even if they are very complicated!

On this point, you can start checking my implementation of dual numbers in `dual/DualNumber.py` and basic functions for them in `dual/functions.py`. Also, you can check examples of using this library in `examples/` folder. For example, in `examples/hard_function_derivative.py` you can see how to calculate derivative of very complicated function using dual numbers.

Basic operations given in `DualNumber.py` are:

1. Addition: $(A + \varepsilon B) + (C + \varepsilon D) = (A + C) + \varepsilon (B + D)$
2. Subtraction: $(A + \varepsilon B) - (C + \varepsilon D) = (A - C) + \varepsilon (B - D)$
3. Multiplication: $(A + \varepsilon B)(C + \varepsilon D) = AC + \varepsilon (AD + BC) + \varepsilon^2 BD = AC + \varepsilon (AD + BC)$
4. Division: $\frac{A + \varepsilon B}{C + \varepsilon D} = \frac{(A + \varepsilon B)(C - \varepsilon D)}{(C + \varepsilon D)(C - \varepsilon D)} = \frac{AC + \varepsilon (BC - AD)}{C^2} = \frac{A}{C} + \varepsilon \frac{BC - AD}{C^2} \quad (C \neq 0)$
5. Logarithm: $\ln(A + \varepsilon B) = \ln A + \varepsilon B \ln'(A) = \ln A + \varepsilon \frac{B}{A}$
6. Power: $(A + \varepsilon B)^{C + \varepsilon D} = e^{(C + \varepsilon D) \ln(A + \varepsilon B)} = e^{C \ln A + \varepsilon (D \ln A + C \frac{B}{A})} = A^C + \varepsilon A^C (D \ln A + C \frac{B}{A}) = A^C + \varepsilon A^{C-1} (AD \ln A + BC)$

The main point is that if we have complex function $f(g(A_1 + \varepsilon B_1))$, we can always easily find coefficients $A_2, B_2$ such that $f(g(A_1 + \varepsilon B_1)) = A_2 + \varepsilon B_2$ if we know only how to calculate $f'(x)$ and $g'(x)$. It will look like this:

$$
f(g(A_1 + \varepsilon B_1)) = f(g(A_1) + \varepsilon B_1 g'(A_1)) = f(g(A_1)) + \varepsilon B_1 g'(A_1) f'(g(A_1)) = A_2 + \varepsilon B_2 \implies \begin{cases}
A_2 = f(g(A_1)) \\
B_2 = B_1 g'(A_1) f'(g(A_1))
\end{cases}
$$

So, maybe it looks hard, but for python it's just product of real numbers. 

As you remember,

$$
f(A + \varepsilon B) = f(A) + \varepsilon B f'(A)
$$

So if we have function f and we want it work in $\mathbb{D}$, we just need to make something like this:

```python
from dual.DualNumber import DualNumber

def f(x: DualNumber) -> DualNumber:
    A = x.real
    B = x.dual
    return DualNumber(real_f(A), B * real_f_derivative(A))
```

And it will work for any function! For example, let's write code for sin(x):

```python
import math
from dual.DualNumber import DualNumber

def sin(x: DualNumber) -> DualNumber:
    A = x.real
    B = x.dual
    
    return DualNumber(math.sin(A), B * math.cos(A))  # because sin'(x) = cos(x)
```

So what if we want to calculate derivative of $\sin(x^2)$? We can just write:

```python
import math
from dual.DualNumber import DualNumber
from dual.functions import sin

def f(x: DualNumber) -> DualNumber:
    return sin(x * x)

x = DualNumber(2, 1) # or just result = sin(x * x), no need to define f, but I want to show how it works with more complex functions
result = f(x)
print(f"f({x.real}) = {result.real})")  # this will print f(2)
print(f"f'({x.real}) = {result.dual})")  # this will print f'(2)
```
Let see table of values for more complex functions in `dual/functions.py`:

| Function $f$ | Derivative $f'$             | $f(A + \varepsilon B)$ |
|--------------|-----------------------------| --- |
| $\sin x$     | $\cos x$                    | $\sin A + \varepsilon B \cos A$ |
| $\cos x$     | $-\sin x$                   | $\cos A - \varepsilon B \sin A$ |
| $\tan x$     | $\tan^2 x + 1$              | $\tan A + \varepsilon B \tan^2 A + \varepsilon B$ |
| $\ln x$      | $\frac{1}{x}$               | $\ln A + \varepsilon \frac{B}{A}$ |
| $e^x$        | $e^x$                       | $e^A + \varepsilon B e^A$ |
| $\sqrt{x}$   | $\frac{1}{2\sqrt{x}}$       | $\sqrt{A} + \varepsilon \frac{B}{2\sqrt{A}}$ |
| $\sinh x$    | $\cosh x$                   | $\sinh A + \varepsilon B \cosh A$ |
| $\cosh x$    | $\sinh x$                   | $\cosh A + \varepsilon B \sinh A$ |
| $\tanh x$    | $1 - \tanh^2 x$             | $\tanh A + \varepsilon B (1 - \tanh^2 A)$ |
| $\arcsin x$  | $\frac{1}{\sqrt{1 - x^2}}$  | $\arcsin A + \varepsilon \frac{B}{\sqrt{1 - A^2}}$ |
| $\arccos x$  | $-\frac{1}{\sqrt{1 - x^2}}$ | $\arccos A - \varepsilon \frac{B}{\sqrt{1 - A^2}}$ |
| $\arctan x$  | $\frac{1}{1 + x^2}$         | $\arctan A + \varepsilon \frac{B}{1 + A^2}$ |