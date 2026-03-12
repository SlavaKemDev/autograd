import matplotlib.pyplot as plt

from dual.DualNumber import DualNumber
from dual.functions import sin, log, cos, sqrt, dabs
import numpy as np


def f(x):  # function from calculus-1 homework
    return (2 ** sin(x) * 3 ** log(x + 3)) / (dabs(cos(2 * x)) ** (log(x ** 2)) * sqrt(x))


X = np.linspace(0.1, 5, 50).tolist()
y = [f(x).real for x in X]

hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
yd_nums = []
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    yd_nums.append([(f(x + h).real - f(x).real) / h for x in X])

yd_true = [f(DualNumber(x, 1)).dual for x in X]

fig, ax = plt.subplots(ncols=2, figsize=(13, 6))

ax[0].plot(X, y, label="f(x)", linewidth=2)
for h, yd_num in zip(hs, yd_nums):
    ax[0].plot(X, yd_num, label=f"numerical derivative with h = {h}", linestyle="--")
ax[0].plot(X, yd_true, label="dual number derivative", linestyle=":", linewidth=2)

for h, yd_num in zip(hs, yd_nums):
    ax[1].plot(X, np.abs(np.array(yd_num) - np.array(yd_true)), label=f"error with h = {h}", linestyle="--")

ax[0].set_yscale('symlog', linthresh=10)
ax[0].grid(True, which="both", ls="--", alpha=0.5)
ax[0].legend()

ax[1].set_yscale('symlog', linthresh=10)
ax[1].grid(True, which="both", ls="--", alpha=0.5)
ax[1].legend()

plt.show()
