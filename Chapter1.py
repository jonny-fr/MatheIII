# import pandas, sympy, numpy, matplotlib, scipy

import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def f(x, y):
    return x**2 + y**2

# Create grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y, indexing='xy')

# Evaluate function and compute gradients (order matters: axis0<-y, axis1<-x)
Z = f(X, Y)
dZdy, dZdx = np.gradient(Z, y, x, edge_order=2)

# Arrow step
step = 10

# Plot
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, levels=20)
fig.colorbar(CS, ax=ax, label='f(x, y)')

# Plot gradients as arrows (steepest ascent)
ax.quiver(
    X[::step, ::step], Y[::step, ::step],
    dZdx[::step, ::step], dZdy[::step, ::step],
    color='red', angles='xy', scale_units='xy', scale=3.0
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Contour plot with gradients')
ax.set_aspect('equal')

plt.show()