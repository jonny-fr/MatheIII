
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definiere wahre Ebenen-Parameter: ax + by + cz + d = 0
true_a, true_b, true_c, true_d = 2.0, -1.5, 3.0, -4.0

# Generiere künstliche Datenpunkte
np.random.seed(42)
n_points = 100
x = np.random.uniform(-5, 5, n_points)
y = np.random.uniform(-5, 5, n_points)
z = -(true_a * x + true_b * y + true_d) / true_c
z_noisy = z + np.random.normal(0, 0.2, n_points)

# Finde Ebene mit Normalengleichung: (A^T A)x = A^T b
A = np.column_stack([x, y, np.ones(n_points)])
b = -z_noisy
params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

est_a, est_b, est_d, est_c = params[0], params[1], params[2], 1.0

# Ausgabe der Parameter
print(f"Geschätzte Ebene: {est_a:.4f}x + {est_b:.4f}y + {est_c:.4f}z + {est_d:.4f} = 0")
print(f"Residuen: {residuals[0]:.4f}")

# 3D-Visualisierung
fig = plt.figure(figsize=(14, 6))

# Plot 1: Geschätzte Ebene
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z_noisy, c='blue', marker='o', alpha=0.6, s=30)
x_range = np.linspace(x.min(), x.max(), 20)
y_range = np.linspace(y.min(), y.max(), 20)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z_estimated = -(est_a * X_grid + est_b * Y_grid + est_d) / est_c
ax1.plot_surface(X_grid, Y_grid, Z_estimated, alpha=0.3, color='red')
ax1.set_title('Geschätzte Ebene durch Normalengleichung')

# Plot 2: Vergleich
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x, y, z_noisy, c='blue', marker='o', alpha=0.6, s=30)
Z_true = -(true_a * X_grid + true_b * Y_grid + true_d) / true_c
ax2.plot_surface(X_grid, Y_grid, Z_true, alpha=0.2, color='green')
ax2.plot_surface(X_grid, Y_grid, Z_estimated, alpha=0.2, color='red')
ax2.set_title('Vergleich: Wahre vs. Geschätzte Ebene')

plt.tight_layout()
plt.show()




