import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

os.makedirs("figs", exist_ok=True)

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)

x = X["total_phenols"].values
y = X["flavanoids"].values

r = np.corrcoef(x, y)[0, 1]
print("Corrélation total_phenols / flavanoids :", round(r, 3))

a, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = a * x_line + b

plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.7)
plt.plot(x_line, y_line)
plt.title(f"total_phenols vs flavanoids (r = {r:.3f})")
plt.xlabel("total_phenols")
plt.ylabel("flavanoids")
plt.tight_layout()
plt.savefig("figs/partA_scatter_reg.png", dpi=150)
plt.close()

x_c = x - x.mean()
y_c = y - y.mean()
Fx = np.fft.rfft(x_c)
Fy = np.fft.rfft(y_c)
freqs = np.fft.rfftfreq(len(x_c), d=1)

plt.figure(figsize=(8, 5))
plt.plot(freqs, np.abs(Fx), label="total_phenols")
plt.plot(freqs, np.abs(Fy), label="flavanoids")
plt.title("Spectres FFT centrés")
plt.xlabel("Fréquence (index normalisé)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("figs/partA_fft.png", dpi=150)
plt.close()
