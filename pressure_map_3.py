import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# glove shape
# fingers: 4 vertical x 20 horizontal
# palm: 16 vertival x 32 horizontal



# =========================== CONFIG ===========================

CSV_FILE = "/Users/jennakazim/Desktop/VAK/PressureMapping/09282025_singleconfig8_pressure_capacitance_CH0_CH4.csv"    # <-- update with your file

ROWS = 4
COLS = 4
GRID_RES = 300

SIGMA_X = 0.8 / COLS
SIGMA_Y = 0.8 / ROWS

# ====================== LOAD CSV ===============================

df = pd.read_csv(CSV_FILE)

# START_TIME = 650.0   # seconds

# df = df[df["timestamp"] >= START_TIME].reset_index(drop=True)


timestamps = df.iloc[:, 0].values        # timestamp column
A_data = df.iloc[:, 1:17].values         # A00..A33 columns (16 total)

num_frames = len(df)

# ================= PRECOMPUTE PER-NODE MIN/MAX =================

A_min = A_data.min(axis=0)
A_max = A_data.max(axis=0)
A_range = np.where((A_max - A_min) == 0, 1e-9, A_max - A_min)

def per_node_intensity(A_flat):
    """Normalize each node independently based on its own min/max."""
    return np.clip((A_flat - A_min) / A_range, 0.0, 1.0)

# ==================== HEATMAP FUNCTIONS =========================

def gaussian1d(sigma_px, radius=None):
    if sigma_px <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(3 * sigma_px))
    xs = np.arange(-radius, radius + 1)
    k = np.exp(-(xs**2) / (2 * sigma_px**2))
    return k / k.sum()

def gaussian_blur(arr, sigma_px=3):
    if sigma_px <= 0:
        return arr
    k = gaussian1d(sigma_px)
    arr = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'),
                              axis=1, arr=arr)
    arr = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'),
                              axis=0, arr=arr)
    return arr

def intensity_to_rgb(I):
    R = np.ones_like(I)
    G = 1.0 - 0.7 * I
    B = 1.0 - 0.7 * I
    return np.stack([R, G, B], axis=-1)

# ===================== HEATMAP SETUP ===========================

x = np.linspace(-1, 1, GRID_RES)
y = np.linspace(-1, 1, GRID_RES)
X, Y = np.meshgrid(x, y)

x_positions = np.linspace(-0.8, 0.8, COLS)
y_positions = np.linspace(0.8, -0.8, ROWS)
sensor_coords = [(x_positions[c], y_positions[r])
                 for r in range(ROWS) for c in range(COLS)]

def field_from_matrix(A_flat):
    F = np.zeros_like(X)
    intensities = per_node_intensity(A_flat)

    for (cx, cy), I in zip(sensor_coords, intensities):
        g = np.exp(-(((X - cx)**2)/(2 * SIGMA_X**2)
                      + ((Y - cy)**2)/(2 * SIGMA_Y**2)))
        F += I * g

    return gaussian_blur(np.clip(F, 0, 1), sigma_px=3)

# ===================== PLOTTING ================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

ax.plot([-1,1,1,-1,-1], [1,1,-1,-1,1], 'k', lw=2)

img = ax.imshow(np.ones((GRID_RES, GRID_RES, 3)),
                extent=[-1,1,-1,1],
                origin='lower',
                interpolation='nearest',
                animated=True)

# labels
labels = []
for i, (cx, cy) in enumerate(sensor_coords):
    r, c = divmod(i, COLS)
    col_label = 4 + c  # label columns as 4..7
    ax.text(cx, cy + 0.08, f"R{r} C{col_label}",
            ha='center', va='center', fontsize=10, weight='bold')
    val = ax.text(cx, cy - 0.08, "--",
                  ha='center', va='center', fontsize=9)
    labels.append(val)

timestamp_text = ax.text(
    0, -1.05,
    "t = 0.000 s",
    ha='center', va='top',
    fontsize=12
)

# ===================== ANIMATION UPDATE ========================

def update(i):
    A_flat = A_data[i]

    F = field_from_matrix(A_flat)
    img.set_data(intensity_to_rgb(F))

    for k, v in enumerate(A_flat):
        labels[k].set_text(f"{v:.4f}")

    timestamp_text.set_text(f"t = {timestamps[i]:.3f}s")

    return [img] + labels + [timestamp_text]

ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)

plt.tight_layout()
plt.show()