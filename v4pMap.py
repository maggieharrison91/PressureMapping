import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
import matplotlib.image as mpimg

HAND_IMG_FILE = "/Users/jennakazim/Desktop/VAK/PressureMapping/handOutline.png"
# glove shape
# fingers: 4 vertical x 20 horizontal = 80 nodes each
# palm: 16 vertival x 32 horizontal = 512 nodes total


# =========================== CONFIG ===========================

CSV_FILE = "/Users/jennakazim/Desktop/VAK/PressureMapping/09282025_singleconfig8_pressure_capacitance_CH0_CH4.csv"
# LAYOUT_FILE = "/Users/jennakazim/Desktop/VAK/PressureMapping/glove.csv"
HAND_OUTLINE_FILE = "/Users/jennakazim/Desktop/VAK/PressureMapping/handOutlineCoordinates_reduced.csv"

GRID_RES = 300

# ====================== LOAD CSV ===============================

df = pd.read_csv(CSV_FILE)

timestamps = df.iloc[:, 0].values        # timestamp column

# We only have 8 physical channels: CH0..CH7
raw_data = df.iloc[:, 1:9].values        # shape: (num_frames, 8)
num_frames = raw_data.shape[0]

def compute_intersections(frame_8ch):
    """
    frame_8ch: length-8 array [CH0, CH1, CH2, CH3, CH4, CH5, CH6, CH7]
      - CH0..CH3 = row strings
      - CH4..CH7 = column strings
    Returns:
      length-16 array for the 4x4 intersection grid.
    """
    rows = frame_8ch[0:4]   # CH0..CH3
    cols = frame_8ch[4:8]   # CH4..CH7

    # Intersection model: outer product (can change to rows+cols if desired)
    inter_matrix = np.outer(rows, cols)   # shape (4, 4)
    return inter_matrix.ravel()           # length 16 (row-major)

# Build a full (num_frames x 16) array for the 4x4 mockup
A_data = np.apply_along_axis(compute_intersections, 1, raw_data)  # shape: (num_frames, 16)

# ===================== GLOVE GEOMETRY =========================

PALM_V = 16  # vertical stripes in palm
PALM_H = 32  # horizontal stripes in palm

FINGER_V = 4  # vertical stripes per finger
FINGER_H = 20 # horizontal stripes per finger
N_FINGERS = 4

FINGER_NAMES = ["index", "middle", "ring", "pinky"]


def build_hand_layout():
    """
    Generate a dense glove layout:
      - Palm: 16 x 32 intersections
      - Fingers: 4 fingers * (4 x 20) intersections each
    Returns:
      x_coords, y_coords: arrays of all node positions (normalized)
      regions: list of region labels ("palm", "index", ...)
      mockup_indices: indices of palm nodes used for a 4x4 mockup patch
    """

    x_list = []
    y_list = []
    regions = []

    # --- Palm grid ---
    # Palm spans most of the width: [-0.7, 0.7]
    palm_x = np.linspace(-0.5, 0.5, PALM_V)
    # Palm height: from wrist-ish to base of fingers: [-0.7, 0]
    palm_y = np.linspace(-0.7, 0, PALM_H)

    # Keep track of palm node indices: 0 .. (PALM_V*PALM_H - 1)
    for j in range(PALM_H):        # y index
        for i in range(PALM_V):    # x index
            x_list.append(palm_x[i])
            y_list.append(palm_y[j])
            regions.append("palm")

    # --- Fingers ---
    # Fingers start just above the palm and go towards the top.
    finger_base_y = 0.1
    finger_tip_y = 0.9
    finger_y = np.linspace(finger_base_y, finger_tip_y, FINGER_H)

    # 4 finger centers across the top of the palm
    finger_centers_x = np.linspace(-0.45, 0.45, N_FINGERS)
    finger_half_width = 0.05  # approximate lateral extent per finger

    for f_idx, fname in enumerate(FINGER_NAMES):
        cx = finger_centers_x[f_idx]
        # 4 vertical stripes within the finger width
        finger_x = np.linspace(cx - finger_half_width, cx + finger_half_width, FINGER_V)

        for j in range(FINGER_H):
            for i in range(FINGER_V):
                x_list.append(finger_x[i])
                y_list.append(finger_y[j])
                regions.append(fname)

    x_coords = np.array(x_list, dtype=float)
    y_coords = np.array(y_list, dtype=float)

    # --- Choose a 4x4 subgrid in the PALM as the "mockup" patch ---
    # Place the 4x4 patch in the middle horizontally and near the bottom
    # of the palm vertically (close to the wrist).
    start_col = (PALM_V - 4) // 2
    mockup_cols = np.arange(start_col, start_col + 4, dtype=int)

    # choose rows slightly above the very bottom rows to avoid edge artifacts
    start_row = 2
    mockup_rows = np.arange(start_row, start_row + 4, dtype=int)

    mockup_indices = []
    for r in mockup_rows:
        for c in mockup_cols:
            idx = r * PALM_V + c  # palm nodes are stored row-major at the beginning
            mockup_indices.append(idx)
    mockup_indices = np.array(mockup_indices, dtype=int)


    return x_coords, y_coords, regions, mockup_indices


# ======== Build layout ========
x_coords, y_coords, regions, all_mockup_indices = build_hand_layout()
sensor_coords = list(zip(x_coords, y_coords))
num_nodes = len(sensor_coords)

# Map data channels (now 16 virtual intersections) --> mockup nodes
n_channels = A_data.shape[1]

if len(all_mockup_indices) < n_channels:
    raise ValueError(
        f"Mockup region has only {len(all_mockup_indices)} nodes "
        f"but data has {n_channels} channels."
    )

# Use as many mockup nodes as there are channels (here, 16)
mockup_indices = all_mockup_indices[:n_channels]

print("Total nodes (palm + fingers):", num_nodes)
print("Data channels (virtual intersections):", n_channels)
print("Mockup nodes available:", len(all_mockup_indices))
print("Mockup nodes used:", len(mockup_indices))


print("\n=== Mapping of CSV Channels to Mockup Nodes ===")
for virtual_idx, node_idx in enumerate(mockup_indices):
    # row & column index in the 4×4 grid
    r, c = divmod(virtual_idx, 4)

    # physical channels involved in this intersection
    row_ch = r          # CH0..CH3
    col_ch = 4 + c      # CH4..CH7

    print(
        f"Mockup node {virtual_idx:2d} "
        f"(grid R{r},C{c}) at glove index {node_idx:4d}  <--  "
        f"Row channel CH{row_ch} × Column channel CH{col_ch}"
    )
print("================================================\n")


# ================= PRECOMPUTE PER-NODE MIN/MAX =================

A_min = A_data.min(axis=0)
A_max = A_data.max(axis=0)
A_range = np.where((A_max - A_min) == 0, 1e-9, A_max - A_min)

def per_node_intensity(A_flat):
    """Normalize each of the 16 mockup channels based on its own min/max."""
    return np.clip((A_flat - A_min) / A_range, 0.0, 1.0)

# ==================== HEATMAP FUNCTIONS =========================

span_x = x_coords.max() - x_coords.min()
span_y = y_coords.max() - y_coords.min()
# Controls how wide each sensor's gaussian bump is (fraction of span)
SPREAD_FACTOR = 0.025  # smaller blobs
SIGMA_X = SPREAD_FACTOR * span_x
SIGMA_Y = SPREAD_FACTOR * span_y

# Additional blur applied to the whole field after summing bumps.
HEAT_BLUR = 1  # reduce to make heatmap sharper

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

# ===================== HEATMAP GRID ===========================

x_grid = np.linspace(-1, 1, GRID_RES)
y_grid = np.linspace(-1, 1, GRID_RES)
X, Y = np.meshgrid(x_grid, y_grid)

# ===================== HAND OUTLINE & MASK ====================

hand_outline = pd.read_csv(HAND_OUTLINE_FILE, comment="#")

poly_x = hand_outline["x_norm"].values
poly_y = hand_outline["y_norm"].values
polygon = np.vstack([poly_x, poly_y]).T

# Path for outline and inside/outside tests
hand_path = Path(polygon)

# Precompute mask on the grid: True inside the hand, False outside
points = np.vstack([X.ravel(), Y.ravel()]).T
inside_flat = hand_path.contains_points(points)
hand_mask = inside_flat.reshape(X.shape)

def field_from_matrix(A_flat):
    """
    A_flat: shape (n_channels,), current frame's readings (16 virtual intersections).
    We:
      - normalize these n_channels,
      - place them on mockup_indices,
      - leave all other nodes at intensity 0,
      - blur,
      - apply hand mask (optional).
    """
    F = np.zeros_like(X)

    mockup_intensities = per_node_intensity(A_flat)  # shape (n_channels,)
    intensities_all = np.zeros(num_nodes)
    intensities_all[mockup_indices] = mockup_intensities

    for (cx, cy), I in zip(sensor_coords, intensities_all):
        if I <= 0:
            continue
        g = np.exp(-(((X - cx)**2) / (2 * SIGMA_X**2)
                     + ((Y - cy)**2) / (2 * SIGMA_Y**2)))
        F += I * g

    F = gaussian_blur(np.clip(F, 0, 1), sigma_px=HEAT_BLUR)
    # Optionally mask outside the hand:
    # F = F * hand_mask

    return F

# ===================== PLOTTING ================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

# Load the hand outline image
hand_img = mpimg.imread(HAND_IMG_FILE)
hand_img = np.flipud(hand_img)
ax.imshow(
    hand_img,
    extent=[-1, 1, -1, 1],   # match your heatmap coordinate system
    origin="lower",          # same as your heatmap
    zorder=0                 # behind everything else
)

# heatmap image (will be updated each frame)
img = ax.imshow(
    np.ones((GRID_RES, GRID_RES, 3)),
    extent=[-1, 1, -1, 1],
    origin='lower',
    interpolation='nearest',
    animated=True,
    alpha=0.7,   # lower = more of the hand outline visible
    zorder=1     # above the PNG, below text/markers
)

# Highlight channel/mockup nodes so they stand out from the rest.
node_colors = ['k'] * num_nodes
for idx in mockup_indices:
    node_colors[idx] = '#8A2BE2'

node_sizes = np.full(num_nodes, 10)
node_sizes[mockup_indices] = 10

ax.scatter(x_coords, y_coords, s=node_sizes, c=node_colors, alpha=0.85, zorder=2)

# labels for the 16 mockup sensors (virtual intersections)
labels = []
for ch_idx, node_idx in enumerate(mockup_indices):
    cx, cy = sensor_coords[node_idx]
    # You can switch to R{r}C{c} if you prefer intersection-style labels:
    # UNCOMMENT TO SEE LABELS
    # r, c = divmod(ch_idx, 4)
    # name = f"R{r}C{c}"
    # name = f"CH{ch_idx}"
    name = ""

    ax.text(cx, cy + 0.05, name,
            ha='center', va='center', fontsize=8, weight='bold')
    val = ax.text(cx, cy - 0.05, "--",
                  ha='center', va='center', fontsize=7)
    labels.append(val)

timestamp_text = ax.text(
    0, -1.05,
    "t = 0.000 s",
    ha='center', va='top',
    fontsize=8
)

# ===================== ANIMATION UPDATE ========================

def update(i):
    A_flat = A_data[i]   # shape (16,)

    F = field_from_matrix(A_flat)
    img.set_data(intensity_to_rgb(F))

    for k, v in enumerate(A_flat):
        # UNCOMMENT TO SEE LABELS
        # labels[k].set_text(f"{v:.4f}")
        labels[k].set_text("")

    timestamp_text.set_text(f"t = {timestamps[i]:.3f}s")

    return [img] + labels + [timestamp_text]

ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)

plt.tight_layout()
plt.show()
