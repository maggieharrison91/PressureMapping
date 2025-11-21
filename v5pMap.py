import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
import matplotlib.image as mpimg


SIMULATE_SERIAL = True   # <<< set to False when using real hardware

HAND_IMG_FILE = "handOutline.png"

# =========================== CONFIG ===========================

CSV_FILE = "09282025_singleconfig8_pressure_capacitance_CH0_CH4.csv"
GRID_RES = 300

# =========================== NEW: SERIAL SETUP ===========================
import serial
import time
from collections import deque

SERIAL_PORT = "/dev/tty.usbserial-2110"
BAUD_RATE = 115200

MAX_FRAMES = 300

if not SIMULATE_SERIAL:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
else:
    ser = None  # no real port

timestamps = deque(maxlen=MAX_FRAMES)
raw_frames = deque(maxlen=MAX_FRAMES)

start_time = time.time()

import math
import random

def read_serial_frame():
    """
    If SIMULATE_SERIAL=True, returns fake but realistic CH0..CH7 readings.
    Otherwise reads from actual serial.
    """
    if SIMULATE_SERIAL:
        t = time.time() - start_time
        fake_row = [200 + 50*math.sin(t + i) for i in range(4)]
        fake_col = [300 + 30*math.cos(t + i*0.5) for i in range(4)]
        return fake_row + fake_col

    # REAL SERIAL MODE
    try:
        line = ser.readline().decode().strip()
        if not line:
            return None

        parts = line.split(',')
        if len(parts) != 8:
            return None

        return [float(p) for p in parts]

    except Exception:
        return None


def serial_data_generator():
    """Continuously read serial and yield newest frame index for animation."""
    frame_idx = 0
    while True:
        frame = read_serial_frame()
        if frame is not None:
            raw_frames.append(frame)
            timestamps.append(time.time() - start_time)
            yield frame_idx
            frame_idx += 1


# ====================== LOAD CSV (DISABLED FOR LIVE) ===============================

# df = pd.read_csv(CSV_FILE)
# timestamps = df.iloc[:, 0].values
# raw_data = df.iloc[:, 1:9].values
# num_frames = raw_data.shape[0]

raw_data = raw_frames
num_frames = None


def compute_intersections(frame_8ch):
    rows = frame_8ch[0:4]
    cols = frame_8ch[4:8]
    inter_matrix = np.outer(rows, cols)
    return inter_matrix.ravel()

# A_data = np.apply_along_axis(compute_intersections, 1, raw_data)

# ===================== GLOVE GEOMETRY =========================

PALM_V = 16
PALM_H = 32

FINGER_V = 4
FINGER_H = 20
N_FINGERS = 4

FINGER_NAMES = ["index", "middle", "ring", "pinky"]


def build_hand_layout():
    x_list = []
    y_list = []
    regions = []

    palm_x = np.linspace(-0.5, 0.5, PALM_V)
    palm_y = np.linspace(-0.7, 0, PALM_H)

    for j in range(PALM_H):
        for i in range(PALM_V):
            x_list.append(palm_x[i])
            y_list.append(palm_y[j])
            regions.append("palm")

    finger_base_y = 0.1
    finger_tip_y = 0.9
    finger_y = np.linspace(finger_base_y, finger_tip_y, FINGER_H)

    finger_centers_x = np.linspace(-0.45, 0.45, N_FINGERS)
    finger_half_width = 0.05

    for f_idx, fname in enumerate(FINGER_NAMES):
        cx = finger_centers_x[f_idx]
        finger_x = np.linspace(cx - finger_half_width, cx + finger_half_width, FINGER_V)

        for j in range(FINGER_H):
            for i in range(FINGER_V):
                x_list.append(finger_x[i])
                y_list.append(finger_y[j])
                regions.append(fname)

    x_coords = np.array(x_list, dtype=float)
    y_coords = np.array(y_list, dtype=float)

    # --- Choose a 4x4 subgrid located at the tip of one finger ---
    # We'll place the 4x4 labeled mockup on the tip of the first finger (index)
    palm_count = PALM_V * PALM_H

    # pick which finger to use for the mockup (0=index, 1=middle, ...)
    target_finger = 0

    # within the finger grid, rows increase from base->tip, so choose the
    # topmost 4 rows to be the 4x4 patch (i.e., near the fingertip)
    finger_row_start = max(0, FINGER_H - 4)
    finger_row_indices = np.arange(finger_row_start, finger_row_start + 4, dtype=int)

    mockup_indices = []
    for r in finger_row_indices:
        for c in range(FINGER_V):
            # index within that finger: r * FINGER_V + c
            idx_within_finger = r * FINGER_V + c
            # global node index = palm_count + finger_offset + idx_within_finger
            finger_offset = target_finger * (FINGER_H * FINGER_V)
            idx = palm_count + finger_offset + idx_within_finger
            mockup_indices.append(idx)

    return x_coords, y_coords, regions, np.array(mockup_indices, dtype=int)


x_coords, y_coords, regions, all_mockup_indices = build_hand_layout()
sensor_coords = list(zip(x_coords, y_coords))
num_nodes = len(sensor_coords)

# n_channels = A_data.shape[1]

# use known 16 intersections
n_channels = 16

if len(all_mockup_indices) < n_channels:
    raise ValueError(
        f"Mockup region has only {len(all_mockup_indices)} nodes "
        f"but data has {n_channels} channels."
    )

mockup_indices = all_mockup_indices[:n_channels]


print("Total nodes:", num_nodes)
print("Data channels:", n_channels)
print("Mockup nodes available:", len(all_mockup_indices))
print("Mockup nodes used:", len(mockup_indices))


print("\n=== Mapping of Channels to Mockup Nodes ===")
for virtual_idx, node_idx in enumerate(mockup_indices):
    r, c = divmod(virtual_idx, 4)
    row_ch = r
    col_ch = 4 + c
    print(
        f"Mockup node {virtual_idx:2d} (grid R{r},C{c}) at glove index {node_idx:4d}  <--  "
        f"Row CH{row_ch} × Column CH{col_ch}"
    )
print("================================================\n")


# ================= HARD-CODED MIN/MAX FOR NORMALIZATION =================
A_min = np.zeros(16)
A_max = np.ones(16) * 250000  # adjust as needed
A_range = A_max - A_min

def per_node_intensity(A_flat):
    return np.clip((A_flat - A_min) / A_range, 0.0, 1.0)


# ==================== HEATMAP FUNCTIONS =========================

span_x = x_coords.max() - x_coords.min()
span_y = y_coords.max() - y_coords.min()

SPREAD_FACTOR = 0.025
SIGMA_X = SPREAD_FACTOR * span_x
SIGMA_Y = SPREAD_FACTOR * span_y
HEAT_BLUR = 1

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

# ===================== FIELD CALC ====================

def field_from_matrix(A_flat):
    F = np.zeros_like(X)

    mockup_intensities = per_node_intensity(A_flat)
    intensities_all = np.zeros(num_nodes)
    intensities_all[mockup_indices] = mockup_intensities

    for (cx, cy), I in zip(sensor_coords, intensities_all):
        if I <= 0:
            continue
        g = np.exp(-(((X - cx)**2) / (2 * SIGMA_X**2)
                     + ((Y - cy)**2) / (2 * SIGMA_Y**2)))
        F += I * g

    F = gaussian_blur(np.clip(F, 0, 1), sigma_px=HEAT_BLUR)

    return F

# ===================== PLOTTING ================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

hand_img = mpimg.imread(HAND_IMG_FILE)
hand_img = np.flipud(hand_img)
ax.imshow(
    hand_img,
    extent=[-1, 1, -1, 1],
    origin="lower",
    zorder=0
)

img = ax.imshow(
    np.ones((GRID_RES, GRID_RES, 3)),
    extent=[-1, 1, -1, 1],
    origin='lower',
    interpolation='nearest',
    animated=True,
    alpha=0.7,
    zorder=1
)

node_colors = ['k'] * num_nodes
for idx in mockup_indices:
    node_colors[idx] = '#8A2BE2'

node_sizes = np.full(num_nodes, 10)
node_sizes[mockup_indices] = 10

ax.scatter(x_coords, y_coords, s=node_sizes, c=node_colors, alpha=0.85, zorder=2)

labels = []
# Labels disabled to reduce clutter — keep empty list for update return

timestamp_text = ax.text(
    0, -1.05,
    "t = 0.000 s",
    ha='center', va='top',
    fontsize=8
)

# ===================== ANIMATION UPDATE ========================

def update(i):
    # A_flat = A_data[i]   # (CSV MODE) — NOT USED NOW

    if len(raw_frames) == 0:
        return [img] + labels + [timestamp_text]

    frame_8ch = raw_frames[-1]
    A_flat = compute_intersections(frame_8ch)

    F = field_from_matrix(A_flat)
    img.set_data(intensity_to_rgb(F))

    # labels are disabled; skip per-node text updates

    timestamp_text.set_text(f"t = {timestamps[-1]:.3f}s")

    return [img] + labels + [timestamp_text]


ani = FuncAnimation(
    fig,
    update,
    frames=serial_data_generator(),
    interval=1,
    blit=False
)

plt.tight_layout()
plt.show()