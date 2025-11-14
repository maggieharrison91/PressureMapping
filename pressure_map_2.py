import threading, time
from collections import defaultdict
import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# desired format of serial data:
# A0: 512
# A1: 600
# ...   
# A11: 300

# this creates a rectangular sensor array heat map with configurable rows and cols
PORT = '/dev/cu.usbmodem2101'
BAUD = 115200
ROWS = 3  # number of rows
COLS = 4  # number of columns
Y_MIN, Y_MAX = 0, 1023 # intended for resistive pressure sensors (ADC range 0-1023)
PLOT_INTERVAL_MS = 16
SER_TIMEOUT = 0.0
GRID_RES = 300
SIGMA_X = 0.8 / COLS
SIGMA_Y = 0.8 / ROWS

# channel labels (was using Arduino channels)
CHANNELS = [f"A{i}" for i in range(ROWS * COLS)]
latest = defaultdict(lambda: None)
lock = threading.Lock()
running = True

# reading serial in background thread
def parse_line(line: bytes):
    s = line.decode('utf-8', errors='ignore').strip()
    if ':' not in s:
        return None
    k, v = s.split(':', 1)
    try:
        return k.strip(), int(v.strip())
    except:
        return None

def serial_reader():
    global running
    try:
        ser = serial.Serial(PORT, BAUD, timeout=SER_TIMEOUT)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"[Serial] Failed to open {PORT}: {e}")
        return
    while running:
        drained = False
        for _ in range(64):
            line = ser.readline()
            if not line:
                break
            drained = True
            pv = parse_line(line)
            if pv is None:
                continue
            key, val = pv
            if key in CHANNELS:
                val = max(Y_MIN, min(Y_MAX, val))
                with lock:
                    latest[key] = val
        if not drained:
            time.sleep(0.001)
    try:
        ser.close()
    except:
        pass

t = threading.Thread(target=serial_reader, daemon=True)
t.start()

# helper functions
def gaussian1d(sigma_px, radius=None):
    if sigma_px <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(3 * sigma_px))
    xs = np.arange(-radius, radius + 1)
    k = np.exp(-(xs**2) / (2 * sigma_px**2))
    k /= k.sum()
    return k

def gaussian_blur(arr, sigma_px=3):
    if sigma_px <= 0:
        return arr
    k = gaussian1d(sigma_px)
    arr = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=1, arr=arr)
    arr = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=0, arr=arr)
    return arr

def adc_to_intensity(v):
    if v is None:
        return 0.0   # show white if no reading
    if Y_MAX == Y_MIN:
        return 0.0
    norm = (v - Y_MIN) / float(Y_MAX - Y_MIN)
    return 1.0 - np.clip(norm, 0.0, 1.0)

def intensity_to_rgb(I):
    I = np.clip(I, 0.0, 1.0)
    R = np.ones_like(I)
    G = 1.0 - I
    B = 1.0 - I
    return np.stack([R, G, B], axis=-1)

# normalize -1 to 1
x = np.linspace(-1, 1, GRID_RES)
y = np.linspace(-1, 1, GRID_RES)
X, Y = np.meshgrid(x, y)

# epicenter spacings
x_positions = np.linspace(-0.8, 0.8, COLS)
y_positions = np.linspace(0.8, -0.8, ROWS)
sensor_coords = [(x_positions[c], y_positions[r])
                 for r in range(ROWS) for c in range(COLS)]

def values_snapshot():
    with lock:
        snap = {ch: latest[ch] for ch in CHANNELS}
    mid = (Y_MIN + Y_MAX) // 2
    return np.array([snap[ch] if snap[ch] is not None else mid for ch in CHANNELS], dtype=float)

# this adds gaussian bump (SIGMA_X/Y control the spread)
def field_from_sensors(sensor_vals):
    intensities = np.array([adc_to_intensity(v) for v in sensor_vals])
    F = np.zeros_like(X)
    for (cx, cy), intensity in zip(sensor_coords, intensities):
        g = np.exp(-(((X - cx)**2) / (2 * SIGMA_X**2) +
                     ((Y - cy)**2) / (2 * SIGMA_Y**2)))
        F += intensity * g
    # normalize
    F = np.clip(F / np.max(F), 0.0, 1.0)
    F = gaussian_blur(F, sigma_px=3)
    return F

# plot init
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axis('off')

# sensor grid visual outline
ax.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], 'k', lw=2)

img = ax.imshow(np.ones((GRID_RES, GRID_RES, 3)),
                extent=[-1, 1, -1, 1], origin='lower',
                interpolation='nearest', animated=True)

# add labels for reading purposes
labels = []
for i, (cx, cy) in enumerate(sensor_coords):
    lbl = ax.text(cx, cy, f"A{i}", color='black', ha='center', va='center',
                  fontsize=12, weight='bold')
    labels.append(lbl)


def update(_):
    vals = values_snapshot() # keep this to read over serial

    # this is for simulating fake data to see how the heatmap looks

    # demo for simulated data (no serial connection)
    # t_now = time.time()
    # vals = np.array([
    #     int(Y_MAX/2 + (Y_MAX/2)*np.sin(t_now + i))
    #     for i in range(ROWS * COLS)
    # ])
    # # print simulated serial lines
    # for i, v in enumerate(vals):
    #     print(f"A{i}: {v}")


    F = field_from_sensors(vals)
    RGB = intensity_to_rgb(F)
    img.set_data(RGB)
    return [img]

def on_close(_evt):
    global running
    running = False

fig.canvas.mpl_connect('close_event', on_close)
ani = FuncAnimation(fig, update, interval=PLOT_INTERVAL_MS, blit=False)
plt.tight_layout()
plt.show()

running = False
t.join(timeout=1.0)