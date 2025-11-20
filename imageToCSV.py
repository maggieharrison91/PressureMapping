import csv
import numpy as np
from PIL import Image

# --- PARAMETERS YOU CAN TUNE ---
IMAGE_PATH = "/Users/jennakazim/Desktop/VAK/PressureMapping/handOutline.png"
OUTPUT_PATH = "handOutlineCoordinates_reduced.csv"

# Threshold for deciding what is "hand" vs background (on grayscale 0-255)
# Lower = stricter (only very dark pixels), higher = looser (includes lighter lines)
HAND_THRESHOLD = 240  

# Maximum number of points to keep (set None to keep all hand pixels)
MAX_POINTS = 1000
# -------------------------------

# Load image
image = Image.open(IMAGE_PATH).convert("L")  # convert to grayscale
width, height = image.size

# Convert to numpy array
img_array = np.array(image)

# Create a mask of "hand" pixels: pixels darker than HAND_THRESHOLD
hand_mask = img_array < HAND_THRESHOLD

# Get (y, x) indices where hand_mask is True
ys, xs = np.where(hand_mask)

# Stack into (N, 2) array [x, y]
hand_coords = np.vstack((xs, ys)).T

if hand_coords.size == 0:
    raise ValueError("No hand pixels found with the current threshold. Try lowering HAND_THRESHOLD.")

# Optionally downsample to at most MAX_POINTS points
if MAX_POINTS is not None and hand_coords.shape[0] > MAX_POINTS:
    # Randomly sample without replacement
    indices = np.random.choice(hand_coords.shape[0], size=MAX_POINTS, replace=False)
    hand_coords = hand_coords[indices]

# Normalize coordinates to [0, 1]
normalized_x = hand_coords[:, 0] / (width - 1.0)
normalized_y = hand_coords[:, 1] / (height - 1.0)
normalized_coords = np.vstack((normalized_x, normalized_y)).T

# Save to CSV
np.savetxt(
    OUTPUT_PATH,
    normalized_coords,
    delimiter=",",
    fmt="%1.4f",
    header="x_norm,y_norm",
    comments=""
)
