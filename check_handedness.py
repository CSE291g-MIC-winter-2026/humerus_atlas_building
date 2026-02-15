import numpy as np
import os
import glob

def check_handedness(sdf_path):
    sdf = np.load(sdf_path)
    # SDF is negative inside. Find center of mass of interior
    mask = sdf < 0
    if not np.any(mask):
        return 0, "empty"
    
    indices = np.argwhere(mask)
    # Centered indices (relative to middle of volume)
    center = np.array(sdf.shape) / 2
    centered_indices = indices - center
    
    # Calculate Mean X offset
    mean_x = np.mean(centered_indices[:, 0])
    
    # In our coordinate system, if the shaft is centered, 
    # the head should be an offset.
    return mean_x, "left" if mean_x < 0 else "right"

processed_dir = 'd:/Winter Assignments/291G/project/data/processed'
npy_files = glob.glob(os.path.join(processed_dir, '*.npy'))

results = []
for f in npy_files:
    mx, side = check_handedness(f)
    results.append((os.path.basename(f), mx, side))

# Sort by subject id
results.sort()

for res in results:
    print(f"{res[0]:20} | Mean X: {res[1]:8.2f} | Detected: {res[2]}")

lefts = sum(1 for r in results if r[2] == 'left')
rights = sum(1 for r in results if r[2] == 'right')
print(f"\nSummary: {lefts} Lefts, {rights} Rights")
