import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from itertools import permutations

# Reference keypad
reference_keys = np.array([
    [0, 0],   # 1 (top-left)
    [1, 0],   # 2 (top-center)
    [2, 0],   # 3 (top-right)
    [0, 1],   # 4 (mid-left)
    [1, 1],   # 5 (center)
    [2, 1],   # 6 (mid-right)
    [0, 2],   # 7 (bot-left)
    [1, 2],   # 8 (bot-center)
    [2, 2],   # 9 (bot-right)
    [1, 3],   # 0 (bottom-center)
])
key_labels = ['1','2','3','4','5','6','7','8','9','0']

def normalize(points):
    points = np.array(points)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    return (points - min_xy) / (max_xy - min_xy + 1e-8)

def process_csv(csv_path, n_clusters=4, n_guesses=10):
    df = pd.read_csv(csv_path)
    led_cols = [col for col in df.columns if col.startswith('led') and ('_x' in col or '_y' in col)]
    n_leds = len(led_cols) // 2

    ring_centers = []
    for idx, row in df.iterrows():
        pts = []
        for led_idx in range(n_leds):
            x = row.get(f'led{led_idx+1}_x')
            y = row.get(f'led{led_idx+1}_y')
            if not (pd.isna(x) or pd.isna(y)):
                pts.append((x, y))
        if pts:
            ring_centers.append(np.mean(pts, axis=0))
        else:
            ring_centers.append(None)

    # Remove None values
    ring_centers = np.array([c for c in ring_centers if c is not None])
    if len(ring_centers) < n_clusters:
        print(f"{os.path.basename(csv_path)}: Not enough valid points for clustering.\n")
        return

    # Smooth the trajectory
    if len(ring_centers) > 9:
        x_smooth = savgol_filter(ring_centers[:,0], 9, 2)
        y_smooth = savgol_filter(ring_centers[:,1], 9, 2)
    else:
        x_smooth = ring_centers[:,0]
        y_smooth = ring_centers[:,1]
    trajectory = np.column_stack([x_smooth, y_smooth])

    # Cluster the trajectory points into 4 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(trajectory)
    cluster_centers = kmeans.cluster_centers_

    # Normalize cluster centers and reference keypad
    cluster_centers_norm = normalize(cluster_centers)
    reference_keys_norm = normalize(reference_keys)

    # For each cluster, find the closest reference key (for all permutations)
    cluster_indices = list(range(n_clusters))
    reduced = []
    for l in labels:
        if len(reduced) == 0 or l != reduced[-1]:
            reduced.append(l)
    reduced = reduced[:4]
    if len(reduced) < 4:
        print(f"{os.path.basename(csv_path)}: Not enough unique clusters in sequence.\n")
        return

    # Try all permutations of mapping clusters to reference keys
    pin_candidates = []
    for perm in permutations(range(len(reference_keys)), n_clusters):
        mapping = {cluster_indices[i]: key_labels[perm[i]] for i in range(n_clusters)}
        pin_digits = [mapping[l] for l in reduced]
        # Score: sum of distances between cluster centers and assigned reference keys
        score = sum(np.linalg.norm(cluster_centers_norm[i] - reference_keys_norm[perm[i]]) for i in range(n_clusters))
        pin_candidates.append((score, ''.join(pin_digits), mapping, perm))

    # Sort by score (lowest = best spatial match)
    pin_candidates.sort()
    print(f"{os.path.basename(csv_path)}: Top {min(n_guesses, len(pin_candidates))} PIN guesses:\n")
    for i, (score, pin, mapping, perm) in enumerate(pin_candidates[:n_guesses]):
        print(f"  Guess {i+1}: {pin} (mapping: {mapping}, score: {score:.3f})")
    print()

def main():
    output_root = './output'
    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                process_csv(csv_path)

if __name__ == "__main__":
    main()
