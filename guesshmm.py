import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations

# --- CONFIGURATION ---
ACTUAL_PIN = '1397'  # Set your actual PIN here for evaluation
OUTPUT_DIR = './output'
REPORT_FOLDER = 'reporthmm'
PINPAD_COORDS = np.array([
    [0,0], [1,0], [2,0],   # 1 2 3
    [0,1], [1,1], [2,1],   # 4 5 6
    [0,2], [1,2], [2,2],   # 7 8 9
    [1,3]                  # 0
])
PINPAD_DIGITS = ['1','2','3','4','5','6','7','8','9','0']

def normalize_trajectory(points, pinpad_coords):
    min_pts, max_pts = points.min(axis=0), points.max(axis=0)
    min_pad, max_pad = pinpad_coords.min(axis=0), pinpad_coords.max(axis=0)
    scale = (max_pad - min_pad) / (max_pts - min_pts + 1e-8)
    norm_points = (points - min_pts) * scale + min_pad
    return norm_points

def cluster_and_order(points, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # For each cluster, get the mean time index (order)
    order = []
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        order.append((c, idxs.mean()))
    order = [c for c, _ in sorted(order, key=lambda x: x[1])]
    centers_ordered = centers[order]
    return centers_ordered

def assign_to_pinpad(centers_ordered):
    # For each permutation of pinpad digits, compute total distance
    best_pins = []
    for pin_digits in permutations(PINPAD_DIGITS, 4):
        pin_indices = [PINPAD_DIGITS.index(d) for d in pin_digits]
        coords = PINPAD_COORDS[pin_indices]
        score = np.sum(np.linalg.norm(centers_ordered - coords, axis=1))
        best_pins.append((''.join(pin_digits), score))
    best_pins.sort(key=lambda x: x[1])
    return best_pins[:100]

def process_csv(csv_path, actual_pin, outdir):
    df = pd.read_csv(csv_path)
    for xcol, ycol in [('ring_x', 'ring_y'), ('center_x', 'center_y'), ('x', 'y')]:
        if xcol in df.columns and ycol in df.columns:
            xs = df[xcol].values
            ys = df[ycol].values
            mask = ~np.isnan(xs) & ~np.isnan(ys)
            points = np.stack([xs[mask], ys[mask]], axis=1)
            break
    else:
        raise ValueError("Could not find ring center columns in CSV.")
    points = normalize_trajectory(points, PINPAD_COORDS)
    centers_ordered = cluster_and_order(points, n_clusters=4)
    top100 = assign_to_pinpad(centers_ordered)
    decoded_pin = top100[0][0]
    found = any(pin == actual_pin for pin, _ in top100)
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(points[:,0], points[:,1], marker='o', label='Trajectory')
    plt.scatter(centers_ordered[:,0], centers_ordered[:,1], c='red', s=80, label='Cluster Centers')
    for i, mu in enumerate(PINPAD_COORDS):
        plt.scatter(mu[0], mu[1], marker='s', s=100, label=f"{PINPAD_DIGITS[i]}")
    plt.title(f"Top guess: {decoded_pin} (actual: {actual_pin})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plot_path = os.path.join(outdir, os.path.basename(os.path.dirname(csv_path)) + '_cluster_decoded.png')
    plt.savefig(plot_path)
    plt.close()
    # Write HTML report
    html_path = os.path.join(outdir, os.path.basename(os.path.dirname(csv_path)) + '_report.html')
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PIN Report for {os.path.basename(os.path.dirname(csv_path))}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .pin {{ font-size: 1.2em; font-weight: bold; }}
        img {{ max-width: 400px; border: 1px solid #ccc; margin-bottom: 10px; }}
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
        .highlight {{ background-color: #d4ffd4; }}
    </style>
</head>
<body>
    <h1>PIN Report: {os.path.basename(os.path.dirname(csv_path))}</h1>
    <h2>Trajectory Plot</h2>
    <img src="{os.path.basename(plot_path)}" alt="Trajectory Plot">
    <div><b>Actual PIN:</b> <span style="color:blue;">{actual_pin}</span></div>
    <div><b>Top guess:</b> <span style="color:green;">{decoded_pin}</span></div>
    <div><b>Match:</b> {'YES' if decoded_pin == actual_pin else 'NO'}</div>
    <div><b>Actual PIN in top 100:</b> {'YES' if found else 'NO'}</div>
    <h2>Top 100 Guesses</h2>
    <table>
        <tr><th>Rank</th><th>PIN</th><th>Score (lower is better)</th></tr>
""")
        for rank, (pin, score) in enumerate(top100, 1):
            highlight = ' class="highlight"' if pin == actual_pin else ''
            f.write(f'<tr{highlight}><td>{rank}</td><td>{pin}</td><td>{score:.2f}</td></tr>\n')
        f.write("""
    </table>
</body>
</html>
""")
    print(f"Report written: {html_path}")
    return found

def main():
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))
    print(f"Found {len(csv_files)} *_ring_center.csv files.")
    found_count = 0
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"Processing {idx}/{len(csv_files)}: {csv_path}")
        found = process_csv(csv_path, ACTUAL_PIN, REPORT_FOLDER)
        if found:
            found_count += 1
    print(f"\nSummary: Found actual PIN in top 100 for {found_count} out of {len(csv_files)} videos.")

if __name__ == '__main__':
    main()
