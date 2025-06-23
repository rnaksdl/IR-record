import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from itertools import permutations
from collections import Counter

# --- CONFIGURATION ---
ACTUAL_PIN = '1397'  # <--- Set your actual PIN here
OUTPUT_DIR = './output'
PIN_FILE = '../rockyou2024/pins_4.txt'
REPORT_FOLDER = 'report'
CLUSTER_METHODS = {
    'KMeans': lambda pts: KMeans(n_clusters=4, random_state=0, n_init=20).fit_predict(pts),
    'Agglomerative': lambda pts: AgglomerativeClustering(n_clusters=4).fit_predict(pts),
    'DBSCAN': lambda pts: DBSCAN(eps=0.7, min_samples=8).fit_predict(pts)
}
PINPAD_COORDS = np.array([
    [0,0], [1,0], [2,0],   # 1 2 3
    [0,1], [1,1], [2,1],   # 4 5 6
    [0,2], [1,2], [2,2],   # 7 8 9
           [1,3]           #   0
])
PINPAD_DIGITS = ['1','2','3','4','5','6','7','8','9','0']
PINPAD_DIGIT_TO_IDX = {d: i for i, d in enumerate(PINPAD_DIGITS)}

def load_pin_priors(pin_file):
    with open(pin_file) as f:
        pins = [line.strip() for line in f if line.strip() and len(line.strip()) == 4]
    counts = Counter(pins)
    total = sum(counts.values())
    def prior(pin):
        return (counts[pin] + 1) / (total + 10000)
    return prior

def find_ring_center_cols(df):
    for xcol, ycol in [('ring_x', 'ring_y'), ('center_x', 'center_y'), ('x', 'y')]:
        if xcol in df.columns and ycol in df.columns:
            return xcol, ycol
    for col in df.columns:
        if 'x' in col and 'ring' in col:
            xcol = col
            ycol = col.replace('x', 'y')
            if ycol in df.columns:
                return xcol, ycol
    raise ValueError("Could not find ring center columns in CSV.")

def get_cluster_centers_and_times(labels, points):
    clusters = np.unique(labels)
    if -1 in clusters:
        clusters = clusters[clusters != -1]
    centers = []
    times = []
    for c in clusters:
        idxs = np.where(labels == c)[0]
        centers.append(points[idxs].mean(axis=0))
        times.append(np.mean(idxs))
    return np.array(centers), np.array(times), clusters

def plot_trajectory(points, labels, out_path, title):
    plt.figure(figsize=(6,6))
    if labels is not None:
        plt.scatter(points[:,0], points[:,1], c=labels, cmap='tab10', s=30)
    else:
        plt.plot(points[:,0], points[:,1], marker='o')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def trajectory_score(centers_ordered, pin_indices):
    cluster_vecs = np.diff(centers_ordered, axis=0)
    pinpad_vecs = np.diff(PINPAD_COORDS[pin_indices], axis=0)
    cluster_vecs = cluster_vecs / (np.linalg.norm(cluster_vecs, axis=1, keepdims=True) + 1e-8)
    pinpad_vecs = pinpad_vecs / (np.linalg.norm(pinpad_vecs, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(cluster_vecs * pinpad_vecs, axis=1)
    return 1 - np.mean(cos_sim)

def pin_total_score(pin, centers_ordered, prior_func, alpha=0.7, beta=0.3):
    try:
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
    except KeyError:
        return np.inf
    dist_score = np.sum(np.linalg.norm(centers_ordered - PINPAD_COORDS[pin_indices], axis=1))
    traj_score = trajectory_score(centers_ordered, pin_indices)
    prior_penalty = -np.log(prior_func(pin))
    return alpha * dist_score + (1 - alpha) * traj_score + beta * prior_penalty

def process_csv(csv_path, prior_func, report_dir, actual_pin):
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    os.makedirs(report_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    xs = df[xcol].values
    ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    points = np.stack([xs, ys], axis=1)
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    results = {}
    found_actual = {}
    for method, cluster_func in CLUSTER_METHODS.items():
        try:
            labels = cluster_func(points_scaled)
        except Exception as e:
            print(f"Clustering failed for {video_name} with {method}: {e}")
            labels = None
        guesses = []
        found = False
        if labels is not None:
            centers, times, clusters = get_cluster_centers_and_times(labels, points_scaled)
            if len(centers) == 4:
                # Order clusters by mean frame index (timing)
                order = np.argsort(times)
                centers_ordered = centers[order]
                # Only use this time order, not all permutations
                for pin_digits in permutations(PINPAD_DIGITS, 4):
                    pin = ''.join(pin_digits)
                    score = pin_total_score(pin, centers_ordered, prior_func, alpha=0.7, beta=0.3)
                    guesses.append((pin, score))
                guesses = sorted(set(guesses), key=lambda x: x[1])
                guesses = guesses[:100]
                found = any(pin == actual_pin for pin, _ in guesses)
        results[method] = {'guesses': guesses, 'labels': labels}
        found_actual[method] = found
        plot_trajectory(points, labels, os.path.join(video_dir, f'trajectory_{method}.png'),
                        f'Trajectory ({method})')
    plot_trajectory(points, None, os.path.join(video_dir, 'trajectory_raw.png'), 'Raw Trajectory')
    # Write HTML report in central report folder
    html_path = os.path.join(report_dir, f'{video_name}.html')
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Report for {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .pin {{ font-size: 1.2em; font-weight: bold; }}
        img {{ max-width: 400px; border: 1px solid #ccc; margin-bottom: 10px; }}
        .method-block {{ margin-bottom: 40px; }}
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
    </style>
</head>
<body>
    <h1>Video: {video_name}</h1>
    <h2>Trajectory Plot</h2>
    <div><b>Actual PIN:</b> <span style="color:blue;">{actual_pin}</span></div>
""")
        for method in CLUSTER_METHODS:
            found = found_actual[method]
            f.write(f'<div class="method-block">\n')
            f.write(f'<h2>{method} Clustering</h2>\n')
            f.write(f'<img src="{os.path.relpath(os.path.join(video_dir, f"trajectory_{method}.png"), report_dir)}" alt="{method} Trajectory">\n')
            if found:
                f.write(f'<div style="color:green;font-weight:bold;">Actual PIN {actual_pin} FOUND in top 100 guesses!</div>\n')
            else:
                f.write(f'<div style="color:red;font-weight:bold;">Actual PIN {actual_pin} NOT found in top 100 guesses.</div>\n')
            guesses = results[method]['guesses']
            if guesses:
                f.write('<table><tr><th>Rank</th><th>PIN</th><th>Score (lower is better)</th></tr>\n')
                for idx, (pin, score) in enumerate(guesses, 1):
                    highlight = ' style="background-color: #d4ffd4;"' if pin == actual_pin else ''
                    f.write(f'<tr{highlight}><td>{idx}</td><td>{pin}</td><td>{score:.4f}</td></tr>\n')
                f.write('</table>\n')
            else:
                f.write('<div class="pin" style="color:gray;">Could not extract a 4-digit PIN from this clustering.</div>\n')
            f.write('</div>\n')
        f.write("</body></html>")
    print(f"    Actual PIN {actual_pin} found in top 100? " +
          ", ".join([f"{m}: {'YES' if found_actual[m] else 'NO'}" for m in CLUSTER_METHODS]))

def main():
    prior_func = load_pin_priors(PIN_FILE)
    report_dir = os.path.join('.', REPORT_FOLDER)
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))
    total = len(csv_files)
    print(f"Found {total} *_ring_center.csv files.")
    if not csv_files:
        print("No *_ring_center.csv files found. Please check your folder structure and file names.")
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"Processing {idx}/{total}: {csv_path}")
        process_csv(csv_path, prior_func, report_dir, ACTUAL_PIN)

if __name__ == '__main__':
    main()
