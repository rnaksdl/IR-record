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
ACTUAL_PIN = ''  # <--- Set your actual PIN here
OUTPUT_DIR = './output'
PIN_FILE = '../rockyou2024/pins_4.txt'
REPORT_FOLDER = './report'
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

# NEW FUNCTION: Analyze LED Y motion for stopped points and keystrokes
def analyze_led_y_motion(led_positions, threshold=2, min_stop_length=3):
    """
    Analyze LED Y positions to find stopped points and potential keystrokes
    
    Args:
        led_positions: numpy array of shape (num_leds, num_frames, 2)
        threshold: maximum Y change to consider "stopped"
        min_stop_length: minimum frames to consider a valid stop
    
    Returns:
        stops_per_led: list of stops for each LED (start_frame, end_frame, duration)
        keystrokes: list of potential keystrokes (led_idx, frame, y_value, segment)
    """
    num_leds, num_frames, _ = led_positions.shape
    stops_per_led = []
    keystrokes = []
    
    for led_idx in range(num_leds):
        y = led_positions[led_idx, :, 1]
        if np.all(np.isnan(y)):
            stops_per_led.append([])
            continue
            
        # Detect stopped points
        stopped = np.abs(np.diff(y, prepend=np.nanmean(y[~np.isnan(y)]))) < threshold
        stops = []
        in_stop = False
        start = 0
        
        for i, val in enumerate(stopped):
            if val and not in_stop and not np.isnan(y[i]):
                in_stop = True
                start = i
            elif (not val or np.isnan(y[i])) and in_stop:
                if i - start >= min_stop_length:
                    stops.append((start, i-1, i-start))  # start, end, duration
                in_stop = False
                
        if in_stop and (num_frames - start) >= min_stop_length:
            stops.append((start, num_frames-1, num_frames-start))
            
        stops_per_led.append(stops)
        
        # Check for down-then-up (jitter/keystroke) in each stopped segment
        for start, end, duration in stops:
            segment = y[start:end+1]
            if np.any(np.isnan(segment)):
                continue
                
            min_idx = np.nanargmin(segment)
            if 0 < min_idx < len(segment)-1:
                if segment[min_idx] < segment[0] and segment[min_idx] < segment[-1]:
                    keystrokes.append({
                        'led': led_idx,
                        'frame': start+min_idx,
                        'y_value': segment[min_idx],
                        'segment': (start, end)
                    })
    
    return stops_per_led, keystrokes

# NEW FUNCTION: Plot Y motion with analysis
def plot_y_motion_with_analysis(led_positions, stops_per_led, keystrokes, output_path):
    """Plot Y motion with stopped regions and keystroke markers"""
    num_leds, num_frames, _ = led_positions.shape
    plt.figure(figsize=(12, 8))
    
    for led_idx in range(num_leds):
        y = led_positions[led_idx, :, 1]
        plt.plot(np.arange(num_frames), y, label=f'LED #{led_idx+1}')
        
        # Mark stopped segments
        for start, end, _ in stops_per_led[led_idx]:
            plt.axvspan(start, end, color=f'C{led_idx}', alpha=0.1)
    
    # Mark keystrokes
    for ks in keystrokes:
        plt.plot(ks['frame'], ks['y_value'], 'rv', markersize=8)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Y Position (pixels)')
    plt.title('Up/Down (Y) Motion Analysis\nShaded: stopped regions, Red triangles: keystrokes')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# NEW FUNCTION: Load LED positions from CSV
def load_led_positions(video_dir):
    """Load LED position data from CSV files in the video directory"""
    # Try to load from led_tracks_raw.csv first
    csv_path = glob.glob(os.path.join(video_dir, '*_led_tracks_raw.csv'))
    if csv_path:
        df = pd.read_csv(csv_path[0])
        num_leds = (len(df.columns) - 1) // 2
        num_frames = len(df)
        
        led_positions = np.full((num_leds, num_frames, 2), np.nan)
        for led_idx in range(num_leds):
            x_col = f'led{led_idx+1}_x'
            y_col = f'led{led_idx+1}_y'
            if x_col in df.columns and y_col in df.columns:
                led_positions[led_idx, :, 0] = df[x_col].values
                led_positions[led_idx, :, 1] = df[y_col].values
        
        return led_positions
    
    return None

def process_csv(csv_path, prior_func, report_dir, actual_pin):
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    os.makedirs(report_dir, exist_ok=True)
    
    # Load LED positions for keystroke analysis
    led_positions = load_led_positions(video_dir)
    
    # Normal processing for PIN guessing
    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    xs = df[xcol].values
    ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    points = np.stack([xs, ys], axis=1)
    scaler = StandardScaler()
    if points.shape[0] == 0:
        print(f"Warning: No valid points in {csv_path}. Skipping.")
        return
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
    
    # If we have LED positions, analyze them for stopped points and keystrokes
    keystroke_analysis_html = ""
    if led_positions is not None:
        # Analyze Y motion for stopped points and keystrokes
        stops_per_led, keystrokes = analyze_led_y_motion(led_positions)
        
        # Plot the Y motion with analysis
        plot_y_motion_with_analysis(led_positions, stops_per_led, keystrokes,
                                  os.path.join(video_dir, 'y_motion_analysis.png'))
        
        # Print analysis results
        print(f"\nY Motion Analysis for {video_name}:")
        keystroke_analysis_html = """
        <h2>LED Motion and Keystroke Analysis</h2>
        <img src="{}" alt="Y Motion Analysis">
        <h3>Stopped Segments (Possible Digit Entry Points)</h3>
        <table>
            <tr><th>LED</th><th>Start Frame</th><th>End Frame</th><th>Duration (frames)</th></tr>
        """.format(os.path.relpath(os.path.join(video_dir, 'y_motion_analysis.png'), report_dir))
        
        for led_idx, stops in enumerate(stops_per_led):
            if stops:
                print(f"  LED #{led_idx+1} stopped segments:")
                for start, end, duration in stops:
                    print(f"    Frames {start}-{end} ({duration} frames)")
                    keystroke_analysis_html += f"""
                    <tr>
                        <td>LED #{led_idx+1}</td>
                        <td>{start}</td>
                        <td>{end}</td>
                        <td>{duration}</td>
                    </tr>"""
        
        keystroke_analysis_html += "</table>"
        
        if keystrokes:
            print(f"  Potential keystrokes detected:")
            keystroke_analysis_html += """
            <h3>Detected Keystrokes (Down-then-Up Motion)</h3>
            <table>
                <tr><th>LED</th><th>Frame</th><th>During Stopped Segment</th></tr>
            """
            
            for ks in keystrokes:
                print(f"    LED #{ks['led']+1} at frame {ks['frame']}")
                keystroke_analysis_html += f"""
                <tr>
                    <td>LED #{ks['led']+1}</td>
                    <td>{ks['frame']}</td>
                    <td>{ks['segment'][0]}-{ks['segment'][1]}</td>
                </tr>"""
            
            keystroke_analysis_html += "</table>"
            
            # Add repeated digit analysis
            keystroke_analysis_html += """
            <h3>Repeated Digit Analysis</h3>
            <p>Segments with significantly longer duration may indicate repeated digits.</p>
            """
            
            # Get average duration of stopped segments
            all_durations = [duration for led_stops in stops_per_led for _, _, duration in led_stops]
            if all_durations:
                avg_duration = np.mean(all_durations)
                keystroke_analysis_html += f"<p>Average stopped segment duration: {avg_duration:.1f} frames</p>"
                
                # Find segments that are much longer than average
                long_segments = []
                for led_idx, stops in enumerate(stops_per_led):
                    for start, end, duration in stops:
                        if duration > 1.5 * avg_duration:  # 50% longer than average
                            long_segments.append((led_idx, start, end, duration))
                
                if long_segments:
                    keystroke_analysis_html += """
                    <table>
                        <tr><th>LED</th><th>Start-End</th><th>Duration</th><th>vs Avg</th></tr>
                    """
                    
                    for led_idx, start, end, duration in long_segments:
                        ratio = duration / avg_duration
                        keystroke_analysis_html += f"""
                        <tr>
                            <td>LED #{led_idx+1}</td>
                            <td>{start}-{end}</td>
                            <td>{duration}</td>
                            <td>{ratio:.1f}x avg</td>
                        </tr>"""
                    
                    keystroke_analysis_html += "</table>"
                    keystroke_analysis_html += "<p><b>These segments with longer duration may indicate repeated digits.</b></p>"
                else:
                    keystroke_analysis_html += "<p>No segments with significantly longer duration found.</p>"
    
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
        img {{ max-width: 600px; border: 1px solid #ccc; margin-bottom: 10px; }}
        .method-block {{ margin-bottom: 40px; }}
        table {{ border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Video: {video_name}</h1>
    <h2>Trajectory Plot</h2>
    <div><b>Actual PIN:</b> <span style="color:blue;">{actual_pin}</span></div>
    {keystroke_analysis_html}
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
    os.makedirs(report_dir, exist_ok=True)
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
