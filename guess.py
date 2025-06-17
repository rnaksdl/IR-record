import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

reference_keys = np.array([
    [-503, -306],   # 1
    [0,   -311],    # 2
    [504,  -310],   # 3
    [-513,     0],  # 4
    [0,      0],    # 5 (center)
    [513,    -2],   # 6
    [-510,   307],  # 7
    [-1,   304],    # 8
    [513,   303],   # 9
    [ -3,   620],   # 0
])
key_labels = ['1','2','3','4','5','6','7','8','9','0']

def remove_outlier_tracks(led_tracks, min_frames=10, min_disp=5):
    filtered = []
    for track in led_tracks:
        pts = np.array([pt for pt in track if pt is not None])
        if len(pts) < min_frames:
            continue
        disp = np.linalg.norm(pts[-1] - pts[0])
        if disp < min_disp:
            continue
        filtered.append(track)
    return filtered

def interpolate_centers(centers):
    centers = np.array([
        [c[0], c[1]] if c is not None else [np.nan, np.nan]
        for c in centers
    ])
    for i in range(2):  # x and y
        valid = ~np.isnan(centers[:, i])
        if np.sum(valid) < 2:
            continue
        centers[:, i] = np.interp(
            np.arange(len(centers)),
            np.flatnonzero(valid),
            centers[valid, i]
        )
    return centers

def smooth_centers(centers, window=9, poly=2):
    if len(centers) < window:
        return centers
    x = savgol_filter(centers[:,0], window, poly)
    y = savgol_filter(centers[:,1], window, poly)
    return np.stack([x, y], axis=1)

def find_stopped_segments(centers, min_length=5, velocity_thresh=2, accel_thresh=0.5):
    velocities = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    accelerations = np.abs(np.diff(velocities))
    stopped = (velocities < velocity_thresh) & (np.concatenate([[False], accelerations < accel_thresh]))
    segments = []
    start = None
    for i, val in enumerate(stopped):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_length:
                segments.append((start, i))
            start = None
    if start is not None and len(centers) - start >= min_length:
        segments.append((start, len(centers)))
    # Merge segments that are very close together
    merged = []
    for seg in segments:
        if not merged or seg[0] - merged[-1][1] > 3:
            merged.append(list(seg))
        else:
            merged[-1][1] = seg[1]
    return [tuple(m) for m in merged]

def beam_search(stops, beam_width=50, max_pins=10):
    # stops: list of lists of (key_label, distance, key_index, duration)
    beams = [([], 0, -1)]  # (partial_pin, total_score, last_key_idx)
    for stop in stops[:4]:
        new_beams = []
        for pin, score, last_idx in beams:
            for key, dist, idx, duration in stop:
                # Allow repeated digits if stops are far apart, else skip repeats
                if key in pin and duration < 10:
                    continue
                # Score: proximity + duration bonus
                new_score = score - dist + duration * 0.2
                if last_idx != -1:
                    # Add transition penalty (prefer small moves)
                    trans_penalty = np.linalg.norm(reference_keys[last_idx] - reference_keys[idx]) / 100.0
                    new_score -= trans_penalty
                new_beams.append((pin + [key], new_score, idx))
        # Keep only top beam_width beams
        new_beams = sorted(new_beams, key=lambda x: -x[1])[:beam_width]
        beams = new_beams
    # Output up to max_pins unique pins
    pins = []
    seen = set()
    for pin, _, _ in beams:
        pin_str = ''.join(pin)
        if pin_str not in seen:
            pins.append(pin_str)
            seen.add(pin_str)
        if len(pins) == max_pins:
            break
    return pins

def process_csv(csv_path, top_n_per_stop=4, max_pins=10):
    df = pd.read_csv(csv_path)
    n_leds = (len(df.columns) - 1) // 2
    led_tracks = []
    for led_idx in range(n_leds):
        track = []
        for _, row in df.iterrows():
            x = row.get(f'led{led_idx+1}_x')
            y = row.get(f'led{led_idx+1}_y')
            if not (np.isnan(x) or np.isnan(y)):
                track.append((x, y))
            else:
                track.append(None)
        led_tracks.append(track)
    led_tracks = remove_outlier_tracks(led_tracks, min_frames=10, min_disp=5)
    if not led_tracks:
        return None, "No valid tracks after outlier removal"
    ring_centers = []
    for frame_idx in range(len(df)):
        pts = [track[frame_idx] for track in led_tracks if track[frame_idx] is not None]
        if pts:
            ring_centers.append(np.mean(pts, axis=0))
        else:
            ring_centers.append(None)
    centers_interp = interpolate_centers(ring_centers)
    centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
    if len(centers_smooth) == 0 or np.isnan(centers_smooth).all():
        return None, "No valid ring center trajectory"
    segments = find_stopped_segments(centers_smooth, min_length=5, velocity_thresh=2, accel_thresh=0.5)
    if len(segments) < 4:
        return None, f"Could not find at least 4 stopped segments (found {len(segments)})"
    # For each segment, get the mean position and the top N closest keys
    stops = []
    for seg_idx, (start, end) in enumerate(segments):
        press_pos = np.mean(centers_smooth[start:end], axis=0)
        dists = np.linalg.norm(reference_keys - press_pos, axis=1)
        top_indices = np.argsort(dists)[:top_n_per_stop]
        duration = end - start
        stops.append([(key_labels[i], dists[i], i, duration) for i in top_indices])
    # Debug: print candidates for each stop
    for i, stop in enumerate(stops[:4]):
        print(f"Stop {i+1} candidates: {[c[0] for c in stop]}")
    # Use beam search to generate plausible PINs
    pins = beam_search(stops, beam_width=50, max_pins=max_pins)
    if not pins:
        return None, "No diverse PINs found (all stops may be on the same key)"
    return pins, None

# Recursively process all CSVs and print 10 most likely PINs for each folder
output_root = './output'
folder_pins = {}
folder_errors = {}
for root, dirs, files in os.walk(output_root):
    csvs = [f for f in files if f.endswith('.csv')]
    if not csvs:
        continue
    found = False
    for csv_file in csvs:
        csv_path = os.path.join(root, csv_file)
        pins, error = process_csv(csv_path, top_n_per_stop=4, max_pins=10)
        if pins is not None and len(pins) > 0:
            folder_pins[root] = pins
            found = True
            break  # Only need one valid PIN set per folder
        elif error:
            folder_errors[root] = error
    if not found and root not in folder_pins:
        if root not in folder_errors:
            folder_errors[root] = "No valid CSVs found"

for folder in sorted(set(list(folder_pins.keys()) + list(folder_errors.keys()))):
    if folder in folder_pins:
        print(f"{folder}:")
        for pin in folder_pins[folder]:
            print(f"  {pin}")
    else:
        print(f"{folder}: Could not find valid PIN - {folder_errors[folder]}")
