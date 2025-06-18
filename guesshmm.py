import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter

# Reference keypad layout (from guess.py)
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

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def detect_keystrokes(y, fps, expected_n=4):
    y_smooth = butter_lowpass_filter(y, cutoff=6, fs=fps)
    accel = np.gradient(np.gradient(y_smooth))
    peaks, props = find_peaks(-accel, prominence=0.5)
    prominences = props['prominences']
    if len(prominences) == 0:
        return np.array([], dtype=int)
    # Pick the top N peaks by prominence
    if len(peaks) > expected_n:
        idx = np.argsort(prominences)[-expected_n:]
        keystroke_frames = np.sort(peaks[idx])
    else:
        keystroke_frames = peaks
    return keystroke_frames

def map_to_keys(xs_k, ys_k):
    pins = []
    for x, y in zip(xs_k, ys_k):
        dists = np.linalg.norm(reference_keys - np.array([x, y]), axis=1)
        idx = np.argmin(dists)
        pins.append(key_labels[idx])
    return pins

def process_csv(csv_path, n_keys=4, fps=30):
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
    valid_idx = [i for i, c in enumerate(ring_centers) if c is not None]
    if len(valid_idx) < n_keys:
        return None, f"Not enough valid ring center frames ({len(valid_idx)})"
    centers_interp = interpolate_centers(ring_centers)
    centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
    if len(centers_smooth) == 0 or np.isnan(centers_smooth).all():
        return None, "No valid ring center trajectory"
    ys = centers_smooth[:,1]
    xs = centers_smooth[:,0]
    keystroke_frames = detect_keystrokes(ys, fps, expected_n=n_keys)
    if len(keystroke_frames) < n_keys:
        return None, f"Not enough keystrokes detected ({len(keystroke_frames)})"
    xs_k = xs[keystroke_frames]
    ys_k = ys[keystroke_frames]
    pin = map_to_keys(xs_k, ys_k)
    return pin, None

def main():
    output_root = './output'
    n_keys = 4
    fps = 30

    folder_pins = {}
    folder_errors = {}
    for root, dirs, files in os.walk(output_root):
        csvs = [f for f in files if f.endswith('.csv')]
        if not csvs:
            continue
        found = False
        for csv_file in csvs:
            csv_path = os.path.join(root, csv_file)
            pin, error = process_csv(csv_path, n_keys=n_keys, fps=fps)
            if pin is not None and len(pin) == n_keys:
                folder_pins[root] = pin
                found = True
                break  # Only need one valid PIN per folder
            elif error:
                folder_errors[root] = error
        if not found and root not in folder_pins:
            if root not in folder_errors:
                folder_errors[root] = "No valid CSVs found"

    for folder in sorted(set(list(folder_pins.keys()) + list(folder_errors.keys()))):
        if folder in folder_pins:
            print(f"{folder}:")
            print(f"  Most likely PIN: {''.join(folder_pins[folder])}")
        else:
            print(f"{folder}: Could not find valid PIN - {folder_errors[folder]}")

if __name__ == "__main__":
    main()
