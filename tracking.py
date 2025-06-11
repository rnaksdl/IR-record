import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# === Easily change your video file here ===
video_path = './testdata/far_down.mp4'
# =========================================

def detect_leds_with_blue_or_purple_halo(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lower threshold to catch dimmer LEDs
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.5 or area > 1000:
            continue
        ((x, y), r) = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r+3, 255, 2)
        ring_pixels = hsv[mask == 255]
        # Blue: 110-130, Purple: 130-170
        blue_purple_pixels = np.sum(
            ((ring_pixels[:, 0] >= 110) & (ring_pixels[:, 0] <= 170)) &
            (ring_pixels[:, 1] > 40) & (ring_pixels[:, 2] > 30)
        )
        if len(ring_pixels) > 0 and blue_purple_pixels > 0.03 * len(ring_pixels):
            detected.append((x, y, r))
    return detected


def draw_leds(frame, centers, ring_center=None):
    for (x, y, r) in centers:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    if ring_center is not None:
        x, y = int(ring_center[0]), int(ring_center[1])
        cv2.circle(frame, (x, y), 8, (255, 0, 0), 2)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), 3)
    return frame

def fit_circle(xs, ys):
    if len(xs) < 3:
        return None
    A = np.c_[2*xs, 2*ys, np.ones(len(xs))]
    b = xs**2 + ys**2
    c, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    return (xc, yc)

def interpolate_centers(centers):
    # Interpolate missing (None) centers linearly
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
    # Apply Savitzky-Golay filter for smoothing
    if len(centers) < window:
        return centers
    x = savgol_filter(centers[:,0], window, poly)
    y = savgol_filter(centers[:,1], window, poly)
    return np.stack([x, y], axis=1)

def main():
    output_folder = 'output'
    frames_folder = os.path.join(output_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    led_tracks = []  # List of lists: one per tracked LED
    ring_centers = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        centers = detect_leds_with_blue_or_purple_halo(frame)
        centers_xy = [(x, y) for (x, y, r) in centers]

        # Tracking logic (simple nearest neighbor)
        if frame_idx == 0:
            for c in centers_xy:
                led_tracks.append([c])
        else:
            prev_centers = [track[-1] if len(track) > 0 else None for track in led_tracks]
            assignments = [-1] * len(centers_xy)
            used = set()
            for i, pc in enumerate(prev_centers):
                if pc is None:
                    continue
                min_dist = float('inf')
                min_j = -1
                for j, cc in enumerate(centers_xy):
                    if j in used:
                        continue
                    dist = np.linalg.norm(np.array(pc) - np.array(cc))
                    if dist < min_dist and dist < 30:
                        min_dist = dist
                        min_j = j
                if min_j != -1:
                    assignments[min_j] = i
                    used.add(min_j)
            assigned = [False] * len(centers_xy)
            for i, track in enumerate(led_tracks):
                found = False
                for j, assign in enumerate(assignments):
                    if assign == i:
                        track.append(centers_xy[j])
                        assigned[j] = True
                        found = True
                        break
                if not found:
                    track.append(None)
            for j, was_assigned in enumerate(assigned):
                if not was_assigned:
                    new_track = [None] * frame_idx
                    new_track.append(centers_xy[j])
                    led_tracks.append(new_track)
        # Estimate ring center by fitting a circle to detected LEDs
        if len(centers_xy) >= 3:
            xs = np.array([pt[0] for pt in centers_xy])
            ys = np.array([pt[1] for pt in centers_xy])
            fit = fit_circle(xs, ys)
            ring_centers.append(fit)
        else:
            ring_centers.append(None)

        # Draw and save frame
        frame_draw = draw_leds(frame.copy(), centers, ring_center=ring_centers[-1])
        frame_save_path = os.path.join(frames_folder, f'frame_{frame_idx:05d}.png')
        cv2.imwrite(frame_save_path, frame_draw)
        frame_idx += 1

    cap.release()

    # Displacement vs. time for each LED
    plt.figure(figsize=(10, 5))
    for idx, track in enumerate(led_tracks):
        track_clean = [pt for pt in track if pt is not None]
        if len(track_clean) < 2:
            continue
        initial = track_clean[0]
        displacements = []
        for pt in track:
            if pt is not None:
                dx = pt[0] - initial[0]
                dy = pt[1] - initial[1]
                displacements.append(np.sqrt(dx**2 + dy**2))
            else:
                displacements.append(None)
        times = [i for i, d in enumerate(displacements) if d is not None]
        disp_values = [d for d in displacements if d is not None]
        plt.plot(times, disp_values, marker='o', label=f'LED #{idx+1}')
    plt.xlabel('Frame Number (Time)')
    plt.ylabel('Displacement (pixels)')
    plt.title('Displacement of Purple LEDs vs. Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'displacement.png'))
    plt.close()

    # Trajectory plot for the ring center (smoothed)
    # Interpolate missing centers
    centers_interp = interpolate_centers(ring_centers)
    # Smooth the trajectory
    centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
    initial = centers_smooth[0]
    x_disp = centers_smooth[:,0] - initial[0]
    y_disp = -(centers_smooth[:,1] - initial[1])  # Invert Y for plotting

    plt.figure(figsize=(8, 8))
    plt.plot(x_disp, y_disp, marker='o')
    plt.xlabel('X Displacement (pixels)')
    plt.ylabel('Y Displacement (pixels)')
    plt.title('Trajectory of Ring Center (Smoothed)')

    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'trajectory_ring_center.png'))
    plt.close()

if __name__ == "__main__":
    main()
