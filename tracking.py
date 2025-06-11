import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_leds_with_blue_halo(image):
    # 1. Threshold for bright white centers
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    _, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)  # Only very bright spots

    # 2. Find contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10 or area > 1000:
            continue
        ((x, y), r) = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        # 3. Check for blue/purple halo just outside the white region
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r+3, 255, 2)  # Ring just outside the white blob
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ring_pixels = hsv[mask == 255]
        # Blue/purple in HSV: H in [110, 160], S > 60, V > 40
        blue_pixels = np.sum(
            (ring_pixels[:, 0] >= 110) & (ring_pixels[:, 0] <= 160) &
            (ring_pixels[:, 1] > 60) & (ring_pixels[:, 2] > 40)
        )
        if len(ring_pixels) > 0 and blue_pixels > 0.05 * len(ring_pixels):  # At least 5% of ring is blue/purple
            detected.append((x, y, r))
    return detected

def draw_leds(frame, centers):
    for (x, y, r) in centers:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    return frame

def main():
    video_path = './testdata/250609_174322_7_8s.mp4'
    output_folder = 'output'
    frames_folder = os.path.join(output_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    led_tracks = []  # List of lists: one per tracked LED
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        centers = detect_leds_with_blue_halo(frame)
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
        frame_idx += 1

        # Draw and save frame
        frame_draw = draw_leds(frame.copy(), centers)
        frame_save_path = os.path.join(frames_folder, f'frame_{frame_idx:05d}.png')
        cv2.imwrite(frame_save_path, frame_draw)

    cap.release()

    # After all frames: make two plots
    # 1. Displacement vs. time for each LED
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
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'displacement.png'))
    plt.close()

    # 2. Trajectory plot for all LEDs
    plt.figure(figsize=(8, 8))
    for idx, track in enumerate(led_tracks):
        x_disp, y_disp = [], []
        track_clean = [pt for pt in track if pt is not None]
        if len(track_clean) < 2:
            continue
        initial = track_clean[0]
        for pt in track:
            if pt is not None:
                dx = pt[0] - initial[0]
                dy = -(pt[1] - initial[1])  # Invert Y for plotting
                x_disp.append(dx)
                y_disp.append(dy)
            else:
                x_disp.append(None)
                y_disp.append(None)
        x_disp_clean = [x for x in x_disp if x is not None]
        y_disp_clean = [y for y in y_disp if y is not None]
        plt.plot(x_disp_clean, y_disp_clean, marker='o', label=f'LED #{idx+1}')
    plt.xlabel('X Displacement (pixels)')
    plt.ylabel('Y Displacement (pixels)')
    plt.title('Trajectory of Purple LEDs')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'trajectory.png'))
    plt.close()

if __name__ == "__main__":
    main()
