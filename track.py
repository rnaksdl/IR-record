import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter

input_folder = './input'  # Changed to jitter_vid from input
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)

def detect_leds_with_blue_or_purple_halo(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find bright spots (keep this part)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours (keep this part)
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    
    for c in cnts:
        # Filter by area (keep this part)
        area = cv2.contourArea(c)
        if area < 0.5 or area > 1000:
            continue
            
        # Instead of fitting a circle, use the actual contour shape
        mask_inner = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask_inner, [c], 0, 255, -1)  # Fill the contour
        
        # Create a dilated version of the contour for the surrounding area
        mask_outer = np.zeros(gray.shape, dtype=np.uint8)
        dilated_contour = cv2.dilate(mask_inner.copy(), kernel, iterations=3)  # Dilate by 3 iterations
        cv2.drawContours(mask_outer, [cv2.findContours(dilated_contour, cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)[0][0]], 0, 255, -1)
        
        # Create a ring mask (outer - inner)
        mask_ring = cv2.subtract(mask_outer, mask_inner)
        
        # Check for blue/purple pixels in the ring (similar to original)
        ring_pixels = hsv[mask_ring == 255]
        blue_purple_pixels = np.sum(
            ((ring_pixels[:, 0] >= 110) & (ring_pixels[:, 0] <= 170)) &
            (ring_pixels[:, 1] > 40) & (ring_pixels[:, 2] > 30)
        )
        
        if len(ring_pixels) > 0 and blue_purple_pixels > 0.03 * len(ring_pixels):
            # Calculate centroid of the contour for tracking
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate equivalent radius for reference
                equiv_radius = int(np.sqrt(area / np.pi))
                
                detected.append((cx, cy, equiv_radius))
                
    # Also detect direct blue shapes (like in the previous enhancement)
    blue_mask = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([140, 255, 255]))
    blue_opening = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    blue_cnts, _ = cv2.findContours(blue_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in blue_cnts:
        area = cv2.contourArea(c)
        if area < 3 or area > 500:
            continue
            
        # Calculate centroid
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate equivalent radius for reference
        equiv_radius = int(np.sqrt(area / np.pi))
        
        # Check for duplicates
        is_duplicate = False
        for (x_det, y_det, r_det) in detected:
            dist = np.sqrt((cx-x_det)**2 + (cy-y_det)**2)
            if dist < 2.5 * max(equiv_radius, r_det):
                is_duplicate = True
                break
                
        if not is_duplicate:
            detected.append((cx, cy, equiv_radius))
    
    # Add a final merging step to combine nearby detections
    if len(detected) > 1:
        merged_detected = []
        min_merge_distance = 25  # Adjust based on LED size
        processed = [False] * len(detected)
        
        for i in range(len(detected)):
            if processed[i]:
                continue
                
            x1, y1, r1 = detected[i]
            cluster = [(x1, y1, r1)]
            processed[i] = True
            
            # Find all points close to this one
            for j in range(i + 1, len(detected)):
                if processed[j]:
                    continue
                    
                x2, y2, r2 = detected[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if dist < min_merge_distance:
                    cluster.append((x2, y2, r2))
                    processed[j] = True
            
            # Merge the cluster into one point
            if len(cluster) == 1:
                merged_detected.append(cluster[0])
            else:
                # Weight by radius (larger points have more influence)
                total_weight = sum(r for _, _, r in cluster)
                merged_x = sum(x * r / total_weight for x, _, r in cluster)
                merged_y = sum(y * r / total_weight for _, y, r in cluster)
                merged_r = max(r for _, _, r in cluster)  # Use largest radius
                
                merged_detected.append((int(merged_x), int(merged_y), merged_r))
        
        return merged_detected
    
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

def find_flat_segments(displacements, n_segments=4, min_length=10):
    displacements = np.array(displacements)
    isnan = np.isnan(displacements)
    displacements[isnan] = np.nanmedian(displacements)
    diffs = np.abs(np.diff(displacements))
    window = max(5, len(displacements)//(n_segments*3))
    flatness = np.convolve(diffs, np.ones(window)/window, mode='valid')
    flat_indices = np.argsort(flatness)[:n_segments]
    segments = []
    for idx in sorted(flat_indices):
        start = idx
        end = min(idx+window, len(displacements))
        if end - start >= min_length:
            segments.append((start, end))
    non_overlap = []
    last_end = -1
    for seg in segments:
        if seg[0] > last_end:
            non_overlap.append(seg)
            last_end = seg[1]
        if len(non_overlap) == n_segments:
            break
    return non_overlap

def process_video(video_path, output_folder, video_idx, total_videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video {video_idx}/{total_videos}: {video_path} ...")
    video_out_folder = os.path.join(output_folder, video_name)
    frames_folder = os.path.join(video_out_folder, 'frames')
    plots_folder = os.path.join(video_out_folder, 'flat_segments')
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    led_tracks = []
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

        # Print frame progress
        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"    Frame {frame_idx+1}/{total_frames} ...")
        frame_idx += 1

    cap.release()

    # 1. Save overall displacement/time and CSV (before outlier removal)
    csv_path = os.path.join(video_out_folder, f'{video_name}_led_tracks_raw.csv')
    os.makedirs(video_out_folder, exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['frame']
        for i in range(len(led_tracks)):
            header += [f'led{i+1}_x', f'led{i+1}_y']
        writer.writerow(header)
        for f in range(frame_idx):
            row = [f]
            for track in led_tracks:
                if f < len(track) and track[f] is not None:
                    row += [track[f][0], track[f][1]]
                else:
                    row += [None, None]
            writer.writerow(row)

    # Displacement vs. time for each LED (before outlier removal)
    plt.figure(figsize=(10, 5))
    all_displacements = []
    for idx, track in enumerate(led_tracks):
        track_clean = [pt for pt in track if pt is not None]
        if len(track_clean) < 2:
            all_displacements.append([np.nan]*len(track))
            continue
        initial = track_clean[0]
        displacements = []
        for pt in track:
            if pt is not None:
                dx = pt[0] - initial[0]
                dy = pt[1] - initial[1]
                displacements.append(np.sqrt(dx**2 + dy**2))
            else:
                displacements.append(np.nan)
        all_displacements.append(displacements)
        times = np.arange(len(displacements))
        plt.plot(times, displacements, marker='o', label=f'LED #{idx+1}')
    plt.xlabel('Frame Number (Time)')
    plt.ylabel('Displacement (pixels)')
    plt.title('Displacement of Purple LEDs vs. Time (Raw)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(video_out_folder, 'displacement_raw.png'))
    plt.close()

    # 2. Remove outlier tracks
    led_tracks = remove_outlier_tracks(led_tracks, min_frames=10, min_disp=5)

    # 3. Trajectory plot for the ring center (smoothed)
    centers_interp = interpolate_centers(ring_centers)
    centers_smooth = smooth_centers(centers_interp, window=9, poly=2)
    if len(centers_smooth) > 0:
        # Save smoothed ring center trajectory as CSV
        ring_center_csv_path = os.path.join(video_out_folder, f'{video_name}_ring_center.csv')
        with open(ring_center_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'ring_x', 'ring_y'])
            for i, (x, y) in enumerate(centers_smooth):
                writer.writerow([i, x, y])

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
        plt.savefig(os.path.join(video_out_folder, 'trajectory_ring_center.png'))
        plt.close()

        # --- Velocity calculation ---
        if len(centers_smooth) > 1:
            velocities = np.linalg.norm(np.diff(centers_smooth, axis=0), axis=1)
            times = np.arange(1, len(centers_smooth))  # Frame numbers for velocity

            # Plot velocity vs. time
            plt.figure(figsize=(10, 5))
            plt.plot(times, velocities, marker='o')
            plt.xlabel('Frame Number (Time)')
            plt.ylabel('Velocity (pixels/frame)')
            plt.title('Velocity of Ring Center vs. Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'velocity_ring_center.png'))
            plt.close()

            # Plot velocity histogram for threshold tuning
            plt.figure()
            plt.hist(velocities, bins=50)
            plt.xlabel('Velocity (pixels/frame)')
            plt.ylabel('Count')
            plt.title('Velocity Histogram')
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'velocity_histogram.png'))
            plt.close()

            velocities_reshape = velocities.reshape(-1, 1)

            # --- KMeans clustering (2 clusters) ---
            kmeans = KMeans(n_clusters=2, random_state=0).fit(velocities_reshape)
            labels_kmeans = kmeans.labels_
            cluster_means = [np.mean(velocities[labels_kmeans == i]) for i in range(2)]
            not_moving_label = np.argmin(cluster_means)
            not_moving_mask_kmeans = np.concatenate([[False], labels_kmeans == not_moving_label])

            plt.figure(figsize=(8, 8))
            plt.plot(x_disp, y_disp, color='lightgray', label='All')
            plt.plot(x_disp[not_moving_mask_kmeans], y_disp[not_moving_mask_kmeans], 'o-', color='blue', label='Not moving (KMeans)')
            plt.xlabel('X Displacement (pixels)')
            plt.ylabel('Y Displacement (pixels)')
            plt.title('Trajectory of Ring Center (Not Moving, KMeans)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'trajectory_ring_center_not_moving_kmeans.png'))
            plt.close()

            # --- KMeans clustering (3 clusters, strictest) ---
            kmeans3 = KMeans(n_clusters=3, random_state=0).fit(velocities_reshape)
            labels_kmeans3 = kmeans3.labels_
            cluster_means3 = [np.mean(velocities[labels_kmeans3 == i]) for i in range(3)]
            not_moving_label3 = np.argmin(cluster_means3)
            not_moving_mask_kmeans3 = np.concatenate([[False], labels_kmeans3 == not_moving_label3])

            plt.figure(figsize=(8, 8))
            plt.plot(x_disp, y_disp, color='lightgray', label='All')
            plt.plot(x_disp[not_moving_mask_kmeans3], y_disp[not_moving_mask_kmeans3], 'o-', color='navy', label='Not moving (KMeans, strict)')
            plt.xlabel('X Displacement (pixels)')
            plt.ylabel('Y Displacement (pixels)')
            plt.title('Trajectory of Ring Center (Not Moving, KMeans 3-cluster)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'trajectory_ring_center_not_moving_kmeans3.png'))
            plt.close()

            # --- Gaussian Mixture Model clustering (2 clusters) ---
            gmm = GaussianMixture(n_components=2, random_state=0).fit(velocities_reshape)
            labels_gmm = gmm.predict(velocities_reshape)
            gmm_means = [np.mean(velocities[labels_gmm == i]) for i in range(2)]
            not_moving_label_gmm = np.argmin(gmm_means)
            not_moving_mask_gmm = np.concatenate([[False], labels_gmm == not_moving_label_gmm])

            plt.figure(figsize=(8, 8))
            plt.plot(x_disp, y_disp, color='lightgray', label='All')
            plt.plot(x_disp[not_moving_mask_gmm], y_disp[not_moving_mask_gmm], 'o-', color='green', label='Not moving (GMM)')
            plt.xlabel('X Displacement (pixels)')
            plt.ylabel('Y Displacement (pixels)')
            plt.title('Trajectory of Ring Center (Not Moving, GMM)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'trajectory_ring_center_not_moving_gmm.png'))
            plt.close()

            # --- Gaussian Mixture Model clustering (3 clusters, strictest) ---
            gmm3 = GaussianMixture(n_components=3, random_state=0).fit(velocities_reshape)
            labels_gmm3 = gmm3.predict(velocities_reshape)
            gmm_means3 = [np.mean(velocities[labels_gmm3 == i]) for i in range(3)]
            not_moving_label_gmm3 = np.argmin(gmm_means3)
            not_moving_mask_gmm3 = np.concatenate([[False], labels_gmm3 == not_moving_label_gmm3])

            plt.figure(figsize=(8, 8))
            plt.plot(x_disp, y_disp, color='lightgray', label='All')
            plt.plot(x_disp[not_moving_mask_gmm3], y_disp[not_moving_mask_gmm3], 'o-', color='darkgreen', label='Not moving (GMM, strict)')
            plt.xlabel('X Displacement (pixels)')
            plt.ylabel('Y Displacement (pixels)')
            plt.title('Trajectory of Ring Center (Not Moving, GMM 3-cluster)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'trajectory_ring_center_not_moving_gmm3.png'))
            plt.close()

            # --- Manual strict threshold ---
            velocity_threshold = 0.5  # adjust as needed for strictness
            not_moving_mask_manual = np.concatenate([[False], velocities < velocity_threshold])

            plt.figure(figsize=(8, 8))
            plt.plot(x_disp, y_disp, color='lightgray', label='All')
            plt.plot(x_disp[not_moving_mask_manual], y_disp[not_moving_mask_manual], 'o-', color='red', label=f'Not moving (<{velocity_threshold})')
            plt.xlabel('X Displacement (pixels)')
            plt.ylabel('Y Displacement (pixels)')
            plt.title('Trajectory of Ring Center (Strict Not Moving, Manual Threshold)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(video_out_folder, 'trajectory_ring_center_not_moving_strict.png'))
            plt.close()

    # 4. Find and plot 4 flat segments (zoomed in Y)
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
                displacements.append(np.nan)
        segments = find_flat_segments(displacements, n_segments=4, min_length=10)
        times = np.arange(len(displacements))
        for seg_idx, (start, end) in enumerate(segments):
            segment = np.array(displacements[start:end])
            valid = ~np.isnan(segment)
            if np.sum(valid) == 0:
                continue  # Skip this segment, nothing to plot
            plt.figure(figsize=(8, 4))
            plt.plot(times[start:end][valid], segment[valid], marker='o')
            plt.xlabel('Frame Number')
            plt.ylabel('Displacement (pixels)')
            plt.title(f'LED #{idx+1} Flat Segment {seg_idx+1}')
            plt.ylim(
                np.nanmin(segment[valid]) - 1,
                np.nanmax(segment[valid]) + 1
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, f'led{idx+1}_flat{seg_idx+1}.png'))
            plt.close()

    # NEW CODE: Calculate the center of the ring using all detected LEDs
    # Build an array of shape (num_leds, num_frames, 2)
    max_len = max(len(track) for track in led_tracks)
    led_positions = []
    for track in led_tracks:
        arr = np.full((max_len, 2), np.nan)
        for i, pt in enumerate(track):
            if pt is not None:
                arr[i] = pt
        led_positions.append(arr)
    led_positions = np.array(led_positions)  # shape: (num_leds, num_frames, 2)
    
    # Calculate ring center for each frame as mean of all detected LEDs
    ring_centers_all_leds = np.nanmean(led_positions, axis=0)  # shape: (num_frames, 2)
    
    # Save the all-LED ring center as CSV
    all_leds_center_csv_path = os.path.join(video_out_folder, f'{video_name}_all_leds_center.csv')
    with open(all_leds_center_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'center_x', 'center_y'])
        for i, (x, y) in enumerate(ring_centers_all_leds):
            writer.writerow([i, x, y])
    
    # Plot the calculated ring center trajectory (using all LEDs)
    plt.figure(figsize=(8, 8))
    plt.plot(ring_centers_all_leds[:,0], ring_centers_all_leds[:,1], marker='o')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Ring Center Trajectory (Mean of All LEDs)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(video_out_folder, 'ring_center_mean_of_leds.png'))
    plt.close()
    
    # NEW CODE: Plot Y (up/down) motion of each LED over time
    plt.figure(figsize=(10, 6))
    for idx, arr in enumerate(led_positions):
        plt.plot(np.arange(arr.shape[0]), arr[:,1], label=f'LED #{idx+1}')
    plt.xlabel('Frame Number')
    plt.ylabel('Y Position (pixels)')
    plt.title('Up/Down (Y) Motion of Each LED Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(video_out_folder, 'leds_y_motion.png'))
    plt.close()

if __name__ == "__main__":
    video_files = [fname for fname in os.listdir(input_folder) 
                  if fname.lower().endswith(('.mp4'))]
    total_videos = len(video_files)
    for idx, fname in enumerate(video_files, 1):
        video_path = os.path.join(input_folder, fname)
        process_video(video_path, output_folder, idx, total_videos)
    print("All videos processed.")
