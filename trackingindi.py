import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Parameters
MAX_DISTANCE = 30  # Maximum distance to consider a match between frames

# Output directory
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture('./testdata/vid2-crop.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_number = 0
next_track_id = 0
tracks = {}  # track_id: {'center': (x, y), 'trace': [(x, y), ...], 'missed': 0}
max_missed = 5  # Remove track if not seen for this many frames

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([125, 100, 100])
    upper_purple = np.array([155, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        center_x = x + w // 2
        center_y = y + h // 2
        detections.append({'center': (center_x, center_y), 'bbox': (x, y, w, h)})

    # Prepare for matching
    track_ids = list(tracks.keys())
    track_centers = [tracks[tid]['center'] for tid in track_ids]
    detection_centers = [d['center'] for d in detections]

    # Match detections to existing tracks
    matched_tracks = set()
    matched_detections = set()
    if track_centers and detection_centers:
        dist_matrix = distance.cdist(track_centers, detection_centers)
        for t_idx, tid in enumerate(track_ids):
            d_idx = np.argmin(dist_matrix[t_idx])
            if dist_matrix[t_idx, d_idx] < MAX_DISTANCE and d_idx not in matched_detections:
                # Update track
                tracks[tid]['center'] = detection_centers[d_idx]
                tracks[tid]['trace'].append(detection_centers[d_idx])
                tracks[tid]['missed'] = 0
                matched_tracks.add(tid)
                matched_detections.add(d_idx)

    # Add new tracks for unmatched detections
    for idx, det in enumerate(detections):
        if idx not in matched_detections:
            tracks[next_track_id] = {
                'center': det['center'],
                'trace': [det['center']],
                'missed': 0
            }
            next_track_id += 1

    # Increment missed count for unmatched tracks
    for tid in track_ids:
        if tid not in matched_tracks:
            tracks[tid]['missed'] += 1

    # Remove tracks that have been missed for too long
    tracks = {tid: t for tid, t in tracks.items() if t['missed'] <= max_missed}

    # Draw bounding boxes and IDs
    for tid, t in tracks.items():
        x, y = t['center']
        # Find the bbox for this center in detections (for visualization)
        for det in detections:
            if det['center'] == t['center']:
                bx, by, bw, bh = det['bbox']
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 255), 2)
                cv2.putText(frame, f'ID {tid}', (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                break
        # Optionally, draw the trace
        for i in range(1, len(t['trace'])):
            cv2.line(frame, t['trace'][i-1], t['trace'][i], (0,255,0), 2)

    # Save the frame
    out_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(out_path, frame)

    frame_number += 1

cap.release()
cv2.destroyAllWindows()
