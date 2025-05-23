import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Create output directory
output_dir = 'output-frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture('./testdata/vid2-crop.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_number = 0
centers = []
displacements = []

# Large kernel for detecting IR light clusters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([125, 100, 100])
    upper_purple = np.array([155, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Dilation to cluster IR lights
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest purple cluster
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        # Get centroid
        M = cv2.moments(c)
        if M['m00'] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
        else:
            center_x = x + w // 2
            center_y = y + h // 2
        
        centers.append((center_x, center_y))
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Fit ellipse if enough points
        if len(c) >= 5:
            try:
                # Get initial ellipse
                ellipse = cv2.fitEllipse(c)
                center, axes, angle = ellipse
                
                # Adjust ellipse size (half the size)
                new_axes = (axes[0] * 0.5, axes[1] * 0.5)
                
                # Adjust Y position (move upward by 15% of height)
                y_offset = new_axes[1] * 0.15
                new_center = (center[0], center[1] - y_offset)
                
                # Create adjusted ellipse
                adjusted_ellipse = (new_center, new_axes, angle)
                
                # Draw ellipse
                cv2.ellipse(frame, adjusted_ellipse, (0, 255, 0), 3)
                
                print(f"Frame {frame_number}: Ellipse fitted successfully")
            except Exception as e:
                print(f"Frame {frame_number}: Ellipse fitting error - {e}")
    else:
        centers.append(None)
        print(f"Frame {frame_number}: No purple cluster found.")

    # Save the frame
    out_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(out_path, frame)

    frame_number += 1

cap.release()

# Calculate displacement from initial position (keep your original code)
if centers:
    for idx, center in enumerate(centers):
        if idx == 0:
            if center is not None:
                initial_center = center
            else:
                for c in centers:
                    if c is not None:
                        initial_center = c
                        break
                else:
                    print("No cluster detected in any frame.")
                    exit()
        
        if center is not None:
            dx = center[0] - initial_center[0]
            dy = center[1] - initial_center[1]
            displacement = np.sqrt(dx**2 + dy**2)
            displacements.append(displacement)
        else:
            displacements.append(None)

    # Plot displacement vs. time
    times = [i for i, d in enumerate(displacements) if d is not None]
    disp_values = [d for d in displacements if d is not None]

    plt.figure(figsize=(10, 5))
    plt.plot(times, disp_values, marker='o')
    plt.xlabel('Frame Number (Time)')
    plt.ylabel('Displacement (pixels)')
    plt.title('Displacement of Purple Cluster vs. Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('displacement_with_ellipse.png')
    
    # Create trajectory visualization
    x_disp = []
    y_disp = []

    for idx, center in enumerate(centers):
        if center is not None:
            dx = center[0] - initial_center[0]
            dy = -(center[1] - initial_center[1])  # Invert Y
            x_disp.append(dx)
            y_disp.append(dy)

    if len(x_disp) > 1:
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        points = np.array([x_disp, y_disp]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create color map
        cmap = LinearSegmentedColormap.from_list(
            "rainbow6",
            [
                (1.0, 0.0, 0.0),    # Red
                (1.0, 0.5, 0.0),    # Orange
                (1.0, 1.0, 0.0),    # Yellow
                (0.0, 1.0, 0.0),    # Green
                (0.0, 0.0, 1.0),    # Blue
                (0.5, 0.0, 1.0),    # Purple
            ],
            N=256
        )
        norm = plt.Normalize(0, len(segments))

        fig, ax = plt.subplots(figsize=(8, 8))
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.arange(len(segments)))
        lc.set_linewidth(3)
        line = ax.add_collection(lc)

        # Plot start and end points
        ax.plot(x_disp[0], y_disp[0], 'o', color='red', markersize=10, label='Start')
        ax.plot(x_disp[-1], y_disp[-1], 'o', color=(0.5, 0.0, 1.0), markersize=10, label='End')

        ax.set_xlabel('X Displacement (pixels)')
        ax.set_ylabel('Y Displacement (pixels)')
        ax.set_title('Controller Movement with Ellipse Detection')
        ax.grid(True)
        ax.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig('trajectory_with_ellipse.png')

