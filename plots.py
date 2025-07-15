import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from itertools import product
import time
import datetime
import shutil
import webbrowser

# --- USER CONFIGURABLE PARAMETERS ---
PIN_LENGTH = 4
OUTPUT_DIR = './output'
REPORT_FOLDER = './report'
TIME_WEIGHT = 0.5  # Weight for time dimension in clustering (0-1)

# Keep the box size parameter for detection purposes, but don't use it for scoring
SAME_DIGIT_BOX_SIZE = 80  # Default was 80, try lowering to 50-60 if needed

# Button dimensions
BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9
X_OFFSET = BUTTON_WIDTH/2
Y_OFFSET = BUTTON_HEIGHT/2

# Create coordinate array with correct Y-axis
PINPAD_COORDS = np.array([
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],    # 1
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],    # 2
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 0*BUTTON_HEIGHT + 0*GAP + Y_OFFSET],    # 3
    
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],    # 4
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],    # 5
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 1*BUTTON_HEIGHT + 1*GAP + Y_OFFSET],    # 6
    
    [0*BUTTON_WIDTH + 0*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],    # 7
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],    # 8
    [2*BUTTON_WIDTH + 2*GAP + X_OFFSET, 2*BUTTON_HEIGHT + 2*GAP + Y_OFFSET],    # 9
    
    [1*BUTTON_WIDTH + 1*GAP + X_OFFSET, 3*BUTTON_HEIGHT + 3*GAP + Y_OFFSET]     # 0
])

PINPAD_DIGITS = ['1','2','3','4','5','6','7','8','9','0']
PINPAD_DIGIT_TO_IDX = {d: i for i, d in enumerate(PINPAD_DIGITS)}

def find_ring_center_cols(df):
    """Find X and Y columns for ring center in the dataframe"""
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

def are_all_points_close(points, max_width=None, max_height=None):
    """
    Check if all points are confined to a box of specified size
    Uses the global SAME_DIGIT_BOX_SIZE by default
    """
    if max_width is None:
        max_width = SAME_DIGIT_BOX_SIZE
    if max_height is None:
        max_height = SAME_DIGIT_BOX_SIZE
        
    if len(points) < 5:  # Not enough points to make a determination
        return False
    
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    return width <= max_width and height <= max_height

def calculate_speeds(points):
    """Calculate speeds between consecutive points"""
    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    # Add a placeholder speed for the first point
    speeds = np.insert(speeds, 0, np.median(speeds))
    return speeds

def filter_by_speed(points, speeds, frame_indices=None):
    """Filter points based on speed using simple percentile threshold"""
    # Keep points with speeds in the lowest 40% (more lenient)
    threshold = np.percentile(speeds, 40)
    slow_mask = speeds <= threshold
    filtered_points = points[slow_mask]
    
    if frame_indices is not None:
        filtered_frames = frame_indices[slow_mask]
        return filtered_points, filtered_frames
    else:
        return filtered_points, np.where(slow_mask)[0]

def time_aware_clustering(points, frame_indices, n_clusters=4, time_weight=TIME_WEIGHT):
    """
    Perform clustering in both space and time dimensions
    This helps distinguish points at the same location but at different times
    """
    if len(points) < n_clusters:
        return None, None, None
    
    # Create a space-time feature matrix
    # Normalize spatial and temporal components
    scaler_space = StandardScaler()
    scaler_time = StandardScaler()
    
    points_scaled = scaler_space.fit_transform(points)
    time_scaled = scaler_time.fit_transform(frame_indices.reshape(-1, 1))
    
    # Combine space and time with appropriate weighting
    # Higher time_weight means more emphasis on temporal separation
    space_time_features = np.hstack([points_scaled, time_weight * time_scaled])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(space_time_features)
    
    # Calculate cluster centers, times, and sizes
    centers, times, sizes = get_cluster_centers_and_times(labels, points, frame_indices)
    
    return centers, times, sizes

def fit_translation_scaling(A, B):
    """
    Find the best translation and uniform scaling (no rotation) that maps A to B.
    Returns the transformed A and the RMS error.
    """
    # Handle edge cases
    if len(A) < 2 or len(B) < 2:
        return float('inf'), None
    
    # Center both trajectories
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Find optimal scaling
    norm_A = np.linalg.norm(AA)
    norm_B = np.linalg.norm(BB)

    if norm_A == 0 or norm_B == 0:
        scale = 1.0
    else:
        scale = norm_B / norm_A

    # Apply scaling and translation
    A2 = AA * scale + centroid_B

    # Compute RMS error
    error = np.sqrt(np.mean(np.sum((A2 - B)**2, axis=1)))
    return error, A2

def get_cluster_centers_and_times(labels, points, frame_indices=None):
    """Get cluster centers, times, and sizes"""
    clusters = np.unique(labels)
    
    # Calculate mean position and time for each cluster
    centers = []
    times = []
    sizes = []
    
    # If frame indices not provided, use position in array
    if frame_indices is None:
        frame_indices = np.arange(len(labels))
    
    for c in clusters:
        idxs = np.where(labels == c)[0]
        centers.append(np.mean(points[idxs], axis=0))
        times.append(np.mean(frame_indices[idxs]))
        sizes.append(len(idxs))
    
    return np.array(centers), np.array(times), np.array(sizes)

def plot_trajectory_on_pinpad(centers_ordered, top_pins, out_path, title):
    """Plot the fitted trajectory on the PIN pad"""
    plt.figure(figsize=(10, 8))

    # Draw PIN pad
    for i, (x, y) in enumerate(PINPAD_COORDS):
        plt.scatter(x, y, s=200, c='lightgray', edgecolor='black', zorder=1)
        plt.annotate(PINPAD_DIGITS[i], xy=(x, y), fontsize=16, ha='center', va='center', zorder=2)

    # Plot top pin trajectory
    if top_pins and len(top_pins) > 0:
        top_pin = top_pins[0][0]
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in top_pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        plt.plot(pin_coords[:,0], pin_coords[:,1], 'b-', 
                linewidth=2, alpha=0.7, label=f"{top_pin} (ideal)")

    # Plot observed trajectory fit
    if len(centers_ordered) > 0 and len(top_pins) > 0:
        top_pin = top_pins[0][0]
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in top_pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        # Fit trajectory to the PIN pad path
        _, transformed_centers = fit_translation_scaling(centers_ordered, pin_coords)
        if transformed_centers is not None:
            # Plot the transformed trajectory
            plt.plot(transformed_centers[:,0], transformed_centers[:,1], 'r--', alpha=0.8, zorder=5,
                     label="Observed trajectory")
            plt.scatter(transformed_centers[:,0], transformed_centers[:,1], 
                        color='red', s=80, edgecolor='white', alpha=0.8, zorder=10)
            
            # Add digit sequence numbers
            for i, (x, y) in enumerate(transformed_centers):
                plt.annotate(f"{i+1}", xy=(x, y), xytext=(-5, 5), 
                            textcoords='offset points', fontsize=10, 
                            color='white', fontweight='bold')

    plt.title(title)
    plt.grid(False)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(out_path)
    plt.close()

# Function for jitter analysis
def detect_jitter_in_trajectory(points, frame_indices):
    """
    Detect jitter (stationary periods) in the trajectory.
    Returns masks for different detection methods and key jitter points.
    """
    # Calculate velocities
    velocities = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    # Initialize with empty arrays for safety
    not_moving_mask_kmeans = np.array([])
    jitter_start_points = []
    
    if len(velocities) > 0:
        velocities_reshape = velocities.reshape(-1, 1)
        
        # KMeans clustering (2 clusters)
        if len(velocities) > 1:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(velocities_reshape)
            labels_kmeans = kmeans.labels_
            cluster_means = [np.mean(velocities[labels_kmeans == i]) for i in range(2)]
            not_moving_label = np.argmin(cluster_means)
            not_moving_mask_kmeans = np.concatenate([[False], labels_kmeans == not_moving_label])
            
            # Find key jitter points (where velocity drops significantly)
            # These are transitions from moving to not moving
            for i in range(1, len(not_moving_mask_kmeans)):
                if not_moving_mask_kmeans[i] and not not_moving_mask_kmeans[i-1]:
                    jitter_start_points.append(i)
    
    return {
        'kmeans': not_moving_mask_kmeans,
        'velocities': np.concatenate([[0], velocities]),  # Add 0 for the first frame
        'jitter_points': jitter_start_points  # Key points where jitter starts
    }

# Improved function to plot trajectory with precise jitter points
def plot_trajectory_with_jitter(points, jitter_masks, out_path, title):
    """Plot the trajectory with exact jitter points highlighted"""
    plt.figure(figsize=(10, 8))
    
    # Calculate displacements from origin
    if len(points) > 0:
        initial = points[0]
        x_disp = points[:, 0] - initial[0]
        y_disp = points[:, 1] - initial[1]
        
        # Plot full trajectory
        plt.plot(x_disp, -y_disp, color='gray', linewidth=1.5, label='Trajectory')
        
        # Extract key jitter points (transition points where velocity drops)
        jitter_points = jitter_masks.get('jitter_points', [])
        
        # Plot key jitter points (start of stationary periods)
        if jitter_points:
            jitter_x = [x_disp[i] for i in jitter_points]
            jitter_y = [-y_disp[i] for i in jitter_points]
            
            # Plot large markers at jitter points
            plt.scatter(jitter_x, jitter_y, color='red', s=120, marker='*', 
                       edgecolor='black', linewidth=1, zorder=10,
                       label='Key Press Points')
            
            # Add labels with point numbers
            for i, (x, y, idx) in enumerate(zip(jitter_x, jitter_y, jitter_points)):
                plt.annotate(f"{i+1}", xy=(x, y), xytext=(10, 0),
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           color='red', backgroundcolor='white',
                           bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="red", alpha=0.8),
                           zorder=11)
    
    plt.title(title)
    plt.xlabel('X Displacement (pixels)')
    plt.ylabel('Y Displacement (pixels, up is positive)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Improved function to plot velocity profile with precise key press points
def plot_velocity_profile_with_keypresses(points, jitter_masks, out_path, title):
    """Plot the velocity profile with precise key press points"""
    plt.figure(figsize=(12, 6))
    
    # Plot velocity profile
    velocities = jitter_masks['velocities']
    plt.plot(velocities, 'b-', linewidth=1.5, alpha=0.7, label='Velocity')
    
    # Extract key jitter points (transition points where velocity drops)
    jitter_points = jitter_masks.get('jitter_points', [])
    
    # Plot key press points
    if jitter_points:
        jitter_v = [velocities[i] for i in jitter_points]
        
        # Plot stars at key press points
        plt.scatter(jitter_points, jitter_v, color='red', s=120, marker='*', 
                   edgecolor='black', linewidth=1, zorder=10,
                   label='Key Press Points')
        
        # Add vertical lines at key press points
        for i, idx in enumerate(jitter_points):
            plt.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
            
            # Add labels with point numbers
            plt.annotate(f"{i+1}", xy=(idx, jitter_v[i]), xytext=(0, 15),
                       textcoords='offset points', fontsize=12, fontweight='bold',
                       color='red', ha='center',
                       bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="red", alpha=0.8),
                       zorder=11)
    
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Velocity (pixels/frame)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def generate_individual_html_report(video_name, pin_scores, is_same_digit, report_dir):
    """Generate an HTML report for a single video"""
    report_path = os.path.join(report_dir, f"{video_name}_report.html")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PIN Analysis for {video_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pin-rank {{
            font-weight: bold;
            color: #e74c3c;
            width: 60px;
            text-align: center;
        }}
        .pin-code {{
            font-family: monospace;
            font-size: 1.2em;
            font-weight: bold;
            width: 100px;
        }}
        .pin-score {{
            color: #7f8c8d;
            width: 100px;
        }}
        .trajectory-image {{
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #95a5a6;
            font-style: italic;
            text-align: right;
            margin-top: 50px;
        }}
        .back-link {{
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        .back-link a {{
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
        }}
        .back-link a:hover {{
            text-decoration: underline;
        }}
        .pin-table-container {{
            max-height: 600px;
            overflow-y: auto;
            margin-bottom: 30px;
        }}
        .same-digit {{
            background-color: #ffe8e8;
        }}
        .repeated-digit {{
            background-color: #f2e8f7;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .image-item {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="back-link">
        <a href="index.html">← Back to main report</a>
    </div>

    <h1>PIN Analysis for Video: {video_name}</h1>
    <p>This report shows detailed PIN candidates based on trajectory analysis.</p>
    <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Add PIN candidates table
    html += """
    <h2>PIN Candidates</h2>
    <div class="pin-table-container">
    <table>
        <tr>
            <th class="pin-rank">Rank</th>
            <th class="pin-code">PIN</th>
            <th class="pin-score">Score</th>
            <th>Description</th>
        </tr>
"""
    
    # Add rows for each PIN candidate with descriptions
    for rank, (pin, score) in enumerate(pin_scores, 1):
        # Generate a description based on the PIN and score
        description = ""
        row_class = ""
        
        # Detect patterns in the PIN (for highlighting purposes only)
        if len(set(pin)) == 1:
            description = "Same digit repeated four times"
            row_class = "same-digit"
        elif pin[0] == pin[2] and pin[1] == pin[3]:
            description = "Alternating pattern (XYXY)"
        elif pin == pin[::-1]:
            description = "Palindrome pattern"
        elif pin.count(pin[0]) > 1 or pin.count(pin[1]) > 1 or pin.count(pin[2]) > 1:
            description = "Contains repeated digits"
            row_class = "repeated-digit"
        else:
            description = "Standard trajectory match"
            
        html += f"""
        <tr class="{row_class}">
            <td class="pin-rank">{rank}</td>
            <td class="pin-code">{pin}</td>
            <td class="pin-score">{score:.4f}</td>
            <td>{description}</td>
        </tr>"""
    
    html += """
    </table>
    </div>
"""
    
    # Add trajectory visualizations in a grid
    html += """
    <h2>Trajectory Visualizations</h2>
    <div class="image-grid">
"""
    
    # Add standard trajectory plot if it exists
    trajectory_img_path = os.path.join(OUTPUT_DIR, video_name, 'trajectory_mapping.png')
    if os.path.exists(trajectory_img_path):
        # Copy the image to report folder for portability
        report_img_path = os.path.join(report_dir, f"{video_name}_trajectory.png")
        shutil.copy(trajectory_img_path, report_img_path)
        html += f"""
        <div class="image-item">
            <img class="trajectory-image" src="{os.path.basename(report_img_path)}" alt="Trajectory plot">
            <p>Trajectory match with top PIN candidate</p>
        </div>
"""
    
    # Add jitter trajectory plot if it exists
    jitter_img_path = os.path.join(OUTPUT_DIR, video_name, 'trajectory_with_jitter.png')
    if os.path.exists(jitter_img_path):
        # Copy the image to report folder for portability
        report_jitter_path = os.path.join(report_dir, f"{video_name}_jitter.png")
        shutil.copy(jitter_img_path, report_jitter_path)
        html += f"""
        <div class="image-item">
            <img class="trajectory-image" src="{os.path.basename(report_jitter_path)}" alt="Jitter detection">
            <p>Trajectory with detected key press locations</p>
        </div>
"""

    # Add velocity profile plot if it exists
    velocity_img_path = os.path.join(OUTPUT_DIR, video_name, 'velocity_profile.png')
    if os.path.exists(velocity_img_path):
        # Copy the image to report folder for portability
        report_velocity_path = os.path.join(report_dir, f"{video_name}_velocity.png")
        shutil.copy(velocity_img_path, report_velocity_path)
        html += f"""
        <div class="image-item">
            <img class="trajectory-image" src="{os.path.basename(report_velocity_path)}" alt="Velocity profile">
            <p>Velocity profile with key press points</p>
        </div>
"""
    
    html += """
    </div>
    <p>The numbered points in the trajectory indicate detected key presses where the velocity suddenly dropped, indicating the exact moments of pressing each digit.</p>
"""
    
    # Close HTML document
    html += """
    <div class="back-link">
        <a href="index.html">← Back to main report</a>
    </div>

    <div class="timestamp">
        <p>Analysis powered by Trajectory-Based PIN Detection</p>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html)
    
    return os.path.basename(report_path)

def generate_main_html_report(results, pattern_info, report_dir, video_reports):
    """Generate the main HTML report summarizing all videos"""
    report_path = os.path.join(report_dir, 'index.html')
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PIN Trajectory Analysis - Summary Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .video-section {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .video-name {
            margin-top: 0;
            color: #2c3e50;
        }
        .pin-rank {
            font-weight: bold;
            color: #e74c3c;
            width: 60px;
            text-align: center;
        }
        .pin-code {
            font-family: monospace;
            font-size: 1.2em;
            font-weight: bold;
            width: 100px;
        }
        .pin-score {
            color: #7f8c8d;
            width: 100px;
        }
        .timestamp {
            color: #95a5a6;
            font-style: italic;
            text-align: right;
            margin-top: 50px;
        }
        .details-link {
            margin-top: 10px;
        }
        .details-link a {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 8px 15px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
        }
        .details-link a:hover {
            background-color: #2980b9;
        }
        .top-pins {
            font-family: monospace;
            color: #555;
        }
        .same-digit {
            background-color: #ffe8e8;
        }
        .config-info {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .config-param {
            font-family: monospace;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>PIN Trajectory Analysis - Summary Report</h1>
    <p>This report summarizes the PIN analysis results across all videos.</p>
    <p>Generated on: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    
    <div class="config-info">
        <h3>Configuration Parameters</h3>
        <p><span class="config-param">TIME_WEIGHT</span>: """ + str(TIME_WEIGHT) + """ - Controls emphasis on time vs. space in clustering</p>
        <p><span class="config-param">SAME_DIGIT_BOX_SIZE</span>: """ + str(SAME_DIGIT_BOX_SIZE) + """ pixels - For same-digit pattern detection</p>
    </div>
    
    <h2>Videos Analyzed</h2>
    <table>
        <tr>
            <th>Video</th>
            <th>Top PIN</th>
            <th>Score</th>
            <th>Top 5 Candidates</th>
            <th>Action</th>
        </tr>
"""
    
    # Add rows for each video with links to detailed reports
    for video_name, pin_scores in results.items():
        if pin_scores:
            top_pin = pin_scores[0][0]
            top_score = pin_scores[0][1]
            report_filename = video_reports.get(video_name, "")
            
            # Get top 5 PINs for summary
            top_5_pins = ", ".join([ps[0] for ps in pin_scores[:5]])
            
            # Add special highlighting for repeated digits (visual only)
            row_class = ""
            if len(set(top_pin)) == 1:  # All same digit
                row_class = "class=\"same-digit\""
            
            html += f"""
        <tr {row_class}>
            <td>{video_name}</td>
            <td class="pin-code">{top_pin}</td>
            <td class="pin-score">{top_score:.4f}</td>
            <td class="top-pins">{top_5_pins}</td>
            <td class="details-link"><a href="{report_filename}">View Details</a></td>
        </tr>"""
    
    html += """
    </table>
    
    <div class="timestamp">
        <p>Analysis powered by Time-Aware Trajectory PIN Detection</p>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"Main HTML report generated at: {report_path}")

def process_csv_trajectory(csv_path, report_dir):
    """Process a CSV file using trajectory matching with time-aware clustering"""
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\nProcessing video: {video_name}")
    
    # Step 1: Load and clean data
    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    xs = df[xcol].values
    ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    points = np.stack([xs, ys], axis=1)
    frame_indices = np.arange(len(df))[mask]
    
    if points.shape[0] == 0:
        print(f"Warning: No valid points in {csv_path}. Skipping.")
        return [], False
    
    # Step 2: Speed filtering - keep only slow points (potential keystrokes)
    speeds = calculate_speeds(points)
    filtered_points, filtered_frames = filter_by_speed(points, speeds, frame_indices)
    
    print(f"  Filtered {len(points)} points to {len(filtered_points)} slow points")
    
    # Step 2.5: Detect jitter in trajectory (periods of minimal movement)
    jitter_masks = detect_jitter_in_trajectory(points, frame_indices)
    jitter_points = jitter_masks.get('jitter_points', [])
    print(f"  Detected {len(jitter_points)} key press events")
    
    # Step 2.6: Generate jitter visualization plots
    if len(points) > 0:
        # Plot trajectory with jitter highlighted
        plot_trajectory_with_jitter(
            points, jitter_masks,
            os.path.join(video_dir, 'trajectory_with_jitter.png'),
            f'Trajectory with Key Press Points: {video_name}'
        )
        
        # Plot velocity profile with key press points
        plot_velocity_profile_with_keypresses(
            points, jitter_masks,
            os.path.join(video_dir, 'velocity_profile.png'),
            f'Velocity Profile with Key Press Points: {video_name}'
        )
    
    # Initialize variables
    best_centers_ordered = None
    
    # Step 3: Check for all-same-digit PIN pattern (for information only)
    is_same_digit = are_all_points_close(filtered_points)  
    
    if is_same_digit:
        print(f"  DETECTED: All points are within a {SAME_DIGIT_BOX_SIZE}x{SAME_DIGIT_BOX_SIZE} pixel box - likely same-digit PIN")
        print(f"  Note: Using trajectory matching only for scoring")
        
        # We'll just use the mean point as center for visualization in this case
        best_centers_ordered = np.mean(filtered_points, axis=0).reshape(1, 2)
    
    # Step 4: Try time-aware clustering with different k values
    if not is_same_digit or best_centers_ordered is None:
        # Try time-aware clustering first with PIN_LENGTH clusters (typically 4)
        centers, times, sizes = time_aware_clustering(filtered_points, filtered_frames, n_clusters=PIN_LENGTH)
        
        if centers is not None and len(centers) == PIN_LENGTH:
            print(f"  Using time-aware clustering: Found {len(centers)} centers")
            # Sort by time
            order = np.argsort(times)
            centers_ordered = centers[order]
            best_centers_ordered = centers_ordered
        else:
            # Fall back to standard clustering methods if time-aware clustering doesn't work well
            print("  Falling back to standard clustering")
            for k in [4, 3, 2]:
                if k > len(filtered_points) // 5:
                    continue  # Skip if too few points
                
                scaler = StandardScaler()
                filtered_points_scaled = scaler.fit_transform(filtered_points)
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(filtered_points_scaled)
                
                # Get cluster centers, times, and sizes
                centers, times, sizes = get_cluster_centers_and_times(labels, filtered_points, filtered_frames)
                order = np.argsort(times)
                centers_ordered = centers[order]
                
                print(f"  Using standard k={k}: Found {len(centers_ordered)} centers")
                
                # Store best centers for visualization
                if best_centers_ordered is None:
                    best_centers_ordered = centers_ordered
    
    # If we still don't have centers, skip this video
    if best_centers_ordered is None:
        print(f"  WARNING: Could not find valid clustering for {video_name}")
        return [], is_same_digit
    
    # Step 5: Perform trajectory matching for all possible PINs
    print("  Generating PIN candidates using trajectory matching...")
    
    # Trajectory matching for all PINs
    pin_scores = []
    
    # Process all possible 4-digit PINs
    for pin in [''.join(p) for p in product(PINPAD_DIGITS, repeat=PIN_LENGTH)]:        
        try:
            pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
            pin_coords = PINPAD_COORDS[pin_indices]
            
            # Calculate trajectory matching score
            error, _ = fit_translation_scaling(best_centers_ordered, pin_coords)
            pin_scores.append((pin, error))
            
        except Exception as e:
            print(f"  Error scoring PIN {pin}: {e}")
    
    # Sort trajectory scores
    pin_scores.sort(key=lambda x: x[1])
    
    # Step 6: Apply dynamic score-based cutoff
    if pin_scores:
        # Get the best score and statistics
        best_score = pin_scores[0][1]
        all_scores = np.array([score for _, score in pin_scores])
        
        # Calculate statistical properties of scores
        score_mean = np.mean(all_scores)
        score_std = np.std(all_scores)
        
        # Use multiple threshold approaches
        # 1. Significant jump detection
        jump_idx = len(pin_scores)
        last_score = best_score
        score_range = max(0.1, best_score)  # Baseline for calculating jumps
        
        for i, (_, score) in enumerate(pin_scores[1:], 1):
            # If score jumps by more than 20% of the best score or 0.05 absolute
            jump_threshold = max(0.05, score_range * 0.20)  # defualt 0.05   0.20
            if score - last_score > jump_threshold:
                jump_idx = i
                break
            last_score = score
        
        # 2. Absolute threshold - scores within reasonable range of best
        absolute_threshold = best_score + min(0.5, best_score)
        absolute_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > absolute_threshold), len(pin_scores))
        
        # 3. Statistical threshold - scores within statistical bounds
        stat_threshold = best_score + 2 * score_std if score_std > 0 else best_score * 2
        stat_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > stat_threshold), len(pin_scores))
        
        # 4. Hard minimum threshold - always show at least 10 candidates
        min_candidates = 1000
        
        # Take the minimum of all approaches, but ensure minimum count
        cutoff_idx = max(min_candidates, min(jump_idx, absolute_idx, stat_idx))
        
        # Apply the cutoff
        pin_scores = pin_scores[:cutoff_idx]
        print(f"  Using dynamic score-based cutoff: Keeping {len(pin_scores)} candidates")
    
    # Step 7: Display top results
    print("\nTop PIN candidates (trajectory matching only):")
    for pin, score in pin_scores[:min(10, len(pin_scores))]:
        print(f"  {pin}: {score:.4f}")
    
    # Step 8: Plot trajectory on PIN pad for top candidates
    if best_centers_ordered is not None and len(best_centers_ordered) > 0:
        title = 'Time-Aware Trajectory Matching'
            
        plot_trajectory_on_pinpad(
            best_centers_ordered, pin_scores[:5] if pin_scores else [],
            os.path.join(video_dir, 'trajectory_mapping.png'),
            f'{title}: Top PIN {pin_scores[0][0] if pin_scores else "N/A"}'
        )
    
    return pin_scores, is_same_digit

def main():
    """Main function to process all CSV files"""
    report_dir = os.path.join('.', REPORT_FOLDER)
    os.makedirs(report_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))
    total = len(csv_files)
    print(f"Found {total} *_ring_center.csv files.")
    
    results = {}
    pattern_info = {}
    video_reports = {}
    
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"Processing {idx}/{total}: {csv_path}")
        video_name = os.path.basename(os.path.dirname(csv_path))
        try:
            pin_scores, is_same_digit = process_csv_trajectory(csv_path, report_dir)
            if pin_scores:
                results[video_name] = pin_scores
                pattern_info[video_name] = is_same_digit
                # Generate individual HTML report for this video
                report_filename = generate_individual_html_report(video_name, pin_scores, is_same_digit, report_dir)
                video_reports[video_name] = report_filename
        except Exception as e:
            print(f"Error processing file {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate main HTML report with links to individual reports
    generate_main_html_report(results, pattern_info, report_dir, video_reports)
    
    # Open the HTML report in the default browser
    main_report_path = os.path.join(report_dir, 'index.html')
    print(f"Opening HTML report in default browser: {main_report_path}")
    webbrowser.open('file://' + os.path.abspath(main_report_path))

if __name__ == '__main__':
    main()
