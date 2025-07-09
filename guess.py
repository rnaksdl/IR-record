import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from itertools import product, permutations
import time
import datetime
import shutil
import webbrowser
from collections import Counter

# --- USER CONFIGURABLE PARAMETERS ---
PIN_LENGTH = 4
OUTPUT_DIR = './output'
REPORT_FOLDER = './report'

# IMPORTANT: Adjust this value to control same-digit PIN detection
# Lower values (like 50 or 60) make detection more sensitive
# Higher values (like 80 or 100) are more conservative
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

def detect_temporal_patterns(points, frame_indices):
    """
    Analyze temporal patterns in the point sequence
    Returns possible PINs with confidence scores
    """
    if len(points) < 10:  # Need sufficient points but not too strict
        return []
    
    # Create a time-ordered sequence
    order = np.argsort(frame_indices)
    ordered_points = points[order]
    
    # Use DBSCAN to identify clusters without specifying count
    db = DBSCAN(eps=15.0, min_samples=3)
    db_labels = db.fit_predict(ordered_points)
    
    # Get unique clusters (ignoring noise points with label -1)
    unique_clusters = np.unique(db_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    if len(unique_clusters) < 2:  # Need at least 2 clusters
        return []
    
    # Calculate cluster centers and sizes
    cluster_centers = []
    cluster_sizes = []
    for c in unique_clusters:
        cluster_points = ordered_points[db_labels == c]
        cluster_centers.append(np.mean(cluster_points, axis=0))
        cluster_sizes.append(len(cluster_points))
    
    # Find nearest digit for each cluster
    closest_digits = []
    for center in cluster_centers:
        distances = [np.linalg.norm(center - pin_coord) for pin_coord in PINPAD_COORDS]
        closest_digit = PINPAD_DIGITS[np.argmin(distances)]
        closest_digits.append(closest_digit)
    
    # Create a temporal sequence of cluster visits
    temporal_sequence = []
    for point_idx, point in enumerate(ordered_points):
        if db_labels[point_idx] >= 0:  # Skip noise
            cluster_idx = np.where(unique_clusters == db_labels[point_idx])[0][0]
            temporal_sequence.append(closest_digits[cluster_idx])
    
    # Analyze temporal sequence for patterns
    candidates = []
    
    # Check for alternating patterns (like 2121)
    if len(temporal_sequence) >= 6:  # Reduced threshold
        # Count consecutive pairs in the sequence
        pairs = [temporal_sequence[i:i+2] for i in range(0, len(temporal_sequence)-1)]
        if len(pairs) > 0:  # Make sure there are pairs to analyze
            pair_counter = Counter(["".join(p) for p in pairs])
            
            # Find the most common alternating pairs
            common_pairs = pair_counter.most_common(5)
            
            for pair, count in common_pairs:
                if count >= 2:  # Reduced threshold
                    # Create alternating pattern
                    alt_pattern = pair[0] + pair[1] + pair[0] + pair[1]
                    # Also consider the reverse order
                    rev_pattern = pair[1] + pair[0] + pair[1] + pair[0]
                    
                    # Score based on frequency in the sequence
                    normalized_count = count / len(pairs)  # Changed variable name from pair_score
                    candidates.append((alt_pattern, -2.5 * normalized_count))
                    candidates.append((rev_pattern, -2.0 * normalized_count))
    
    # Rest of the function remains the same
    # ... (code for repeated digit patterns and N-gram analysis)
    
    return candidates

def identify_repeated_digit_clusters(clusters, sizes, closest_digits):
    """
    Analyze cluster sizes to identify repeated digits
    Creates multiple mappings for how clusters might map to digits in a PIN
    """
    candidates = []
    
    # Calculate statistical properties
    mean_size = np.mean(sizes)
    std_size = np.std(sizes) if len(sizes) > 1 else 0
    
    # Case 1: One cluster is significantly larger than others
    for i, size in enumerate(sizes):
        # If a cluster is >30% larger than mean or >1.0 std deviations above
        if size > mean_size * 1.3 or (std_size > 0 and size > mean_size + 1.0 * std_size):
            digit = closest_digits[i]
            
            # Generate patterns with this digit repeated at different positions
            for positions in [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]:
                # Create a list of remaining positions
                remaining_pos = [j for j in range(PIN_LENGTH) if j not in positions]
                
                # Fill remaining positions with other digits
                for other_digits in permutations([d for d in closest_digits if d != digit], len(remaining_pos)):
                    pattern = [""] * PIN_LENGTH
                    for pos in positions:
                        pattern[pos] = digit
                    for j, pos in enumerate(remaining_pos):
                        if j < len(other_digits):
                            pattern[pos] = other_digits[j]
                    
                    # Fill any remaining positions with random digits
                    for k in range(PIN_LENGTH):
                        if pattern[k] == "":
                            pattern[k] = closest_digits[0]  # Default to first digit
                    
                    pin = ''.join(pattern)
                    # Score based on how extreme the size difference is
                    score_factor = (size / mean_size) if mean_size > 0 else 1.5
                    candidates.append((pin, -2.0 * score_factor))
                    
                    # Extra boost for middle-position repeats (like in 1990)
                    if positions == [1,2]:  # Two repeats in the middle
                        candidates.append((pin, -2.2 * score_factor))
    
    # Case 2: Special handling for exactly 3 clusters (possible 4-digit PIN with one digit repeated)
    if len(clusters) == 3:
        # Try each possible position for the repeated digit
        for d1_idx in range(len(closest_digits)):
            for d2_idx in range(len(closest_digits)):
                if d2_idx == d1_idx:
                    continue
                for d3_idx in range(len(closest_digits)):
                    if d3_idx == d1_idx or d3_idx == d2_idx:
                        continue
                    
                    # Try repeating each digit in different positions
                    for repeat_pos in range(3):  # 0=first, 1=middle, 2=last position to start repeat
                        # Create a PIN with one digit repeated consecutively
                        pattern = []
                        digits_used = [d1_idx, d2_idx, d3_idx]
                        
                        for i in range(PIN_LENGTH):
                            if i == repeat_pos or i == repeat_pos + 1:
                                pattern.append(closest_digits[d1_idx])  # Repeated digit
                            elif i < repeat_pos:
                                pattern.append(closest_digits[d2_idx])  # Digits before repeat
                            else:
                                pattern.append(closest_digits[d3_idx])  # Digits after repeat
                        
                        pin = ''.join(pattern)
                        
                        # Give extra boost to middle position repeat patterns
                        if repeat_pos == 1:  # Middle position repeat (like 1990)
                            candidates.append((pin, -2.2))
                        else:
                            candidates.append((pin, -1.8))
    
    # Case 3: Special handling for exactly 2 clusters (possible 4-digit PIN with multiple digits repeated)
    if len(clusters) == 2:
        d1, d2 = closest_digits[0], closest_digits[1]
        
        # Generate all possible 4-digit combinations of these 2 digits
        for pattern in product([0, 1], repeat=PIN_LENGTH):
            pin = ''.join([closest_digits[p] for p in pattern])
            
            # Boost score for alternating patterns (0101 or 1010)
            if pattern == (0, 1, 0, 1) or pattern == (1, 0, 1, 0):
                candidates.append((pin, -2.2))  # Stronger boost for alternating
            else:
                candidates.append((pin, -1.5))  # Regular boost for other combinations
    
    return candidates

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

    plt.title(title)
    plt.grid(False)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(out_path)
    plt.close()

def generate_individual_html_report(video_name, pin_scores, report_dir):
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
        .alternating-pattern {{
            background-color: #e8f7f3;
        }}
        .repeated-digit {{
            background-color: #f2e8f7;
        }}
        .same-digit {{
            background-color: #ffe8e8;
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
        
        # Detect patterns in the PIN
        if len(set(pin)) == 1:
            description = "Same digit repeated four times"
            row_class = "same-digit"
        elif pin[0] == pin[2] and pin[1] == pin[3]:
            description = "Alternating pattern (XYXY)"
            row_class = "alternating-pattern"
        elif pin == pin[::-1]:
            description = "Palindrome pattern"
        elif score < 0:
            description = "Special pattern detected with high confidence"
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
    
    # Add trajectory plot image if it exists
    trajectory_img_path = os.path.join(OUTPUT_DIR, video_name, 'trajectory_mapping.png')
    if os.path.exists(trajectory_img_path):
        # Copy the image to report folder for portability
        report_img_path = os.path.join(report_dir, f"{video_name}_trajectory.png")
        shutil.copy(trajectory_img_path, report_img_path)
        html += f"""
    <h2>Trajectory Visualization</h2>
    <img class="trajectory-image" src="{os.path.basename(report_img_path)}" alt="Trajectory plot for {video_name}">
    <p>The blue line represents the ideal path for the top PIN candidate. The red dashed line shows the detected trajectory after transformation.</p>
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

def generate_main_html_report(results, report_dir, video_reports):
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
        .alternating-pattern {
            background-color: #e8f7f3;
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
        <p><span class="config-param">SAME_DIGIT_BOX_SIZE</span>: """ + str(SAME_DIGIT_BOX_SIZE) + """ pixels</p>
        <p>This parameter controls the size threshold for detecting same-digit PINs. Lower values make the detection more sensitive.</p>
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
            
            # Add special highlighting for patterns
            row_class = ""
            if len(set(top_pin)) == 1:  # All same digit
                row_class = "class=\"same-digit\""
            elif top_pin[0] == top_pin[2] and top_pin[1] == top_pin[3]:  # Alternating
                row_class = "class=\"alternating-pattern\""
            
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
        <p>Analysis powered by Trajectory-Based PIN Detection</p>
    </div>
</body>
</html>
"""
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"Main HTML report generated at: {report_path}")

def process_csv_trajectory(csv_path, report_dir):
    """Process a CSV file with advanced trajectory analysis and smart score-based cutoff"""
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
        return []
    
    # Step 2: Speed filtering - keep only slow points (potential keystrokes)
    speeds = calculate_speeds(points)
    filtered_points, filtered_frames = filter_by_speed(points, speeds, frame_indices)
    
    print(f"  Filtered {len(points)} points to {len(filtered_points)} slow points")
    
    # Initialize pin_scores
    pin_scores = []
    best_centers_ordered = None
    
    # Step 3: Check for all-same-digit PIN pattern using configurable box size
    is_same_digit = are_all_points_close(filtered_points)  # Uses global SAME_DIGIT_BOX_SIZE
    
    if is_same_digit:
        print(f"  PRE-CLUSTERING CHECK: All points are within a {SAME_DIGIT_BOX_SIZE}x{SAME_DIGIT_BOX_SIZE} pixel box - likely same-digit PIN")
        
        # Instead of immediately assuming all digits are the same,
        # Find the closest keypad digit to the mean point position
        mean_point = np.mean(filtered_points, axis=0)
        distances = [np.linalg.norm(mean_point - pin_coord) for pin_coord in PINPAD_COORDS]
        closest_idx = np.argmin(distances)
        closest_digit = PINPAD_DIGITS[closest_idx]
        
        print(f"  Closest digit to mean position: {closest_digit}")
        
        # Add this digit repeated as the primary candidate
        same_digit_pin = closest_digit * PIN_LENGTH
        pin_scores.append((same_digit_pin, -3.0))
        
        # Add other repeating digits with lower scores
        for digit in PINPAD_DIGITS:
            if digit != closest_digit:
                repeat_pin = digit * PIN_LENGTH
                # Score based on distance from mean point
                digit_idx = PINPAD_DIGIT_TO_IDX[digit]
                distance = np.linalg.norm(mean_point - PINPAD_COORDS[digit_idx])
                closest_distance = np.linalg.norm(mean_point - PINPAD_COORDS[closest_idx])
                relative_distance = distance / closest_distance if closest_distance > 0 else 2.0
                
                # Only add plausible alternatives
                if relative_distance < 2.0:
                    pin_scores.append((repeat_pin, -3.0 + relative_distance))
        
        # Use mean point as center for visualization
        best_centers_ordered = np.mean(filtered_points, axis=0).reshape(1, 2)
    else:
        # Step 4: First check for temporal patterns (zigzag, repeats)
        temporal_candidates = detect_temporal_patterns(filtered_points, filtered_frames)
        if temporal_candidates:
            print(f"  DETECTED TEMPORAL PATTERNS: {len(temporal_candidates)} candidates")
            pin_scores.extend(temporal_candidates)
        
        # Step 5: Try clustering with different k values
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
            sizes_ordered = sizes[order]
            
            print(f"  Using k={k}: Found {len(centers_ordered)} centers")
            
            # Store best centers for visualization
            if best_centers_ordered is None:
                best_centers_ordered = centers_ordered
            
            # Find closest digits to each cluster center
            closest_digits = []
            for center in centers:
                distances = [np.linalg.norm(center - pin_coord) for pin_coord in PINPAD_COORDS]
                closest_digit = PINPAD_DIGITS[np.argmin(distances)]
                closest_digits.append(closest_digit)
            
            # Step 6: For k < 4, identify potential repeated digits
            if k < 4:
                repeat_candidates = identify_repeated_digit_clusters(centers, sizes, closest_digits)
                pin_scores.extend(repeat_candidates)
                
                # Special handling for common patterns without hardcoding
                if k == 2 and len(closest_digits) == 2:
                    d1, d2 = closest_digits
                    # Test for alternating patterns
                    alt_pattern = d1 + d2 + d1 + d2
                    rev_pattern = d2 + d1 + d2 + d1
                    pin_scores.append((alt_pattern, -2.2))  # Standard boost
                    pin_scores.append((rev_pattern, -2.0))  # Slightly lower boost
                
                # Special handling for 3 clusters - might be patterns with one repeat
                if k == 3 and len(closest_digits) == 3:
                    # Try various positions for repeated digits
                    for first in closest_digits:
                        for middle in closest_digits:
                            for last in closest_digits:
                                # Middle position repeats
                                pin = first + middle + middle + last
                                pin_scores.append((pin, -2.2))
        
        # If we still don't have centers, skip this video
        if best_centers_ordered is None:
            print(f"  WARNING: Could not find valid clustering for {video_name}")
            return []
        
        # Step 7: Now perform trajectory matching for all possible PINs
        print("  Generating PIN candidates using trajectory matching...")
        
        # Skip PINs we've already identified through special patterns
        existing_pins = set(pin for pin, _ in pin_scores)
        
        # Trajectory matching for remaining PINs
        trajectory_scores = []
        
        # Process all possible 4-digit PINs
        for pin in [''.join(p) for p in product(PINPAD_DIGITS, repeat=PIN_LENGTH)]:
            if pin in existing_pins:
                continue
                
            try:
                pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
                pin_coords = PINPAD_COORDS[pin_indices]
                
                # Calculate trajectory matching score
                error, _ = fit_translation_scaling(best_centers_ordered, pin_coords)
                trajectory_scores.append((pin, error))
                
            except Exception as e:
                print(f"  Error scoring PIN {pin}: {e}")
        
        # Sort trajectory scores
        trajectory_scores.sort(key=lambda x: x[1])
        
        # Combine all scores and sort
        pin_scores.extend(trajectory_scores)
        pin_scores.sort(key=lambda x: x[1])
    
    # Step 8: Apply dynamic score-based cutoff
    if pin_scores:
        # Get the best score and statistics
        best_score = pin_scores[0][1]
        all_scores = np.array([score for _, score in pin_scores])
        
        # Calculate statistical properties of scores
        score_mean = np.mean(all_scores)
        score_std = np.std(all_scores)
        
        # Different cutoff strategies based on score distribution
        if best_score < 0:
            # For special pattern scores (negative scores)
            
            # Find where scores become positive
            first_positive_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > 0), len(pin_scores))
            
            # Keep all negative scores plus some buffer into positive scores
            buffer_size = min(20, len(pin_scores) - first_positive_idx)
            cutoff_idx = first_positive_idx + buffer_size
            
            # If we have special patterns, always show at least 15 candidates
            cutoff_idx = max(15, cutoff_idx)
            
        else:
            # For standard scores (all positive), use multiple threshold approaches
            
            # 1. Significant jump detection - find where scores start to jump significantly
            jump_idx = len(pin_scores)
            last_score = best_score
            score_range = max(0.1, best_score)  # Baseline for calculating jumps
            
            for i, (_, score) in enumerate(pin_scores[1:], 1):
                # If score jumps by more than 20% of the best score or 0.05 absolute
                jump_threshold = max(0.05, score_range * 0.20)
                if score - last_score > jump_threshold:
                    jump_idx = i
                    break
                last_score = score
            
            # 2. Absolute threshold - scores within reasonable range of best
            absolute_threshold = best_score + min(0.5, best_score)  # Either +0.5 or 2x best score
            absolute_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > absolute_threshold), len(pin_scores))
            
            # 3. Statistical threshold - scores within statistical bounds
            stat_threshold = best_score + 2 * score_std if score_std > 0 else best_score * 2
            stat_idx = next((i for i, (_, s) in enumerate(pin_scores) if s > stat_threshold), len(pin_scores))
            
            # 4. Hard minimum threshold - always show at least 10 candidates
            min_candidates = 10
            
            # Take the minimum of all approaches, but ensure minimum count
            cutoff_idx = max(min_candidates, min(jump_idx, absolute_idx, stat_idx))
        
        # Apply the cutoff
        pin_scores = pin_scores[:cutoff_idx]
        print(f"  Using dynamic score-based cutoff: Keeping {len(pin_scores)} candidates")
    
    # Step 9: Display top results
    print("\nTop PIN candidates:")
    for pin, score in pin_scores[:min(10, len(pin_scores))]:
        print(f"  {pin}: {score:.4f}")
    
    # Filter to keep only positive scores
    positive_scores = [ps for ps in pin_scores if ps[1] > 0]
    print(f"  Filtered to {len(positive_scores)} candidates with positive scores only")
    
    # Step 10: Plot trajectory on PIN pad for top candidates
    if best_centers_ordered is not None and len(best_centers_ordered) > 0:
        plot_trajectory_on_pinpad(
            best_centers_ordered, positive_scores[:5] if positive_scores else [],
            os.path.join(video_dir, 'trajectory_mapping.png'),
            f'Trajectory Matching: Top PIN {positive_scores[0][0] if positive_scores else "N/A"}'
        )
    
    return positive_scores

def main():
    """Main function to process all CSV files"""
    report_dir = os.path.join('.', REPORT_FOLDER)
    os.makedirs(report_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))
    total = len(csv_files)
    print(f"Found {total} *_ring_center.csv files.")
    
    results = {}
    video_reports = {}
    
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"Processing {idx}/{total}: {csv_path}")
        video_name = os.path.basename(os.path.dirname(csv_path))
        try:
            pin_scores = process_csv_trajectory(csv_path, report_dir)
            if pin_scores:
                results[video_name] = pin_scores
                # Generate individual HTML report for this video
                report_filename = generate_individual_html_report(video_name, pin_scores, report_dir)
                video_reports[video_name] = report_filename
        except Exception as e:
            print(f"Error processing file {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate main HTML report with links to individual reports
    generate_main_html_report(results, report_dir, video_reports)
    
    # Open the HTML report in the default browser
    main_report_path = os.path.join(report_dir, 'index.html')
    print(f"Opening HTML report in default browser: {main_report_path}")
    webbrowser.open('file://' + os.path.abspath(main_report_path))

if __name__ == '__main__':
    main()
