import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from itertools import permutations, product, combinations
from collections import Counter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score

# --- CONFIGURATION ---
PIN_LENGTH = 4
ACTUAL_PIN = ''  # Set your actual PIN here
OUTPUT_DIR = './output'
REPORT_FOLDER = './report'
# Button dimensions
BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9

# Calculate center positions
X_OFFSET = BUTTON_WIDTH/2
Y_OFFSET = BUTTON_HEIGHT/2

# Create coordinate array with correct Y-axis (positive values = down)
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

# Common PIN patterns by type
COMMON_PATTERNS = {
    'rows': ['1234', '2345', '3456', '4567', '5678', '6789', '7890'],
    'columns': ['1470', '2580', '3690', '1478', '2589', '3690'],
    'diagonals': ['1590', '3570', '1397', '3971', '7531'],
    'repeated': ['1111', '2222', '3333', '4444', '5555', '6666', '7777', '8888', '9999', '0000']
}

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

# Calculate speeds between frames
def calculate_speeds(points):
    """Calculate speeds between consecutive points"""
    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    # Add a placeholder speed for the first point
    speeds = np.insert(speeds, 0, np.median(speeds))
    return speeds

# Filter points by speed
def filter_by_speed(points, speeds):
    """Filter points based on speed using KMeans clustering"""
    # Reshape speeds for KMeans
    speeds_2d = speeds.reshape(-1, 1)
    
    # Cluster speeds into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(speeds_2d)
    labels = kmeans.labels_
    
    # Sort clusters by mean speed (ascending)
    cluster_speeds = [np.mean(speeds[labels == i]) for i in range(3)]
    sorted_indices = np.argsort(cluster_speeds)
    
    # Take only the slowest cluster
    slowest_cluster = sorted_indices[0]
    slow_mask = (labels == slowest_cluster)
    
    return points[slow_mask], np.where(slow_mask)[0]

# NEW FUNCTION: Remove small clusters early (before elbow method)
def remove_small_clusters_early(points, min_ratio=0.15):
    """
    Remove points belonging to small clusters (< min_ratio * largest cluster size).
    Returns filtered points.
    """
    if len(points) < 5:  # Not enough points to meaningfully cluster
        return points
        
    # Use DBSCAN to find clusters
    dbscan = DBSCAN(eps=15, min_samples=2)
    labels = dbscan.fit_predict(points)
    
    # Count points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Skip noise points (label -1)
    if len(unique_labels) == 1 and unique_labels[0] == -1:
        print("  All points classified as noise, keeping all")
        return points
        
    # Find valid labels (not noise and above size threshold)
    valid_clusters = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]
    
    if len(valid_clusters) == 0:
        print("  No valid clusters found, keeping all points")
        return points
        
    max_size = np.max(valid_counts)
    size_threshold = min_ratio * max_size
    
    # Identify clusters to keep
    keep_labels = [label for label, count in zip(valid_clusters, valid_counts) 
                  if count >= size_threshold]
                  
    if len(keep_labels) == 0:
        print("  No clusters above threshold, keeping largest")
        keep_labels = [valid_clusters[np.argmax(valid_counts)]]
    
    # Create mask for points to keep
    keep_mask = np.isin(labels, keep_labels)
    
    # Report removed clusters
    removed_count = len(points) - np.sum(keep_mask)
    if removed_count > 0:
        print(f"  Early cluster filtering: removed {removed_count} points from small clusters")
        print(f"  Original cluster sizes: {[count for label, count in zip(unique_labels, counts) if label != -1]}")
        print(f"  Kept cluster sizes: {[count for label, count in zip(unique_labels, counts) if label in keep_labels]}")
    
    return points[keep_mask]

# Find optimal number of clusters using elbow method
def find_optimal_clusters(points, max_clusters=10):
    """Determine optimal number of clusters using elbow method"""
    if len(points) < max_clusters:
        max_clusters = len(points) // 2
        if max_clusters < 2:
            return 2  # Minimum clusters
    
    max_clusters = min(max_clusters, 8)  # Cap at 8 for practical PIN analysis
    
    inertias = []
    silhouettes = []
    
    # Calculate inertia and silhouette scores for different cluster counts
    for k in range(2, max_clusters + 1):
        if len(points) < k:
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(points)
        
        # Inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score (if more than one cluster with data)
        if len(np.unique(labels)) > 1 and len(points) > k:
            silhouette = silhouette_score(points, labels)
            silhouettes.append(silhouette)
        else:
            silhouettes.append(0)
    
    # Find elbow point in inertia curve
    optimal_k = 4  # Default to 4 for PIN length
    
    if len(inertias) > 2:
        # Calculate rate of change in inertia
        inertia_changes = np.diff(inertias)
        rate_changes = np.diff(inertia_changes)
        
        # Find the point where the rate of change stabilizes (elbow)
        elbow_candidates = np.argsort(np.abs(rate_changes))
        if len(elbow_candidates) > 0:
            optimal_k = elbow_candidates[0] + 3  # +3 because we started at k=2
        
        # If silhouette scores available, consider them too
        if silhouettes and max(silhouettes) > 0.5:
            silhouette_k = np.argmax(silhouettes) + 2  # +2 because we started at k=2
            
            # If silhouette strongly suggests a different k, use it
            if abs(silhouette_k - optimal_k) <= 1 or silhouettes[silhouette_k-2] > 0.7:
                optimal_k = silhouette_k
    
    # Cap optimal clusters at PIN_LENGTH
    return min(optimal_k, PIN_LENGTH)

# Plot elbow method results
def plot_elbow_method(points, max_clusters, out_path):
    """Plot elbow method results for finding optimal clusters"""
    if len(points) < max_clusters:
        max_clusters = len(points) // 2
        if max_clusters < 2:
            return 2  # Minimum clusters
    
    max_clusters = min(max_clusters, 8)  # Cap at 8 for practical PIN analysis
    
    inertias = []
    silhouettes = []
    
    # Calculate inertia and silhouette scores for different cluster counts
    for k in range(2, max_clusters + 1):
        if len(points) < k:
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(points)
        
        # Inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score (if more than one cluster with data)
        if len(np.unique(labels)) > 1 and len(points) > k:
            silhouette = silhouette_score(points, labels)
            silhouettes.append(silhouette)
        else:
            silhouettes.append(0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot inertia
    k_range = list(range(2, 2 + len(inertias)))
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    
    # Plot silhouette score
    ax2.plot(k_range, silhouettes, 'ro-')
    ax2.set_title('Silhouette Score Method')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def get_cluster_centers_and_times(labels, points, frame_indices=None):
    """Get cluster centers, times, and sizes"""
    clusters = np.unique(labels)
    if -1 in clusters:
        clusters = clusters[clusters != -1]
    
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
    
    return np.array(centers), np.array(times), np.array(sizes), clusters

# Check for minimal movement (potential repeated digits)
def is_minimal_movement(centers_ordered, threshold=0.3):
    """Check if centers are very close, suggesting repeated digits"""
    if len(centers_ordered) < 2:
        return False
    dists = [np.linalg.norm(a-b) for a, b in combinations(centers_ordered, 2)]
    return all(d < threshold for d in dists)

# Generate plausible mappings from clusters to digits
def plausible_cluster_digit_mappings(num_centers, pin_length):
    """Generate mappings from clusters to PIN digits, allowing for repeats"""
    return product(range(num_centers), repeat=pin_length)

# Shape-preserving trajectory fitting
def fit_translation_scaling(A, B):
    """
    Find the best translation and uniform scaling (no rotation) that maps A to B.
    Returns the transformed A and the RMS error.
    
    Parameters:
    A : np.array, shape (n, 2) - Source points (observed trajectory)
    B : np.array, shape (n, 2) - Target points (PIN pad trajectory)
    
    Returns:
    error : float - RMS error after transformation
    A2 : np.array - Transformed version of A
    """
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

# Score PIN by shape-preserving trajectory fit
def score_pin_by_shape(observed_centers, pin):
    """
    Score PIN candidates by fitting the observed trajectory to the PIN pad trajectory.
    Only translation and uniform scaling are allowed (no rotation).
    """
    try:
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
    except KeyError:
        print(f"Error: Invalid digits in PIN {pin}")
        return np.inf, None
    
    pin_coords = PINPAD_COORDS[pin_indices]
    
    # Need at least 2 points for a meaningful comparison
    if len(observed_centers) < 2 or len(pin_coords) < 2:
        print(f"Warning: Not enough points for meaningful comparison. Observed: {len(observed_centers)}, PIN: {len(pin_coords)}")
        
        # Special case for repeated digit PINs - create a small pattern
        if len(set(pin)) == 1:
            digit = pin[0]
            idx = PINPAD_DIGIT_TO_IDX[digit]
            coord = PINPAD_COORDS[idx]
            # Create a small circle around this coordinate
            radius = 0.2
            angles = np.linspace(0, 2*np.pi, 8)
            fake_centers = np.array([
                [coord[0] + radius * np.cos(angle), coord[1] + radius * np.sin(angle)]
                for angle in angles
            ])
            print(f"Created artificial pattern for repeated digit PIN {pin}")
            return 0.1, fake_centers
            
        return np.inf, None
    
    # For repeated digit PINs with single point trajectory
    if len(set(pin)) == 1 and len(observed_centers) <= 2:
        digit = pin[0]
        idx = PINPAD_DIGIT_TO_IDX[digit]
        coord = PINPAD_COORDS[idx]
        # Create a small pattern
        radius = 0.2
        angles = np.linspace(0, 2*np.pi, 8)
        fake_centers = np.array([
            [coord[0] + radius * np.cos(angle), coord[1] + radius * np.sin(angle)]
            for angle in angles
        ])
        print(f"Created artificial pattern for repeated digit PIN {pin}")
        return 0.1, fake_centers
    
    try:
        error, transformed_centers = fit_translation_scaling(observed_centers, pin_coords)
        
        # Verify no NaN values
        if np.isnan(transformed_centers).any():
            print(f"Warning: NaN values in transformed centers for PIN {pin}")
            return np.inf, None
            
        return error, transformed_centers
    except Exception as e:
        print(f"Error in trajectory fitting for PIN {pin}: {e}")
        return np.inf, None

def remove_small_clusters(centers, times, sizes):
    """
    Remove clusters that are extreme outliers in size (likely noise).
    """
    if len(sizes) <= 1:
        return centers, times, sizes, 0

    # Remove clusters that are less than 15% the size of the largest cluster
    max_size = np.max(sizes)
    min_ratio = 0.15
    valid_indices = np.where(sizes >= min_ratio * max_size)[0]

    # If this would remove all clusters, keep the largest one
    if len(valid_indices) == 0:
        valid_indices = [np.argmax(sizes)]

    removed_count = len(sizes) - len(valid_indices)
    if removed_count > 0:
        print(f"  Removing {removed_count} outlier clusters (size < {min_ratio*100:.0f}% of max)")
        print(f"  Original cluster sizes: {sizes}")
        print(f"  Filtered cluster sizes: {sizes[valid_indices]}")

    return centers[valid_indices], times[valid_indices], sizes[valid_indices], removed_count

# Check if all points are within an 80x80 pixel box
def are_all_points_close(points):
    """
    Returns True if all points are within an 80x80 pixel box.
    Simple check for all-same-digit PINs like 1111, 2222, etc.
    """
    if len(points) < 2:
        return True
    
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    print(f"  Point spread analysis: width={width:.2f}px, height={height:.2f}px (threshold=80px)")
    
    # Return True if all points fit within an 80x80 pixel box
    return width <= 80 and height <= 80

# Check for clusters that suggest repeated digits based on point count
def identify_repeated_digit_clusters(sizes):
    """
    Identify clusters that likely correspond to repeated digits based on cluster sizes.
    When there are fewer clusters than PIN_LENGTH, calculate repetition counts.
    
    Parameters:
    sizes : np.array - Number of points in each cluster
    
    Returns:
    dict - Maps cluster indices to repetition counts (how many times each digit appears)
    """
    if len(sizes) == 0:
        return {}
    
    # Case 1: Only one cluster - all digits are the same
    if len(sizes) == 1:
        return {0: PIN_LENGTH}  # The single cluster represents all digits
    
    # Case 2: Fewer clusters than PIN length - some digits must be repeated
    if len(sizes) < PIN_LENGTH:
        print(f"  Fewer clusters ({len(sizes)}) than PIN length ({PIN_LENGTH}) - analyzing for repeated digits")
        # Calculate relative sizes to determine repetition counts
        total_points = np.sum(sizes)
        relative_sizes = sizes / total_points
        
        print(f"  Cluster relative sizes: {relative_sizes}")
        
        # Start by assigning one position to each cluster
        repetitions = {i: 1 for i in range(len(sizes))}
        remaining_positions = PIN_LENGTH - len(sizes)
        
        # Distribute remaining positions based on relative cluster sizes
        if remaining_positions > 0:
            # Create a weighted distribution based on cluster sizes
            weighted_reps = relative_sizes * remaining_positions
            
            # Assign remaining positions to clusters with largest sizes
            for _ in range(remaining_positions):
                max_idx = np.argmax(weighted_reps)
                repetitions[max_idx] = repetitions.get(max_idx, 0) + 1
                weighted_reps[max_idx] = 0  # Mark as assigned
            
            print(f"  Estimated digit repetitions: {repetitions}")
            
        return {idx: count for idx, count in repetitions.items() if count > 1}
    
    # Case 3: Normal case - look for statistical outliers
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    
    # Consider a cluster as "repeated digit" if significantly more points than average
    threshold = mean_size + 1.5 * std_size
    
    # Find clusters that exceed the threshold
    repetitions = {}
    for i, size in enumerate(sizes):
        if size > threshold:
            # Estimate repetition count based on how much larger the cluster is
            rep_factor = size / mean_size
            if rep_factor > 2.5:
                repetitions[i] = 3  # Appears 3 times
            elif rep_factor > 1.75:
                repetitions[i] = 2  # Appears 2 times
            else:
                repetitions[i] = 2  # Default to appearing twice
    
    return repetitions

# Improved PIN scoring function with shape-based trajectory fitting
def score_pin(pin, centers_ordered, cluster_sizes=None, used_filtering=True, num_major_moves=None):
    """Score PIN candidates using shape-based trajectory fitting and cluster size analysis"""
    # Skip invalid PINs
    try:
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
    except KeyError:
        return np.inf
    
    num_centers = len(centers_ordered)
    num_unique_digits = len(set(pin))
    has_repeats = num_unique_digits < PIN_LENGTH
    
    # If fewer clusters than PIN length, we need special handling
    if num_centers < PIN_LENGTH:
        # If PIN doesn't have repeated digits but we have fewer clusters, penalize
        if not has_repeats:
            return np.inf
            
        # For minimal movement, strongly favor all-same-digit PINs
        if are_all_points_close(centers_ordered) and num_unique_digits == 1:
            return -2.0  # Very strong score for all-same-digit when minimal movement
            
        # Get repetition pattern from cluster sizes
        largest_idx = np.argmax(cluster_sizes)
        repeats = PIN_LENGTH - num_centers + 1
        
        # Check if PIN follows the expected repetition pattern
        digit_counts = Counter(pin)
        repeated_digit = None
        for d, count in digit_counts.items():
            if count == repeats:
                repeated_digit = d
                break
                
        # If no digit is repeated the expected number of times, penalize
        if repeated_digit is None:
            return np.inf
            
        # Get the digits in the order they would appear in the trajectory
        expected_digits = []
        for i in range(num_centers):
            if i == largest_idx:
                expected_digits.extend([repeated_digit] * repeats)
            else:
                # Find a digit that appears exactly once
                for d in pin:
                    if digit_counts[d] == 1 and d not in expected_digits:
                        expected_digits.append(d)
                        break
        
        # If we couldn't build the expected digits correctly, penalize
        if len(expected_digits) != PIN_LENGTH:
            return np.inf
            
        # Give bonus for repeated digit patterns (no more prior probability)
        return -1.0  # Strong bonus for matching repetition pattern
    
    # Standard case: use shape-based scoring for regular PINs
    min_movement = is_minimal_movement(centers_ordered)
    shape_score, _ = score_pin_by_shape(centers_ordered, pin)
    
    # Check if PIN is in common patterns for bonus
    pattern_bonus = 0.0
    for pattern_type, patterns in COMMON_PATTERNS.items():
        if pin in patterns:
            pattern_bonus = 0.3
            break
    
    # Penalize mismatch between unique digits and major moves
    move_penalty = 0.0
    if num_major_moves is not None:
        move_digit_mismatch = abs(num_unique_digits - num_major_moves)
        move_penalty = move_digit_mismatch * 0.5  # Strong penalty for mismatch
    
    # Add bonus for PINs that match repeated digit patterns from cluster size analysis
    cluster_size_bonus = 0.0
    if cluster_sizes is not None and len(cluster_sizes) > 1:
        # Identify clusters that might indicate repeated digits
        repeated_indices = identify_repeated_digit_clusters(cluster_sizes)
        
        # Count repeated digits in PIN and their positions
        digit_counts = Counter(pin)
        repeated_digits = [d for d, count in digit_counts.items() if count > 1]
        
        # If both PIN and clusters have repetitions, give bonus
        if repeated_indices and repeated_digits:
            # Higher bonus if counts match
            if len(repeated_indices) == len(repeated_digits):
                cluster_size_bonus = 0.5
            else:
                cluster_size_bonus = 0.3
    
    # Adjust scores based on conditions
    if min_movement and has_repeats:
        # If minimal movement and PIN has repeats, reduce shape score penalty
        shape_score *= 0.5
    elif not min_movement and has_repeats and num_centers >= 3:
        # If clear movement but PIN has repeats, increase shape score penalty
        shape_score *= 2.0
    
    # Special case: if only 1 center (stationary), favor same-digit PINs
    if num_centers == 1 and has_repeats and num_unique_digits == 1:
        shape_score *= 0.1  # Strongly favor same-digit PINs
    
    # Total score (lower is better) - no more prior score
    total_score = shape_score + move_penalty - pattern_bonus - cluster_size_bonus
    
    return total_score

def generate_repeated_digit_pin_candidates(centers_ordered, cluster_sizes, used_filtering=True, num_major_moves=None, pin_length=4):
    """
    Generate and score repeated-digit PIN candidates by following the trajectory order
    and repeating the digit corresponding to the largest cluster.
    """
    num_clusters = len(centers_ordered)
    if num_clusters >= pin_length:
        return []  # Not a repeated-digit case

    # Find which cluster is largest
    largest_idx = np.argmax(cluster_sizes)
    # How many times to repeat the largest cluster's digit
    repeats = pin_length - num_clusters + 1

    print(f"  Generating repeated digit PINs with cluster {largest_idx+1} repeated {repeats} times")
    
    candidates = []
    for digits in product(PINPAD_DIGITS, repeat=num_clusters):
        # Build the PIN by repeating the largest cluster's digit
        pin_digits = []
        for i in range(num_clusters):
            if i == largest_idx:
                pin_digits.extend([digits[i]] * repeats)
            else:
                pin_digits.append(digits[i])
        # Only keep if length matches PIN_LENGTH
        if len(pin_digits) == pin_length:
            pin = ''.join(pin_digits)
            # Strong bonus for repeated digit PINs - no more prior score
            candidates.append((pin, -1.0))
    # Sort by score (lower is better)
    return sorted(candidates, key=lambda x: x[1])

# Generate all possible PIN candidates with scores
def generate_all_pin_candidates(centers_ordered, cluster_sizes=None, used_filtering=True, num_major_moves=None):
    """Generate all possible PINs with scores"""
    guesses = []
    num_centers = len(centers_ordered)
    
    # Check if all cluster centers are in a small area (80x80 pixel box)
    if are_all_points_close(centers_ordered):
        print("  All clusters are close together: assuming all-same-digit PIN")
        for digit in PINPAD_DIGITS:
            pin = digit * PIN_LENGTH  # e.g., "1111", "2222", etc.
            guesses.append((pin, -2.0))  # Strong score for all-same-digit PINs
        return sorted(guesses, key=lambda x: x[1])
    
    # Special case: fewer clusters than PIN_LENGTH = some repeated digits
    if num_centers < PIN_LENGTH and cluster_sizes is not None:
        print(f"  Found {num_centers} clusters < {PIN_LENGTH} PIN length - generating repeated digit PINs")
        return generate_repeated_digit_pin_candidates(centers_ordered, cluster_sizes, 
                                                    used_filtering, num_major_moves)
    
    # Normal case: proceed with standard PIN candidate generation
    common_pins = []
    for patterns in COMMON_PATTERNS.values():
        for pin in patterns:
            if len(pin) == PIN_LENGTH:
                common_pins.append(pin)
    
    for pin in set(common_pins):
        score = score_pin(pin, centers_ordered, cluster_sizes, used_filtering, num_major_moves)
        if score < np.inf:
            guesses.append((pin, score))
    
    seen_pins = set(g[0] for g in guesses)
    for digits in product(PINPAD_DIGITS, repeat=PIN_LENGTH):
        pin = ''.join(digits)
        if pin not in seen_pins:
            score = score_pin(pin, centers_ordered, cluster_sizes, used_filtering, num_major_moves)
            if score < np.inf:
                guesses.append((pin, score))
            if len(guesses) >= 10000:
                break
    return sorted(guesses, key=lambda x: x[1])

# Visualization functions
def plot_filtered_points(original_points, filtered_points, out_path, title):
    """Plot original vs filtered points"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(original_points[:,0], original_points[:,1], c='gray', s=15, alpha=0.5)
    plt.title('All Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.scatter(filtered_points[:,0], filtered_points[:,1], c='blue', s=20)
    plt.title('Slow Points Only')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_clusters_with_trajectory(points, labels, centers_ordered, out_path, title, cluster_sizes=None):
    """Plot clusters with ordered trajectory path"""
    plt.figure(figsize=(10, 8))
    
    # Plot points with cluster colors
    if labels is not None:
        plt.scatter(points[:,0], points[:,1], c=labels, cmap='tab10', s=30, alpha=0.7)
    
    # Plot ordered centers with numbers and connecting line
    plt.plot(centers_ordered[:,0], centers_ordered[:,1], 'r-', linewidth=2)
    
    # Visualize cluster sizes with circle size
    if cluster_sizes is not None:
        # Normalize sizes for better visualization
        min_size = 80
        max_size = 250
        if len(cluster_sizes) > 0:
            size_range = np.max(cluster_sizes) - np.min(cluster_sizes)
            if size_range > 0:
                norm_sizes = min_size + (max_size - min_size) * (cluster_sizes - np.min(cluster_sizes)) / size_range
            else:
                norm_sizes = np.full_like(cluster_sizes, (min_size + max_size) / 2)
        else:
            norm_sizes = np.array([min_size])
            
        # Find clusters that might indicate repeated digits
        repeated_indices = identify_repeated_digit_clusters(cluster_sizes)
        
        for i, center in enumerate(centers_ordered):
            # Use different color for repeated digit clusters
            if i in repeated_indices:
                color = 'purple'
                edge_color = 'white'
                annotation = f"{i+1}\n({cluster_sizes[i]} pts)"
            else:
                color = 'red'
                edge_color = 'black'
                annotation = f"{i+1}"
                
            plt.scatter(center[0], center[1], c=color, s=norm_sizes[i], edgecolor=edge_color, zorder=10)
            plt.annotate(annotation, xy=center, fontsize=11, fontweight='bold',
                       ha='center', va='center', color='white', zorder=11)
    else:
        # Original behavior without cluster sizes
        for i, center in enumerate(centers_ordered):
            plt.scatter(center[0], center[1], c='red', s=100, edgecolor='black', zorder=10)
            plt.annotate(f"{i+1}", xy=center, fontsize=12, fontweight='bold',
                       ha='center', va='center', color='white', zorder=11)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(out_path)
    plt.close()

def fit_pin_to_clusters(pin, clusters):
    """
    Maps PIN digits to available clusters when clusters < PIN_LENGTH.
    Returns coordinates for the unique digits in order they appear in the PIN.
    """
    # Get unique digits in order of first appearance
    unique_digits = []
    for d in pin:
        if d not in unique_digits:
            unique_digits.append(d)
            
    # If we have exactly the number of unique digits as clusters, use those in order
    if len(unique_digits) == len(clusters):
        unique_coords = np.array([PINPAD_COORDS[PINPAD_DIGIT_TO_IDX[d]] for d in unique_digits])
        return unique_coords
        
    # If clusters < unique digits, take first N unique digits
    return np.array([PINPAD_COORDS[PINPAD_DIGIT_TO_IDX[d]] for d in unique_digits[:len(clusters)]])

def plot_trajectory_on_pinpad(centers_ordered, top_pins, out_path, title, cluster_sizes=None):
    """Plot trajectory overlaid on PIN pad with top PIN candidates"""
    print(f"\nPIN Pad Mapping Debug:")
    print(f"- centers_ordered shape: {centers_ordered.shape if isinstance(centers_ordered, np.ndarray) else 'Not an array'}")
    print(f"- top_pins count: {len(top_pins) if top_pins else 0}")
    
    fig = plt.figure(figsize=(10, 8))
    
    # Draw PIN pad
    for i, (x, y) in enumerate(PINPAD_COORDS):
        plt.scatter(x, y, s=200, c='lightgray', edgecolor='black', zorder=1)
        plt.annotate(PINPAD_DIGITS[i], xy=(x, y), fontsize=16, ha='center', va='center', zorder=2)
    
    # Plot trajectory for top PINs
    if top_pins and len(top_pins) > 0:
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, (pin, score) in enumerate(top_pins[:min(5, len(top_pins))]):
            try:
                pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
                pin_coords = PINPAD_COORDS[pin_indices]
                print(f"- Plotting PIN {pin} with coords shape: {pin_coords.shape}")
                plt.plot(pin_coords[:,0], pin_coords[:,1], '-', color=colors[i], 
                        linewidth=3-i*0.5, alpha=0.7, label=f"{pin} (score: {score:.3f})")
            except Exception as e:
                print(f"  Error plotting PIN {pin}: {e}")
    else:
        print("- No PIN candidates to plot")
    
    # Plot observed trajectory fit for the top candidate PIN
    if len(centers_ordered) > 0 and len(top_pins) > 0:
        top_pin = top_pins[0][0]
        print(f"- Fitting trajectory for top PIN: {top_pin}")
        
        # NEW: Special handling for fewer clusters than PIN length
        if len(centers_ordered) < len(top_pin):
            print(f"- Using special fit for {len(centers_ordered)} clusters < {len(top_pin)} PIN length")
            # Get coordinates for unique digits in the PIN
            target_coords = fit_pin_to_clusters(top_pin, centers_ordered)
            error, transformed_centers = fit_translation_scaling(centers_ordered, target_coords)
            print(f"- Special fit score: {error}")
        else:
            # Standard case - normal shape fit
            score, transformed_centers = score_pin_by_shape(centers_ordered, top_pin)
            print(f"- Standard fit score: {score}")
        
        if transformed_centers is not None:
            print(f"- Transformed centers shape: {transformed_centers.shape}")
            
            # Draw markers based on cluster sizes
            if cluster_sizes is not None:
                # Normalize sizes for better visualization
                min_size = 50
                max_size = 150
                if len(cluster_sizes) > 0:
                    size_range = np.max(cluster_sizes) - np.min(cluster_sizes)
                    if size_range > 0:
                        norm_sizes = min_size + (max_size - min_size) * (cluster_sizes - np.min(cluster_sizes)) / size_range
                    else:
                        norm_sizes = np.full_like(cluster_sizes, (min_size + max_size) / 2)
                
                    # Find clusters that might indicate repeated digits
                    repeated_indices = identify_repeated_digit_clusters(cluster_sizes)
                    
                    for i, center in enumerate(transformed_centers):
                        # Check for NaN values
                        if np.isnan(center).any():
                            print(f"- Warning: NaN in center {i}")
                            continue
                            
                        # Use different color for repeated digit clusters
                        if i in repeated_indices:
                            color = 'purple'
                            edge_color = 'white'
                            # Add annotation for large clusters
                            plt.annotate(f"{cluster_sizes[i]} pts", xy=center, 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=9, color='purple')
                        else:
                            color = 'black'
                            edge_color = 'white'
                            
                        plt.scatter(center[0], center[1], color=color, s=norm_sizes[i], 
                                   edgecolor=edge_color, alpha=0.7, zorder=10)
                else:
                    plt.scatter(transformed_centers[:,0], transformed_centers[:,1], 
                               color='black', s=80, edgecolor='white', alpha=0.7, zorder=10)
            else:
                plt.scatter(transformed_centers[:,0], transformed_centers[:,1], 
                           color='black', s=80, edgecolor='white', alpha=0.7, zorder=10)
            
            # Connect centers with lines
            plt.plot(transformed_centers[:,0], transformed_centers[:,1], 'k--', alpha=0.5, zorder=5,
                     label="Fitted observed trajectory")
        else:
            print("- Warning: transformed_centers is None - cannot plot fitted trajectory")
    
    plt.title(title)
    plt.grid(False)
    
    # Dynamically calculate axis limits to ensure everything is visible
    pad = 1.0
    x_min = min(coord[0] for coord in PINPAD_COORDS) - pad
    x_max = max(coord[0] for coord in PINPAD_COORDS) + pad
    y_min = min(coord[1] for coord in PINPAD_COORDS) - pad
    y_max = max(coord[1] for coord in PINPAD_COORDS) + pad
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.legend(title="PIN candidates & fitted trajectory", 
               loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Important: Invert Y-axis for correct orientation
    plt.savefig(out_path)
    print(f"- Saved PIN pad mapping to {out_path}")
    plt.close()

# Load LED positions and analyze motion
def load_led_positions(video_dir):
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

def analyze_led_y_motion(led_positions, threshold=2, min_stop_length=3):
    """Analyze LED Y positions to find stopped points and potential keystrokes"""
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
        
        # Check for down-then-up motion in each stopped segment
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
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(output_path)
    plt.close()

# Main processing function
def process_csv(csv_path, report_dir, actual_pin):
    """
    Process a single CSV file to guess PINs based on trajectory analysis.
    Uses a multi-stage approach: 
    1. Speed filtering 
    2. Remove small clusters
    3. Check for all-same-digit PIN
    4. Clustering 
    5. PIN inference
    """
    # ------- 1. SETUP, LOADING, CLEANING, PREP -------
    video_dir = os.path.dirname(csv_path)
    video_name = os.path.basename(video_dir)
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\nProcessing video: {video_name}")
    
    # Load and clean data
    df = pd.read_csv(csv_path)
    xcol, ycol = find_ring_center_cols(df)
    xs = df[xcol].values
    ys = df[ycol].values
    mask = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[mask], ys[mask]
    points = np.stack([xs, ys], axis=1)
    frame_indices = np.arange(len(df))[mask]  # Original frame numbers
    
    if points.shape[0] == 0:
        print(f"Warning: No valid points in {csv_path}. Skipping.")
        return
    
    # Initialize results containers
    results = {}
    found_actual = {}
    
    # ------- 2. SPEED FILTERING (PRIMARY ANALYSIS PATH) -------
    print("Performing speed filtering...")
    # Calculate speeds between consecutive points
    speeds = calculate_speeds(points)
    
    # Filter points based on speed (keep only the slowest group = likely keystrokes)
    filtered_points, slow_indices = filter_by_speed(points, speeds)
    filtered_frames = frame_indices[slow_indices]
    
    print(f"  Filtered {len(points)} points to {len(filtered_points)} slow points")
    
    # ------- 3. NEW STEP: EARLY SMALL CLUSTER REMOVAL -------
    print("Removing small clusters from filtered points...")
    filtered_points = remove_small_clusters_early(filtered_points)
    print(f"  After removing small clusters: {len(filtered_points)} points remain")
    
    # Visualize filtering results
    plot_filtered_points(points, filtered_points, 
                        os.path.join(video_dir, 'filtered_points.png'),
                        f'Speed-Based Filtering & Small Cluster Removal: {video_name}')
    
    # ------- 4. CHECK FOR ALL-SAME-DIGIT PIN (BEFORE CLUSTERING) -------
    # Estimate frame dimensions
    frame_height = np.max(ys) - np.min(ys) + 100  # Add padding
    frame_width = np.max(xs) - np.min(xs) + 100   # Add padding
    
    if are_all_points_close(filtered_points):
        print("  PRE-CLUSTERING CHECK: All points are within an 80x80 pixel box - assuming all-same-digit PIN")
        # Generate all-same-digit PINs with very strong scores
        guesses_filtered = [(d*PIN_LENGTH, -3.0) for d in PINPAD_DIGITS]
        
        # Use mean point as center for visualization
        center_point = np.mean(filtered_points, axis=0).reshape(1, 2)
        
        # Plot trajectory on PIN pad
        plot_trajectory_on_pinpad(
            center_point, guesses_filtered[:5],
            os.path.join(video_dir, 'pinpad_mapping_filtered.png'),
            f'Top PIN Candidates: All-same-digit PINs (80x80 pixel box)',
            cluster_sizes=np.array([len(filtered_points)]))
        
        # Store results
        results['filtered'] = {
            'guesses': guesses_filtered, 
            'centers_ordered': center_point,
            'sizes_ordered': np.array([len(filtered_points)]),
            'repeated_clusters': {0: PIN_LENGTH},
            'optimal_k': 1,
            'num_major_moves': 1
        }
        found_actual['filtered'] = any(pin == actual_pin for pin, _ in guesses_filtered)
        
        # Skip clustering and proceed with raw data analysis
        is_filtered_processed = True
    else:
        # Continue with normal clustering workflow
        is_filtered_processed = False
        
        # Scale data for better clustering performance
        scaler = StandardScaler()
        filtered_points_scaled = scaler.fit_transform(filtered_points)
        
        # ------- FILTERED DATA PROCESSING PATH -------
        try:
            print("Processing filtered data (main analysis path)...")
            
            # ------- 5. ELBOW METHOD - Find optimal number of clusters -------
            optimal_k_filtered = find_optimal_clusters(filtered_points_scaled, max_clusters=8)
            plot_elbow_method(filtered_points_scaled, 8, os.path.join(video_dir, 'elbow_filtered.png'))
            print(f"  Optimal clusters for filtered data: {optimal_k_filtered}")
            
            # Apply KMeans with optimal cluster count
            kmeans_filtered = KMeans(n_clusters=optimal_k_filtered, random_state=0, n_init=20)
            labels_filtered = kmeans_filtered.fit_predict(filtered_points_scaled)
            
            # Get initial clusters, times and sizes
            centers_filtered, times_filtered, sizes_filtered, _ = get_cluster_centers_and_times(
                labels_filtered, filtered_points, filtered_frames)
                
            # ------- 6. SIZE FILTERING - Remove small noise clusters -------
            centers_filtered, times_filtered, sizes_filtered, removed_count = remove_small_clusters(
                centers_filtered, times_filtered, sizes_filtered)
                
            # Skip if no valid clusters remained after filtering
            if len(centers_filtered) < 1:
                print("  No valid clusters found after filtering")
                results['filtered'] = {
                    'guesses': [], 
                    'centers_ordered': np.array([]),
                    'sizes_ordered': np.array([]),
                    'repeated_clusters': [],
                    'optimal_k': optimal_k_filtered,
                    'num_major_moves': 0
                }
                found_actual['filtered'] = False
            else:
                # ------- 7. TEMPORAL ORDERING - Order clusters by time -------
                order_filtered = np.argsort(times_filtered)
                centers_ordered_filtered = centers_filtered[order_filtered]
                sizes_ordered_filtered = sizes_filtered[order_filtered]
                
                # Count major moves (after filtering small clusters)
                num_major_moves_filtered = len(centers_ordered_filtered)
                print(f"  Number of major moving parts (filtered): {num_major_moves_filtered}")
                
                # ------- 8. REPEATED DIGIT DETECTION -------
                repeated_clusters_filtered = identify_repeated_digit_clusters(sizes_ordered_filtered)
                if repeated_clusters_filtered:
                    print(f"  Potential repeated digit clusters: {[i+1 for i in repeated_clusters_filtered]}")
                    print(f"  Cluster sizes: {sizes_ordered_filtered}")
                
                # ------- 9. VISUALIZATION - Show trajectory and clusters -------
                plot_clusters_with_trajectory(
                    filtered_points, labels_filtered, centers_ordered_filtered,
                    os.path.join(video_dir, 'trajectory_filtered.png'),
                    f'Filtered Data: {num_major_moves_filtered} major moves',
                    cluster_sizes=sizes_ordered_filtered)
                    
                # ------- 10. POST-CLUSTERING SAME-DIGIT CHECK (FALLBACK) -------
                # Only run as a fallback if pre-clustering check didn't catch it
                if are_all_points_close(centers_ordered_filtered):
                    print("  All clusters are close together (within 80x80 pixel box): assuming all-same-digit PIN")
                    guesses_filtered = [(d*PIN_LENGTH, -2.0) for d in PINPAD_DIGITS]
                    
                    plot_trajectory_on_pinpad(
                        centers_ordered_filtered, guesses_filtered[:5],
                        os.path.join(video_dir, 'pinpad_mapping_filtered.png'),
                        f'Top PIN Candidates: All-same-digit PINs (clusters in 80x80 box)',
                        cluster_sizes=sizes_ordered_filtered)
                    
                    results['filtered'] = {
                        'guesses': guesses_filtered, 
                        'centers_ordered': centers_ordered_filtered,
                        'sizes_ordered': sizes_ordered_filtered,
                        'repeated_clusters': repeated_clusters_filtered,
                        'optimal_k': optimal_k_filtered,
                        'num_major_moves': num_major_moves_filtered
                    }
                    found_actual['filtered'] = any(pin == actual_pin for pin, _ in guesses_filtered)
                
                # Normal case: generate PIN candidates based on trajectory and clustering
                else:
                    print("  Generating standard PIN candidates...")
                    guesses_filtered = generate_all_pin_candidates(
                        centers_ordered_filtered, 
                        cluster_sizes=sizes_ordered_filtered,
                        used_filtering=True,
                        num_major_moves=num_major_moves_filtered
                    )
                    
                    # Add debugging for PIN pad mapping
                    print("\nPreparing to generate PIN pad mapping...")
                    print(f"centers_ordered_filtered shape: {centers_ordered_filtered.shape}")
                    print(f"Number of guesses: {len(guesses_filtered)}")
                    if len(guesses_filtered) > 0:
                        print(f"Top 3 guesses: {guesses_filtered[:3]}")
                        top_pin = guesses_filtered[0][0]
                        print(f"Testing shape fit for top PIN {top_pin}...")
                        test_score, test_transformed = score_pin_by_shape(centers_ordered_filtered, top_pin)
                        if test_transformed is None:
                            print("WARNING: Transformation failed - PIN pad plot may be empty")
                        else:
                            print(f"Transformation successful, score: {test_score}")
                    
                    # Display top PIN candidates on the PIN pad
                    plot_trajectory_on_pinpad(
                        centers_ordered_filtered, guesses_filtered[:5],
                        os.path.join(video_dir, 'pinpad_mapping_filtered.png'),
                        f'Top PIN Candidates: {num_major_moves_filtered} major moves',
                        cluster_sizes=sizes_ordered_filtered)
                    
                    # Check if actual PIN is found
                    found_filtered = False
                    if actual_pin:
                        found_filtered = any(pin == actual_pin for pin, _ in guesses_filtered)
                        if found_filtered:
                            for idx, (pin, score) in enumerate(guesses_filtered):
                                if pin == actual_pin:
                                    print(f"  Filtered data: Actual PIN found at position {idx+1} with score {score:.4f}")
                                    break
                        else:
                            print(f"  Filtered data: Actual PIN not found")
                    
                    # Store results for reporting
                    results['filtered'] = {
                        'guesses': guesses_filtered, 
                        'centers_ordered': centers_ordered_filtered,
                        'sizes_ordered': sizes_ordered_filtered,
                        'repeated_clusters': repeated_clusters_filtered,
                        'optimal_k': optimal_k_filtered,
                        'num_major_moves': num_major_moves_filtered
                    }
                    found_actual['filtered'] = found_filtered
        
        except Exception as e:
            print(f"  Error processing filtered data: {e}")
            import traceback
            traceback.print_exc()
            
            results['filtered'] = {
                'guesses': [], 
                'centers_ordered': np.array([]),
                'sizes_ordered': np.array([]),
                'repeated_clusters': [],
                'optimal_k': 0,
                'num_major_moves': 0
            }
            found_actual['filtered'] = False
    
    # ------- RAW DATA ANALYSIS (SECONDARY PATH) -------
    try:
        print("\nProcessing raw data (secondary analysis)...")
        
        # Also check raw data for all-same-digit PIN before clustering
        if are_all_points_close(points):
            print("  PRE-CLUSTERING RAW CHECK: All points are within an 80x80 pixel box - assuming all-same-digit PIN")
            guesses_raw = [(d*PIN_LENGTH, -3.0) for d in PINPAD_DIGITS]
            center_point = np.mean(points, axis=0).reshape(1, 2)
            
            plot_trajectory_on_pinpad(
                center_point, guesses_raw[:5],
                os.path.join(video_dir, 'pinpad_mapping_raw.png'),
                f'Top PIN Candidates (Raw): All-same-digit PINs (80x80 pixel box)',
                cluster_sizes=np.array([len(points)]))
                
            results['raw'] = {
                'guesses': guesses_raw,
                'centers_ordered': center_point,
                'sizes_ordered': np.array([len(points)]),
                'repeated_clusters': {0: PIN_LENGTH},
                'optimal_k': 1,
                'num_major_moves': 1
            }
            found_actual['raw'] = any(pin == actual_pin for pin, _ in guesses_raw)
        else:
            # Apply early small cluster removal to raw data too
            print("Removing small clusters from raw points...")
            raw_points_filtered = remove_small_clusters_early(points)
            print(f"  After removing small clusters: {len(raw_points_filtered)} points remain")
            
            # Proceed with normal raw data analysis
            scaler = StandardScaler()
            raw_points_scaled = scaler.fit_transform(raw_points_filtered)
            
            optimal_k_raw = find_optimal_clusters(raw_points_scaled, max_clusters=8)
            plot_elbow_method(raw_points_scaled, 8, os.path.join(video_dir, 'elbow_raw.png'))
            print(f"  Optimal clusters for raw data: {optimal_k_raw}")
            
            kmeans_raw = KMeans(n_clusters=optimal_k_raw, random_state=0, n_init=20)
            labels_raw = kmeans_raw.fit_predict(raw_points_scaled)
            
            centers_raw, times_raw, sizes_raw, _ = get_cluster_centers_and_times(
                labels_raw, raw_points_filtered, frame_indices[np.isin(frame_indices, slow_indices)])
            
            centers_raw, times_raw, sizes_raw, small_clusters_removed_raw = remove_small_clusters(
                centers_raw, times_raw, sizes_raw)
                
            if len(centers_raw) < 1:
                print("  No valid clusters found in raw data")
                results['raw'] = {
                    'guesses': [], 
                    'centers_ordered': np.array([]),
                    'sizes_ordered': np.array([]),
                    'repeated_clusters': [],
                    'optimal_k': 0,
                    'num_major_moves': 0
                }
                found_actual['raw'] = False
            else:
                order_raw = np.argsort(times_raw)
                centers_ordered_raw = centers_raw[order_raw]
                sizes_ordered_raw = sizes_raw[order_raw]
                
                repeated_clusters_raw = identify_repeated_digit_clusters(sizes_ordered_raw)
                if repeated_clusters_raw:
                    print(f"  Potential repeated digit clusters (raw): {[i+1 for i in repeated_clusters_raw]}")
                    print(f"  Cluster sizes (raw): {sizes_ordered_raw}")
                
                num_major_moves_raw = len(centers_ordered_raw)
                print(f"  Number of major moving parts (raw): {num_major_moves_raw}")
                
                plot_clusters_with_trajectory(
                    raw_points_filtered, labels_raw, centers_ordered_raw,
                    os.path.join(video_dir, 'trajectory_raw.png'),
                    f'Raw Data: {num_major_moves_raw} major moves',
                    cluster_sizes=sizes_ordered_raw)
                    
                # Check if raw cluster centers are within 80x80 pixel box
                if are_all_points_close(centers_ordered_raw):
                    print("  RAW CLUSTER CHECK: All clusters are within an 80x80 pixel box - assuming all-same-digit PIN")
                    guesses_raw = [(d*PIN_LENGTH, -3.0) for d in PINPAD_DIGITS]
                else:
                    guesses_raw = generate_all_pin_candidates(
                        centers_ordered_raw, 
                        cluster_sizes=sizes_ordered_raw,
                        used_filtering=False, 
                        num_major_moves=num_major_moves_raw
                    )
                
                # Debug PIN pad mapping for raw data
                print("\nPreparing raw data PIN pad mapping...")
                if len(guesses_raw) > 0:
                    top_pin = guesses_raw[0][0]
                    print(f"Testing shape fit for top PIN {top_pin}...")
                    test_score, test_transformed = score_pin_by_shape(centers_ordered_raw, top_pin)
                    if test_transformed is None:
                        print("WARNING: Raw data transformation failed")
                    
                plot_trajectory_on_pinpad(
                    centers_ordered_raw, guesses_raw[:5],
                    os.path.join(video_dir, 'pinpad_mapping_raw.png'),
                    f'Top PIN Candidates (Raw): {num_major_moves_raw} major moves',
                    cluster_sizes=sizes_ordered_raw)
                
                found_raw = False
                if actual_pin:
                    found_raw = any(pin == actual_pin for pin, _ in guesses_raw)
                    if found_raw:
                        for idx, (pin, score) in enumerate(guesses_raw):
                            if pin == actual_pin:
                                print(f"  Raw data: Actual PIN found at position {idx+1} with score {score:.4f}")
                                break
                    else:
                        print(f"  Raw data: Actual PIN not found")
                
                results['raw'] = {
                    'guesses': guesses_raw, 
                    'centers_ordered': centers_ordered_raw,
                    'sizes_ordered': sizes_ordered_raw,
                    'repeated_clusters': repeated_clusters_raw,
                    'optimal_k': optimal_k_raw,
                    'num_major_moves': num_major_moves_raw
                }
                found_actual['raw'] = found_raw
    except Exception as e:
        print(f"  Error processing raw data: {e}")
        results['raw'] = {
            'guesses': [], 
            'centers_ordered': np.array([]),
            'sizes_ordered': np.array([]),
            'repeated_clusters': [],
            'optimal_k': 0,
            'num_major_moves': 0
        }
        found_actual['raw'] = False
    
    # ------- Generate HTML Report -------
    led_positions = load_led_positions(video_dir)
    
    # Generate streamlined HTML report
    html_path = os.path.join(report_dir, f'{video_name}.html')
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PIN Analysis for {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .pin {{ font-size: 1.2em; font-weight: bold; }}
        img {{ max-width: 600px; border: 1px solid #ccc; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .common {{ background-color: #fff0d0; }}
        .found {{ background-color: #d4ffd4; }}
        .repeated {{ background-color: #ffedcc; }}
    </style>
</head>
<body>
    <h1>PIN Analysis for {video_name}</h1>
    
    <h2>1. Speed-Filtered Analysis</h2>
    <img src="{os.path.relpath(os.path.join(video_dir, 'filtered_points.png'), report_dir)}" alt="Filtered Points">
""")

        if 'filtered' in results and len(results['filtered']['centers_ordered']) > 0:
            # Add cluster size information
            cluster_size_info = ""
            if len(results['filtered']['sizes_ordered']) > 0:
                cluster_size_info = f"<p>Cluster sizes: {', '.join(map(str, results['filtered']['sizes_ordered']))}</p>"
                if results['filtered']['repeated_clusters']:
                    repeated_idxs = [i+1 for i in results['filtered']['repeated_clusters']]
                    cluster_size_info += f"<p><b>Potential repeated digits at clusters: {', '.join(map(str, repeated_idxs))}</b></p>"
                
            f.write(f"""
    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
        <div>
            <h3>KMeans Elbow Method</h3>
            <img src="{os.path.relpath(os.path.join(video_dir, 'elbow_filtered.png'), report_dir)}" alt="Elbow Method">
            <p>Optimal clusters: {results['filtered']['optimal_k']}</p>
            <p><b>Major moving parts: {results['filtered']['num_major_moves']}</b></p>
            {cluster_size_info}
        </div>
        <div>
            <h3>Filtered Trajectory</h3>
            <img src="{os.path.relpath(os.path.join(video_dir, 'trajectory_filtered.png'), report_dir)}" alt="Filtered Trajectory">
            <p>Circle sizes represent number of points in each cluster.</p>
        </div>
    </div>
""")

            # Add PIN pad mapping visualization if it exists
            pinpad_filtered_path = os.path.join(video_dir, 'pinpad_mapping_filtered.png')
            if os.path.exists(pinpad_filtered_path):
                f.write(f"""
    <div>
        <h3>PIN Pad Mapping</h3>
        <img src="{os.path.relpath(pinpad_filtered_path, report_dir)}" alt="PIN Pad Mapping">
        <p>The dotted black line shows the observed trajectory fitted to the top candidate PIN.</p>
    </div>
""")

            f.write(f"""    
    <h3>Top 20 PIN Guesses</h3>
""")

            if results['filtered']['guesses']:
                f.write("""    <table>
        <tr><th>Rank</th><th>PIN</th><th>Score</th><th>Notes</th></tr>
""")
                for idx, (pin, score) in enumerate(results['filtered']['guesses'][:20], 1):
                    # Check if it's in common patterns
                    is_common = False
                    pattern_type = ""
                    for p_type, patterns in COMMON_PATTERNS.items():
                        if pin in patterns:
                            is_common = True
                            pattern_type = p_type
                            break
                    
                    is_repeated = pin in COMMON_PATTERNS['repeated']
                    
                    highlight = ' class="found"' if pin == actual_pin else ''
                    if is_repeated and not highlight:
                        highlight = ' class="repeated"'
                    elif is_common and not highlight:
                        highlight = ' class="common"'
                    
                    note = f"Common {pattern_type} pattern" if is_common else ""
                    if len(set(pin)) < len(pin):
                        repeats = [d for d, count in Counter(pin).items() if count > 1]
                        repeat_str = ", ".join(repeats)
                        note += f"{'; ' if note else ''}Has repeated digit{'s' if len(repeats) > 1 else ''}: {repeat_str}"
                    
                    # Add major moves match info
                    num_unique = len(set(pin))
                    if num_unique == results['filtered']['num_major_moves']:
                        note += f"{'; ' if note else ''}Matches {results['filtered']['num_major_moves']} major moves"
                    
                    f.write(f"""        <tr{highlight}><td>{idx}</td><td class="pin">{pin}</td><td>{score:.4f}</td><td>{note}</td></tr>
""")
                f.write("    </table>\n")
                
                # Add link to full results
                full_results_path = os.path.join(report_dir, f'{video_name}_full_results.html')
                with open(full_results_path, 'w') as full_f:
                    full_f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Full PIN Results for {video_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        table {{ border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .found {{ background-color: #d4ffd4; }}
    </style>
</head>
<body>
    <h1>Full PIN Rankings for {video_name}</h1>
    <p><a href="{os.path.basename(html_path)}">Back to main report</a></p>
    
    <h2>All PINs Ranked by Score</h2>
    <table>
        <tr><th>Rank</th><th>PIN</th><th>Score</th></tr>
""")
                    for idx, (pin, score) in enumerate(results['filtered']['guesses'], 1):
                        highlight = ' class="found"' if pin == actual_pin else ''
                        full_f.write(f"""        <tr{highlight}><td>{idx}</td><td>{pin}</td><td>{score:.4f}</td></tr>
""")
                    full_f.write("""    </table>
</body>
</html>""")
                
                f.write(f"""
    <p><a href="{os.path.basename(full_results_path)}" target="_blank">View Complete PIN Rankings</a></p>
""")
            else:
                f.write("    <p>No valid PIN guesses found.</p>\n")
        else:
            f.write("    <p>No valid clusters found.</p>\n")
        
        # End of HTML
        f.write("""
</body>
</html>""")

    return results, found_actual

def main():
    """Main function to process all CSV files"""
    report_dir = os.path.join('.', REPORT_FOLDER)
    os.makedirs(report_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))
    total = len(csv_files)
    print(f"Found {total} *_ring_center.csv files.")
    print(f"Using PIN_LENGTH = {PIN_LENGTH}")
    
    if not csv_files:
        print("No *_ring_center.csv files found. Please check your folder structure and file names.")
        return
    
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"Processing {idx}/{total}: {csv_path}")
        try:
            process_csv(csv_path, report_dir, ACTUAL_PIN)
        except Exception as e:
            print(f"Error processing file {idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
