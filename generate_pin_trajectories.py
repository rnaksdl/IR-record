"""
Generate and store PIN trajectories for different PIN lengths.
This pre-computes all possible PIN combinations to avoid recalculation.
"""
import numpy as np
import pickle
import os
from itertools import product
import time

# Button dimensions - must match those in guess.py
BUTTON_WIDTH = 10.0
BUTTON_HEIGHT = 5.5
GAP = 0.9
X_OFFSET = BUTTON_WIDTH/2
Y_OFFSET = BUTTON_HEIGHT/2

# Create coordinate array for virtual keypad
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

# Directory to store trajectory files
TRAJECTORIES_DIR = './pin_trajectories'
os.makedirs(TRAJECTORIES_DIR, exist_ok=True)

def generate_pin_trajectories(pin_length):
    """Generate trajectories for all possible PINs of given length."""
    print(f"Generating trajectories for {pin_length}-digit PINs...")
    start_time = time.time()
    
    # Dictionary to store PIN trajectories
    trajectories = {}
    
    # Generate all possible PIN combinations
    total_pins = 10**pin_length
    pin_combinations = [''.join(p) for p in product(PINPAD_DIGITS, repeat=pin_length)]
    
    # Calculate trajectory for each PIN
    for i, pin in enumerate(pin_combinations):
        if i % 10000 == 0:
            progress = (i / total_pins) * 100
            print(f"  Progress: {progress:.1f}% ({i}/{total_pins})")
            
        # Get coordinates for each digit
        pin_indices = [PINPAD_DIGIT_TO_IDX[d] for d in pin]
        pin_coords = PINPAD_COORDS[pin_indices]
        
        # Store the coordinates
        trajectories[pin] = pin_coords
    
    # Save trajectories to file
    output_path = os.path.join(TRAJECTORIES_DIR, f'pin{pin_length}_trajectories.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    elapsed = time.time() - start_time
    print(f"Done! Generated {len(trajectories)} trajectories in {elapsed:.2f} seconds")
    print(f"Saved to: {output_path}")
    
    return trajectories

if __name__ == "__main__":
    # Generate trajectories for 4, 5, and 6 digit PINs
    for length in [4, 5, 6]:
        generate_pin_trajectories(length)
