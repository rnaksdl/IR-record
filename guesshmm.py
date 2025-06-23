import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --- CONFIGURATION ---
ACTUAL_PIN = '1397'  # Set your actual PIN here for evaluation
OUTPUT_DIR = './output'
REPORT_FOLDER = 'report'
PINPAD_COORDS = np.array([
    [0,0], [1,0], [2,0],   # 1 2 3
    [0,1], [1,1], [2,1],   # 4 5 6
    [0,2], [1,2], [2,2],   # 7 8 9
    [1,3]                  # 0
])
PINPAD_DIGITS = ['1','2','3','4','5','6','7','8','9','0']
N_STATES = len(PINPAD_DIGITS)
SIGMA = 0.5  # Standard deviation for emission (tune for your data)

def load_trajectory(csv_path):
    df = pd.read_csv(csv_path)
    for xcol, ycol in [('ring_x', 'ring_y'), ('center_x', 'center_y'), ('x', 'y')]:
        if xcol in df.columns and ycol in df.columns:
            xs = df[xcol].values
            ys = df[ycol].values
            mask = ~np.isnan(xs) & ~np.isnan(ys)
            return np.stack([xs[mask], ys[mask]], axis=1)
    raise ValueError("Could not find ring center columns in CSV.")

def build_emission_probs(points):
    # emission_probs[t, s] = P(observation at t | state s)
    emission_probs = np.zeros((len(points), N_STATES))
    for s, mu in enumerate(PINPAD_COORDS):
        rv = multivariate_normal(mean=mu, cov=SIGMA**2 * np.eye(2))
        emission_probs[:, s] = rv.pdf(points)
    # Normalize for numerical stability
    emission_probs = emission_probs / (emission_probs.sum(axis=1, keepdims=True) + 1e-12)
    return emission_probs

def build_transition_probs():
    # Uniform transition probability (can be replaced with PIN statistics)
    trans = np.ones((N_STATES, N_STATES)) / N_STATES
    return trans

def viterbi(emission_probs, trans_probs, start_probs):
    T, N = emission_probs.shape
    log_emiss = np.log(emission_probs + 1e-12)
    log_trans = np.log(trans_probs + 1e-12)
    log_start = np.log(start_probs + 1e-12)
    dp = np.zeros((T, N))  # dp[t, s]: max log-prob of path ending at state s at time t
    ptr = np.zeros((T, N), dtype=int)
    dp[0] = log_start + log_emiss[0]
    for t in range(1, T):
        for s in range(N):
            seq_probs = dp[t-1] + log_trans[:, s]
            ptr[t, s] = np.argmax(seq_probs)
            dp[t, s] = seq_probs[ptr[t, s]] + log_emiss[t, s]
    # Backtrack
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(dp[-1])
    for t in range(T-2, -1, -1):
        states[t] = ptr[t+1, states[t+1]]
    return states, np.max(dp[-1])

def process_csv(csv_path, actual_pin):
    points = load_trajectory(csv_path)
    emission_probs = build_emission_probs(points)
    trans_probs = build_transition_probs()
    start_probs = np.ones(N_STATES) / N_STATES  # Uniform start
    states, logprob = viterbi(emission_probs, trans_probs, start_probs)
    decoded_digits = [PINPAD_DIGITS[s] for s in states]
    # Collapse consecutive repeats (user may linger on a key)
    pin_guess = []
    for d in decoded_digits:
        if not pin_guess or d != pin_guess[-1]:
            pin_guess.append(d)
    pin_guess = ''.join(pin_guess)
    # Only keep first 4 digits as the PIN guess
    pin_guess = pin_guess[:4]
    print(f"Decoded PIN: {pin_guess} (actual: {actual_pin})")
    print(f"Match: {'YES' if pin_guess == actual_pin else 'NO'}")
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(points[:,0], points[:,1], marker='o', label='Trajectory')
    for i, mu in enumerate(PINPAD_COORDS):
        plt.scatter(mu[0], mu[1], marker='s', s=100, label=f"{PINPAD_DIGITS[i]}")
    plt.title(f"Decoded PIN: {pin_guess} (actual: {actual_pin})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(csv_path), REPORT_FOLDER)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, 'hmm_decoded.png'))
    plt.close()
    # Save result
    with open(os.path.join(outdir, 'hmm_result.txt'), 'w') as f:
        f.write(f"Decoded PIN: {pin_guess}\n")
        f.write(f"Actual PIN: {actual_pin}\n")
        f.write(f"Match: {'YES' if pin_guess == actual_pin else 'NO'}\n")
        f.write(f"Log-probability: {logprob}\n")

def main():
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*', '*_ring_center.csv'))
    print(f"Found {len(csv_files)} *_ring_center.csv files.")
    for idx, csv_path in enumerate(csv_files, 1):
        print(f"Processing {idx}/{len(csv_files)}: {csv_path}")
        process_csv(csv_path, ACTUAL_PIN)

if __name__ == '__main__':
    main()
