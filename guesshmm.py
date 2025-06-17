import os
import numpy as np
import pandas as pd
from hmmlearn import hmm

# Keypad and move definitions
keypad = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [None, 0, None]
]
move_labels = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right', 'stay']
move_vectors = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
    (0, 0)
]

def direction_to_label(angle):
    dirs = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4])
    labels = ['right', 'up-right', 'up', 'up-left', 'left', 'down-left', 'down', 'down-right']
    idx = (np.abs(np.angle(np.exp(1j*(angle - dirs))))).argmin()
    return labels[idx]

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['x', 'y'])
    return df

def extract_observations(df):
    directions = df['direction'].values
    obs_labels = [direction_to_label(a) for a in directions]
    label_to_idx = {label: i for i, label in enumerate(move_labels)}
    obs_idx = [label_to_idx.get(l, label_to_idx['stay']) for l in obs_labels]
    return np.array(obs_idx).reshape(-1, 1), obs_labels

def build_hmm(n_components=9):
    model = hmm.MultinomialHMM(n_components=n_components, n_iter=100, init_params="")
    model.startprob_ = np.ones(n_components) / n_components
    model.transmat_ = np.ones((n_components, n_components)) / n_components
    model.emissionprob_ = np.ones((n_components, len(move_labels))) / len(move_labels)
    return model

def enumerate_pin_candidates(obs_labels, pin_length=4):
    candidates = []
    for start_row in range(4):
        for start_col in range(3):
            if keypad[start_row][start_col] is None:
                continue
            row, col = start_row, start_col
            sequence = [keypad[row][col]]
            for move in obs_labels[:pin_length-1]:
                if move not in move_labels:
                    break
                idx = move_labels.index(move)
                dx, dy = move_vectors[idx]
                new_row, new_col = row + dx, col + dy
                if 0 <= new_row < 4 and 0 <= new_col < 3 and keypad[new_row][new_col] is not None:
                    row, col = new_row, new_col
                    sequence.append(keypad[row][col])
                else:
                    break
            if len(sequence) == pin_length:
                candidates.append(sequence)
    return candidates

def find_feature_files(root_folder):
    feature_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "ring_center_features.csv":
                feature_files.append(os.path.join(dirpath, filename))
    return feature_files

def main():
    root_folder = "outputhmm"
    pin_length = 4
    feature_files = find_feature_files(root_folder)
    if not feature_files:
        print("No ring_center_features.csv files found in outputhmm.")
        return

    for csv_path in feature_files:
        print(f"\nProcessing: {csv_path}")
        try:
            df = load_features(csv_path)
            obs_idx, obs_labels = extract_observations(df)
            model = build_hmm(n_components=9)
            # logprob, hidden_states = model.decode(obs_idx, algorithm="viterbi")  # Not used here
            pin_candidates = enumerate_pin_candidates(obs_labels, pin_length=pin_length)
            print("  Most likely observed moves:", obs_labels[:pin_length-1])
            print("  Possible PIN candidates:")
            for pin in pin_candidates:
                print("   ", "".join(str(d) for d in pin))
            if not pin_candidates:
                print("   No valid candidates found for this file.")
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")

if __name__ == "__main__":
    main()
