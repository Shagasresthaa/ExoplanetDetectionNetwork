import os
import pandas as pd

def find_shortest_sequence_length(indices):
    """
    Find the shortest sequence length across all light curve CSVs.

    Parameters:
        indices (list of str): List of paths to index CSV files (positive and negative).

    Returns:
        int: Shortest sequence length found.
    """
    shortest_length = float('inf')

    for index_csv in indices:
        print(f"Processing index: {index_csv}")
        df = pd.read_csv(index_csv)

        for lc_path in df['LC_Path'].dropna():
            if not os.path.exists(lc_path):
                print(f"Warning: File not found: {lc_path}")
                continue

            # Load light curve CSV
            try:
                lc_data = pd.read_csv(lc_path)
                sequence_length = len(lc_data)
                if sequence_length < shortest_length:
                    shortest_length = sequence_length
            except Exception as e:
                print(f"Error reading {lc_path}: {e}")

    print(f"Shortest sequence length found: {shortest_length}")
    return shortest_length


def trim_sequences_to_length(indices, target_length):
    """
    Trim all light curve sequences to the target length.

    Parameters:
        indices (list of str): List of paths to index CSV files (positive and negative).
        target_length (int): Length to which all sequences will be trimmed.
    """
    for index_csv in indices:
        print(f"Processing index: {index_csv}")
        df = pd.read_csv(index_csv)

        for lc_path in df['LC_Path'].dropna():
            if not os.path.exists(lc_path):
                print(f"Warning: File not found: {lc_path}")
                continue

            # Load and trim light curve CSV
            try:
                lc_data = pd.read_csv(lc_path)
                trimmed_data = lc_data.iloc[:target_length]  # Trim to target length
                trimmed_data.to_csv(lc_path, index=False)  # Overwrite the file
                print(f"Trimmed and saved: {lc_path}")
            except Exception as e:
                print(f"Error processing {lc_path}: {e}")


# Example usage
positive_index = "data/positive_tic_extracts.csv"  # Positive index CSV
negative_index = "data/negative_tic_extracts.csv"  # Negative index CSV

indices = [positive_index, negative_index]

# Step 1: Find the shortest sequence length
shortest_length = find_shortest_sequence_length(indices)

# Step 2: Trim all sequences to the shortest length
trim_sequences_to_length(indices, shortest_length)
