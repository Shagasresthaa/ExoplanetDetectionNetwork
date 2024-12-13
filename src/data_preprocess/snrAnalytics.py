import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_noise_metrics(lc_csv):
    """
    Calculate noise metrics (STD, MAD, RMS) for a light curve.

    Parameters:
        lc_csv (str): Path to the light curve CSV file.

    Returns:
        dict: A dictionary containing noise metrics.
    """
    try:
        lc_data = pd.read_csv(lc_csv)
        flux = lc_data['FLUX'].dropna()  # Drop NaN values to avoid errors

        # Noise metrics
        std_dev = np.std(flux)  # Standard Deviation
        mad = np.median(np.abs(flux - np.median(flux)))  # Median Absolute Deviation
        rms = np.sqrt(np.mean((flux - np.mean(flux))**2))  # Root Mean Square

        return {"STD": std_dev, "MAD": mad, "RMS": rms}
    except Exception as e:
        print(f"Error processing {lc_csv}: {e}")
        return None


def analyze_denoising(index_csv, thresholds):
    """
    Analyze and count how many light curves need denoising.

    Parameters:
        index_csv (str): Path to the index CSV file.
        thresholds (dict): Thresholds for STD, MAD, and RMS to determine noisiness.

    Returns:
        dict: Counts of clean and noisy light curves.
    """
    counts = {"Clean": 0, "Needs Denoising": 0}

    try:
        df = pd.read_csv(index_csv)

        for lc_path in df['LC_Path'].dropna():
            if not os.path.exists(lc_path):
                print(f"File not found: {lc_path}")
                continue

            metrics = calculate_noise_metrics(lc_path)
            if metrics:
                is_noisy = (
                    metrics["STD"] > thresholds["STD"] or
                    metrics["MAD"] > thresholds["MAD"] or
                    metrics["RMS"] > thresholds["RMS"]
                )
                if is_noisy:
                    counts["Needs Denoising"] += 1
                else:
                    counts["Clean"] += 1
    except Exception as e:
        print(f"Error reading index CSV: {e}")

    return counts


def plot_denoising_counts(counts, title="Denoising Requirements"):
    """
    Plot a bar chart for denoising counts.

    Parameters:
        counts (dict): Counts of clean and noisy light curves.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(counts.keys(), counts.values(), color=['green', 'red'], alpha=0.7, edgecolor='black')
    plt.ylabel('Number of Light Curves')
    plt.title(title)
    plt.grid(axis='y')
    plt.show()


# Example usage
positive_index = "data/positive_tic_extracts.csv"  # Positive index CSV
negative_index = "data/negative_tic_extracts.csv"  # Negative index CSV

# Define thresholds for noisiness
noise_thresholds = {"STD": 100, "MAD": 50, "RMS": 100}

# Analyze denoising requirements for positive and negative datasets
print("Analyzing Positive Data...")
positive_counts = analyze_denoising(positive_index, noise_thresholds)

print("\nAnalyzing Negative Data...")
negative_counts = analyze_denoising(negative_index, noise_thresholds)

# Plot results
print("\nPlotting results for Positive Data...")
plot_denoising_counts(positive_counts, title="Denoising Requirements for Positive Data")

print("\nPlotting results for Negative Data...")
plot_denoising_counts(negative_counts, title="Denoising Requirements for Negative Data")
