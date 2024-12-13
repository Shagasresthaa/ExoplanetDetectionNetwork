import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
index_csv_path = "data/preprocessed/datasetIndex.csv"  # Current dataset index file
missing_csv_path = "data/preprocessed/missing_files.csv"  # Path to save missing files
plots_folder = "data/preprocessed/LC_EXT_trimmed_plots"  # Folder for trimmed/padded plots

# Target sequence length
target_length = 6000

# Ensure the plots folder exists
os.makedirs(plots_folder, exist_ok=True)

# Load the dataset index
try:
    data = pd.read_csv(index_csv_path)
    print(f"Loaded dataset index with {len(data)} entries.")
except FileNotFoundError:
    print(f"Index file not found: {index_csv_path}")
    exit()

# Track missing files
missing_files = []

# Process each light curve file
for idx, row in data.iterrows():
    lc_path = row['LC_Path']
    if not os.path.exists(lc_path):
        print(f"Missing file: {lc_path}")
        missing_files.append(lc_path)
        continue

    try:
        # Load light curve data
        lc_data = pd.read_csv(lc_path).dropna(subset=['FLUX'])  # Drop rows with NaN in FLUX
        flux = lc_data['FLUX'].values
        
        # Adjust sequence length
        if len(flux) > target_length:
            flux = flux[:target_length]  # Trim
        elif len(flux) < target_length:
            flux = np.pad(flux, (0, target_length - len(flux)), 'constant')  # Pad
        
        # Save the adjusted light curve back to the same file
        pd.DataFrame({'FLUX': flux}).to_csv(lc_path, index=False)

        # Generate and save the plot
        plot_path = os.path.join(plots_folder, f"{os.path.basename(lc_path).replace('.csv', '.png')}")
        plt.figure(figsize=(10, 5))
        plt.plot(flux, color="blue", label="Trimmed/Padded Light Curve")
        plt.xlabel("Time Steps")
        plt.ylabel("Flux")
        plt.title(f"Light Curve: {os.path.basename(lc_path)}")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

    except Exception as e:
        print(f"Error processing {lc_path}: {e}")

# Remove missing entries from the dataset index
data = data[~data['LC_Path'].isin(missing_files)]

# Save the updated dataset index
data.to_csv(index_csv_path, index=False)
print(f"Updated dataset index saved to: {index_csv_path}")

# Save missing file paths
if missing_files:
    pd.DataFrame({'Missing_LC_Files': missing_files}).to_csv(missing_csv_path, index=False)
    print(f"Missing file paths saved to: {missing_csv_path}")
else:
    print("No missing files found.")
