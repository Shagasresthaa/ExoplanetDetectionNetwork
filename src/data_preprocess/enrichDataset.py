import os
import numpy as np
import pandas as pd
from lightkurve import TessLightCurve
from scipy.signal import find_peaks
from scipy.stats import skew
from tqdm import tqdm

# Define the dataset path
CSV_PATH = "data/preprocessed/balanced_datasetIndex.csv"  # Path to the original ExoFOP dataset

def extract_lightcurve_features_from_fits(fits_path):
    """
    Extract additional features from a TESS FITS file.

    Parameters:
        fits_path (str): Path to the FITS file.

    Returns:
        dict: Extracted features.
    """
    features = {
        "Transit_Skewness": np.nan,
        "Periodicity_Confidence": np.nan,
        "Centroid_Shift": np.nan,
    }
    
    try:
        # Load light curve
        lc = TessLightCurve.read(fits_path)
        
        # Normalize the light curve
        lc = lc.normalize().remove_nans()
        
        # Transit shape (skewness)
        folded_lc = lc.fold(period=lc.meta.get('PERIOD', 1), transit_midpoint=lc.meta.get('EPOCH', 0))
        features["Transit_Skewness"] = skew(folded_lc.flux)

        # Periodicity confidence (signal-to-noise of the main period)
        periodogram = lc.to_periodogram()
        features["Periodicity_Confidence"] = periodogram.max_power.value / np.median(periodogram.power.value)
        
        # Centroid analysis (placeholder)
        features["Centroid_Shift"] = np.nan  # Implement if centroid data is available

    except Exception as e:
        print(f"Failed to process FITS file at {fits_path}: {e}")
    
    return features

# Load ExoFOP data
exofop_data = pd.read_csv(CSV_PATH)

# Extract light curve features for each TIC using FITS_File_Path
feature_rows = []
for _, row in tqdm(exofop_data.iterrows(), total=len(exofop_data)):
    fits_path = row.get("FITS_File_Path")
    if not os.path.exists(fits_path):
        print(f"FITS file not found: {fits_path}")
        feature_rows.append({
            "Transit_Skewness": np.nan,
            "Periodicity_Confidence": np.nan,
            "Centroid_Shift": np.nan,
        })
        continue

    features = extract_lightcurve_features_from_fits(fits_path)
    feature_rows.append(features)

# Create a DataFrame with extracted features
feature_df = pd.DataFrame(feature_rows)

# Merge features with ExoFOP data
enriched_data = pd.concat([exofop_data.reset_index(drop=True), feature_df], axis=1)

# Ensure `Label` is the last column
columns = [col for col in enriched_data.columns if col != "Label"] + ["Label"]
enriched_data = enriched_data[columns]

# Rewrite the original CSV with enriched data
enriched_data.to_csv(CSV_PATH, index=False)

print(f"Enriched dataset saved to {CSV_PATH}")
