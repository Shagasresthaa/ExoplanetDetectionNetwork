import pandas as pd
from astropy.io import fits
import os
import logging
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# Specify paths
source_index_path = "data/truePositivesRawData/raw_fits_file_index.csv"
base_path = "/run/media/maverick/X10 Pro/exoplanetDataset/truePositivesExtracts"
lc_path = os.path.join(base_path, "LC")
lcap_path = os.path.join(base_path, "LCAP")
lcplots_path = os.path.join(base_path, "LCPlots")
lcapimg_path = os.path.join(base_path, "LCAPIMG")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Setup logging
log_file = os.path.join("logs/extraction_logs/", f"extraction_log_{timestamp}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load index file
index_df = pd.read_csv(source_index_path)

# Ensure output directories exist
os.makedirs(lc_path, exist_ok=True)
os.makedirs(lcap_path, exist_ok=True)
os.makedirs(lcplots_path, exist_ok=True)
os.makedirs(lcapimg_path, exist_ok=True)

# Generate timestamped filename for the updated index file
updated_index_path = f"data/true_positives_extract/extracted_LC_data_{timestamp}.csv"

os.makedirs("data/true_positives_extract", exist_ok=True)

# Initialize list to store new index data
updated_index_data = []

# Track occurrence counts for each tic_id
tic_id_counts = defaultdict(int)

for _, row in index_df.iterrows():
    # Only process LC files
    if row['file_type'] == 'LC':
        tic_id = row['tic_id']
        file_path = row['file_path']
        
        # Increment the occurrence count for the current tic_id
        tic_id_counts[tic_id] += 1
        occurrence_num = tic_id_counts[tic_id]
        
        # Define paths for the LC CSV, LCAP CSV, LCPlot image, and LCAP Image
        lc_filename = f"{tic_id}_{occurrence_num}_lc.csv"
        lc_csv_path = os.path.join(lc_path, lc_filename)
        
        lcap_filename = f"{tic_id}_{occurrence_num}_lcap.csv"
        lcap_csv_path = os.path.join(lcap_path, lcap_filename)
        
        lcplot_filename = f"{tic_id}_{occurrence_num}_lcplot.png"
        lcplot_path = os.path.join(lcplots_path, lcplot_filename)
        
        lcapimg_filename = f"{tic_id}_{occurrence_num}_lcapimg.png"
        lcapimg_path_full = os.path.join(lcapimg_path, lcapimg_filename)
        
        try:
            # Open the FITS file and extract data
            with fits.open(file_path) as hdul:
                # Extract and save the aperture mask (pixel-level data) from the third HDU
                aperture_data = hdul[2].data  # Assuming the aperture data is in the third HDU
                
                # Convert aperture data to DataFrame and save as CSV
                aperture_df = pd.DataFrame(aperture_data)
                aperture_df.to_csv(lcap_csv_path, index=False)
                
                # Save the aperture mask as an image
                plt.figure(figsize=(6, 6))
                plt.imshow(aperture_data, cmap="gray", origin="lower")
                plt.colorbar(label="Pixel Intensity")
                plt.title(f"TIC {tic_id} Aperture Mask - Observation {occurrence_num}")
                plt.savefig(lcapimg_path_full)
                plt.close()  # Close the plot to free memory
                
                # Extract and save light curve data (TIME and PDCSAP_FLUX) for TCNN processing
                lightcurve_data = hdul[1].data  # Assuming lightcurve data is in the second HDU
                time_data = lightcurve_data['TIME']
                flux_data = lightcurve_data['PDCSAP_FLUX']
                
                # Save light curve data to CSV
                lightcurve_df = pd.DataFrame({
                    'TIME_BJD': time_data,
                    'PDCSAP_FLUX': flux_data
                })
                lightcurve_df.to_csv(lc_csv_path, index=False)
                
                # Plot TIME vs PDCSAP_FLUX
                plt.figure(figsize=(10, 6))
                plt.plot(time_data, flux_data, label="PDCSAP Flux")
                plt.xlabel("Time (BJD)")
                plt.ylabel("PDCSAP Flux")
                plt.title(f"TIC {tic_id} - Observation {occurrence_num}")
                plt.legend()
                plt.grid()
                
                # Save the plot
                plt.savefig(lcplot_path)
                plt.close()  # Close the plot to free memory
                
            # Log successful extraction
            logging.info(f"Successfully extracted data to {lc_csv_path}, {lcap_csv_path}, plot to {lcplot_path}, and aperture image to {lcapimg_path_full}")
            
            # Add paths to the row and append to updated data list
            updated_row = row.to_dict()
            updated_row["lc_csv_path"] = lc_csv_path
            updated_row["lcap_csv_path"] = lcap_csv_path
            updated_row["lcplot_path"] = lcplot_path
            updated_row["lcapimg_path"] = lcapimg_path_full
            updated_index_data.append(updated_row)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

# Create updated DataFrame with new CSV and plot paths
updated_index_df = pd.DataFrame(updated_index_data)

# Save updated index to the timestamped CSV
updated_index_df.to_csv(updated_index_path, index=False)
logging.info(f"Updated index saved at {updated_index_path}")
