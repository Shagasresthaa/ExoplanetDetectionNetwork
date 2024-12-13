import os
import logging
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Set up logging with timestamp
LOG_DIR = "logs/extraction_logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists
log_file_name = f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=os.path.join(LOG_DIR, log_file_name),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def extract_fits_data(index_csv, extract_folder):
    logging.info(f"Starting extraction process using index: {index_csv}")

    # Ensure target directories exist
    lc_ext_dir = os.path.join(extract_folder, "LC_EXT")
    ap_ext_dir = os.path.join(extract_folder, "AP_EXT")
    ap_img_dir = os.path.join(extract_folder, "AP_IMG")
    lc_plot_dir = os.path.join(extract_folder, "LC_EXT_plots")
    os.makedirs(lc_ext_dir, exist_ok=True)
    os.makedirs(ap_ext_dir, exist_ok=True)
    os.makedirs(ap_img_dir, exist_ok=True)
    os.makedirs(lc_plot_dir, exist_ok=True)

    # Load the index CSV
    try:
        df = pd.read_csv(index_csv)
        logging.info(f"Loaded index CSV: {index_csv}")
    except Exception as e:
        logging.error(f"Failed to load index CSV: {e}")
        return

    # Ensure the CSV has the necessary column for FITS file paths
    if 'FITS_File_Path' not in df.columns or 'TIC_ID' not in df.columns:
        logging.error("Index CSV must contain 'FITS_File_Path' and 'TIC_ID' columns.")
        return

    # Add new columns for extracted paths
    df['LC_Path'] = None
    df['AP_Path'] = None
    df['LC_Plot_Path'] = None

    # Track counters for each TIC ID
    tic_id_counters = defaultdict(int)

    for idx, row in df.iterrows():
        fits_path = row['FITS_File_Path']
        tic_id = row['TIC_ID']  # Ensure TIC ID is available

        try:
            if not os.path.exists(fits_path):
                logging.warning(f"FITS file not found: {fits_path}")
                continue

            # Increment the counter for this TIC ID
            tic_id_counters[tic_id] += 1
            counter = tic_id_counters[tic_id]

            with fits.open(fits_path) as hdul:
                # Extract light curve data
                light_curve_data = hdul[1].data
                time = light_curve_data['TIME']
                flux = light_curve_data['PDCSAP_FLUX']

                # Save light curve data to CSV
                lc_csv_path = os.path.join(lc_ext_dir, f"{tic_id}_{counter}_LC.csv")
                pd.DataFrame({'TIME': time, 'FLUX': flux}).to_csv(lc_csv_path, index=False)
                df.at[idx, 'LC_Path'] = lc_csv_path
                logging.info(f"Saved light curve data to: {lc_csv_path}")

                # Extract aperture image
                aperture_mask = hdul[2].data
                ap_raw_path = os.path.join(ap_ext_dir, f"{tic_id}_{counter}_AP.npy")
                np.save(ap_raw_path, aperture_mask)
                df.at[idx, 'AP_Path'] = ap_raw_path
                logging.info(f"Saved aperture mask to: {ap_raw_path}")

                # Save aperture mask as PNG
                ap_png_path = os.path.join(ap_img_dir, f"{tic_id}_{counter}_AP.png")
                plt.figure(figsize=(5, 5))
                plt.imshow(aperture_mask, cmap='gray')
                plt.colorbar(label='Flux')
                plt.title(f"Aperture Mask: {tic_id} (Instance {counter})")
                plt.axis('off')
                plt.savefig(ap_png_path, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved aperture mask PNG to: {ap_png_path}")

                # Plot and save light curve
                lc_plot_path = os.path.join(lc_plot_dir, f"{tic_id}_{counter}_LCPlot.png")
                plt.figure(figsize=(10, 5))
                plt.plot(time, flux, label='Light Curve', color='blue')
                plt.xlabel('Time (days)')
                plt.ylabel('Flux')
                plt.title(f"Light Curve: {tic_id} (Instance {counter})")
                plt.legend()
                plt.savefig(lc_plot_path)
                plt.close()
                df.at[idx, 'LC_Plot_Path'] = lc_plot_path
                logging.info(f"Saved light curve plot to: {lc_plot_path}")

        except Exception as e:
            logging.error(f"Error processing FITS file {fits_path}: {e}")

    # Save the updated index CSV
    updated_csv_path = "data/negative_tic_extracts.csv"
    df.to_csv(updated_csv_path, index=False)
    logging.info(f"Updated index CSV saved to: {updated_csv_path}")


# Example usage
index_csv = "data/negative_tics.csv"  # Index CSV built earlier
extract_folder = "/run/media/maverick/X10 Pro/exoplanetDatasetReindexedFinal/negative_extract"  # Target folder for extracted data

os.makedirs(extract_folder, exist_ok=True)
extract_fits_data(index_csv, extract_folder)
