import os
import shutil
import logging
import pandas as pd
from astropy.io import fits

# Set up logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "classification.log"),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Define positive and negative dispositions
positive_dispositions = {'PC', 'CP', 'APC', 'KP'}  # Valid positives
negative_dispositions = {'EB', 'FP', 'FA', 'IS', 'V', 'O'}  # Negatives

def classify_tic_ids_with_dispositions(fits_dir1, fits_dir2, csv_path, positive_dir, negative_dir, positive_csv, negative_csv):
    logging.info("Starting TIC classification process.")

    # Ensure output directories exist
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)

    # Load the CSV file and skip the first 4 rows (comments)
    try:
        df = pd.read_csv(csv_path, skiprows=4)
        logging.info(f"Loaded CSV file: {csv_path}")
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return

    # Prepare output DataFrames
    positive_data = []
    negative_data = []

    # Combine FITS files from both directories
    fits_files = [
        os.path.join(fits_dir1, f) for f in os.listdir(fits_dir1) if f.endswith('.fits')
    ] + [
        os.path.join(fits_dir2, f) for f in os.listdir(fits_dir2) if f.endswith('.fits')
    ]

    total_files = len(fits_files)
    logging.info(f"Total FITS files to process: {total_files}")

    for file_path in fits_files:
        try:
            # Open the FITS file and extract the TIC ID
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                tic_id = str(header.get('TICID', None))  # Ensure TIC ID is a string

                if tic_id and tic_id in df['TIC'].astype(str).values:
                    # Fetch the corresponding dispositions and other data
                    row = df.loc[df['TIC'].astype(str) == tic_id].iloc[0]
                    toi_disposition = row['TOI Disposition']
                    exofop_disposition = row['EXOFOP Disposition']

                    # Extract required columns
                    ra = row['TIC Right Ascension']
                    dec = row['TIC Declination']
                    tmag = row['TMag Value']
                    tmag_unc = row['TMag Uncertainty']
                    vmag = row['VMag Value']
                    vmag_unc = row['VMag Uncertainty']
                    epoch = row['Epoch Value']
                    epoch_err = row['Epoch Error']
                    period = row['Orbital Period Value']
                    duration = row['Transit Duration Value']
                    depth = row['Transit Depth Value']

                    # Classification logic
                    if toi_disposition == 'EB' and exofop_disposition in {'APC', 'PC'}:
                        destination = os.path.join(positive_dir, os.path.basename(file_path))
                        shutil.copy(file_path, destination)
                        positive_data.append([tic_id, ra, dec, tmag, tmag_unc, vmag, vmag_unc, epoch, epoch_err, period, duration, depth, destination])
                        logging.debug(f"TIC {tic_id}: EB with APC/PC -> Positive")
                    elif toi_disposition == 'EB' and exofop_disposition in negative_dispositions:
                        destination = os.path.join(negative_dir, os.path.basename(file_path))
                        shutil.copy(file_path, destination)
                        negative_data.append([tic_id, ra, dec, tmag, tmag_unc, vmag, vmag_unc, epoch, epoch_err, period, duration, depth, destination])
                        logging.debug(f"TIC {tic_id}: EB with Negative EXOFOP -> Negative")
                    elif toi_disposition == 'PC' and (pd.isna(exofop_disposition) or exofop_disposition in positive_dispositions):
                        destination = os.path.join(positive_dir, os.path.basename(file_path))
                        shutil.copy(file_path, destination)
                        positive_data.append([tic_id, ra, dec, tmag, tmag_unc, vmag, vmag_unc, epoch, epoch_err, period, duration, depth, destination])
                        logging.debug(f"TIC {tic_id}: PC with Positive EXOFOP -> Positive")
                    elif exofop_disposition in negative_dispositions:
                        destination = os.path.join(negative_dir, os.path.basename(file_path))
                        shutil.copy(file_path, destination)
                        negative_data.append([tic_id, ra, dec, tmag, tmag_unc, vmag, vmag_unc, epoch, epoch_err, period, duration, depth, destination])
                        logging.debug(f"TIC {tic_id}: Negative EXOFOP -> Negative")
                    else:
                        logging.warning(f"TIC {tic_id}: Unmatched")
                else:
                    logging.warning(f"TIC ID {tic_id} not found in CSV.")
        except Exception as e:
            logging.error(f"Error reading FITS file {file_path}: {e}")

    # Write results to CSV
    positive_columns = ['TIC_ID', 'RA', 'Dec', 'TMag', 'TMag_Unc', 'VMag', 'VMag_Unc', 'Epoch', 'Epoch_Err', 'Period', 'Duration', 'Depth', 'FITS_File_Path']
    pd.DataFrame(positive_data, columns=positive_columns).to_csv(positive_csv, index=False)
    pd.DataFrame(negative_data, columns=positive_columns).to_csv(negative_csv, index=False)

    logging.info(f"Positive TICs saved to {positive_csv}")
    logging.info(f"Negative TICs saved to {negative_csv}")

# Example usage
fits_directory1 = "/run/media/maverick/X10 Pro/exoplanetDataset/truePositivesRaw/LC"  # Replace with your first FITS directory path
fits_directory2 = "/run/media/maverick/X10 Pro/exoplanetDataset/falsePositivesRaw/LC"  # Replace with your second FITS directory path
csv_file_path = "data/csv-file-toi-catalog.csv"  # Replace with your CSV file path
positive_directory = "/run/media/maverick/X10 Pro/exoplanetDatasetReindexedFinal/positive"  # Directory to store positive FITS files
negative_directory = "/run/media/maverick/X10 Pro/exoplanetDatasetReindexedFinal/negative"  # Directory to store negative FITS files
positive_csv_file = "data/positive_tics.csv"  # CSV to save positive TIC data
negative_csv_file = "data/negative_tics.csv"  # CSV to save negative TIC data

classify_tic_ids_with_dispositions(fits_directory1, fits_directory2, csv_file_path, positive_directory, negative_directory, positive_csv_file, negative_csv_file)
