import os
import pandas as pd
import logging
from datetime import datetime
from astroquery.mast import Observations
import argparse
import time
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch TESS TOI TP files with additional downloads up to 5 files.")
parser.add_argument("csv_file", type=str, help="Path to the input CSV file with TOI data")
parser.add_argument("--delay", type=float, help="Delay between downloads (in seconds)", default=1.0)
args = parser.parse_args()

csv_file = args.csv_file
delay_between_downloads = args.delay

# Setup logging
log_dir = 'logs/data_fetch/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'tp_fits_data_fetch_log_{timestamp}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("TP Data Fetch Initiated......")

# Output CSV for tracking downloaded TP files
output_csv = 'data/rawData/tp_fits_file_index.csv'
tp_dir = 'data/rawData/TP/'
os.makedirs(tp_dir, exist_ok=True)

# Load the input CSV and get a list of distinct TOI IDs with matching obs_ids
df = pd.read_csv(csv_file)
distinct_toi_ids = df['tic_id'].unique()
#print(distinct_toi_ids)
# Modification to get from specified last toi id
last_toi_id = 381976956

for i in range(0, len(distinct_toi_ids)):
    if distinct_toi_ids[i] == last_toi_id:
        index = i
        #print(i)

distinct_toi_ids = distinct_toi_ids[index:]
#print(distinct_toi_ids)

def save_to_csv(row_dict):
    """Append data to TP index file."""
    if not os.path.exists(output_csv):
        pd.DataFrame([row_dict]).to_csv(output_csv, mode='w', index=False)
    else:
        pd.DataFrame([row_dict]).to_csv(output_csv, mode='a', header=False, index=False)

# Main loop to fetch TP files for each distinct TOI ID
for toi_id in distinct_toi_ids:
    try:
        logging.info(f"Querying TP files for TOI: {toi_id}")

        # Primary match: Find TP files matching obs_id entries from the CSV
        matched_obs_ids = df[df['tic_id'] == toi_id]['obs_id'].unique()
        tp_products = Observations.query_criteria(target_name=toi_id, project="TESS", dataproduct_type="timeseries")
        tp_files = Observations.get_product_list(tp_products)
        
        # Convert relevant column(s) to standard Series before filtering
        tp_files_df = tp_files.to_pandas()  # Convert entire table to DataFrame
        tp_files_df['parent_obsid'] = tp_files_df['parent_obsid'].astype(str)

        # Filter TP files that match the obs_ids and are of '_tp.fits' type
        matched_tp_files = tp_files_df[(tp_files_df['parent_obsid'].isin(matched_obs_ids.astype(str))) &
                                       (tp_files_df['productFilename'].str.contains('_tp.fits'))]

        # Download matched files
        download_count = 0
        for _, row in matched_tp_files.iterrows():
            if download_count >= 5:
                break
            file_url = row['dataURI']
            filename = row['productFilename']
            local_path = os.path.join(tp_dir, filename)
            download_result = Observations.download_file(file_url, local_path=local_path)

            if download_result[0] == 'COMPLETE':
                save_to_csv({
                    'tic_id': toi_id,
                    'obs_id': row['parent_obsid'],
                    'file_type': 'TP',
                    'file_path': local_path
                })
                logging.info(f"Downloaded TP file: {filename} for TOI: {toi_id}")
                download_count += 1
            else:
                logging.error(f"Failed to download TP file for TOI: {toi_id}")
            time.sleep(delay_between_downloads)

        # If fewer than 5 files, get additional TP files until count reaches 5
        if download_count < 5:
            remaining_tp_files = tp_files_df[(~tp_files_df['parent_obsid'].isin(matched_obs_ids.astype(str))) & 
                                             (tp_files_df['productFilename'].str.contains('_tp.fits'))]
            for _, row in remaining_tp_files.iterrows():
                if download_count >= 5:
                    break
                file_url = row['dataURI']
                filename = row['productFilename']
                local_path = os.path.join(tp_dir, filename)
                download_result = Observations.download_file(file_url, local_path=local_path)

                if download_result[0] == 'COMPLETE':
                    save_to_csv({
                        'tic_id': toi_id,
                        'obs_id': row['parent_obsid'],
                        'file_type': 'TP',
                        'file_path': local_path
                    })
                    logging.info(f"Downloaded additional TP file: {filename} for TOI: {toi_id}")
                    download_count += 1
                else:
                    logging.error(f"Failed to download additional TP file for TOI: {toi_id}")
                time.sleep(delay_between_downloads)

    except Exception as e:
        logging.error(f"Error processing TP files for TOI: {toi_id}, Error: {str(e)}")
        continue

logging.info("TP Data Fetch Completed......")
