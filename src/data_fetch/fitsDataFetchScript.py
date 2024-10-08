import os
import pandas as pd
import logging
from datetime import datetime
from astroquery.mast import Observations
import argparse
import time
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch TESS TOI FITS files.")
parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
parser.add_argument("--limit", type=int, help="Limit the number of TOI IDs to process", default=None)
parser.add_argument("--delay", type=float, help="Delay between downloads (in seconds)", default=1.0)
args = parser.parse_args()

csv_file = args.csv_file
limit = args.limit
delay_between_downloads = args.delay

# Setup logging
log_dir = 'logs/data_fetch/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'tess_fits_data_fetch_log_{timestamp}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Data Fetch Initiated......")

# Output CSV for tracking downloaded files
output_csv = 'data/rawData/raw_fits_file_index.csv'
raw_data_dir = 'data/rawData/'
lc_dir = os.path.join(raw_data_dir, 'LC/')
llc_dir = os.path.join(raw_data_dir, 'LLC/')
dvt_dir = os.path.join(raw_data_dir, 'DVT/')
for directory in [lc_dir, llc_dir, dvt_dir]:
    os.makedirs(directory, exist_ok=True)

# File to track distinct TOI IDs and last processed TOI ID
distinct_toi_file = 'data/rawData/distinct_toi_ids.json'
last_toi_file = 'data/rawData/last_processed_toi.json'

# Load the input CSV
df = pd.read_csv(csv_file)

def save_last_processed_toi(tic_id):
    with open(last_toi_file, 'w') as f:
        json.dump({"last_toi_id": tic_id}, f)

def load_last_processed_toi():
    if os.path.exists(last_toi_file):
        with open(last_toi_file, 'r') as f:
            data = json.load(f)
            return data.get("last_toi_id")
    return None

def save_distinct_toi_ids(toi_ids):
    toi_ids = [int(tic_id) for tic_id in toi_ids]
    with open(distinct_toi_file, 'w') as f:
        json.dump(toi_ids, f)

def load_distinct_toi_ids():
    if os.path.exists(distinct_toi_file):
        with open(distinct_toi_file, 'r') as f:
            return json.load(f)
    return None

# Load the list of distinct TOI IDs
distinct_toi_ids = load_distinct_toi_ids()

# If it's a fresh run, generate the distinct TOI IDs and save them
if distinct_toi_ids is None:
    unique_toi_ids = df['tic_id'].unique()
    if limit:
        unique_toi_ids = unique_toi_ids[:limit]
    save_distinct_toi_ids(list(unique_toi_ids))
else:
    unique_toi_ids = distinct_toi_ids

# Filter the original DataFrame to only include rows for the selected distinct TOI IDs
df = df[df['tic_id'].isin(unique_toi_ids)]

# Load the last processed TOI ID and skip entries up to that ID
last_processed_toi = load_last_processed_toi()

if last_processed_toi:
    last_toi_index = unique_toi_ids.index(last_processed_toi)
    remaining_toi_ids = unique_toi_ids[last_toi_index:]  # Resume from the last TOI ID
else:
    remaining_toi_ids = unique_toi_ids  # No last TOI ID, so start from the beginning

def save_to_csv(row_dict):
    if not os.path.exists(output_csv):
        pd.DataFrame([row_dict]).to_csv(output_csv, mode='w', index=False)
    else:
        pd.DataFrame([row_dict]).to_csv(output_csv, mode='a', header=False, index=False)

# Main loop to process each TOI ID
for toi_id in remaining_toi_ids:
    toi_df = df[df['tic_id'] == toi_id]
    try:
        for _, row in toi_df.iterrows():
            tic_id = row['tic_id']
            ra = row['ra']
            dec = row['dec']
            calib_level = row['calib_level']
            t_min = row['t_min']
            t_max = row['t_max']
            logging.info(f"Processing TOI: {tic_id}, RA: {ra}, DEC: {dec}")

            file_url = row['dataURL']
            if '_lc.fits' in file_url:
                file_type = 'LC'
                save_dir = lc_dir
            elif '_llc.fits' in file_url:
                file_type = 'LLC'
                save_dir = llc_dir
            elif '_dvt.fits' in file_url:
                file_type = 'DVT'
                save_dir = dvt_dir
            else:
                continue

            # Set the local filename for saving the file
            filename = file_url.split('/')[-1]
            local_path = os.path.join(save_dir, filename)
            download_result = Observations.download_file(file_url, local_path=local_path)

            if download_result[0] == 'COMPLETE':
                save_to_csv({
                    'tic_id': tic_id,
                    'obs_id': row['obs_id'],
                    'ra': ra,
                    'dec': dec,
                    'file_type': file_type,
                    'calib_level': calib_level,
                    't_min': t_min,
                    't_max': t_max,
                    'file_path': local_path,
                    'proposal_pi': row['proposal_pi'],
                    'obs_title': row['obs_title']
                })
                logging.info(f"Downloaded {file_type} file: {filename} for TOI: {tic_id}")
            else:
                logging.error(f"Failed to download {file_type} file for TOI: {tic_id}")
            
            logging.info(f"Sleeping for {delay_between_downloads} seconds to respect API limits")
            time.sleep(delay_between_downloads)

        # Save the last processed TOI ID after each successful TOI ID
        save_last_processed_toi(toi_id)

    except Exception as e:
        logging.error(f"Error processing TOI: {toi_id}, Error: {str(e)}")
        continue

logging.info("Data Fetch Completed......")
