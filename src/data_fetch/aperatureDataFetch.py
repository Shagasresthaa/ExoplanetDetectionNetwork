import os
import pandas as pd
import logging
from datetime import datetime
from astroquery.mast import Observations
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch TESS TOI TP files.")
parser.add_argument("csv_file", type=str, help="Path to the input CSV file with TOI data")
parser.add_argument("index_file", type=str, help="Path to the existing index CSV file")
parser.add_argument("--delay", type=float, help="Delay between downloads (in seconds)", default=1.0)
args = parser.parse_args()

csv_file = args.csv_file
index_file = args.index_file
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

# Load the input CSV and index CSV
df = pd.read_csv(csv_file)
index_df = pd.read_csv(index_file)
distinct_toi_ids = df['tic_id'].unique()

def save_to_csv(row_dict):
    """Append data to TP index file."""
    if not os.path.exists(output_csv):
        pd.DataFrame([row_dict]).to_csv(output_csv, mode='w', index=False)
    else:
        pd.DataFrame([row_dict]).to_csv(output_csv, mode='a', header=False, index=False)

# Main loop to fetch TP files for each distinct TOI ID
for toi_id in distinct_toi_ids:
    try:
        # Limit to 5 TP files per TOI ID by matching `obs_id` in the existing index
        matched_obs_ids = index_df[index_df['tic_id'] == toi_id]['obs_id'].unique()
        if len(matched_obs_ids) == 0:
            logging.warning(f"No matching obs_id found in index for TOI: {toi_id}")
            continue
        
        logging.info(f"Querying TP files for TOI: {toi_id}")
        tp_products = Observations.query_criteria(target_name=toi_id, project="TESS", dataproduct_type="timeseries")
        tp_files = Observations.get_product_list(tp_products)
        tp_files = tp_files[tp_files['obs_id'].isin(matched_obs_ids) & tp_files['productFilename'].str.contains('_tp.fits')].head(5)
        
        if tp_files is None or len(tp_files) == 0:
            logging.warning(f"No TP files found or matched for TOI: {toi_id}")
            continue

        # Download each TP file and log to the CSV
        for _, row in tp_files.iterrows():
            file_url = row['dataURI']
            filename = row['productFilename']
            local_path = os.path.join(tp_dir, filename)
            download_result = Observations.download_file(file_url, local_path=local_path)

            if download_result[0] == 'COMPLETE':
                save_to_csv({
                    'tic_id': toi_id,
                    'obs_id': row['obs_id'],
                    'file_type': 'TP',
                    'file_path': local_path,
                    'proposal_pi': row.get('proposal_pi', 'N/A'),
                    'obs_title': row.get('obs_title', 'N/A')
                })
                logging.info(f"Downloaded TP file: {filename} for TOI: {toi_id}")
            else:
                logging.error(f"Failed to download TP file for TOI: {toi_id}")

            logging.info(f"Sleeping for {delay_between_downloads} seconds to respect API limits")
            time.sleep(delay_between_downloads)

    except Exception as e:
        logging.error(f"Error processing TP files for TOI: {toi_id}, Error: {str(e)}")
        continue

logging.info("TP Data Fetch Completed......")
