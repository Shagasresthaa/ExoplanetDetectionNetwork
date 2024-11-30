from astroquery.mast import Catalogs, Observations
import pandas as pd
import os
import logging
from datetime import datetime

# Logging setup for run logs
log_dir = 'logs/data_fetch'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'query_log_{timestamp}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("False Positive Metadata fetch initiated.......")

# File paths
input_file = "data/negatives.csv"
output_file = 'data/negative_matched_observations_lc_data.csv'

# Read input CSV (skipping first 4 rows as specified)
toi_list = pd.read_csv(input_file)

# Extract TIC IDs
tic_ids = toi_list['TIC'].tolist()

# Create the matched observations file and write headers initially
if not os.path.exists(output_file):
    pd.DataFrame(columns=[
        'tic_id', 'obs_id', 'ra', 'dec', 'calib_level', 
        't_min', 't_max', 'dataURL', 'obs_title', 'proposal_pi'
    ]).to_csv(output_file, index=False)

# Loop over TIC IDs and fetch data
for tic_id in tic_ids[2501:3000]:
    logging.info(f"Fetching data for TIC ID: {tic_id}")
    
    try:
        # Query TIC catalog
        tic_catalog_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
        
        if len(tic_catalog_data) > 0:
            tic_obj = tic_catalog_data[0]
            ra = tic_obj['ra']  # Right Ascension
            dec = tic_obj['dec']  # Declination
            logging.info(f"TIC ID: {tic_id} - RA: {ra}, Dec: {dec}")

            # Query MAST for light curve observations
            obs_table = Observations.query_criteria(
                coordinates=f"{ra} {dec}",
                radius="0.02 deg",
                dataproduct_type="timeseries",
                obs_collection="TESS",
                provenance_name=["LC", "SPOC", "QLP"],
                calib_level=[2, 3, 4]
            )
            obs_df = obs_table.to_pandas()

            # Save full observation data for reference
            obs_df.to_csv(f'data/falsePositivesObservationsData/observations_table_{tic_id}.csv', index=False)
            logging.info(f"Saved observation data for TIC ID {tic_id} to CSV.")

            # Filter for light curve files
            filtered_obs = obs_df[
                obs_df['dataURL'].str.endswith(('_lc.fits', '_llc.fits', '_dvt.fits'))
            ]

            if not filtered_obs.empty:
                logging.info(f"Found light curve data for TIC ID {tic_id}, saving filtered observations.")
                matched_observations = []

                for _, obs in filtered_obs.iterrows():
                    matched_observations.append({
                        'tic_id': tic_id,
                        'obs_id': obs['obs_id'],
                        'ra': ra,
                        'dec': dec,
                        'calib_level': obs['calib_level'],
                        't_min': obs['t_min'],
                        't_max': obs['t_max'],
                        'dataURL': obs['dataURL'],
                        'obs_title': obs['obs_title'],
                        'proposal_pi': obs['proposal_pi']
                    })

                # Append matched observations to output CSV
                pd.DataFrame(matched_observations).to_csv(output_file, mode='a', header=False, index=False)
                logging.info(f"Appended data for TIC ID {tic_id} to '{output_file}'.")
        else:
            logging.warning(f"No data found for TIC ID {tic_id}.")
    except Exception as e:
        logging.error(f"Error fetching data for TIC ID {tic_id}: {e}")

logging.info("False Positive Metadata fetch completed.")
