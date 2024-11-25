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
log_filename = os.path.join(log_dir, f'false_positives_query_log_{timestamp}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("False Positive Candidate Metadata fetch Initiated.......")

fp = "data/exofop_tess_tois.csv"

toi_list = pd.read_csv(fp, skiprows=1)

# Filter for False Positives in TESS Disposition and TFOPWG Disposition
toi_list = toi_list[
    (toi_list['TESS Disposition'].isin(['EB', 'V'])) | 
    (toi_list['TFOPWG Disposition'].isin(['FP', 'FA']))
]

tic_ids = toi_list['TIC ID'].tolist()

#print(len(toi_list))

# Create the matched observations file and write headers initially
output_file = 'data/false_positive_matched_observations_lc_data.csv'
if not os.path.exists(output_file):
    pd.DataFrame(columns=['tic_id', 'obs_id', 'ra', 'dec', 'calib_level', 't_min', 't_max', 'dataURL', 'obs_title', 'proposal_pi']).to_csv(output_file, index=False)

for tic_id in tic_ids[396:]:
    logging.info(f"Fetching data for TIC ID: {tic_id}")
    
    try:
        # First, query the TESS Input Catalog (TIC) to get information for this TIC ID
        tic_catalog_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
        
        # If the TIC object exists in the catalog, get the coordinates (RA, Dec)
        if len(tic_catalog_data) > 0:
            tic_obj = tic_catalog_data[0]
            ra = tic_obj['ra']  # Right Ascension
            dec = tic_obj['dec']  # Declination

            logging.info(f"TIC ID: {tic_id} - RA: {ra}, Dec: {dec}")

            # Query MAST for light curve observations at the coordinates (RA, Dec)
            obs_table = Observations.query_criteria(
                coordinates=f"{ra} {dec}",
                radius="0.02 deg",
                dataproduct_type="timeseries",
                obs_collection="TESS",
                provenance_name=["LC", "SPOC", "QLP"],
                calib_level=[2, 3, 4]
            )
            obs_df = obs_table.to_pandas()

            # Save the full observation DataFrame to CSV
            obs_df.to_csv(f'data/falsePositivesObservationsData/observations_table_{tic_id}.csv', index=False)
            logging.info(f"Saved observation data for TIC ID {tic_id} to CSV at path data/falsePositivesObservationsData/observations_table_{tic_id}.csv.")

            # Filter for rows where 'dataURL' ends with '_lc.fits' or '_llc.fits' or '_dvt.fits' (Light curve or Long Light Curve Data Files or Aperature Data Files)
            filtered_obs = obs_df[
                obs_df['dataURL'].str.endswith(('_lc.fits', '_llc.fits', '_dvt.fits'))
            ]

            if len(filtered_obs) > 0:
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

                # Convert matched_observations to a DataFrame and append to CSV file
                pd.DataFrame(matched_observations).to_csv(output_file, mode='a', header=False, index=False)
                logging.info(f"Appended data for TIC ID {tic_id} to '{output_file}'.")

        else:
            logging.warning(f"No data found for TIC ID {tic_id}.")
    except Exception as e:
        logging.error(f"Error fetching data for TIC ID {tic_id}: {e}")

logging.info("False Positives Candidate Metadata fetch completed.......")