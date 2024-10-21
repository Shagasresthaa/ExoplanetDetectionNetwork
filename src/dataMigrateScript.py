import os
import shutil
import logging
import time
from datetime import datetime

# Configure logging
log_dir = 'logs/data_migration/'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'data_migration_log_{timestamp}.log')

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# List of file types to exclude
excluded_files = ['raw_fits_file_index.csv', 'tp_fits_file_index.csv', 'last_processed_index.txt', 'distinct_toi_ids.json', 'last_processed_toi.json']

# Function to move files from source to destination, maintaining the folder structure
def move_data(source_dir, dest_dir, delay_between_moves=180):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file is in the excluded list
            if any(file.endswith(excluded) for excluded in excluded_files):
                continue

            source_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, source_dir)
            dest_file_dir = os.path.join(dest_dir, relative_path)
            dest_file_path = os.path.join(dest_file_dir, file)

            os.makedirs(dest_file_dir, exist_ok=True)  # Ensure subdirectories are created

            try:
                shutil.move(source_file_path, dest_file_path)  # Move file to destination

                # Log and print file movement
                log_message = f"Moved: {source_file_path} -> {dest_file_path}"
                logging.info(log_message)
                print(log_message)

                # # Sleep between moves to avoid overwhelming the system
                # logging.info(f"Sleeping for {delay_between_moves} seconds...")
                # print(f"Sleeping for {delay_between_moves} seconds...")
                # time.sleep(delay_between_moves)
            except Exception as e:
                error_message = f"Error moving file: {source_file_path}. Error: {str(e)}"
                logging.error(error_message)
                print(error_message)

if __name__ == "__main__":
    # Specify the source directory and destination directory
    source_directory = 'data/rawData/'  # Local folder where your data is stored
    destination_directory = '/run/media/maverick/X10 Pro/data_bkp/dataset'  # External SSD path

    # Start moving data with a delay of 180 seconds between moves
    move_data(source_directory, destination_directory, delay_between_moves=180)
