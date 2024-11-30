import os
import pandas as pd
from astropy.io import fits

# Define positive and negative dispositions
positive_dispositions = {'PC', 'CP', 'APC', 'KP'}  # Valid positives
negative_dispositions = {'EB', 'FP', 'FA', 'IS', 'V', 'O'}  # Negatives

def classify_tic_ids_with_dispositions(fits_dir1, fits_dir2, csv_path, positive_file, negative_file):
    """
    Classify TIC IDs from FITS files based on TOI and EXOFOP dispositions, ensuring unique entries,
    and count the number of FITS files processed for each category from two directories.

    Parameters:
        fits_dir1 (str): Path to the first directory containing FITS files.
        fits_dir2 (str): Path to the second directory containing FITS files.
        csv_path (str): Path to the CSV file with TIC IDs and dispositions.
        positive_file (str): Path to save TIC IDs with positive dispositions.
        negative_file (str): Path to save TIC IDs with negative dispositions.
    """
    # Load the CSV file and skip the first 4 rows (comments)
    df = pd.read_csv(csv_path, skiprows=4)

    positive_tics = set()  # Use sets for uniqueness
    negative_tics = set()
    unmatched_tics = []  # Store unmatched TICs with details
    positive_file_count = 0
    negative_file_count = 0

    # Combine FITS files from both directories
    fits_files = [
        os.path.join(fits_dir1, f) for f in os.listdir(fits_dir1) if f.endswith('.fits')
    ] + [
        os.path.join(fits_dir2, f) for f in os.listdir(fits_dir2) if f.endswith('.fits')
    ]

    total_files = len(fits_files)

    for file_path in fits_files:
        try:
            # Open the FITS file and extract the TIC ID
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                tic_id = str(header.get('TICID', None))  # Ensure TIC ID is a string

                if tic_id and tic_id in df['TIC'].astype(str).values:
                    # Fetch the corresponding dispositions
                    row = df.loc[df['TIC'].astype(str) == tic_id].iloc[0]
                    toi_disposition = row['TOI Disposition']
                    exofop_disposition = row['EXOFOP Disposition']

                    # Updated classification logic
                    if toi_disposition == 'EB' and exofop_disposition == 'APC':
                        positive_tics.add(tic_id)
                        positive_file_count += 1
                    elif toi_disposition == 'EB' and exofop_disposition == 'PC':
                        positive_tics.add(tic_id)
                        positive_file_count += 1
                    elif toi_disposition == 'EB' and exofop_disposition in negative_dispositions:
                        negative_tics.add(tic_id)
                        negative_file_count += 1
                    elif toi_disposition == 'PC' and (pd.isna(exofop_disposition) or exofop_disposition in positive_dispositions):
                        positive_tics.add(tic_id)
                        positive_file_count += 1
                    elif exofop_disposition in negative_dispositions:
                        negative_tics.add(tic_id)
                        negative_file_count += 1
                    elif toi_disposition in positive_dispositions:
                        positive_tics.add(tic_id)
                        positive_file_count += 1
                    else:
                        unmatched_tics.append({
                            "TIC_ID": tic_id,
                            "TOI_Disposition": toi_disposition,
                            "EXOFOP_Disposition": exofop_disposition
                        })
                else:
                    print(f"TIC ID {tic_id} not found in CSV.")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Save unique results to files
    pd.DataFrame(sorted(positive_tics), columns=['TIC']).to_csv(positive_file, index=False, header=False)
    pd.DataFrame(sorted(negative_tics), columns=['TIC']).to_csv(negative_file, index=False, header=False)

    # Print unmatched TICs
    if unmatched_tics:
        print(f"\nUnmatched TICs ({len(unmatched_tics)}):")
        for tic in unmatched_tics:
            print(f"TIC ID: {tic['TIC_ID']}, TOI Disposition: {tic['TOI_Disposition']}, EXOFOP Disposition: {tic['EXOFOP_Disposition']}")
    else:
        print("\nAll TICs matched classification criteria.")

    # Print the statistics
    print(f"\nTotal FITS files processed: {total_files}")
    print(f"Positive TICs count: {len(positive_tics)}")
    print(f"Total FITS files for positive TICs: {positive_file_count}")
    print(f"Negative TICs count: {len(negative_tics)}")
    print(f"Total FITS files for negative TICs: {negative_file_count}")
    print(f"Positive TICs saved to {positive_file}")
    print(f"Negative TICs saved to {negative_file}")

# Example usage
fits_directory1 = "/run/media/maverick/X10 Pro/exoplanetDataset/truePositivesRaw/LC"  # Replace with your first FITS directory path
fits_directory2 = "/run/media/maverick/X10 Pro/exoplanetDataset/falsePositivesRaw/LC"  # Replace with your second FITS directory path
csv_file_path = "data/csv-file-toi-catalog.csv"  # Replace with your CSV file path
positive_output_file = "data/positive_tics.txt"  # File to save positive TICs
negative_output_file = "data/negative_tics.txt"  # File to save negative TICs

classify_tic_ids_with_dispositions(fits_directory1, fits_directory2, csv_file_path, positive_output_file, negative_output_file)
