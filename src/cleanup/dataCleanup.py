import pandas as pd
import os

# Define file paths
csv1_path = 'data/true_positives_extract/extracted_LC_data_20241112_152509.csv'  # Replace with the path to your first CSV
csv2_path = 'data/csv-file-toi-catalog.csv'
# Load the CSV files
csv1 = pd.read_csv(csv1_path)
csv2 = pd.read_csv(csv2_path, skiprows=4)  # Skip the first 4 comment lines
csv2['TIC'] = csv2['TIC'].astype(str)
# Define positive and negative dispositions
positive_dispositions = {'PC', 'CP', 'APC', 'KP'}  # Includes KP as positive
negative_dispositions = {'EB', 'FP', 'FA', 'IS', 'V', 'O'}  # Negative dispositions

# Process the first CSV
rows_to_remove = []

for index, row in csv1.iterrows():
    toi_id = str(row['tic_id'])  # Ensure TOI ID is compared as a string

    # Filter rows in the second CSV matching the TOI ID
    match = csv2[csv2['TIC'].str.contains(toi_id, na=False)]

    # Check for any negative dispositions in the matching rows
    has_negative = not match[
        (match['TOI Disposition'].isin(positive_dispositions)) |
        (match['EXOFOP Disposition'].isin(positive_dispositions))
    ].empty

    # If any negative disposition exists, treat as a false positive
    if has_negative:
        file_paths = [
            row['file_path'],
            row['lc_csv_path'],
            row['lcap_csv_path'],
            row['lcplot_path'],
            row['lcapimg_path']
        ]
        
        # Delete the files
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Mark the row for removal
        rows_to_remove.append(index)

# Remove rows from the first CSV
csv1.drop(rows_to_remove, inplace=True)

# Save the cleaned-up first CSV
cleaned_csv1_path = 'data/true_positives_extract/extracted_LC_data_20241112_152509.csv' 
csv1.to_csv(cleaned_csv1_path, index=False)

print(f"Cleaned CSV saved to {cleaned_csv1_path}")
