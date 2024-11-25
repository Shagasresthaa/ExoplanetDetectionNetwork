import pandas as pd

# Define file paths
csv1_path = 'data/true_positives_extract/extracted_LC_data_20241112_152509.csv'  # Replace with the path to your first CSV
csv2_path = 'data/csv-file-toi-catalog.csv'

# Load the CSV files
csv1 = pd.read_csv(csv1_path)
csv2 = pd.read_csv(csv2_path, skiprows=4)  # Skip the first 4 comment lines

# Convert relevant columns to strings for consistent matching
csv1['tic_id'] = csv1['tic_id'].astype(str)
csv2['Full TOI ID'] = csv2['Full TOI ID'].astype(str)

# Define positive dispositions
positive_dispositions = {'PC', 'CP', 'APC', 'KP'}  # Valid positives

# List to track negatives
negatives = []

# Check each entry in CSV1 against CSV2
for index, row in csv1.iterrows():
    tic_id = row['tic_id']
    
    # Find matching rows in CSV2 based on TIC ID
    match = csv2[csv2['Full TOI ID'].str.contains(tic_id, na=False)]
    
    # Determine if this TIC ID has any positive disposition
    has_positive = not match[
        (match['TOI Disposition'].isin(positive_dispositions)) |
        (match['EXOFOP Disposition'].isin(positive_dispositions))
    ].empty
    
    # If no positive disposition is found, classify as negative
    if not has_positive:
        negatives.append(row)

# Create a DataFrame for negatives
negatives_df = pd.DataFrame(negatives)

# Output the report
report_path = 'negative_objects_report.csv'
negatives_df.to_csv(report_path, index=False)

print(f"Report generated: {report_path}")
print("\nList of negative objects:")
print(negatives_df)
