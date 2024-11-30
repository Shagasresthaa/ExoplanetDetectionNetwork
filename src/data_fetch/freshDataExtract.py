import pandas as pd

# Load the CSV, skipping the first 4 rows
csv2 = pd.read_csv('data/csv-file-toi-catalog.csv', skiprows=4)

# Define positive and negative dispositions
positive_dispositions = {'PC', 'CP', 'APC', 'KP'}  # Valid positives
negative_dispositions = {'EB', 'FP', 'FA', 'IS', 'V', 'O'}  # Negatives

# Helper function to classify rows
def classify_row(row):
    exofop_disposition = row['EXOFOP Disposition']
    toi_disposition = row['TOI Disposition']
    
    # If EXOFOP Disposition is False Positive, classify as negative
    if exofop_disposition == 'FP':
        return 'negative'
    # If EXOFOP Disposition is empty, classify as positive
    elif pd.isna(exofop_disposition):
        return 'positive'
    # Otherwise, check TOI Disposition and EXOFOP Disposition against positive list
    elif toi_disposition in positive_dispositions or exofop_disposition in positive_dispositions:
        return 'positive'
    # If none match, classify as negative
    else:
        return 'negative'

# Apply classification to each row
csv2['classification'] = csv2.apply(classify_row, axis=1)

# Separate into positives and negatives based on classification
positives = csv2[csv2['classification'] == 'positive'].drop(columns=['classification'])
negatives = csv2[csv2['classification'] == 'negative'].drop(columns=['classification'])

# Define output file paths
positives_csv_path = 'data/positives.csv'
negatives_csv_path = 'data/negatives.csv'

# Save the positive and negative entries to separate CSV files
positives.to_csv(positives_csv_path, index=False)
negatives.to_csv(negatives_csv_path, index=False)

print(f"Positive entries saved to: {positives_csv_path}")
print(f"Negative entries saved to: {negatives_csv_path}")
