import pandas as pd

def merge_and_shuffle_csvs(positive_csv, negative_csv, output_csv, random_state=42):
    # Load positive and negative CSVs
    positive_df = pd.read_csv(positive_csv)
    negative_df = pd.read_csv(negative_csv)

    # Add Label column
    positive_df['Label'] = 1  # 1 for positive
    negative_df['Label'] = 0  # 0 for negative

    # Merge the dataframes
    merged_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Shuffle the rows
    shuffled_df = merged_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Save the shuffled CSV
    shuffled_df.to_csv(output_csv, index=False)
    print(f"Merged and shuffled CSV saved to: {output_csv}")


# Example usage
positive_csv = "data/positive_tic_extracts.csv"
negative_csv = "data/negative_tic_extracts.csv"
output_csv = "data/preprocessed/datasetIndex.csv"

merge_and_shuffle_csvs(positive_csv, negative_csv, output_csv)
