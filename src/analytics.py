import pandas as pd
import matplotlib.pyplot as plt

# Load the first CSV file (e.g., true positives)
file_path_1 = 'data/true_positive_matched_observations_lc_data.csv'
data_1 = pd.read_csv(file_path_1)

# Load the second CSV file (e.g., negatives)
file_path_2 = 'data/negative_matched_observations_lc_data.csv'
data_2 = pd.read_csv(file_path_2)

# Scatter plot with different colors
plt.figure(figsize=(10, 6))

# Plot the first dataset in one color
plt.scatter(data_1['ra'], data_1['dec'], s=10, alpha=0.7, label='True Positives', color='blue')

# Plot the second dataset in another color
plt.scatter(data_2['ra'], data_2['dec'], s=10, alpha=0.7, label='Negatives', color='red')

# Add plot details
plt.title("RA vs Dec Scatter Plot")
plt.xlabel("Right Ascension (RA)")
plt.ylabel("Declination (Dec)")
plt.legend()  # Show the legend to differentiate datasets
plt.grid(True)
plt.show()
