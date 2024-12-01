import os
import numpy as np
import matplotlib.pyplot as plt

def convert_npy_to_png(npy_folder, png_folder):
    """
    Convert .npy aperture mask files to PNG images.

    Parameters:
        npy_folder (str): Path to the folder containing .npy files.
        png_folder (str): Path to the folder where PNG images will be saved.
    """
    # Ensure the PNG folder exists
    os.makedirs(png_folder, exist_ok=True)

    # Get list of all .npy files in the folder
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

    print(f"Found {len(npy_files)} .npy files in {npy_folder}")

    for npy_file in npy_files:
        npy_path = os.path.join(npy_folder, npy_file)
        try:
            # Load the .npy file
            aperture_mask = np.load(npy_path)

            # Save as PNG
            png_path = os.path.join(png_folder, npy_file.replace('.npy', '.png'))
            plt.figure(figsize=(5, 5))
            plt.imshow(aperture_mask, cmap='gray')
            plt.colorbar(label='Flux')
            plt.title(f"Aperture Mask: {npy_file}")
            plt.axis('off')
            plt.savefig(png_path, bbox_inches='tight')
            plt.close()

            print(f"Converted {npy_file} to {png_path}")
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")

# Example usage
npy_folder = "/run/media/maverick/X10 Pro/exoplanetDatasetReindexedFinal/positive/positive_extract/AP_EXT"  # Folder with .npy files
png_folder = "/run/media/maverick/X10 Pro/exoplanetDatasetReindexedFinal/positive/positive_extract/AP_EXT_images"  # Folder to save PNG images

convert_npy_to_png(npy_folder, png_folder)
