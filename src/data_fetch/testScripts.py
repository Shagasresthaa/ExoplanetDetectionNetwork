from astroquery.mast import Observations

# Replace this with the actual dataURL from your CSV
data_url = "mast:TESS/product/tess2020186164531-s0027-0000000231663901-0189-s_lc.fits"

# Download the file using the dataURL
file_path = Observations.download_file(data_url)

# Check and print the downloaded file path
if file_path:
    print(f"File downloaded to: {file_path}")
else:
    print("Download failed or no file was downloaded.")
