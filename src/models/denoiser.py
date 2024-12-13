import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Ensure directories exist
SAVE_DIR = "model_saves/autoencoder/"
LOG_DIR = "logs/training/"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "autoencoder_training.log"),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Set GPU to exclusive mode
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    logging.info("Using GPU: Set to exclusive mode.")

# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, sequence_length):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # (16, seq_len/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # (32, seq_len/4)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # (64, seq_len/8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # (32, seq_len/4)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # (16, seq_len/2)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # (1, seq_len)
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Dataset Class for Light Curves
class LightCurveDataset(Dataset):
    def __init__(self, index_csv, mode='train', validation_split=0.2, noise_std=0.1, sequence_length=200):
        self.data = pd.read_csv(index_csv)
        self.noise_std = noise_std
        self.sequence_length = sequence_length

        # Split into training and validation sets
        split_idx = int((1 - validation_split) * len(self.data))
        if mode == 'train':
            self.data = self.data.iloc[:split_idx]
        elif mode == 'val':
            self.data = self.data.iloc[split_idx:]
        else:
            raise ValueError("Mode must be either 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        lc_path = row['LC_Path']
        lc_data = pd.read_csv(lc_path).dropna()  # Drop NaN values

        # Handle empty light curves
        if lc_data.empty or 'FLUX' not in lc_data.columns or lc_data['FLUX'].isna().all():
            logging.warning(f"Empty or invalid light curve at index {idx}, file: {lc_path}")
            return None, None  # Skip this entry

        flux = lc_data['FLUX'].values

        # Handle empty or invalid flux array
        if len(flux) == 0:
            logging.warning(f"Flux data is empty at index {idx}, file: {lc_path}")
            return None, None

        # Normalize flux to range [-1, 1]
        if np.max(flux) == np.min(flux):
            logging.warning(f"Flux data has no variation at index {idx}, file: {lc_path}")
            return None, None

        flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux)) * 2 - 1

        # Enforce fixed sequence length
        if len(flux) > self.sequence_length:
            flux = flux[:self.sequence_length]  # Trim to the required length
        elif len(flux) < self.sequence_length:
            flux = np.pad(flux, (0, self.sequence_length - len(flux)), 'constant')  # Pad with zeros

        # Add Gaussian noise to create a noisy version
        noisy_flux = flux + np.random.normal(0, self.noise_std, size=flux.shape)

        return torch.tensor(noisy_flux, dtype=torch.float32).unsqueeze(0), \
               torch.tensor(flux, dtype=torch.float32).unsqueeze(0)  # Clean flux as target


# Custom collate function to handle invalid entries
def custom_collate_fn(batch):
    """
    Custom collate function to handle empty light curves.
    """
    batch = [item for item in batch if item[0] is not None]  # Filter out invalid entries
    if len(batch) == 0:
        return None  # Handle case where entire batch is empty
    return torch.utils.data.dataloader.default_collate(batch)


# Training Function
def train_autoencoder(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                val_loss += loss.item()

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - "
                     f"Train Loss: {train_loss / len(train_loader):.4f} - "
                     f"Val Loss: {val_loss / len(val_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model weights saved to: {save_path}")


# Main Script
if __name__ == "__main__":
    # Hyperparameters
    index_csv = "data/preprocessed/datasetIndex.csv" 
    sequence_length = 6000
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001
    noise_std = 0.2
    save_path = os.path.join(SAVE_DIR, "denoiser_autoencoder.pth")
    device = "cuda"

    logging.info(f"Starting training: sequence_length={sequence_length}, batch_size={batch_size}, "
                 f"num_epochs={num_epochs}, learning_rate={learning_rate}, noise_std={noise_std}")

    # Create training and validation datasets and data loaders
    train_dataset = LightCurveDataset(index_csv, mode='train', validation_split=0.2, noise_std=noise_std, sequence_length=sequence_length)
    val_dataset = LightCurveDataset(index_csv, mode='val', validation_split=0.2, noise_std=noise_std, sequence_length=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize the autoencoder
    model = Autoencoder(sequence_length)

    # Train the autoencoder
    train_autoencoder(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path)
