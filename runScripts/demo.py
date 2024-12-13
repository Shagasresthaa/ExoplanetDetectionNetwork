import torch
import numpy as np
from astropy.io import fits
import sys

# Define the autoencoder architecture
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the two-view model architecture
class TwoViewModel(torch.nn.Module):
    def __init__(self):
        super(TwoViewModel, self).__init__()
        self.lc_branch = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 3000, 128),  # Adjust based on seq_length/2
        )
        self.ap_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 64, 128),  # Assuming aperture is 64x64
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),  # Adjust based on number of classes
        )

    def forward(self, lc, ap):
        lc_features = self.lc_branch(lc)
        ap_features = self.ap_branch(ap)
        combined = torch.cat((lc_features, ap_features), dim=1)
        return self.fc(combined)

def trim_or_pad_sequence(data, target_length):
    return data[:target_length] if len(data) > target_length else np.pad(data, (0, target_length - len(data)), 'constant')

def load_fits_data(file_path):
    with fits.open(file_path) as hdul:
        lc_data = hdul[1].data['FLUX']
        ap_data = hdul[2].data
    return lc_data, ap_data

def preprocess_lc(lc_data, seq_length):
    return trim_or_pad_sequence(lc_data, seq_length)

def denoise_light_curve(lc_data, autoencoder):
    lc_tensor = torch.tensor(lc_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return autoencoder(lc_tensor).squeeze(0).squeeze(0).detach().numpy()

def classify_data(lc_data, ap_data, model):
    lc_tensor = torch.tensor(lc_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ap_tensor = torch.tensor(ap_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(lc_tensor, ap_tensor)
    return torch.argmax(output, dim=1).item()

if __name__ == "__main__":
    file_path = sys.argv[1]
    autoencoder_path = "runScripts/denoiser_autoencoder.pth"
    two_view_model_path = "runScripts/checkpoint_epoch_37.pth"
    seq_length = 6000

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.eval()

    two_view_model = TwoViewModel()
    two_view_model.load_state_dict(torch.load(two_view_model_path))
    two_view_model.eval()

    lc_data, ap_data = load_fits_data(file_path)
    lc_data = preprocess_lc(lc_data, seq_length)
    lc_denoised = denoise_light_curve(lc_data, autoencoder)
    classification = classify_data(lc_denoised, ap_data, two_view_model)

    np.savetxt("denoised_lc.csv", lc_denoised, delimiter=",")
    print(classification)
