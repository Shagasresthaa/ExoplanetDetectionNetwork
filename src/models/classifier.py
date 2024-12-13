import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from torchvision import transforms

SAVE_DIR = "model_saves/classifier/"
LOG_DIR = "logs/training/"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"two_view_classifier_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    logging.info("Using GPU: Set to exclusive mode.")

class LightCurveDataset(Dataset):
    def __init__(self, index_csv, sequence_length, zoom_length, scalar_features, mode="train", validation_split=0.2, balance_classes=False):
        self.data = pd.read_csv(index_csv)
        self.sequence_length = sequence_length
        self.zoom_length = zoom_length
        self.scalar_features = scalar_features

        for feature in scalar_features:
            if feature in self.data.columns:
                self.data[feature] = (self.data[feature] - self.data[feature].min()) / (
                    self.data[feature].max() - self.data[feature].min() + 1e-6
                )

        split_idx = int((1 - validation_split) * len(self.data))
        if mode == "train":
            self.data = self.data.iloc[:split_idx]
        elif mode == "val":
            self.data = self.data.iloc[split_idx:]
        else:
            raise ValueError("Mode must be 'train' or 'val'.")

        if balance_classes and mode == "train":
            class_counts = self.data["Label"].value_counts()
            min_class_count = class_counts.min()
            self.data = self.data.groupby("Label").apply(lambda x: x.sample(n=min_class_count, random_state=42))
            self.data = self.data.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        lc_path = row["LC_Path"]
        ap_path = row["AP_Path"]
        label = int(row["Label"])

        flux = pd.read_csv(lc_path)["FLUX"].values
        flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-6) * 2 - 1

        full_view = np.pad(flux[:self.sequence_length], (0, max(0, self.sequence_length - len(flux[:self.sequence_length]))), "constant")
        epoch, duration = row["Epoch"], row["Duration"]
        start_idx, end_idx = max(0, int(epoch - duration)), int(epoch + duration)
        zoomed_view = np.pad(flux[start_idx:end_idx], (0, max(0, self.zoom_length - len(flux[start_idx:end_idx]))), "constant")

        ap_image = np.load(ap_path).astype(np.float32)
        ap_image = self.transform(ap_image)

        scalar_features = row[self.scalar_features].fillna(0).values.astype(np.float32)

        return (
            torch.tensor(full_view, dtype=torch.float32).unsqueeze(0),
            torch.tensor(zoomed_view, dtype=torch.float32).unsqueeze(0),
            ap_image,
            torch.tensor(scalar_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss

class TwoViewClassifier(nn.Module):
    def __init__(self, sequence_length, zoom_length, num_classes, scalar_feature_size, d_model=128, dropout=0.5):
        super(TwoViewClassifier, self).__init__()
        self.full_view_cnn = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.zoom_view_cnn = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.ap_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        total_features_dim = d_model * 2 + scalar_feature_size + 128 * 16 * 16
        self.fc = nn.Sequential(
            nn.Linear(total_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, full_view, zoomed_view, ap_image, scalar_features):
        full_features = self.full_view_cnn(full_view).mean(dim=2)
        zoomed_features = self.zoom_view_cnn(zoomed_view).mean(dim=2)
        ap_features = self.ap_cnn(ap_image).flatten(start_dim=1)
        combined_features = torch.cat([full_features, zoomed_features, scalar_features, ap_features], dim=1)
        return self.fc(combined_features)

def train_two_view_classifier(model, train_loader, val_loader, num_epochs, learning_rate, device, save_dir):
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=learning_rate, step_size_up=10)

    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for full_view, zoomed_view, ap_image, scalar_features, labels in train_loader:
            full_view, zoomed_view, ap_image, scalar_features, labels = (
                full_view.to(device),
                zoomed_view.to(device),
                ap_image.to(device),
                scalar_features.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(full_view, zoomed_view, ap_image, scalar_features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_accuracy = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct, all_labels, all_preds = 0.0, 0, [], []
        with torch.no_grad():
            for full_view, zoomed_view, ap_image, scalar_features, labels in val_loader:
                full_view, zoomed_view, ap_image, scalar_features, labels = (
                    full_view.to(device),
                    zoomed_view.to(device),
                    ap_image.to(device),
                    scalar_features.to(device),
                    labels.to(device),
                )
                outputs = model(full_view, zoomed_view, ap_image, scalar_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.argmax(1).cpu().numpy())

        val_accuracy = val_correct / len(val_loader.dataset)
        scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
            logging.info(f"Checkpoint saved with Val Accuracy: {val_accuracy:.4f}")

        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"], output_dict=True)
        f1 = report["weighted avg"]["f1-score"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        logging.info(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    index_csv = "data/preprocessed/balanced_datasetIndex.csv"
    sequence_length, zoom_length, batch_size, num_epochs, learning_rate = 6000, 1000, 64, 150, 0.001
    scalar_features = ["Period", "Duration"]
    device = "cuda"

    train_dataset = LightCurveDataset(index_csv, sequence_length, zoom_length, scalar_features, mode="train", validation_split=0.2, balance_classes=True)
    val_dataset = LightCurveDataset(index_csv, sequence_length, zoom_length, scalar_features, mode="val", validation_split=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TwoViewClassifier(sequence_length, zoom_length, num_classes=2, scalar_feature_size=len(scalar_features))
    train_two_view_classifier(model, train_loader, val_loader, num_epochs, learning_rate, device, SAVE_DIR)
