import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio.transforms as T
from PIL import Image

class SignalDataset(Dataset):
    def __init__(self, signal_dir, annotation_path, transform=None):
        self.signal_dir = signal_dir
        self.annotations = pd.read_csv(annotation_path)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        signal_id = self.annotations.iloc[idx]['id']
        signal_path = os.path.join(self.signal_dir, f'{signal_id}.npy')
        signal = np.load(signal_path)
        spectrograms = [T.Spectrogram()(torch.tensor(signal[i])) for i in range(3)]
        spectrogram_stack = np.stack(spectrograms, axis=-1)
        image = Image.fromarray(spectrogram_stack)
        if self.transform:
            image = self.transform(image)
        label = self.annotations.iloc[idx]['label']
        return image, label

def create_dataloader(signal_dir, annotation_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SignalDataset(signal_dir, annotation_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
