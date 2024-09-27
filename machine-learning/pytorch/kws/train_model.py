import os
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import fft
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import Resample
import torchaudio

# Impostazioni globali
SAMPLE_RATE = 16000
KEYWORD = 'yes'  # Parola chiave da rilevare
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
AUDIO_DURATION = 1  # Durata fissa in secondi
MAX_AUDIO_LENGTH = SAMPLE_RATE * AUDIO_DURATION  # Lunghezza massima in campioni

# Funzione per caricare e preprocessare un singolo file audio
def load_and_preprocess(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    audio = waveform.numpy().flatten()
    # Padding o troncamento
    if len(audio) < MAX_AUDIO_LENGTH:
        pad_length = MAX_AUDIO_LENGTH - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    else:
        audio = audio[:MAX_AUDIO_LENGTH]
    # Pre-elaborazione: trasformata di Fourier
    fft_audio = np.abs(fft(audio))
    fft_audio = fft_audio[:len(fft_audio)//2]
    # Normalizzazione
    fft_audio = fft_audio / np.max(fft_audio)
    return fft_audio

# Dataset personalizzato per il keyword spotting
class KWSDataset(Dataset):
    def __init__(self, keyword_dir, other_dirs):
        self.files = []
        self.labels = []
        # Aggiungi file della parola chiave (label 1)
        for f in os.listdir(keyword_dir):
            if f.endswith('.wav'):
                self.files.append(os.path.join(keyword_dir, f))
                self.labels.append(1)
        # Aggiungi file delle altre parole (label 0)
        for other_dir in other_dirs:
            for f in os.listdir(other_dir):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(other_dir, f))
                    self.labels.append(0)
                        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio = load_and_preprocess(self.files[idx])
        label = self.labels[idx]
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        
# Definizione del modello MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
def train_model():
    # Percorsi alle cartelle delle parole chiave e delle altre parole
    keyword_dir = f'./speech_commands_v0.02/{KEYWORD}'
    other_words = ['no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    other_dirs = [f'./speech_commands_v0.02/{word}' for word in other_words]

    # Controlla se le cartelle esistono
    for dir_path in [keyword_dir] + other_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory non trovata: {dir_path}")

    # Creazione del dataset
    dataset = KWSDataset(keyword_dir, other_dirs)
    
    # Suddivisione in training e validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inizializzazione del modello
    input_size = MAX_AUDIO_LENGTH // 2  # Dopo la FFT prendiamo metà dello spettro
    hidden_sizes = [256, 128]
    output_size = 1
    model = MLP(input_size, hidden_sizes, output_size)
    
    # Loss function e optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Ciclo di addestramento
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validazione
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                labels = labels.view(-1, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Salvataggio del modello
    torch.save(model.state_dict(), 'mlp_kws_model.pth')
    print("Modello addestrato e salvato come 'mlp_kws_model.pth'")

if __name__ == "__main__":
    train_model()
