import tkinter as tk
from tkinter import messagebox
import threading
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
from scipy.fft import fft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy.io.wavfile as wavfile

# Parametri audio
SAMPLE_RATE = 16000  # Frequenza di campionamento
DURATION = 2  # Durata della registrazione in secondi (aumentata per il test)
MAX_AUDIO_LENGTH = SAMPLE_RATE * DURATION  # Lunghezza massima in campioni

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

# Funzione per registrare audio
def record_audio(duration, sample_rate):
    print("Inizia la registrazione...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Registrazione terminata.")
    return audio.flatten()

# Funzione per rilevare se l'audio contiene voce
def detect_voice(audio, threshold=0.001):  # Soglia ridotta
    energy = np.sum(audio ** 2) / len(audio)
    print(f"Energy of audio: {energy}")
    return energy > threshold

# Funzione per pre-elaborare l'audio
def preprocess_audio(audio):
    # Salva l'audio per il debug
    wavfile.write('test_audio.wav', SAMPLE_RATE, audio)
    # Padding o troncamento
    if len(audio) < MAX_AUDIO_LENGTH:
        pad_length = MAX_AUDIO_LENGTH - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    else:
        audio = audio[:MAX_AUDIO_LENGTH]
    # Trasformata di Fourier
    fft_audio = np.abs(fft(audio))
    # Normalizzazione
    if np.max(fft_audio) > 0:
        fft_audio = fft_audio / np.max(fft_audio)
    else:
        fft_audio = fft_audio
    return fft_audio[:len(fft_audio)//2]  # Prendiamo solo la metà positiva dello spettro

class KWSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keyword Spotting App")
        
        # Caricamento del modello
        self.model = self.load_model()
        
        # Creazione degli elementi della GUI
        self.create_widgets()
        
    def load_model(self):
        input_size = MAX_AUDIO_LENGTH // 2
        hidden_sizes = [256, 128]
        output_size = 1
        model = MLP(input_size, hidden_sizes, output_size)
        model.load_state_dict(torch.load('mlp_kws_model.pth'))
        model.eval()
        return model
    
    def create_widgets(self):
        # Pulsante per avviare la registrazione
        self.record_button = tk.Button(self.root, text="Avvia Registrazione", command=self.start_recording)
        self.record_button.pack(pady=10)
        
        # Etichetta per mostrare il risultato
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)
        
        # Canvas per il grafico dello spettro
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Spettro dell'audio")
        self.ax.set_xlabel("Frequenza")
        self.ax.set_ylabel("Ampiezza")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        # Canvas per il grafico del risultato
        self.result_figure = Figure(figsize=(4, 3), dpi=100)
        self.result_ax = self.result_figure.add_subplot(111)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, master=self.root)
        self.result_canvas.get_tk_widget().pack()
        
    def start_recording(self):
        # Disabilita il pulsante durante la registrazione
        self.record_button.config(state=tk.DISABLED)
        self.result_label.config(text="")
        threading.Thread(target=self.process_audio).start()
        
    def process_audio(self):
        # Passo 1: Acquisizione audio
        audio = record_audio(DURATION, SAMPLE_RATE)
        
        # Passo 2: Rilevamento voce
        if detect_voice(audio):
            print("Voce rilevata. Procedo con l'inferenza.")
        else:
            print("Nessuna voce rilevata.")
            self.result_label.config(text="Nessuna voce rilevata.")
            self.record_button.config(state=tk.NORMAL)
            return
        
        # Passo 3: Pre-elaborazione
        processed_audio = preprocess_audio(audio)
        
        # Aggiorna il grafico dello spettro
        self.ax.clear()
        self.ax.plot(processed_audio)
        self.ax.set_title("Spettro dell'audio")
        self.ax.set_xlabel("Frequenza")
        self.ax.set_ylabel("Ampiezza")
        self.canvas.draw()
        
        # Conversione in tensore PyTorch
        input_tensor = torch.tensor(processed_audio, dtype=torch.float32)
        input_tensor = input_tensor.view(1, -1)
        
        # Passo 4: Inferenza
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
            print(f"Probabilità parola chiave: {probability:.4f}")
            
            # Decisione basata sulla soglia
            if probability > 0.5:
                result_text = "Parola chiave rilevata!"
            else:
                result_text = "Parola chiave non rilevata."
            self.result_label.config(text=result_text)
            
            # Aggiorna il grafico del risultato
            self.result_ax.clear()
            self.result_ax.bar(['Non Parola Chiave', 'Parola Chiave'], [1 - probability, probability], color=['red', 'green'])
            self.result_ax.set_ylabel("Probabilità")
            self.result_ax.set_title("Risultato dell'inferenza")
            self.result_canvas.draw()
        
        # Riabilita il pulsante dopo la registrazione
        self.record_button.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = KWSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
