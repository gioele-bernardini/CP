import torch
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import fft
from MLP_model import MLP  # Importa il modello MLP

# Parametri audio
SAMPLE_RATE = 16000  # Frequenza di campionamento
DURATION = 1  # Durata della registrazione in secondi
MAX_AUDIO_LENGTH = SAMPLE_RATE * DURATION  # Lunghezza massima in campioni

# Funzione per registrare audio
def record_audio(duration, sample_rate):
  print("Inizia la registrazione...")
  audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
  sd.wait()
  print("Registrazione terminata.")
  return audio.flatten()

# Funzione per rilevare se l'audio contiene voce
def detect_voice(audio, threshold=0.01):
  energy = np.sum(audio ** 2) / len(audio)
  return energy > threshold

# Funzione per pre-elaborare l'audio
def preprocess_audio(audio):
  # Padding o troncamento
  if len(audio) < MAX_AUDIO_LENGTH:
    pad_length = MAX_AUDIO_LENGTH - len(audio)
    audio = np.pad(audio, (0, pad_length), mode='constant')
  else:
    audio = audio[:MAX_AUDIO_LENGTH]
  # Trasformata di Fourier
  fft_audio = np.abs(fft(audio))
  # Normalizzazione
  fft_audio = fft_audio / np.max(fft_audio)
  return fft_audio[:len(fft_audio)//2]  # Prendiamo solo la metà positiva dello spettro

# Funzione principale
def main():
  # Passo 1: Acquisizione audio
  audio = record_audio(DURATION, SAMPLE_RATE)
  
  # Passo 2: Rilevamento voce
  if detect_voice(audio):
    print("Voce rilevata. Procedo con l'inferenza.")
  else:
    print("Nessuna voce rilevata. Termino il programma.")
    return
  
  # Passo 3: Pre-elaborazione
  processed_audio = preprocess_audio(audio)
  
  # Visualizzazione dello spettro
  plt.figure(figsize=(10, 4))
  plt.plot(processed_audio)
  plt.title("Spettro dell'audio")
  plt.xlabel("Frequenza")
  plt.ylabel("Ampiezza")
  plt.show()
  
  # Conversione in tensore PyTorch
  input_tensor = torch.tensor(processed_audio, dtype=torch.float32)
  
  # Passo 4: Inferenza
  input_size = MAX_AUDIO_LENGTH // 2  # Deve corrispondere all'input_size del modello
  hidden_sizes = [256, 128]
  output_size = 1  # Output binario
  
  model = MLP(input_size, hidden_sizes, output_size)
  
  # Caricare pesi pre-addestrati
  model.load_state_dict(torch.load('mlp_kws_model.pth'))
  
  # Mettere il modello in modalità eval
  model.eval()
  
  # Eseguire l'inferenza
  with torch.no_grad():
    # Aggiungere dimensione batch
    input_tensor = input_tensor.view(1, -1)
    output = model(input_tensor)
    probability = output.item()
    print(f"Probabilità parola chiave: {probability:.4f}")
    
    # Decisione basata sulla soglia
    if probability > 0.5:
      print("Parola chiave rilevata!")
    else:
      print("Parola chiave non rilevata.")
      
    # Output grafico
    plt.figure()
    plt.bar(['Non Parola Chiave', 'Parola Chiave'], [1 - probability, probability])
    plt.title("Risultato dell'inferenza")
    plt.ylabel("Probabilità")
    plt.show()

if __name__ == "__main__":
  main()
