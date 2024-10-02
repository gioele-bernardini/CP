import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import threading

# Parametri audio
SAMPLE_RATE = 16000  # Frequenza di campionamento
BUFFER_SIZE = 1024   # Dimensione del buffer per la lettura dei dati audio

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Copia i dati nel buffer
    audio_plot.data = indata[:, 0]

class AudioPlot:
    def __init__(self):
        self.data = np.zeros(BUFFER_SIZE)
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.data)
        self.ax.set_ylim([-1, 1])
        self.ax.set_xlim([0, BUFFER_SIZE])
        self.ax.set_title('Segnale Audio in Tempo Reale')
        self.ax.set_xlabel('Campioni')
        self.ax.set_ylabel('Ampiezza')
        plt.show(block=False)

    def update_plot(self):
        while True:
            self.line.set_ydata(self.data)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

def main():
    global audio_plot
    plt.ion()  # Modalità interattiva di matplotlib
    audio_plot = AudioPlot()
    # Avvia il thread per aggiornare il grafico
    plot_thread = threading.Thread(target=audio_plot.update_plot, daemon=True)
    plot_thread.start()
    # Avvia lo stream audio
    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=BUFFER_SIZE):
        print("Premi Ctrl+C per terminare.")
        try:
            while True:
                pass  # Mantiene lo script in esecuzione
        except KeyboardInterrupt:
            print("Programma terminato.")

if __name__ == '__main__':
    main()
