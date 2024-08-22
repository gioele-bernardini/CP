Setup della Rete Neurale (fino a riga 145!)

    Importazione delle Librerie:
        Abbiamo importato tutte le librerie necessarie per gestire i dati, definire il modello, e preparare l'addestramento.

    Caricamento e Preprocessing dei Dati:
        Abbiamo caricato il dataset MNIST, applicato trasformazioni necessarie (conversione in tensori), e suddiviso i dati in training e validation set.

    Creazione dei DataLoader:
        Abbiamo configurato i DataLoader per iterare sui dati in mini-batch, il che facilita l'addestramento della rete neurale.

    Definizione della Struttura della Rete:
        Abbiamo creato la classe Net che definisce l'architettura della rete, con i layer completamente connessi (fully connected) e i dropout per la regolarizzazione.
        Abbiamo implementato il forward pass per definire come i dati passano attraverso la rete.

    Impostazione della Funzione di Perdita e dell'Ottimizzatore:
        Abbiamo scelto una funzione di perdita adatta (CrossEntropyLoss) e configurato l'ottimizzatore (SGD) per gestire l'aggiornamento dei pesi durante l'addestramento.

    Preparazione per l'Addestramento:
        Abbiamo deciso il numero di epoche e predisposto una variabile per tenere traccia della perdita minima durante la validazione, per poter salvare il miglior modello.

Prossimi Passi:

Ora che il setup è completo, siamo pronti a implementare il ciclo di addestramento vero e proprio, dove:

    Eseguiamo il forward pass su ogni batch.
    Calcoliamo la perdita.
    Eseguiamo il backward pass per calcolare i gradienti.
    Aggiorniamo i pesi usando l'ottimizzatore.
    Monitoriamo la perdita di validazione per salvare il modello migliore.

