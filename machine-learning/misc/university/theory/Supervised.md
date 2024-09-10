### Definizione di Supervised Learning
Il **Supervised Learning** è una tecnica di **machine learning** in cui un algoritmo apprende da dati etichettati. In altre parole, ogni esempio nel set di dati ha un input associato (feature) e un output (etichetta o target). L'obiettivo dell'algoritmo è apprendere una funzione che mappa gli input agli output corretti, in modo tale che, una volta addestrato, possa predire correttamente gli output per nuovi dati non etichettati.

Formalmente, data una coppia di variabili \((X, Y)\), dove \(X\) rappresenta il vettore degli input e \(Y\) l'output, il problema di Supervised Learning consiste nel trovare una funzione \(f\) tale che:

\[
Y = f(X) + \epsilon
\]

dove \(\epsilon\) è il rumore (o errore) aleatorio associato al processo di apprendimento.

### Procedura per sviluppare un sistema di ML supervisato

#### 1. Raccolta e Pre-elaborazione dei Dati
- **Dati rappresentativi**: Il dataset deve essere rappresentativo del problema che si vuole risolvere. Se i dati non coprono tutte le possibili varianti degli input, il modello non sarà in grado di generalizzare bene.
- **Pre-processing**: È importante pulire i dati, gestire i valori mancanti, normalizzare o scalare i dati, ed eventualmente effettuare trasformazioni per migliorare le performance dell'algoritmo.
  
#### 2. Divisione del Dataset
Il dataset va diviso in tre parti:
- **Training set**: È il dataset usato per addestrare il modello.
- **Validation set**: È utilizzato durante la fase di sviluppo per valutare le prestazioni del modello e ottimizzare i suoi parametri iper.
- **Test set**: È riservato per la valutazione finale del modello, simulando la sua capacità di generalizzare a dati non visti.

La divisione può essere fatta con una tecnica come la **cross-validation**, che aiuta a stimare le performance del modello riducendo il rischio di overfitting.

#### 3. Selezione del Modello
Scelta dell'algoritmo di apprendimento più adatto, ad esempio:
- **Regressione lineare** per problemi di regressione.
- **K-Nearest Neighbors (KNN)** o **Support Vector Machines (SVM)** per problemi di classificazione.
- **Random Forest** o **Reti Neurali** per problemi più complessi e non lineari.

#### 4. Addestramento del Modello
Durante questa fase, l'algoritmo apprende dal training set. L'algoritmo minimizza una **funzione di costo** o **loss function**, ad esempio:
- **Mean Squared Error (MSE)** per problemi di regressione:
  \[
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
  dove \(y_i\) è l'output reale e \(\hat{y}_i\) è la predizione del modello.
  
- **Cross-Entropy Loss** per problemi di classificazione binaria:
  \[
  L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]

Durante l'addestramento, gli **iperparametri** (come la profondità di un albero decisionale, la dimensione di batch nelle reti neurali, ecc.) vengono ottimizzati per migliorare le prestazioni.

#### 5. Validazione del Modello
Dopo l'addestramento, il modello viene valutato su un validation set. In questa fase, si può applicare **early stopping** per prevenire l'overfitting, oppure ottimizzare gli iperparametri tramite tecniche come la **ricerca a griglia** o **ricerca casuale**.

#### 6. Test e Valutazione Finale
Una volta ottimizzato, il modello viene testato sul test set. Si utilizzano metriche di valutazione specifiche, come:
- **Accuracy**: per classificazione, misura la frazione di esempi correttamente classificati.
  \[
  \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
  \]
  
- **Precision, Recall, F1-score**: per classificazioni sbilanciate.
- **Mean Absolute Error (MAE)**, **R-squared**, per regressione.

### Errori Comuni da Evitare

1. **Overfitting**: Quando il modello apprende troppo bene i dettagli e il rumore del training set, performa male su dati nuovi. Si previene con tecniche come la **regolarizzazione** (L1, L2), **early stopping**, e **data augmentation**.

2. **Underfitting**: Quando il modello è troppo semplice e non riesce a catturare la complessità dei dati. Può essere mitigato scegliendo un modello più complesso o aumentando le feature.

3. **Data leakage**: Quando le informazioni nel test set finiscono nel training set, portando a performance artificialmente alte. È importante mantenere una rigorosa separazione tra i dati di addestramento e di test.

4. **Scelta di metriche inappropriate**: Utilizzare metriche che non riflettono correttamente le prestazioni del modello può portare a interpretazioni errate. Per esempio, in un dataset sbilanciato, l'accuracy potrebbe non essere la metrica più indicata.

5. **Dati non sufficientemente rappresentativi**: Se i dati non coprono la varietà di scenari reali, il modello non sarà in grado di generalizzare bene.

Seguendo questo processo rigoroso e evitando gli errori comuni, è possibile sviluppare un sistema di machine learning supervisionato robusto ed efficace.
