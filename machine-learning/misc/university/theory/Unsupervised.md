### Definizione di Unsupervised Learning
L'**Unsupervised Learning** è una tecnica di **machine learning** in cui l'algoritmo apprende dai dati senza etichette. In altre parole, al contrario del supervised learning, non esistono output noti associati agli input. L'obiettivo dell'unsupervised learning è quello di identificare modelli, strutture o pattern nascosti nei dati. L'algoritmo cerca relazioni tra i dati per raggrupparli (clustering), ridurre la dimensionalità, o scoprire distribuzioni latenti.

Formalmente, dato un dataset \(\{X_1, X_2, ..., X_n\}\), con \(X_i \in \mathbb{R}^d\) per ogni \(i\), il problema dell'unsupervised learning consiste nel trovare una funzione \(f: \mathbb{R}^d \to \mathbb{R}^k\) che descriva la struttura intrinseca dei dati, dove \(k \leq d\). Non c'è un output noto \(\{Y_1, Y_2, ..., Y_n\}\), e l'obiettivo è spesso raggruppare i dati o ridurre la dimensionalità.

### Principali Tipi di Unsupervised Learning
1. **Clustering**: Raggruppa i dati in sottoinsiemi o gruppi simili. Esempi di algoritmi sono:
   - **K-Means**: Cerca di suddividere \(n\) osservazioni in \(k\) cluster, minimizzando la distanza intra-cluster (somma delle distanze quadratiche dagli elementi del cluster al loro centroide).
     \[
     \underset{C}{\text{min}} \sum_{i=1}^{k} \sum_{x_j \in C_i} \| x_j - \mu_i \|^2
     \]
     Dove \(C_i\) è il cluster \(i\)-esimo e \(\mu_i\) è il suo centroide.
   
   - **Hierarchical Clustering**: Crea una gerarchia di cluster mediante un approccio agglomerativo (bottom-up) o divisivo (top-down).
   
   - **DBSCAN**: Rileva cluster basati sulla densità di punti, ideale per cluster di forme irregolari e dataset con outliers.

2. **Riduzione della dimensionalità**: Tecniche per ridurre il numero di variabili mantenendo il più possibile l'informazione. Esempi includono:
   - **Principal Component Analysis (PCA)**: Riduce la dimensionalità trovando le componenti principali, ovvero le direzioni di massima varianza nel dataset. Formalmente, si risolve il problema di massimizzare la varianza proiettata.
     \[
     \underset{w}{\text{max}} \left( \frac{1}{n} \sum_{i=1}^{n} (w^T X_i)^2 \right)
     \]
     Soggetto alla condizione che \(\| w \| = 1\).
   
   - **t-SNE**: Una tecnica non lineare per la riduzione dimensionale, usata per visualizzare dataset ad alta dimensionalità in spazi bidimensionali o tridimensionali.

3. **Anomaly Detection**: Identifica punti anomali nel dataset che non si conformano ai pattern osservati. Un esempio è l'**Isolation Forest**, che cerca di isolare le anomalie dividendole iterativamente dalle osservazioni normali.

### Procedura per sviluppare un sistema di Unsupervised Learning

#### 1. Raccolta e Pre-elaborazione dei Dati
- **Raccolta dei dati**: Come nel supervised learning, è essenziale raccogliere dati rappresentativi del fenomeno di interesse.
- **Pre-processing**: Pulizia dei dati, normalizzazione o scaling (per esempio, standardizzazione dei dati). Alcuni algoritmi di clustering, come K-Means, sono sensibili alla scala delle feature, quindi è fondamentale normalizzare i dati prima dell'uso.

#### 2. Selezione dell'Algoritmo
La selezione dell'algoritmo dipende dal tipo di problema:
- **Clustering**: Se il problema richiede la divisione dei dati in gruppi distinti, si può scegliere un algoritmo come K-Means, DBSCAN, o Hierarchical Clustering.
- **Riduzione della dimensionalità**: Se si vuole ridurre la complessità del dataset, si può utilizzare PCA, t-SNE o altri metodi non lineari come l'Autoencoder.
- **Anomaly detection**: Per la rilevazione di anomalie, si possono usare metodi come Isolation Forest o One-Class SVM.

#### 3. Addestramento del Modello
- Nel clustering, l'algoritmo suddivide i dati in gruppi in base alla similarità. Ad esempio, in K-Means, l'obiettivo è minimizzare la somma delle distanze intra-cluster.
  
  La funzione di costo da minimizzare in K-Means è:
  \[
  J = \sum_{i=1}^{k} \sum_{x_j \in C_i} \| x_j - \mu_i \|^2
  \]
  dove \(C_i\) è il \(i\)-esimo cluster, \(\mu_i\) è il centroide del cluster, e \(x_j\) sono i punti all'interno del cluster.

- In PCA, si massimizza la varianza proiettata lungo le componenti principali, trovando gli autovettori della matrice di covarianza del dataset.

#### 4. Valutazione del Modello
La valutazione in unsupervised learning è meno diretta rispetto al supervised learning, poiché non si hanno etichette su cui basarsi. Tuttavia, ci sono tecniche che possono essere utilizzate per valutare le prestazioni:
- **Silhouette score**: Usato per valutare quanto bene sono separati i cluster.
  \[
  s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
  \]
  dove \(a(i)\) è la distanza media tra il punto \(i\) e tutti gli altri punti nel suo cluster, mentre \(b(i)\) è la distanza media tra il punto \(i\) e i punti del cluster più vicino.
  
- **Varianza spiegata**: In PCA, indica quanta della varianza totale dei dati è catturata dalle prime componenti principali.
  
- **Elbow method**: In K-Means, si traccia il valore della funzione di costo in funzione del numero di cluster \(k\), cercando il punto in cui l'incremento di cluster non migliora significativamente la qualità della suddivisione.

#### 5. Test e Generalizzazione
Come nel supervised learning, è importante testare il modello su un set di dati non utilizzato durante l'addestramento per valutare la sua capacità di generalizzazione.

### Errori Comuni da Evitare

1. **Mancata normalizzazione dei dati**: Algoritmi come K-Means o PCA sono molto sensibili alla scala delle feature. Non normalizzare i dati può portare a risultati non corretti.
  
2. **Scelta errata del numero di cluster**: Nel clustering, scegliere il numero di cluster \(k\) arbitrariamente può portare a soluzioni subottimali. L'elbow method o tecniche basate su criteri di valutazione come il Silhouette score possono aiutare nella selezione corretta.

3. **Overfitting nella riduzione della dimensionalità**: Anche se l'obiettivo è semplificare i dati, è possibile ridurre eccessivamente la dimensionalità, perdendo informazioni cruciali. È importante trovare un buon compromesso tra riduzione della dimensionalità e conservazione dell'informazione.

4. **Uso di metriche inappropriate**: Non tutti gli algoritmi di unsupervised learning possono essere valutati con le stesse metriche. Per esempio, il Silhouette score è più adatto al clustering, mentre la varianza spiegata è più adatta per la PCA.

5. **Data leakage involontario**: Sebbene non ci siano etichette, è importante evitare che i dati di validazione influenzino l'addestramento del modello in modo da mantenere una stima imparziale delle prestazioni.

### Conclusione
L'unsupervised learning offre un potente strumento per esplorare e scoprire strutture nascoste nei dati non etichettati. Tuttavia, richiede attenzione nell'interpretazione e valutazione dei risultati, poiché l'assenza di etichette rende più difficile misurare direttamente la bontà delle soluzioni.