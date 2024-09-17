La **feature selection** è un passo cruciale nel machine learning (ML) che implica la scelta di un sottoinsieme rilevante di variabili (features) dal dataset per migliorare la performance del modello. La selezione delle feature mira a ridurre la dimensionalità, eliminando feature ridondanti o irrilevanti, migliorando sia l'efficacia che l'efficienza del modello.

### Motivazioni per cui è necessaria la Feature Selection

1. **Riduzione della dimensionalità**: Un dataset con molte feature può soffrire del problema della dimensionalità, noto anche come "curse of dimensionality". Man mano che il numero di feature aumenta, lo spazio di ricerca diventa esponenzialmente più grande, rendendo difficile per un algoritmo di ML trovare un pattern significativo.
   
2. **Miglioramento della generalizzazione**: Troppe feature aumentano il rischio di overfitting, dove il modello si adatta troppo ai dati di addestramento ma si comporta male su nuovi dati. La feature selection aiuta a ridurre l'overfitting, migliorando la capacità di generalizzazione del modello.

3. **Efficienza computazionale**: Meno feature significano meno dati da elaborare, portando a un miglioramento della velocità di addestramento e inferenza.

4. **Interpretabilità del modello**: Un modello con un numero limitato di feature è più facile da comprendere e interpretare.

### Metodi di Feature Selection

Esistono vari metodi per selezionare le feature:

#### 1. **Wrapper Methods**
I metodi wrapper valutano iterativamente sottoinsiemi di feature e costruiscono modelli su ciascuno di essi per determinare quali feature migliorano la performance del modello. Ad esempio, il metodo "Forward Selection" inizia con un set vuoto di feature e aggiunge progressivamente la feature che migliora maggiormente la performance.

- **Vantaggi**: Spesso danno ottime prestazioni perché si basano direttamente sulle metriche del modello.
- **Svantaggi**: Sono costosi in termini computazionali, specialmente con dataset di grandi dimensioni.

#### 2. **Filter Methods**
I metodi filter valutano l'importanza delle feature utilizzando criteri statistici, come la correlazione con la variabile target. Un esempio è la selezione delle feature basata sul valore del **chi-squared test** o della **mutual information**.

- **Vantaggi**: Veloci e indipendenti dall'algoritmo di ML usato.
- **Svantaggi**: Possono ignorare interazioni complesse tra feature.

#### 3. **Embedded Methods**
Nei metodi embedded, la selezione delle feature avviene durante il processo di costruzione del modello. Ad esempio, i modelli di regressione Lasso e Ridge penalizzano i coefficienti delle feature meno rilevanti, portando implicitamente alla selezione delle feature.

- **Vantaggi**: Bilanciano la performance del modello e la selezione delle feature in modo efficiente.
- **Svantaggi**: Dipendono strettamente dall'algoritmo scelto.

#### 4. **Approcci Bottom-Up e Top-Down**
- **Bottom-Up**: Si inizia con un insieme vuoto e si aggiungono feature fino a trovare la migliore combinazione.
- **Top-Down**: Si parte da tutte le feature disponibili e si rimuovono progressivamente quelle meno utili.

### Relazione con i Modelli Lineari e Normalizzazione dei Pesi

Nei modelli lineari, come la **regressione lineare**, ogni feature viene associata a un peso (coefficiente). Feature irrilevanti o correlate possono avere pesi piccoli o nulli dopo il processo di ottimizzazione. Se le feature sono mal scalate o hanno valori disparati, è comune applicare tecniche di **normalizzazione** per standardizzare i valori delle feature, migliorando la stabilità numerica e consentendo un confronto più equo dei pesi associati a ciascuna feature.

Ad esempio, la **regressione Lasso** usa una penalizzazione L1, che forza alcuni coefficienti a zero, eliminando effettivamente alcune feature non necessarie. La **regressione Ridge**, invece, usa una penalizzazione L2, che riduce i coefficienti ma non li porta a zero, mantenendo tutte le feature.

### Esempi di Non-Linearità e Relazioni Non-Mutuali

Un caso comune di relazione non-mutuale è l'interazione tra feature che, singolarmente, non forniscono informazioni utili ma, combinate, rivelano un pattern. Ad esempio, consideriamo la combinazione di un "hamburger" e un "dessert" in un contesto nutrizionale. Presi singolarmente, non danno un'indicazione precisa se il pasto è sano o meno, ma insieme possono suggerire una combinazione meno salutare rispetto ad altri pasti. Questo è un esempio di **interazioni non lineari**.

Nei modelli lineari, tali interazioni possono essere modellate solo introducendo manualmente termini di interazione, mentre nei modelli non lineari (come le **reti neurali**) queste interazioni possono emergere automaticamente.

### Coefficiente di Correlazione di Pearson

<div align="center">

### Coefficiente di Correlazione di Pearson

</div>

Il coefficiente di correlazione di Pearson misura la relazione lineare tra due variabili, con valori che vanno da -1 (correlazione negativa perfetta) a +1 (correlazione positiva perfetta). La formula è:

<div align="center">

  $r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$

</div>

Dove $(x_i)$ e $(y_i)$ sono i valori osservati delle due variabili, e $(\bar{x})$ e $(\bar{y})$ sono le medie. Questo coefficiente assume una distribuzione lineare: se due variabili seguono una linea retta, avranno una correlazione di 1 o -1.

- **Linea retta**: Correlazione perfetta ($r = \pm1$).
- **Linea pendente**: Se $r > 0$, la pendenza è positiva; se $r < 0$, la pendenza è negativa.
- **Linea piatta**: Se $r = 0$, non c’è relazione lineare.

### Correlation Ratio

<div align="center">

### Correlation Ratio

</div>

Il **correlation ratio** ($\eta$) misura la non linearità tra due variabili. È definito come:

<div align="center">

  $\eta^2 = \frac{\sum_{i=1}^{n} (f_i - \bar{y})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$

</div>

Dove $f_i$ rappresenta la media condizionata di $y$ dato $x$. Può catturare relazioni non lineari, cosa che il coefficiente di Pearson non può fare.

<div align="center">

### Test Statistico delle Ipotesi e Chi-Squared Test

</div>

Il **test delle ipotesi** viene utilizzato per verificare se esiste una relazione significativa tra due variabili o se una caratteristica è importante. L'**ipotesi nulla** ($H_0$) rappresenta una dichiarazione di "assenza di effetto" o "assenza di relazione". Un esempio può essere verificare se la lunghezza di un articolo è correlata con la categoria di appartenenza.

Il **chi-squared test** misura la dipendenza tra due variabili categoriche. Se $O_i$ sono i valori osservati e $E_i$ i valori attesi, la formula è:

<div align="center">

  $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$

</div>

Un valore alto di $\chi^2$ indica che le variabili sono probabilmente correlate, rifiutando l'ipotesi nulla.

Questi metodi matematici e statistici sono fondamentali per la feature selection e per garantire che il modello di machine learning utilizzi solo le variabili più rilevanti, migliorando così performance e interpretabilità.

