# Numpy e Pandas

# Numpy
## Info generiche
Numpy e' implementato in **C**, cio' lo rende efficiente
paragonato ai metodi classici di Python

> E' pratica comune **vettorizzare** per utilizzare numpy!

Altre librerie si basano su _Numpy_, ad esempio <u>SciPy</u>

esempio
```python
v1 = np.array[2.0, 2.0]
v2 = np.array[5.0, 5.0]

# Angolo tra i due vettori
dot_prod = np.dot(v1, v2)
math.acos(dot_prod / (np.linalg.norm(v1) * np.linalg.norm(v2))) 
# math.degrees works the same!
```

### Funzioni principali

## Pandas
### Info generiche
Viene utilizzato per la lettura da file, spesso _in combo_ con Numpy e altre librerie per l'analisi dei dati

### Funzioni principali
pd.Series(np.array([1, 2, 3, 4, 5]), index=['a', 'b'...])