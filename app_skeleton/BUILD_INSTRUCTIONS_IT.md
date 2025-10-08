# Guida al Build dell'Eseguibile RAG Prefettura

## Prerequisiti

Prima di creare l'eseguibile, assicurati di avere:

1. **Python 3.11+** installato sul sistema
2. **Tutti i pacchetti** necessari installati:
   ```bash
   pip install -r requirements.txt
   ```

## Creazione dell'Eseguibile

### Passo 1: Installare PyInstaller

Se non l'hai già fatto:
```bash
pip install pyinstaller
```

### Passo 2: Eseguire lo Script di Build

```bash
python build_exe.py
```

Lo script eseguirà automaticamente tutti i passaggi necessari:
- ✅ Pulizia delle cartelle di build precedenti
- ✅ Creazione della configurazione Streamlit
- ✅ Generazione del file .spec personalizzato
- ✅ Build dell'eseguibile con PyInstaller
- ✅ Copia dei file necessari (data, configurazioni)
- ✅ Creazione del launcher batch
- ✅ Generazione del README per utenti finali

### Passo 3: Testare l'Eseguibile

1. Vai nella cartella `dist/`
2. Fai doppio click su `Avvia_RAG_Prefettura.bat`
3. L'applicazione dovrebbe avviarsi e aprire il browser

## Struttura File Generati

Dopo il build, troverai nella cartella `dist/`:

```
dist/
├── RAG_Prefettura.exe              # Eseguibile principale (~200-500 MB)
├── Avvia_RAG_Prefettura.bat        # Launcher automatico
├── README.txt                       # Istruzioni per utenti finali
├── src/                             # Codice sorgente embedded
├── data/                            # Knowledge base e dati
│   ├── faiss_index/                # Indici vettoriali
│   └── knowledge_base/             # Documenti base
└── .streamlit/                     # Configurazioni Streamlit
    └── config.toml
```

## Distribuzione

Per distribuire l'applicazione agli utenti finali:

1. **Copia l'intera cartella `dist/`** su una chiavetta USB o comprimi in un file ZIP
2. Gli utenti dovranno semplicemente:
   - Estrarre/copiare la cartella sul loro computer
   - Fare doppio click su `Avvia_RAG_Prefettura.bat`

**Nota**: L'eseguibile include tutte le dipendenze Python necessarie, quindi gli utenti **NON** devono avere Python installato!

## Dimensione File

- **Eseguibile base**: ~200-300 MB
- **Con modello fine-tuned**: dipende dalla dimensione del modello (può arrivare a diversi GB)
- **Totale cartella dist/**: varia in base ai dati nella knowledge base

## Personalizzazione

### Cambiare l'Icona

1. Crea o scarica un file `.ico` (dimensione consigliata: 256x256)
2. Rinominalo in `icon.ico` e mettilo nella cartella principale
3. Lo script lo includerà automaticamente nell'eseguibile

### Modificare le Impostazioni

Puoi modificare il file `.streamlit/config.toml` prima del build per personalizzare:
- Porta dell'applicazione (default: 8501)
- Tema colori
- Altre configurazioni Streamlit

## Risoluzione Problemi

### Build Fallito

Se il build fallisce, controlla:

1. **Tutte le dipendenze sono installate?**
   ```bash
   pip install -r requirements.txt
   ```

2. **PyInstaller è aggiornato?**
   ```bash
   pip install --upgrade pyinstaller
   ```

3. **Spazio disco sufficiente?**
   - Serve almeno 2-3 GB di spazio libero per il processo di build

### Eseguibile Non Si Avvia

1. **Antivirus**: Alcuni antivirus bloccano eseguibili PyInstaller
   - Aggiungi un'eccezione per `RAG_Prefettura.exe`

2. **Dipendenze Mancanti**: Se manca Tesseract OCR
   - Includi Tesseract nella cartella dist/ oppure
   - Chiedi agli utenti di installarlo separatamente

3. **Test in Console**: Esegui da prompt dei comandi per vedere gli errori
   ```cmd
   cd dist
   RAG_Prefettura.exe
   ```

### Dimensione Eccessiva

Per ridurre la dimensione:

1. **Escludi file non necessari** modificando `build_exe.py`:
   ```python
   excludes=['matplotlib', 'jupyter', 'notebook', 'pytest'],
   ```

2. **Usa UPX** per comprimere l'eseguibile (già abilitato)

3. **Rimuovi modelli non utilizzati** dalla cartella data/

## Build per Altri Sistemi Operativi

**Importante**: PyInstaller crea eseguibili solo per il sistema su cui viene eseguito:
- Build su Windows → file .exe (per Windows)
- Build su Linux → eseguibile Linux
- Build su macOS → app macOS

Non è possibile fare cross-compilation!

## Aggiornamenti

Per aggiornare l'applicazione:

1. Modifica il codice sorgente
2. Ri-esegui `python build_exe.py`
3. Ridistribuisci la cartella `dist/` aggiornata

Gli utenti dovranno sostituire la loro cartella con la nuova versione.

## Performance

L'eseguibile potrebbe essere leggermente più lento all'avvio rispetto all'esecuzione con Python diretto:
- **Primo avvio**: 10-30 secondi (estrazione file temporanei)
- **Avvii successivi**: 5-15 secondi

Questo è normale per applicazioni PyInstaller.

## Note Finali

- L'eseguibile è **standalone** e non richiede installazione
- Tutti i file sono inclusi, ma **non è portable al 100%** (usa cartelle temporanee)
- Per un'installazione più professionale, considera MSI/NSIS installers
- Il modello fine-tuned deve essere nella cartella `models/` prima del build

## Supporto

Per problemi o domande:
1. Controlla i log nella cartella `logs/`
2. Esegui in modalità console per vedere gli errori
3. Verifica i requisiti di sistema

---

**Versione Script**: 1.0.0  
**Compatibilità**: Windows 10/11, Python 3.11+
