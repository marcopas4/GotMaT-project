# RAG Prefettura - Assistente AI per Illeciti Amministrativi

Un'applicazione RAG (Retrieval-Augmented Generation) sviluppata per assistere la prefettura nella gestione e consultazione di documenti relativi agli illeciti amministrativi.

## 🎯 Caratteristiche

- **Interfaccia intuitiva** simile a ChatGPT ottimizzata per utenti non tecnici
- **Upload documenti** con drag & drop (PDF, DOCX, TXT, immagini)
- **Dual-mode search**: knowledge base precaricata + documenti caricati dall'utente
- **Modello fine-tuned** specializzato nel dominio degli illeciti amministrativi
- **Vector database FAISS** per ricerca semantica veloce
- **OCR integrato** per processamento immagini e PDF scansionati
- **Deploy Docker** ottimizzato per Windows

## 📋 Prerequisiti

### Sistema Operativo
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (10.15+)

### Software Richiesto
- **Python 3.11+**
- **Docker Desktop** (per deployment containerizzato)
- **Git**
- **Tesseract OCR** (per processamento immagini)
  - Windows: [Download qui](https://github.com/tesseract-ocr/tesseract/wiki)
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-ita`
  - macOS: `brew install tesseract tesseract-lang`

### Hardware Raccomandato
- **RAM**: 8GB+ (16GB consigliati)
- **CPU**: 4 core+
- **Storage**: 10GB spazio libero
- **GPU**: Opzionale (accelera inferenza modello)

## 🚀 Installazione Rapida

### 1. Clone del Repository
```bash
git clone https://github.com/your-org/rag-prefettura.git
cd rag-prefettura
```

### 2. Setup Automatico
```bash
# Installazione completa
python setup.py install

# Solo configurazione (se dipendenze già presenti)
python setup.py configure
```

### 3. Avvio Applicazione
```bash
# Locale
streamlit run app.py

# Docker
docker-compose up --build
```

### 4. Accesso
Apri il browser su: http://localhost:8501

## 📁 Struttura del Progetto

```
rag-prefettura/
├── app.py                      # Applicazione Streamlit principale
├── requirements.txt            # Dipendenze Python
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container setup
├── setup.py                    # Script installazione automatica
├── README.md                   # Documentazione (questo file)
├── .env                        # Configurazioni (generato da setup)
│
├── src/                        # Codice sorgente modulare
│   ├── __init__.py
│   ├── document_processor.py   # 📝 Estrazione testo da documenti
│   ├── vector_store.py         # 🔍 Database vettoriale FAISS  
│   ├── llm_handler.py          # 🧠 Gestione modello fine-tuned
│   └── utils.py                # 🛠️ Funzioni di utilità
│
├── data/                       # Dati dell'applicazione
│   ├── knowledge_base/         # 📚 Knowledge base precaricata
│   ├── uploads/                # 📤 Documenti caricati utenti
│   └── faiss_index/           # 💾 Indici FAISS persistenti
│
├── models/                     # 🤖 Modelli ML
│   └── fine_tuned_model/       # Il vostro modello fine-tuned
│
├── logs/                       # 📋 File di log
├── temp/                       # 🗂️ File temporanei
└── .streamlit/                # ⚙️ Configurazioni Streamlit
    └── config.toml
```

## 🔧 Configurazione per i Tuoi Colleghi

Il sistema è strutturato con **placeholder ben definiti** per permettere ai tuoi colleghi di implementare facilmente i componenti mancanti:

### 1. Document Processor (`src/document_processor.py`)
**TODO per i tuoi colleghi:**
- Implementare `_process_pdf()` con PyPDF2/pdfplumber
- Implementare `_process_docx()` con python-docx
- Implementare `_process_image()` con pytesseract OCR
- Migliorare `_clean_text()` e `_create_chunks()`

**Esempio implementazione PDF:**
```python
def _process_pdf(self, file_path: str) -> Dict[str, Any]:
    import pdfplumber
    
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    return self._create_document_result(text, file_path, 'pdf')
```

### 2. Vector Store (`src/vector_store.py`)
**TODO per i tuoi colleghi:**
- Implementare `_initialize_embedding_model()` con sentence-transformers
- Implementare `create_embeddings()` 
- Implementare ricerca FAISS in `search_knowledge_base()`
- Implementare persistenza indici

**Esempio implementazione embeddings:**
```python
def _initialize_embedding_model(self):
    from sentence_transformers import SentenceTransformer
    
    self.embedding_model = SentenceTransformer(
        'sentence-transformers/distiluse-base-multilingual-cased'
    )
    
def create_embeddings(self, texts: List[str]) -> np.ndarray:
    return self.embedding_model.encode(texts, normalize_embeddings=True)
```

### 3. LLM Handler (`src/llm_handler.py`)
**TODO per i tuoi colleghi:**
- Implementare `_initialize_model()` con il vostro modello fine-tuned
- Implementare `_generate_with_model()` per la generazione
- Personalizzare i prompt template per il dominio specifico

**Esempio implementazione con Transformers:**
```python
def _initialize_model(self):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
```

## 🏃‍♂️ Guida Rapida all'Uso

### Per gli Utenti della Prefettura

1. **Avvia l'applicazione** e vai su http://localhost:8501
2. **Carica documenti** (opzionale):
   - Trascina i file nell'area di upload nella sidebar
   - Clicca "Carica Documenti"
3. **Fai una domanda**:
   - Scrivi la tua domanda nell'area di testo
   - Scegli se cercare nella knowledge base o nei tuoi documenti
   - Clicca "Invia Domanda"
4. **Ricevi la risposta** con fonti e referenze

 