# Multi-stage build per ottimizzare dimensioni
FROM python:3.10-slim as builder

# Metadata
LABEL maintainer="RAG Prefettura Team"
LABEL description="RAG Pipeline + Chat Assistant for Italian Administrative Law"

# Variabili ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    tesseract-ocr \
    tesseract-ocr-ita \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Crea directory di lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements.txt .

# Installa dipendenze in ordine ottimizzato per caching
# Prima le dipendenze pesanti che cambiano raramente
RUN pip install --user --no-warn-script-location \
    torch==2.8.0 torchvision==0.23.0 

# Poi il resto delle dipendenze
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage finale
FROM python:3.10-slim

# Installa solo runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ita \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia dipendenze Python da builder
COPY --from=builder /root/.local /root/.local

# Imposta PATH per Python packages
ENV PATH=/root/.local/bin:$PATH

# Crea directory di lavoro
WORKDIR /app

# Crea directory necessarie
RUN mkdir -p data/uploads data/indexes storage evaluation/results logs

# Copia il codice dell'applicazione
COPY . .

# Download modello spaCy italiano (opzionale ma consigliato)
RUN python -m spacy download it_core_news_sm || echo "Spacy model download skipped"

# Esponi porta Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando di avvio
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]