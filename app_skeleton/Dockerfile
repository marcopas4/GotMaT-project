# Dockerfile per RAG Prefettura
# Ottimizzato per Windows e deployment locale

FROM python:3.11-slim

# Metadata
LABEL maintainer="RAG Prefettura Team"
LABEL version="1.0.0"
LABEL description="RAG Application per Assistente Illeciti Amministrativi"

# Variabili di ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Installa dipendenze sistema necessarie
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    tesseract-ocr \
    tesseract-ocr-ita \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Crea utente non-root per sicurezza
RUN useradd -m -u 1000 raguser && \
    mkdir -p /app && \
    chown raguser:raguser /app

# Imposta directory di lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Scarica modelli spaCy per italiano
RUN python -m spacy download it_core_news_sm

# Scarica modelli NLTK necessari
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Crea directory necessarie
RUN mkdir -p data/knowledge_base \
    data/uploads \
    data/faiss_index \
    models \
    logs \
    temp

# Copia codice applicazione
COPY src/ ./src/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Copia knowledge base di esempio (se presente)
COPY data/knowledge_base/ ./data/knowledge_base/

# Imposta permessi
RUN chown -R raguser:raguser /app

# Cambia a utente non-root
USER raguser

# Espone porta Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando di avvio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]