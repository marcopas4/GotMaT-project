# Stage 1: Builder - Installa tutte le dipendenze
FROM python:3.10-slim as builder

LABEL maintainer="RAG Prefettura Team"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Installa dipendenze di sistema (incluse quelle per OpenCV e pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    tesseract-ocr \
    tesseract-ocr-ita \
    libmagic1 \
    # Corretto per OpenCV su Debian/ARM64
    libgl1 \
    libglib2.0-0 \
    # Aggiunta per pdf2image
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Installa torch separatamente per caching
RUN pip install --user --no-warn-script-location torch==2.8.0 torchvision==0.23.0

# Installa il resto delle dipendenze
RUN pip install --user --no-warn-script-location -r requirements.txt


# Stage 2: Final - Crea l'immagine di produzione
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Installa solo le dipendenze di sistema per l'esecuzione
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-ita \
    libmagic1 \
    # Corretto per OpenCV su Debian/ARM64
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copia le dipendenze Python dallo stage builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
RUN mkdir -p data/uploads data/indexes storage evaluation/results logs
COPY . .

# Esponi porta e avvia
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]