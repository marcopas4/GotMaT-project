# 🐳 Docker Setup - RAG Prefettura

Guida completa per eseguire l'applicazione RAG Prefettura con Docker.

---

## 📋 Prerequisiti

### Requisiti Software
- **Docker Desktop** 4.25+ ([Download](https://www.docker.com/products/docker-desktop/))
- **Docker Compose** 2.20+ (incluso in Docker Desktop)
- Almeno **8GB RAM** disponibile per Docker
- **20GB spazio disco** libero (per modelli e immagini)

### Requisiti Hardware Consigliati
- **CPU**: 4+ core
- **RAM**: 16GB+ totali (8GB+ per Docker)
- **GPU** (opzionale): NVIDIA con CUDA per accelerazione

---

## 🚀 Quick Start

### Linux/macOS

```bash
# 1. Clona/naviga nella directory del progetto
cd /path/to/rag-prefettura

# 2. Rendi eseguibile lo script di setup
chmod +x app_setup.sh

# 3. Esegui il setup automatico
./app_setup.sh

# 4. Se hai modificato codice Python, usa --build per ricostruire l'immagine
./app_setup.sh --build
```

### Windows

```batch
REM 1. Apri PowerShell o CMD nella directory del progetto
cd C:\path\to\rag-prefettura

REM 2. Esegui lo script di setup
app_setup.bat

REM 3. Se hai modificato codice Python, ricostruisci manualmente l'immagine
docker compose build rag-app
docker compose up -d
```

### Avvio Manuale

Se preferisci configurare manualmente:

```bash
# 1. Avvia i container
docker compose up -d

# 2. Attendi che Ollama sia pronto (circa 10 secondi)
docker compose logs -f ollama

# 3. Scarica il modello LLM
docker exec rag-ollama ollama pull llama3.2:3b-instruct-q4_K_M

# 4. Accedi all'app
# Browser: http://localhost:8501
```

---

## 🏗️ Architettura

Il `docker-compose.yml` configura 2 servizi:

### 1. **ollama** (LLM Service)
- **Porta**: 11434
- **Volume**: `ollama_data` per persistenza modelli
- **GPU**: Supporto NVIDIA (opzionale)
- Gestisce l'inferenza dei modelli LLM

### 2. **rag-app** (Streamlit Application)
- **Porta**: 8501 (interfaccia web)
- **Volumi montati**:
  - `./data` → Documenti caricati e indici
  - `./storage` → Storage LlamaIndex
  - `./logs` → Log applicazione
  - `./evaluation` → Risultati valutazione
- **Dipendenze**: Ollama (deve essere attivo)

---

## ⚙️ Configurazione

### Senza GPU (CPU Only)

Se non hai una GPU NVIDIA, modifica `docker-compose.yml`:

```yaml
services:
  ollama:
    # Commenta questa sezione:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
```

### Personalizza Porte

Modifica le porte in `docker-compose.yml`:

```yaml
services:
  ollama:
    ports:
      - "11434:11434"  # Cambia la prima porta per binding esterno
  
  rag-app:
    ports:
      - "8501:8501"    # Cambia la prima porta per accesso web
```

### Variabili Ambiente

Modifica `.env` (crea il file se non esiste):

```env
# Ollama
OLLAMA_BASE_URL=http://ollama:11434

# Logging
LOG_LEVEL=INFO

# Performance
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

---

## 📦 Gestione Container

### Comandi Base

```bash
# Avvia i servizi
docker compose up -d

# Ferma i servizi
docker compose down

# Riavvia un servizio specifico
docker compose restart rag-app

# Ferma e rimuovi tutto (inclusi volumi)
docker compose down -v
```

### Monitoraggio

```bash
# Vedi log in tempo reale
docker compose logs -f

# Log di un servizio specifico
docker compose logs -f rag-app

# Stato dei container
docker compose ps

# Risorse utilizzate
docker stats
```

### Shell nei Container

```bash
# Accedi al container dell'app
docker exec -it rag-streamlit-app bash

# Accedi al container Ollama
docker exec -it rag-ollama bash

# Esegui comandi Python nell'app
docker exec -it rag-streamlit-app python -c "import torch; print(torch.__version__)"
```

---

## 🧪 Testing

### Verifica Ollama

```bash
# Testa API Ollama
curl http://localhost:11434/api/tags

# Lista modelli installati
docker exec rag-ollama ollama list

# Testa generazione
docker exec rag-ollama ollama run llama3.2:3b-instruct-q4_K_M "Ciao, come stai?"
```

### Verifica App

```bash
# Health check Streamlit
curl http://localhost:8501/_stcore/health

# Log dell'app
docker compose logs rag-app | tail -50
```

---

## 🔧 Troubleshooting

### Problema: Ollama non risponde

```bash
# Controlla stato
docker compose ps ollama

# Vedi log per errori
docker compose logs ollama

# Riavvia il servizio
docker compose restart ollama

# Se persiste, ricrea il container
docker compose up -d --force-recreate ollama
```

### Problema: App non si connette a Ollama

Verifica che:
1. Ollama sia attivo: `docker compose ps`
2. Network sia corretto: `docker network inspect rag-network`
3. Variabile ambiente: `OLLAMA_BASE_URL=http://ollama:11434`

```bash
# Testa connessione dall'app
docker exec rag-streamlit-app curl http://ollama:11434/api/tags
```

### Problema: Porta 8501 già in uso

```bash
# Trova processo che usa la porta
# Linux/macOS:
lsof -i :8501

# Windows:
netstat -ano | findstr :8501

# Cambia porta in docker-compose.yml
ports:
  - "8502:8501"  # Usa 8502 invece
```

### Problema: Out of Memory

```bash
# Aumenta memoria Docker Desktop
# Settings → Resources → Memory → 16GB

# Oppure ridimensiona il modello
docker exec rag-ollama ollama pull llama3.2:1b-instruct-q4_K_M  # Modello più piccolo
```

### Problema: Modelli non scaricati

```bash
# Scarica manualmente
docker exec rag-ollama ollama pull llama3.2:3b-instruct-q4_K_M

# Verifica modelli installati
docker exec rag-ollama ollama list
```

---

## 🗂️ Persistenza Dati

### Volumi Docker

I dati persistenti sono salvati in:

```
ollama_data/         → Modelli LLM Ollama
./data/uploads/      → Documenti caricati
./data/indexes/      → Indici FAISS
./storage/           → Storage LlamaIndex
./logs/              → Log applicazione
```

### Backup

```bash
# Backup volume Ollama
docker run --rm -v rag-ollama-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama-backup.tar.gz -C /data .

# Backup dati applicazione
tar czf rag-data-backup.tar.gz data/ storage/ logs/
```

### Restore

```bash
# Restore volume Ollama
docker run --rm -v rag-ollama-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/ollama-backup.tar.gz -C /data

# Restore dati applicazione
tar xzf rag-data-backup.tar.gz
```

---

## 🔄 Update & Maintenance

### Aggiorna Immagini

```bash
# Pull ultime versioni
docker compose pull

# Ricrea container
docker compose up -d --force-recreate

# Rimuovi immagini vecchie
docker image prune -a
```

### Aggiorna Codice

Se hai modificato file Python (come `app.py` o moduli in `rag_pipeline/`):

```bash
# Metodo 1: Usa lo script con flag --build (o -b)
./app_setup.sh --build

# Metodo 2: Rebuild manuale
docker compose build rag-app
docker compose up -d
```

Se hai solo aggiornato il codice da git senza modifiche:

```bash
# Pull codice da git
git pull origin main

# Rebuild immagine
./app_setup.sh -b
```

### Pulizia Sistema

```bash
# Rimuovi container fermi
docker container prune

# Rimuovi immagini non usate
docker image prune -a

# Rimuovi volumi non usati
docker volume prune

# Pulizia completa (ATTENZIONE: rimuove TUTTO)
docker system prune -a --volumes
```

---

## 📊 Performance Tips

### CPU Mode (Senza GPU)

Per ottimizzare su CPU:

1. Usa modelli quantizzati più piccoli:
   ```bash
   docker exec rag-ollama ollama pull llama3.2:1b-instruct-q4_K_M
   ```

2. Riduci `context_window` in `settings.py`:
   ```python
   context_window: int = 2048  # invece di 4096
   ```

### GPU Mode (Con NVIDIA)

1. Installa [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Verifica supporto:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Il `docker-compose.yml` è già configurato per GPU

---

## 🆘 Supporto

### Log Dettagliati

```bash
# Abilita debug logging
docker compose down
docker compose up  # Senza -d per vedere output in tempo reale
```

### Informazioni Sistema

```bash
# Info Docker
docker version
docker compose version

# Info sistema
docker info

# Risorse
docker stats --no-stream
```

### Report Issue

Quando segnali un problema, includi:
1. Output di `docker compose logs`
2. Configurazione sistema (`docker info`)
3. Versione Docker (`docker version`)
4. File `docker-compose.yml` (se modificato)

---

## 📚 Risorse

- [Docker Documentation](https://docs.docker.com/)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

---

**Buon utilizzo! 🚀**