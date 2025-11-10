#!/bin/bash

# Setup script per RAG Prefettura
# Scarica modelli Ollama e prepara l'ambiente

set -e

echo "🚀 RAG Prefettura - Setup Script"
echo "================================="
echo ""

# Colori per output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Controlla se Docker è in esecuzione
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker non è in esecuzione!${NC}"
    echo "Avvia Docker Desktop e riprova."
    exit 1
fi

echo -e "${GREEN}✅ Docker è in esecuzione${NC}"
echo ""

# Avvia i container
echo "📦 Avvio container con docker compose..."
docker compose up -d

echo ""
echo "⏳ Attendo che Ollama sia pronto..."
sleep 10

# Funzione per controllare se Ollama è ready
wait_for_ollama() {
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama è pronto!${NC}"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "Tentativo $attempt/$max_attempts..."
        sleep 2
    done
    
    echo -e "${RED}❌ Timeout: Ollama non risponde${NC}"
    return 1
}

if ! wait_for_ollama; then
    echo "Verifica i log con: docker compose logs ollama"
    exit 1
fi

echo ""
echo "📥 Download modelli Ollama..."
echo ""

# Scarica il modello LLM principale
echo -e "${YELLOW}Downloading llama3.2:3b-instruct-q4_K_M...${NC}"
docker exec rag-ollama ollama pull llama3.2:3b-instruct-q4_K_M

echo ""
echo -e "${GREEN}✅ Modello LLM scaricato!${NC}"

# Scarica embedding model (Ollama non supporta direttamente HuggingFace)
# L'embedding verrà gestito da sentence-transformers nel container Python
echo ""
echo -e "${YELLOW}ℹ️  Embedding model (nomic-ai/nomic-embed-text-v1.5) verrà scaricato al primo utilizzo${NC}"

echo ""
echo "🎉 Setup completato!"
echo ""
echo "📊 Stato servizi:"
docker compose ps

echo ""
echo "🌐 Accedi all'applicazione:"
echo -e "${GREEN}   → http://localhost:8501${NC}"
echo ""
echo "📝 Comandi utili:"
echo "   • Vedere i log:      docker compose logs -f"
echo "   • Fermare servizi:   docker compose down"
echo "   • Riavviare:         docker compose restart"
echo "   • Shell nel container: docker exec -it rag-streamlit-app bash"
echo ""
echo -e "${YELLOW}⚠️  Al primo avvio, il download degli embedding potrebbe richiedere qualche minuto${NC}"
echo ""