@echo off
REM Setup script per RAG Prefettura - Windows
REM Scarica modelli Ollama e prepara l'ambiente

echo ========================================
echo    RAG Prefettura - Setup Script
echo ========================================
echo.

REM Controlla se Docker è in esecuzione
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker non e' in esecuzione!
    echo Avvia Docker Desktop e riprova.
    pause
    exit /b 1
)

echo [OK] Docker e' in esecuzione
echo.

REM Avvia i container
echo [INFO] Avvio container con docker compose...
docker compose up -d

echo.
echo [INFO] Attendo che Ollama sia pronto...
timeout /t 10 /nobreak >nul

REM Attendi che Ollama sia pronto
:wait_ollama
set /a attempt=0
:wait_loop
if %attempt% geq 30 (
    echo [ERROR] Timeout: Ollama non risponde
    echo Verifica i log con: docker compose logs ollama
    pause
    exit /b 1
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    set /a attempt+=1
    echo Tentativo %attempt%/30...
    timeout /t 2 /nobreak >nul
    goto wait_loop
)

echo [OK] Ollama e' pronto!
echo.

REM Scarica modelli
echo [INFO] Download modelli Ollama...
echo.

echo [INFO] Downloading llama3.2:3b-instruct-q4_K_M...
docker exec rag-ollama ollama pull llama3.2:3b-instruct-q4_K_M

echo.
echo [OK] Modello LLM scaricato!
echo.

echo [INFO] Embedding model (nomic-ai/nomic-embed-text-v1.5) verra' scaricato al primo utilizzo
echo.

echo ========================================
echo           Setup completato!
echo ========================================
echo.

echo [INFO] Stato servizi:
docker compose ps

echo.
echo [INFO] Accedi all'applicazione:
echo    -^> http://localhost:8501
echo.
echo [INFO] Comandi utili:
echo    - Vedere i log:        docker compose logs -f
echo    - Fermare servizi:     docker compose down
echo    - Riavviare:           docker compose restart
echo    - Shell nel container: docker exec -it rag-streamlit-app bash
echo.
echo [WARNING] Al primo avvio, il download degli embedding potrebbe richiedere qualche minuto
echo.

pause