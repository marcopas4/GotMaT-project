#!/usr/bin/env python3
"""
Setup Script per RAG Prefettura
===============================

Script di installazione e configurazione automatica per l'applicazione RAG.
Gestisce l'installazione delle dipendenze, configurazione dell'ambiente,
e setup iniziale del sistema.

Usage:
    python setup.py install      # Installazione completa
    python setup.py configure    # Solo configurazione
    python setup.py test         # Test del sistema
    python setup.py clean        # Pulizia cache e temp files
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
import requests
import zipfile
from urllib.parse import urlparse
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPrefetturaSetup:
    """Classe per gestire l'installazione e configurazione del sistema."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.temp_dir = self.project_root / "temp"
        
    def create_directories(self):
        """Crea le directory necessarie."""
        logger.info("Creazione directory del progetto...")
        
        directories = [
            self.data_dir / "knowledge_base",
            self.data_dir / "uploads", 
            self.data_dir / "faiss_index",
            self.models_dir,
            self.temp_dir,
            self.project_root / "logs",
            self.project_root / "src",
            self.project_root / ".streamlit"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Creata directory: {directory}")
    
    def install_python_dependencies(self):
        """Installa le dipendenze Python."""
        logger.info("Installazione dipendenze Python...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ])
            
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            
            logger.info("✓ Dipendenze Python installate con successo")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Errore nell'installazione dipendenze: {e}")
            return False
        
        return True
    
    def install_spacy_model(self):
        """Installa il modello spaCy per l'italiano."""
        logger.info("Installazione modello spaCy italiano...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "it_core_news_sm"
            ])
            logger.info("✓ Modello spaCy italiano installato")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠ Errore installazione modello spaCy: {e}")
            logger.info("Puoi installarlo manualmente con: python -m spacy download it_core_news_sm")
    
    def download_nltk_data(self):
        """Scarica i dati NLTK necessari."""
        logger.info("Download dati NLTK...")
        
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logger.info("✓ Dati NLTK scaricati")
            
        except Exception as e:
            logger.warning(f"⚠ Errore download NLTK: {e}")
    
    def check_tesseract(self):
        """Verifica installazione Tesseract OCR."""
        logger.info("Verifica Tesseract OCR...")
        
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ Tesseract OCR disponibile")
                return True
            else:
                logger.warning("⚠ Tesseract non trovato")
                return False
                
        except FileNotFoundError:
            logger.warning("⚠ Tesseract OCR non installato")
            logger.info("Installa Tesseract da: https://github.com/tesseract-ocr/tesseract")
            return False
    
    def create_sample_knowledge_base(self):
        """Crea una knowledge base di esempio."""
        logger.info("Creazione knowledge base di esempio...")
        
        kb_dir = self.data_dir / "knowledge_base"
        
        # Crea alcuni file di esempio
        sample_files = {
            "illeciti_amministrativi_guida.txt": """
GUIDA AGLI ILLECITI AMMINISTRATIVI

Gli illeciti amministrativi sono violazioni di norme amministrative che comportano 
l'applicazione di sanzioni pecuniarie o altre misure amministrative.

TIPI PRINCIPALI:
1. Illeciti tributari
2. Illeciti ambientali  
3. Illeciti edilizi
4. Illeciti commerciali

PROCEDIMENTO SANZIONATORIO:
1. Accertamento della violazione
2. Notificazione dell'atto di contestazione
3. Eventuale audizione dell'interessato
4. Determinazione della sanzione
5. Notificazione dell'ordinanza-ingiunzione

TERMINI E SCADENZE:
- Presentazione memorie: 30 giorni dalla notifica
- Ricorso amministrativo: 60 giorni
- Ricorso giurisdizionale: 120 giorni
            """,
            
            "codice_procedura_amministrativa.txt": """
CODICE DEL PROCEDIMENTO AMMINISTRATIVO

Art. 1 - Principi generali
L'attività amministrativa persegue i fini determinati dalla legge ed è retta 
da criteri di economicità, di efficacia, di imparzialità, di pubblicità e 
di trasparenza.

Art. 2 - Definizioni  
Ai fini della presente legge si intende per:
a) "amministrazione pubblica": tutti i soggetti di diritto pubblico
b) "procedimento": la sequenza di atti preordinati all'emanazione di un 
   provvedimento finale

Art. 7 - Comunicazione di avvio del procedimento
L'avvio del procedimento è comunicato ai soggetti nei confronti dei quali 
il provvedimento finale è destinato a produrre effetti diretti.
            """,
            
            "sanzioni_amministrative_pecuniarie.txt": """
SANZIONI AMMINISTRATIVE PECUNIARIE

Le sanzioni amministrative pecuniarie sono disciplinate dalla Legge 689/1981.

CRITERI PER LA DETERMINAZIONE:
1. Gravità della violazione
2. Opera svolta dall'agente per l'eliminazione o attenuazione delle conseguenze
3. Personalità dell'agente
4. Condizioni economiche del trasgressore

RIDUZIONE DELLA SANZIONE:
- Pagamento entro 60 giorni: riduzione del 30%
- Pagamento entro 30 giorni: riduzione del 50% (se previsto)

RATEIZZAZIONE:
Possibile in presenza di comprovate difficoltà economiche, 
fino a 24 rate mensili.
            """
        }
        
        for filename, content in sample_files.items():
            file_path = kb_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✓ Creato file di esempio: {filename}")
    
    def create_env_file(self):
        """Crea file .env con configurazioni."""
        logger.info("Creazione file di configurazione...")
        
        env_content = f"""# RAG Prefettura Configuration
# Personalizza questi valori secondo le tue necessità

# Paths
MODEL_PATH={self.models_dir}/fine_tuned_model
KNOWLEDGE_BASE_PATH={self.data_dir}/knowledge_base
VECTOR_DB_PATH={self.data_dir}/faiss_index
UPLOAD_PATH={self.data_dir}/uploads
TEMP_PATH={self.temp_dir}
LOG_PATH={self.project_root}/logs

# Model Settings
MAX_TOKENS=512
TEMPERATURE=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# File Upload Settings
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=pdf,docx,txt,jpg,png

# Streamlit Settings
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost

# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG=false

# Security (per deployment)
ENABLE_AUTH=false
SECRET_KEY=your-secret-key-here
"""
        
        env_path = self.project_root / ".env"
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info(f"✓ File di configurazione creato: {env_path}")
    
    def test_installation(self):
        """Testa l'installazione."""
        logger.info("Test dell'installazione...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test import principali
        test_imports = [
            'streamlit',
            'numpy', 
            'pandas',
            'transformers',
            'sentence_transformers',
            'faiss',
            'PIL',
            'docx'
        ]
        
        for module in test_imports:
            total_tests += 1
            try:
                __import__(module)
                logger.info(f"✓ {module} importato correttamente")
                tests_passed += 1
            except ImportError as e:
                logger.error(f"✗ Errore importazione {module}: {e}")
        
        # Test directory
        total_tests += 1
        if all(d.exists() for d in [self.data_dir, self.models_dir]):
            logger.info("✓ Directory create correttamente")
            tests_passed += 1
        else:
            logger.error("✗ Errore nella creazione delle directory")
        
        # Test file configurazione
        total_tests += 1
        if (self.project_root / ".env").exists():
            logger.info("✓ File di configurazione presente")
            tests_passed += 1
        else:
            logger.error("✗ File di configurazione mancante")
        
        logger.info(f"\nTest completati: {tests_passed}/{total_tests} superati")
        
        if tests_passed == total_tests:
            logger.info("🎉 Installazione completata con successo!")
            return True
        else:
            logger.error("❌ Installazione incompleta. Controlla gli errori sopra.")
            return False
    
    def clean_cache(self):
        """Pulisce cache e file temporanei."""
        logger.info("Pulizia cache e file temporanei...")
        
        # Directory da pulire
        cache_dirs = [
            self.temp_dir,
            self.project_root / "__pycache__",
            self.project_root / "src" / "__pycache__",
            self.project_root / ".pytest_cache"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"✓ Rimossa cache: {cache_dir}")
        
        # Ricrea directory necessarie
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info("✓ Pulizia completata")
    
    def print_next_steps(self):
        """Stampa i passi successivi."""
        print("\n" + "="*60)
        print("🎉 INSTALLAZIONE COMPLETATA!")
        print("="*60)
        
        print("\n📋 PROSSIMI PASSI:")
        print("\n1. I tuoi colleghi devono implementare i moduli:")
        print("   - src/document_processor.py (estrazione testo)")
        print("   - src/vector_store.py (FAISS e embeddings)")
        print("   - src/llm_handler.py (modello fine-tuned)")
        
        print("\n2. Configurazione:")
        print(f"   - Modifica {self.project_root}/.env per le tue impostazioni")
        print(f"   - Aggiungi documenti in {self.data_dir}/knowledge_base/")
        print(f"   - Posiziona il modello fine-tuned in {self.models_dir}/")
        
        print("\n3. Avvio applicazione:")
        print("   # Locale:")
        print("   streamlit run app.py")
        print("   ")
        print("   # Docker:")
        print("   docker-compose up --build")
        
        print("\n4. Accesso:")
        print("   http://localhost:8501")
        
        print("\n" + "="*60)
        print("📚 Documentazione: README.md")
        print("🐛 Issues: Controlla i log in logs/")
        print("="*60)


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description='Setup RAG Prefettura Application')
    parser.add_argument('command', choices=['install', 'configure', 'test', 'clean'],
                       help='Comando da eseguire')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Salta installazione dipendenze Python')
    parser.add_argument('--skip-models', action='store_true', 
                       help='Salta download modelli')
    
    args = parser.parse_args()
    
    setup = RAGPrefetturaSetup()
    
    if args.command == 'install':
        logger.info("🚀 Avvio installazione RAG Prefettura...")
        
        setup.create_directories()
        
        if not args.skip_deps:
            if not setup.install_python_dependencies():
                sys.exit(1)
        
        if not args.skip_models:
            setup.install_spacy_model()
            setup.download_nltk_data()
        
        setup.check_tesseract()
        setup.create_sample_knowledge_base()
        setup.create_env_file()
        
        if setup.test_installation():
            setup.print_next_steps()
        else:
            sys.exit(1)
    
    elif args.command == 'configure':
        logger.info("⚙️ Configurazione sistema...")
        setup.create_directories()
        setup.create_env_file()
        logger.info("✓ Configurazione completata")
    
    elif args.command == 'test':
        logger.info("🧪 Test sistema...")
        setup.test_installation()
    
    elif args.command == 'clean':
        logger.info("🧹 Pulizia sistema...")
        setup.clean_cache()

if __name__ == "__main__":
    main()