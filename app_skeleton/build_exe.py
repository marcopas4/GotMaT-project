"""
Build Script per Generare l'Eseguibile Windows
==============================================

Questo script crea un file .exe standalone dell'applicazione RAG Prefettura
utilizzando PyInstaller.

Prerequisiti:
    pip install pyinstaller

Utilizzo:
    python build_exe.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Configurazione
APP_NAME = "RAG_Prefettura"
MAIN_SCRIPT = "app.py"
ICON_FILE = "icon.ico"  # Opzionale - aggiungi un'icona se disponibile

def clean_build_folders():
    """Pulisce le cartelle di build precedenti"""
    print("🧹 Pulizia cartelle di build...")
    folders_to_clean = ['build', 'dist', '__pycache__']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"   ✓ Rimossa cartella: {folder}")
    
    # Rimuovi file .spec precedenti
    spec_files = list(Path('.').glob('*.spec'))
    for spec_file in spec_files:
        os.remove(spec_file)
        print(f"   ✓ Rimosso file: {spec_file}")

def create_spec_file():
    """Crea un file .spec personalizzato per PyInstaller"""
    print("\n📝 Creazione file .spec personalizzato...")
    
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Analisi dipendenze
a = Analysis(
    ['{MAIN_SCRIPT}'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('data', 'data'),
        ('.streamlit', '.streamlit'),
    ],
    hiddenimports=[
        'streamlit',
        'streamlit.runtime.scriptrunner.magic_funcs',
        'streamlit_extras',
        'PIL',
        'PIL._imaging',
        'pytesseract',
        'torch',
        'transformers',
        'sentence_transformers',
        'faiss',
        'numpy',
        'scipy',
        'pandas',
        'spacy',
        'nltk',
        'loguru',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{APP_NAME}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # True per vedere output console, False per nasconderla
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{ICON_FILE}' if os.path.exists('{ICON_FILE}') else None,
)
"""
    
    spec_filename = f"{APP_NAME}.spec"
    with open(spec_filename, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print(f"   ✓ Creato file: {spec_filename}")
    return spec_filename

def build_executable(spec_file):
    """Esegue PyInstaller per creare l'eseguibile"""
    print(f"\n🔨 Building eseguibile con PyInstaller...")
    print("   Questo processo potrebbe richiedere diversi minuti...\n")
    
    try:
        # Esegui PyInstaller
        cmd = [
            sys.executable,
            '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            spec_file
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("   ✓ Build completata con successo!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Errore durante il build:")
        print(e.stderr)
        return False

def create_launcher_bat():
    """Crea un file .bat per lanciare l'applicazione facilmente"""
    print("\n📄 Creazione launcher batch...")
    
    bat_content = f"""@echo off
echo Avvio RAG Prefettura...
echo.

REM Imposta la directory di lavoro
cd /d "%~dp0"

REM Avvia l'applicazione
start "" "{APP_NAME}.exe"

REM Attendi 3 secondi
timeout /t 3 /nobreak >nul

REM Apri il browser sulla porta di Streamlit
start http://localhost:8501

exit
"""
    
    bat_filename = "dist/Avvia_RAG_Prefettura.bat"
    os.makedirs('dist', exist_ok=True)
    
    with open(bat_filename, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print(f"   ✓ Creato launcher: {bat_filename}")

def create_readme():
    """Crea un README per la distribuzione"""
    print("\n📖 Creazione README per distribuzione...")
    
    readme_content = """# RAG Prefettura - Applicazione Desktop

## Come Avviare l'Applicazione

### Metodo 1: Launcher Automatico (Consigliato)
1. Fai doppio click su `Avvia_RAG_Prefettura.bat`
2. L'applicazione si avvierà automaticamente e il browser si aprirà

### Metodo 2: Manuale
1. Fai doppio click su `RAG_Prefettura.exe`
2. Attendi che si avvii (potrebbero volerci alcuni secondi)
3. Apri il browser e vai su: http://localhost:8501

## Requisiti di Sistema

- **Sistema Operativo**: Windows 10 o superiore
- **RAM**: Minimo 8GB (16GB consigliati)
- **Spazio Disco**: Almeno 2GB liberi
- **CPU**: Processore multi-core (4+ core consigliati)

## Struttura File

```
dist/
├── RAG_Prefettura.exe          # Eseguibile principale
├── Avvia_RAG_Prefettura.bat    # Launcher automatico
├── README.txt                   # Questo file
├── src/                         # Codice sorgente
├── data/                        # Dati e knowledge base
└── .streamlit/                  # Configurazioni
```

## Risoluzione Problemi

### L'applicazione non si avvia
- Verifica che il firewall non blocchi l'eseguibile
- Verifica di avere i permessi di esecuzione
- Prova a eseguire come amministratore

### Il browser non si apre automaticamente
- Apri manualmente il browser
- Vai su: http://localhost:8501

### Errori di memoria
- Chiudi altre applicazioni
- Verifica di avere almeno 8GB di RAM disponibili

### Porta già in uso
- Assicurati che nessun'altra applicazione usi la porta 8501
- Puoi cambiare la porta nel file `.streamlit/config.toml`

## Supporto

Per assistenza tecnica, contattare il team di sviluppo.

Versione: 1.0.0
Data compilazione: {datetime.now().strftime('%d/%m/%Y')}
"""
    
    readme_filename = "dist/README.txt"
    with open(readme_filename, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"   ✓ Creato README: {readme_filename}")

def copy_essential_files():
    """Copia file essenziali nella cartella dist"""
    print("\n📦 Copia file essenziali...")
    
    essential_dirs = ['data', '.streamlit']
    
    for dir_name in essential_dirs:
        if os.path.exists(dir_name):
            dest = os.path.join('dist', dir_name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(dir_name, dest)
            print(f"   ✓ Copiato: {dir_name}")

def create_config_if_missing():
    """Crea configurazione Streamlit se non esiste"""
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    if not os.path.exists(config_file):
        print("\n⚙️ Creazione configurazione Streamlit...")
        config_content = """[server]
port = 8501
headless = true
runOnSave = false

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "#1f4e79"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f8ff"
textColor = "#262730"
font = "sans serif"
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"   ✓ Creato: {config_file}")

def main():
    """Funzione principale per il build"""
    print("=" * 70)
    print("     RAG PREFETTURA - BUILD SCRIPT PER ESEGUIBILE WINDOWS")
    print("=" * 70)
    print()
    
    # Verifica che PyInstaller sia installato
    try:
        import PyInstaller
    except ImportError:
        print("❌ PyInstaller non trovato!")
        print("   Installa con: pip install pyinstaller")
        sys.exit(1)
    
    # Verifica che il file principale esista
    if not os.path.exists(MAIN_SCRIPT):
        print(f"❌ File principale non trovato: {MAIN_SCRIPT}")
        sys.exit(1)
    
    try:
        # Step 1: Pulizia
        clean_build_folders()
        
        # Step 2: Crea configurazione se mancante
        create_config_if_missing()
        
        # Step 3: Crea file .spec
        spec_file = create_spec_file()
        
        # Step 4: Build eseguibile
        success = build_executable(spec_file)
        
        if not success:
            print("\n❌ Build fallito!")
            sys.exit(1)
        
        # Step 5: Copia file essenziali
        copy_essential_files()
        
        # Step 6: Crea launcher e README
        create_launcher_bat()
        create_readme()
        
        # Successo!
        print("\n" + "=" * 70)
        print("✅ BUILD COMPLETATO CON SUCCESSO!")
        print("=" * 70)
        print()
        print(f"📁 L'eseguibile si trova in: dist/{APP_NAME}.exe")
        print(f"🚀 Per avviare: Fai doppio click su dist/Avvia_RAG_Prefettura.bat")
        print()
        print("📦 File nella cartella dist/:")
        print(f"   - {APP_NAME}.exe")
        print("   - Avvia_RAG_Prefettura.bat")
        print("   - README.txt")
        print("   - src/ (codice sorgente)")
        print("   - data/ (knowledge base)")
        print()
        print("💡 Puoi distribuire l'intera cartella 'dist/' agli utenti")
        print()
        
    except Exception as e:
        print(f"\n❌ Errore durante il build: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()
