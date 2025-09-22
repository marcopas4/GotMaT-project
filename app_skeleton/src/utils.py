"""
Utility Functions
=================

Modulo contenente funzioni di utilità per l'applicazione RAG.
Include helper per gestione file, formattazione, validazione, etc.
"""

import os
import tempfile
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import streamlit as st
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file) -> str:
    """
    Salva un file caricato dall'utente in una cartella temporanea.
    
    Args:
        uploaded_file: File caricato tramite Streamlit
        
    Returns:
        Percorso del file salvato temporaneamente
    """
    try:
        # Crea un file temporaneo con l'estensione corretta
        file_extension = Path(uploaded_file.name).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        logger.info(f"File salvato temporaneamente: {uploaded_file.name} -> {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Errore nel salvataggio del file {uploaded_file.name}: {str(e)}")
        raise

def get_file_type(filename: str) -> str:
    """
    Determina il tipo di file dall'estensione.
    
    Args:
        filename: Nome del file
        
    Returns:
        Tipo di file normalizzato
    """
    file_extension = Path(filename).suffix.lower()
    
    type_mapping = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',  # Trattiamo DOC come DOCX
        '.txt': 'txt',
        '.jpg': 'jpg',
        '.jpeg': 'jpg',
        '.png': 'png',
        '.bmp': 'png',  # Trattiamo BMP come PNG
        '.gif': 'png',  # Trattiamo GIF come PNG
    }
    
    return type_mapping.get(file_extension, 'unknown')

def validate_file_upload(uploaded_file, max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Valida un file caricato dall'utente.
    
    Args:
        uploaded_file: File da validare
        max_size_mb: Dimensione massima in MB
        
    Returns:
        Dizionario con risultato validazione e dettagli
    """
    validation_result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'file_info': {}
    }
    
    try:
        # Informazioni base del file
        file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': get_file_type(uploaded_file.name)
        }
        validation_result['file_info'] = file_info
        
        # Controllo dimensione
        max_size_bytes = max_size_mb * 1024 * 1024
        if uploaded_file.size > max_size_bytes:
            validation_result['errors'].append(
                f"File troppo grande: {uploaded_file.size / (1024*1024):.1f}MB > {max_size_mb}MB"
            )
        
        # Controllo tipo file supportato
        if file_info['type'] == 'unknown':
            validation_result['errors'].append(
                f"Tipo di file non supportato: {Path(uploaded_file.name).suffix}"
            )
        
        # Controllo nome file
        if not uploaded_file.name or len(uploaded_file.name.strip()) == 0:
            validation_result['errors'].append("Nome file vuoto")
        
        # Controllo caratteri speciali nel nome
        if re.search(r'[<>:"/\\|?*]', uploaded_file.name):
            validation_result['warnings'].append(
                "Il nome del file contiene caratteri speciali che potrebbero causare problemi"
            )
        
        # File vuoto
        if uploaded_file.size == 0:
            validation_result['errors'].append("File vuoto")
        
        # Se nessun errore, il file è valido
        validation_result['valid'] = len(validation_result['errors']) == 0
        
    except Exception as e:
        validation_result['errors'].append(f"Errore nella validazione: {str(e)}")
    
    return validation_result

def format_file_size(size_bytes: int) -> str:
    """
    Formatta la dimensione del file in modo leggibile.
    
    Args:
        size_bytes: Dimensione in bytes
        
    Returns:
        Stringa formattata (es: "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def format_response(response: str) -> str:
    """
    Formatta una risposta per la visualizzazione nell'interfaccia.
    
    Args:
        response: Testo della risposta
        
    Returns:
        Risposta formattata con HTML/Markdown
    """
    # Sostituzioni per migliorare la leggibilità
    formatted = response
    
    # Converti elenchi numerati in HTML
    formatted = re.sub(r'^(\d+)\.\s+(.+)$', r'<strong>\1.</strong> \2', formatted, flags=re.MULTILINE)
    
    # Evidenzia sezioni con **testo**
    formatted = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted)
    
    # Evidenzia testo tra backticks
    formatted = re.sub(r'`(.+?)`', r'<code>\1</code>', formatted)
    
    # Converti newline in <br> per HTML
    formatted = formatted.replace('\n', '<br>')
    
    return formatted

def generate_document_id(content: str, filename: str) -> str:
    """
    Genera un ID univoco per un documento basato su contenuto e nome.
    
    Args:
        content: Contenuto del documento
        filename: Nome del file
        
    Returns:
        ID univoco del documento
    """
    # Combina contenuto e filename per l'hash
    combined = f"{filename}_{content[:1000]}"  # Prima parte del contenuto
    
    # Genera hash MD5
    doc_hash = hashlib.md5(combined.encode()).hexdigest()
    
    # Timestamp per unicità
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"doc_{timestamp}_{doc_hash[:8]}"

def clean_text_for_display(text: str, max_length: int = 200) -> str:
    """
    Pulisce e tronca il testo per la visualizzazione.
    
    Args:
        text: Testo da pulire
        max_length: Lunghezza massima
        
    Returns:
        Testo pulito e troncato
    """
    # Rimuovi caratteri speciali e spazi multipli
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Tronca se necessario
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Estrae parole chiave da un testo.
    
    Args:
        text: Testo da analizzare
        max_keywords: Numero massimo di parole chiave
        
    Returns:
        Lista di parole chiave
        
    PLACEHOLDER - DA MIGLIORARE:
    - Utilizzare NLP avanzato
    - Rimozione stop words
    - Analisi frequenza TF-IDF
    """
    # TODO: Implementare estrazione avanzata con spaCy o NLTK
    
    # Estrazione semplice per ora
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
    
    # Stop words italiane base
    stop_words = {
        'essere', 'avere', 'fare', 'dire', 'quello', 'questo', 'dove', 'quando',
        'come', 'perché', 'cosa', 'molto', 'tutto', 'anche', 'ancora', 'già',
        'sempre', 'mai', 'più', 'meno', 'bene', 'male', 'solo', 'prima',
        'dopo', 'sopra', 'sotto', 'dentro', 'fuori', 'insieme', 'contro',
        'sulla', 'della', 'nella', 'alla', 'dalla', 'dalla', 'questa',
        'quello', 'questi', 'quelle'
    }
    
    # Filtra stop words e conta frequenza
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Ordina per frequenza e prendi le top
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in keywords[:max_keywords]]

def sanitize_filename(filename: str) -> str:
    """
    Pulisce un nome file da caratteri non validi.
    
    Args:
        filename: Nome file da pulire
        
    Returns:
        Nome file sanitizzato
    """
    # Rimuovi caratteri non validi
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Rimuovi spazi multipli
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Assicurati che non sia vuoto
    if not sanitized:
        sanitized = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return sanitized

def create_backup_filename(original_path: str) -> str:
    """
    Crea un nome file di backup.
    
    Args:
        original_path: Percorso file originale
        
    Returns:
        Percorso file di backup
    """
    path = Path(original_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
    return str(path.parent / backup_name)

def log_user_action(action: str, details: Dict[str, Any] = None):
    """
    Logga un'azione dell'utente per analytics/debugging.
    
    Args:
        action: Tipo di azione
        details: Dettagli aggiuntivi
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'details': details or {}
    }
    
    # Per ora loggiamo solo nel logger, ma potrebbe essere esteso
    # per salvare in database o file di analytics
    logger.info(f"User action: {action} - {details}")

def validate_query(query: str) -> Dict[str, Any]:
    """
    Valida una query dell'utente.
    
    Args:
        query: Query da validare
        
    Returns:
        Dizionario con risultato validazione
    """
    validation = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'suggestions': []
    }
    
    # Controlli base
    if not query or not query.strip():
        validation['errors'].append("La domanda non può essere vuota")
        return validation
    
    if len(query.strip()) < 3:
        validation['errors'].append("La domanda è troppo corta (minimo 3 caratteri)")
        return validation
    
    if len(query) > 1000:
        validation['warnings'].append("La domanda è molto lunga, considera di dividerla")
    
    # Controllo caratteri strani
    if len(re.findall(r'[^\w\s\?\.\!\,\;\:\(\)\-\']', query)) > len(query) * 0.1:
        validation['warnings'].append("La domanda contiene molti caratteri speciali")
    
    # Suggerimenti per migliorare la query
    if '?' not in query:
        validation['suggestions'].append("Considera di formulare la domanda in forma interrogativa")
    
    if len(query.split()) < 3:
        validation['suggestions'].append("Prova a essere più specifico nella tua domanda")
    
    # Se nessun errore, è valida
    validation['valid'] = len(validation['errors']) == 0
    
    return validation

def chunk_text_smart(text: str, max_chunk_size: int = 1000, overlap_size: int = 200) -> List[Dict[str, Any]]:
    """
    Suddivide il testo in chunk intelligenti preservando la semantica.
    
    Args:
        text: Testo da suddividere
        max_chunk_size: Dimensione massima chunk
        overlap_size: Sovrapposizione tra chunk
        
    Returns:
        Lista di chunk con metadati
        
    PLACEHOLDER - DA MIGLIORARE:
    - Chunking basato su paragrafi/frasi
    - Preservazione contesto semantico
    - Chunking adattivo basato sul contenuto
    """
    chunks = []
    
    # TODO: Implementare chunking più intelligente
    # - Dividere per paragrafi quando possibile
    # - Mantenere frasi intere
    # - Considerare struttura del documento
    
    # Chunking semplice per caratteri per ora
    for i in range(0, len(text), max_chunk_size - overlap_size):
        chunk_text = text[i:i + max_chunk_size]
        
        if chunk_text.strip():
            chunks.append({
                'text': chunk_text,
                'start': i,
                'end': min(i + max_chunk_size, len(text)),
                'chunk_id': len(chunks),
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text)
            })
    
    return chunks

def get_system_info() -> Dict[str, Any]:
    """
    Restituisce informazioni sul sistema per debugging.
    
    Returns:
        Dizionario con informazioni sistema
    """
    import platform
    import psutil
    
    return {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:\\').percent
    }

class ProgressTracker:
    """
    Classe helper per tracciare il progresso di operazioni lunghe.
    """
    
    def __init__(self, total_steps: int, description: str = "Processando..."):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step_increment: int = 1, status_message: str = None):
        """Aggiorna il progresso."""
        self.current_step += step_increment
        progress = min(self.current_step / self.total_steps, 1.0)
        
        self.progress_bar.progress(progress)
        
        if status_message:
            self.status_text.text(status_message)
        else:
            self.status_text.text(f"{self.description} ({self.current_step}/{self.total_steps})")
    
    def complete(self, final_message: str = "Completato!"):
        """Completa il progresso."""
        self.progress_bar.progress(1.0)
        self.status_text.text(final_message)
    
    def cleanup(self):
        """Rimuove la progress bar dall'interfaccia."""
        self.progress_bar.empty()
        self.status_text.empty()

# Costanti utili
SUPPORTED_FILE_TYPES = {
    'pdf': 'PDF Document',
    'docx': 'Word Document',
    'txt': 'Text Document',
    'jpg': 'JPEG Image',
    'png': 'PNG Image'
}

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 50

# Regex patterns utili
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_PATTERN = re.compile(r'^\+?[\d\s\-\(\)]{7,}$')
DATE_PATTERN = re.compile(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}')