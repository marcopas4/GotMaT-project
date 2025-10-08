import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_handler import LLMHandler
from src.utils import save_uploaded_file, get_file_type, format_response

# Configurazione pagina
st.set_page_config(
    page_title="RAG Prefettura - Assistente per Illeciti Amministrativi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per migliorare l'aspetto
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    
    .upload-section {
        background-color: #fafafa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #ddd;
        text-align: center;
        margin: 1rem 0;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inizializza lo stato della sessione"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = None
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = None

def initialize_components():
    """Inizializza i componenti principali"""
    try:
        if st.session_state.document_processor is None:
            with st.spinner("Inizializzazione processore documenti..."):
                st.session_state.document_processor = DocumentProcessor()
        
        if st.session_state.vector_store is None:
            with st.spinner("Inizializzazione database vettoriale..."):
                st.session_state.vector_store = VectorStore()
                # Carica la knowledge base preesistente
                st.session_state.vector_store.load_knowledge_base()
        
        if st.session_state.llm_handler is None:
            with st.spinner("Inizializzazione modello linguistico..."):
                st.session_state.llm_handler = LLMHandler()
        
        return True
    except Exception as e:
        st.error(f"Errore nell'inizializzazione: {str(e)}")
        return False

def handle_file_upload(uploaded_files):
    """Gestisce l'upload dei file"""
    if not uploaded_files:
        return
    
    successful_uploads = []
    failed_uploads = []
    
    for uploaded_file in uploaded_files:
        try:
            # Salva il file temporaneamente
            temp_path = save_uploaded_file(uploaded_file)
            
            # Determina il tipo di file
            file_type = get_file_type(uploaded_file.name)
            
            # Processa il documento
            with st.spinner(f"Processando {uploaded_file.name}..."):
                processed_content = st.session_state.document_processor.process_document(
                    temp_path, file_type
                )
            
            # Aggiungi al vector store
            document_id = st.session_state.vector_store.add_document(
                content=processed_content,
                metadata={
                    'filename': uploaded_file.name,
                    'file_type': file_type,
                    'size': uploaded_file.size
                }
            )
            
            successful_uploads.append({
                'filename': uploaded_file.name,
                'file_type': file_type,
                'document_id': document_id,
                'size': uploaded_file.size
            })
            
            # Pulizia file temporaneo
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            failed_uploads.append({
                'filename': uploaded_file.name,
                'error': str(e)
            })
    
    # Aggiorna lo stato
    st.session_state.uploaded_documents.extend(successful_uploads)
    
    # Mostra risultati
    if successful_uploads:
        st.markdown('<div class="status-box status-success">✅ Documenti caricati con successo:</div>', unsafe_allow_html=True)
        for doc in successful_uploads:
            st.write(f"• {doc['filename']} ({doc['file_type']})")
    
    if failed_uploads:
        st.markdown('<div class="status-box status-error">❌ Errori nel caricamento:</div>', unsafe_allow_html=True)
        for doc in failed_uploads:
            st.write(f"• {doc['filename']}: {doc['error']}")

def handle_query(query: str):
    """Gestisce le query dell'utente"""
    if not query.strip():
        return
    
    # Aggiungi messaggio utente alla cronologia
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": st.session_state.get('current_time', '')
    })
    
    try:
        with st.spinner("Elaborando la tua domanda..."):
            # Recupera contesto rilevante dai documenti caricati
            # La knowledge base è già integrata nel modello fine-tuned
            context = st.session_state.vector_store.search_uploaded_documents(query)
            
            # Genera risposta con LLM (usa sempre i documenti caricati)
            response = st.session_state.llm_handler.generate_response(
                query=query,
                context=context,
                use_uploaded_docs=True
            )
            
            # Aggiungi risposta alla cronologia
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "context": context,
                "timestamp": st.session_state.get('current_time', '')
            })
    
    except Exception as e:
        st.error(f"Errore nell'elaborazione della query: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Mi dispiace, si è verificato un errore: {str(e)}",
            "timestamp": st.session_state.get('current_time', '')
        })

def display_chat_history():
    """Mostra la cronologia della chat"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>Tu:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistente:</strong> {format_response(message["content"])}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Funzione principale dell'applicazione"""
    
    # Header principale
    st.markdown("""
    <div class="main-header">
        🏛️ RAG Prefettura
        <br><small>Assistente AI per Illeciti Amministrativi</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Inizializzazione
    initialize_session_state()
    
    # Sidebar per configurazioni e upload
    with st.sidebar:
        st.header("📁 Gestione Documenti")
        
        # Sezione upload
        st.subheader("Carica Nuovi Documenti")
        uploaded_files = st.file_uploader(
            "Trascina i file qui o clicca per selezionare",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png'],
            help="Formati supportati: PDF, DOCX, TXT, JPG, PNG"
        )
        
        if st.button("📤 Carica Documenti", disabled=not uploaded_files):
            handle_file_upload(uploaded_files)
        
        # Mostra documenti caricati
        if st.session_state.uploaded_documents:
            st.subheader("📋 Documenti Caricati")
            for doc in st.session_state.uploaded_documents:
                with st.expander(f"{doc['filename']} ({doc['file_type']})"):
                    st.write(f"**Dimensione:** {doc['size']} bytes")
                    st.write(f"**ID:** {doc['document_id']}")
        
        st.divider()
        
        # Statistiche
        st.subheader("📊 Statistiche")
        st.metric("Documenti caricati", len(st.session_state.uploaded_documents))
        st.metric("Messaggi nella chat", len(st.session_state.messages))
        
        # Pulsante per pulire la cronologia
        if st.button("🗑️ Pulisci Cronologia"):
            st.session_state.messages = []
            st.rerun()
    
    # Area principale - inizializzazione componenti
    if not initialize_components():
        st.stop()
    
    # Area chat principale
    st.header("💬 Assistente AI")
    
    # Container per la cronologia chat
    chat_container = st.container()
    
    with chat_container:
        display_chat_history()
    
    # Input per nuove domande
    st.divider()
    
    # Forma di input
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_area(
            "Fai una domanda:",
            placeholder="Ad esempio: 'Quali sono le sanzioni per il mancato pagamento delle tasse?' oppure 'Carica un documento e chiedi informazioni specifiche'",
            height=100
        )
        
        submitted = st.form_submit_button("🚀 Invia Domanda", use_container_width=True)
    
    # Gestisci l'invio della query
    if submitted and user_query:
        handle_query(user_query)
        st.rerun()
    
    # Footer con informazioni
    st.divider()
    with st.expander("ℹ️ Informazioni sull'Applicazione"):
        st.markdown("""
        **Come usare l'applicazione:**
        
        1. **Carica documenti** (opzionale): Usa la sidebar per caricare PDF, DOCX o immagini
        2. **Fai domande**: Scrivi la tua domanda nell'area di testo in basso
        3. **Ricevi risposte**: L'AI analizzerà i documenti caricati e fornirà risposte pertinenti basate sul modello fine-tuned
        
        **Caratteristiche:**
        - Il modello è stato fine-tuned sulla knowledge base degli illeciti amministrativi
        - I documenti caricati vengono sempre utilizzati per fornire risposte contestuali
        - Il sistema integra automaticamente le conoscenze del modello con i tuoi documenti
        
        **Tipi di domande che puoi fare:**
        - Domande generali sugli illeciti amministrativi
        - Richieste di chiarimenti su procedure specifiche
        - Analisi di documenti caricati
        - Confronti tra normative
        
        **Formati supportati:** PDF, DOCX, TXT, JPG, PNG
        """)

if __name__ == "__main__":
    main()