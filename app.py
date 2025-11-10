import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str((Path(__file__).parent / "rag_pipeline").resolve()))
sys.path.append(str((Path(__file__).parent / "app_skeleton").resolve()))

import streamlit as st
from typing import List, Dict, Any

from app_skeleton.src.utils import get_file_type, format_response
from rag_pipeline.config.settings import RAGConfig
from rag_pipeline.core.pipeline import OptimizedRAGPipeline
from rag_pipeline.core.chat_pipeline import ChatPipeline

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
    
    .main-header-chat {
        font-size: 2.5rem;
        font-weight: 700;
        color: #7b1fa2;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f3e5f5, #e1bee7);
        border-radius: 10px;
        border-left: 5px solid #7b1fa2;
    }
    
    .mode-toggle {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #dee2e6;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
        color: #000000 !important;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .user-message-chat {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }
    
    .assistant-message-chat {
        background-color: #fce4ec;
        border-left-color: #e91e63;
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

def handle_file_upload(uploaded_files):
    """Gestisce l'upload dei file"""
    if not uploaded_files:
        return
    
    successful_uploads = []
    failed_uploads = []
    
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        try:
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_type = get_file_type(uploaded_file.name)
            
            successful_uploads.append({
                'filename': uploaded_file.name,
                'file_type': file_type,
                'file_path': str(file_path),
                'size': uploaded_file.size
            })
                
        except Exception as e:
            failed_uploads.append({
                'filename': uploaded_file.name,
                'error': str(e)
            })
    
    # Aggiorna lo stato
    st.session_state.uploaded_files_info.extend(successful_uploads)
    
    # ✅ COSTRUISCI L'INDICE SUBITO
    if successful_uploads:
        try:
            with st.spinner("📚 Indicizzando documenti con RAG pipeline..."):
                file_paths = [doc['file_path'] for doc in st.session_state.uploaded_files_info]
                
                # Build index con TUTTI i file
                st.session_state.rag_pipeline.build_index(file_paths=file_paths)
                
                # Setup query engine
                st.session_state.rag_pipeline.setup_query_engine()
                
                # Flag che l'indice è pronto
                st.session_state.index_ready = True
                
            st.success(f"✅ {len(successful_uploads)} documenti indicizzati!")
            
        except Exception as e:
            st.error(f"❌ Errore indicizzazione: {e}")
            return
    
    # Mostra risultati
    if successful_uploads:
        st.markdown('<div class="status-box status-success">✅ Documenti caricati:</div>', unsafe_allow_html=True)
        for doc in successful_uploads:
            st.write(f"• {doc['filename']} ({doc['file_type']})")
    
    if failed_uploads:
        st.markdown('<div class="status-box status-error">❌ Errori:</div>', unsafe_allow_html=True)
        for doc in failed_uploads:
            st.write(f"• {doc['filename']}: {doc['error']}")

def handle_query(query: str):
    """Gestisce le query dell'utente in base alla modalità"""
    if not query.strip():
        return
    
    current_mode = st.session_state.app_mode
    
    # Aggiungi messaggio utente alla cronologia corretta
    messages_key = f"messages_{current_mode}"
    st.session_state[messages_key].append({
        "role": "user",
        "content": query,
        "timestamp": ""
    })
    
    try:
        if current_mode == "rag":
            # Modalità RAG - verifica documenti caricati
            if not st.session_state.uploaded_files_info:
                st.error("⚠️ Carica almeno un documento prima di fare domande!")
                return
            
            if not st.session_state.get('index_ready', False):
                st.error("⚠️ Indice non costruito. Ricarica i documenti.")
                return
            
            with st.spinner("🤔 Elaborando la tua domanda con RAG..."):
                # Query RAG
                result = st.session_state.rag_pipeline.query(
                    query, 
                    enhance_query=True
                )
                
                response = result.get('answer', 'Nessuna risposta generata')
                sources = result.get('sources', [])
                metadata = result.get('query_metadata', {})
                
                # Aggiungi risposta con metadata completi
                st.session_state[messages_key].append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                    "metadata": metadata,
                    "response_time": result.get('response_time', 0),
                    "timestamp": ""
                })
        
        else:
            # Modalità Chat
            with st.spinner("💭 Pensando..."):
                if st.session_state.chat_pipeline:
                    # Usa ChatPipeline
                    result = st.session_state.chat_pipeline.query(query)
                    response = result.get('answer', 'Nessuna risposta generata')
                else:
                    # Fallback
                    response = "⚠️ ChatPipeline non inizializzato correttamente."
                
                # Aggiungi risposta semplice (senza metadata)
                st.session_state[messages_key].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": ""
                })
    
    except Exception as e:
        st.error(f"❌ Errore: {str(e)}")
        st.session_state[messages_key].append({
            "role": "assistant",
            "content": f"Mi dispiace, si è verificato un errore: {str(e)}",
            "timestamp": ""
        })

def display_chat_history():
    """Mostra la cronologia della chat in base alla modalità"""
    current_mode = st.session_state.app_mode
    messages_key = f"messages_{current_mode}"
    messages = st.session_state.get(messages_key, [])
    
    for message in messages:
        if message["role"] == "user":
            user_class = "user-message-chat" if current_mode == "chat" else "user-message"
            st.markdown(f"""
            <div class="chat-message {user_class}">
                <strong>🧑 Tu:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            assistant_class = "assistant-message-chat" if current_mode == "chat" else "assistant-message"
            icon = "💬" if current_mode == "chat" else "🤖"
            st.markdown(f"""
            <div class="chat-message {assistant_class}">
                <strong>{icon} Assistente:</strong> {format_response(message["content"])}
            </div>
            """, unsafe_allow_html=True)
            
            # Mostra metadata SOLO in modalità RAG
            if current_mode == "rag" and "metadata" in message and message["metadata"]:
                metadata = message["metadata"]
                
                with st.expander("🔍 Query Analysis & Pipeline Details"):
                    # Query Analysis
                    if 'expansions' in metadata:
                        st.markdown("**🔍 Query Analysis:**")
                        exp = metadata['expansions']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'intent' in exp:
                                st.info(f"**Intent:** {exp['intent']}")
                            if 'keywords' in exp and exp['keywords']:
                                keywords = ', '.join(exp['keywords'][:5])
                                st.info(f"**Keywords:** {keywords}")
                        
                        with col2:
                            if 'semantic_variants' in exp:
                                st.info(f"**Semantic variants:** {len(exp['semantic_variants'])}")
                            if 'num_queries_generated' in metadata:
                                st.info(f"**Total queries:** {metadata['num_queries_generated']}")
                    
                    st.divider()
                    
                    # Retrieval Pipeline
                    st.markdown("**📊 Retrieval Pipeline:**")
                    
                    if 'retrieval' in metadata:
                        ret = metadata['retrieval']
                        nodes = ret.get('total_nodes_retrieved', 0)
                        st.success(f"**1. Multi-retrieval:** {nodes} nodes retrieved")
                    
                    if 'deduplication' in metadata:
                        dedup = metadata['deduplication']
                        before = dedup.get('nodes_before', 0)
                        after = dedup.get('nodes_after', 0)
                        removed = dedup.get('duplicates_removed', 0)
                        st.success(f"**2. Deduplication:** {before} → {after} nodes (-{removed} duplicates)")
                    
                    if 'reranking' in metadata:
                        rerank = metadata['reranking']
                        if rerank.get('applied'):
                            before = rerank.get('nodes_before', 0)
                            after = rerank.get('nodes_after', 0)
                            st.success(f"**3. Reranking:** {before} → {after} top nodes ✓")
                        else:
                            st.warning("**3. Reranking:** Not applied")
                    
                    if 'response_time' in message:
                        st.divider()
                        st.metric("⚡ Total time", f"{message['response_time']:.3f}s")
            
            # Mostra fonti SOLO in modalità RAG
            if current_mode == "rag" and "sources" in message and message["sources"]:
                with st.expander(f"📚 Top Sources ({len(message['sources'])} total)"):
                    for i, source in enumerate(message['sources'][:3], 1):
                        st.markdown(f"**[{i}] Score: {source['score']:.3f}**")
                        st.text(source['text'][:200] + "...")
                        
                        if source.get('metadata'):
                            st.caption(f"📄 {source['metadata'].get('file_name', 'N/A')}")
                        
                        if source.get('reranked'):
                            st.success("✓ Reranked")
                        
                        st.divider()

def initialize_session_state():
    """Inizializza lo stato della sessione"""
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'rag'
    
    if 'messages_rag' not in st.session_state:
        st.session_state.messages_rag = []
    if 'messages_chat' not in st.session_state:
        st.session_state.messages_chat = []
    
    if 'uploaded_files_info' not in st.session_state:
        st.session_state.uploaded_files_info = []
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'index_ready' not in st.session_state:
        st.session_state.index_ready = False
    
    if 'chat_pipeline' not in st.session_state:
        st.session_state.chat_pipeline = None

def reset_mode_data(mode):
    """Resetta i dati di una specifica modalità"""
    if mode == 'rag':
        upload_dir = Path("data/uploads")
        if upload_dir.exists():
            for file_info in st.session_state.uploaded_files_info:
                file_path = Path(file_info['file_path'])
                if file_path.exists():
                    file_path.unlink()
        
        st.session_state.uploaded_files_info = []
        st.session_state.messages_rag = []
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
    
    elif mode == 'chat':
        st.session_state.messages_chat = []
        st.session_state.chat_pipeline = None

def initialize_components():
    """Inizializza i componenti principali in base alla modalità"""
    try:
        if st.session_state.app_mode == 'rag':
            if st.session_state.rag_pipeline is None:
                with st.spinner("Inizializzazione RAG Pipeline..."):
                    config = RAGConfig(
                        llm_model="llama3.2:3b-instruct-q4_K_M",
                        embedding_model="nomic-ai/nomic-embed-text-v1.5",
                        chunk_sizes=[2048, 512],
                        temperature=0.3,
                        context_window=4096,
                        use_reranker=True,
                        use_automerging=True,
                        chunk_overlap=150
                    )
                    st.session_state.rag_pipeline = OptimizedRAGPipeline(config)
        else:
            if st.session_state.chat_pipeline is None:
                with st.spinner("Inizializzazione Chat Pipeline..."):
                    config = RAGConfig(
                        llm_model="llama3.2:3b-instruct-q4_K_M",
                        temperature=0.3,
                        context_window=4096
                    )
                    st.session_state.chat_pipeline = ChatPipeline(config)
        
        return True
    except Exception as e:
        st.error(f"Errore nell'inizializzazione: {str(e)}")
        return False

def main():
    """Funzione principale dell'applicazione"""
    initialize_session_state()
    
    # Mode Toggle Button
    st.markdown("""
    <div class="mode-toggle">
        <h3>🔄 Seleziona Modalità</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_mode = st.session_state.app_mode
        
        mode_col1, mode_col2 = st.columns(2)
        
        with mode_col1:
            if st.button("📚 RAG Mode", 
                        type="primary" if current_mode == "rag" else "secondary",
                        use_container_width=True):
                if current_mode != "rag":
                    reset_mode_data("chat")
                    st.session_state.app_mode = "rag"
                    st.rerun()
        
        with mode_col2:
            if st.button("💬 Chat Mode", 
                        type="primary" if current_mode == "chat" else "secondary",
                        use_container_width=True):
                if current_mode != "chat":
                    reset_mode_data("rag")
                    st.session_state.app_mode = "chat"
                    st.rerun()
    
    # Header principale
    if st.session_state.app_mode == "rag":
        st.markdown("""
        <div class="main-header">
            🏛️ RAG Prefettura - Modalità Documenti
            <br><small>Analizza i tuoi documenti con AI</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="main-header-chat">
            💬 RAG Prefettura - Modalità Chat
            <br><small>Assistente legale specializzato in diritto amministrativo</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar condizionale
    if st.session_state.app_mode == "rag":
        with st.sidebar:
            st.header("📁 Gestione Documenti")
            
            st.subheader("Carica Nuovi Documenti")
            uploaded_files = st.file_uploader(
                "Trascina i file qui o clicca per selezionare",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png'],
                help="Formati supportati: PDF, DOCX, TXT, JPG, PNG"
            )
            
            if st.button("📤 Carica Documenti", disabled=not uploaded_files):
                handle_file_upload(uploaded_files)
                st.rerun()
            
            if st.session_state.uploaded_files_info:
                st.subheader("📋 Documenti Caricati")
                for doc in st.session_state.uploaded_files_info:
                    with st.expander(f"{doc['filename']} ({doc['file_type']})"):
                        st.write(f"**Dimensione:** {doc['size']} bytes")
                        st.write(f"**Path:** {doc['file_path']}")
            
            st.divider()
            
            st.subheader("📊 Statistiche RAG")
            
            if st.session_state.rag_pipeline and st.session_state.get('index_ready', False):
                try:
                    stats = st.session_state.rag_pipeline.get_statistics()
                    
                    perf = stats.get('performance', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Query totali", perf.get('total_queries', 0))
                    with col2:
                        st.metric("Tempo medio", f"{perf.get('avg_response_time', 0):.2f}s")
                    
                    data = stats.get('data', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Documenti", data.get('total_documents', 0))
                    with col2:
                        st.metric("Nodi", data.get('total_nodes', 0))
                    
                    if 'retrieval_stats' in stats:
                        rs = stats['retrieval_stats']
                        with st.expander("📈 Retrieval Statistics"):
                            st.metric("Avg nodes retrieved", f"{rs.get('avg_nodes_retrieved', 0):.1f}")
                            st.metric("Avg dedup reduction", f"{rs.get('avg_dedup_reduction', 0)*100:.1f}%")
                    
                    with st.expander("⚙️ Configurazione"):
                        config = stats.get('configuration', {})
                        st.json({
                            "LLM": config.get('llm_model', 'N/A'),
                            "Embedding": config.get('embedding_model', 'N/A'),
                            "Index": config.get('index_type', 'N/A'),
                            "Reranker": "✅" if config.get('reranker_enabled') else "❌"
                        })
                
                except Exception as e:
                    st.error(f"Errore nel recupero statistiche: {e}")
            else:
                st.info("Carica documenti per vedere le statistiche")
            
            st.divider()
            
            if st.button("🔄 Reset RAG"):
                reset_mode_data('rag')
                st.success("✅ RAG resettato!")
                st.rerun()
    
    else:
        with st.sidebar:
            st.header("💬 Modalità Chat")
            st.info("Chat specializzata in diritto amministrativo italiano e illeciti amministrativi.")
            
            st.subheader("📊 Statistiche Chat")
            messages_count = len(st.session_state.messages_chat)
            st.metric("Messaggi totali", messages_count)
            
            if st.session_state.chat_pipeline:
                try:
                    stats = st.session_state.chat_pipeline.get_statistics()
                    perf = stats.get('performance', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Query", perf.get('total_queries', 0))
                    with col2:
                        st.metric("Success", perf.get('success_rate', 'N/A'))
                    
                    st.metric("Tempo medio", perf.get('avg_response_time', 'N/A'))
                    
                    with st.expander("ℹ️ Specializzazione"):
                        sys_info = stats.get('system_prompt', {})
                        st.write(f"**Dominio:** {sys_info.get('domain', 'N/A')}")
                        st.write(f"**Lingua:** {sys_info.get('language', 'N/A')}")
                    
                except Exception as e:
                    st.warning(f"Statistiche non disponibili: {e}")
            else:
                st.warning("⚠️ ChatPipeline non inizializzato")
            
            st.divider()
            
            if st.button("🔄 Reset Chat"):
                reset_mode_data('chat')
                st.success("✅ Chat resettata!")
                st.rerun()
    
    # Area principale
    if not initialize_components():
        st.stop()
    
    chat_title = "💬 Chat Libera" if st.session_state.app_mode == "chat" else "🤖 Assistente Documenti"
    st.header(chat_title)
    
    chat_container = st.container()
    
    with chat_container:
        display_chat_history()
    
    st.divider()
    
    # Input form
    with st.form("query_form", clear_on_submit=True):
        current_mode = st.session_state.app_mode
        
        if current_mode == "rag":
            if not st.session_state.uploaded_files_info:
                st.warning("⚠️ Carica almeno un documento prima di fare domande")
            
            user_query = st.text_area(
                "Fai una domanda sui tuoi documenti:",
                placeholder="Carica prima dei documenti, poi fai domande come: 'Riassumi il contenuto' o 'Quali sono i punti chiave?'",
                height=100,
                disabled=not st.session_state.uploaded_files_info
            )
            
            submit_disabled = not st.session_state.uploaded_files_info
            submit_text = "🚀 Analizza Documenti"
        
        else:
            user_query = st.text_area(
                "Fai una domanda su diritto amministrativo:",
                placeholder="Es: 'Cosa sono gli illeciti amministrativi?' o 'Come funziona un ricorso amministrativo?'",
                height=100
            )
            
            submit_disabled = False
            submit_text = "💬 Invia"
        
        submitted = st.form_submit_button(
            submit_text,
            use_container_width=True,
            disabled=submit_disabled
        )
    
    if submitted and user_query:
        handle_query(user_query)
        st.rerun()
    
    # Footer
    st.divider()
    current_mode = st.session_state.app_mode
    
    if current_mode == "rag":
        info_title = "ℹ️ Informazioni - Modalità RAG"
        info_content = """
        **Come usare la modalità RAG:**
        
        1. **Carica documenti**: Usa la sidebar per caricare PDF, DOCX o immagini
        2. **Fai domande**: Scrivi domande specifiche sui tuoi file
        3. **Analisi dettagliata**: Ricevi risposte con fonti e metadata completi
        
        **Formati supportati:** PDF, DOCX, TXT, JPG, PNG
        """
    else:
        info_title = "ℹ️ Informazioni - Modalità Chat"
        info_content = """
        **Come usare la modalità Chat:**
        
        - Assistente specializzato in **diritto amministrativo italiano**
        - Focus su **illeciti amministrativi** e procedure prefettizie
        - Nessun documento richiesto - conversazione diretta
        - Risposte basate su normativa italiana
        
        **Nota:** Questo è uno strumento informativo, non sostituisce la consulenza legale professionale.
        """
    
    with st.expander(info_title):
        st.markdown(info_content)

if __name__ == "__main__":
    main()