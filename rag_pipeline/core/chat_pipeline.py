"""
ChatPipeline - Sistema di chat conversazionale per assistenza legale

Pipeline semplificata per chat diretta senza RAG, ottimizzata per 
rispondere a domande su illeciti amministrativi e diritto amministrativo.
"""

from typing import Dict, Any
import time
import logging
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from config.settings import RAGConfig

logger = logging.getLogger(__name__)


class ChatPipeline:
    """
    Pipeline di chat conversazionale per assistenza legale.
    
    Utilizza un LLM con system prompt specifico per diritto amministrativo
    e illeciti amministrativi, senza retrieval da documenti.
    """
    
    # System prompt fisso per guidare il modello
    SYSTEM_PROMPT = """Sei un assistente AI specializzato in diritto amministrativo italiano e illeciti amministrativi della Prefettura.

Il tuo ruolo è fornire supporto e informazioni su:
- Illeciti amministrativi e relative sanzioni
- Procedimenti amministrativi presso la Prefettura
- Normativa italiana in materia di diritto amministrativo
- Interpretazione di leggi e regolamenti amministrativi
- Procedure per ricorsi e opposizioni amministrative

ISTRUZIONI OPERATIVE:
1. Rispondi sempre in italiano corretto e formale
2. Fornisci informazioni precise e basate sulla normativa italiana vigente
3. Cita gli articoli di legge pertinenti quando possibile
4. Se non sei sicuro di una risposta, ammetti i limiti della tua conoscenza
5. Evita di dare consulenza legale vincolante - suggerisci sempre di consultare un professionista
6. Struttura le risposte in modo chiaro: premessa, spiegazione, conclusione
7. Usa esempi pratici quando utile per chiarire concetti complessi

FORMATO RISPOSTE:
- Sii conciso ma esaustivo
- Evidenzia i punti chiave
- Usa elenchi puntati per chiarezza quando appropriato
- Mantieni un tono professionale ma accessibile

Ricorda: sei uno strumento di supporto informativo, non sostituisci la consulenza di un avvocato o professionista legale."""
    
    def __init__(self, config: RAGConfig, llm=None):
        """
        Inizializza ChatPipeline con configurazione.
        
        Args:
            config: Configurazione RAG (riutilizzata per LLM settings)
            llm: LLM preconfigurato (opzionale)
        """
        self.config = config or RAGConfig()
        # ✅ USA LLM PASSATO O CREANE UNO NUOVO
        self.llm = llm if llm is not None else self._initialize_llm()
        
        # Configura settings globali
        self._configure_global_settings()
        
        # Statistiche
        self.stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "total_errors": 0
        }
        
        logger.info(f"ChatPipeline initialized with model: {self.config.llm_model}")
        logger.info("Specializzazione: Diritto amministrativo e illeciti amministrativi")
    
    def _initialize_llm(self):
        """Configura LLM Ollama con parametri ottimizzati per chat"""
        try:
            llm = Ollama(
                model=self.config.llm_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
                context_window=self.config.context_window,
                request_timeout=120.0,
                additional_kwargs={
                    "num_thread": self.config.num_threads,
                    "num_gpu": 1 if self.config.use_gpu else 0,
                    "repeat_penalty": 1.1,
                    "top_k": 40,
                    "top_p": 0.9
                }
                # ❌ RIMUOVI system_prompt DA QUI - verrà aggiunto nei messaggi
            )
            logger.info(f"LLM configured: {self.config.llm_model}")
            return llm
        except Exception as e:
            logger.error(f"Errore nella configurazione LLM: {e}")
            raise
    
    def _configure_global_settings(self):
        """Configura settings globali LlamaIndex"""
        Settings.llm = self.llm
        Settings.chunk_size = self.config.chunk_sizes[0]
        Settings.chunk_overlap = self.config.chunk_overlap
        Settings.num_output = 512
    
    def query(
        self, 
        question: str,
        conversation_history: list = None
    ) -> Dict[str, Any]:
        """
        Esegue una query conversazionale con il modello.
        
        Args:
            question: Domanda dell'utente
            conversation_history: Lista opzionale di messaggi precedenti 
                                 [{"role": "user"/"assistant", "content": "..."}]
        
        Returns:
            Dizionario con risposta e metadata:
            {
                "question": str,
                "answer": str,
                "response_time": float,
                "model": str,
                "error": str (opzionale)
            }
        """
        # Validazione input
        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Per favore, inserisci una domanda valida.",
                "response_time": 0.0,
                "model": self.config.llm_model,
                "error": "Empty question"
            }
        
        start_time = time.time()
        
        try:
            # ✅ COSTRUISCI MESSAGGI PER CHAT API
            messages = self._build_chat_messages(question, conversation_history)
            
            logger.debug(f"Query: '{question[:100]}...'")
            
            # ✅ USA CHAT INVECE DI COMPLETE
            from llama_index.core.llms import ChatMessage
            
            chat_messages = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
            
            response = self.llm.chat(chat_messages)
            
            # Estrai testo risposta
            answer = str(response.message.content) if hasattr(response, 'message') else str(response)
            
            response_time = time.time() - start_time
            
            # Aggiorna statistiche
            self._update_stats(response_time, success=True)
            
            result = {
                "question": question,
                "answer": answer,
                "response_time": response_time,
                "model": self.config.llm_model,
                "temperature": self.config.temperature,
                "mode": "chat"
            }
            
            logger.info(f"Query completata in {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            error_msg = self._handle_error(e)
            
            # Aggiorna statistiche errori
            self._update_stats(error_time, success=False)
            
            logger.error(f"Errore durante query: {e}")
            
            return {
                "question": question,
                "answer": error_msg,
                "response_time": error_time,
                "model": self.config.llm_model,
                "error": str(e)
            }
    
    def _build_chat_messages(
        self, 
        question: str, 
        conversation_history: list = None
    ) -> list:
        """
        Costruisce la lista di messaggi per la chat API.
        
        Args:
            question: Domanda corrente
            conversation_history: Cronologia messaggi precedenti
        
        Returns:
            Lista di dizionari con formato {"role": str, "content": str}
        """
        messages = []
        
        # ✅ AGGIUNGI SYSTEM PROMPT COME PRIMO MESSAGGIO
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        
        # ✅ AGGIUNGI CRONOLOGIA (LIMITATA)
        if conversation_history:
            max_history = 10  # Limita a 10 scambi precedenti
            recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
            
            for msg in recent_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Normalizza ruolo per API
                if role in ["user", "assistant"]:
                    messages.append({
                        "role": role,
                        "content": content
                    })
        
        # ✅ AGGIUNGI DOMANDA CORRENTE
        messages.append({
            "role": "user",
            "content": question
        })
        
        return messages
    
    def _build_prompt(
        self, 
        question: str, 
        conversation_history: list = None
    ) -> str:
        """
        [DEPRECATED] Metodo legacy - ora usa _build_chat_messages
        Mantenuto per compatibilità.
        """
        # Non più usato con chat API
        return question

    def _handle_error(self, error: Exception) -> str:
        """
        Gestisce errori e ritorna messaggi user-friendly in italiano.
        
        Args:
            error: Eccezione catturata
        
        Returns:
            Messaggio di errore user-friendly
        """
        error_str = str(error).lower()
        
        # Timeout Ollama
        if "timeout" in error_str or "timed out" in error_str:
            return ("Mi dispiace, il sistema sta impiegando troppo tempo a rispondere. "
                   "Per favore riprova tra qualche istante.")
        
        # Connessione a Ollama fallita
        if "connection" in error_str or "refused" in error_str:
            return ("Non riesco a connettermi al servizio di AI. "
                   "Verifica che Ollama sia in esecuzione e riprova.")
        
        # Modello non trovato
        if "model" in error_str and ("not found" in error_str or "404" in error_str):
            return (f"Il modello AI '{self.config.llm_model}' non è disponibile. "
                   "Verifica che il modello sia installato in Ollama.")
        
        # Context window overflow
        if "context" in error_str or "too long" in error_str:
            return ("La conversazione è diventata troppo lunga. "
                   "Per favore, inizia una nuova chat o riduci il messaggio.")
        
        # Errore generico
        return ("Si è verificato un errore imprevisto. "
               "Per favore riprova o contatta l'assistenza se il problema persiste.")
    
    def _update_stats(self, response_time: float, success: bool = True):
        """
        Aggiorna statistiche interne.
        
        Args:
            response_time: Tempo di risposta in secondi
            success: Se la query è andata a buon fine
        """
        self.stats["total_queries"] += 1
        
        if not success:
            self.stats["total_errors"] += 1
        
        # Media mobile per tempo di risposta
        prev_avg = self.stats["avg_response_time"]
        n = self.stats["total_queries"]
        self.stats["avg_response_time"] = (prev_avg * (n - 1) + response_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Ottieni statistiche complete della pipeline.
        
        Returns:
            Dizionario con statistiche
        """
        success_rate = (
            (self.stats["total_queries"] - self.stats["total_errors"]) / 
            self.stats["total_queries"] * 100
            if self.stats["total_queries"] > 0 else 0
        )
        
        return {
            "configuration": {
                "llm_model": self.config.llm_model,
                "temperature": self.config.temperature,
                "context_window": self.config.context_window,
                "mode": "chat",
                "specialization": "Diritto amministrativo italiano"
            },
            "performance": {
                "total_queries": self.stats["total_queries"],
                "total_errors": self.stats["total_errors"],
                "success_rate": f"{success_rate:.1f}%",
                "avg_response_time": f"{self.stats['avg_response_time']:.3f}s"
            },
            "system_prompt": {
                "enabled": True,
                "domain": "Illeciti amministrativi e diritto amministrativo",
                "language": "Italiano"
            }
        }
    
    def reset_statistics(self):
        """Resetta le statistiche"""
        self.stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "total_errors": 0
        }
        logger.info("Statistiche resettate")
