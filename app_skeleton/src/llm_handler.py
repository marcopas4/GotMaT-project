"""
LLM Handler Module
==================

Questo modulo gestisce il modello di linguaggio fine-tuned per rispondere 
alle domande sugli illeciti amministrativi.

I tuoi colleghi dovranno implementare:
- Caricamento del modello fine-tuned
- Creazione prompt template specifici per il dominio
- Gestione della generazione delle risposte
- Post-processing delle risposte

PLACEHOLDER - DA IMPLEMENTARE:
- Modello fine-tuned locale (transformers, llama.cpp, etc.)
- Template prompt per illeciti amministrativi
- Controllo qualità risposte
- Gestione contesto e memoria conversazione
"""

import os
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# TODO: Importare librerie necessarie
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# oppure
# from llama_cpp import Llama
# oppure altri framework per modelli locali

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Gestisce il modello di linguaggio per la generazione delle risposte.
    
    Questa classe si occupa di:
    - Caricare il modello fine-tuned
    - Gestire i prompt template
    - Generare risposte contestuali
    - Applicare filtri e controlli di qualità
    """
    
    def __init__(self, 
                 model_path: str = "models/fine_tuned_model/",
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 device: str = "cpu"):
        """
        Inizializza l'LLM handler.
        
        Args:
            model_path: Percorso al modello fine-tuned
            max_tokens: Numero massimo di token per risposta
            temperature: Parametro di temperatura per la generazione
            device: Dispositivo da usare (cpu/cuda)
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device
        
        # Componenti del modello
        self.model = None
        self.tokenizer = None
        
        # Template prompt
        self.prompt_templates = self._load_prompt_templates()
        
        # Storia conversazione (per contesto)
        self.conversation_history = []
        
        # Inizializzazione
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Carica il modello fine-tuned e il tokenizer.
        
        PLACEHOLDER - DA IMPLEMENTARE:
        - Caricamento modello da Hugging Face transformers
        - Configurazione per CPU/GPU
        - Ottimizzazioni per inferenza
        - Gestione modelli quantizzati se necessario
        """
        logger.info(f"Inizializzazione modello LLM: {self.model_path}")
        
        # TODO: Implementare caricamento modello reale
        # Esempio con transformers:
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # import torch
        # 
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path,
        #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        #     device_map="auto" if self.device == "cuda" else None
        # )
        # 
        # # Configurazione per generazione
        # self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Esempio con llama.cpp (per modelli GGUF):
        # from llama_cpp import Llama
        # self.model = Llama(
        #     model_path=os.path.join(self.model_path, "model.gguf"),
        #     n_ctx=2048,  # Contesto
        #     n_threads=4,  # Thread CPU
        #     verbose=False
        # )
        
        # PLACEHOLDER - Simulazione
        logger.info("✅ Modello LLM inizializzato (PLACEHOLDER)")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Carica i template dei prompt specifici per illeciti amministrativi.
        
        PLACEHOLDER - DA PERSONALIZZARE:
        - Template specifici per il dominio
        - Prompt engineering ottimizzato
        - Varianti per diversi tipi di domande
        """
        templates = {
            'base_prompt': """Sei un assistente AI specializzato in illeciti amministrativi e normative italiane.
Il tuo compito è fornire risposte accurate e utili basate sui documenti forniti e sulla tua conoscenza del diritto amministrativo.

CONTESTO:
{context}

DOMANDA: {query}

ISTRUZIONI:
- Fornisci una risposta precisa e professionale
- Cita sempre le fonti quando possibile
- Se non hai informazioni sufficienti, dillo chiaramente
- Usa un linguaggio tecnico ma comprensibile
- Struttura la risposta in modo chiaro e logico

RISPOSTA:""",
            
            'document_analysis': """Analizza il seguente documento nel contesto della domanda posta.

DOCUMENTO:
{context}

DOMANDA: {query}

Fornisci un'analisi dettagliata evidenziando:
1. Informazioni rilevanti trovate
2. Articoli o sezioni pertinenti
3. Possibili interpretazioni
4. Raccomandazioni pratiche

ANALISI:""",
            
            'comparison_prompt': """Confronta le seguenti informazioni da diverse fonti per rispondere alla domanda.

FONTI MULTIPLE:
{context}

DOMANDA: {query}

Fornisci un confronto strutturato evidenziando:
- Punti comuni tra le fonti
- Eventuali discrepanze
- Conclusioni basate sull'analisi comparata

CONFRONTO:""",
            
            'procedural_prompt': """Fornisci una guida procedurale basata sui documenti forniti.

RIFERIMENTI NORMATIVI:
{context}

RICHIESTA: {query}

Struttura la risposta come guida passo-passo:
1. Premesse normative
2. Procedura da seguire
3. Documentazione necessaria
4. Tempistiche e scadenze
5. Possibili criticità

GUIDA PROCEDURALE:"""
        }
        
        return templates
    
    def generate_response(self, 
                         query: str, 
                         context: List[Dict[str, Any]],
                         use_uploaded_docs: bool = False,
                         prompt_type: str = "base_prompt") -> str:
        """
        Genera una risposta alla query dell'utente.
        
        Args:
            query: Domanda dell'utente
            context: Lista di contesti rilevanti dal vector store
            use_uploaded_docs: Se True, indica che si stanno usando documenti caricati
            prompt_type: Tipo di prompt template da usare
            
        Returns:
            Risposta generata dal modello
            
        PLACEHOLDER - DA IMPLEMENTARE:
        - Costruzione prompt con contesto
        - Chiamata al modello per generazione
        - Post-processing della risposta
        - Controlli di qualità e filtri
        """
        logger.info(f"Generando risposta per query: '{query[:50]}...'")
        
        try:
            # Prepara il contesto
            context_text = self._prepare_context(context, use_uploaded_docs)
            
            # Costruisci il prompt
            prompt = self._build_prompt(query, context_text, prompt_type)
            
            # Genera la risposta
            response = self._generate_with_model(prompt)
            
            # Post-processing
            processed_response = self._post_process_response(response, query, context)
            
            # Aggiorna cronologia conversazione
            self._update_conversation_history(query, processed_response, context)
            
            logger.info("✅ Risposta generata con successo")
            return processed_response
            
        except Exception as e:
            logger.error(f"Errore nella generazione risposta: {str(e)}")
            return self._get_error_response(str(e))
    
    def _prepare_context(self, context: List[Dict[str, Any]], use_uploaded_docs: bool) -> str:
        """
        Prepara il contesto dai risultati del vector store.
        
        Args:
            context: Lista di risultati dal vector store
            use_uploaded_docs: Se si stanno usando documenti caricati
            
        Returns:
            Testo del contesto formattato
        """
        if not context:
            return "Nessun contesto specifico disponibile."
        
        context_parts = []
        
        for i, ctx in enumerate(context[:5]):  # Limita a 5 risultati più rilevanti
            source_info = f"Fonte: {ctx.get('source', 'Sconosciuta')}"
            score_info = f"(Rilevanza: {ctx.get('score', 0):.2f})"
            content = ctx.get('content', '')
            
            context_parts.append(f"""
--- DOCUMENTO {i+1} ---
{source_info} {score_info}

{content}
""")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, prompt_type: str) -> str:
        """
        Costruisce il prompt finale combinando template, query e contesto.
        
        Args:
            query: Domanda dell'utente
            context: Contesto formattato
            prompt_type: Tipo di template da usare
            
        Returns:
            Prompt completo per il modello
        """
        template = self.prompt_templates.get(prompt_type, self.prompt_templates['base_prompt'])
        
        # Aggiungi contesto conversazionale se presente
        conversation_context = self._get_conversation_context()
        if conversation_context:
            context = f"{conversation_context}\n\n{context}"
        
        prompt = template.format(
            query=query,
            context=context,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        
        return prompt
    
    def _generate_with_model(self, prompt: str) -> str:
        """
        Chiama il modello per generare la risposta.
        
        Args:
            prompt: Prompt completo per il modello
            
        Returns:
            Risposta grezza dal modello
            
        PLACEHOLDER - DA IMPLEMENTARE:
        - Tokenizzazione input
        - Chiamata modello con parametri ottimizzati
        - Decodifica output
        - Gestione errori di generazione
        """
        # TODO: Implementare chiamata al modello reale
        
        # Esempio con transformers:
        # inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        # if self.device == "cuda":
        #     inputs = inputs.to("cuda")
        # 
        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         inputs,
        #         max_new_tokens=self.max_tokens,
        #         temperature=self.temperature,
        #         do_sample=True,
        #         top_p=0.9,
        #         repetition_penalty=1.1,
        #         eos_token_id=self.tokenizer.eos_token_id,
        #         pad_token_id=self.tokenizer.pad_token_id
        #     )
        # 
        # response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Esempio con llama.cpp:
        # response = self.model(
        #     prompt,
        #     max_tokens=self.max_tokens,
        #     temperature=self.temperature,
        #     top_p=0.9,
        #     repeat_penalty=1.1,
        #     stop=["</s>", "[INST]", "[/INST]"]
        # )
        # return response['choices'][0]['text']
        
        # PLACEHOLDER - Simulazione risposta
        placeholder_response = f"""Basandomi sui documenti forniti, posso rispondere alla tua domanda sugli illeciti amministrativi.

Per la tua domanda specifica, è importante considerare:

1. **Aspetti normativi**: La normativa italiana prevede specifiche procedure per questo tipo di situazioni.

2. **Procedure amministrative**: Esistono passaggi ben definiti che devono essere seguiti secondo la legge.

3. **Documentazione richiesta**: È necessario presentare specifica documentazione per avviare la procedura.

4. **Tempistiche**: Ci sono termini specifici da rispettare per non incorrere in decadenze.

**Raccomandazione**: Ti consiglio di consultare sempre un professionista qualificato per il tuo caso specifico.

*Nota: Questa è una risposta simulata. I tuoi colleghi implementeranno il modello fine-tuned reale.*"""
        
        return placeholder_response
    
    def _post_process_response(self, response: str, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Applica post-processing alla risposta del modello.
        
        Args:
            response: Risposta grezza dal modello
            query: Domanda originale
            context: Contesto utilizzato
            
        Returns:
            Risposta post-processata
            
        PLACEHOLDER - DA IMPLEMENTARE:
        - Rimozione ripetizioni
        - Correzione errori comuni
        - Formattazione migliorata
        - Validazione contenuto
        - Aggiunta riferimenti normativi
        """
        # TODO: Implementare post-processing avanzato
        
        # Pulizia base
        processed = response.strip()
        
        # Rimuovi eventuali marker di fine generazione
        end_markers = ["</s>", "[END]", "---END---"]
        for marker in end_markers:
            processed = processed.replace(marker, "")
        
        # Aggiungi disclaimer se necessario
        if not self._contains_disclaimer(processed):
            processed += "\n\n*Nota: Questa risposta è basata sui documenti forniti. Per casi specifici, consultare sempre un professionista qualificato.*"
        
        # TODO: Aggiungere controlli più sofisticati
        # - Verifica coerenza con il contesto
        # - Controllo presenza informazioni errate
        # - Formattazione automatica (elenchi, sezioni)
        # - Aggiunta links/riferimenti quando possibile
        
        return processed
    
    def _contains_disclaimer(self, text: str) -> bool:
        """Controlla se il testo contiene già un disclaimer."""
        disclaimer_keywords = ["consultare", "professionista", "caso specifico", "disclaimer"]
        return any(keyword in text.lower() for keyword in disclaimer_keywords)
    
    def _get_conversation_context(self) -> str:
        """
        Recupera il contesto della conversazione precedente.
        
        Returns:
            Contesto formattato delle interazioni precedenti
        """
        if not self.conversation_history:
            return ""
        
        # Prendi solo le ultime 3 interazioni per non saturare il contesto
        recent_history = self.conversation_history[-3:]
        
        context_parts = []
        for interaction in recent_history:
            context_parts.append(f"Q: {interaction['query']}")
            context_parts.append(f"A: {interaction['response'][:200]}...")
        
        return "CONVERSAZIONE PRECEDENTE:\n" + "\n".join(context_parts)
    
    def _update_conversation_history(self, query: str, response: str, context: List[Dict[str, Any]]):
        """
        Aggiorna la cronologia della conversazione.
        
        Args:
            query: Domanda dell'utente
            response: Risposta generata
            context: Contesto utilizzato
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'context_sources': [ctx.get('source', 'Unknown') for ctx in context],
            'context_scores': [ctx.get('score', 0) for ctx in context]
        }
        
        self.conversation_history.append(interaction)
        
        # Mantieni solo le ultime 10 interazioni per gestire la memoria
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _get_error_response(self, error_message: str) -> str:
        """
        Genera una risposta di errore user-friendly.
        
        Args:
            error_message: Messaggio di errore tecnico
            
        Returns:
            Risposta di errore formattata per l'utente
        """
        return f"""Mi dispiace, si è verificato un problema durante l'elaborazione della tua richiesta.

**Errore**: {error_message}

**Cosa puoi fare**:
1. Riprova con una domanda formulata diversamente
2. Assicurati che i documenti caricati siano leggibili
3. Controlla che la connessione sia stabile

Se il problema persiste, contatta l'assistenza tecnica.

*Nota: Questo errore è stato registrato per il debug.*"""
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sul modello caricato.
        
        Returns:
            Dizionario con informazioni sul modello
        """
        return {
            'model_path': self.model_path,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'device': self.device,
            'conversation_history_length': len(self.conversation_history),
            'available_prompts': list(self.prompt_templates.keys()),
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None
        }
    
    def clear_conversation_history(self):
        """
        Pulisce la cronologia della conversazione.
        """
        self.conversation_history = []
        logger.info("Cronologia conversazione pulita")
    
    def set_temperature(self, temperature: float):
        """
        Modifica la temperatura di generazione.
        
        Args:
            temperature: Nuovo valore di temperatura (0.0-1.0)
        """
        if 0.0 <= temperature <= 1.0:
            self.temperature = temperature
            logger.info(f"Temperatura aggiornata a: {temperature}")
        else:
            raise ValueError("La temperatura deve essere tra 0.0 e 1.0")
    
    def add_custom_prompt_template(self, name: str, template: str):
        """
        Aggiunge un template personalizzato per i prompt.
        
        Args:
            name: Nome del template
            template: Template string con placeholder {query} e {context}
        """
        if '{query}' not in template or '{context}' not in template:
            raise ValueError("Il template deve contenere i placeholder {query} e {context}")
        
        self.prompt_templates[name] = template
        logger.info(f"Template '{name}' aggiunto")
    
    def validate_response_quality(self, response: str, query: str) -> Dict[str, Any]:
        """
        Valuta la qualità della risposta generata.
        
        Args:
            response: Risposta da validare
            query: Domanda originale
            
        Returns:
            Dizionario con metriche di qualità
            
        PLACEHOLDER - DA IMPLEMENTARE:
        - Controllo lunghezza appropriata
        - Rilevamento allucinazioni
        - Verifica coerenza con query
        - Score di confidenza
        """
        # TODO: Implementare controlli di qualità avanzati
        
        quality_metrics = {
            'length_appropriate': 50 <= len(response) <= 2000,
            'contains_answer': query.lower() in response.lower() or any(
                word in response.lower() for word in query.lower().split()
            ),
            'has_structure': '\n' in response or '.' in response,
            'professional_tone': not any(word in response.lower() for word in ['non so', 'boh', 'mah']),
            'has_disclaimer': self._contains_disclaimer(response),
            'word_count': len(response.split()),
            'sentence_count': response.count('.') + response.count('!') + response.count('?')
        }
        
        # Score complessivo (0-1)
        passed_checks = sum(1 for check in quality_metrics.values() if isinstance(check, bool) and check)
        total_checks = sum(1 for check in quality_metrics.values() if isinstance(check, bool))
        quality_metrics['overall_score'] = passed_checks / total_checks if total_checks > 0 else 0
        
        return quality_metrics
    
    def generate_batch_responses(self, queries: List[str], contexts: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Genera risposte per multiple query in batch.
        
        Args:
            queries: Lista di domande
            contexts: Lista di contesti per ogni query
            
        Returns:
            Lista di risposte generate
            
        PLACEHOLDER - DA IMPLEMENTARE per efficienza
        """
        responses = []
        
        for i, (query, context) in enumerate(zip(queries, contexts)):
            logger.info(f"Processando query {i+1}/{len(queries)}")
            response = self.generate_response(query, context)
            responses.append(response)
        
        return responses
    
    def export_conversation_history(self, filepath: str):
        """
        Esporta la cronologia della conversazione in un file JSON.
        
        Args:
            filepath: Percorso dove salvare il file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Cronologia esportata in: {filepath}")
        except Exception as e:
            logger.error(f"Errore nell'esportazione cronologia: {str(e)}")
            raise
    
    def load_conversation_history(self, filepath: str):
        """
        Carica la cronologia della conversazione da un file JSON.
        
        Args:
            filepath: Percorso del file da caricare
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            logger.info(f"Cronologia caricata da: {filepath}")
        except Exception as e:
            logger.error(f"Errore nel caricamento cronologia: {str(e)}")
            raise
    
    def __del__(self):
        """
        Cleanup quando l'oggetto viene distrutto.
        """
        # TODO: Cleanup risorse del modello se necessario
        # if hasattr(self, 'model') and self.model is not None:
        #     del self.model
        # if hasattr(self, 'tokenizer') and self.tokenizer is not None:
        #     del self.tokenizer
        pass