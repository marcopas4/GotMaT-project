from typing import List, Dict, Any, Set
from pathlib import Path
import time
import json
import numpy as np
import logging
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
    Settings
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from config.settings import RAGConfig, ResponseMode
from core.embedding_manager import EmbeddingManager
from core.indexing import FAISSIndexManager
from core.document_processor import DocumentProcessor
from core.retrieval import RetrievalManager
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from core.query_expansion import QueryProcessor, QueryConfig
from llama_index.core.schema import QueryBundle

logger = logging.getLogger(__name__)


class OptimizedRAGPipeline:
    """Pipeline RAG principale che coordina tutti i componenti"""
    
    def __init__(self, config: RAGConfig = None, llm=None):
        """
        Inizializza pipeline con configurazione
        
        Args:
            config: Configurazione RAG (usa default se None)
            llm: LLM preconfigurato (opzionale, condiviso)
        """
        self.config = config or RAGConfig()
        
        # Inizializza componenti core
        self.embedding_manager = EmbeddingManager(self.config.embedding_model)
        
        # FLAT index - sempre deterministico
        self.faiss_manager = FAISSIndexManager(
            dimension=self.embedding_manager.config.dimension,
            index_type="Flat"
        )
        
        self.document_processor = DocumentProcessor(
            self.config.chunk_sizes,
            self.config.chunk_overlap
        )
        self.retrieval_manager = RetrievalManager(self.config)
        
        # Query processor per enhancement
        self.query_processor = None
        
        # ✅ USA LLM CONDIVISO O INIZIALIZZANE UNO NUOVO
        if llm is not None:
            self.llm = llm
            logger.info("Using shared LLM instance")
        else:
            # Inizializza LLM
            self._setup_llm()
        
        # Configura settings globali
        self._configure_global_settings()
        
        # Componenti indice
        self.storage_context = None
        self.index = None
        self.query_engine = None
        
        # Statistiche
        self.stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "index_built": False,
            "reranking_enabled": self.config.use_reranker,
            "total_nodes_retrieved": 0,
            "total_nodes_after_dedup": 0,
            "total_nodes_after_rerank": 0
        }
        
        # Crea directory necessarie
        self._create_directories()
        
        logger.info("RAG Pipeline initialized with FLAT index (deterministic)")
        if self.config.use_reranker:
            logger.info("Jina reranker enabled")
    
    def _setup_llm(self):
        """Configura LLM Ollama - SENZA system prompt per RAG"""
        self.llm = Ollama(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.temperature,
            context_window=self.config.context_window,
            request_timeout=240.0,
            additional_kwargs={
                "num_thread": self.config.num_threads,
                "num_gpu": 1 if self.config.use_gpu else 0,
                "repeat_penalty": 1.1,
                "top_k": 40,
                "top_p": 0.9
            }
            # ✅ NO system_prompt - il RAG usa il contesto recuperato
        )
        logger.info(f"LLM configured for RAG: {self.config.llm_model}")
    
    def _configure_global_settings(self):
        """Configura settings globali LlamaIndex"""
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_manager.model
        Settings.chunk_size = self.config.chunk_sizes[0]
        Settings.chunk_overlap = self.config.chunk_overlap
        Settings.num_output = 512
    
    def _create_directories(self):
        """Crea directory necessarie"""
        Path(self.config.faiss_index_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
    
    def build_index(
        self,
        file_paths: List[str] = None,
        directories: List[str] = None,
        documents: List[Document] = None,
        batch_size: int = 50  # 🆕 Parametro batch size
    ) -> VectorStoreIndex:
        """
        Costruisce indice FLAT deterministico dai documenti con batching
        
        Args:
            file_paths: Lista di file da indicizzare
            directories: Liste di directory da indicizzare
            documents: Documenti già caricati (opzionale)
            batch_size: Numero di nodi per batch (default: 50)
            
        Returns:
            VectorStoreIndex costruito
        """
        logger.info("Building FLAT index (deterministic) with batching...")
        
        # Carica documenti se non forniti
        if documents is None:
            documents = self.document_processor.load_documents(
                file_paths=file_paths,
                directories=directories
            )
            
            if not documents:
                raise ValueError("No documents to index")
        
        # Crea nodi gerarchici
        leaf_nodes, all_nodes = self.document_processor.create_hierarchical_nodes(documents)
        
        logger.info(f"Total nodes to index: {len(leaf_nodes)}")
        
        # Setup FAISS FLAT index
        faiss_index = self.faiss_manager.create_index()
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Docstore per tutti i nodi
        docstore = SimpleDocumentStore()
        docstore.add_documents(all_nodes)
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )
        
        # 🆕 BATCHING: dividi nodi in batch
        num_batches = (len(leaf_nodes) + batch_size - 1) // batch_size
        logger.info(f"Processing {num_batches} batches of ~{batch_size} nodes each")
        
        # 🆕 Costruisci indice in batch
        import streamlit as st
        
        # Progress bar opzionale (se streamlit disponibile)
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
        except:
            progress_bar = None
            status_text = None
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(leaf_nodes))
            batch_nodes = leaf_nodes[start_idx:end_idx]
            
            batch_info = f"Batch {batch_idx + 1}/{num_batches}: {len(batch_nodes)} nodes"
            logger.info(batch_info)
            
            if status_text:
                status_text.text(f"🔄 {batch_info}")
            
            try:
                # 🆕 Costruisci indice per questo batch
                if batch_idx == 0:
                    # Primo batch: crea indice
                    self.index = VectorStoreIndex(
                        batch_nodes,
                        storage_context=self.storage_context,
                        show_progress=False,
                        use_async=self.config.async_processing
                    )
                else:
                    # Batch successivi: inserisci nodi nell'indice esistente
                    for node in batch_nodes:
                        self.index.insert_nodes([node])
                
                # Aggiorna progress bar
                if progress_bar:
                    progress_bar.progress((batch_idx + 1) / num_batches)
                
                # 🆕 Piccola pausa per evitare overload
                if batch_idx < num_batches - 1:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx + 1}: {e}")
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
                raise
    
        # Cleanup progress indicators
        if progress_bar:
            progress_bar.empty()
        if status_text:
            status_text.empty()
        
        # Setup retriever
        self.retrieval_manager.create_retriever(self.index, self.storage_context)
        
        # Inizializza query processor
        config = QueryConfig(
            llm=self.llm,
            embed_model=self.embedding_manager.model,
            use_llm_variants=True
        )
        
        self.query_processor = QueryProcessor(
            index=self.index,
            config=config
        )
        
        self.stats["index_built"] = True
        logger.info(f"FLAT index built successfully with {len(leaf_nodes)} nodes in {num_batches} batches")
        
        return self.index
    
    def setup_query_engine(self, response_mode: ResponseMode = ResponseMode.COMPACT):
        """Configura query engine"""
        if not self.retrieval_manager.retriever:
            raise ValueError("Index not built. Call build_index() first")
        
        # Setup postprocessors
        postprocessors = self.retrieval_manager.setup_postprocessors()
        
        # Crea query engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retrieval_manager.retriever,
            response_mode=response_mode.value,
            node_postprocessors=postprocessors,
            use_async=self.config.async_processing,
            streaming=False,
            response_synthesizer_kwargs={
                "use_async": self.config.async_processing
            }
        )
        
        logger.info(f"Query engine configured with mode: {response_mode.value}")
    
    
    def _compute_text_hash(self, text: str) -> int:
        """
        Calcola SimHash del testo per deduplication veloce
        
        Args:
            text: Testo da hashare
            
        Returns:
            Hash intero del testo (64-bit fingerprint)
        """
        import hashlib
        
        tokens = text.lower().split()
        
        # Crea vettore di bit
        hash_bits = 64
        v = [0] * hash_bits
        
        for token in tokens:
            # Hash del token
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            # Aggiorna vettore
            for i in range(hash_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Converti in hash finale
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Calcola Hamming distance tra due hash
        
        Args:
            hash1: Primo hash
            hash2: Secondo hash
            
        Returns:
            Hamming distance (numero di bit diversi)
        """
        xor = hash1 ^ hash2
        distance = 0
        
        for i in range(64):
            if xor & (1 << i):
                distance += 1
        
        return distance
    
    def _deduplicate_nodes(
        self,
        nodes: List,
        similarity_threshold: float = 0.85
    ) -> List:
        """
        Rimuove nodi duplicati usando SimHash con Hamming distance
        
        Args:
            nodes: Lista di nodi da dedupplicare
            similarity_threshold: Soglia di similarità [0, 1]
            
        Returns:
            Lista di nodi dedupplicati
        """
        if not nodes:
            return []
        
        dedup_start = time.time()
        unique_nodes = []
        seen_hashes = []
        duplicates_removed = 0
        
        # Calcola max Hamming distance dalla threshold
        hash_bits = 64
        max_hamming = int(hash_bits * (1 - similarity_threshold))
        
        for node in nodes:
            node_text = node.text
            node_hash = self._compute_text_hash(node_text)
            
            is_duplicate = False
            
            # Confronta con hash già visti
            for seen_hash in seen_hashes:
                distance = self._hamming_distance(node_hash, seen_hash)
                
                if distance <= max_hamming:
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            # Aggiungi solo se non è duplicato
            if not is_duplicate:
                unique_nodes.append(node)
                seen_hashes.append(node_hash)
        
        dedup_time = time.time() - dedup_start
        
        logger.info(
            f"Deduplication (SimHash): {len(nodes)} → {len(unique_nodes)} nodes "
            f"({duplicates_removed} duplicates removed, "
            f"threshold={similarity_threshold}, "
            f"time={dedup_time:.3f}s)"
        )
        
        return unique_nodes
    
    def query(self, question: str, enhance_query: bool = True) -> Dict[str, Any]:
        """
        Esegue query con enhancement, deduplication e reranking
        
        Flusso:
        1. Espandi query in multiple varianti
        2. Retrieval per ogni query
        3. Deduplicazione con SimHash
        4. Reranking con Jina
        5. Synthesis della risposta
        
        Args:
            question: Domanda da porre
            enhance_query: Se True, usa query enhancement
            
        Returns:
            Risultato con risposta e metadata
        """
        # Validazione input
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        start_time = time.time()
        
        # Setup query engine se necessario
        if not self.query_engine:
            self.setup_query_engine()
        
        # ✅ VERIFICA CHE similarity_top_k ESISTA
        similarity_top_k = getattr(self.config, 'similarity_top_k', 10)
        
        # Metadata tracking
        query_metadata = {
            "original_query": question,
            "enhanced": enhance_query
        }
        reranking_applied = False
        all_nodes = []
        
        # STEP 1: Query Enhancement
        if enhance_query and self.query_processor:
            try:
                expansion_start = time.time()
                
                # Genera query espanse
                expansion_result = self.query_processor.expand(
                    query=question,
                    max_queries=getattr(self.config, "max_queries", 10)
                )
                
                queries = expansion_result["queries"]
                expansions = expansion_result["expansions"]
                
                expansion_time = time.time() - expansion_start
                
                query_metadata.update({
                    "num_queries_generated": len(queries),
                    "queries": queries,
                    "expansions": {
                        "keywords": expansions.get("keywords", []),
                        "intent": expansions.get("intent", "general"),
                        "semantic_variants": expansions.get("semantic_variants", []),
                        "sub_queries": expansions.get("sub_queries", [])
                    },
                    "expansion_time": f"{expansion_time:.3f}s"
                })
                
                logger.info(f"Query expansion: {len(queries)} queries generated in {expansion_time:.3f}s")
                
                # STEP 2: Multi-Query Retrieval
                retrieval_start = time.time()
                
                for i, q in enumerate(queries):
                    try:
                        # ✅ MIGLIORA GESTIONE FALLBACK RETRIEVER
                        if not self.retrieval_manager.retriever:
                            logger.warning("Retriever not configured, creating fallback retriever")
                            retriever = self.index.as_retriever(
                                similarity_top_k=similarity_top_k
                            )
                        else:
                            retriever = self.retrieval_manager.retriever

                        nodes = retriever.retrieve(q)
                        all_nodes.extend(nodes)
                        
                        logger.debug(f"Query {i+1}/{len(queries)}: '{q[:50]}...' → {len(nodes)} nodes")
                        
                    except Exception as e:
                        logger.warning(f"Retrieval failed for query '{q[:50]}...': {e}")
                        continue
                
                retrieval_time = time.time() - retrieval_start
                
                query_metadata["retrieval"] = {
                    "total_nodes_retrieved": len(all_nodes),
                    "retrieval_time": f"{retrieval_time:.3f}s",
                    "avg_nodes_per_query": len(all_nodes) / len(queries) if queries else 0
                }
                
                self.stats["total_nodes_retrieved"] += len(all_nodes)
                
                logger.info(f"Multi-retrieval: {len(all_nodes)} total nodes from {len(queries)} queries in {retrieval_time:.3f}s")
                
                # STEP 3: Deduplication con SimHash
                if all_nodes:
                    dedup_threshold = getattr(self.config, "dedup_threshold", 0.85)
                    unique_nodes = self._deduplicate_nodes(all_nodes, dedup_threshold)
                    
                    query_metadata["deduplication"] = {
                        "nodes_before": len(all_nodes),
                        "nodes_after": len(unique_nodes),
                        "duplicates_removed": len(all_nodes) - len(unique_nodes),
                        "threshold": dedup_threshold
                    }
                    
                    self.stats["total_nodes_after_dedup"] += len(unique_nodes)
                    
                    all_nodes = unique_nodes
                
                # STEP 4: Reranking con Jina
                if self.config.use_reranker and all_nodes:
                    try:
                        rerank_start = time.time()
                        original_count = len(all_nodes)
                        
                        # ✅ USA similarity_top_k VERIFICATO
                        reranked_nodes = self.retrieval_manager.rerank_nodes(
                            query=question,
                            nodes=all_nodes,
                            top_n=min(similarity_top_k, len(all_nodes))
                        )
                        
                        rerank_time = time.time() - rerank_start
                        reranking_applied = True
                        
                        query_metadata["reranking"] = {
                            "applied": True,
                            "nodes_before": original_count,
                            "nodes_after": len(reranked_nodes),
                            "rerank_time": f"{rerank_time:.3f}s"
                        }
                        
                        self.stats["total_nodes_after_rerank"] += len(reranked_nodes)
                        
                        all_nodes = reranked_nodes
                        
                        logger.info(f"Reranking: {original_count} → {len(reranked_nodes)} nodes in {rerank_time:.3f}s")
                        
                    except Exception as e:
                        logger.warning(f"Reranking failed: {e}, continuing without reranking")
                        query_metadata["reranking"] = {
                            "applied": False,
                            "error": str(e)
                        }
                
            except Exception as e:
                logger.warning(f"Query enhancement pipeline failed: {e}, falling back to standard retrieval")
                all_nodes = []
                query_metadata["enhancement_error"] = str(e)
        
        # STEP 5: Synthesis
        try:
            synthesis_start = time.time()
            
            # Se abbiamo nodi enhanced, usiamo quelli
            if all_nodes:
                qb = QueryBundle(query_str=question)
                response = self.query_engine.synthesize(
                    query_bundle=qb,
                    nodes=all_nodes
                )
            else:
                # Fallback alla query normale
                logger.info("Falling back to standard query (no enhanced nodes)")
                response = self.query_engine.query(question)
            
            synthesis_time = time.time() - synthesis_start
            query_metadata["synthesis_time"] = f"{synthesis_time:.3f}s"
            
            # Prepara risultato
            result = {
                "question": question,
                "answer": str(response),
                "response_time": time.time() - start_time,
                "model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "index_type": "FLAT (deterministic)",
                "reranking_applied": reranking_applied,
                "query_metadata": query_metadata
            }
            
            # Aggiungi sources
            if all_nodes:
                sources = []
                for i, node in enumerate(all_nodes[:5]):
                    source = {
                        "text": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                        "score": float(node.score) if hasattr(node, 'score') and node.score else 0.0,
                        "metadata": node.metadata,
                        "reranked": reranking_applied
                    }
                    sources.append(source)
                
                result["sources"] = sources
                result["num_sources"] = len(sources)
            
            # Aggiorna statistiche
            self._update_stats(result["response_time"])
            
            logger.info(
                f"Query completed in {result['response_time']:.2f}s "
                f"(enhanced: {enhance_query}, reranked: {reranking_applied})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "response_time": time.time() - start_time,
                "reranking_applied": False,
                "query_metadata": query_metadata
            }
    
    def _update_stats(self, response_time: float):
        """Aggiorna statistiche interne"""
        self.stats["total_queries"] += 1
        
        # Media mobile per tempo di risposta
        prev_avg = self.stats["avg_retrieval_time"]
        n = self.stats["total_queries"]
        self.stats["avg_retrieval_time"] = (prev_avg * (n - 1) + response_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Ottieni statistiche complete della pipeline
        
        Returns:
            Dizionario con tutte le statistiche
        """
        stats = {
            "configuration": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.embedding_manager.config.dimension,
                "index_type": "FLAT (deterministic)",
                "chunk_sizes": self.config.chunk_sizes,
                "context_window": self.config.context_window,
                "temperature": self.config.temperature,
                "reranker_enabled": self.config.use_reranker,
                "dedup_threshold": getattr(self.config, "dedup_threshold", 0.85)
            },
            "data": self.document_processor.stats,
            "performance": {
                "total_queries": self.stats["total_queries"],
                "avg_response_time": f"{self.stats['avg_retrieval_time']:.3f}s",
                "index_built": self.stats["index_built"]
            },
            "retrieval_stats": {
                "total_nodes_retrieved": self.stats["total_nodes_retrieved"],
                "total_nodes_after_dedup": self.stats["total_nodes_after_dedup"],
                "total_nodes_after_rerank": self.stats["total_nodes_after_rerank"],
                "avg_nodes_retrieved": (
                    self.stats["total_nodes_retrieved"] / self.stats["total_queries"]
                    if self.stats["total_queries"] > 0 else 0
                ),
                "avg_dedup_reduction": (
                    1 - (self.stats["total_nodes_after_dedup"] / self.stats["total_nodes_retrieved"])
                    if self.stats["total_nodes_retrieved"] > 0 else 0
                )
            }
        }
        
        # Aggiungi metriche del reranker se disponibili
        if self.config.use_reranker:
            reranker_metrics = self.retrieval_manager.get_reranker_metrics()
            if reranker_metrics:
                stats["reranker"] = reranker_metrics
        
        return stats
