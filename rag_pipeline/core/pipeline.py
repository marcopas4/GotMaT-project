from typing import List, Dict, Any
from pathlib import Path
import time
import json
import numpy as np
import logging as logger
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
    Settings
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from config.settings import RAGConfig, IndexType, ResponseMode
from core.embedding_manager import EmbeddingManager
from core.indexing import FAISSIndexManager
from core.document_processor import DocumentProcessor
from core.retrieval import RetrievalManager
from core.cache import QueryCacheManager
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from core.query_expansion import QueryProcessor, QueryConfig, FusionMethod
from llama_index.core.schema import QueryBundle

class OptimizedRAGPipeline:
    """Pipeline RAG principale che coordina tutti i componenti"""
    
    def __init__(self, config: RAGConfig = None):
        """
        Inizializza pipeline con configurazione
        
        Args:
            config: Configurazione RAG (usa default se None)
        """
        self.config = config or RAGConfig()
        
        # Inizializza componenti
        self.embedding_manager = EmbeddingManager(self.config.embedding_model)
        self.faiss_manager = FAISSIndexManager(
            self.embedding_manager.config.dimension,
            self.config.index_type
        )
        self.document_processor = DocumentProcessor(
            self.config.chunk_sizes,
            self.config.chunk_overlap
        )
        self.retrieval_manager = RetrievalManager(self.config)
        self.cache_manager = QueryCacheManager(
            self.config.enable_cache,
            self.config.cache_size
        )
        
        # Query processor per enhancement
        self.query_processor = None
        
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
            "index_built": False
        }
        
        # Crea directory necessarie
        self._create_directories()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _setup_llm(self):
        """Configura LLM Ollama"""
        self.llm = Ollama(
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
        )
        logger.info(f"LLM configured: {self.config.llm_model}")
    
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
        documents: List[Document] = None
    ) -> VectorStoreIndex:
        """
        Costruisce indice dai documenti
        
        Args:
            file_paths: Lista di file da indicizzare
            directories: Liste di directory da indicizzare
            documents: Documenti già caricati (opzionale)
            
        Returns:
            VectorStoreIndex costruito
        """
        logger.info("Building optimized index...")
        
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
        
        # Setup FAISS index
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
        
        # Costruisci indice
        self.index = VectorStoreIndex(
            leaf_nodes,
            storage_context=self.storage_context,
            show_progress=True,
            use_async=self.config.async_processing
        )
        
        # Training per IVF se necessario
        if self.config.index_type == IndexType.IVF:
            self._train_ivf_index(leaf_nodes)
        
        # Setup retriever
        self.retrieval_manager.create_retriever(self.index, self.storage_context)
        
        # Inizializza query processor con accesso al vector store
        config = QueryConfig(
            llm=self.llm,
            embed_model=self.embedding_manager.model,
            fusion_method=FusionMethod.RECIPROCAL_RANK.value,
            cache_enabled=True,
            use_llm_variants=True
        )
        
        self.query_processor = QueryProcessor(
            config=config,
        )
        # Passa l'index al query processor
        self.query_processor.index = self.index
        
        # Salva indice
        self.save_index()
        
        self.stats["index_built"] = True
        logger.info("Index built successfully!")
        
        return self.index
    
    def _train_ivf_index(self, nodes: List, max_training: int = 1000):
        """Addestra IVF index con subset di nodi"""
        logger.info("Training IVF index...")
        
        training_nodes = nodes[:min(max_training, len(nodes))]
        training_vectors = []
        
        for node in training_nodes:
            embedding = self.embedding_manager.get_embedding(node.text)
            training_vectors.append(embedding)
        
        if training_vectors:
            training_array = np.vstack(training_vectors).astype('float32')
            self.faiss_manager.train_index(
                self.storage_context.vector_store._faiss_index,
                training_array
            )
    
    def setup_query_engine(self, response_mode: ResponseMode = ResponseMode.TREE_SUMMARIZE):
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
    
    def query(self, question: str, enhance_query: bool = True) -> Dict[str, Any]:
        """
        Esegue query con enhancement avanzato
        
        Args:
            question: Domanda da porre
            enhance_query: Se True, usa query enhancement
            
        Returns:
            Risultato con risposta e metadata
        """
        start_time = time.time()
        
        # Setup query engine se necessario
        if not self.query_engine:
            self.setup_query_engine()
        
        # Check cache
        query_hash = self.cache_manager.compute_hash(question)
        cached_result = self.cache_manager.get(query_hash)
        
        if cached_result:
            cached_result["from_cache"] = True
            return cached_result
        
        # Query enhancement avanzato
        enhanced_results = None
        query_metadata = {}
        
        if enhance_query and self.query_processor:
            try:
                # Usa il nuovo enhanced_retrieval invece del process_query
                enhanced_results = self.query_processor.retrieve(
                    query=question,
                    top_k=self.config.similarity_top_k,
                    max_queries=getattr(self.config, "max_queries", 10),
                    fusion_method=FusionMethod.RECIPROCAL_RANK.value
                )
                
                qi = enhanced_results.get("query_info", {})
                query_metadata = {
                    "original_query": question,
                    "method": qi.get("fusion_method") or enhanced_results.get("method"),
                    "fusion_method": qi.get("fusion_method"),
                    "num_queries": qi.get("num_queries_executed", len(qi.get("queries", []))),
                    "queries_executed": qi.get("queries", [])
                }

                logger.info(f"Enhanced retrieval completed with {query_metadata['num_queries']} queries")
                
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")
                enhanced_results = None
    
        # Esegui query
        try:
            # Se abbiamo risultati enhanced, usiamo quelli
            if enhanced_results and enhanced_results.get("nodes"):
                qb = QueryBundle(query_str=question)
                response = self.query_engine.synthesize(
                    query_bundle=qb,
                    nodes=enhanced_results["nodes"]
                )
            else:
                # Fallback alla query normale
                response = self.query_engine.query(question)
        
            # Prepara risultato
            result = {
                "question": question,
                "answer": str(response),
                "response_time": time.time() - start_time,
                "model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "from_cache": False,
                "query_metadata": query_metadata
            }
            
            # Aggiungi sources e metadata dal retrieval avanzato
            if enhanced_results and enhanced_results.get("nodes"):
                sources = []
                for i, node in enumerate(enhanced_results["nodes"][:5]):
                    source = {
                        "text": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                        "score": float(enhanced_results["scores"][i]) if enhanced_results.get("scores") else 0.0,
                        "metadata": node.metadata,
                        "fusion_metadata": enhanced_results.get("metadata", [])[i] if enhanced_results.get("metadata") else {}
                    }
                    sources.append(source)
                
                result["sources"] = sources
                result["num_sources"] = len(sources)
                result["fusion_details"] = enhanced_results.get("fusion_details", {})
        
            # Cache risultato
            self.cache_manager.set(query_hash, result)
            
            # Aggiorna statistiche
            self._update_stats(result["response_time"])
            
            logger.info(f"Query completed in {result['response_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "response_time": time.time() - start_time,
                "from_cache": False
            }
    
    def _update_stats(self, response_time: float):
        """Aggiorna statistiche interne"""
        self.stats["total_queries"] += 1
        
        # Media mobile per tempo di risposta
        prev_avg = self.stats["avg_retrieval_time"]
        n = self.stats["total_queries"]
        self.stats["avg_retrieval_time"] = (prev_avg * (n - 1) + response_time) / n
    
    def save_index(self):
        """Salva indice e metadata su disco"""
        if not self.index or not self.storage_context:
            logger.warning("No index to save")
            return
        
        # Salva storage context
        self.storage_context.persist(persist_dir=self.config.storage_path)
        
        # Salva FAISS index
        faiss_index = self.storage_context.vector_store._faiss_index
        index_path = f"{self.config.faiss_index_path}/index.faiss"
        self.faiss_manager.save_index(faiss_index, index_path)
        
        # Salva metadata
        metadata = {
            "config": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "chunk_sizes": self.config.chunk_sizes,
                "index_type": self.config.index_type.value
            },
            "stats": {
                **self.stats,
                **self.document_processor.stats,
                "cache": self.cache_manager.get_stats()
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = f"{self.config.faiss_index_path}/metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Index saved to {self.config.storage_path}")
    
    def load_index(self) -> bool:
        """
        Carica indice esistente da disco
        
        Returns:
            True se caricato con successo, False altrimenti
        """
        try:
            # Check esistenza
            if not Path(self.config.storage_path).exists():
                logger.info("No existing index found")
                return False
            
            # Carica metadata
            metadata_path = f"{self.config.faiss_index_path}/metadata.json"
            if Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.stats.update(metadata.get("stats", {}))
                    logger.info(f"Metadata loaded - Created: {metadata.get('created_at', 'N/A')}")
            
            # Carica FAISS index
            index_path = f"{self.config.faiss_index_path}/index.faiss"
            if Path(index_path).exists():
                faiss_index = self.faiss_manager.load_index(index_path)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
            else:
                faiss_index = self.faiss_manager.create_index()
                vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # Ricostruisci storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=self.config.storage_path
            )
            
            # Carica indice
            self.index = load_index_from_storage(
                self.storage_context,
                embed_model=self.embedding_manager.model
            )
            
            # Setup retriever
            self.retrieval_manager.create_retriever(self.index, self.storage_context)
            config = QueryConfig(
                llm=self.llm,
                embed_model=self.embedding_manager.model,
                fusion_method=FusionMethod.RECIPROCAL_RANK.value,
                cache_enabled=True,
                use_llm_variants=True
            )
            # Inizializza query processor
            self.query_processor = QueryProcessor(
                config=config,
                index =self.index,
            )
            
            self.stats["index_built"] = True
            logger.info("Index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def update_index(self, file_paths: List[str] = None, directories: List[str] = None):
        """
        Aggiorna indice con nuovi documenti
        
        Args:
            file_paths: Nuovi file da aggiungere
            directories: Nuove directory da aggiungere
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        # Carica nuovi documenti
        new_documents = self.document_processor.load_documents(
            file_paths=file_paths,
            directories=directories
        )
        
        if not new_documents:
            logger.warning("No new documents to add")
            return
        
        # Crea nodi
        leaf_nodes, all_nodes = self.document_processor.create_hierarchical_nodes(new_documents)
        
        # Inserisci nell'indice
        self.index.insert_nodes(leaf_nodes)
        
        # Aggiorna docstore
        if self.storage_context.docstore:
            self.storage_context.docstore.add_documents(all_nodes)
        
        # Salva indice aggiornato
        self.save_index()
        
        logger.info(f"Index updated with {len(new_documents)} new documents")
    
    def clear_cache(self):
        """Pulisce cache delle query"""
        self.cache_manager.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Ottieni statistiche complete della pipeline
        
        Returns:
            Dizionario con tutte le statistiche
        """
        return {
            "configuration": {
                "llm_model": self.config.llm_model,
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.embedding_manager.config.dimension,
                "index_type": self.config.index_type.value,
                "chunk_sizes": self.config.chunk_sizes,
                "context_window": self.config.context_window,
                "temperature": self.config.temperature
            },
            "data": self.document_processor.stats,
            "performance": {
                "total_queries": self.stats["total_queries"],
                "avg_response_time": f"{self.stats['avg_retrieval_time']:.3f}s",
                "index_built": self.stats["index_built"]
            },
            "cache": self.cache_manager.get_stats()
        }
