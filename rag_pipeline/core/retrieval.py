import logging
logger = logging.getLogger(__name__)
from typing import List
from config.settings import RAGConfig
from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.retrievers import AutoMergingRetriever
# ❌ RIMOSSO: Non serve più con reranker
# from llama_index.core.postprocessor import LongContextReorder
from core.jina_reranker import JinaReranker
import torch

class RetrievalManager:
    """Gestisce retrieval e postprocessing"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = None
        self.postprocessors = []
        self.jina_reranker = None
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Inizializza Jina reranker se abilitato
        if self.config.use_reranker:
            try:
                self.jina_reranker = JinaReranker(
                    model_name="jinaai/jina-reranker-v2-base-multilingual",
                    device=self.device,
                    use_fp16=True,
                    batch_size=32,
                    max_length=1024
                )
                logger.info("Jina reranker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Jina reranker: {e}")
                self.jina_reranker = None
    
    def create_retriever(
        self, 
        index: VectorStoreIndex, 
        storage_context: StorageContext
    ) -> AutoMergingRetriever:
        """Crea retriever ottimizzato"""
        base_retriever = index.as_retriever(
            similarity_top_k=self.config.similarity_top_k
        )
        
        if self.config.use_automerging:
            self.retriever = AutoMergingRetriever(
                base_retriever,
                storage_context,
                verbose=False
            )
            logger.info("AutoMergingRetriever configured")
        else:
            self.retriever = base_retriever
            logger.info("Standard retriever configured")
        
        return self.retriever
    
    def setup_postprocessors(self) -> List:
        """
        Configura postprocessors per query engine
        
        NOTA: Con Jina reranker attivo, non servono postprocessors aggiuntivi.
        Il reranking ML fornisce già l'ordine ottimale dei nodi.
        
        Postprocessors come LongContextReorder sono utili SOLO quando
        il ranking è basato su semplice similarità embedding (no reranker).
        """
        postprocessors = []
        
        # ❌ NESSUN POSTPROCESSOR con reranker attivo
        # Motivo: Jina reranker fornisce già ordine ottimale ML-based
        
        self.postprocessors = postprocessors
        logger.info(
            f"Configured {len(postprocessors)} postprocessors "
            f"(reranker handles optimal ordering)"
        )
        
        return postprocessors
    
    def rerank_nodes(self, query: str, nodes: List, top_n: int = 5) -> List:
        """
        Rerank nodes usando Jina reranker ML model
        
        Fornisce ordine ottimale basato su relevance semantica query-document.
        Nessun postprocessor aggiuntivo necessario dopo questo step.
        
        Args:
            query: Query string originale
            nodes: Lista di nodi da rerankare (già dedupplicati)
            top_n: Numero di nodi top da ritornare
            
        Returns:
            Lista di nodi rerankati in ordine ottimale (top_n migliori)
        """
        if not self.jina_reranker or not nodes:
            logger.warning("Jina reranker not available or no nodes to rerank")
            return nodes[:top_n]
        
        try:
            # Prepara documenti per reranking
            documents = [node.get_content() for node in nodes]
            
            # Rerank con Jina ML model
            response = self.jina_reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_n
            )
            
            # Riordina nodi secondo i risultati e aggiorna score
            reranked_nodes = []
            for result in response.results:
                original_node = nodes[result.index]
                # ✅ Aggiorna score con quello del reranker (più accurato)
                original_node.score = result.relevance_score
                reranked_nodes.append(original_node)
            
            logger.info(
                f"Reranked {len(nodes)} nodes → top {len(reranked_nodes)} "
                f"(scores: {[f'{n.score:.3f}' for n in reranked_nodes[:3]]})"
            )
            return reranked_nodes
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original nodes sorted by score")
            # Fallback: ordina per score esistente
            sorted_nodes = sorted(
                nodes, 
                key=lambda n: getattr(n, 'score', 0.0), 
                reverse=True
            )
            return sorted_nodes[:top_n]
    
    def get_reranker_metrics(self):
        """Ottieni metriche del reranker"""
        if self.jina_reranker:
            return self.jina_reranker.get_metrics()
        return {}
