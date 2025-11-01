import logging
logger = logging.getLogger(__name__)
from typing import List
from config.settings import RAGConfig
from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    LongContextReorder
)
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
                    device=self.device,  # Apple Silicon M4
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
        """Configura postprocessors per migliorare retrieval"""
        postprocessors = []
        
        # Filtro similarity
        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=0.45)
        )
        
        # Long context reordering
        postprocessors.append(LongContextReorder())
        
        self.postprocessors = postprocessors
        logger.info(f"Configured {len(postprocessors)} postprocessors")
        
        return postprocessors
    
    def rerank_nodes(self, query: str, nodes: List, top_n: int = 5) -> List:
        """
        Rerank nodes usando Jina reranker
        
        Args:
            query: Query string
            nodes: Lista di nodi da rerankare
            top_n: Numero di nodi top da ritornare
            
        Returns:
            Lista di nodi rerankati
        """
        if not self.jina_reranker or not nodes:
            return nodes
        
        try:
            # Prepara documenti per reranking
            documents = [node.get_content() for node in nodes]
            
            # Rerank con Jina
            response = self.jina_reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_n
            )
            
            # Riordina nodi secondo i risultati
            reranked_nodes = []
            for result in response.results:
                reranked_nodes.append(nodes[result.index])
            
            logger.info(f"Reranked {len(nodes)} nodes to top {len(reranked_nodes)}")
            return reranked_nodes
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original nodes")
            return nodes[:top_n]
    
    def get_reranker_metrics(self):
        """Ottieni metriche del reranker"""
        if self.jina_reranker:
            return self.jina_reranker.get_metrics()
        return {}
