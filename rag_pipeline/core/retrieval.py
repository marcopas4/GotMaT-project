import logging as logger
from typing import List
from rag_pipeline.config.settings import RAGConfig
from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
    LongContextReorder
)

class RetrievalManager:
    """Gestisce retrieval e postprocessing"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = None
        self.postprocessors = []
    
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
        
        if self.config.use_reranker:
            # Reranking con cross-encoder
            postprocessors.append(
                SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                    top_n=5
                )
            )
        
        # Long context reordering
        postprocessors.append(LongContextReorder())
        
        self.postprocessors = postprocessors
        logger.info(f"Configured {len(postprocessors)} postprocessors")
        
        return postprocessors
