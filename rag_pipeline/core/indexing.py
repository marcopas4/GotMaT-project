import faiss
from config.settings import IndexType
import numpy as np  
import logging as logger


class FAISSIndexManager:
    """Gestisce creazione e configurazione degli indici FAISS"""
    
    def __init__(self, dimension: int, index_type: IndexType = IndexType.HNSW):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        
    def create_index(self) -> faiss.Index:
        """Crea un nuovo indice FAISS ottimizzato"""
        if self.index_type == IndexType.HNSW:
            return self._create_hnsw_index()
        elif self.index_type == IndexType.IVF:
            return self._create_ivf_index()
        else:
            return self._create_flat_index()
    
    def _create_hnsw_index(self) -> faiss.Index:
        """Crea indice HNSW ottimizzato per M4"""
        M = 32  # Connessioni per nodo
        ef_construction = 200
        ef_search = 50
        
        index = faiss.IndexHNSWFlat(self.dimension, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        
        logger.info(f"HNSW index created (M={M}, ef_search={ef_search})")
        return index
    
    def _create_ivf_index(self) -> faiss.Index:
        """Crea indice IVF per dataset grandi"""
        nlist = 100
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        logger.info(f"IVF index created (nlist={nlist})")
        return index
    
    def _create_flat_index(self) -> faiss.Index:
        """Crea indice flat per ricerca esaustiva"""
        index = faiss.IndexFlatL2(self.dimension)
        logger.info("Flat L2 index created")
        return index
    
    def train_index(self, index: faiss.Index, training_vectors: np.ndarray):
        """Addestra indice IVF se necessario"""
        if self.index_type == IndexType.IVF and not index.is_trained:
            index.train(training_vectors)
            logger.info(f"Index trained with {len(training_vectors)} vectors")
    
    def save_index(self, index: faiss.Index, path: str):
        """Salva indice su disco"""
        faiss.write_index(index, path)
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str) -> faiss.Index:
        """Carica indice da disco"""
        index = faiss.read_index(path)
        
        # Ripristina configurazioni HNSW se necessario
        if self.index_type == IndexType.HNSW and hasattr(index, "hnsw"):
            index.hnsw.efSearch = 50
            
        logger.info(f"Index loaded from {path}")
        return index

