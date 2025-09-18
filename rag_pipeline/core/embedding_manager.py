from config.settings import EmbeddingConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import numpy as np
import logging as logger

class EmbeddingManager:
    """Gestisce modelli di embedding e configurazioni"""
    
    # Configurazioni predefinite per diversi modelli
    CONFIGS = {
        "nomic-ai/nomic-embed-text-v1.5": EmbeddingConfig(
            name="nomic-ai/nomic-embed-text-v1.5",
            dimension=768,
            max_length=8192,
            description="Ottimizzato per ARM, lungo contesto"
        ),
        "intfloat/e5-small-v2": EmbeddingConfig(
            name="intfloat/e5-small-v2",
            dimension=384,
            max_length=512,
            description="Veloce, accurato, leggero"
        ),
        "BAAI/bge-small-en-v1.5": EmbeddingConfig(
            name="BAAI/bge-small-en-v1.5",
            dimension=384,
            max_length=512,
            description="Buon compromesso velocità/qualità"
        ),
        "sentence-transformers/all-MiniLM-L6-v2": EmbeddingConfig(
            name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            max_length=256,
            description="Ultra veloce, buono per prototipazione"
        )
    }
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.config = self.CONFIGS.get(model_name, self.CONFIGS["BAAI/bge-small-en-v1.5"])
        self.model = self._initialize_model()
        logger.info(f"Embedding model initialized: {self.config.name} (dim={self.config.dimension})")
    
    def _initialize_model(self) -> HuggingFaceEmbedding:
        """Inizializza il modello di embedding"""
        return HuggingFaceEmbedding(
            model_name=self.config.name,
            cache_folder="./embeddings_cache",
            embed_batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            normalize=self.config.normalize,
            trust_remote_code=True
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Ottiene embedding per un testo"""
        return np.array(self.model.get_text_embedding(text)).astype('float32')
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Ottiene embeddings per batch di testi"""
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
        return np.vstack(embeddings)

