from dataclasses import dataclass, field
from enum import Enum
from typing import List
import os
from pathlib import Path

# Ottieni la directory dove si trova questo file config
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent  # rag_pipeline/

class IndexType(Enum):
    HNSW = "hnsw"
    IVF = "ivf"
    FLAT = "flat"

class ResponseMode(Enum):
    TREE_SUMMARIZE = "tree_summarize"
    COMPACT = "compact"
    SIMPLE = "simple"
    REFINE = "refine"

@dataclass
class EmbeddingConfig:
    name: str
    dimension: int
    max_length: int
    description: str
    batch_size: int = 32
    normalize: bool = True
    
@dataclass
class RAGConfig:
    """Configurazione completa della pipeline RAG"""
    # Modelli
    llm_model: str = "llama3.2:3b-instruct-q4_K_M"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    
    # Chunking
    chunk_sizes: List[int] = field(default_factory=lambda: [2048, 512, 128])
    chunk_overlap: int = 50
    
    # FAISS
    index_type: IndexType = IndexType.HNSW
    faiss_index_path: str = str(_PROJECT_ROOT / "data" / "indexes")
    
    # Storage
    storage_path: str = str(_PROJECT_ROOT / "storage")
    
    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    context_window: int = 4096
    
    # Cache
    enable_cache: bool = True
    cache_size: int = 100
    embeddings_cache_path: str = str(_PROJECT_ROOT / "embeddings_cache")
    
    # Data paths
    data_path: str = str(_PROJECT_ROOT / "data")
    evaluation_path: str = str(_PROJECT_ROOT / "evaluation")
    docs_path: str = str(_PROJECT_ROOT / "data" / "documents")
    
    # Performance
    num_threads: int = 8
    use_gpu: bool = True
    async_processing: bool = True
    
    # Retrieval
    similarity_top_k: int = 10
    use_reranker: bool = True
    use_automerging: bool = True
