# -*- coding: utf-8 -*-
"""
Configuration and Data Classes for Query Processor
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class FusionMethod(Enum):
    """Metodi di fusion disponibili"""
    RECIPROCAL_RANK = "reciprocal_rank"
    WEIGHTED_SUM = "weighted_sum"
    MAX_SCORE = "max_score"
    HYBRID = "hybrid"  # Combina più metodi


class QueryIntent(Enum):
    """Tipi di intent per le query"""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"
    LIST = "list"
    GENERAL = "general"


@dataclass
class QueryConfig:
    """Configurazione principale del Query Processor"""
    
    # Modelli
    embed_model: Optional[Any] = None
    llm: Optional[Any] = None
    
    # Fusion settings
    fusion_method: str = FusionMethod.RECIPROCAL_RANK.value
    query_weights: Dict[str, float] = field(default_factory=lambda: {
        "original": 2.0,
        "semantic_variant": 1.5,
        "sub_query": 1.2,
        "expanded_term": 0.8
    })
    
    # Cache settings
    cache_enabled: bool = True
    cache_size: int = 100
    cache_ttl: int = 3600  # 1 ora default
    
    # Expansion settings
    use_llm_variants: bool = True
    max_expanded_terms: int = 10
    max_semantic_variants: int = 3
    max_sub_queries: int = 3
    
    # Retrieval settings
    default_top_k: int = 10
    max_queries: int = 10
    
    # Performance
    async_retrieval: bool = False
    batch_size: int = 32


@dataclass
class QueryExpansion:
    """Container per tutti i tipi di espansione query"""
    original: str
    expanded_terms: List[str] = field(default_factory=list)
    semantic_variants: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    intent: str = QueryIntent.GENERAL.value
    keywords: List[str] = field(default_factory=list)
    language: str = "it"  # Default italiano
    
    def get_all_expansions(self) -> List[str]:
        """Ritorna tutte le espansioni in una lista unica"""
        all_expansions = []
        all_expansions.extend(self.semantic_variants)
        all_expansions.extend(self.sub_queries)
        all_expansions.extend(self.expanded_terms)
        return all_expansions
    
    def count_expansions(self) -> int:
        """Conta il numero totale di espansioni"""
        return (
            len(self.semantic_variants) +
            len(self.sub_queries) +
            len(self.expanded_terms)
        )


@dataclass
class RetrievalResult:
    """Risultato di una singola query di retrieval"""
    query: str
    query_type: str  # "original", "expanded_term", "semantic_variant", "sub_query"
    nodes: List[Any]
    scores: List[float]
    weight: float = 1.0
    
    def __post_init__(self):
        """Validazione post-inizializzazione"""
        if len(self.nodes) != len(self.scores):
            raise ValueError(
                f"Mismatch between nodes ({len(self.nodes)}) "
                f"and scores ({len(self.scores)})"
            )
    
    def get_top_score(self) -> float:
        """Ritorna lo score più alto"""
        return max(self.scores) if self.scores else 0.0
    
    def get_avg_score(self) -> float:
        """Ritorna lo score medio"""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


@dataclass
class FusionResult:
    """Risultato della fusion di multiple queries"""
    nodes: List[Any]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    fusion_stats: Dict[str, Any]
    
    def get_node_by_rank(self, rank: int) -> Optional[Any]:
        """Ritorna il nodo a un dato rank (1-based)"""
        if 0 < rank <= len(self.nodes):
            return self.nodes[rank - 1]
        return None
