# -*- coding: utf-8 -*-
"""
Configuration and Data Classes for Query Processor
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
from enum import Enum


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
    
    # Expansion settings
    use_llm_variants: bool = True
    max_expanded_terms: int = 10
    max_semantic_variants: int = 3
    max_sub_queries: int = 3
    
    # Query generation limits
    max_queries: int = 10


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
