# -*- coding: utf-8 -*-
"""
Query Processor - Query Expansion Only
Genera query multiple per retrieval avanzato
"""

import logging
from typing import List, Dict, Any, Optional

from .config import QueryConfig, QueryExpansion
from .expansion import QueryExpansionModule

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Generatore di query multiple per retrieval
    NON esegue retrieval - solo espansione query
    """
    
    def __init__(
        self,
        index,
        config: Optional[QueryConfig] = None
    ):
        """
        Args:
            index: Indice LlamaIndex (serve per expansion module)
            config: Configurazione del query processor
        """
        self.config = config or QueryConfig()
        
        # Inizializza modulo di espansione
        self.expansion_module = QueryExpansionModule(
            index=index,
            embed_model=self.config.embed_model,
            llm=self.config.llm
        )
        
        logger.info("QueryProcessor initialized for query expansion")
    
    def expand(
        self,
        query: str,
        max_queries: int = 10
    ) -> Dict[str, Any]:
        """
        Espande query in multiple varianti
        
        Args:
            query: Query originale
            max_queries: Numero massimo di query da generare
            
        Returns:
            Dict con query espanse e metadata
        """
        # Genera espansioni
        expansions = self.expansion_module.expand(query)
        
        # Costruisci lista prioritizzata
        queries = self._build_query_list(expansions, max_queries)
        
        result = {
            "queries": queries,
            "expansions": {
                "original": expansions.original,
                "keywords": expansions.keywords,
                "intent": expansions.intent,
                "semantic_variants": expansions.semantic_variants,
                "sub_queries": expansions.sub_queries,
                "expanded_terms": expansions.expanded_terms
            }
        }
        
        logger.info(f"Expanded '{query[:50]}...' into {len(queries)} queries")
        return result
    
    def _build_query_list(
        self,
        expansions: QueryExpansion,
        max_queries: int
    ) -> List[str]:
        """
        Costruisce lista prioritizzata di query (solo stringhe)
        
        Priority order:
        1. Original query (always first)
        2. Semantic variants (high priority)
        3. Sub-queries (medium priority)
        4. Expanded terms (low priority)
        """
        queries = []
        
        # 1. Query originale
        queries.append(expansions.original)
        
        # 2. Varianti semantiche (max 2)
        for variant in expansions.semantic_variants[:2]:
            if len(queries) < max_queries:
                queries.append(variant)
        
        # 3. Sub-queries (max 2)
        for sub_query in expansions.sub_queries[:2]:
            if len(queries) < max_queries:
                queries.append(sub_query)
        
        # 4. Termini espansi (max 3, min length 3)
        for term in expansions.expanded_terms[:3]:
            if len(queries) < max_queries and len(term) > 3:
                queries.append(term)
        
        logger.debug(f"Built {len(queries)} queries from expansions")
        return queries
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche di utilizzo"""
        stats = {
            "config": {
                "max_queries_default": 10,
                "query_types": ["original", "semantic_variants", "sub_queries", "expanded_terms"],
                "limits": {
                    "semantic_variants": 2,
                    "sub_queries": 2,
                    "expanded_terms": 3
                }
            }
        }
        
        # Aggiungi stats da expansion module se disponibili
        try:
            expansion_stats = self.expansion_module.get_stats()
            stats["expansion"] = expansion_stats
        except (AttributeError, Exception) as e:
            logger.debug(f"Expansion stats not available: {e}")
        
        return stats
