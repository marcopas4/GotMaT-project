# -*- coding: utf-8 -*-
"""
Query Processor - Main Orchestrator
Coordina tutti i moduli per multi-query retrieval
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import QueryConfig, QueryExpansion, RetrievalResult
from .expansion import QueryExpansionModule
from .fusion import FusionStrategyFactory
from .cache import CacheManager

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Orchestratore principale per multi-query retrieval con fusion
    """
    
    def __init__(
        self,
        index,
        config: Optional[QueryConfig] = None
):
        """
        Args:
            index: Indice LlamaIndex per retrieval
            config: Configurazione del query processor
        """
        self.index = index
        self.config = config or QueryConfig()
        
        # Inizializza moduli
        self.expansion_module = QueryExpansionModule(
            index=index,
            embed_model=self.config.embed_model,
            llm=self.config.llm
        )
        
        self.fusion_strategy = FusionStrategyFactory.create(
            method=self.config.fusion_method,
            weights=self.config.query_weights
        )
        
        self.cache_manager = CacheManager(
            enabled=self.config.cache_enabled,
            max_size=self.config.cache_size
        )
        
        logger.info(f"QueryProcessor initialized with {self.config.fusion_method} fusion")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_queries: int = 10,
        fusion_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Esegue multi-query retrieval con fusion
        
        Args:
            query: Query originale
            top_k: Numero documenti finali da ritornare
            max_queries: Numero massimo di query da eseguire
            fusion_method: Override del metodo di fusion configurato
            
        Returns:
            Dizionario con nodi fusi, scores e metadata
        """
        # Check cache
        cache_key = f"{query}_{top_k}_{fusion_method or self.config.fusion_method}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result
        
        # 1. Genera espansioni query
        expansions = self.expansion_module.expand(query)
        
        # 2. Costruisci lista di query da eseguire
        queries_to_run = self._build_query_list(expansions, max_queries)
        
        # 3. Esegui tutte le ricerche
        all_results = self._execute_multi_retrieval(queries_to_run, top_k)
        
        # 4. Scegli strategia di fusion
        if fusion_method and fusion_method != self.config.fusion_method:
            fusion = FusionStrategyFactory.create(fusion_method, self.config.query_weights)
        else:
            fusion = self.fusion_strategy
        
        # 5. Fondi i risultati
        fused_results = fusion.fuse(all_results, top_k)
        
        # 6. Aggiungi metadata
        fused_results["query_info"] = {
            "original_query": query,
            "num_queries_executed": len(all_results),
            "fusion_method": fusion_method or self.config.fusion_method,
            "queries": [r.query for r in all_results],
            "expansions": {
                "keywords": expansions.keywords,
                "intent": expansions.intent,
                "variants": expansions.semantic_variants,
                "expanded_terms": expansions.expanded_terms
            }
        }
        
        # 7. Cache risultato
        self.cache_manager.set(cache_key, fused_results)
        
        return fused_results
    
    def _build_query_list(
        self,
        expansions: QueryExpansion,
        max_queries: int
    ) -> List[Dict[str, Any]]:
        """
        Costruisce lista prioritizzata di query da eseguire
        """
        queries = []
        weights = self.config.query_weights
        
        # 1. Query originale (sempre prima, peso maggiore)
        queries.append({
            "query": expansions.original,
            "type": "original",
            "weight": weights.get("original", 2.0),
            "top_k_multiplier": 2  # Recupera più risultati
        })
        
        # 2. Varianti semantiche (alta priorità)
        for variant in expansions.semantic_variants[:2]:
            if len(queries) < max_queries:
                queries.append({
                    "query": variant,
                    "type": "semantic_variant",
                    "weight": weights.get("semantic_variant", 1.5),
                    "top_k_multiplier": 1
                })
        
        # 3. Sub-queries (media priorità)
        for sub_query in expansions.sub_queries[:2]:
            if len(queries) < max_queries:
                queries.append({
                    "query": sub_query,
                    "type": "sub_query",
                    "weight": weights.get("sub_query", 1.2),
                    "top_k_multiplier": 1
                })
        
        # 4. Termini espansi (bassa priorità)
        for term in expansions.expanded_terms[:3]:
            if len(queries) < max_queries and len(term) > 3:
                queries.append({
                    "query": term,
                    "type": "expanded_term",
                    "weight": weights.get("expanded_term", 0.8),
                    "top_k_multiplier": 1
                })
        
        logger.debug(f"Built {len(queries)} queries from expansions")
        return queries
    
    def _execute_multi_retrieval(
        self,
        queries: List[Dict],
        base_top_k: int
    ) -> List[RetrievalResult]:
        """
        Esegue tutte le query di retrieval
        """
        results = []
        
        for query_info in queries:
            # Calcola top_k per questa query
            top_k = base_top_k * query_info["top_k_multiplier"]
            
            # Log query execution
            logger.debug(
                f"Retrieving: '{query_info['query'][:50]}...' "
                f"(type: {query_info['type']}, top_k: {top_k})"
            )
            
            # Esegui retrieval
            retrieval_result = self._single_retrieval(query_info["query"], top_k)
            
            # Crea risultato con metadata
            if retrieval_result["nodes"]:
                results.append(RetrievalResult(
                    query=query_info["query"],
                    query_type=query_info["type"],
                    nodes=retrieval_result["nodes"],
                    scores=retrieval_result["scores"],
                    weight=query_info["weight"]
                ))
            else:
                logger.warning(f"No results for query: {query_info['query'][:50]}...")
        
        return results
    
    def _single_retrieval(
        self,
        query: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Esegue singola query con cache di secondo livello
        """
        # Check cache per questa specifica query
        cache_key = f"single_{query}_{top_k}"
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached
        
        try:
            # Retrieve usando l'index
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            
            result = {
                "nodes": nodes,
                "scores": [float(n.score) if n.score else 0.0 for n in nodes]
            }
            
            # Cache risultato
            self.cache_manager.set(cache_key, result, ttl=300)  # 5 min cache
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
            return {"nodes": [], "scores": []}
    
    def clear_cache(self):
        """Pulisce tutte le cache"""
        self.cache_manager.clear()
        self.expansion_module.clear_cache()
        logger.info("All caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche di utilizzo"""
        return {
            "cache_stats": self.cache_manager.get_stats(),
            "expansion_stats": self.expansion_module.get_stats(),
            "config": {
                "fusion_method": self.config.fusion_method,
                "cache_enabled": self.config.cache_enabled,
                "weights": self.config.query_weights
            }
        }
