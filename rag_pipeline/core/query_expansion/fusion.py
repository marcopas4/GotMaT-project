# -*- coding: utf-8 -*-
"""
Fusion Strategies for Multi-Query Retrieval
Implementa diversi metodi per combinare risultati da multiple queries
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

from .config import RetrievalResult, FusionResult

logger = logging.getLogger(__name__)


class FusionStrategy(ABC):
    """Classe base astratta per strategie di fusion"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Pesi per tipo di query
        """
        self.weights = weights or {
            "original": 2.0,
            "semantic_variant": 1.5,
            "sub_query": 1.2,
            "expanded_term": 0.8
        }
    
    @abstractmethod
    def fuse(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Fonde risultati da multiple queries
        
        Args:
            results: Lista di risultati da fondere
            top_k: Numero di documenti da ritornare
            
        Returns:
            Dizionario con nodi fusi, scores e metadata
        """
        pass
    
    def _get_node_id(self, node) -> str:
        """
        Genera ID univoco per un nodo
        """
        if hasattr(node, 'node_id'):
            return node.node_id
        elif hasattr(node, 'text'):
            return hashlib.md5(node.text.encode()).hexdigest()
        return str(id(node))


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF)
    Score = Σ(weight * 1/(k + rank))
    """
    
    def __init__(self, weights: Dict[str, float] = None, k: int = 60):
        """
        Args:
            weights: Pesi per tipo di query
            k: Parametro k per RRF (default 60)
        """
        super().__init__(weights)
        self.k = k
    
    def fuse(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Implementa Reciprocal Rank Fusion
        """
        # Accumula scores
        node_scores = defaultdict(float)
        node_objects = {}
        node_metadata = defaultdict(lambda: {
            "sources": [],
            "ranks": [],
            "types": [],
            "weights": []
        })
        
        # Processa ogni risultato
        for result in results:
            weight = self.weights.get(result.query_type, 1.0) * result.weight
            
            for rank, (node, score) in enumerate(zip(result.nodes, result.scores), 1):
                node_id = self._get_node_id(node)
                
                # Formula RRF con peso
                fusion_score = weight * (1.0 / (self.k + rank))
                node_scores[node_id] += fusion_score
                
                # Salva nodo se è la prima volta
                if node_id not in node_objects:
                    node_objects[node_id] = node
                
                # Accumula metadata
                node_metadata[node_id]["sources"].append(result.query)
                node_metadata[node_id]["ranks"].append(rank)
                node_metadata[node_id]["types"].append(result.query_type)
                node_metadata[node_id]["weights"].append(weight)
        
        # Ordina per score finale
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepara output
        final_nodes = []
        final_scores = []
        final_metadata = []
        
        for node_id, fusion_score in sorted_nodes:
            final_nodes.append(node_objects[node_id])
            final_scores.append(fusion_score)
            
            meta = node_metadata[node_id]
            final_metadata.append({
                "fusion_score": fusion_score,
                "appeared_in": len(meta["sources"]),
                "avg_rank": np.mean(meta["ranks"]),
                "min_rank": min(meta["ranks"]),
                "max_rank": max(meta["ranks"]),
                "query_types": list(set(meta["types"])),
                "avg_weight": np.mean(meta["weights"])
            })
        
        return {
            "nodes": final_nodes,
            "scores": final_scores,
            "metadata": final_metadata,
            "fusion_stats": {
                "method": "reciprocal_rank",
                "total_unique_nodes": len(node_scores),
                "k_parameter": self.k,
                "queries_processed": len(results)
            }
        }


class WeightedSumFusion(FusionStrategy):
    """
    Weighted Sum Fusion
    Score = Σ(weight * score) * appearance_bonus
    """
    
    def __init__(self, weights: Dict[str, float] = None, appearance_bonus: float = 0.1):
        """
        Args:
            weights: Pesi per tipo di query
            appearance_bonus: Bonus per ogni apparizione aggiuntiva
        """
        super().__init__(weights)
        self.appearance_bonus = appearance_bonus
    
    def fuse(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Implementa Weighted Sum Fusion
        """
        node_scores = defaultdict(float)
        node_objects = {}
        node_appearances = defaultdict(int)
        node_sources = defaultdict(list)
        
        # Accumula scores
        for result in results:
            weight = self.weights.get(result.query_type, 1.0) * result.weight
            
            for node, score in zip(result.nodes, result.scores):
                node_id = self._get_node_id(node)
                
                # Accumula score pesato
                node_scores[node_id] += weight * score
                node_appearances[node_id] += 1
                node_sources[node_id].append(result.query_type)
                
                if node_id not in node_objects:
                    node_objects[node_id] = node
        
        # Applica bonus per apparizioni multiple
        for node_id in node_scores:
            bonus = 1.0 + (self.appearance_bonus * (node_appearances[node_id] - 1))
            node_scores[node_id] *= bonus
        
        # Ordina per score
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepara output
        return {
            "nodes": [node_objects[nid] for nid, _ in sorted_nodes],
            "scores": [score for _, score in sorted_nodes],
            "metadata": [
                {
                    "appearances": node_appearances[nid],
                    "sources": list(set(node_sources[nid]))
                }
                for nid, _ in sorted_nodes
            ],
            "fusion_stats": {
                "method": "weighted_sum",
                "total_unique_nodes": len(node_scores),
                "appearance_bonus": self.appearance_bonus
            }
        }


class MaxScoreFusion(FusionStrategy):
    """
    Max Score Fusion
    Score = max(all weighted scores for node)
    """
    
    def fuse(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Implementa Max Score Fusion
        """
        node_max_scores = {}
        node_objects = {}
        node_best_source = {}
        node_all_scores = defaultdict(list)
        
        # Trova score massimo per ogni nodo
        for result in results:
            weight = self.weights.get(result.query_type, 1.0) * result.weight
            
            for node, score in zip(result.nodes, result.scores):
                node_id = self._get_node_id(node)
                weighted_score = weight * score
                
                # Traccia tutti gli scores
                node_all_scores[node_id].append(weighted_score)
                
                # Aggiorna se è il massimo
                if node_id not in node_max_scores or weighted_score > node_max_scores[node_id]:
                    node_max_scores[node_id] = weighted_score
                    node_objects[node_id] = node
                    node_best_source[node_id] = result.query_type
        
        # Ordina per score massimo
        sorted_nodes = sorted(
            node_max_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepara output
        return {
            "nodes": [node_objects[nid] for nid, _ in sorted_nodes],
            "scores": [score for _, score in sorted_nodes],
            "metadata": [
                {
                    "best_source": node_best_source[nid],
                    "max_score": node_max_scores[nid],
                    "all_scores": node_all_scores[nid],
                    "score_variance": np.var(node_all_scores[nid])
                }
                for nid, _ in sorted_nodes
            ],
            "fusion_stats": {
                "method": "max_score",
                "total_unique_nodes": len(node_max_scores)
            }
        }


class HybridFusion(FusionStrategy):
    """
    Hybrid Fusion - Combina multiple strategie
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        strategies_weights: Dict[str, float] = None
    ):
        """
        Args:
            weights: Pesi per tipo di query
            strategies_weights: Pesi per combinare diverse strategie
        """
        super().__init__(weights)
        
        self.strategies_weights = strategies_weights or {
            "rrf": 0.5,
            "weighted_sum": 0.3,
            "max_score": 0.2
        }
        
        # Inizializza strategie componenti
        self.rrf = ReciprocalRankFusion(weights)
        self.weighted_sum = WeightedSumFusion(weights)
        self.max_score = MaxScoreFusion(weights)
    
    def fuse(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Combina risultati da multiple strategie
        """
        # Ottieni risultati da ogni strategia
        rrf_result = self.rrf.fuse(results, top_k * 2)
        ws_result = self.weighted_sum.fuse(results, top_k * 2)
        ms_result = self.max_score.fuse(results, top_k * 2)
        
        # Combina scores
        combined_scores = defaultdict(float)
        all_nodes = {}
        
        # Aggiungi scores da RRF
        for node, score in zip(rrf_result["nodes"], rrf_result["scores"]):
            node_id = self._get_node_id(node)
            combined_scores[node_id] += score * self.strategies_weights["rrf"]
            all_nodes[node_id] = node
        
        # Aggiungi scores da Weighted Sum
        for node, score in zip(ws_result["nodes"], ws_result["scores"]):
            node_id = self._get_node_id(node)
            # Normalizza score per renderlo comparabile
            normalized_score = score / max(ws_result["scores"]) if ws_result["scores"] else 0
            combined_scores[node_id] += normalized_score * self.strategies_weights["weighted_sum"]
            all_nodes[node_id] = node
        
        # Aggiungi scores da Max Score
        for node, score in zip(ms_result["nodes"], ms_result["scores"]):
            node_id = self._get_node_id(node)
            # Normalizza score
            normalized_score = score / max(ms_result["scores"]) if ms_result["scores"] else 0
            combined_scores[node_id] += normalized_score * self.strategies_weights["max_score"]
            all_nodes[node_id] = node
        
        # Ordina per score combinato
        sorted_nodes = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepara output
        return {
            "nodes": [all_nodes[nid] for nid, _ in sorted_nodes],
            "scores": [score for _, score in sorted_nodes],
            "metadata": [
                {"combined_score": score}
                for _, score in sorted_nodes
            ],
            "fusion_stats": {
                "method": "hybrid",
                "strategies_used": list(self.strategies_weights.keys()),
                "total_unique_nodes": len(combined_scores)
            }
        }


class FusionStrategyFactory:
    """Factory per creare strategie di fusion"""
    
    @staticmethod
    def create(
        method: str,
        weights: Dict[str, float] = None
    ) -> FusionStrategy:
        """
        Crea una strategia di fusion
        
        Args:
            method: Nome del metodo di fusion
            weights: Pesi per tipo di query
            
        Returns:
            Istanza della strategia appropriata
            
        Raises:
            ValueError: Se il metodo non è supportato
        """
        method = method.lower()
        
        if method == "reciprocal_rank" or method == "rrf":
            return ReciprocalRankFusion(weights)
        elif method == "weighted_sum":
            return WeightedSumFusion(weights)
        elif method == "max_score":
            return MaxScoreFusion(weights)
        elif method == "hybrid":
            return HybridFusion(weights)
        else:
            raise ValueError(
                f"Unknown fusion method: {method}. "
                f"Supported: reciprocal_rank, weighted_sum, max_score, hybrid"
            )