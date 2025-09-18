"""
Query Expansion Avanzata con Multiple Retrieval Strategies
Implementa fusion di risultati da query multiple invece di concatenazione
"""

import logging
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Risultato di una singola query di retrieval"""
    query: str
    query_type: str  # "original", "expanded_term", "semantic_variant"
    nodes: List[Any]
    scores: List[float]
    
    
class QueryProcessor:
    """
    Query processor con strategie di retrieval multiple e fusion avanzata
    invece di semplice concatenazione di termini
    """
    
    def __init__(self, embed_model, vector_store=None, index=None):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.index = index
        self.retrieval_cache = {}
        
    def enhanced_retrieval(
        self, 
        query: str, 
        top_k: int = 10,
        expansion_strategy: str = "multi_query",  # "concatenation" vs "multi_query"
        fusion_method: str = "reciprocal_rank"  # "max_score", "reciprocal_rank", "weighted_sum"
    ) -> Dict[str, Any]:
        """
        Esegue retrieval avanzato con multiple strategie
        
        Args:
            query: Query originale
            top_k: Numero di documenti da recuperare
            expansion_strategy: Strategia di espansione
            fusion_method: Metodo di fusione risultati
        """
        
        # 1. Process query per ottenere espansioni
        query_data = self._process_query_advanced(query)
        
        if expansion_strategy == "concatenation":
            # Metodo VECCHIO: concatena tutto
            return self._retrieval_concatenation(query, query_data, top_k)
        else:
            # Metodo NUOVO: multiple queries + fusion
            return self._retrieval_multi_query(query, query_data, top_k, fusion_method)
    
    def _process_query_advanced(self, query: str) -> Dict[str, Any]:
        """Processa query con analisi avanzata"""
        return {
            "original": query,
            "expanded_terms": self._expand_query_smart(query),
            "semantic_variants": self._generate_semantic_variants(query),
            "sub_queries": self._decompose_query(query),
            "intent": self._classify_intent(query),
            "keywords": self._extract_keywords(query)
        }
    
    def _retrieval_concatenation(
        self, 
        query: str, 
        query_data: Dict, 
        top_k: int
    ) -> Dict[str, Any]:
        """
        METODO VECCHIO: Concatenazione diretta (problematico)
        """
        # Concatena tutto insieme
        expanded_query = f"{query} {' '.join(query_data['expanded_terms'][:5])}"
        
        # Single retrieval con query espansa
        results = self._single_retrieval(expanded_query, top_k)
        
        return {
            "method": "concatenation",
            "final_query": expanded_query,
            "results": results,
            "num_queries": 1
        }
    
    def _retrieval_multi_query(
        self, 
        query: str, 
        query_data: Dict,
        top_k: int,
        fusion_method: str
    ) -> Dict[str, Any]:
        """
        METODO NUOVO: Multiple queries indipendenti + fusion
        """
        all_retrievals = []
        
        # 1. Query originale (peso maggiore)
        logger.info(f"Retrieving with original query: {query}")
        original_results = self._single_retrieval(query, top_k * 2)  # Più risultati
        all_retrievals.append(RetrievalResult(
            query=query,
            query_type="original",
            nodes=original_results["nodes"],
            scores=original_results["scores"]
        ))
        
        # 2. Top expanded terms individuali
        for term in query_data["expanded_terms"][:3]:  # Solo top 3 termini
            if len(term) > 3:  # Evita termini troppo corti
                logger.info(f"Retrieving with expanded term: {term}")
                term_results = self._single_retrieval(term, top_k)
                all_retrievals.append(RetrievalResult(
                    query=term,
                    query_type="expanded_term",
                    nodes=term_results["nodes"],
                    scores=term_results["scores"]
                ))
        
        # 3. Semantic variants (query riformulate)
        for variant in query_data["semantic_variants"][:2]:
            logger.info(f"Retrieving with semantic variant: {variant}")
            variant_results = self._single_retrieval(variant, top_k)
            all_retrievals.append(RetrievalResult(
                query=variant,
                query_type="semantic_variant",
                nodes=variant_results["nodes"],
                scores=variant_results["scores"]
            ))
        
        # 4. Sub-queries se la query è complessa
        if query_data["sub_queries"]:
            for sub_query in query_data["sub_queries"][:2]:
                logger.info(f"Retrieving with sub-query: {sub_query}")
                sub_results = self._single_retrieval(sub_query, top_k)
                all_retrievals.append(RetrievalResult(
                    query=sub_query,
                    query_type="sub_query",
                    nodes=sub_results["nodes"],
                    scores=sub_results["scores"]
                ))
        
        # 5. FUSION dei risultati
        fused_results = self._fuse_results(all_retrievals, fusion_method, top_k)
        
        return {
            "method": "multi_query_fusion",
            "fusion_method": fusion_method,
            "num_queries": len(all_retrievals),
            "queries_executed": [r.query for r in all_retrievals],
            "results": fused_results
        }
    
    def _single_retrieval(self, query: str, top_k: int) -> Dict[str, Any]:
        """Esegue singola query di retrieval"""
        # Check cache
        cache_key = f"{query}_{top_k}"
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        # Retrieve usando l'index
        if self.index:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            
            result = {
                "nodes": nodes,
                "scores": [n.score for n in nodes]
            }
            
            # Cache result
            self.retrieval_cache[cache_key] = result
            return result
        
        return {"nodes": [], "scores": []}
    
    def _fuse_results(
        self, 
        all_retrievals: List[RetrievalResult],
        fusion_method: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Fusione intelligente dei risultati da multiple queries
        """
        if fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(all_retrievals, top_k)
        elif fusion_method == "weighted_sum":
            return self._weighted_sum_fusion(all_retrievals, top_k)
        else:  # max_score
            return self._max_score_fusion(all_retrievals, top_k)
    
    def _reciprocal_rank_fusion(
        self, 
        all_retrievals: List[RetrievalResult],
        top_k: int,
        k: int = 60
    ) -> Dict[str, Any]:
        """
        Reciprocal Rank Fusion (RRF) - Metodo più robusto
        Combina ranking da multiple queries
        """
        # Dizionario per accumulare scores
        node_scores = defaultdict(float)
        node_objects = {}
        node_metadata = defaultdict(lambda: {"sources": [], "ranks": []})
        
        # Weight per tipo di query
        query_weights = {
            "original": 2.0,        # Query originale pesa di più
            "semantic_variant": 1.5,
            "sub_query": 1.2,
            "expanded_term": 0.8    # Termini espansi pesano meno
        }
        
        for retrieval in all_retrievals:
            weight = query_weights.get(retrieval.query_type, 1.0)
            
            for rank, node in enumerate(retrieval.nodes, start=1):
                # Genera ID univoco per il nodo
                node_id = self._get_node_id(node)
                
                # Reciprocal Rank Fusion formula
                score = weight * (1.0 / (k + rank))
                node_scores[node_id] += score
                
                # Salva oggetto nodo
                if node_id not in node_objects:
                    node_objects[node_id] = node
                
                # Track metadata
                node_metadata[node_id]["sources"].append(retrieval.query)
                node_metadata[node_id]["ranks"].append(rank)
        
        # Ordina per score finale
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepara risultati finali
        final_nodes = []
        final_scores = []
        final_metadata = []
        
        for node_id, score in sorted_nodes[:top_k]:
            final_nodes.append(node_objects[node_id])
            final_scores.append(score)
            final_metadata.append({
                "fusion_score": score,
                "appeared_in": node_metadata[node_id]["sources"],
                "ranks": node_metadata[node_id]["ranks"],
                "avg_rank": np.mean(node_metadata[node_id]["ranks"])
            })
        
        return {
            "nodes": final_nodes,
            "scores": final_scores,
            "metadata": final_metadata,
            "fusion_details": {
                "total_unique_nodes": len(node_scores),
                "fusion_method": "reciprocal_rank",
                "k_parameter": k
            }
        }
    
    def _weighted_sum_fusion(
        self, 
        all_retrievals: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Weighted Sum Fusion - Somma pesata degli scores
        """
        node_scores = defaultdict(float)
        node_objects = {}
        node_counts = defaultdict(int)
        
        # Pesi per tipo di query
        query_weights = {
            "original": 3.0,
            "semantic_variant": 2.0,
            "sub_query": 1.5,
            "expanded_term": 1.0
        }
        
        for retrieval in all_retrievals:
            weight = query_weights.get(retrieval.query_type, 1.0)
            
            for node, score in zip(retrieval.nodes, retrieval.scores):
                node_id = self._get_node_id(node)
                
                # Accumula score pesato
                node_scores[node_id] += weight * score
                node_counts[node_id] += 1
                
                if node_id not in node_objects:
                    node_objects[node_id] = node
        
        # Normalizza per numero di apparizioni (opzionale)
        for node_id in node_scores:
            # Bonus per nodi che appaiono in multiple queries
            appearance_bonus = 1.0 + (0.1 * (node_counts[node_id] - 1))
            node_scores[node_id] *= appearance_bonus
        
        # Ordina e ritorna top-k
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_nodes = []
        final_scores = []
        
        for node_id, score in sorted_nodes[:top_k]:
            final_nodes.append(node_objects[node_id])
            final_scores.append(score)
        
        return {
            "nodes": final_nodes,
            "scores": final_scores,
            "fusion_details": {
                "method": "weighted_sum",
                "total_unique_nodes": len(node_scores)
            }
        }
    
    def _max_score_fusion(
        self, 
        all_retrievals: List[RetrievalResult],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Max Score Fusion - Prende il punteggio massimo per ogni documento
        """
        node_max_scores = {}
        node_objects = {}
        
        for retrieval in all_retrievals:
            for node, score in zip(retrieval.nodes, retrieval.scores):
                node_id = self._get_node_id(node)
                
                # Mantieni solo il punteggio massimo
                if node_id not in node_max_scores or score > node_max_scores[node_id]:
                    node_max_scores[node_id] = score
                    node_objects[node_id] = node
        
        # Ordina per score massimo
        sorted_nodes = sorted(node_max_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_nodes = []
        final_scores = []
        
        for node_id, score in sorted_nodes[:top_k]:
            final_nodes.append(node_objects[node_id])
            final_scores.append(score)
        
        return {
            "nodes": final_nodes,
            "scores": final_scores,
            "fusion_details": {
                "method": "max_score",
                "total_unique_nodes": len(node_max_scores)
            }
        }
    
    def _get_node_id(self, node) -> str:
        """Genera ID univoco per un nodo"""
        if hasattr(node, 'node_id'):
            return node.node_id
        elif hasattr(node, 'text'):
            return hashlib.md5(node.text.encode()).hexdigest()
        else:
            return str(id(node))
    
    def _expand_query_smart(self, query: str) -> List[str]:
        """
        Espansione intelligente che genera termini correlati
        invece di semplici varianti morfologiche
        """
        expanded = []
        
        # 1. Estrai concetti principali dalla query
        concepts = self._extract_key_concepts(query)
        
        # 2. Per ogni concetto, trova termini correlati nel corpus
        for concept in concepts:
            # Semantic neighbors dal vector space
            neighbors = self._find_semantic_neighbors(concept, k=5)
            expanded.extend(neighbors)
            
            # Domain-specific expansions
            domain_terms = self._get_domain_expansions(concept)
            expanded.extend(domain_terms)
        
        # 3. Rimuovi duplicati e termini troppo generici
        expanded = list(set(expanded))
        expanded = [term for term in expanded if len(term) > 3 and term.lower() not in query.lower()]
        
        return expanded[:10]  # Limita a 10 termini
    
    def _generate_semantic_variants(self, query: str) -> List[str]:
        """
        Genera varianti semantiche della query
        (query alternative che esprimono lo stesso concetto)
        """
        variants = []
        
        # Pattern-based reformulation
        if query.startswith("What is"):
            variants.append(query.replace("What is", "Define"))
            variants.append(query.replace("What is", "Explain"))
        elif query.startswith("How"):
            variants.append(query.replace("How", "What is the process to"))
            variants.append(query.replace("How", "Steps to"))
        
        # Intent-based reformulation
        intent = self._classify_intent(query)
        if intent == "definition":
            base = query.replace("?", "").strip()
            variants.extend([
                f"Definition of {base}",
                f"Meaning of {base}",
                f"{base} explanation"
            ])
        elif intent == "comparison":
            variants.append(f"Difference between {query}")
            variants.append(f"Compare and contrast {query}")
        
        return variants[:3]  # Max 3 varianti
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Estrae concetti chiave dalla query"""
        # Implementazione semplificata
        import re
        
        # Rimuovi stop words e parole comuni
        stop_words = {'what', 'is', 'the', 'how', 'to', 'a', 'an', 'and', 'or', 'but'}
        words = query.lower().split()
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Estrai anche bigrammi significativi
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i] not in stop_words or words[i+1] not in stop_words:
                if len(bigram) > 5:
                    concepts.append(bigram)
        
        return concepts[:5]
    
    def _find_semantic_neighbors(self, term: str, k: int = 5) -> List[str]:
        """
        Trova termini semanticamente vicini nel vector space
        """
        # Questa è una versione semplificata
        # In produzione, useresti il vector store per trovare termini simili
        
        # Esempio di mapping semantico predefinito
        semantic_map = {
            "faiss": ["vector search", "similarity", "indexing", "embedding"],
            "rag": ["retrieval", "generation", "llm", "context"],
            "embedding": ["vector", "representation", "encoding", "features"],
            "llm": ["language model", "gpt", "transformer", "generation"]
        }
        
        term_lower = term.lower()
        for key, values in semantic_map.items():
            if key in term_lower or term_lower in key:
                return values[:k]
        
        return []
    
    def _get_domain_expansions(self, concept: str) -> List[str]:
        """Espansioni specifiche del dominio"""
        # Mappings domain-specific
        domain_expansions = {
            "machine learning": ["ml", "artificial intelligence", "ai", "deep learning"],
            "retrieval": ["search", "query", "lookup", "fetch"],
            "vector": ["embedding", "representation", "features", "tensor"],
            "database": ["storage", "index", "query", "persistence"]
        }
        
        concept_lower = concept.lower()
        for key, expansions in domain_expansions.items():
            if key in concept_lower or concept_lower in key:
                return expansions[:3]
        
        return []
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompone query complesse in sub-queries"""
        # Splitting su congiunzioni
        import re
        
        # Pattern per identificare query composte
        connectors = r'\s+(and|also|as well as|inoltre|e anche)\s+'
        sub_queries = re.split(connectors, query, flags=re.IGNORECASE)
        
        # Filtra e pulisci
        sub_queries = [q.strip() for q in sub_queries 
                      if len(q.strip()) > 10 and q.strip().lower() not in ['and', 'also', 'inoltre']]
        
        # Se la query contiene domande multiple
        if '?' in query:
            questions = query.split('?')
            sub_queries.extend([q.strip() + '?' for q in questions if len(q.strip()) > 5])
        
        return list(set(sub_queries))[:3]  # Max 3 sub-queries
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Estrae keywords principali dalla query"""
        import re
        
        # Tokenizzazione base
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Rimuovi stop words
        stop_words = {'what', 'is', 'the', 'how', 'to', 'a', 'an', 'and', 'or', 'but',
                     'in', 'on', 'at', 'for', 'with', 'about', 'as', 'by', 'of'}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _classify_intent(self, query: str) -> str:
        """Classifica l'intent della query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition', "cos'è"]):
            return "definition"
        elif any(word in query_lower for word in ['how', 'come', 'steps', 'process']):
            return "explanation"
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return "comparison"
        elif any(word in query_lower for word in ['list', 'enumerate', 'elenca', 'examples']):
            return "list"
        else:
            return "general"


# ============================================================================
# INTEGRAZIONE CON LA PIPELINE PRINCIPALE
# ============================================================================

class EnhancedRAGPipeline:
    """
    Estensione della pipeline principale con retrieval migliorato
    """
    
    def __init__(self, config):
        self.config = config
        self.query_processor = None
        # ... altri componenti
    
    def query_with_advanced_retrieval(
        self,
        question: str,
        retrieval_strategy: str = "multi_query",  # o "concatenation" per confronto
        fusion_method: str = "reciprocal_rank",
        compare_methods: bool = False
    ) -> Dict[str, Any]:
        """
        Query con retrieval avanzato
        
        Args:
            question: Domanda utente
            retrieval_strategy: "multi_query" o "concatenation"
            fusion_method: Metodo di fusion per multi_query
            compare_methods: Se True, confronta entrambi i metodi
        """
        
        if compare_methods:
            # Esegui entrambi i metodi per confronto
            results = {}
            
            # Metodo vecchio (concatenazione)
            concat_result = self.query_processor.enhanced_retrieval(
                question,
                expansion_strategy="concatenation"
            )
            results["concatenation"] = concat_result
            
            # Metodo nuovo (multi-query fusion)
            fusion_result = self.query_processor.enhanced_retrieval(
                question,
                expansion_strategy="multi_query",
                fusion_method=fusion_method
            )
            results["multi_query"] = fusion_result
            
            # Confronta metriche
            results["comparison"] = self._compare_retrieval_methods(
                concat_result, 
                fusion_result
            )
            
            return results
        else:
            # Esegui solo il metodo richiesto
            return self.query_processor.enhanced_retrieval(
                question,
                expansion_strategy=retrieval_strategy,
                fusion_method=fusion_method
            )
    
    def _compare_retrieval_methods(
        self, 
        concat_result: Dict,
        fusion_result: Dict
    ) -> Dict[str, Any]:
        """Confronta i risultati dei due metodi"""
        comparison = {
            "num_queries": {
                "concatenation": concat_result["num_queries"],
                "multi_query": fusion_result["num_queries"]
            },
            "unique_nodes": {
                "concatenation": len(set([str(n) for n in concat_result["results"].get("nodes", [])])),
                "multi_query": fusion_result["results"]["fusion_details"]["total_unique_nodes"]
            },
            "top_scores": {
                "concatenation": concat_result["results"].get("scores", [])[:5],
                "multi_query": fusion_result["results"]["scores"][:5]
            }
        }
        
        # Calcola overlap dei top-k risultati
        if concat_result["results"].get("nodes") and fusion_result["results"]["nodes"]:
            concat_ids = [self._get_node_id(n) for n in concat_result["results"]["nodes"][:10]]
            fusion_ids = [self._get_node_id(n) for n in fusion_result["results"]["nodes"][:10]]
            
            overlap = len(set(concat_ids) & set(fusion_ids))
            comparison["top_10_overlap"] = f"{overlap}/10 nodes in common"
        
        return comparison


