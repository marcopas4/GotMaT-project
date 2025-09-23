# -*- coding: utf-8 -*-
"""
Term Expander
Espande termini usando il vector store e co-occurrence analysis
"""

import logging
import re
from typing import List, Dict, Any, Set, Optional
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class TermExpander:
    """
    Espande termini trovando sinonimi e termini correlati nel corpus
    """
    
    def __init__(self, index, embed_model=None):
        """
        Args:
            index: Indice per retrieval
            embed_model: Modello embeddings per similarity
        """
        self.index = index
        self.embed_model = embed_model
        self.expansion_cache = {}
        
        # Dizionario fallback per domini comuni
        self.domain_expansions = self._load_domain_expansions()
        
        logger.info("TermExpander initialized")
    
    def expand(
        self,
        keywords: List[str],
        max_terms: int = 10
    ) -> List[str]:
        """
        Espande lista di keywords trovando termini correlati
        
        Args:
            keywords: Lista di keywords da espandere
            max_terms: Numero massimo di termini espansi
            
        Returns:
            Lista di termini espansi
        """
        expanded_terms = set()
        
        # Se non abbiamo index, usa solo fallback
        if not self.index:
            logger.warning("No index available, using fallback expansion")
            return self._expand_fallback(keywords, max_terms)
        
        # Espandi ogni keyword
        for keyword in keywords[:5]:  # Limita per performance
            if len(keyword) < 3:
                continue
            
            # Check cache
            if keyword in self.expansion_cache:
                expanded_terms.update(self.expansion_cache[keyword])
                continue
            
            # 1. Recupera chunks simili
            similar_chunks = self._retrieve_similar_chunks(keyword, top_k=5)
            
            if similar_chunks:
                # 2. Estrai termini co-occorrenti
                co_occurring = self._extract_cooccurring_terms(
                    similar_chunks,
                    keyword,
                    keywords
                )
                expanded_terms.update(co_occurring)
                
                # 3. Trova semantic neighbors se abbiamo embeddings
                if self.embed_model:
                    neighbors = self._find_semantic_neighbors(keyword, similar_chunks)
                    expanded_terms.update(neighbors)
                
                # Cache risultato
                keyword_expansions = co_occurring + neighbors if self.embed_model else co_occurring
                self.expansion_cache[keyword] = keyword_expansions[:5]
        
        # 4. Aggiungi espansioni di dominio
        domain_terms = self._get_domain_expansions(keywords)
        expanded_terms.update(domain_terms)
        
        # 5. Filtra e ordina
        filtered = self._filter_expanded_terms(list(expanded_terms), keywords, max_terms)
        
        # Se pochi risultati, aggiungi fallback
        if len(filtered) < 3:
            fallback = self._expand_fallback(keywords, max_terms)
            filtered.extend(fallback)
            filtered = list(dict.fromkeys(filtered))[:max_terms]
        
        return filtered
    
    def _retrieve_similar_chunks(self, keyword: str, top_k: int = 5) -> List[Any]:
        """Recupera chunks simili dal vector store"""
        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(keyword)
            return nodes
        except Exception as e:
            logger.debug(f"Failed to retrieve chunks for '{keyword}': {e}")
            return []
    
    def _extract_cooccurring_terms(
        self,
        chunks: List[Any],
        keyword: str,
        all_keywords: List[str],
        window_size: int = 10
    ) -> List[str]:
        """Estrae termini che co-occorrono con il keyword"""
        co_occurring = Counter()
        
        # Pattern per parole significative
        word_pattern = re.compile(r'\b[a-zA-ZÀ-ÿ]{3,}\b')
        
        # Stop words
        stop_words = self._get_stop_words()
        
        for chunk in chunks:
            # Estrai testo dal chunk
            if hasattr(chunk, 'text'):
                text = chunk.text.lower()
            elif hasattr(chunk, 'node') and hasattr(chunk.node, 'text'):
                text = chunk.node.text.lower()
            else:
                continue
            
            words = word_pattern.findall(text)
            
            # Trova posizioni del keyword
            keyword_positions = [
                i for i, w in enumerate(words)
                if w == keyword.lower()
            ]
            
            # Estrai termini nella finestra
            for pos in keyword_positions:
                start = max(0, pos - window_size)
                end = min(len(words), pos + window_size + 1)
                
                for word in words[start:end]:
                    # Filtra
                    if (word != keyword.lower() and
                        word not in stop_words and
                        word not in [kw.lower() for kw in all_keywords] and
                        len(word) > 3):
                        co_occurring[word] += 1
        
        # Ritorna i più frequenti
        min_occurrences = 2 if len(chunks) > 3 else 1
        frequent_terms = [
            term for term, count in co_occurring.most_common(20)
            if count >= min_occurrences
        ]
        
        return frequent_terms[:10]
    
    def _find_semantic_neighbors(
        self,
        keyword: str,
        chunks: List[Any]
    ) -> List[str]:
        """Trova termini semanticamente simili usando embeddings"""
        if not self.embed_model:
            return []
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Embedding del keyword
            keyword_embedding = self.embed_model.get_text_embedding(keyword)
            keyword_embedding = np.array(keyword_embedding).reshape(1, -1)
            
            # Estrai candidati dai chunks
            candidates = self._extract_candidates_from_chunks(chunks)
            
            if not candidates:
                return []
            
            # Calcola embeddings dei candidati
            candidate_list = list(candidates)[:20]
            candidate_embeddings = []
            
            for candidate in candidate_list:
                try:
                    emb = self.embed_model.get_text_embedding(candidate)
                    candidate_embeddings.append(emb)
                except:
                    continue
            
            if not candidate_embeddings:
                return []
            
            # Calcola similarità
            candidate_embeddings = np.array(candidate_embeddings)
            similarities = cosine_similarity(keyword_embedding, candidate_embeddings)[0]
            
            # Filtra per threshold
            threshold = 0.5
            similar_indices = np.where(similarities > threshold)[0]
            
            # Ordina per similarità
            similar_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]]
            
            return [candidate_list[i] for i in similar_indices[:5]]
            
        except Exception as e:
            logger.debug(f"Semantic neighbor extraction failed: {e}")
            return []
    
    def _extract_candidates_from_chunks(self, chunks: List[Any]) -> Set[str]:
        """Estrae termini candidati dai chunks"""
        candidates = set()
        
        # Pattern per termini interessanti
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Proper nouns
            r'\b[a-z]+(?:_[a-z]+)+\b',              # snake_case
            r'\b[a-z]+(?:-[a-z]+)+\b',              # kebab-case
            r'\b[A-Z]{2,}\b',                       # Acronyms
        ]
        
        for chunk in chunks[:3]:  # Top 3 chunks
            if hasattr(chunk, 'text'):
                text = chunk.text
            elif hasattr(chunk, 'node') and hasattr(chunk.node, 'text'):
                text = chunk.node.text
            else:
                continue
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                candidates.update([m.lower() for m in matches if len(m) > 3])
        
        return candidates
    
    def _filter_expanded_terms(
        self,
        terms: List[str],
        original_keywords: List[str],
        max_terms: int
    ) -> List[str]:
        """Filtra e ordina termini espansi"""
        filtered = []
        seen = set()
        
        # Set di keywords originali
        original_set = {kw.lower() for kw in original_keywords}
        
        # Termini generici da escludere
        generic_terms = self._get_generic_terms()
        
        for term in terms:
            term_lower = term.lower().strip()
            
            if (term_lower not in seen and
                term_lower not in original_set and
                term_lower not in generic_terms and
                3 < len(term_lower) < 30 and
                not term_lower.isdigit()):
                
                filtered.append(term_lower)
                seen.add(term_lower)
                
                if len(filtered) >= max_terms:
                    break
        
        return filtered
    
    def _expand_fallback(
        self,
        keywords: List[str],
        max_terms: int
    ) -> List[str]:
        """Espansione fallback usando dizionario predefinito"""
        expanded = []
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            
            # Cerca match esatti
            if kw_lower in self.domain_expansions:
                expanded.extend(self.domain_expansions[kw_lower])
            
            # Cerca match parziali
            for key, values in self.domain_expansions.items():
                if (key in kw_lower or kw_lower in key) and key != kw_lower:
                    expanded.extend(values[:2])
        
        # Deduplica
        seen = set()
        unique = []
        for term in expanded:
            if term.lower() not in seen and term.lower() not in [k.lower() for k in keywords]:
                unique.append(term)
                seen.add(term.lower())
        
        return unique[:max_terms]
    
    def _get_domain_expansions(self, keywords: List[str]) -> List[str]:
        """Ottiene espansioni specifiche del dominio"""
        expansions = []
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            if kw_lower in self.domain_expansions:
                expansions.extend(self.domain_expansions[kw_lower][:3])
        
        return expansions
    
    def _load_domain_expansions(self) -> Dict[str, List[str]]:
        """Carica dizionario di espansioni di dominio"""
        return {
            # AI/ML
            "faiss": ["vector search", "similarity search", "ann", "approximate nearest neighbor"],
            "rag": ["retrieval augmented generation", "retrieval", "generation"],
            "embedding": ["vector representation", "dense vector", "encoding"],
            "llm": ["language model", "large language model", "ai model"],
            
            # Technical
            "database": ["db", "storage", "datastore"],
            "index": ["indexing", "search index"],
            "query": ["search", "question", "request"],
            
            # Italian
            "intelligenza": ["ai", "ia", "artificiale"],
            "ricerca": ["search", "query", "retrieval"],
            "modello": ["model", "algoritmo"],
            
            # Abbreviations
            "ml": ["machine learning", "apprendimento automatico"],
            "ai": ["artificial intelligence", "intelligenza artificiale"],
            "nlp": ["natural language processing", "elaborazione linguaggio"]
        }
    
    def _get_stop_words(self) -> Set[str]:
        """Ritorna set di stop words"""
        return {
            # English
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are',
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'with', 'from', 'for', 'to',
            'of', 'in', 'by', 'about', 'through', 'during', 'before', 'after',
            # Italian
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
            'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
            'e', 'o', 'ma', 'se', 'che', 'chi', 'cui', 'come',
            'è', 'sono', 'era', 'erano'
        }
    
    def _get_generic_terms(self) -> Set[str]:
        """Ritorna termini troppo generici da escludere"""
        return {
            'data', 'information', 'system', 'method', 'approach',
            'technique', 'process', 'model', 'example', 'case',
            'dati', 'informazione', 'sistema', 'metodo', 'approccio',
            'tecnica', 'processo', 'modello', 'esempio', 'caso',
            'thing', 'things', 'stuff', 'way', 'ways'
        }
    
    def clear_cache(self):
        """Pulisce cache espansioni"""
        self.expansion_cache.clear()
        logger.info("Term expansion cache cleared")