# -*- coding: utf-8 -*-
"""
Query Expansion Module
Gestisce l'espansione delle query con multiple strategie
"""

import logging
from typing import List, Dict, Any, Optional

from .config import QueryExpansion, QueryIntent
from .extractors import KeywordExtractor, IntentClassifier
from .variants import SemanticVariantGenerator
from .terms import TermExpander

logger = logging.getLogger(__name__)


class QueryExpansionModule:
    """
    Modulo principale per l'espansione delle query
    Coordina tutti i componenti di espansione
    """
    
    def __init__(
        self,
        index,
        embed_model=None,
        llm=None
    ):
        """
        Args:
            index: Indice per retrieval
            embed_model: Modello embeddings
            llm: LLM per generazione varianti
        """
        self.index = index
        
        # Inizializza componenti
        self.keyword_extractor = KeywordExtractor()
        self.intent_classifier = IntentClassifier()
        self.variant_generator = SemanticVariantGenerator(llm)
        self.term_expander = TermExpander(index, embed_model)
        
        # Stats
        self.stats = {
            "total_expansions": 0
        }
        
        logger.info("QueryExpansionModule initialized")
    
    def expand(self, query: str) -> QueryExpansion:
        """
        Genera tutte le espansioni per una query
        
        Args:
            query: Query originale
            
        Returns:
            QueryExpansion con tutti i tipi di espansione
        """
        self.stats["total_expansions"] += 1
        
        # Crea oggetto espansione
        expansion = QueryExpansion(original=query)
        
        # 1. Classifica intent
        expansion.intent = self.intent_classifier.classify(query)
        logger.debug(f"Query intent: {expansion.intent}")
        
        # 2. Estrai keywords
        expansion.keywords = self.keyword_extractor.extract(query)
        logger.debug(f"Extracted {len(expansion.keywords)} keywords")
        
        # 3. Genera varianti semantiche basate su intent
        expansion.semantic_variants = self.variant_generator.generate(
            query,
            intent=expansion.intent
        )
        logger.debug(f"Generated {len(expansion.semantic_variants)} semantic variants")
        
        # 4. Decomponi query complesse
        if self._is_complex_query(query):
            expansion.sub_queries = self._decompose_query(query)
            logger.debug(f"Decomposed into {len(expansion.sub_queries)} sub-queries")
        
        # 5. Espandi termini usando il corpus
        if expansion.keywords:
            expansion.expanded_terms = self.term_expander.expand(
                expansion.keywords
            )
            logger.debug(f"Expanded to {len(expansion.expanded_terms)} additional terms")
        
        # 6. Rileva lingua
        expansion.language = self._detect_language(query)
        
        return expansion
    
    def _is_complex_query(self, query: str) -> bool:
        """Determina se una query è complessa e necessita decomposizione"""
        conjunctions = ['e', 'o', 'ma', 'inoltre', 'oppure', 'and', 'or', 'but', 'also']
        
        has_conjunctions = any(f" {conj} " in query.lower() for conj in conjunctions)
        is_long = len(query.split()) > 12
        has_multiple_questions = query.count('?') > 1
        has_multiple_clauses = query.count(',') > 1
        
        return has_conjunctions or is_long or has_multiple_questions or has_multiple_clauses
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompone query complesse in sub-queries"""
        import re
        
        sub_queries = []
        
        # Pattern per split su congiunzioni
        conjunction_pattern = r'\s+(?:e|o|ma|inoltre|oppure|and|or|but|also)\s+'
        parts = re.split(conjunction_pattern, query, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            
            if len(part) < 10:
                continue
            
            if part.lower() in ['e', 'o', 'ma', 'inoltre']:
                continue
            
            if self._is_question(part) and '?' not in part:
                part += '?'
            
            sub_queries.append(part)
        
        # Gestisci domande multiple
        if '?' in query:
            questions = query.split('?')
            for q in questions:
                q = q.strip()
                if len(q) > 5 and q not in sub_queries:
                    sub_queries.append(q + '?')
        
        # Deduplica
        seen = set()
        unique_subs = []
        for sub in sub_queries:
            sub_lower = sub.lower()
            if sub_lower not in seen and sub_lower != query.lower():
                unique_subs.append(sub)
                seen.add(sub_lower)
        
        return unique_subs[:3]
    
    def _is_question(self, text: str) -> bool:
        """Determina se un testo è una domanda"""
        question_words = [
            'cosa', 'come', 'quando', 'dove', 'chi', 'quale', 'quanto',
            'perché', 'what', 'how', 'when', 'where', 'who', 'which',
            'why', 'is', 'are', 'can', 'could', 'would'
        ]
        
        text_lower = text.lower()
        return any(text_lower.startswith(qw) for qw in question_words)
    
    def _detect_language(self, query: str) -> str:
        """Rileva la lingua della query"""
        italian_indicators = [
            'cosa', 'come', 'quando', 'dove', 'chi', 'quale',
            'è', 'sono', 'questo', 'quello', 'il', 'la', 'i', 'le',
            'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra'
        ]
        
        query_lower = query.lower()
        italian_score = sum(1 for word in italian_indicators if word in query_lower)
        
        return "it" if italian_score >= 2 else "en"
    
    def get_stats(self) -> Dict[str, Any]:
        """Ritorna statistiche del modulo"""
        return {
            "total_expansions": self.stats["total_expansions"],
            "components": {
                "keyword_extractor": "active",
                "intent_classifier": "active",
                "variant_generator": "active" if self.variant_generator.llm else "pattern-based",
                "term_expander": "active"
            }
        }
