# -*- coding: utf-8 -*-
"""
Extractors for Keywords and Intent Classification
"""

import logging
import re
from typing import List, Set, Dict
from .config import QueryIntent

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Estrae keywords da query usando spaCy o fallback regex
    """
    
    def __init__(self):
        self._nlp = None
        self._load_spacy()
        
        # Stop words italiane e inglesi
        self.stop_words = {
            # Italiano
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
            'e', 'o', 'ma', 'se', 'che', 'chi', 'cui', 'come', 'quando',
            'dove', 'mentre', 'anche', 'ancora', 'già', 'più', 'molto',
            'poco', 'tanto', 'tutto', 'questo', 'quello',
            'essere', 'avere', 'fare', 'dire', 'andare', 'venire',
            'è', 'sono', 'era', 'erano', 'sarà', 'saranno',
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
            'into', 'through', 'during', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might'
        }
    
    def _load_spacy(self):
        """Carica spaCy se disponibile"""
        try:
            import spacy
            
            # Prova a caricare modello italiano
            try:
                self._nlp = spacy.load("it_core_news_sm")
                logger.debug("Loaded Italian spaCy model")
            except OSError:
                try:
                    # Fallback su modello inglese
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.debug("Loaded English spaCy model")
                except OSError:
                    logger.warning("No spaCy models found")
                    self._nlp = None
        except ImportError:
            logger.warning("spaCy not installed")
            self._nlp = None
    
    def extract(self, query: str) -> List[str]:
        """
        Estrae keywords dalla query
        
        Args:
            query: Query da processare
            
        Returns:
            Lista di keywords estratte
        """
        if self._nlp:
            return self._extract_with_spacy(query)
        else:
            return self._extract_with_regex(query)
    
    def _extract_with_spacy(self, query: str) -> List[str]:
        """
        Estrazione keywords usando spaCy
        """
        doc = self._nlp(query)
        keywords = []
        seen = set()
        
        # 1. Token importanti per POS
        important_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ'}
        auxiliary_verbs = {
            'essere', 'avere', 'fare', 'potere', 'dovere', 'volere',
            'be', 'have', 'do', 'can', 'could', 'would', 'should', 'will'
        }
        
        for token in doc:
            # Skip non importanti
            if token.is_punct or token.is_space or token.is_stop:
                continue
            
            # Skip verbi ausiliari
            if token.pos_ == 'VERB' and token.lemma_.lower() in auxiliary_verbs:
                continue
            
            # Aggiungi se importante
            if token.pos_ in important_pos and len(token.text) > 2:
                keyword = token.lemma_ if token.pos_ == 'VERB' else token.text
                keyword_lower = keyword.lower()
                
                if keyword_lower not in seen and keyword_lower not in self.stop_words:
                    keywords.append(keyword_lower)
                    seen.add(keyword_lower)
        
        # 2. Named Entities
        for ent in doc.ents:
            entity_text = ent.text.lower()
            if entity_text not in seen and len(entity_text) > 2:
                keywords.append(entity_text)
                seen.add(entity_text)
        
        # 3. Noun chunks
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 3:
                chunk_text = self._clean_noun_chunk(chunk.text.lower())
                if chunk_text and chunk_text not in seen and len(chunk_text) > 3:
                    keywords.append(chunk_text)
                    seen.add(chunk_text)
        
        return keywords
    
    def _extract_with_regex(self, query: str) -> List[str]:
        """
        Fallback extraction usando regex
        """
        # Pattern per parole significative
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+(?:[-_][a-zA-ZÀ-ÿ]+)*\b', query.lower())
        
        # Filtra stop words e parole corte
        keywords = [
            w for w in words 
            if w not in self.stop_words and len(w) > 2
        ]
        
        return keywords
    
    def _clean_noun_chunk(self, chunk: str) -> str:
        """Pulisce noun chunks"""
        determiners = {
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
            'the', 'a', 'an', 'this', 'that', 'these', 'those'
        }
        
        words = chunk.split()
        while words and words[0] in determiners:
            words = words[1:]
        
        return ' '.join(words)


class IntentClassifier:
    """
    Classifica l'intent di una query
    """
    
    def __init__(self):
        # Pattern per classificazione intent
        self.patterns = {
            QueryIntent.DEFINITION: [
                r"(?i)^(cos'è|cosa è|che cos'è|cosa sono)",
                r"(?i)^(what is|what are|what's)",
                r"(?i)(definizione|definition|significato|meaning)",
                r"(?i)^(define|spiega cosa)"
            ],
            QueryIntent.EXPLANATION: [
                r"(?i)^(come|in che modo)",
                r"(?i)^(how|why)",
                r"(?i)(funziona|works?|processo|process)",
                r"(?i)(spiega|explain|descrivi|describe)"
            ],
            QueryIntent.COMPARISON: [
                r"(?i)(differenza|difference|confronto|compare)",
                r"(?i)(versus|vs\.?)",
                r"(?i)(meglio|migliore|better|best)",
                r"(?i)(paragone|comparison)"
            ],
            QueryIntent.TUTORIAL: [
                r"(?i)(come fare|how to)",
                r"(?i)(tutorial|guida|guide)",
                r"(?i)(passi|steps|istruzioni|instructions)",
                r"(?i)(configurare|installare|setup|install)"
            ],
            QueryIntent.LIST: [
                r"(?i)(elenca|list|enumera|enumerate)",
                r"(?i)(esempi|examples|tipi|types)",
                r"(?i)(quali sono|what are the)",
                r"(?i)(mostra|show me)"
            ]
        }
    
    def classify(self, query: str) -> str:
        """
        Classifica l'intent della query
        
        Args:
            query: Query da classificare
            
        Returns:
            Intent classificato (stringa)
        """
        query_lower = query.lower()
        
        # Check pattern per ogni intent
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.debug(f"Intent matched: {intent.value} with pattern: {pattern}")
                    return intent.value
        
        # Default
        return QueryIntent.GENERAL.value
    
    def get_intent_confidence(self, query: str) -> Dict[str, float]:
        """
        Calcola confidence per ogni intent
        
        Args:
            query: Query da analizzare
            
        Returns:
            Dizionario intent -> confidence score
        """
        scores = {}
        query_lower = query.lower()
        
        for intent, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            
            # Normalizza per numero di pattern
            scores[intent.value] = score / len(patterns) if patterns else 0.0
        
        # Se nessun pattern matcha, general ha score 1.0
        if all(s == 0.0 for s in scores.values()):
            scores[QueryIntent.GENERAL.value] = 1.0
        
        return scores