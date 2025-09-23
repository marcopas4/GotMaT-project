# -*- coding: utf-8 -*-
"""
Semantic Variant Generator
Genera varianti semantiche delle query usando LLM o pattern
"""

import logging
import re
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class SemanticVariantGenerator:
    """
    Genera varianti semantiche delle query
    """
    
    def __init__(self, llm=None):
        """
        Args:
            llm: Istanza LLM (Ollama) per generazione avanzata
        """
        self.llm = llm
        self.use_llm = llm is not None
        self.variants_cache = {}
        
        logger.info(f"SemanticVariantGenerator initialized (LLM: {self.use_llm})")
    
    def generate(
        self,
        query: str,
        intent: str = "general",
        max_variants: int = 3
    ) -> List[str]:
        """
        Genera varianti semantiche della query
        
        Args:
            query: Query originale
            intent: Intent classificato
            max_variants: Numero massimo di varianti
            
        Returns:
            Lista di varianti semantiche
        """
        # Check cache
        cache_key = f"{query}_{intent}"
        if cache_key in self.variants_cache:
            logger.debug(f"Using cached variants for: {query[:30]}...")
            return self.variants_cache[cache_key][:max_variants]
        
        variants = []
        
        # Usa LLM se disponibile
        if self.use_llm:
            try:
                variants = self._generate_with_llm(query, intent, max_variants)
                logger.debug(f"Generated {len(variants)} variants with LLM")
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, using fallback")
                variants = self._generate_with_patterns(query, intent, max_variants)
        else:
            variants = self._generate_with_patterns(query, intent, max_variants)
        
        # Cache risultato
        if variants:
            self.variants_cache[cache_key] = variants
        
        return variants[:max_variants]
    
    def _generate_with_llm(
        self,
        query: str,
        intent: str,
        max_variants: int
    ) -> List[str]:
        """
        Genera varianti usando LLM
        """
        # Prepara prompt con few-shot examples
        examples = self._get_examples_for_intent(intent)
        
        prompt = f"""Genera {max_variants} modi alternativi per porre questa domanda, mantenendo esattamente lo stesso significato.

        {examples}

        Query originale: "{query}"

        Genera SOLO le riformulazioni, una per riga, senza numerazione o commenti.
        Le varianti devono essere naturali e grammaticalmente corrette."""
        
        # Chiama LLM
        response = self.llm.complete(prompt)
        
        if hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        
        # Parsa e pulisci risposta
        lines = response_text.strip().split('\n')
        variants = []
        
        for line in lines[:max_variants * 2]:  # Prendi più linee per sicurezza
            cleaned = self._clean_llm_output(line)
            if self._is_valid_variant(cleaned, query):
                variants.append(cleaned)
                if len(variants) >= max_variants:
                    break
        
        return variants
    
    def _generate_with_patterns(
        self,
        query: str,
        intent: str,
        max_variants: int
    ) -> List[str]:
        """
        Genera varianti usando pattern predefiniti
        """
        variants = []
        language = self._detect_language(query)
        
        if intent == "definition":
            variants = self._generate_definition_variants(query, language)
        elif intent == "explanation":
            variants = self._generate_explanation_variants(query, language)
        elif intent == "comparison":
            variants = self._generate_comparison_variants(query, language)
        elif intent == "tutorial":
            variants = self._generate_tutorial_variants(query, language)
        elif intent == "list":
            variants = self._generate_list_variants(query, language)
        else:
            variants = self._generate_general_variants(query, language)
        
        return self._filter_variants(variants, query, max_variants)
    
    def _generate_definition_variants(self, query: str, language: str) -> List[str]:
        """Genera varianti per query di definizione"""
        base = query.replace("?", "").strip()
        variants = []
        
        if language == "it":
            # Rimuovi prefissi comuni
            for prefix in ["cos'è", "cosa è", "che cos'è", "cosa sono"]:
                if prefix in query.lower():
                    base = query.lower().replace(prefix, "").strip()
                    break
            
            variants = [
                f"Definizione di {base}",
                f"Significato di {base}",
                f"{base}: spiegazione",
                f"Cosa si intende per {base}"
            ]
        else:
            # Inglese
            for prefix in ["what is", "what are", "what's"]:
                if prefix in query.lower():
                    base = query.lower().replace(prefix, "").replace("?", "").strip()
                    break
            
            variants = [
                f"Define {base}",
                f"{base} definition",
                f"Meaning of {base}",
                f"Explain {base}"
            ]
        
        return variants
    
    def _generate_explanation_variants(self, query: str, language: str) -> List[str]:
        """Genera varianti per query di spiegazione"""
        variants = []
        
        if language == "it":
            if "come" in query.lower():
                variants = [
                    query.replace("Come", "Spiega come"),
                    query.replace("Come", "In che modo"),
                    query.replace("Come", "Descrivi come")
                ]
        else:
            if "how" in query.lower():
                variants = [
                    query.replace("How", "Explain how"),
                    query.replace("How", "Describe how"),
                    query.replace("How", "What is the process of how")
                ]
        
        return variants
    
    def _generate_comparison_variants(self, query: str, language: str) -> List[str]:
        """Genera varianti per query di confronto"""
        base = query.replace("?", "").strip()
        
        if language == "it":
            return [
                f"Differenze tra {base}",
                f"Confronto tra {base}",
                f"Paragone tra {base}"
            ]
        else:
            return [
                f"Differences between {base}",
                f"Compare and contrast {base}",
                f"Comparison of {base}"
            ]
    
    def _generate_tutorial_variants(self, query: str, language: str) -> List[str]:
        """Genera varianti per query tutorial"""
        base = query.replace("?", "").strip()
        
        if language == "it":
            return [
                f"Guida per {base}",
                f"Tutorial {base}",
                f"Istruzioni per {base}"
            ]
        else:
            return [
                f"Step-by-step guide to {base}",
                f"Tutorial for {base}",
                f"Instructions for {base}"
            ]
    
    def _generate_list_variants(self, query: str, language: str) -> List[str]:
        """Genera varianti per query di lista"""
        variants = []
        
        if language == "it":
            if "esempi" in query.lower():
                base = query.replace("esempi di", "").replace("?", "").strip()
                variants = [
                    f"Lista di {base}",
                    f"Quali sono i {base}",
                    f"Elenca {base}"
                ]
        else:
            if "examples" in query.lower():
                base = query.replace("examples of", "").replace("?", "").strip()
                variants = [
                    f"List of {base}",
                    f"What are some {base}",
                    f"Enumerate {base}"
                ]
        
        return variants
    
    def _generate_general_variants(self, query: str, language: str) -> List[str]:
        """Genera varianti generiche"""
        # Semplici riformulazioni
        variants = []
        
        if "?" in query:
            statement = query.replace("?", "").strip()
            if language == "it":
                variants.append(f"Informazioni su {statement}")
                variants.append(f"Dettagli riguardo {statement}")
            else:
                variants.append(f"Information about {statement}")
                variants.append(f"Details regarding {statement}")
        
        return variants
    
    def _get_examples_for_intent(self, intent: str) -> str:
        """Ritorna esempi few-shot per l'intent"""
        examples = {
            "definition": """Esempi:
Query: "Cos'è Python?"
Varianti:
- Definizione di Python
- Python linguaggio di programmazione spiegazione
- Cosa si intende per Python""",
            
            "explanation": """Esempi:
Query: "Come funziona FAISS?"
Varianti:
- Spiega il funzionamento di FAISS
- FAISS processo e meccanismo
- Descrizione del funzionamento di FAISS""",
            
            "comparison": """Esempi:
Query: "Differenza tra Python e Java"
Varianti:
- Confronto Python vs Java
- Come si differenziano Python e Java
- Paragone tra i linguaggi Python e Java""",
            
            "tutorial": """Esempi:
Query: "Come installare Python?"
Varianti:
- Guida installazione Python
- Tutorial per installare Python
- Istruzioni passo-passo installazione Python""",
            
            "list": """Esempi:
Query: "Esempi di database NoSQL"
Varianti:
- Lista database NoSQL
- Quali sono i database NoSQL
- Elenco dei principali database NoSQL""",
            
            "general": """Esempi:
Query: "Machine learning applicazioni"
Varianti:
- Usi del machine learning
- Dove si applica il machine learning
- Applicazioni pratiche del machine learning"""
        }
        
        return examples.get(intent, examples["general"])
    
    def _clean_llm_output(self, line: str) -> str:
        """Pulisce output LLM"""
        # Rimuovi numerazione
        line = re.sub(r'^[\d\-\*•\.]+\s*', '', line)
        
        # Rimuovi quotes
        line = line.strip('"\'')
        
        # Rimuovi prefissi comuni
        prefixes = ['Variante:', 'Query:', 'Domanda:', '-']
        for prefix in prefixes:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        
        return line.strip()
    
    def _is_valid_variant(self, variant: str, original: str) -> bool:
        """Valida una variante"""
        if not variant or len(variant) < 5:
            return False
        
        # Non identica all'originale
        if variant.lower() == original.lower():
            return False
        
        # Non troppo lunga
        if len(variant) > len(original) * 3:
            return False
        
        # Calcola similarità Jaccard
        variant_words = set(variant.lower().split())
        original_words = set(original.lower().split())
        
        if not original_words:
            return True
        
        jaccard = len(variant_words & original_words) / len(variant_words | original_words)
        
        # Deve essere simile ma non identica (0.2 < sim < 0.9)
        return 0.2 < jaccard < 0.9
    
    def _filter_variants(
        self,
        variants: List[str],
        original: str,
        max_variants: int
    ) -> List[str]:
        """Filtra e deduplica varianti"""
        filtered = []
        seen = set()
        
        for variant in variants:
            variant = variant.strip()
            variant_lower = variant.lower()
            
            if (variant_lower not in seen and 
                variant_lower != original.lower() and
                variant):
                filtered.append(variant)
                seen.add(variant_lower)
                
                if len(filtered) >= max_variants:
                    break
        
        return filtered
    
    def _detect_language(self, query: str) -> str:
        """Rileva lingua della query"""
        italian_words = ['cosa', 'come', 'quando', 'dove', 'chi', 'è', 'sono']
        query_lower = query.lower()
        
        italian_count = sum(1 for word in italian_words if word in query_lower)
        return "it" if italian_count >= 2 else "en"
    
    def clear_cache(self):
        """Pulisce cache varianti"""
        self.variants_cache.clear()
        logger.info("Variants cache cleared")