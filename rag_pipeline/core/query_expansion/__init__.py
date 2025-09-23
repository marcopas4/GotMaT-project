# -*- coding: utf-8 -*-
"""
Query Processor Package
Multi-query retrieval with fusion for improved RAG performance
"""

from .config import (
    QueryConfig,
    QueryExpansion,
    RetrievalResult,
    FusionResult,
    FusionMethod,
    QueryIntent
)

from .base import QueryProcessor

from .fusion import (
    FusionStrategy,
    ReciprocalRankFusion,
    WeightedSumFusion,
    MaxScoreFusion,
    HybridFusion,
    FusionStrategyFactory
)

from .cache import CacheManager
from .expansion import QueryExpansionModule
from .extractors import KeywordExtractor, IntentClassifier
from .variants import SemanticVariantGenerator
from .terms import TermExpander

__version__ = "2.0.0"

__all__ = [
    # Main class
    'QueryProcessor',
    
    # Config classes
    'QueryConfig',
    'QueryExpansion',
    'RetrievalResult',
    'FusionResult',
    'FusionMethod',
    'QueryIntent',
    
    # Fusion strategies
    'FusionStrategy',
    'ReciprocalRankFusion',
    'WeightedSumFusion',
    'MaxScoreFusion',
    'HybridFusion',
    'FusionStrategyFactory',
    
    # Components
    'CacheManager',
    'QueryExpansionModule',
    'KeywordExtractor',
    'IntentClassifier',
    'SemanticVariantGenerator',
    'TermExpander'
]