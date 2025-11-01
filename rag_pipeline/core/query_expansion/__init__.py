# -*- coding: utf-8 -*-
"""
Query Processor Package
Query expansion for improved RAG performance
"""

from .config import (
    QueryConfig,
    QueryExpansion,
    QueryIntent
)

from .base import QueryProcessor

__version__ = "2.0.0"

__all__ = [
    # Main class
    'QueryProcessor',
    
    # Config classes
    'QueryConfig',
    'QueryExpansion',
    'QueryIntent'
]