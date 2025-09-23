# -*- coding: utf-8 -*-
"""
Cache Manager for Query Processor
Gestisce caching con TTL e statistiche
"""

import logging
import time
from typing import Any, Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gestisce cache con supporto TTL e LRU eviction
    """
    
    def __init__(
        self,
        enabled: bool = True,
        max_size: int = 100,
        default_ttl: int = 3600
    ):
        """
        Args:
            enabled: Se la cache è abilitata
            max_size: Numero massimo di elementi in cache
            default_ttl: Time-to-live di default in secondi
        """
        self.enabled = enabled
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Usa OrderedDict per LRU
        self.cache = OrderedDict() if enabled else None
        
        # Statistiche
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
            "total_queries": 0
        }
        
        logger.info(
            f"CacheManager initialized "
            f"(enabled={enabled}, max_size={max_size}, ttl={default_ttl}s)"
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera valore dalla cache
        
        Args:
            key: Chiave cache
            
        Returns:
            Valore cached o None se non trovato/scaduto
        """
        if not self.enabled or not self.cache:
            return None
        
        self.stats["total_queries"] += 1
        
        if key in self.cache:
            # Check TTL
            entry = self.cache[key]
            if self._is_expired(entry):
                # Entry scaduta
                del self.cache[key]
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                logger.debug(f"Cache expired for key: {key[:30]}...")
                return None
            
            # Hit - sposta in fondo (most recently used)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for key: {key[:30]}...")
            return entry["value"]
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Salva valore in cache
        
        Args:
            key: Chiave cache
            value: Valore da salvare
            ttl: Time-to-live in secondi (None = usa default)
        """
        if not self.enabled or not self.cache:
            return
        
        # Prepara entry con timestamp
        entry = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
        
        # Se key già esiste, aggiorna e sposta in fondo
        if key in self.cache:
            self.cache[key] = entry
            self.cache.move_to_end(key)
        else:
            # Nuova entry
            self.cache[key] = entry
            
            # Check dimensione massima
            if len(self.cache) > self.max_size:
                # Rimuovi least recently used (primo elemento)
                evicted = self.cache.popitem(last=False)
                self.stats["evictions"] += 1
                logger.debug(f"Cache eviction: {evicted[0][:30]}...")
    
    def _is_expired(self, entry: Dict) -> bool:
        """
        Verifica se una entry è scaduta
        
        Args:
            entry: Entry cache con timestamp e ttl
            
        Returns:
            True se scaduta
        """
        elapsed = time.time() - entry["timestamp"]
        return elapsed > entry["ttl"]
    
    def clear(self):
        """
        Pulisce completamente la cache
        """
        if self.cache:
            size = len(self.cache)
            self.cache.clear()
            logger.info(f"Cache cleared ({size} entries removed)")
    
    def cleanup_expired(self):
        """
        Rimuove tutte le entry scadute
        """
        if not self.cache:
            return
        
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.stats["expired"] += 1
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Ritorna statistiche della cache
        
        Returns:
            Dizionario con statistiche dettagliate
        """
        if not self.enabled:
            return {"enabled": False}
        
        total = self.stats["total_queries"]
        hit_rate = (self.stats["hits"] / max(1, total)) * 100 if total > 0 else 0
        
        return {
            "enabled": True,
            "current_size": len(self.cache) if self.cache else 0,
            "max_size": self.max_size,
            "total_queries": total,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "evictions": self.stats["evictions"],
            "expired": self.stats["expired"]
        }
    
    def get_size_info(self) -> Dict[str, Any]:
        """
        Ritorna informazioni sulla dimensione della cache
        
        Returns:
            Info su utilizzo memoria approssimativo
        """
        if not self.cache:
            return {"size_bytes": 0, "entries": 0}
        
        # Stima approssimativa della dimensione
        import sys
        
        total_size = 0
        for key, entry in self.cache.items():
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(entry)
        
        return {
            "entries": len(self.cache),
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "avg_entry_size": total_size // max(1, len(self.cache))
        }