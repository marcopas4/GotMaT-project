import hashlib
import logging as logger
from typing import Optional, Dict

class QueryCacheManager:
    """Gestisce caching delle query"""
    
    def __init__(self, enabled: bool = True, max_size: int = 100):
        self.enabled = enabled
        self.max_size = max_size
        self.cache = {} if enabled else None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0
        }
    
    def get(self, query_hash: str) -> Optional[Dict]:
        """Recupera risultato dalla cache"""
        if not self.enabled or not self.cache:
            return None
        
        self.stats["total_queries"] += 1
        
        if query_hash in self.cache:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for hash: {query_hash[:8]}")
            return self.cache[query_hash]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, query_hash: str, result: Dict):
        """Salva risultato in cache"""
        if not self.enabled or not self.cache:
            return
        
        self.cache[query_hash] = result
        
        # Limita dimensione cache (FIFO)
        if len(self.cache) > self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
    
    def clear(self):
        """Pulisce la cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Query cache cleared")
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche cache"""
        if not self.enabled:
            return {"enabled": False}
        
        hit_rate = (self.stats["hits"] / max(1, self.stats["total_queries"])) * 100
        
        return {
            "enabled": True,
            "current_size": len(self.cache) if self.cache else 0,
            "max_size": self.max_size,
            "total_queries": self.stats["total_queries"],
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%"
        }
    
    @staticmethod
    def compute_hash(query: str) -> str:
        """Calcola hash MD5 della query"""
        return hashlib.md5(query.encode()).hexdigest()
