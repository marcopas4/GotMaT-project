import json
import logging as logger
from pathlib import Path
from typing import Dict, Any
from config import RAGConfig
class EvaluationManager:
    """Gestisce valutazione e logging dei risultati"""

    def __init__(self, output_dir: str = RAGConfig.evaluation_path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.csv_path = self.output_dir / "rag_evaluation.csv"
        self.metrics_path = self.output_dir / "metrics.json"
    
    def save_query_result(self, result: Dict[str, Any]):
        """Salva risultato query per valutazione"""
        import csv
        from datetime import datetime
        
        row = {
            "timestamp": datetime.now().isoformat(),
            "question": result["question"],
            "answer": result["answer"][:500],
            "response_time": result.get("response_time", 0),
            "from_cache": result.get("from_cache", False),
            "num_sources": result.get("num_sources", 0),
            "model": result.get("model", ""),
            "embedding_model": result.get("embedding_model", ""),
            "intent": result.get("query_metadata", {}).get("intent", ""),
            "query_type": result.get("query_metadata", {}).get("query_type", "")
        }
        
        # Aggiungi top scores se presenti
        if "sources" in result and result["sources"]:
            top_scores = [s.get("score", 0) for s in result["sources"][:3]]
            row["top_3_scores"] = ", ".join([f"{s:.3f}" for s in top_scores])
        
        # Scrivi CSV
        file_exists = self.csv_path.exists()
        
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        logger.info(f"Result saved to {self.csv_path}")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calcola metriche di valutazione aggregate"""
        import csv
        
        if not self.csv_path.exists():
            return {}
        
        metrics = {
            "total_queries": 0,
            "avg_response_time": 0,
            "cache_hit_rate": 0,
            "avg_num_sources": 0,
            "intent_distribution": {},
            "query_type_distribution": {}
        }
        
        response_times = []
        cache_hits = 0
        num_sources_list = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                metrics["total_queries"] += 1
                
                # Response time
                rt = float(row.get("response_time", 0))
                response_times.append(rt)
                
                # Cache hits
                if row.get("from_cache") == "True":
                    cache_hits += 1
                
                # Sources
                ns = int(row.get("num_sources", 0))
                num_sources_list.append(ns)
                
                # Intent distribution
                intent = row.get("intent", "unknown")
                metrics["intent_distribution"][intent] = metrics["intent_distribution"].get(intent, 0) + 1
                
                # Query type distribution
                qtype = row.get("query_type", "unknown")
                metrics["query_type_distribution"][qtype] = metrics["query_type_distribution"].get(qtype, 0) + 1
        
        # Calcola medie
        if response_times:
            metrics["avg_response_time"] = sum(response_times) / len(response_times)
            metrics["min_response_time"] = min(response_times)
            metrics["max_response_time"] = max(response_times)
        
        if metrics["total_queries"] > 0:
            metrics["cache_hit_rate"] = (cache_hits / metrics["total_queries"]) * 100
        
        if num_sources_list:
            metrics["avg_num_sources"] = sum(num_sources_list) / len(num_sources_list)
        
        # Salva metriche
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

