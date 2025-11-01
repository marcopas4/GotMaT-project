import faiss
import logging as logger


class FAISSIndexManager:
    """Gestisce creazione e configurazione dell'indice FAISS FLAT (deterministico)"""
    
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Inizializza manager per indice FLAT
        
        Args:
            dimension: Dimensione dei vettori di embedding
            index_type: Tipo di indice (sempre "Flat" per determinismo)
        """
        self.dimension = dimension
        self.index_type = index_type  # Mantenuto per compatibilità, ma sempre "Flat"
        self.index = None
        
    def create_index(self) -> faiss.Index:
        """
        Crea un nuovo indice FAISS FLAT L2
        
        FLAT usa ricerca esaustiva (exact nearest neighbor) - completamente deterministico
        
        Returns:
            Indice FAISS FLAT
        """
        index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"FLAT L2 index created (dimension={self.dimension}, deterministic=True)")
        return index
    
    def save_index(self, index: faiss.Index, path: str):
        """
        Salva indice su disco
        
        Args:
            index: Indice FAISS da salvare
            path: Percorso file di destinazione
        """
        faiss.write_index(index, path)
        logger.info(f"FLAT index saved to {path}")
    
    def load_index(self, path: str) -> faiss.Index:
        """
        Carica indice FLAT da disco
        
        Args:
            path: Percorso file indice
            
        Returns:
            Indice FAISS caricato
        """
        index = faiss.read_index(path)
        logger.info(f"FLAT index loaded from {path} (dimension={index.d})")
        return index