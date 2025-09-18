from typing import List, Optional, Tuple
from pathlib import Path
import logging as logger
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes
)
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
)

class DocumentProcessor:
    """Gestisce caricamento e processing dei documenti"""
    
    SUPPORTED_EXTENSIONS = [".txt", ".text", ".md", ".markdown"]
    
    def __init__(self, chunk_sizes: List[int], chunk_overlap: int = 50):
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        self.stats = {
            "total_documents": 0,
            "total_nodes": 0,
            "total_chars": 0
        }
    
    def load_documents(
        self, 
        file_paths: List[str] = None, 
        directories: List[str] = None
    ) -> List[Document]:
        """Carica documenti da file e directory"""
        all_documents = []
        
        # Carica file specifici
        if file_paths:
            for file_path in file_paths:
                doc = self._load_single_file(file_path)
                if doc:
                    all_documents.append(doc)
        
        # Carica da directory
        if directories:
            for directory in directories:
                docs = self._load_directory(directory)
                all_documents.extend(docs)
        
        # Rimuovi duplicati
        unique_docs = self._remove_duplicates(all_documents)
        
        # Aggiorna statistiche
        self.stats["total_documents"] = len(unique_docs)
        self.stats["total_chars"] = sum(len(doc.text) for doc in unique_docs)
        
        logger.info(f"Loaded {len(unique_docs)} unique documents")
        return unique_docs
    
    def _load_single_file(self, file_path: str) -> Optional[Document]:
        """Carica singolo file"""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
            
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported extension: {path.suffix}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty file: {file_path}")
                return None
            
            doc = Document(
                text=content,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_size": len(content),
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    "line_count": len(content.splitlines())
                }
            )
            
            logger.info(f"Loaded: {path.name} ({len(content)} chars)")
            return doc
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _load_directory(self, directory: str) -> List[Document]:
        """Carica documenti da directory"""
        if not Path(directory).exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        try:
            reader = SimpleDirectoryReader(
                input_dir=directory,
                recursive=True,
                filename_as_id=True,
                required_exts=self.SUPPORTED_EXTENSIONS,
                encoding="utf-8"
            )
            
            documents = reader.load_data()
            
            # Arricchisci metadata
            for doc in documents:
                if doc.metadata and doc.text:
                    doc.metadata.update({
                        "char_count": len(doc.text),
                        "word_count": len(doc.text.split()),
                        "line_count": len(doc.text.splitlines())
                    })
            
            logger.info(f"Loaded {len(documents)} documents from {directory}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading directory {directory}: {e}")
            return []
    
    def _remove_duplicates(self, documents: List[Document]) -> List[Document]:
        """Rimuove documenti duplicati basandosi sul path"""
        seen_sources = set()
        unique_docs = []
        
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source and source not in seen_sources:
                seen_sources.add(source)
                unique_docs.append(doc)
            elif not source:
                unique_docs.append(doc)
        
        if len(documents) > len(unique_docs):
            logger.info(f"Removed {len(documents) - len(unique_docs)} duplicates")
        
        return unique_docs
    
    def create_hierarchical_nodes(self, documents: List[Document]) -> Tuple[List, List]:
        """Crea nodi gerarchici dai documenti"""
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            chunk_overlap=self.chunk_overlap
        )
        
        nodes = node_parser.get_nodes_from_documents(documents)
        
        leaf_nodes = get_leaf_nodes(nodes)
        root_nodes = get_root_nodes(nodes)
        
        # Aggiungi metadata sui livelli
        for node in nodes:
            text_len = len(node.text)
            if text_len >= self.chunk_sizes[0] * 0.8:
                level = "parent"
            elif text_len >= self.chunk_sizes[1] * 0.8:
                level = "middle"
            else:
                level = "child"
            node.metadata["chunk_level"] = level
        
        self.stats["total_nodes"] = len(nodes)
        
        logger.info(f"Created {len(nodes)} total nodes ({len(leaf_nodes)} leaf, {len(root_nodes)} root)")
        return leaf_nodes, nodes