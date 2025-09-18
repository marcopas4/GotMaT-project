import sys
from pathlib import Path

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAGConfig, IndexType, ResponseMode
from core import OptimizedRAGPipeline
from evaluation import EvaluationManager


def main():
    """Main execution function with interactive mode"""
    
    print("\n" + "="*60)
    print("🚀 RAG Pipeline Modulare - Apple Silicon M4 Optimized")
    print("="*60)
    
    # Configurazione ottimizzata
    config = RAGConfig(
        llm_model="llama3.2:3b-instruct-q4_K_M",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        chunk_sizes=[2048, 512, 128],
        index_type=IndexType.HNSW,
        temperature=0.3,
        context_window=4096,
        enable_cache=True,
        use_reranker=False,
        use_automerging=True,
        storage_path="./storage",
    )
    
    print("\n📊 Configuration:")
    print(f"  LLM: {config.llm_model}")
    print(f"  Embeddings: {config.embedding_model}")
    print(f"  Index: {config.index_type.value}")
    print(f"  Chunks: {config.chunk_sizes}")
    print("="*60)
    
    # Inizializza pipeline
    pipeline = OptimizedRAGPipeline(config)
    
    # Inizializza evaluation manager
    evaluator = EvaluationManager()
    
    # Prova a caricare indice esistente
    if pipeline.load_index():
        print("✅ Existing index loaded successfully")
    else:
        print("📚 Building new index...")
        
        # Crea documenti di esempio se necessario
        docs_dir = "./data/documents"
        if not Path(docs_dir).exists():
            print(f"❌ Documents directory '{docs_dir}' not found. Please add documents and restart.")
            return
        
        # Costruisci indice
        try:
            pipeline.build_index(directories=[docs_dir])
            print("✅ Index built successfully")
        except Exception as e:
            print(f"❌ Error building index: {e}")
            return
    
    # Setup query engine
    pipeline.setup_query_engine(ResponseMode.TREE_SUMMARIZE)
    
    # Mostra statistiche iniziali
    stats = pipeline.get_statistics()
    print("\n📊 Initial Statistics:")
    print(f"  Documents: {stats['data']['total_documents']}")
    print(f"  Nodes: {stats['data']['total_nodes']}")
    print(f"  Cache: {'Enabled' if stats['cache']['enabled'] else 'Disabled'}")
    
    # Menu interattivo
    print("\n" + "="*60)
    print("💬 Interactive Mode")
    print("Commands:")
    print("  'quit' - Exit")
    print("  'stats' - Show statistics")
    print("  'clear' - Clear cache")
    print("  'metrics' - Calculate evaluation metrics")
    print("  'update <path>' - Update index with new documents")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                stats = pipeline.get_statistics()
                print("\n📊 Current Statistics:")
                for category, values in stats.items():
                    print(f"\n{category.upper()}:")
                    for key, value in values.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for k, v in value.items():
                                print(f"    {k}: {v}")
                        else:
                            print(f"  {key}: {value}")
            
            elif user_input.lower() == 'clear':
                pipeline.clear_cache()
                print("✅ Cache cleared")
            
            elif user_input.lower() == 'metrics':
                metrics = evaluator.calculate_metrics()
                print("\n📊 Evaluation Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            
            elif user_input.lower().startswith('update'):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    path = parts[1]
                    try:
                        if Path(path).is_dir():
                            pipeline.update_index(directories=[path])
                        else:
                            pipeline.update_index(file_paths=[path])
                        print("✅ Index updated")
                    except Exception as e:
                        print(f"❌ Update failed: {e}")
                else:
                    print("❌ Please specify a path to update")
            
            elif user_input:
                # Esegui query con enhancement
                result = pipeline.query(user_input, enhance_query=True)
                
                # Mostra risposta
                print(f"\n{'='*60}")
                print(f"💡 Answer:\n")
                print(result['answer'])
                
                # Mostra query metadata se presente
                if 'query_metadata' in result and result['query_metadata']:
                    metadata = result['query_metadata']
                    print(f"\n🔍 Query Analysis:")
                    print(f"  Intent: {metadata.get('intent', 'unknown')}")
                    print(f"  Type: {metadata.get('query_type', 'unknown')}")
                    if metadata.get('expanded_terms'):
                        print(f"  Expanded terms: {', '.join(metadata['expanded_terms'])}")
                
                # Mostra fonti
                if 'sources' in result and result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"\n  [{i}] Score: {source['score']:.3f} | Level: {source['chunk_level']}")
                        print(f"      {source['text'][:150]}...")
                
                # Mostra performance
                print(f"\n⚡ Performance:")
                print(f"  Response time: {result['response_time']:.3f}s")
                print(f"  From cache: {'Yes' if result['from_cache'] else 'No'}")
                
                # Salva risultato per valutazione
                evaluator.save_query_result(result)
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()