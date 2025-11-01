from pathlib import Path

from config import RAGConfig, ResponseMode
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
        chunk_sizes=[2048, 512],
        temperature=0.3,
        context_window=4096,
        use_reranker=True,
        use_automerging=True,
        chunk_overlap=150
    )
    
    print("\n📊 Configuration:")
    print(f"  LLM: {config.llm_model}")
    print(f"  Embeddings: {config.embedding_model}")
    print(f"  Chunks: {config.chunk_sizes}")
    print(f"  Reranker: {'Enabled' if config.use_reranker else 'Disabled'}")
    print("="*60)
    
    # Inizializza pipeline
    pipeline = OptimizedRAGPipeline(config)
    
    # Inizializza evaluation manager
    evaluator = EvaluationManager()
    
    # Verifica esistenza directory documenti
    docs_dir = config.docs_path
    if not Path(docs_dir).exists():
        print(f"❌ Documents directory '{docs_dir}' not found. Please add documents and restart.")
        return
    
    # Costruisci indice
    print("\n📚 Building index...")
    try:
        pipeline.build_index(directories=[docs_dir])
        print("✅ Index built successfully")
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup query engine
    try:
        pipeline.setup_query_engine(ResponseMode.TREE_SUMMARIZE)
        print("✅ Query engine configured")
    except Exception as e:
        print(f"❌ Error setting up query engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Mostra statistiche iniziali
    try:
        stats = pipeline.get_statistics()
        print("\n📊 Initial Statistics:")
        print(f"  Documents: {stats['data']['total_documents']}")
        print(f"  Nodes: {stats['data']['total_nodes']}")
    except Exception as e:
        print(f"⚠️  Warning: Could not retrieve statistics: {e}")
    
    # Menu interattivo
    print("\n" + "="*60)
    print("💬 Interactive Mode")
    print("Commands:")
    print("  'quit' - Exit")
    print("  'stats' - Show statistics")
    print("  'metrics' - Calculate evaluation metrics")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                try:
                    stats = pipeline.get_statistics()
                    print("\n📊 Pipeline Statistics:")
                    print(f"\n  Configuration:")
                    print(f"    LLM: {stats['configuration']['llm_model']}")
                    print(f"    Embeddings: {stats['configuration']['embedding_model']}")
                    print(f"    Index: {stats['configuration']['index_type']}")
                    print(f"    Reranker: {'Enabled' if stats['configuration']['reranker_enabled'] else 'Disabled'}")
                    
                    print(f"\n  Data:")
                    print(f"    Documents: {stats['data']['total_documents']}")
                    print(f"    Nodes: {stats['data']['total_nodes']}")
                    
                    print(f"\n  Performance:")
                    print(f"    Total queries: {stats['performance']['total_queries']}")
                    print(f"    Avg response time: {stats['performance']['avg_response_time']}")
                    
                    if 'retrieval_stats' in stats:
                        rs = stats['retrieval_stats']
                        print(f"\n  Retrieval:")
                        print(f"    Total nodes retrieved: {rs['total_nodes_retrieved']}")
                        print(f"    After deduplication: {rs['total_nodes_after_dedup']}")
                        print(f"    After reranking: {rs['total_nodes_after_rerank']}")
                        print(f"    Avg dedup reduction: {rs['avg_dedup_reduction']:.1%}")
                    
                    if 'reranker' in stats:
                        print(f"\n  Reranker:")
                        for key, value in stats['reranker'].items():
                            print(f"    {key}: {value}")
                            
                except Exception as e:
                    print(f"❌ Error retrieving statistics: {e}")
            
            elif user_input.lower() == 'metrics':
                try:
                    metrics = evaluator.calculate_metrics()
                    print("\n📊 Evaluation Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for k, v in value.items():
                                print(f"    {k}: {v}")
                        else:
                            print(f"  {key}: {value}")
                except Exception as e:
                    print(f"❌ Error calculating metrics: {e}")
            
            elif user_input:
                try:
                    # Esegui query con enhancement
                    result = pipeline.query(user_input, enhance_query=True)
                    
                    # Mostra risposta
                    print(f"\n{'='*60}")
                    print(f"💡 Answer:\n")
                    print(result['answer'])
                    
                    # Mostra query metadata se presente
                    if 'query_metadata' in result and result['query_metadata']:
                        metadata = result['query_metadata']
                        
                        # Query Analysis
                        print(f"\n🔍 Query Analysis:")
                        
                        # Expansions info
                        if 'expansions' in metadata:
                            exp = metadata['expansions']
                            
                            # Intent
                            if 'intent' in exp:
                                print(f"  Intent: {exp['intent']}")
                            
                            # Keywords (top 5)
                            if 'keywords' in exp and exp['keywords']:
                                keywords = exp['keywords'][:5]
                                print(f"  Keywords: {', '.join(keywords)}")
                            
                            # Semantic variants count
                            if 'semantic_variants' in exp and exp['semantic_variants']:
                                print(f"  Semantic variants: {len(exp['semantic_variants'])}")
                            
                            # Sub-queries count
                            if 'sub_queries' in exp and exp['sub_queries']:
                                print(f"  Sub-queries: {len(exp['sub_queries'])}")
                        
                        # Number of queries generated
                        if 'num_queries_generated' in metadata:
                            print(f"  Total queries generated: {metadata['num_queries_generated']}")
                        
                        # Retrieval Pipeline Stats
                        print(f"\n📊 Retrieval Pipeline:")
                        
                        # Retrieval
                        if 'retrieval' in metadata:
                            ret = metadata['retrieval']
                            nodes_retrieved = ret.get('total_nodes_retrieved', 0)
                            print(f"  1. Multi-retrieval: {nodes_retrieved} nodes")
                        
                        # Deduplication
                        if 'deduplication' in metadata:
                            dedup = metadata['deduplication']
                            nodes_before = dedup.get('nodes_before', 0)
                            nodes_after = dedup.get('nodes_after', 0)
                            duplicates = dedup.get('duplicates_removed', 0)
                            print(f"  2. Deduplication: {nodes_before} → {nodes_after} (-{duplicates} duplicates)")
                        
                        # Reranking
                        if 'reranking' in metadata:
                            rerank = metadata['reranking']
                            if rerank.get('applied'):
                                nodes_before = rerank.get('nodes_before', 0)
                                nodes_after = rerank.get('nodes_after', 0)
                                print(f"  3. Reranking: {nodes_before} → {nodes_after} nodes ✓")
                            else:
                                print(f"  3. Reranking: Not applied")
                    
                    # Mostra fonti
                    if 'sources' in result and result['sources']:
                        print(f"\n📚 Top Sources ({len(result['sources'])} total):")
                        for i, source in enumerate(result['sources'][:3], 1):
                            print(f"\n  [{i}] Score: {source['score']:.3f}")
                            print(f"      {source['text'][:150]}...")
                            if source.get('reranked'):
                                print(f"      [✓ Reranked]")
                    
                    # Performance summary (solo tempo totale)
                    print(f"\n⚡ Total time: {result['response_time']:.3f}s")
                    
                    # Salva risultato per valutazione
                    try:
                        evaluator.save_query_result(result)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not save result: {e}")
                        
                except Exception as e:
                    print(f"❌ Query error: {e}")
                    import traceback
                    traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()