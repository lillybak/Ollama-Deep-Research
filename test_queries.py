#!/usr/bin/env python3
"""
Query Testing Framework for D&D RAG System

Use this script to test and optimize your queries for RAG-only results.
"""

from src.ollama_deep_researcher.rag import create_rag_system, RAGConfig
from typing import List, Dict, Any

def test_query_effectiveness(query: str, config: RAGConfig = None) -> Dict[str, Any]:
    """Test a single query and return detailed results."""
    rag = create_rag_system()
    if config:
        rag.config = config
    
    # Test both regular and hybrid search
    regular_results = rag.search_documents(query)
    hybrid_results = rag.hybrid_search(query, boost_dnd_terms=True)
    
    # Get formatted output lengths
    regular_formatted = rag.format_search_results(regular_results)
    hybrid_formatted = rag.format_search_results(hybrid_results)
    
    # Check if would skip web search
    regular_skip_web = (len(regular_formatted) >= 100 and 
                       'No relevant documents found' not in regular_formatted)
    hybrid_skip_web = (len(hybrid_formatted) >= 100 and 
                      'No relevant documents found' not in hybrid_formatted)
    
    return {
        'query': query,
        'regular': {
            'docs': len(regular_results),
            'chars': len(regular_formatted),
            'best_score': regular_results[0]['metadata']['score'] if regular_results else 0,
            'skip_web': regular_skip_web
        },
        'hybrid': {
            'docs': len(hybrid_results),
            'chars': len(hybrid_formatted),
            'best_score': hybrid_results[0]['metadata']['score'] if hybrid_results else 0,
            'skip_web': hybrid_skip_web
        }
    }

def test_query_batch(queries: List[str]) -> None:
    """Test multiple queries and display results."""
    print("🧪 D&D RAG Query Effectiveness Test\n")
    print("Legend: ✅ = RAG-only (skips web search), ❌ = Needs web search\n")
    
    for query in queries:
        result = test_query_effectiveness(query)
        
        print(f"📝 \"{query}\"")
        
        # Regular search results
        reg = result['regular']
        status = "✅" if reg['skip_web'] else "❌"
        print(f"   Regular:  {status} {reg['docs']} docs, {reg['chars']} chars, score: {reg['best_score']:.3f}")
        
        # Hybrid search results  
        hyb = result['hybrid']
        status = "✅" if hyb['skip_web'] else "❌"
        improvement = hyb['best_score'] - reg['best_score']
        print(f"   Hybrid:   {status} {hyb['docs']} docs, {hyb['chars']} chars, score: {hyb['best_score']:.3f} ({improvement:+.3f})")
        print()

def test_configurations(query: str) -> None:
    """Test different RAG configurations for a single query."""
    configs = [
        ("Strict", RAGConfig(score_threshold=0.7, top_k=5)),
        ("Balanced", RAGConfig(score_threshold=0.6, top_k=5)),
        ("Permissive", RAGConfig(score_threshold=0.5, top_k=7)),
    ]
    
    print(f"🎛️ Configuration Test for: \"{query}\"\n")
    
    for name, config in configs:
        result = test_query_effectiveness(query, config)
        reg = result['regular']
        status = "✅" if reg['skip_web'] else "❌"
        print(f"{name:>12}: {status} {reg['docs']} docs, {reg['chars']} chars, best: {reg['best_score']:.3f}")

if __name__ == "__main__":
    # Test different query styles
    sample_queries = [
        # Specific D&D mechanics
        "wizard spell slots and preparation",
        "rogue sneak attack damage calculation", 
        "paladin divine smite mechanics",
        
        # Natural language questions
        "How does multiclassing affect spell progression",
        "What are the different types of armor class",
        "Which classes get extra attack feature",
        
        # Potentially challenging queries
        "dungeon master advice",
        "campaign setting information",
        "player character optimization",
    ]
    
    print("=" * 60)
    test_query_batch(sample_queries)
    
    print("=" * 60)
    test_configurations("barbarian rage mechanics and resistances")
