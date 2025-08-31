"""RAG (Retrieval-Augmented Generation) module for DnD document search using Qdrant.

This module provides functionality to search and retrieve relevant DnD documents
from a Qdrant vector database for DnD document retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG operations."""
    
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "DnD_Documents"
    embedding_model: str = "mxbai-embed-large"
    embedding_base_url: str = "http://localhost:11434"
    top_k: int = 5
    score_threshold: float = 0.7
    max_content_length: int = 1000


class QdrantRAG:
    """RAG implementation using Qdrant vector database for DnD document retrieval."""
    
    def __init__(self, config: RAGConfig = None):
        """Initialize the RAG system with Qdrant client and embeddings.
        
        Args:
            config: RAG configuration object. If None, uses default config.
        """
        self.config = config or RAGConfig()
        self.client = None
        self.embeddings = None
        self._initialize_client()
        self._initialize_embeddings()
    
    def _initialize_client(self):
        """Initialize Qdrant client connection."""
        try:
            self.client = QdrantClient(url=self.config.qdrant_url)
            # Test connection
            self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.config.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant server: {e}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.embedding_base_url
            )
            logger.info(f"Initialized embeddings with model: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise RuntimeError(f"Cannot initialize embedding model: {e}")
    
    def search_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents in the Qdrant collection.
        
        Args:
            query: Search query string
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List of documents with metadata and scores
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If search operation fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        top_k = top_k or self.config.top_k
        
        try:
            # Generate query embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=self.config.score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                doc = {
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        "source": result.payload.get("source", "Unknown"),
                        "page": result.payload.get("page", 0),
                        "title": result.payload.get("title", ""),
                        "section": result.payload.get("section", ""),
                        "score": result.score
                    },
                    "id": result.id
                }
                formatted_results.append(doc)
            
            logger.info(f"Found {len(formatted_results)} documents for query: {query[:50]}...")
            return formatted_results
            
        except UnexpectedResponse as e:
            logger.error(f"Qdrant search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string for LLM consumption.
        
        Args:
            results: List of search results from search_documents()
            
        Returns:
            Formatted string with document contents and sources
        """
        if not results:
            return "No relevant documents found in the D&D knowledge base."
        
        formatted_sections = []
        
        for i, doc in enumerate(results, 1):
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Truncate content if too long
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length] + "..."
            
            section = f"""--- D&D Reference {i} ---
Source: {metadata.get('source', 'Unknown')}
Title: {metadata.get('title', 'N/A')}
Section: {metadata.get('section', 'N/A')}
Page: {metadata.get('page', 'N/A')}
Relevance Score: {metadata.get('score', 0):.3f}

Content:
{content}

"""
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            # Handle different vector config structures
            vector_config = collection_info.config.params.vectors
            if hasattr(vector_config, 'size'):
                vector_size = vector_config.size
            elif isinstance(vector_config, dict) and 'size' in vector_config:
                vector_size = vector_config['size']
            else:
                vector_size = "Unknown"
                
            return {
                "name": self.config.collection_name,
                "vector_size": vector_size,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def hybrid_search(self, query: str, boost_dnd_terms: bool = True) -> List[Dict[str, Any]]:
        """Perform enhanced search with D&D term boosting.
        
        Args:
            query: Search query
            boost_dnd_terms: Whether to boost D&D terminology
            
        Returns:
            Enhanced search results
        """
        # D&D terms that should be boosted
        dnd_terms = [
            "spell", "magic", "class", "race", "ability", "skill", "feat",
            "monster", "creature", "dungeon", "dragon", "character", "equipment", 
            "weapon", "armor", "adventure", "campaign", "dice", "level"
        ]
        
        enhanced_query = query
        if boost_dnd_terms:
            # Add weight to D&D terms if they appear in query
            for term in dnd_terms:
                if term.lower() in query.lower():
                    enhanced_query += f" {term} dnd"
        
        return self.search_documents(enhanced_query)
    
    def search_by_category(self, query: str, category: str) -> List[Dict[str, Any]]:
        """Search documents filtered by D&D category.
        
        Args:
            query: Search query
            category: D&D category (e.g., "spells", "monsters", "classes", "equipment")
            
        Returns:
            Filtered search results
        """
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Create filter for category
            category_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category.lower())
                    )
                ]
            )
            
            search_results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=category_filter,
                limit=self.config.top_k,
                score_threshold=self.config.score_threshold,
                with_payload=True
            )
            
            formatted_results = []
            for result in search_results:
                doc = {
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        "source": result.payload.get("source", "Unknown"),
                        "category": result.payload.get("category", ""),
                        "page": result.payload.get("page", 0),
                        "score": result.score
                    },
                    "id": result.id
                }
                formatted_results.append(doc)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Category search failed: {e}")
            return []


def create_rag_system(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "DnD_Documents",
    embedding_model: str = "mxbai-embed-large"
) -> QdrantRAG:
    """Factory function to create a RAG system with custom configuration.
    
    Args:
        qdrant_url: Qdrant server URL (full URL including protocol and port)
        collection_name: Name of the Qdrant collection
        embedding_model: Name of the embedding model
        
    Returns:
        Configured QdrantRAG instance
    """
    config = RAGConfig(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    return QdrantRAG(config)


# Utility functions for document ingestion (if needed later)
def prepare_dnd_document(
    content: str, 
    source: str, 
    title: str = "", 
    page: int = 0, 
    section: str = "",
    category: str = ""
) -> Dict[str, Any]:
    """Prepare a D&D document for ingestion into Qdrant.
    
    Args:
        content: Document content
        source: Source file/book name
        title: Document title
        page: Page number
        section: Section within document
        category: D&D category (e.g., "spells", "monsters", "classes", "equipment")
        
    Returns:
        Document dictionary ready for Qdrant ingestion
    """
    return {
        "content": content,
        "source": source,
        "title": title,
        "page": page,
        "section": section,
        "category": category.lower() if category else "",
        "content_length": len(content)
    }


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize RAG system
        rag = create_rag_system()
        
        # Test connection
        collection_info = rag.get_collection_info()
        print(f"Collection info: {collection_info}")
        
        # Example search
        query = "What are the different types of dragons in D&D?"
        results = rag.search_documents(query)
        
        print(f"\nSearch results for: {query}")
        print(rag.format_search_results(results))
        
    except Exception as e:
        print(f"Error: {e}")
