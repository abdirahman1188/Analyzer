import os
import logging
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from google import genai
# Import the GeminiAPI utility
from api_utils import GeminiAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VectorStore:
    """Create and manage vector embeddings for document chunks using Gemini API."""
    
    def __init__(self, api_key: str, embedding_model: str = "models/embedding-001"):
        """
        Initialize the vector store.
        
        Args:
            api_key: Google API key for Gemini API access
            embedding_model: Gemini embedding model to use
        """
        self.api_key = api_key
        self.embedding_model = embedding_model
        
        # Initialize API utility
        self.api = GeminiAPI(api_key)
        
        # Initialize FAISS index (to be created when embeddings are generated)
        self.index = None
        self.documents = []  # Store document info for retrieval
        self.dimension = None  # Will be set when first embedding is created
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a text using Gemini API.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Use the API utility to generate embeddings with retry logic
            embedding = self.api.generate_embedding(text)
            
            # Convert to numpy array
            embedding_array = np.array(embedding["embedding"]).astype(np.float32)
            
            # Set dimension if not already set
            if self.dimension is None:
                self.dimension = embedding_array.shape[0]
                
            return embedding_array
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            if self.dimension:
                return np.zeros(self.dimension).astype(np.float32)
            else:
                # Default dimension for Gemini embeddings
                return np.zeros(768).astype(np.float32)
    
    def create_vector_store(self, documents: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
        """
        Create a vector store from document chunks.
        
        Args:
            documents: List of documents with filename, path, and chunks
            save_path: Optional path to save the vector store
        """
        if not documents:
            logging.warning("No documents provided for vector store creation")
            return
            
        total_chunks = sum(doc.get("chunk_count", 0) for doc in documents)
        logging.info(f"Creating vector store with {total_chunks} chunks from {len(documents)} documents")
        
        # Generate embeddings for all chunks
        embeddings = []
        self.documents = []
        
        for doc in tqdm(documents, desc="Creating embeddings"):
            filename = doc.get("filename", "")
            
            for i, chunk_text in enumerate(doc.get("chunks", [])):
                # Store document data for later retrieval
                chunk_doc = {
                    "id": len(self.documents),
                    "text": chunk_text,
                    "filename": filename,
                    "chunk_id": i,
                    "total_chunks": len(doc.get("chunks", [])),
                    "path": doc.get("path", "")
                }
                self.documents.append(chunk_doc)
                
                # Generate embedding
                embedding = self.generate_embedding(chunk_text)
                embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        # Create FAISS index
        self.dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
        
        logging.info(f"Created FAISS index with {self.index.ntotal} vectors of dimension {self.dimension}")
        
        # Save if path provided
        if save_path:
            self.save(save_path)
    
    def save(self, save_path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            save_path: Directory path to save the vector store
        """
        if not self.index:
            logging.warning("No index to save")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, "faiss_index.bin"))
        
        # Save document info (excluding full text to reduce file size)
        docs_for_save = []
        for doc in self.documents:
            doc_copy = doc.copy()
            # Store only the first 100 chars of text as a preview
            preview = doc.get("text", "")[:100] + "..." if len(doc.get("text", "")) > 100 else doc.get("text", "")
            doc_copy["text_preview"] = preview
            doc_copy["text"] = ""  # Remove full text
            docs_for_save.append(doc_copy)
            
        with open(os.path.join(save_path, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(docs_for_save, f, ensure_ascii=False, indent=2)
            
        # Save full text separately in chunks
        chunk_size = 1000  # Store 1000 docs per file to avoid huge files
        for i in range(0, len(self.documents), chunk_size):
            chunk_docs = self.documents[i:i+chunk_size]
            # Only save id and text fields
            text_data = [{
                "id": doc.get("id"),
                "text": doc.get("text", "")
            } for doc in chunk_docs]
            
            with open(os.path.join(save_path, f"texts_{i//chunk_size}.json"), 'w', encoding='utf-8') as f:
                json.dump(text_data, f, ensure_ascii=False)
            
        # Save metadata (dimension, etc.)
        with open(os.path.join(save_path, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
                "num_vectors": self.index.ntotal if self.index else 0,
                "num_documents": len(self.documents)
            }, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Vector store saved to {save_path}")
    
    def load(self, load_path: str) -> None:
        """
        Load a vector store from disk.
        
        Args:
            load_path: Directory path to load the vector store from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
            
            # Load document info
            with open(os.path.join(load_path, "documents.json"), 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            # Load metadata
            with open(os.path.join(load_path, "metadata.json"), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.dimension = metadata.get("dimension")
                self.embedding_model = metadata.get("embedding_model")
            
            # Load text data
            i = 0
            while os.path.exists(os.path.join(load_path, f"texts_{i}.json")):
                with open(os.path.join(load_path, f"texts_{i}.json"), 'r', encoding='utf-8') as f:
                    text_data = json.load(f)
                    
                    # Update documents with full text
                    for text_item in text_data:
                        doc_id = text_item.get("id")
                        text = text_item.get("text", "")
                        
                        # Find and update the corresponding document
                        for doc in self.documents:
                            if doc.get("id") == doc_id:
                                doc["text"] = text
                                break
                
                i += 1
            
            logging.info(f"Loaded vector store from {load_path} with {self.index.ntotal} vectors")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents to a query.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional filter to apply to results (e.g. {"filename": "paper1.pdf"})
            
        Returns:
            List of similar documents with scores
        """
        if not self.index:
            logging.warning("No index for search")
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS index
        # Get more results if filtering
        search_k = k * 10 if filter_dict else k  
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Process results
        results = []
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx].copy()
            
            # Apply filter if provided
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    if key in doc and doc[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # Add distance score
            doc['score'] = float(1.0 / (1.0 + distance))  # Convert distance to similarity score
            results.append(doc)
            
            # Stop if we have enough results after filtering
            if len(results) >= k:
                break
        
        return results[:k]  # Return top k results

if __name__ == "__main__":
    # Test implementation
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    parser = argparse.ArgumentParser(description="Test vector store functionality")
    parser.add_argument("--chunks_dir", default="data/chunks", help="Directory containing document chunks")
    parser.add_argument("--output_dir", default="results/vector_store", help="Directory to save vector store")
    parser.add_argument("--query", default="What is AI readiness assessment?", help="Test query")
    
    args = parser.parse_args()
    
    # Load chunks from directory
    from document_processor import process_pdfs
    
    # Get input directory from chunks directory
    input_dir = os.path.dirname(args.chunks_dir)
    docs = process_pdfs(input_dir, args.chunks_dir)
    
    # Create vector store
    vector_store = VectorStore(api_key)
    vector_store.create_vector_store(docs, args.output_dir)
    
    # Test search
    results = vector_store.similarity_search(args.query, k=3)
    
    print(f"\nSearch results for: '{args.query}'")
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
        print(f"  File: {result['filename']}")
        print(f"  Chunk: {result['chunk_id']} of {result['total_chunks']}")
        print(f"  Text preview: {result['text'][:150]}...")