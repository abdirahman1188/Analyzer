import os
import json
import logging
from typing import List, Dict, Any, Optional

from api_utils import GeminiAPI
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentQueryEngine:
    """Engine for querying documents using vector search and LLM."""
    
    def __init__(self, vector_store: VectorStore, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the document query engine.
        
        Args:
            vector_store: Vector store with document embeddings
            api_key: Google API key for Gemini API access
            model_name: Name of the LLM model to use
        """
        self.vector_store = vector_store
        self.api = GeminiAPI(api_key, model_name)
    
    def query(self, query_text: str, document_filter: Optional[str] = None, num_results: int = 5) -> str:
        """
        Query documents with optional filtering by document name.
        
        Args:
            query_text: The user's query
            document_filter: Optional filename to filter results by
            num_results: Number of document chunks to retrieve
            
        Returns:
            Generated response based on relevant document chunks
        """
        logger.info(f"Querying: '{query_text}' with filter: {document_filter}")
        
        # Search for relevant documents
        if document_filter:
            # Filter results to specific document
            results = [doc for doc in self.vector_store.similarity_search(query_text, num_results) 
                      if doc.get("filename") and document_filter in doc.get("filename")]
            
            # If no results with filter, fall back to unfiltered (but log warning)
            if not results:
                logger.warning(f"No results found for filter '{document_filter}', falling back to unfiltered search")
                results = self.vector_store.similarity_search(query_text, num_results)
        else:
            # Standard search with no filter
            results = self.vector_store.similarity_search(query_text, num_results)
        
        # Build context from search results
        context = self._build_context(results, document_filter)
        
        # Generate response
        return self._generate_response(query_text, context, document_filter)
    
    def _build_context(self, results: List[Dict[str, Any]], document_filter: Optional[str] = None) -> str:
        """
        Build context from search results.
        
        Args:
            results: List of document chunks from vector search
            document_filter: Optional filename filter that was applied
            
        Returns:
            Formatted context string for the LLM
        """
        if document_filter:
            context = f"Based on the document '{document_filter}', here are relevant excerpts:\n\n"
        else:
            context = "Based on the provided documents, here are relevant excerpts:\n\n"
            
        for i, doc in enumerate(results):
            filename = doc.get("filename", "Unknown document")
            # Get just the base filename without path
            base_filename = os.path.basename(filename) if filename else "Unknown document"
            
            context += f"Document: {base_filename}\n"
            context += f"Excerpt {i+1}:\n{doc.get('text', '')}\n\n"
            
        return context
    
    def _generate_response(self, query: str, context: str, document_filter: Optional[str] = None) -> str:
        """
        Generate response using LLM.
        
        Args:
            query: The user's query
            context: Document context from vector search
            document_filter: Optional filename filter that was applied
            
        Returns:
            Generated response from LLM
        """
        # Customize instruction based on whether this is a single document or full collection
        if document_filter:
            instruction = f"You are an AI assistant analyzing a document about AI readiness in Somalia. Focus specifically on document '{document_filter}'."
        else:
            instruction = "You are an AI assistant analyzing multiple documents about AI readiness in Somalia."
        
        prompt = f"""{instruction}

Task: Answer the following question based ONLY on the provided context.
Question: {query}

CONTEXT:
{context}

Provide a detailed, well-structured answer addressing the question directly.
If the information is not in the context, say "I don't have enough information to answer this question based on the provided documents."

Format your response using markdown for readability. Include headers, bullet points, and emphasis where appropriate.
"""
        
        try:
            response = self.api.generate_content(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_report(self, topic: str, document_filters: Optional[List[str]] = None, max_length: Optional[int] = None) -> str:
        """
        Generate a comprehensive report on a topic.
        
        Args:
            topic: The report topic
            document_filters: Optional list of documents to focus on
            max_length: Optional maximum length for the report
            
        Returns:
            Generated report as markdown text
        """
        # For report generation, we need to build a more comprehensive prompt
        if document_filters:
            doc_filter_str = ", ".join(document_filters)
            instruction = f"You are an AI research assistant tasked with creating a comprehensive report on '{topic}' focusing on the documents: {doc_filter_str}."
        else:
            instruction = f"You are an AI research assistant tasked with creating a comprehensive report on '{topic}' based on all available documents."
        
        # Since reports are longer, we'll do a broader search with more results
        results = []
        
        # Get relevant document chunks - either filtered or all
        if document_filters:
            for doc_filter in document_filters:
                filtered_results = [doc for doc in self.vector_store.similarity_search(topic, 10) 
                                   if doc.get("filename") and doc_filter in doc.get("filename")]
                results.extend(filtered_results)
        else:
            results = self.vector_store.similarity_search(topic, 20)  # More results for comprehensive report
        
        # Build context
        context = self._build_context(results, None)
        
        prompt = f"""{instruction}

TASK:
Create a comprehensive, well-structured report on "{topic}" based ONLY on the information in the provided context.

CONTEXT:
{context}

REPORT FORMAT:
1. Executive Summary
2. Introduction
   - Background on AI readiness in Somalia
   - Purpose of this report
3. Current State Analysis
   - Key findings from the documents
   - Strengths and opportunities
   - Challenges and barriers
4. Recommendations
   - Strategic priorities
   - Implementation suggestions
   - Key stakeholders to involve
5. Conclusion

Format your report using markdown with proper headers, bullet points, and emphasis.
Focus on actionable insights and practical recommendations.
If certain sections cannot be addressed with the available information, note this briefly.

Your report should be comprehensive yet concise, focusing on the most important insights from the documents.
"""
        
        try:
            report = self.api.generate_content(prompt)
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"