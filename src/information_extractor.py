import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from api_utils import GeminiAPI
from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class InformationExtractor:
    """Extract structured information from documents using vector search and LLM."""
    
    def __init__(self, api_key: str, vector_store: VectorStore, model_name: str = "models/gemini-pro"):
        """
        Initialize the information extractor.
        
        Args:
            api_key: Google API key for Gemini API access
            vector_store: Vector store with document embeddings
            model_name: Name of the LLM model to use
        """
        self.api = GeminiAPI(api_key, model_name)
        self.vector_store = vector_store
    
    def extract_information(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Extract structured information for a query.
        
        Args:
            query: The query or topic to extract information about
            num_results: Number of similar documents to retrieve
            
        Returns:
            Structured information about the query
        """
        logger.info(f"Extracting information for query: {query}")
        
        # Search for relevant documents
        search_results = self.vector_store.similarity_search(query, num_results)
        
        # Build context from search results
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(search_results)])
        
        # Generate prompt for information extraction
        prompt = self._build_extraction_prompt(query, context)
        
        # Get structured information from LLM
        try:
            response = self.api.generate_content(prompt)
            
            # Try to parse as JSON, if not return as text
            try:
                # Attempt to extract JSON from the response
                json_str = self._extract_json_from_response(response)
                structured_info = json.loads(json_str)
                return structured_info
            except json.JSONDecodeError:
                logger.warning(f"Could not parse response as JSON: {response[:100]}...")
                return {"text": response}
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            return {"error": str(e)}
    
    def _build_extraction_prompt(self, query: str, context: str) -> str:
        """Build a prompt for information extraction."""
        return f"""You are an AI assistant specialized in analyzing documents about AI readiness and digital transformation.

TASK:
Extract structured information about the following query based on the provided context documents:
Query: {query}

CONTEXT DOCUMENTS:
{context}

Analyze the context and extract key information about the query. Include:
1. Overview of the topic
2. Key challenges mentioned
3. Potential solutions or recommendations
4. Relevant stakeholders
5. Any metrics or indicators mentioned
6. Required capabilities or infrastructure

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
{{
  "overview": "Brief overview of the topic based on context",
  "challenges": ["challenge 1", "challenge 2", ...],
  "solutions": ["solution 1", "solution 2", ...],
  "stakeholders": ["stakeholder 1", "stakeholder 2", ...],
  "metrics": ["metric 1", "metric 2", ...],
  "required_capabilities": ["capability 1", "capability 2", ...]
}}

ONLY INCLUDE INFORMATION PRESENT IN THE CONTEXT. If information for a section is not available, use an empty array or "Not specified in context" as appropriate.
IMPORTANT: Ensure your response is a valid, parseable JSON object.
"""
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON string from LLM response."""
        # Try to find content between curly braces
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx+1]
        
        return response  # Return original if no JSON found
    
    def analyze_readiness(self, country: str = "Somalia") -> Dict[str, Any]:
        """
        Analyze AI readiness for a specific country.
        
        Args:
            country: The country to analyze
            
        Returns:
            Structured analysis of AI readiness for the country
        """
        query = f"AI readiness assessment for {country}"
        
        # Extract general information
        general_info = self.extract_information(query)
        
        # Get specific components
        components = [
            "digital infrastructure", 
            "data availability and quality", 
            "human capital and skills",
            "innovation ecosystem", 
            "policy and regulation", 
            "ethics and responsible AI"
        ]
        
        component_analyses = {}
        
        for component in components:
            component_query = f"{component} for AI readiness in {country}"
            component_analyses[component.replace(" ", "_")] = self.extract_information(component_query, num_results=3)
        
        # Compile complete analysis
        return {
            "country": country,
            "general_assessment": general_info,
            "component_analyses": component_analyses
        }
    
    def generate_recommendations(self, country: str = "Somalia") -> Dict[str, Any]:
        """
        Generate recommendations for improving AI readiness.
        
        Args:
            country: The country to generate recommendations for
            
        Returns:
            Structured recommendations
        """
        # First get an assessment
        assessment = self.analyze_readiness(country)
        
        # Extract challenges from assessment
        challenges = []
        
        # Add general challenges
        if "challenges" in assessment.get("general_assessment", {}):
            challenges.extend(assessment["general_assessment"]["challenges"])
        
        # Add component-specific challenges
        for component, info in assessment.get("component_analyses", {}).items():
            if "challenges" in info:
                for challenge in info["challenges"]:
                    if challenge not in challenges:
                        challenges.append(challenge)
        
        # Build prompt for recommendations
        challenge_text = "\n".join([f"- {challenge}" for challenge in challenges])
        prompt = f"""You are an AI assistant specialized in digital transformation and AI readiness.

TASK:
Generate strategic recommendations for improving AI readiness in {country} based on the identified challenges.

IDENTIFIED CHALLENGES:
{challenge_text}

FORMAT YOUR RESPONSE AS A VALID JSON OBJECT with the following structure:
{{
  "vision": "A brief vision statement for AI readiness in {country}",
  "strategic_priorities": [
    {{
      "priority": "Name of strategic priority 1",
      "description": "Brief description of this priority",
      "recommendations": [
        {{
          "recommendation": "Specific recommendation 1",
          "timeframe": "short/medium/long term",
          "resource_intensity": "low/medium/high",
          "stakeholders": ["stakeholder 1", "stakeholder 2"]
        }},
        // More recommendations...
      ]
    }},
    // More strategic priorities...
  ],
  "implementation_roadmap": {{
    "immediate_actions": ["action 1", "action 2"],
    "medium_term_actions": ["action 1", "action 2"],
    "long_term_actions": ["action 1", "action 2"]
  }}
}}

IMPORTANT: 
1. Make recommendations specific and actionable
2. Consider resource constraints of {country}
3. Prioritize high-impact interventions
4. Ensure your response is a valid, parseable JSON object
"""
        
        # Generate recommendations
        try:
            response = self.api.generate_content(prompt)
            
            # Try to parse as JSON
            try:
                json_str = self._extract_json_from_response(response)
                recommendations = json.loads(json_str)
                return recommendations
            except json.JSONDecodeError:
                logger.warning(f"Could not parse recommendations as JSON")
                return {"text": response}
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}
    
    def process_all_documents(self, filenames: List[str], output_dir: str) -> Dict[str, Any]:
        """
        Process all documents and extract structured information from each.
        
        Args:
            filenames: List of document filenames
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary mapping filenames to extracted information
        """
        logger.info(f"Processing {len(filenames)} documents for information extraction")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for filename in filenames:
            # Extract base filename without path and extension
            base_filename = os.path.basename(filename)
            output_file = os.path.join(output_dir, f"{base_filename}.json")
            
            # Skip if already processed
            if os.path.exists(output_file):
                logger.info(f"Skipping already processed file: {base_filename}")
                with open(output_file, 'r') as f:
                    results[base_filename] = json.load(f)
                continue
            
            logger.info(f"Extracting information from: {base_filename}")
            
            # Build query from filename
            query = f"Key information and insights from document: {base_filename}"
            
            # Extract information
            try:
                info = self.extract_information(query)
                
                # Save to file
                with open(output_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                results[base_filename] = info
                
            except Exception as e:
                logger.error(f"Error processing {base_filename}: {str(e)}")
                results[base_filename] = {"error": str(e)}
        
        # Save summary file
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            summary = {
                "total_documents": len(filenames),
                "processed_documents": len(results),
                "timestamp": str(datetime.now()),
                "results": results
            }
            json.dump(summary, f, indent=2)
        
        logger.info(f"Completed information extraction for {len(results)} documents")
        return results

# For testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    parser = argparse.ArgumentParser(description="Test information extraction")
    parser.add_argument("--vector_store", default="results/vector_store", help="Vector store directory")
    parser.add_argument("--query", default="AI readiness assessment framework for Somalia", help="Test query")
    
    args = parser.parse_args()
    
    # Load vector store
    vector_store = VectorStore(api_key)
    vector_store.load(args.vector_store)
    
    # Create information extractor
    extractor = InformationExtractor(api_key, vector_store)
    
    # Extract information for query
    info = extractor.extract_information(args.query)
    
    print("\nExtracted Information:")
    print(json.dumps(info, indent=2))