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
    
    def query(self, query_text: str, document_filter: Optional[str] = None, num_results: int = 5, 
              use_model_knowledge: bool = True) -> str:
        """
        Query documents with optional filtering by document name.
        
        Args:
            query_text: The user's query
            document_filter: Optional filename to filter results by
            num_results: Number of document chunks to retrieve
            use_model_knowledge: Whether to allow the model to use its trained knowledge
            
        Returns:
            Generated response based on relevant document chunks
        """
        logger.info(f"Querying: '{query_text}' with filter: {document_filter}")
        
        # Search for relevant documents
        if document_filter:
            # Filter results to specific document
            results = [doc for doc in self.vector_store.similarity_search(query_text, num_results*2) 
                      if doc.get("filename") and document_filter in doc.get("filename")]
            
            # If no results with filter, fall back to unfiltered (but log warning)
            if not results:
                logger.warning(f"No results found for filter '{document_filter}', falling back to unfiltered search")
                results = self.vector_store.similarity_search(query_text, num_results)
            else:
                results = results[:num_results]  # Limit to requested number of results
        else:
            # Standard search with no filter
            results = self.vector_store.similarity_search(query_text, num_results)
        
        # Build context from search results
        context = self._build_context(results, document_filter)
        
        # Generate response
        return self._generate_response(query_text, context, document_filter, use_model_knowledge)
    
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
    
    def _generate_response(self, query: str, context: str, document_filter: Optional[str] = None, 
                          use_model_knowledge: bool = True) -> str:
        """
        Generate response using LLM.
        
        Args:
            query: The user's query
            context: Document context from vector search
            document_filter: Optional filename filter that was applied
            use_model_knowledge: Whether to allow model to use its trained knowledge
            
        Returns:
            Generated response from LLM
        """
        # Customize instruction based on whether this is a single document or full collection
        if document_filter:
            instruction = f"You are an AI assistant analyzing a document about AI readiness in Somalia. Focus specifically on document '{document_filter}'."
        else:
            instruction = "You are an AI assistant analyzing multiple documents about AI readiness in Somalia."
        
        # Add knowledge constraint based on parameter
        knowledge_instruction = ""
        if use_model_knowledge:
            knowledge_instruction = """
You can use both the provided context AND your general knowledge about AI readiness, digital transformation, 
and Somalia's specific circumstances to provide a comprehensive answer. When you use information not in the 
context, clearly indicate this by saying things like "Based on general knowledge..." or "In addition to the 
document information..."
"""
        else:
            knowledge_instruction = """
Use ONLY the information provided in the context. If the context doesn't contain enough information to 
answer the question fully, say "The provided documents don't contain sufficient information about [specific aspect]."
"""
        
        prompt = f"""{instruction}

Task: Answer the following question about Somalia's AI readiness.
Question: {query}

CONTEXT FROM DOCUMENTS:
{context}

KNOWLEDGE CONSTRAINTS:
{knowledge_instruction}

FORMAT:
Provide a detailed, well-structured answer addressing the question directly.
Format your response using markdown for readability. Include headers, bullet points, and emphasis where appropriate.
"""
        
        try:
            response = self.api.generate_content(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_report(self, topic: str, document_filters: Optional[List[str]] = None, 
                       max_length: Optional[str] = None, report_type: str = "standard") -> str:
        """
        Generate a comprehensive report on a topic.
        
        Args:
            topic: The report topic
            document_filters: Optional list of documents to focus on
            max_length: Optional indication of report length ("short", "standard", "long")
            report_type: Type of report ("standard", "comprehensive", "executive", "questionnaire", etc.)
            
        Returns:
            Generated report as markdown text
        """
        logger.info(f"Generating {report_type} report on '{topic}' with length '{max_length}'")
        
        # For report generation, we need to build a more comprehensive prompt
        if document_filters:
            doc_filter_str = ", ".join(document_filters)
            instruction = f"You are an AI research assistant tasked with creating a {report_type} report on '{topic}' focusing on the documents: {doc_filter_str}."
        else:
            instruction = f"You are an AI research assistant tasked with creating a {report_type} report on '{topic}' based on all available documents."
        
        # Determine how many chunks to retrieve based on requested length
        if max_length == "short":
            num_chunks = 15
        elif max_length == "long":
            num_chunks = 40
        else:  # standard
            num_chunks = 25
        
        # Special handling for Somalia AI readiness reports
        is_somalia_report = "somalia" in topic.lower() and ("ai" in topic.lower() or "readiness" in topic.lower())
        
        # Get relevant document chunks from multiple queries to ensure comprehensive coverage
        results = []
        
        # For Somalia AI readiness, use multiple targeted queries
        if is_somalia_report:
            search_queries = [
                "Somalia AI readiness infrastructure",
                "Somalia AI policy and governance",
                "Somalia digital transformation challenges",
                "Somalia AI skills and education",
                "Somalia technology infrastructure",
                "AI readiness assessment framework Somalia",
                "Somalia data ecosystem",
                "Somalia innovation ecosystem",
                "Somalia digital divide"
            ]
            
            # Allocate chunks across different queries to get diverse content
            chunks_per_query = max(2, num_chunks // len(search_queries))
            
            for query in search_queries:
                # Get chunks for this query
                query_results = self.vector_store.similarity_search(query, chunks_per_query)
                
                # Apply document filter if specified
                if document_filters:
                    query_results = [doc for doc in query_results if any(f in doc.get("filename", "") for f in document_filters)]
                
                results.extend(query_results)
            
            # Remove duplicates based on content fingerprint
            seen_texts = set()
            unique_results = []
            
            for doc in results:
                text = doc.get('text', '')[:100]  # Use first 100 chars as fingerprint
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append(doc)
                    
            results = unique_results[:num_chunks]  # Limit to prevent context overflow
        
        # Standard approach for other reports or if Somalia-specific approach didn't yield enough results
        if not is_somalia_report or len(results) < num_chunks // 2:
            # Get documents for the main topic
            main_results = self.vector_store.similarity_search(topic, num_chunks)
            
            # Apply document filter if specified
            if document_filters:
                main_results = [doc for doc in main_results if any(f in doc.get("filename", "") for f in document_filters)]
            
            # If we already have some results from the Somalia-specific approach, supplement them
            if results:
                # Remove duplicates when combining
                existing_texts = {doc.get('text', '')[:100] for doc in results}
                for doc in main_results:
                    text = doc.get('text', '')[:100]
                    if text not in existing_texts:
                        results.append(doc)
                        existing_texts.add(text)
                        
                # Limit to prevent context overflow
                results = results[:num_chunks]
            else:
                results = main_results
        
        # Build context from all retrieved chunks
        context = self._build_context(results, None)
        
        # Different prompt templates based on report type
        if report_type == "questionnaire":
            prompt = self._build_questionnaire_prompt(topic, context)
        elif report_type == "comprehensive" or (is_somalia_report and max_length == "long"):
            prompt = self._build_comprehensive_report_prompt(topic, context, is_somalia_report)
        elif report_type == "executive":
            prompt = self._build_executive_report_prompt(topic, context)
        else:  # standard
            prompt = self._build_standard_report_prompt(topic, context, is_somalia_report)
        
        try:
            report = self.api.generate_content(prompt)
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def _build_questionnaire_prompt(self, topic: str, context: str) -> str:
        """Build prompt for questionnaire generation"""
        return f"""You are an AI readiness assessment expert tasked with creating a questionnaire.

TASK:
Create a comprehensive questionnaire for assessing Somalia's AI readiness based on the provided document context.

CONTEXT FROM DOCUMENTS:
{context}

QUESTIONNAIRE REQUIREMENTS:
1. Create a well-structured questionnaire with 20-30 questions organized into relevant sections
2. Questions should be specifically tailored to Somalia's unique context and challenges
3. Include a mix of question types (multiple choice, Likert scale, open-ended)
4. Cover key dimensions of AI readiness including:
   - Infrastructure and connectivity
   - Data ecosystem and governance
   - Human capital and skills
   - Policy and regulatory environment
   - Business and innovation ecosystem
   - Funding and resources
   - Specific application areas for Somalia

FORMAT:
- Use proper markdown formatting with clear section headers
- For multiple choice questions, provide appropriate options
- For Likert scale questions, use a 5-point scale
- Include brief instructions for each section
- Begin with basic information collection and progress to more specific assessments

ADDITIONAL GUIDELINES:
- Make questions specific and actionable
- Avoid technical jargon where possible, but include appropriate technical questions
- Questions should help assess both current state and readiness for future development
- Include questions that address Somalia's specific challenges as identified in the documents
"""
    
    def _build_comprehensive_report_prompt(self, topic: str, context: str, is_somalia_focus: bool = False) -> str:
        """Build prompt for comprehensive reports (longer format)"""
        
        # Somalia-specific comprehensive report structure
        if is_somalia_focus:
            return f"""You are an expert researcher on AI readiness with deep knowledge of Somalia and developing nations.

TASK:
Create a comprehensive, detailed report on "Somalia's AI Readiness" that combines information from the provided documents with your general knowledge of AI readiness frameworks and Somalia's context.

CONTEXT FROM DOCUMENTS:
{context}

REPORT FORMAT:
# Somalia AI Readiness Assessment
## Executive Summary

## 1. Introduction
   - 1.1 Background and Context
   - 1.2 Importance of AI for Somalia's Development
   - 1.3 Assessment Framework and Methodology

## 2. Current Digital Landscape in Somalia
   - 2.1 Telecommunications Infrastructure
   - 2.2 Internet Access and Affordability
   - 2.3 Digital Service Adoption
   - 2.4 Power Infrastructure and Reliability

## 3. Data Ecosystem Analysis
   - 3.1 Data Availability and Sources
   - 3.2 Data Governance Frameworks
   - 3.3 Data Quality and Standardization
   - 3.4 Open Data Initiatives

## 4. Human Capital and Skills Assessment
   - 4.1 Digital Literacy Levels
   - 4.2 Technical Education and Training Programs
   - 4.3 AI/ML Expertise Availability
   - 4.4 Innovation and Research Capacity

## 5. Policy and Regulatory Environment
   - 5.1 Current Technology Policies
   - 5.2 Regulatory Frameworks for Data Protection
   - 5.3 Intellectual Property Considerations
   - 5.4 Governance Structures for Technology

## 6. Business and Innovation Ecosystem
   - 6.1 Technology Startup Landscape
   - 6.2 Private Sector Technology Adoption
   - 6.3 Investment and Funding Environment
   - 6.4 Industry-Academia Collaboration

## 7. Comparative Regional Analysis
   - 7.1 Comparison with East African Nations
   - 7.2 Lessons from Similar Developing Economies
   - 7.3 Applicable Best Practices

## 8. Key Challenges and Barriers
   - 8.1 Infrastructure Limitations
   - 8.2 Resource Constraints
   - 8.3 Skill Gaps
   - 8.4 Policy and Regulatory Challenges
   - 8.5 Security and Stability Considerations

## 9. Strategic Recommendations
   - 9.1 Short-term Actions (0-2 years)
   - 9.2 Medium-term Priorities (2-5 years)
   - 9.3 Long-term Vision (5+ years)
   - 9.4 Implementation Roadmap

## 10. Stakeholder Engagement Framework
    - 10.1 Government Agencies and Roles
    - 10.2 Private Sector Engagement
    - 10.3 Academic and Research Institutions
    - 10.4 International Partners and Donors

## 11. Conclusion and Way Forward

IMPORTANT GUIDELINES:
1. Base your report primarily on the provided document context, but supplement with your knowledge where gaps exist
2. When using information not explicitly in the documents, indicate this with phrases like "Generally, in developing economies..." or "Based on established AI readiness frameworks..."
3. Be specific and provide concrete examples wherever possible
4. Use proper markdown formatting (headers, bullet points, emphasis) for readability
5. Make the report comprehensive and substantive, equivalent to approximately 10 pages
6. Focus on actionable insights and practical recommendations
7. Be balanced - acknowledge both challenges and opportunities
8. Consider Somalia's unique context including resource constraints, recent history, and development priorities
"""
        # Generic comprehensive report structure
        else:
            return f"""You are an expert researcher tasked with creating a comprehensive report.

TASK:
Create a comprehensive, detailed report on "{topic}" based primarily on the provided context, supplemented with your general knowledge where appropriate.

CONTEXT FROM DOCUMENTS:
{context}

FORMAT AND STRUCTURE:
Create a well-structured report with appropriate sections and subsections.
Use markdown formatting for readability.
The report should be comprehensive (equivalent to approximately 10 pages).
Include concrete examples and specific details.
When using information not explicitly in the documents, indicate this clearly.

REPORT CONTENT SHOULD INCLUDE:
- Executive summary
- Introduction and background
- Comprehensive analysis of current state
- Detailed examination of challenges and opportunities
- Evidence-based recommendations
- Implementation considerations
- Conclusions and next steps

IMPORTANT:
Focus on actionable insights and practical recommendations.
Be balanced in your assessment.
Provide specific, detailed information rather than general statements.
Use proper citations when referring to specific documents.
"""

    def _build_executive_report_prompt(self, topic: str, context: str) -> str:
        """Build prompt for executive summary reports (shorter format)"""
        return f"""You are an expert consultant tasked with creating an executive brief.

TASK:
Create a concise executive brief on "{topic}" based on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

FORMAT AND STRUCTURE:
- Keep the report concise (equivalent to 2-3 pages)
- Use markdown for formatting
- Focus on key insights and actionable recommendations
- Prioritize information most relevant to executive decision-makers

SECTIONS TO INCLUDE:
1. Executive Summary (1-2 paragraphs)
2. Key Findings (3-5 bullet points)
3. Current Situation (brief assessment)
4. Strategic Implications (what this means for stakeholders)
5. Recommendations (prioritized, actionable items)
6. Next Steps (immediate actions to take)

IMPORTANT:
- Be direct and to the point
- Highlight decision-critical information
- Quantify impacts and benefits where possible
- Focus on strategic rather than operational details
"""

    def _build_standard_report_prompt(self, topic: str, context: str, is_somalia_focus: bool = False) -> str:
        """Build prompt for standard reports (medium format)"""
        
        # Somalia-specific standard report
        if is_somalia_focus:
            return f"""You are an expert researcher on AI readiness with knowledge of Somalia and developing nations.

TASK:
Create a well-structured report on "Somalia's AI Readiness" based on the provided documents, supplemented with your general knowledge where gaps exist.

CONTEXT FROM DOCUMENTS:
{context}

REPORT FORMAT:
# Somalia AI Readiness Assessment

## Executive Summary

## 1. Introduction
   - Background on AI readiness in Somalia
   - Purpose of this assessment

## 2. Current State Analysis
   - Infrastructure and connectivity
   - Data ecosystem
   - Human capital and skills
   - Policy and regulatory environment
   - Business and innovation ecosystem

## 3. Key Challenges
   - Identify major barriers to AI adoption
   - Resource constraints
   - Skill gaps

## 4. Opportunities and Potential
   - Priority sectors for AI application
   - Quick wins and long-term opportunities

## 5. Recommendations
   - Strategic priorities
   - Implementation suggestions
   - Key stakeholders to involve

## 6. Conclusion
   - Summary of key findings
   - Critical next steps

FORMAT GUIDELINES:
1. Base your report primarily on the provided document context
2. When using general knowledge not in the documents, indicate this clearly
3. Use markdown formatting for readability
4. Make recommendations specific and actionable
5. Consider Somalia's unique context and constraints
"""
        # Generic standard report
        else:
            return f"""You are an AI research assistant tasked with creating a report.

TASK:
Create a comprehensive, well-structured report on "{topic}" based on the information in the provided context.

CONTEXT FROM DOCUMENTS:
{context}

REPORT FORMAT:
1. Executive Summary
2. Introduction
   - Background on the topic
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

FORMAT GUIDELINES:
- Use markdown with proper headers, bullet points, and emphasis
- Focus on actionable insights and practical recommendations
- If certain sections cannot be addressed with the available information, note this briefly
- Your report should be comprehensive yet concise, focusing on the most important insights

Feel free to use your general knowledge to provide context and fill gaps, but base your key findings and recommendations on the document content.
"""