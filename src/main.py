import os
import argparse
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_readiness_analyzer.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main function to run the AI readiness assessment analysis pipeline."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Readiness Assessment Analyzer")
    parser.add_argument("--papers_dir", default="data/papers", help="Directory containing PDF papers")
    parser.add_argument("--chunks_dir", default="data/chunks", help="Directory to save processed chunks")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--model", default="models/gemini-pro", help="Gemini model to use")
    parser.add_argument("--skip_processing", action="store_true", help="Skip document processing if already done")
    parser.add_argument("--skip_vector_store", action="store_true", help="Skip vector store creation if already done")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip information extraction if already done")
    parser.add_argument("--skip_questionnaire", action="store_true", help="Skip questionnaire generation if already done")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.chunks_dir, exist_ok=True)
    
    vector_store_dir = os.path.join(args.output_dir, "vector_store")
    analysis_dir = os.path.join(args.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Step 1: Process documents
    from document_processor import process_pdfs
    
    if not args.skip_processing and (not os.path.exists(args.chunks_dir) or len(os.listdir(args.chunks_dir)) == 0):
        logging.info("Step 1: Processing PDF documents...")
        processed_docs = process_pdfs(args.papers_dir, args.chunks_dir)
    else:
        logging.info("Skipping document processing...")
        # Load document information if needed later
        processed_docs = None
    
    # Step 2: Create vector store
    from vector_store import VectorStore
    
    vector_store = VectorStore(api_key)
    
    if not args.skip_vector_store and (not os.path.exists(vector_store_dir) or 
                                      not os.path.exists(os.path.join(vector_store_dir, "faiss_index.bin"))):
        logging.info("Step 2: Creating vector store...")
        
        # If we skipped processing but need to create vector store, 
        # process documents without saving chunks
        if processed_docs is None:
            processed_docs = process_pdfs(args.papers_dir)
        
        vector_store.create_vector_store(processed_docs, vector_store_dir)
    else:
        logging.info("Loading existing vector store...")
        vector_store.load(vector_store_dir)
    
    # Step 3: Extract information
    from information_extractor import InformationExtractor
    
    all_questions_file = os.path.join(analysis_dir, "all_questions.json")
    
    if not args.skip_analysis:
        logging.info("Step 3: Extracting information from documents...")
        extractor = InformationExtractor(api_key, vector_store, model_name=args.model)
        
        # Get unique filenames
        filenames = list(set(doc.get("filename", "") for doc in vector_store.documents if doc.get("filename")))
        logging.info(f"Found {len(filenames)} unique documents")
        
        # Extract information
        extraction_results = extractor.process_all_documents(filenames, analysis_dir)
        
        # Create the all_questions.json file by extracting questions from each document
        logging.info("Creating all_questions.json file...")
        all_questions = []
        
        # For each document, extract questions if they exist or generate them
        for filename in filenames:
            base_filename = os.path.basename(filename)
            doc_file = os.path.join(analysis_dir, f"{base_filename}.json")
            if os.path.exists(doc_file):
                with open(doc_file, 'r', encoding='utf-8') as f:
                    try:
                        doc_data = json.load(f)
                        
                        # Extract questions if they exist, otherwise generate from the information
                        if "questions" in doc_data:
                            doc_questions = doc_data["questions"]
                            all_questions.extend(doc_questions)
                        else:
                            # Extract key points to generate questions
                            overview = doc_data.get("overview", "")
                            challenges = doc_data.get("challenges", [])
                            solutions = doc_data.get("solutions", [])
                            capabilities = doc_data.get("required_capabilities", [])
                            stakeholders = doc_data.get("stakeholders", [])
                            
                            # Generate questions based on content
                            for item in challenges:
                                if item and item != "Not specified in context":
                                    all_questions.append({
                                        "question": f"How is Somalia addressing the challenge of: {item}?",
                                        "category": "Challenges",
                                        "source": base_filename
                                    })
                            
                            for item in solutions:
                                if item and item != "Not specified in context":
                                    all_questions.append({
                                        "question": f"To what extent has Somalia implemented the solution: {item}?",
                                        "category": "Solutions",
                                        "source": base_filename
                                    })
                            
                            for item in capabilities:
                                if item and item != "Not specified in context":
                                    all_questions.append({
                                        "question": f"Does Somalia have the capability for: {item}?",
                                        "category": "Capabilities",
                                        "source": base_filename
                                    })
                            
                            for item in stakeholders:
                                if item and item != "Not specified in context":
                                    all_questions.append({
                                        "question": f"What role does {item} play in Somalia's AI readiness?",
                                        "category": "Stakeholders",
                                        "source": base_filename
                                    })
                            
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse JSON from {doc_file}")
        
        # Remove duplicates by converting questions to string and using a set
        unique_questions = []
        seen_questions = set()
        
        for q in all_questions:
            q_text = q.get("question", "")
            if q_text and q_text not in seen_questions:
                seen_questions.add(q_text)
                unique_questions.append(q)
        
        # Save to all_questions.json
        with open(all_questions_file, 'w', encoding='utf-8') as f:
            json.dump(unique_questions, f, indent=2)
        
        logging.info(f"Created {all_questions_file} with {len(unique_questions)} unique questions")
        
        # Update all_questions variable for questionnaire generation
        all_questions = unique_questions
    else:
        logging.info("Skipping information extraction...")
        # Load extracted questions for questionnaire generation
        if os.path.exists(all_questions_file):
            with open(all_questions_file, 'r', encoding='utf-8') as f:
                all_questions = json.load(f)
        else:
            # If the file doesn't exist but we're skipping analysis, we need to create it from existing document analysis
            if os.path.exists(analysis_dir) and len(os.listdir(analysis_dir)) > 0:
                logging.info("Creating all_questions.json from existing document analysis...")
                all_questions = []
                
                # Process each existing document analysis file
                for file in os.listdir(analysis_dir):
                    if file.endswith('.json') and file != 'all_questions.json' and file != 'summary.json':
                        doc_file = os.path.join(analysis_dir, file)
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            try:
                                doc_data = json.load(f)
                                
                                # Same logic as above for extracting or generating questions
                                if "questions" in doc_data:
                                    doc_questions = doc_data["questions"]
                                    all_questions.extend(doc_questions)
                                else:
                                    # Extract key points to generate questions
                                    challenges = doc_data.get("challenges", [])
                                    solutions = doc_data.get("solutions", [])
                                    capabilities = doc_data.get("required_capabilities", [])
                                    stakeholders = doc_data.get("stakeholders", [])
                                    
                                    # Generate questions based on content
                                    for item in challenges:
                                        if item and item != "Not specified in context":
                                            all_questions.append({
                                                "question": f"How is Somalia addressing the challenge of: {item}?",
                                                "category": "Challenges",
                                                "source": file
                                            })
                                    
                                    for item in solutions:
                                        if item and item != "Not specified in context":
                                            all_questions.append({
                                                "question": f"To what extent has Somalia implemented the solution: {item}?",
                                                "category": "Solutions",
                                                "source": file
                                            })
                                    
                                    for item in capabilities:
                                        if item and item != "Not specified in context":
                                            all_questions.append({
                                                "question": f"Does Somalia have the capability for: {item}?",
                                                "category": "Capabilities",
                                                "source": file
                                            })
                                            
                                    for item in stakeholders:
                                        if item and item != "Not specified in context":
                                            all_questions.append({
                                                "question": f"What role does {item} play in Somalia's AI readiness?",
                                                "category": "Stakeholders",
                                                "source": file
                                            })
                                    
                            except json.JSONDecodeError:
                                logging.warning(f"Could not parse JSON from {doc_file}")
                
                # Remove duplicates
                unique_questions = []
                seen_questions = set()
                
                for q in all_questions:
                    q_text = q.get("question", "")
                    if q_text and q_text not in seen_questions:
                        seen_questions.add(q_text)
                        unique_questions.append(q)
                
                # Save to all_questions.json
                with open(all_questions_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_questions, f, indent=2)
                
                logging.info(f"Created {all_questions_file} with {len(unique_questions)} unique questions")
                
                # Update all_questions variable for questionnaire generation
                all_questions = unique_questions
            else:
                logging.warning("No document analysis found and skipping analysis. Cannot generate questions.")
                all_questions = []
    
    # Step 4: Generate Somalia questionnaire
    from questionnaire_generator import QuestionnaireGenerator
    
    questionnaire_file = os.path.join(args.output_dir, "somalia_ai_readiness_questionnaire.json")
    
    if not args.skip_questionnaire and not os.path.exists(questionnaire_file):
        logging.info("Step 4: Generating Somalia-specific questionnaire...")
        questionnaire_gen = QuestionnaireGenerator(api_key, model_name=args.model)
        
        # Check if we have questions to work with
        if not all_questions:
            logging.warning("No questions available for questionnaire generation")
            # Create minimal questionnaire with placeholder
            questionnaire = {
                "title": "Somalia AI Readiness Assessment",
                "description": "Questionnaire to assess Somalia's readiness for AI adoption",
                "sections": [
                    {
                        "title": "General Assessment",
                        "description": "No questions available - please run information extraction first",
                        "questions": []
                    }
                ]
            }
        else:
            # Generate questionnaire from available questions
            questionnaire = questionnaire_gen.generate_somalia_questionnaire(
                all_questions,
                questionnaire_file
            )
        
        # Generate human-readable version
        questionnaire_gen.generate_human_readable_version(
            questionnaire,
            os.path.join(args.output_dir, "somalia_ai_readiness_questionnaire.md")
        )
    else:
        logging.info("Skipping questionnaire generation...")
    
    logging.info(f"Process complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()