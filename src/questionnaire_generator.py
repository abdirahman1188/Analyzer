import json
import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# Import the GeminiAPI utility
from api_utils import GeminiAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class QuestionnaireGenerator:
    """Generate a Somalia-specific AI readiness questionnaire using Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-pro"):
        """
        Initialize the questionnaire generator.
        
        Args:
            api_key: Google API key for Gemini API access
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize API utility
        self.api = GeminiAPI(api_key, model_name)
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from Gemini API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Generated response
        """
        return self.api.generate_content(prompt, temperature=0.7)
    
    def generate_somalia_questionnaire(self, 
                                      questions: List[Dict[str, Any]], 
                                      output_path: str) -> Dict[str, Any]:
        """
        Generate a Somalia-specific AI readiness questionnaire.
        
        Args:
            questions: List of questions extracted from papers
            output_path: Path to save the generated questionnaire
            
        Returns:
            Generated questionnaire
        """
        # Format questions for the prompt
        question_examples = []
        categories = {}
        
        # Group questions by category for the prompt
        for q in questions[:200]:  # Limit to 200 questions to avoid token limits
            category = q.get("category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(q.get("text", ""))
        
        # Format grouped questions
        for category, q_list in categories.items():
            question_examples.append(f"Category: {category}")
            for i, q in enumerate(q_list[:10]):  # Take up to 10 questions per category
                question_examples.append(f"  - {q}")
        
        formatted_questions = "\n".join(question_examples)
        
        somalia_context = """
        Somalia Context Information:
        - Somalia is recovering from decades of civil war and instability.
        - Limited infrastructure with only about 36% of the population having access to electricity.
        - Internet penetration is around 12-15% of the population, mostly in urban areas.
        - Mobile penetration is higher, with about 50-60% of the population having access to mobile phones.
        - Limited AI talent pool and tech education opportunities.
        - Few AI startups or private sector investments in technology.
        - Government digital infrastructure is in early development stages.
        - Challenges include security issues, limited regulatory frameworks, and infrastructure gaps.
        - Opportunities include a young population, growing mobile adoption, and potential for technological leapfrogging.
        - Potential sectors for AI application include agriculture, healthcare, security, and governance.
        """
        
        prompt = f"""
        You are tasked with creating a comprehensive AI readiness assessment questionnaire specifically for Somalia. This questionnaire will help assess Somalia's readiness for AI adoption and implementation. Use the examples of questions from other AI readiness assessments as inspiration, but adapt them to Somalia's specific context.

        {somalia_context}
        
        Examples of questions from other AI readiness assessments:
        {formatted_questions}
        
        Create a comprehensive AI readiness questionnaire for Somalia with the following:
        
        1. A title and brief introduction explaining the purpose of the assessment
        2. 6-7 sections covering different aspects of AI readiness, such as:
           - Infrastructure and Data
           - Government and Policy
           - Skills and Education
           - Private Sector Development
           - Ethics and Inclusion
           - Sectoral Applications
        3. Each section should have 5-8 questions
        4. Include a mix of question types (specify for each question):
           - Likert scale (1-5 agreement)
           - Multiple choice
           - Open-ended questions
        5. Questions must be culturally appropriate and relevant to Somalia's context
        6. Consider Somalia's specific challenges and opportunities
        
        Format your response as a valid JSON object with the following structure:
        {{
            "title": "Questionnaire title",
            "introduction": "Introduction text explaining purpose and how to use the assessment",
            "sections": [
                {{
                    "name": "Section name",
                    "description": "Brief description of this section's focus",
                    "questions": [
                        {{
                            "text": "Question text",
                            "type": "Likert/Multiple Choice/Open-ended",
                            "options": ["Option 1", "Option 2"] (include only for multiple choice questions)
                        }}
                    ]
                }}
            ]
        }}
        
        Ensure your response is ONLY the JSON object, with no additional text before or after.
        """
        
        logging.info("Generating Somalia-specific questionnaire...")
        response_text = self.generate_response(prompt)
        
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                questionnaire = json.loads(json_str)
                
                # Save the questionnaire
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(questionnaire, f, indent=2, ensure_ascii=False)
                
                logging.info(f"Somalia AI readiness questionnaire saved to {output_path}")
                return questionnaire
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logging.error(f"Error generating questionnaire: {e}")
            
            # Try to save the raw response for debugging
            debug_path = output_path.replace('.json', '_raw.txt')
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
                
            logging.info(f"Saved raw response to {debug_path}")
            return {"error": "Failed to parse questionnaire", "debug_file": debug_path}
    
    def generate_human_readable_version(self, questionnaire: Dict[str, Any], output_path: str) -> None:
        """
        Generate a human-readable version of the questionnaire in Markdown.
        
        Args:
            questionnaire: Questionnaire dictionary
            output_path: Path to save the Markdown version
        """
        if "error" in questionnaire:
            logging.error("Cannot generate readable version due to questionnaire error")
            return
            
        markdown = f"# {questionnaire.get('title', 'AI Readiness Assessment for Somalia')}\n\n"
        markdown += questionnaire.get('introduction', '') + "\n\n"
        
        for i, section in enumerate(questionnaire.get('sections', [])):
            markdown += f"## Section {i+1}: {section.get('name', 'Unnamed Section')}\n\n"
            markdown += section.get('description', '') + "\n\n"
            
            for j, question in enumerate(section.get('questions', [])):
                markdown += f"{j+1}. **{question.get('text', '')}**\n"
                markdown += f"   *Type: {question.get('type', 'Not specified')}*\n"
                
                if 'options' in question and question['options']:
                    markdown += "   Options:\n"
                    for option in question['options']:
                        markdown += f"   - {option}\n"
                        
                markdown += "\n"
            
            markdown += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logging.info(f"Human-readable questionnaire saved to {output_path}")

if __name__ == "__main__":
    # Test implementation
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    parser = argparse.ArgumentParser(description="Generate Somalia-specific AI readiness questionnaire")
    parser.add_argument("--analysis_dir", default="results/analysis", help="Directory containing analysis")
    parser.add_argument("--output_dir", default="results", help="Directory to save questionnaire")
    
    args = parser.parse_args()
    
    # Load extracted questions
    questions_file = os.path.join(args.analysis_dir, "all_questions.json")
    
    if not os.path.exists(questions_file):
        raise ValueError(f"Questions file not found: {questions_file}")
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Generate questionnaire
    questionnaire_gen = QuestionnaireGenerator(api_key)
    
    questionnaire = questionnaire_gen.generate_somalia_questionnaire(
        questions,
        os.path.join(args.output_dir, "somalia_ai_readiness_questionnaire.json")
    )
    
    # Generate human-readable version
    questionnaire_gen.generate_human_readable_version(
        questionnaire,
        os.path.join(args.output_dir, "somalia_ai_readiness_questionnaire.md")
    )