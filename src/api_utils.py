import logging
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from google import genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GeminiAPI:
    """Utility class for making requests to the Gemini API with proper error handling."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the API utility.
        
        Args:
            api_key: Google API key for Gemini API access
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key
        
        # Always use gemini-2.0-flash regardless of what was passed
        # This ensures we use a model that works with the current API
        self.model_name = "gemini-2.0-flash"
        
        if model_name != "gemini-2.0-flash":
            logger.info(f"Using gemini-2.0-flash instead of {model_name}")
        
        # Initialize the API client
        self.client = genai.Client(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.0, min=1, max=60),
        retry=retry_if_exception_type((Exception)),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def generate_content(self, prompt: str, temperature: float = None) -> str:
        """
        Generate content using Gemini with retry logic.
        
        Args:
            prompt: Prompt text to send to the model
            temperature: Optional temperature parameter
            
        Returns:
            Generated text response
        """
        try:
            # Add a small random delay to avoid rate limits
            time.sleep(random.uniform(0.2, 1.0))
            
            # Configure generation parameters
            config = None
            if temperature is not None:
                # Import types only if needed
                from google.genai import types
                config = types.GenerateContentConfig(temperature=temperature)
            
            # Log the model being used
            logger.info(f"Generating content with model: {self.model_name}")
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=config
            )
            
            # Extract text from response
            return response.text
                
        except Exception as e:
            # Check for rate limit errors
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(f"Rate limit hit: {str(e)}")
                # Add extra delay before retry
                time.sleep(5 + random.uniform(1.0, 3.0))
            else:
                logger.error(f"Error generating content: {str(e)}")
            
            # Re-raise for retry mechanism
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=30),
        retry=retry_if_exception_type((Exception)),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def generate_embedding(self, text: str):
        """
        Generate an embedding for text with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding result from the API
        """
        try:
            # Add a small random delay to avoid rate limits
            time.sleep(random.uniform(0.2, 1.0))
            
            # Truncate text if too long (embedding models have token limits)
            if len(text) > 10000:
                text = text[:10000]
            
            # Generate embedding using the models.embed_content method 
            # as shown in the latest documentation
            embedding_model = "text-embedding-004"  # Use the newer model
            
            # Create embedding config if needed
            config = None
            try:
                from google.genai import types
                config = types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            except ImportError:
                logger.warning("Could not import types.EmbedContentConfig")
            
            # Generate embedding
            result = self.client.models.embed_content(
                model=embedding_model,
                contents=text,
                config=config
            )
            
            # For debugging
            if not hasattr(self, '_debug_printed'):
                logger.info(f"Embedding response keys: {dir(result)}")
                self._debug_printed = True
            
            # Extract embedding values according to the documentation
            if hasattr(result, 'embeddings') and result.embeddings:
                # For batch request
                embedding_vector = result.embeddings[0].values
            elif hasattr(result, 'embedding') and hasattr(result.embedding, 'values'):
                # For single request
                embedding_vector = result.embedding.values
            else:
                # Fallback to zero vector if we can't extract
                logger.warning("Could not extract embedding from response. Using zero vector.")
                embedding_vector = [0.0] * 768
            
            # Return in the expected format for vector store
            return {
                "embedding": embedding_vector,
                "text": text[:100] + "..." if len(text) > 100 else text
            }
            
        except Exception as e:
            # Check for rate limit errors
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(f"Rate limit hit during embedding: {str(e)}")
                # Add extra delay before retry
                time.sleep(5 + random.uniform(1.0, 3.0))
            else:
                logger.error(f"Error generating embedding: {str(e)}")
            
            # Re-raise for retry mechanism
            raise