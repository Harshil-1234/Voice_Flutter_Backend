"""
LocalLLMService: Manages local LLM inference via llama-cpp-python for text analysis.
- Uses llama-cpp-python to load Gemma-2-2b-it GGUF model directly in Python process
- Analyzes articles for summary + UPSC relevance + tagging
- CPU-only inference (n_gpu_layers=0)

Prerequisites:
1. Install llama-cpp-python: pip install llama-cpp-python
2. Model is auto-downloaded from Hugging Face on first run
3. First inference will take longer due to model initialization

Model Details:
- Name: Gemma-2-2b-it (quantized Q4_K_M)
- Source: https://huggingface.co/second-state/Gemma-2-2b-it-GGUF
- Size: ~1.3GB
- Context: 4096 tokens
"""

import ast
import json
import logging
import os
import re
from typing import Dict, Optional
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is not installed. "
        "Install with: pip install llama-cpp-python"
    )

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Model configuration
HF_REPO_ID = "bartowski/gemma-2-2b-it-GGUF"
MODEL_FILENAME = "gemma-2-2b-it-Q4_K_M.gguf"
CACHE_DIR = Path.home() / ".cache" / "llm_models"
CONTEXT_SIZE = 4096
N_GPU_LAYERS = 0  # CPU only
N_THREADS = 2  # Adjust based on your CPU core count


class LocalLLMService:
    """Encapsulates local LLM inference via llama-cpp-python for article analysis."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the LocalLLMService (singleton)."""
        if self._initialized:
            return
        
        self._initialized = True
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Download and load the Gemma-2-2b-it GGUF model."""
        try:
            logger.info(f"Loading {MODEL_FILENAME} from {HF_REPO_ID}...")
            logger.info(f"Cache directory: {CACHE_DIR}")
            logger.info(f"Threads: {N_THREADS}, GPU layers: {N_GPU_LAYERS}, Context: {CONTEXT_SIZE}")
            
            # Ensure cache directory exists
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Cache directory ready: {CACHE_DIR}")
            
            # Download model from Hugging Face if not cached
            logger.info(f"Starting model download from HuggingFace...")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                cache_dir=str(CACHE_DIR),
                resume_download=True
            )
            
            logger.info(f"✅ Model file ready at: {model_path}")
            logger.info(f"   File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
            
            # Load model with llama-cpp-python
            logger.info(f"Initializing llama-cpp-python with model...")
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=N_GPU_LAYERS,  # CPU only
                n_ctx=CONTEXT_SIZE,
                n_threads=N_THREADS,
                verbose=False  # Set to True for debug logs
            )
            
            logger.info(
                f"✅ Model loaded successfully! "
                f"Ready for inference."
            )
        
        except ImportError as e:
            logger.error(f"❌ ImportError: {e}")
            logger.error("This likely means llama-cpp-python is not installed.")
            logger.error("Install with: pip install llama-cpp-python")
            raise RuntimeError(
                f"llama-cpp-python import failed: {e}\n"
                "Install with: pip install llama-cpp-python"
            ) from e
        
        except Exception as e:
            logger.error(f"❌ Error loading model: {type(e).__name__}: {e}")
            import traceback
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise RuntimeError(
                f"Failed to load Gemma-2-2b-it model: {e}\n"
                f"Error type: {type(e).__name__}\n"
                "Make sure you have llama-cpp-python installed:\n"
                "pip install llama-cpp-python"
            ) from e
    
    def _check_model_loaded(self):
        """Verify model is loaded before inference."""
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Check logs for loading errors."
            )
    
    def analyze_article(self, text: str, max_length: int = 2048) -> Dict:
        """
        Analyze article text for summary, UPSC relevance, and tags.
        
        Args:
            text: Article content to analyze
            max_length: Truncate input to this length (characters)
        
        Returns:
            Dict with keys:
            - summary (str): 80-word concise summary
            - upsc_relevant (bool): True if relevant to UPSC Civil Services
            - tags (List[str] | None): UPSC tags if relevant, else None
        
        Returns empty dict on error.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to analyze_article.")
            return {
                "summary": "",
                "upsc_relevant": False,
                "tags": None
            }
        
        self._check_model_loaded()
        
        # Truncate text to save tokens
        text = text[:max_length]
        
        system_prompt = (
            "You are a strict UPSC Civil Services Exam Gatekeeper. Analyze this news article.\n\n"
            "Criteria for 'upsc_relevant':\n"
            "✅ TRUE ONLY IF: It involves National Policy (Govt Schemes), Supreme Court Rulings, Economy (RBI/GDP), International Relations (G20/UN), or Science/Environment (ISRO/Climate).\n"
            "❌ FALSE IF: It is Local Crime, Accidents, Sports results, Celebrity/Movie news, or partisan political rallies.\n\n"
            "Your Task:\n"
            "1. Write a 80-word concise summary.\n"
            "2. Determine boolean 'upsc_relevant' based on strict criteria above.\n"
            "3. If True, add tags (e.g., ['GS-2', 'Polity']). If False, set tags: null.\n"
            "4. Return ONLY valid JSON: {'summary': '...', 'upsc_relevant': bool, 'tags': [...]}"
        )
        
        try:
            # FIX: Combine System + User into one prompt block to avoid "System role not supported" error
            combined_prompt = f"{system_prompt}\n\nArticle Text:\n{text}"
            
            # Call llama-cpp-python for inference using ONLY 'user' role
            response = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": combined_prompt
                    }
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=512
            )
            
            # Extract response text
            response_text = response["choices"][0]["message"]["content"].strip()
            
            if not response_text:
                logger.warning("Empty response from model.")
                return {}
            
            # Parse JSON from response
            result = self._parse_json_response(response_text)
            
            # Validate structure
            if self._validate_result(result):
                return result
            else:
                logger.warning(f"Invalid response structure: {result}")
                return {}
        
        except Exception as e:
            logger.error(f"Error during analyze_article: {e}")
            return {}
    
    @staticmethod
    def _parse_json_response(response_text: str) -> Dict:
        """
        Extract and parse JSON from response text.
        
        Handles:
        1. Markdown code blocks (```json ... ```)
        2. Python-style dictionaries with single quotes
        3. Standard JSON with double quotes
        4. Regex extraction of {...} blocks as fallback
        
        Returns:
            Parsed dictionary or empty dict if parsing fails
        """
        if not response_text or not response_text.strip():
            logger.warning("Empty response text provided to _parse_json_response.")
            return {}
        
        # Step 1: Strip Markdown code blocks
        cleaned_text = response_text.strip()
        
        # Remove ```json ... ``` or ``` ... ```
        if "```" in cleaned_text:
            # Match ```json...``` or ```...```
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()
        
        # Step 2: Try standard JSON parsing (handles double quotes)
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.debug(f"Standard JSON parsing failed: {e}")
        
        # Step 3: Try ast.literal_eval (handles single quotes and Python-style dicts)
        try:
            result = ast.literal_eval(cleaned_text)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError) as e:
            logger.debug(f"ast.literal_eval failed: {e}")
        
        # Step 4: Fallback - regex extract {...} block and retry
        try:
            # Find the first { and last } and extract that substring
            match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if match:
                extracted = match.group(0).strip()
                
                # Try JSON parsing on extracted block
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass
                
                # Try ast.literal_eval on extracted block
                try:
                    result = ast.literal_eval(extracted)
                    if isinstance(result, dict):
                        return result
                except (ValueError, SyntaxError):
                    pass
        except Exception as e:
            logger.debug(f"Regex extraction fallback failed: {e}")
        
        # All parsing attempts failed
        logger.warning(
            f"Could not parse JSON from response. "
            f"Text (first 300 chars): {response_text[:300]}"
        )
        return {}
    
    @staticmethod
    def _validate_result(result: Dict) -> bool:
        """Validate result has required keys."""
        required_keys = {"summary", "upsc_relevant", "tags"}
        if not isinstance(result, dict):
            return False
        
        # Check all required keys exist
        if not all(k in result for k in required_keys):
            return False
        
        # Type checks
        if not isinstance(result.get("summary"), str):
            return False
        if not isinstance(result.get("upsc_relevant"), bool):
            return False
        tags = result.get("tags")
        if tags is not None and not isinstance(tags, list):
            return False
        
        return True
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
            logger.info("Model cleaned up.")


# Convenience functions
_service = None


def get_local_llm_service() -> LocalLLMService:
    """Get or create the singleton LocalLLMService instance."""
    global _service
    if _service is None:
        _service = LocalLLMService()
    return _service


def analyze_article(text: str) -> Dict:
    """Wrapper function to analyze article."""
    service = get_local_llm_service()
    return service.analyze_article(text)
