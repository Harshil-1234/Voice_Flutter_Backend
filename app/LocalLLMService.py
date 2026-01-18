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

import json
import logging
import os
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
N_THREADS = 4  # Adjust based on your CPU core count


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
            - summary (str): 60-word concise summary
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
            "You are a UPSC Exam Expert. Analyze this news article.\n\n"
            "Your task:\n"
            "1. Write a 60-word concise summary (factual, no opinions).\n"
            "2. Evaluate relevance to UPSC Civil Services Syllabus (GS-1, GS-2, GS-3, GS-4).\n"
            "3. Return ONLY valid JSON (no other text):\n"
            "{'summary': '...', 'upsc_relevant': bool, 'tags': ['GS-x', 'Topic']}\n\n"
            "If irrelevant (Sports, Entertainment, Local Crime, Partisan Politics), "
            "set upsc_relevant: false and tags: null.\n"
            "If relevant, set upsc_relevant: true and tags: list of applicable tags."
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
        Extract JSON from response text. Handles cases where model returns
        extra text before/after JSON.
        """
        try:
            # Try direct parse first
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"Could not parse JSON from response: {response_text[:200]}")
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
