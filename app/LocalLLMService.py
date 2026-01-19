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
import threading
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
# Gemma supports up to 8k, but 4k is safe and covers long articles
CONTEXT_SIZE = 4096
N_GPU_LAYERS = 0         # CPU only
N_THREADS = 3            # Leave 1 vCPU free for the API/Database
N_BATCH = 512            # Process prompt in smaller chunks to prevent RAM spikes


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

        self.lock = threading.Lock()
        self._initialized = True
        self.model = None
        self._load_model()

    def _load_model(self):
        """Download and load the Gemma-2-2b-it GGUF model."""
        try:
            logger.info(f"Loading {MODEL_FILENAME} from {HF_REPO_ID}...")
            logger.info(f"Cache directory: {CACHE_DIR}")
            logger.info(
                f"Threads: {N_THREADS}, GPU layers: {N_GPU_LAYERS}, Context: {CONTEXT_SIZE}")

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
            logger.info(
                f"   File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")

            # Load model with llama-cpp-python
            logger.info(f"Initializing llama-cpp-python with model...")
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=N_GPU_LAYERS,
                n_ctx=CONTEXT_SIZE,
                n_threads=N_THREADS,
                n_batch=N_BATCH,    # Crucial for stability
                verbose=False       # Set to True only if debugging crashes
            )

            logger.info(
                f"✅ Model loaded successfully! "
                f"Ready for inference."
            )

        except ImportError as e:
            logger.error(f"❌ ImportError: {e}")
            logger.error(
                "This likely means llama-cpp-python is not installed.")
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

        Implements "Enhance Mode" for short articles:
        - Tier 3 (< 30 words): Reject as garbage (headline-only)
        - Tier 1 (30-300 words): Enhance mode - rewrite as professional brief
        - Tier 2 (> 300 words): Full analysis - UPSC relevance + summary + tags

        Args:
            text: Article content to analyze
            max_length: Truncate input to this length (characters)

        Returns:
            Dict with keys:
            - summary (str): Enhanced brief or concise summary
            - upsc_relevant (bool): True if relevant to UPSC Civil Services
            - tags (List[str] | None): UPSC tags if relevant, else None

        Returns empty dict on error or for garbage content (< 30 words).
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

        # Count words to determine processing tier
        word_count = len(text.split())
        logger.info(f"Article word count: {word_count}")

        # TIER 3: Garbage (< 30 words) - Reject
        if word_count < 30:
            logger.warning(
                f"Article too short ({word_count} words). Rejecting as garbage.")
            return {}

        # TIER 1: Short/Snippet (30-300 words) - Enhance Mode
        if 30 <= word_count <= 300:
            logger.info(
                f"Short article ({word_count} words). Using Enhance Mode.")
            system_prompt = (
                "You are a News Editor. The input is a short news snippet. Your task:\n\n"
                "1. Rewrite it into a smooth, professional news brief (80-120 words).\n"
                "2. Remove any HTML artifacts, 'read more' text, or formatting noise.\n"
                "3. Do NOT just repeat the headline. Make it sound like a complete story.\n"
                "4. Evaluate if it's relevant to UPSC Civil Services (National Policy, Court Rulings, Economy, International Relations, Science/Environment).\n"
                "5. Return ONLY valid JSON with DOUBLE QUOTES:\n"
                "   {\"summary\": \"...\", \"upsc_relevant\": true/false, \"tags\": [\"GS-x\", \"Topic\"] or null}\n\n"
                "IMPORTANT: Use DOUBLE QUOTES (\") for all keys and values. No single quotes."
            )

        # TIER 2: Full Article (> 300 words) - Full Analysis
        else:
            logger.info(
                f"Full article ({word_count} words). Using UPSC Analysis Mode.")
            system_prompt = (
            """
                You are a senior editor for a news app that serves both general readers and UPSC aspirants. Your goal is to provide balanced summaries and filter news for UPSC relevance based on direct impact on India.

                ### 1. EVALUATION CRITERIA (UPSC_RELEVANT)
                •⁠  ⁠TRUE ONLY IF: The news involves National Policy (Govt Schemes), Supreme Court/Judicial Rulings, Indian Economy (RBI/GDP/Budget), International Relations (specifically involving India or impacting India's strategic interests), Science/Environment (ISRO/Climate/Pollution), Internal Security (Defense/Terrorism in India), or Governance.
                •⁠  ⁠FALSE IF: It is local crime, accidents, sports, celebrity news, partisan political rallies, or internal affairs of foreign countries (e.g., US local policy, UK elections) that have NO direct strategic, economic, or diplomatic consequence for India.

                ### 2. TAGGING DICTIONARY
                If 'upsc_relevant' is true, you MUST use ONLY these tags from this fixed list:
                •⁠  ⁠Papers: ["GS-1", "GS-2", "GS-3", "GS-4"]
                •⁠  ⁠Categories: ["Polity", "Economy", "IR", "Environment", "Science", "Security", "Geography", "History", "Society", "Ethics"]
                If 'upsc_relevant' is false, return 'tags' as an empty array [].

                ### 3.  MANDATORY TASKS
                1.⁠ ⁠summary: Write a 80-word concise summary..
                2.⁠ ⁠upsc_relevant: Set to true or false.
                3.⁠ ⁠tags: An array of strings from the Tagging Dictionary. If false, return [].

                ### 4. OUTPUT FORMAT (STRICT JSON)
                Return ONLY a valid JSON object. Do not include markdown code blocks (```json), no preambles, and no conversational text.
                **IMPORTANT: Use DOUBLE QUOTES (\") for all keys and strings.** Do not use single quotes.\n"
                "   ✅ GOOD: {\"summary\": \"India's economy...\", \"upsc_relevant\": true}\n"
                "   ❌ BAD: {'summary': 'India's economy...', 'upsc_relevant': True}\n\n"
                {
                    "summary": "string",
                    "upsc_relevant": boolean,
                    "tags": ["string"]
                }
            """
            )

        try:
            # Combine System + User into one prompt block
            combined_prompt = f"{system_prompt}\n\nArticle Text:\n{text}"

            # CRITICAL FIX: Lock ensures only one thread accesses the model at a time
            # This prevents segmentation faults from concurrent C++ inference calls
            with self.lock:
                response = self.model.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": combined_prompt
                        }
                    ],
                    temperature=0.2,   # Low temp = Strict JSON, less hallucination
                    top_p=0.95,        # Slightly higher than 0.9 to allow natural summaries
                    max_tokens=1024,   # CRITICAL: Increased from 512 to prevent JSON cutoff
                    repeat_penalty=1.1  # Prevents the AI from repeating sentences
                )

            # Extract response text
            response_text = response["choices"][0]["message"]["content"].strip(
            )

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
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Tries to find JSON. If it fails, treats the entire response as a summary.
        This prevents 'KeyError: summary' in your UI.
        """
        # 1. Look for a JSON-like structure in the text
        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if match:
            json_str = match.group(0).strip()
            
            # Simple repair for truncated text
            if not json_str.endswith("}"):
                json_str += "}"

            # Try Standard JSON (Double Quotes)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # Try Python Literal (Single Quotes - Common with Gemma-2)
            try:
                import ast
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                pass

        # 2. FINAL FALLBACK: If we reach here, no JSON was found or parsed.
        # We treat the entire raw output as a summary to avoid crashing the app.
        logger.warning("No valid JSON found. Using raw response as summary fallback.")
        
        # Remove common markdown fluff if present
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        return {
            "summary": clean_text,
            "upsc_relevant": False, # Assume false if it failed to format
            "tags": []
        }

    @staticmethod
    def _validate_result(result: Dict) -> bool:
        """
        Validate result has required keys and correct types.
        Automatically sets missing 'tags' to None if not provided.
        """
        required_keys = {"summary", "upsc_relevant"}
        if not isinstance(result, dict):
            return False

        # Check all required keys exist (tags is optional, defaults to None)
        if not all(k in result for k in required_keys):
            return False

        # Type checks
        if not isinstance(result.get("summary"), str):
            return False
        if not isinstance(result.get("upsc_relevant"), bool):
            return False

        # Tags validation: if missing, set to None; if present, must be list or None
        if "tags" not in result:
            result["tags"] = None
        else:
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
