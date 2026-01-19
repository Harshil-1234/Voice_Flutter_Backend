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
            """
            LocalLLMService: Manages local LLM inference via llama-cpp-python for text analysis.
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
                    "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
                )

            from huggingface_hub import hf_hub_download

            logger = logging.getLogger(__name__)

            # Model configuration
            HF_REPO_ID = "bartowski/gemma-2-2b-it-GGUF"
            MODEL_FILENAME = "gemma-2-2b-it-Q4_K_M.gguf"
            CACHE_DIR = Path.home() / ".cache" / "llm_models"
            CONTEXT_SIZE = 4096
            N_GPU_LAYERS = 0
            N_THREADS = 3
            N_BATCH = 512


            class LocalLLMService:
                """Encapsulates local LLM inference via llama-cpp-python for article analysis."""

                _instance = None

                def __new__(cls):
                    if cls._instance is None:
                        cls._instance = super().__new__(cls)
                        cls._instance._initialized = False
                    return cls._instance

                def __init__(self):
                    if self._initialized:
                        return
                    self.lock = threading.Lock()
                    self._initialized = True
                    self.model = None
                    self._load_model()

                def _load_model(self):
                    try:
                        logger.info(f"Loading {MODEL_FILENAME} from {HF_REPO_ID}...")
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        model_path = hf_hub_download(
                            repo_id=HF_REPO_ID, filename=MODEL_FILENAME, cache_dir=str(CACHE_DIR), resume_download=True
                        )
                        self.model = Llama(
                            model_path=model_path,
                            n_gpu_layers=N_GPU_LAYERS,
                            n_ctx=CONTEXT_SIZE,
                            n_threads=N_THREADS,
                            n_batch=N_BATCH,
                            verbose=False,
                        )
                        logger.info("Model loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}")
                        raise

                def _check_model_loaded(self):
                    if self.model is None:
                        raise RuntimeError("Model not loaded. Check logs for loading errors.")

                def analyze_article(self, text: str, max_length: int = 2048) -> Dict:
                    """Analyze article and return dict with keys: summary, upsc_relevant, tags."""
                    if not text or not text.strip():
                        logger.warning("Empty text provided to analyze_article.")
                        return {"summary": "", "upsc_relevant": False, "tags": None}

                    self._check_model_loaded()
                    text = text[:max_length]
                    word_count = len(text.split())
                    logger.info(f"Article word count: {word_count}")

                    if word_count < 30:
                        logger.warning(f"Article too short ({word_count} words). Rejecting as garbage.")
                        return {}

                    if 30 <= word_count <= 300:
                        logger.info(f"Short article ({word_count} words). Using Enhance Mode.")
                        system_prompt = (
                            "You are a News Editor. The input is a short news snippet. Your task:\n\n"
                            "1. Rewrite it into a smooth, professional news brief (60 words).\n"
                            "2. Remove any HTML artifacts, 'read more' text, or formatting noise.\n"
                            "3. Do NOT just repeat the headline. Make it sound like a complete story.\n"
                            "4. Evaluate if it's relevant to UPSC Civil Services (National Policy, Court Rulings, Economy, International Relations, Science/Environment).\n\n"
                            "### OUTPUT FORMAT (Strictly Follow This):\n"
                            "SUMMARY: [Write the 60-word summary here]\n"
                            "UPSC_RELEVANT: [TRUE or FALSE]\n"
                            "TAGS: [Tag1, Tag2, Tag3] (or NONE)\n\n"
                            "Return ONLY the text in the format above; do NOT return JSON or code fences."
                        )
                    else:
                        logger.info(f"Full article ({word_count} words). Using UPSC Analysis Mode.")
                        system_prompt = (
                            "You are a senior editor for a news app that serves both general readers and UPSC aspirants. Your goal is to provide balanced summaries and filter news for UPSC relevance based on direct impact on India.\n\n"
                            "### 1. EVALUATION CRITERIA (UPSC_RELEVANT)\n"
                            "TRUE ONLY IF: The news involves National Policy (Govt Schemes), Supreme Court/Judicial Rulings, Indian Economy (RBI/GDP/Budget), International Relations (specifically involving India or impacting India's strategic interests), Science/Environment (ISRO/Climate/Pollution), Internal Security (Defense/Terrorism in India), or Governance.\n"
                            "FALSE IF: It is local crime, accidents, sports, celebrity news, partisan political rallies, or internal affairs of foreign countries that have NO direct strategic, economic, or diplomatic consequence for India.\n\n"
                            "### 2. TAGGING DICTIONARY\n"
                            "If 'upsc_relevant' is true, you MUST use ONLY these tags from this fixed list: Papers: [\"GS-1\", \"GS-2\", \"GS-3\", \"GS-4\"] and Categories: [\"Polity\", \"Economy\", \"IR\", \"Environment\", \"Science\", \"Security\", \"Geography\", \"History\", \"Society\", \"Ethics\"]\n"
                            "If 'upsc_relevant' is false, return 'TAGS' as NONE.\n\n"
                            "### MANDATORY TASKS\n"
                            "1. summary: Write a 80-word concise summary.\n"
                            "2. UPSC_RELEVANT: TRUE or FALSE.\n"
                            "3. TAGS: Comma-separated tags from the Tagging Dictionary, or NONE.\n\n"
                            "### OUTPUT FORMAT (Strictly Follow This):\n"
                            "SUMMARY: [Write the 80-word summary here]\n"
                            "UPSC_RELEVANT: [TRUE or FALSE]\n"
                            "TAGS: [Tag1, Tag2, Tag3] (or NONE)\n\n"
                            "Return ONLY the text in the format above; do NOT return JSON or code fences."
                        )

                    try:
                        combined_prompt = f"{system_prompt}\n\nArticle Text:\n{text}"
                        with self.lock:
                            response = self.model.create_chat_completion(
                                messages=[{"role": "user", "content": combined_prompt}],
                                temperature=0.2,
                                top_p=0.95,
                                max_tokens=1024,
                                repeat_penalty=1.1,
                            )

                        response_text = response["choices"][0]["message"]["content"].strip()
                        if not response_text:
                            logger.warning("Empty response from model.")
                            return {}

                        result = self._parse_custom_format(response_text)
                        if self._validate_result(result):
                            return result
                        else:
                            logger.warning(f"Invalid response structure: {result}")
                            return {}

                    except Exception as e:
                        logger.error(f"Error during analyze_article: {e}")
                        return {}

                @staticmethod
                def _parse_custom_format(text: str) -> Dict:
                    result = {"summary": "", "upsc_relevant": False, "tags": []}
                    lines = text.split('\n')
                    current_section = None
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        if line.upper().startswith("SUMMARY:"):
                            current_section = "summary"
                            result["summary"] = line.split("SUMMARY:", 1)[1].strip()
                        elif line.upper().startswith("UPSC_RELEVANT:"):
                            val = line.split("UPSC_RELEVANT:", 1)[1].strip().upper()
                            result["upsc_relevant"] = (val == "TRUE")
                        elif line.upper().startswith("TAGS:"):
                            tags_str = line.split("TAGS:", 1)[1].strip()
                            if tags_str and "NONE" not in tags_str.upper():
                                result["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
                        elif current_section == "summary":
                            result["summary"] += " " + line

                    if not result["summary"]:
                        return {}
                    return result

                @staticmethod
                def _validate_result(result: Dict) -> bool:
                    required_keys = {"summary", "upsc_relevant"}
                    if not isinstance(result, dict):
                        return False
                    if not all(k in result for k in required_keys):
                        return False
                    if not isinstance(result.get("summary"), str):
                        return False
                    if not isinstance(result.get("upsc_relevant"), bool):
                        return False
                    if "tags" not in result:
                        result["tags"] = None
                    else:
                        tags = result.get("tags")
                        if tags is not None and not isinstance(tags, list):
                            return False
                    return True

                def cleanup(self):
                    if self.model:
                        del self.model
                        self.model = None
                        logger.info("Model cleaned up.")


            # Convenience functions
            _service = None


            def get_local_llm_service() -> LocalLLMService:
                global _service
                if _service is None:
                    _service = LocalLLMService()
                return _service


            def analyze_article(text: str) -> Dict:
                service = get_local_llm_service()
                return service.analyze_article(text)
                return service.analyze_article(text)
