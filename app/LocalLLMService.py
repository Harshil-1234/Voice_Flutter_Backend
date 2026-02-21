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

"""
LocalLLMService: Manages local LLM inference via llama-cpp-python for text analysis.
"""

import logging
import threading
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration
HF_REPO_ID = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILENAME = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
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
            # Local import to avoid failing module import if llama-cpp-python is not installed
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download

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
        except ImportError as e:
            logger.error("llama-cpp-python is not installed or could not be imported.")
            raise
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

        # Use the UPSC Analysis prompt for all non-garbage articles.
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

    def judge_argument(self, statement: str, argument: str) -> dict:
        """
        AI Judge function using local Llama-3-8B.
        Scores the argument 1-10 based on logic, facts, and lack of fallacies.
        """
        self._check_model_loaded()

        system_prompt = """You are an unbiased Logic Professor. 
Your task is to score the user's argument between 1 and 10 based purely on facts, logic, and lack of logical fallacies. 
Do not agree or disagree with the conclusion; evaluate only the reasoning.

Respond ONLY with a valid JSON object in the exact following format, without any markdown formatting or extra text:
{
    "score": <int 1-10>,
    "feedback": "<short string explanation of the score>",
    "key_factors": ["<factor 1>", "<factor 2>"],
    "detailed_analysis": "<long string detailed analysis of the argument's strengths and weaknesses>"
}"""

        user_prompt = f"Topic Statement: {statement}\nUser Argument: {argument}"
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            with self.lock:
                response = self.model.create_chat_completion(
                    messages=[{"role": "user", "content": combined_prompt}],
                    temperature=0.2, # Low temperature for more analytical/consistent scoring
                    top_p=0.95,
                    max_tokens=512,
                    response_format={"type": "json_object"}
                )
            
            response_content = response["choices"][0]["message"]["content"].strip()
            if not response_content:
                 raise ValueError("Empty response from local LLM.")
                 
            # Llama-cpp might still wrap it in markdown block, so we strip it if needed
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
                
            return json.loads(response_content.strip())
            
        except Exception as e:
            logger.error(f"Error during judge_argument local inference: {e}")
            # Fallback response in case of error
            return {
                "score": 5,
                "feedback": "AI analysis temporarily unavailable due to a server error.",
                "key_factors": ["Could not parse argument logically."],
                "detailed_analysis": "An error occurred while the Logic Professor was analyzing your argument. Please try again."
            }

    def generate_debate_topic(self, articles_text: str) -> dict:
        """Generates a debate topic statement from recent Indian politics articles."""
        self._check_model_loaded()
        system_prompt = """You are a highly analytical political journalist in India. 
Based on the provided recent news headlines and summaries, identify ONE highly debatable, controversial topic. 
Frame a clear, compelling Statement (e.g., 'India should ban algorithmic social feeds') and provide a brief context.

Respond ONLY with a valid JSON object in the exact following format:
{
    "statement": "<The controversial statement>",
    "context": "<1-2 sentences explaining why this is currently debated>"
}"""
        try:
            with self.lock:
                response = self.model.create_chat_completion(
                    messages=[{"role": "user", "content": f"{system_prompt}\n\nNews Data:\n{articles_text}"}],
                    temperature=0.4,
                    max_tokens=256,
                    response_format={"type": "json_object"}
                )
            content = response["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Error generating debate topic: {e}")
            return {
                "statement": "The implementation of simultaneous elections (One Nation, One Election) will strengthen Indian democracy.",
                "context": "Simultaneous elections are being proposed to reduce election expenditure and policy paralysis."
            }

    def conclude_debate(self, statement: str, winning_side: str, top_args: str) -> str:
        """Summarizes why the winning side won based on top arguments."""
        self._check_model_loaded()
        system_prompt = f"""You are a debate moderator. The debate statement was: '{statement}'.
The winning side was: '{winning_side}'.
Based on the following top arguments from the winning side, write a concise, compelling 2-3 sentence conclusion summarizing why they won.
Do not use JSON. Write plain text."""

        try:
            with self.lock:
                response = self.model.create_chat_completion(
                    messages=[{"role": "user", "content": f"{system_prompt}\n\nTop Arguments:\n{top_args}"}],
                    temperature=0.3,
                    max_tokens=256
                )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error concluding debate: {e}")
            return "The winning side presented stronger logical points and practical evidence to support their stance."


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

def judge_argument(statement: str, argument: str) -> dict:
    service = get_local_llm_service()
    return service.judge_argument(statement, argument)

def generate_debate_topic(articles_text: str) -> dict:
    service = get_local_llm_service()
    return service.generate_debate_topic(articles_text)

def conclude_debate(statement: str, winning_side: str, top_args: str) -> str:
    service = get_local_llm_service()
    return service.conclude_debate(statement, winning_side, top_args)

