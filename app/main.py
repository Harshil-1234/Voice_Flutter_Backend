import os
import hashlib
import time
import random
from typing import List, Optional, Tuple

from fastapi import FastAPI, Query, Path, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import json
from pydantic import BaseModel
from pathlib import Path as PPath
from dotenv import load_dotenv
from datetime import datetime, timedelta
# from transformers import pipeline  # Commented out - using Groq instead of BART
import trafilatura
from .riddle_generator import generate_daily_riddle, get_latest_riddle, check_today_riddle_exists

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Environment: load from .env if present (root or app folder)
_ENV_CANDIDATES = [
    PPath(__file__).resolve().parent.parent / ".env",  # voice_backend/.env
    PPath(__file__).resolve().parent / ".env",         # voice_backend/app/.env
]
for _p in _ENV_CANDIDATES:
    try:
        if _p.exists():
            load_dotenv(dotenv_path=_p, override=True)
            break
    except Exception:
        # Non-fatal; fall back to system env
        pass

# Read environment variables after attempting to load .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Support multiple summary API keys (comma-separated) for fallback/rotation
# Format: GROQ_SUMMARY_API_KEY=key1,key2,key3,key4,key5
# Falls back to GROQ_API_KEY if not set
_summary_keys_str = os.getenv("GROQ_SUMMARY_API_KEY") or GROQ_API_KEY
GROQ_SUMMARY_API_KEYS = [key.strip() for key in _summary_keys_str.split(",") if key.strip()]
if not GROQ_SUMMARY_API_KEYS and GROQ_API_KEY:
    GROQ_SUMMARY_API_KEYS = [GROQ_API_KEY]  # Fallback to main key

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
# Default batch size set to 20 per requirements
SUMMARIZE_BATCH_SIZE = int(os.getenv("SUMMARIZE_BATCH_SIZE", "20"))

# BART summarizer commented out - using Groq instead
# summarizer = pipeline(
#     "summarization",
#     model="facebook/bart-large-cnn",  # You can switch to smaller models below
#     tokenizer="facebook/bart-large-cnn",
#     device=-1  # CPU; use device=0 if you have GPU
# )

CATEGORIES = [
    "general",
    "technology",
    "sports",
    "science",
    "health",
    "entertainment",
    "world",
    "politics",
    "business",
    "environment",
]


class ArticleOut(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    url: str
    urlToImage: Optional[str] = None
    publishedAt: Optional[str] = None
    createdAt: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None


def supabase_client():
    from supabase import create_client, Client
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase env vars missing: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def md5_lower(text: str) -> str:
    return hashlib.md5(text.lower().encode("utf-8")).hexdigest()

# _summarizer = None

# def _load_summarizer():
#     global _summarizer
#     if _summarizer is None:
#         try:
#             _summarizer = pipeline(
#                 "summarization",
#                 model="sshleifer/distilbart-cnn-6-6",
#                 tokenizer="sshleifer/distilbart-cnn-6-6",
#                 device=-1  # CPU only
#             )
#             print("✅ Loaded lightweight summarizer (DistilBART, CPU)")
#         except Exception as e:
#             print(f"⚠️ Could not load summarizer: {e}")
#             _summarizer = None
#     return _summarizer

def fetch_full_article_text(url: str) -> str:
    """
    Extract full article text from a news URL using trafilatura.
    Returns cleaned text or empty string if extraction fails.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text or ""
    except Exception as e:
        print(f"❌ Error fetching article: {url} — {e}")
        return ""

# BART-based summarization commented out - using Groq instead
# def summarize_text_if_possible(content, titles=None):
#     """
#     Summarize text using BART if length is sufficient.
#     - If `content` is a single string → returns a single summary (string or None).
#     - If `content` is a list of strings → returns a list of summaries aligned with input.
#     Titles are optional, mainly used for debugging/logging.
#     """
#     try:
#         # Case 1: Single string
#         if isinstance(content, str):
#             if not content or len(content.split()) < 30:
#                 return None
#
#             summary = summarizer(
#                 content,
#                 max_length=100,
#                 min_length=30,
#                 do_sample=False,
#                 num_beams=6,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 early_stopping=True
#             )
#             return summary[0]["summary_text"]
#
#         # Case 2: List of strings
#         elif isinstance(content, list):
#             results = []
#             for idx, text in enumerate(content):
#                 if not text or len(text.split()) < 30:
#                     results.append(None)
#                     continue
#                 summary = summarizer(
#                     text,
#                     max_length=100,
#                     min_length=30,
#                     do_sample=False,
#                     num_beams=6,
#                     no_repeat_ngram_size=3,
#                     length_penalty=1.0,
#                     early_stopping=True
#                 )
#                 results.append(summary[0]["summary_text"])
#
#                 # optional: log which title got summarized
#                 if titles and idx < len(titles):
#                     snippet = summary[0]["summary_text"][:120].replace("\n", " ")
#                     print(f"✅ Summarized: {titles[idx]} → {snippet}...")
#
#             return results
#
#         else:
#             return None
#
#     except Exception as e:
#         print(f"Summarization error: {e}")
#         if isinstance(content, str):
#             return None
#         elif isinstance(content, list):
#             return [None] * len(content)


# Global counter for tracking first few summaries to log
_summary_log_counter = 0
_MAX_LOGGED_SUMMARIES = 5

# API key rotation tracking (simple round-robin for now)
_summary_key_index = 0
_summary_key_lock = threading.Lock()  # Thread-safe key rotation

def _get_next_summary_key():
    """Get next API key in rotation for summarization."""
    global _summary_key_index
    if not GROQ_SUMMARY_API_KEYS:
        return None
    with _summary_key_lock:
        key = GROQ_SUMMARY_API_KEYS[_summary_key_index]
        _summary_key_index = (_summary_key_index + 1) % len(GROQ_SUMMARY_API_KEYS)
        return key

def summarize_with_groq(text: str, title: Optional[str] = None) -> Optional[str]:
    """
    Summarize text using Groq API with llama-3.1-8b-instant model.
    Uses multiple GROQ_SUMMARY_API_KEYS with rotation/fallback for rate limit handling.
    Returns a summary string or None if summarization fails.
    Includes exponential backoff retry logic and automatic key rotation on rate limits.
    """
    global _summary_log_counter
    
    if not GROQ_SUMMARY_API_KEYS:
        print("⚠️ GROQ_SUMMARY_API_KEYS not set, skipping summarization")
        return None
    
    if not text or len(text.split()) < 30:
        return None
    
    # Get initial API key (round-robin)
    current_key = _get_next_summary_key()
    if not current_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {current_key}",
        "Content-Type": "application/json",
    }
    
    system_prompt = (
        "You are an expert news article summarizer. "
        "Create concise, informative summaries that capture the key points of news articles. "
        "Keep summaries between 60-100 words. Focus on the main facts, events, and implications."
    )
    
    user_prompt = f"Summarize the following news article{' titled: ' + title if title else ''}:\n\n{text[:2000]}"
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 200,
    }
    
    # Log first few articles being summarized
    if _summary_log_counter < _MAX_LOGGED_SUMMARIES:
        _summary_log_counter += 1
        text_preview = text[:200].replace("\n", " ").strip()
        print(f"\n📝 [Summary #{_summary_log_counter}] Starting Groq summarization:")
        print(f"   Title: {title[:80] if title else 'N/A'}")
        print(f"   Text preview: {text_preview}...")
        print(f"   Text length: {len(text)} chars, {len(text.split())} words")
        print(f"   Requested model in payload: {payload['model']}")
    
    # Exponential backoff retry logic with key rotation on rate limits
    max_key_rotations = len(GROQ_SUMMARY_API_KEYS)  # Try all keys before giving up
    retries_per_key = 2  # Retries per key before rotating
    backoffs = [5, 10]  # Wait times in seconds
    attempt = 0
    key_attempt = 0
    keys_tried = set()
    
    try:
        while key_attempt < max_key_rotations:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                # Log the actual model used by the API
                actual_model = data.get("model", "unknown")
                requested_model = payload.get("model", "unknown")
                if actual_model != requested_model:
                    print(f"\n⚠️ [MODEL MISMATCH] Requested: {requested_model}, API used: {actual_model}")
                    print(f"   Title: {title[:80] if title else 'N/A'}")
                    print(f"   Response model field: {actual_model}")
                    print("-" * 80)
                
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    summary = content.strip()
                    
                    # Log first few successful summaries
                    if _summary_log_counter <= _MAX_LOGGED_SUMMARIES:
                        key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                        print(f"\n✅ [Summary #{_summary_log_counter}] Successfully summarized (Key #{key_num}):")
                        print(f"   Title: {title[:80] if title else 'N/A'}")
                        print(f"   Requested model: {requested_model}")
                        print(f"   Actual model used: {actual_model}")
                        print(f"   Summary: {summary}")
                        print(f"   Summary length: {len(summary)} chars, {len(summary.split())} words")
                        print("-" * 80)
                    
                    return summary
                return None
                    
            elif resp.status_code == 429:
                # Rate limit hit - try next API key or retry with backoff
                if attempt < retries_per_key:
                    wait_s = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                    attempt += 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                    print(f"\n🚨 [RATE LIMIT HIT - Key #{key_num}] {timestamp}")
                    print(f"   Article: {title[:80] if title else 'N/A'}")
                    print(f"   Status: 429 Too Many Requests")
                    print(f"   Retry #{attempt}/{retries_per_key} on same key: Waiting {wait_s} seconds...")
                    time.sleep(wait_s)
                    continue
                else:
                    # Rotate to next key
                    keys_tried.add(current_key)
                    if len(keys_tried) >= len(GROQ_SUMMARY_API_KEYS):
                        print(f"\n❌ [ALL KEYS EXHAUSTED] All {len(GROQ_SUMMARY_API_KEYS)} API keys hit rate limits")
                        return None
                    
                    current_key = _get_next_summary_key()
                    while current_key in keys_tried and len(keys_tried) < len(GROQ_SUMMARY_API_KEYS):
                        current_key = _get_next_summary_key()
                    
                    if current_key and current_key not in keys_tried:
                        headers["Authorization"] = f"Bearer {current_key}"
                        attempt = 0  # Reset attempt counter for new key
                        key_attempt += 1
                        key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                        print(f"\n🔄 [ROTATING TO KEY #{key_num}] Trying next API key...")
                        continue
                    else:
                        print(f"\n❌ [NO MORE KEYS] All keys exhausted")
                        return None
            else:
                # Non-rate-limit error - try next key once, then give up
                if key_attempt == 0 and len(GROQ_SUMMARY_API_KEYS) > 1:
                    current_key = _get_next_summary_key()
                    headers["Authorization"] = f"Bearer {current_key}"
                    key_attempt += 1
                    key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                    print(f"\n⚠️ [API ERROR {resp.status_code}] Rotating to Key #{key_num}...")
                    continue
                else:
                    print(f"⚠️ Groq summarization error {resp.status_code}: {resp.text[:200]}")
                    if _summary_log_counter <= _MAX_LOGGED_SUMMARIES:
                        print(f"   Failed article: {title[:80] if title else 'N/A'}")
                    return None
                
    except Exception as e:
        print(f"⚠️ Groq summarization exception: {e}")
        if _summary_log_counter <= _MAX_LOGGED_SUMMARIES:
            print(f"   Failed article: {title[:80] if title else 'N/A'}")
        return None


def summarize_batch_with_groq(articles_data: List[Tuple[str, str, int]]) -> List[Optional[str]]:
    """
    Summarize multiple articles in a single Groq API call with proper mapping.
    Uses multiple GROQ_SUMMARY_API_KEYS with rotation/fallback for rate limit handling.
    articles_data: List of (title, text, index) tuples
    Returns a list of summaries aligned with input order (indexed properly).
    """
    if not GROQ_SUMMARY_API_KEYS:
        print("⚠️ GROQ_SUMMARY_API_KEYS not set, skipping batch summarization")
        return [None] * len(articles_data)
    
    if not articles_data:
        return []
    
    # Get initial API key (round-robin)
    current_key = _get_next_summary_key()
    if not current_key:
        return [None] * len(articles_data)
    
    try:
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
        }
        
        system_prompt = (
            "You are an expert news article summarizer. "
            "I will provide multiple news articles numbered starting from 0. For each article, create a concise, "
            "informative summary that captures the key points (60-100 words). Focus on main facts, events, and implications. "
            "Return ONLY a valid JSON object with this exact format: "
            '{"summaries": [{"index": 0, "summary": "summary text"}, {"index": 1, "summary": "summary text"}, ...]} '
            "The index MUST match the article number (0, 1, 2, etc.) in the order provided."
        )
        
        # Build user prompt with numbered articles (using local index 0, 1, 2...)
        articles_text = []
        for local_idx, (title, text, original_idx) in enumerate(articles_data):
            article_text = text[:1500]  # Limit per article to fit in context
            articles_text.append(f"Article {local_idx} (Title: {title[:100]}):\n{article_text}")
        
        user_prompt = (
            f"Summarize the following {len(articles_data)} news articles. Return a JSON object with a 'summaries' array "
            "where each entry has 'index' (matching the article number 0, 1, 2, etc.) and 'summary' fields.\n\n" +
            "\n\n---\n\n".join(articles_text)
        )
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 200 * len(articles_data),  # Allocate tokens for all summaries
            "response_format": {"type": "json_object"}
        }
        
        # Exponential backoff retry logic with key rotation on rate limits
        max_key_rotations = len(GROQ_SUMMARY_API_KEYS)
        retries_per_key = 2
        backoffs = [5, 10]
        attempt = 0
        key_attempt = 0
        keys_tried = set()
        
        while key_attempt < max_key_rotations:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,  # Longer timeout for batch processing
            )
            
            if resp.status_code == 200:
                data = resp.json()
                # Log the actual model used by the API
                actual_model = data.get("model", "unknown")
                requested_model = payload.get("model", "unknown")
                if actual_model != requested_model:
                    print(f"\n⚠️ [BATCH MODEL MISMATCH] Requested: {requested_model}, API used: {actual_model}")
                    print(f"   Response model field: {actual_model}")
                    print("-" * 80)
                
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if content:
                    try:
                        # Parse JSON response
                        result = json.loads(content)
                        summaries_data = result.get("summaries", [])
                        
                        # Create a dictionary mapping local index to summary
                        # Local index is 0, 1, 2... matching the order in articles_data
                        summary_map = {}
                        for item in summaries_data:
                            idx = item.get("index")
                            if idx is not None and isinstance(idx, int):
                                summary_map[idx] = item.get("summary", "").strip()
                        
                        # Build results list in correct order (local index matches position in articles_data)
                        results = [None] * len(articles_data)
                        for local_idx in range(len(articles_data)):
                            if local_idx in summary_map:
                                results[local_idx] = summary_map[local_idx]
                        
                        success_count = len([r for r in results if r])
                        key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                        print(f"✅ Batch summarized {success_count}/{len(articles_data)} articles (Key #{key_num}, model: {actual_model})")
                        return results
                        
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse batch summary JSON: {e}")
                        print(f"Response content: {content[:500]}")
                        # Fallback to individual summarization
                        return [None] * len(articles_data)
                else:
                    return [None] * len(articles_data)
                    
            elif resp.status_code == 429:
                # Rate limit hit - try next API key or retry with backoff
                if attempt < retries_per_key:
                    wait_s = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                    attempt += 1
                    key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                    print(f"⚠️ [BATCH RATE LIMIT - Key #{key_num}] Retrying in {wait_s}s (attempt {attempt}/{retries_per_key})...")
                    time.sleep(wait_s)
                    continue
                else:
                    # Rotate to next key
                    keys_tried.add(current_key)
                    if len(keys_tried) >= len(GROQ_SUMMARY_API_KEYS):
                        print(f"❌ [BATCH ALL KEYS EXHAUSTED] All {len(GROQ_SUMMARY_API_KEYS)} API keys hit rate limits")
                        return [None] * len(articles_data)
                    
                    current_key = _get_next_summary_key()
                    while current_key in keys_tried and len(keys_tried) < len(GROQ_SUMMARY_API_KEYS):
                        current_key = _get_next_summary_key()
                    
                    if current_key and current_key not in keys_tried:
                        headers["Authorization"] = f"Bearer {current_key}"
                        attempt = 0
                        key_attempt += 1
                        key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                        print(f"🔄 [BATCH ROTATING TO KEY #{key_num}] Trying next API key...")
                        continue
                    else:
                        print(f"❌ [BATCH NO MORE KEYS] All keys exhausted")
                        return [None] * len(articles_data)
            else:
                # Non-rate-limit error - try next key once, then give up
                if key_attempt == 0 and len(GROQ_SUMMARY_API_KEYS) > 1:
                    current_key = _get_next_summary_key()
                    headers["Authorization"] = f"Bearer {current_key}"
                    key_attempt += 1
                    key_num = GROQ_SUMMARY_API_KEYS.index(current_key) + 1 if current_key in GROQ_SUMMARY_API_KEYS else "?"
                    print(f"⚠️ [BATCH API ERROR {resp.status_code}] Rotating to Key #{key_num}...")
                    continue
                else:
                    print(f"⚠️ Groq batch summarization error {resp.status_code}: {resp.text[:200]}")
                    return [None] * len(articles_data)
            
    except Exception as e:
        print(f"⚠️ Groq batch summarization exception: {e}")
    
    return [None] * len(articles_data)


def summarize_text_if_possible(content, titles=None):
    """
    Summarize text using Groq API if length is sufficient.
    - If `content` is a single string → returns a single summary (string or None).
    - If `content` is a list of strings → returns a list of summaries aligned with input using batch processing.
    Titles are optional, mainly used for debugging/logging.
    """
    try:
        # Case 1: Single string
        if isinstance(content, str):
            title = titles if isinstance(titles, str) else None
            summary = summarize_with_groq(content, title)
            return summary

        # Case 2: List of strings - use batch processing for efficiency
        elif isinstance(content, list):
            if len(content) == 0:
                return []
            
            # Use parallel batch summarization for multiple articles
            if len(content) > 1:
                return parallel_summarize_texts(content, titles, max_workers=25, batch_size=10)
            else:
                # Single item - use individual call
                title = titles[0] if titles and len(titles) > 0 else None
                summary = summarize_with_groq(content[0], title)
                return [summary] if summary else [None]

        else:
            return None

    except Exception as e:
        print(f"Summarization error: {e}")
        if isinstance(content, str):
            return None
        elif isinstance(content, list):
            return [None] * len(content)


def call_groq_summarize(titles: List[str], texts: List[str], max_workers: int = 25, batch_size: int = 10) -> List[str]:
    """
    Parallelize Groq summarization safely for multiple articles using batch processing.
    Returns a list of summaries aligned with the input.
    
    Args:
        titles: List of article titles
        texts: List of article texts
        max_workers: Maximum concurrent batch API calls (default 25)
        batch_size: Number of articles per batch API call (default 10)
    """
    # Use the batch summarization function
    batch_data = [(titles[idx] if titles and idx < len(titles) else "", text, idx) for idx, text in enumerate(texts)]
    
    # Process in parallel batches
    batches = []
    for i in range(0, len(batch_data), batch_size):
        batches.append((batch_data[i:i + batch_size], i))
    
    results = [None] * len(texts)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(summarize_batch_with_groq, batch_data): (batch_data, start_idx)
            for batch_data, start_idx in batches
        }

        for future in as_completed(future_to_batch):
            batch_data, start_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                for local_idx, summary in enumerate(batch_results):
                    global_idx = start_idx + local_idx
                    if global_idx < len(results):
                        results[global_idx] = summary
                        if summary and titles and global_idx < len(titles):
                            print(f"✅ Summarized (Groq batch): {titles[global_idx][:60]}")
            except Exception as e:
                print(f"⚠️ Batch summarization failed for batch at index {start_idx}: {e}")
    
    return results


def parallel_summarize_texts(texts: List[str], titles: Optional[List[str]] = None, max_workers: int = 25, batch_size: int = 10) -> List[Optional[str]]:
    """
    Parallelize Groq summarization for multiple texts using batch API calls.
    Uses batch summarization (multiple articles per API call) for efficiency.
    Returns a list of summaries aligned with the input texts.
    
    Args:
        texts: List of article texts to summarize
        titles: Optional list of article titles
        max_workers: Maximum concurrent batch API calls (default 25, within Groq free tier limits)
        batch_size: Number of articles per batch API call (default 10)
    """
    if not texts:
        return []
    
    results = [None] * len(texts)
    
    # Group articles into batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_titles = titles[i:i + batch_size] if titles else [None] * len(batch_texts)
        # Create tuples of (title, text, original_index)
        batch_data = [(batch_titles[j] or "", batch_texts[j], i + j) for j in range(len(batch_texts))]
        batches.append((batch_data, i))  # Store batch data and starting index
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(summarize_batch_with_groq, batch_data): (batch_data, start_idx)
            for batch_data, start_idx in batches
        }

        for future in as_completed(future_to_batch):
            batch_data, start_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                # Map batch results to correct positions in results array
                for local_idx, summary in enumerate(batch_results):
                    global_idx = start_idx + local_idx
                    if global_idx < len(results):
                        results[global_idx] = summary
                        if summary and titles and global_idx < len(titles):
                            snippet = summary[:80].replace("\n", " ")
                            print(f"✅ Batch Groq summary [{global_idx}]: {titles[global_idx][:60]} → {snippet}...")
            except Exception as e:
                print(f"⚠️ Batch summarization failed for batch starting at index {start_idx}: {e}")
                # Fallback to individual summarization for this batch
                for local_idx, (title, text, _) in enumerate(batch_data):
                    global_idx = start_idx + local_idx
                    if global_idx < len(results):
                        try:
                            results[global_idx] = summarize_with_groq(text, title)
                        except Exception as e2:
                            print(f"⚠️ Individual fallback failed for index {global_idx}: {e2}")
                            results[global_idx] = None
    
    return results

CACHE_FILE = "summary_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        summary_cache = json.load(f)
else:
    summary_cache = {}

def get_cached_summary(url_hash):
    return summary_cache.get(url_hash)

def save_summary_to_cache(url_hash, summary_text):
    summary_cache[url_hash] = summary_text
    if len(summary_cache) % 20 == 0:  # Save every 20 entries
        with open(CACHE_FILE, "w") as f:
            json.dump(summary_cache, f)

def _summarize_in_batches(articles: List[dict]) -> Tuple[int, int]:
    """
    Summarize articles in batches using parallel threads + caching.
    Saves each summary individually to Supabase.
    Returns (num_batches, num_summarized).
    """
    to_sum = []
    for a in articles:
        title = a.get("title") or ""
        url = a.get("url")
        full_text = ""

        # Try fetching full article text
        if url:
            full_text = fetch_full_article_text(url)
            full_text = full_text[:1000]  # limit length increased to 1000
            if full_text:
                try:
                    title_dbg = (title or a.get("title") or "")[:80]
                    print(f"Summary input source: full_page_text (len={len(full_text)}) for title='{title_dbg}'")
                except Exception:
                    pass

        # Fallback to short content/description
        if not full_text:
            chosen = ""
            if a.get("content"):
                full_text = a.get("content")
                chosen = "content"
            else:
                full_text = a.get("description") or ""
                chosen = "description" if a.get("description") else "none"
            try:
                title_dbg = (title or a.get("title") or "")[:80]
                print(f"Summary input source: {chosen} (len={len(full_text)}) for title='{title_dbg}'")
            except Exception:
                pass

        if full_text:
            to_sum.append((a, title, full_text))

    if not to_sum:
        print("⚠️ No articles with text available for summarization.")
        return (0, 0)

    # Process ALL articles, no batch size limit
    print(f"📊 Starting summarization for {len(to_sum)} articles...")
    summarized_count = 0
    client = supabase_client()

    # Prepare all titles and texts for batch summarization
    all_titles = [t for (_, t, _) in to_sum]
    all_texts = [x for (_, _, x) in to_sum]

    # Use batch summarization with proper mapping (10 articles per API call, 25 concurrent calls)
    # This ensures all articles are summarized and summaries are correctly mapped
    summaries = parallel_summarize_texts(all_texts, all_titles, max_workers=25, batch_size=10)
    
    # Count batches (approximate based on batch_size)
    batches = (len(to_sum) + 10 - 1) // 10  # Ceiling division

    # Map summaries back to articles and save to Supabase
    for (article, title, _), summary_text in zip(to_sum, summaries):
        if not summary_text:
            continue

        try:
            url_hash = hashlib.md5(article["url"].lower().encode()).hexdigest()

            # Cache check/save
            if get_cached_summary(url_hash):
                continue
            save_summary_to_cache(url_hash, summary_text)

            # Update Supabase - ensure summary is mapped to correct article by url_hash
            client.table("articles").update({
                "summary": summary_text,
                "summarized": True,
                "summarization_needed": False,
                "updated_at": "now()"
            }).eq("url_hash", url_hash).execute()

            summarized_count += 1
            print(f"✅ Saved summary for: {title[:80]}")

        except Exception as e:
            print(f"⚠️ Error saving summary for {title[:80]}: {e}")

    print(f"✅ Summarization complete — batches: {batches}, summarized: {summarized_count}")
    return (batches, summarized_count)


def _fetch_pending_by_category(client, per_category_limit: int = 2):
    out = {}
    for c in CATEGORIES:
        try:
            res = (
                client.table("articles")
                .select("id,url,title,content,description,summary,summarized,summarization_needed")
                .eq("category", c)
                .eq("summarization_needed", True)
                .order("created_at", desc=True)
                .limit(per_category_limit)
                .execute()
            )
            items = res.data or []
            if items:
                out[c] = items
        except Exception as e:
            print(f"Fetch pending error for {c}: {e}")
    return out


def _prepare_texts(items):
    # Prefer full page text; fallback to content/description
    from concurrent.futures import ThreadPoolExecutor
    texts = []
    titles = []

    def build_text(item):
        url = item.get("url")
        full = fetch_full_article_text(url)[:1000] if url else ""
        if full:
            try:
                print(f"Summary input source: full_page_text (len={len(full)}) for title='{(item.get('title') or '')[:80]}'")
            except Exception:
                pass
            return full

        fallback = item.get("content") or item.get("description") or ""
        try:
            src = "content" if item.get("content") else ("description" if item.get("description") else "none")
            print(f"Summary input source: {src} (len={len(fallback)}) for title='{(item.get('title') or '')[:80]}'")
        except Exception:
            pass
        return fallback

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(build_text, items))

    for item, txt in zip(items, results):
        titles.append(item.get("title") or "")
        texts.append(txt)
    return titles, texts


def summarize_pending_round_robin(per_category_limit: int = 2, max_cycles: int = 50):
    client = supabase_client()
    cycles = 0
    total_updated = 0

    while cycles < max_cycles:
        cycles += 1
        pending_map = _fetch_pending_by_category(client, per_category_limit)
        if not pending_map:
            break

        # Round‑robin flattening: preserve fairness
        combined = []
        has_more = True
        while has_more:
            has_more = False
            for c, items in list(pending_map.items()):
                if items:
                    combined.append(items.pop(0))
                    has_more = True

        if not combined:
            break

        # Prepare texts
        titles, texts = _prepare_texts(combined)

        # Summarize in one go (model batches over all categories)
        summaries = summarize_text_if_possible(texts, titles) or []

        # Persist results
        for item, summary_text in zip(combined, summaries):
            if not summary_text:
                continue
            try:
                client.table("articles").update({
                    "summary": summary_text,
                    "summarized": True,
                    "summarization_needed": False,
                    "updated_at": "now()"
                }).eq("id", item["id"]).execute()
                total_updated += 1
            except Exception as e:
                print(f"Save summary error for id={item.get('id')}: {e}")

    print(f"✅ Round‑robin summarization complete: cycles={cycles}, updated={total_updated}")


def ingest_category(category: str, page_size: int = 100, summarize_now: bool = True):
    """Ingest one category with strict NewsAPI call count and GNews fallback.
    Returns a dict of metrics for logging and verification.
    If summarize_now is False, this function will only store articles and
    leave summarization for the background round-robin summarizer.
    """
    metrics = {
        "category": category,
        "newsapi_requested": False,
        "newsapi_status": None,
        "newsapi_articles": 0,
        "gnews_fallback": False,
        "gnews_articles": 0,
        "stored": 0,
        "summarized_batches": 0,
        "summarized_count": 0,
    }

    if not NEWS_API_KEY:
        print("NEWS_API_KEY missing; skipping ingestion")
        return metrics

    params = {
        "apiKey": NEWS_API_KEY,
        "pageSize": str(page_size),
        "language": "en",
        "sortBy": "publishedAt",
        "q": category,
        "from": time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 24 * 3600)),
    }
    url = "https://newsapi.org/v2/everything"
    metrics["newsapi_requested"] = True
    resp = requests.get(url, params=params, timeout=30)
    metrics["newsapi_status"] = resp.status_code
    articles = []
    newsapi_articles = []
    if resp.status_code == 200:
        data = resp.json()
        if data.get("status") == "ok":
            newsapi_articles = data.get("articles", [])
            articles = newsapi_articles
        else:
            print(f"NewsAPI response not ok: {data}")
    else:
        print(f"NewsAPI error {resp.status_code}: {resp.text[:200]}")

    metrics["newsapi_articles"] = len(newsapi_articles)

    # Explicit fallback to GNews on quota hit (429) or empty result
    if resp.status_code == 429 or not articles:
        try:
            gparams = {
                "apikey": GNEWS_API_KEY or "",
                "max": str(page_size),
                "lang": "en",
                "country": "us",
                "q": category,
            }
            gurl = "https://gnews.io/api/v4/search"
            gresp = requests.get(gurl, params=gparams, timeout=30)
            if gresp.status_code == 200:
                gdata = gresp.json()
                garts = gdata.get("articles", [])
                articles = garts
                metrics["gnews_fallback"] = True
                metrics["gnews_articles"] = len(garts)
                print(f"Fallback to GNews for category '{category}': {len(garts)} articles")
            else:
                print(f"GNews error {gresp.status_code}: {gresp.text[:200]}")
        except Exception as ge:
            print(f"GNews fallback failed: {ge}")

    client = supabase_client()

    # Prepare rows; store raw first, then batch summarize and update
    rows = []
    for a in articles:
        try:
            if not a.get("url") or a.get("title") == "[Removed]":
                continue
            url_val = a.get("url")
            rows.append({
                "url": url_val,
                "title": a.get("title") or "",
                "description": a.get("description"),
                "content": a.get("content"),
                "author": a.get("author"),
                "source": (a.get("source") or {}).get("name"),
                "image_url": a.get("urlToImage") or a.get("image"),
                "category": category,
                "published_at": a.get("publishedAt") or a.get("published_at"),
                "summary": None,
                "summarized": False,
                "summarization_needed": True,
            })
        except Exception as e:
            print(f"Row build error: {e}")

    # Deduplicate rows by URL to avoid ON CONFLICT double-update errors
    if rows:
        dedup_by_url = {}
        for r in rows:
            k = (r.get("url") or "").strip().lower()
            if k and k not in dedup_by_url:
                dedup_by_url[k] = r
        rows = list(dedup_by_url.values())
        fetched_count = len(rows)
        try:
            client.table("articles").upsert(rows, on_conflict="url_hash").execute()
            metrics["stored"] = len(rows)
            print(f"Stored {len(rows)} articles for category '{category}'")
        except Exception as e:
            print(f"Bulk insert error: {e}")

        # Summarize in batches where possible (optional)
        if summarize_now:
            batches, summarized = _summarize_in_batches(rows)
            metrics["summarized_batches"] = batches
            metrics["summarized_count"] = summarized
            if summarized:
                try:
                    client.table("articles").upsert(rows, on_conflict="url_hash").execute()
                    print(f"Updated {summarized} summaries for category '{category}' in {batches} batches")
                except Exception as e:
                    print(f"Bulk summary update error: {e}")

        # Verification logging per category
        try:
            print(f"Category={category}, fetched={fetched_count}, batches={batches}, summarized={summarized}")
        except Exception:
            pass

    return metrics


def ingest_all_categories():
    print("Starting scheduled ingestion for categories...")
    print(f"Planned NewsAPI requests this cycle: {len(CATEGORIES)} (pageSize=100)")
    cycle_metrics = []
    # Sequential ingestion with small delay to avoid rate limits
    for c in CATEGORIES:
        try:
            m = ingest_category(c, page_size=100, summarize_now=False)
            cycle_metrics.append(m)
            # Respect API limits between category calls
            time.sleep(1.5)
        except Exception as e:
            print(f"Category {c} ingestion failed: {e}")
            cycle_metrics.append({'category': c, 'error': str(e)})
    # Verification logging
    newsapi_calls = sum(1 for m in cycle_metrics if m.get("newsapi_requested"))
    stored_total = sum(m.get("stored", 0) for m in cycle_metrics)
    summarized_total = sum(m.get("summarized_count", 0) for m in cycle_metrics)
    print(
        f"Cycle verification: categories={len(cycle_metrics)}, newsapi_calls={newsapi_calls}, "
        f"stored_total={stored_total}, summarized_total={summarized_total}"
    )
    for m in cycle_metrics:
        print(
            f"Category {m.get('category')}: newsapi_status={m.get('newsapi_status')}, "
            f"newsapi_articles={m.get('newsapi_articles')}, gnews_fallback={m.get('gnews_fallback')}, "
            f"gnews_articles={m.get('gnews_articles')}, stored={m.get('stored')}, "
            f"summ_batches={m.get('summarized_batches')}, summarized={m.get('summarized_count')}"
        )
    print("Ingestion cycle complete.")


def fetch_recent_summarized_articles() -> List[dict]:
    """Fetch summarized articles from the last 24 hours."""
    client = supabase_client()
    
    # Calculate timestamp for 24 hours ago
    twenty_four_hours_ago = (datetime.now() - timedelta(hours=24)).isoformat()
    
    try:
        result = (
            client.table("articles")
            .select("*")
            .eq("summarized", True)
            .gte("created_at", twenty_four_hours_ago)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"Error fetching summarized articles: {e}")
        return []


def call_groq_generate_quiz_questions(articles: List[dict]) -> List[dict]:
    """
    Generate UPSC-style quiz questions from summarized articles using Groq API.
    Returns a list of question objects with all required fields.
    """
    if not GROQ_API_KEY or not articles:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # System prompt for UPSC-style question generation
        system_prompt = (
            "You are an expert UPSC current affairs question generator. "
            "Create high-quality multiple-choice questions based on recent news summaries. "
            "Each question must have:\n"
            "- Clear, concise question text in UPSC style\n"
            "- 4 plausible options (A, B, C, D)\n"
            "- Correct option index (0-3)\n"
            "- Brief explanation of the correct answer\n"
            "- Appropriate difficulty level (easy, medium, hard)\n"
            "- Relevant topic category (Politics, Economy, Environment, International Relations, Science & Technology, etc.)\n"
            "\nFormat the response as a JSON array of objects with these fields:\n"
            "- question_text: string\n"
            "- options: array of 4 strings\n"
            "- correct_option: integer (0-3)\n"
            "- explanation: string\n"
            "- difficulty: string\n"
            "- topic: string\n"
            "- source_article_title: string (for reference)\n"
        )
        
        # Prepare article summaries for context
        article_contexts = []
        for i, article in enumerate(articles[:30], 1):  # Limit to 30 articles
            context = f"Article {i}: {article.get('title', '')}\n"
            context += f"Summary: {article.get('summary', '')}\n"
            context += f"Category: {article.get('category', '')}\n"
            article_contexts.append(context)
        
        user_content = (
            "Generate 30 UPSC-style current affairs questions based on these recent news summaries. "
            "Ensure questions are diverse across different topics and difficulty levels. "
            "Make options plausible but distinct, with only one correct answer. "
            "Return ONLY valid JSON array format.\n\n"
            "Recent News Summaries:\n" + "\n".join(article_contexts)
        )
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }
        
        # Exponential backoff for rate limit handling
        retries = 3
        backoffs = [10, 20, 30]
        attempt = 0
        
        while True:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=120,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                try:
                    # Parse the JSON response
                    questions_data = json.loads(content)
                    # The response should be a JSON object with a questions array
                    questions = questions_data.get("questions", [])
                    if isinstance(questions, list):
                        return questions
                    else:
                        print("Invalid response format: questions not found as array")
                        return []
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    print(f"Response content: {content[:200]}")
                    return []
                    
            elif resp.status_code == 429 and attempt < retries:
                wait_s = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                attempt += 1
                print(f"Groq rate limit reached (429). Retrying in {wait_s}s (attempt {attempt}/{retries}).")
                time.sleep(wait_s)
                continue
            else:
                print(f"Groq quiz generation error {resp.status_code}: {resp.text[:200]}")
                break
                
    except Exception as e:
        print(f"Groq quiz generation exception: {e}")
    
    return []


def check_existing_questions(source_article_titles: List[str]) -> set:
    """Check which articles already have questions to avoid duplicates."""
    client = supabase_client()
    
    try:
        # Get unique article titles that already have questions
        result = (
            client.table("quiz_questions")
            .select("source_article_id")
            .in_("source_article_id", source_article_titles)
            .execute()
        )
        
        existing_titles = {q.get("source_article_id") for q in result.data if q.get("source_article_id")}
        return existing_titles
        
    except Exception as e:
        print(f"Error checking existing questions: {e}")
        return set()


def insert_quiz_questions(questions: List[dict]) -> int:
    """Insert generated quiz questions into the database."""
    if not questions:
        return 0
    
    client = supabase_client()
    
    # Prepare questions for insertion
    rows = []
    for q in questions:
        try:
            row = {
                "question_text": q.get("question_text", ""),
                "options": json.dumps(q.get("options", [])),
                "correct_answer": q.get("correct_option", 0),
                "explanation": q.get("explanation", ""),
                "topic": q.get("topic", "General"),
                "difficulty": q.get("difficulty", "medium"),
                "source_article_id": q.get("source_article_title", ""),
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Validate required fields
            if (row["question_text"] and row["options"] and 
                len(json.loads(row["options"])) == 4 and 
                row["explanation"] and row["topic"]):
                rows.append(row)
                
        except Exception as e:
            print(f"Error preparing question row: {e}")
    
    if not rows:
        return 0
    
    try:
        # Insert in batches to avoid overwhelming the API
        batch_size = 10
        inserted_count = 0
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            result = client.table("quiz_questions").insert(batch).execute()
            inserted_count += len(result.data or [])
        
        print(f"Successfully inserted {inserted_count} quiz questions")
        return inserted_count
        
    except Exception as e:
        print(f"Error inserting quiz questions: {e}")
        return 0


def generate_daily_quiz_questions():
    """Main function to generate daily UPSC-style quiz questions."""
    print("Starting daily quiz question generation...")
    
    # Fetch recent summarized articles
    articles = fetch_recent_summarized_articles()
    print(f"Found {len(articles)} summarized articles from last 24 hours")
    
    if not articles:
        print("No summarized articles found for quiz generation")
        return
    
    # Check for existing questions to avoid duplicates
    article_titles = [article.get('title', '') for article in articles]
    existing_titles = check_existing_questions(article_titles)
    
    # Filter out articles that already have questions
    new_articles = [a for a in articles if a.get('title', '') not in existing_titles]
    
    if not new_articles:
        print("All articles already have questions generated")
        return
    
    print(f"Generating questions for {len(new_articles)} new articles")
    
    # Generate quiz questions using Groq in two batches to increase pool size
    half = max(1, len(new_articles) // 2)
    batch_a = new_articles[:half]
    batch_b = new_articles[half:]
    questions_a = call_groq_generate_quiz_questions(batch_a)
    questions_b = call_groq_generate_quiz_questions(batch_b)
    questions = (questions_a or []) + (questions_b or [])
    print(f"Generated {len(questions)} quiz questions across two calls")
    
    # Insert questions into database
    inserted_count = insert_quiz_questions(questions)
    print(f"Daily quiz generation complete. Inserted {inserted_count} new questions")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _fetch_prioritized_articles(
    client,
    base_query,
    page: int,
    page_size: int,
):
    # First, fetch summarized articles
    offset = (page - 1) * page_size
    summarized_query = (
        base_query.eq("summarized", True)
        .order("created_at", desc=True)
        .range(offset, offset + page_size - 1)
    )
    summarized_res = summarized_query.execute()
    items = summarized_res.data or []

    # If not enough, fetch raw articles to fill the page
    if len(items) < page_size:
        remaining = page_size - len(items)
        raw_offset = (page - 1) * page_size
        raw_query = (
            base_query.eq("summarized", False)
            .order("created_at", desc=True)
            .range(raw_offset, raw_offset + remaining - 1)
        )
        raw_res = raw_query.execute()
        items.extend(raw_res.data or [])

    out: List[ArticleOut] = []
    for r in items:
        out.append(
            ArticleOut(
                id=r.get("id"),
                title=r.get("title"),
                description=r.get("description"),
                url=r.get("url"),
                urlToImage=r.get("image_url"),
                publishedAt=r.get("published_at"),
                createdAt=r.get("created_at"),
                author=r.get("author"),
                source=r.get("source"),
                category=r.get("category"),
                content=r.get("content"),
                summary=r.get("summary"),
            )
        )
    return out


# New endpoints per app contract
@app.get("/news/latest", response_model=List[ArticleOut])
def get_latest_news(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    client = supabase_client()
    base_query = client.table("articles").select("*")
    return _fetch_prioritized_articles(client, base_query, page, page_size)


@app.get("/news/category/{category}", response_model=List[ArticleOut])
def get_category_news(
    category: str = Path(..., min_length=2),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    client = supabase_client()
    base_query = client.table("articles").select("*").eq("category", category)
    return _fetch_prioritized_articles(client, base_query, page, page_size)


scheduler = BackgroundScheduler()


@app.on_event("startup")
def schedule_jobs():
    # Run news ingestion immediately on startup, then every 3 hours
    scheduler.add_job(ingest_all_categories, "interval", hours=3, id="ingest_news", replace_existing=True)
    
    # Add periodic round-robin summarizer (keeps summaries flowing across categories)
    scheduler.add_job(
        summarize_pending_round_robin,
        "interval",
        minutes=3,
        id="summarize_rr",
        replace_existing=True,
        kwargs={"per_category_limit": 2, "max_cycles": 10}
    )

    # Run quiz generation daily at 2 AM
    scheduler.add_job(generate_daily_quiz_questions, "cron", hour=2, minute=0, id="generate_quiz", replace_existing=True)
    
    # Run riddle generation daily at 11:59 PM
    scheduler.add_job(generate_daily_riddle, "cron", hour=23, minute=59, id="generate_riddle", replace_existing=True)
    
    scheduler.start()
    
    # Kick off initial runs asynchronously
    try:
        import threading
        # Start news ingestion
        threading.Thread(target=ingest_all_categories, daemon=True).start()
        
        # Kick the summarizer once on startup for quicker initial coverage
        threading.Thread(
            target=lambda: summarize_pending_round_robin(per_category_limit=2, max_cycles=20),
            daemon=True
        ).start()

        # Start quiz generation immediately on startup to ensure fresh questions
        # This prevents empty quiz pools after backend restarts
        def immediate_quiz_generation():
            print("🚀 Generating quiz questions immediately on backend startup...")
            generate_daily_quiz_questions()
            print("✅ Initial quiz question generation completed")
        
        threading.Thread(target=immediate_quiz_generation, daemon=True).start()
        
        # Start riddle generation immediately on startup if no riddle exists for today
        def immediate_riddle_generation():
            print("🧩 Checking for today's riddle...")
            if not check_today_riddle_exists():
                print("🚀 Generating daily riddle immediately on backend startup...")
                generate_daily_riddle()
                print("✅ Initial riddle generation completed")
            else:
                print("✅ Today's riddle already exists")
        
        threading.Thread(target=immediate_riddle_generation, daemon=True).start()
        
    except Exception as e:
        print(f"❌ Error starting initial jobs: {e}")


@app.on_event("shutdown")
def shutdown_jobs():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)


# New search endpoint with summarized-first behavior
@app.get("/news/search", response_model=List[ArticleOut])
def search_news_new(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    client = supabase_client()
    base = (
        client.table("articles")
        .select("*")
        .or_(
            f"title.ilike.%{query}%,description.ilike.%{query}%,content.ilike.%{query}%"
        )
    )
    return _fetch_prioritized_articles(client, base, page, page_size)


# Search endpoint: query stored articles
@app.get("/api/search", response_model=List[ArticleOut])
def search_news(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    client = supabase_client()
    pattern = f"%{query}%"
    # Filter across title/description/content
    res = (
        client.table("articles")
        .select("*")
        .or_(f"title.ilike.{pattern},description.ilike.{pattern},content.ilike.{pattern}")
        .order("summarized", desc=True)
        .order("created_at", desc=True)
        .range((page - 1) * page_size, (page * page_size) - 1)
        .execute()
    )

    out: List[ArticleOut] = []
    for r in res.data or []:
        out.append(
            ArticleOut(
                id=r.get("id"),
                title=r.get("title"),
                description=r.get("description"),
                url=r.get("url"),
                urlToImage=r.get("image_url"),
                publishedAt=r.get("published_at"),
                createdAt=r.get("created_at"),
                author=r.get("author"),
                source=r.get("source"),
                category=r.get("category"),
                content=r.get("content"),
                summary=r.get("summary"),
            )
        )
    return out


# Ingest endpoint: accept articles from app and store (with summarization attempt)
class IncomingArticle(BaseModel):
    title: str
    url: str
    description: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    urlToImage: Optional[str] = None
    publishedAt: Optional[str] = None
    category: Optional[str] = None


# Quiz models
class QuizQuestion(BaseModel):
    id: str
    question_text: str
    options: List[str]
    correct_answer: int
    explanation: str
    topic: str
    difficulty: str
    source_article_id: Optional[str] = None
    source_url: Optional[str] = None
    created_at: Optional[str] = None


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    quiz_type: str
    total_questions: int


class QuizSubmission(BaseModel):
    quiz_type: str
    questions: List[str]  # Question IDs
    user_answers: List[int]  # Selected option indices
    time_spent: Optional[int] = None  # Seconds


class QuizResult(BaseModel):
    quiz_attempt_id: str
    score: int
    total_questions: int
    correct_answers: List[int]
    user_answers: List[int]


class QuizAttemptInfo(BaseModel):
    quiz_type: str
    remaining_attempts: int
    max_attempts: int
    next_reset: Optional[str] = None
    can_attempt: bool


class AdUnlockResponse(BaseModel):
    success: bool
    message: str
    remaining_attempts: int
    max_attempts: int
    next_reset: Optional[str] = None


# Riddle models
class Riddle(BaseModel):
    id: str
    question: str
    answer: str
    explanation: Optional[str] = None
    created_at: str


class RiddleResponse(BaseModel):
    riddle: Optional[Riddle] = None
    message: str


@app.post("/api/ingest")
def ingest_articles(items: List[IncomingArticle]):
    client = supabase_client()
    inserted = 0
    for a in items:
        try:
            summary = None
            if a.content:
                summary = summarize_text_if_possible(a.content, a.title)
            row = {
                "url": a.url,
                "title": a.title,
                "description": a.description,
                "content": a.content,
                "author": a.author,
                "source": a.source,
                "image_url": a.urlToImage,
                "category": a.category or "general",
                "published_at": a.publishedAt,
                "summary": summary,
                "summarized": bool(summary),
                "summarization_needed": not bool(summary),
            }
            client.table("articles").upsert(row, on_conflict="url_hash").execute()
            inserted += 1
        except Exception as e:
            print(f"Ingest error: {e}")
    return {"inserted": inserted}


# Quiz endpoints
@app.get("/quiz/daily", response_model=QuizResponse)
def get_daily_quiz():
    """
    Get daily quiz - 15 questions from today's new batch
    Free: 1 attempt/day, Max: 3 attempts/day via rewarded ads
    """
    client = supabase_client()
    user_id = _get_user_id_from_auth()
    
    # Check attempt limits
    attempt_info = _check_quiz_attempt_limits(user_id, "daily")
    if not attempt_info["can_attempt"]:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily quiz attempt limit exceeded",
                "remaining_attempts": attempt_info["remaining_attempts"],
                "max_attempts": attempt_info["max_attempts"],
                "next_reset": attempt_info["next_reset"]
            }
        )
    
    # Build exclusion set for this period
    exclude_ids = _get_excluded_question_ids(client, user_id, "daily")

    # Get today's window
    today = datetime.now().date()
    pool_res = (
        client.table("quiz_questions")
        .select("*")
        .eq("is_active", True)
        .gte("created_at", today.isoformat())
        .lt("created_at", (today + timedelta(days=1)).isoformat())
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    pool = [q for q in (pool_res.data or []) if str(q.get("id")) not in exclude_ids]

    randomized_questions = _shuffle_and_map_questions(pool)[:15]

    # Exhaustion fallback: if still short, fill from global active pool
    if len(randomized_questions) < 15:
        remaining = 15 - len(randomized_questions)
        filler_res = (
            client.table("quiz_questions")
            .select("*")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(300)
            .execute()
        )
        already_ids = {str(q.get("id")) for q in randomized_questions}
        filler_pool = [q for q in (filler_res.data or []) if str(q.get("id")) not in already_ids]
        filler = _shuffle_and_map_questions(filler_pool)[:remaining]
        randomized_questions.extend(filler)

    return QuizResponse(
        questions=randomized_questions,
        quiz_type="daily",
        total_questions=len(randomized_questions)
    )


@app.get("/quiz/weekly", response_model=QuizResponse)
def get_weekly_quiz():
    """
    Get weekly quiz - 30 questions: 70% new (last 7 days), 30% wrong answers
    Free: 1 attempt/day, Max: 5 attempts/week via rewarded ads
    """
    client = supabase_client()
    
    # Get user ID from auth (if available)
    user_id = _get_user_id_from_auth()
    
    # Check attempt limits
    attempt_info = _check_quiz_attempt_limits(user_id, "weekly")
    if not attempt_info["can_attempt"]:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Weekly quiz attempt limit exceeded",
                "remaining_attempts": attempt_info["remaining_attempts"],
                "max_attempts": attempt_info["max_attempts"],
                "next_reset": attempt_info["next_reset"]
            }
        )
    
    # Calculate 70% of 30 = 21 new questions
    new_questions_count = 21
    wrong_questions_count = 9

    # Build exclusion set for this period
    exclude_ids = _get_excluded_question_ids(client, user_id, "weekly")
    
    # Fetch new questions from last 7 days
    seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
    
    new_questions_result = (
        client.table("quiz_questions")
        .select("*")
        .eq("is_active", True)
        .gte("created_at", seven_days_ago)
        .order("created_at", desc=True)
        .limit(300)
        .execute()
    )
    new_pool = [q for q in (new_questions_result.data or []) if str(q.get("id")) not in exclude_ids]
    random.shuffle(new_pool)
    new_questions = new_pool[:new_questions_count]
    
    # Fetch wrong answers if user is authenticated
    wrong_questions = []
    if user_id:
        wrong_pool = _get_user_wrong_questions(client, user_id, wrong_questions_count * 2, seven_days_ago)
        wrong_pool = [q for q in (wrong_pool or []) if str(q.get("id")) not in exclude_ids]
        random.shuffle(wrong_pool)
        wrong_questions = wrong_pool[:wrong_questions_count]
    
    # If no wrong questions or not enough, fill with new questions
    if len(wrong_questions) < wrong_questions_count:
        needed = wrong_questions_count - len(wrong_questions)
        additional_new = (
            client.table("quiz_questions")
            .select("*")
            .eq("is_active", True)
            .gte("created_at", seven_days_ago)
            .order("created_at", desc=True)
            .limit(needed)
            .execute()
        )
        add_pool = [q for q in (additional_new.data or []) if str(q.get("id")) not in exclude_ids]
        random.shuffle(add_pool)
        new_questions.extend(add_pool[:needed])
    
    # Combine and randomize
    combined = new_questions + wrong_questions
    randomized_questions = _shuffle_and_map_questions(combined)[:30]

    # Exhaustion fallback
    if len(randomized_questions) < 30:
        remaining = 30 - len(randomized_questions)
        filler_res = (
            client.table("quiz_questions")
            .select("*")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(400)
            .execute()
        )
        already_ids = {str(q.get("id")) for q in randomized_questions}
        filler_pool = [q for q in (filler_res.data or []) if str(q.get("id")) not in already_ids]
        filler = _shuffle_and_map_questions(filler_pool)[:remaining]
        randomized_questions.extend(filler)

    return QuizResponse(
        questions=randomized_questions,
        quiz_type="weekly",
        total_questions=len(randomized_questions)
    )


@app.get("/quiz/monthly", response_model=QuizResponse)
def get_monthly_quiz():
    """
    Get monthly quiz - 50 questions: 70% new (last 30 days), 30% wrong answers
    Free: 1 attempt/week, Max: 5 attempts/month via rewarded ads
    """
    client = supabase_client()
    
    # Get user ID from auth (if available)
    user_id = _get_user_id_from_auth()
    
    # Check attempt limits
    attempt_info = _check_quiz_attempt_limits(user_id, "monthly")
    if not attempt_info["can_attempt"]:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Monthly quiz attempt limit exceeded",
                "remaining_attempts": attempt_info["remaining_attempts"],
                "max_attempts": attempt_info["max_attempts"],
                "next_reset": attempt_info["next_reset"]
            }
        )
    
    # Calculate 70% of 50 = 35 new questions
    new_questions_count = 35
    wrong_questions_count = 15

    # Build exclusion set for this period
    exclude_ids = _get_excluded_question_ids(client, user_id, "monthly")
    
    # Fetch new questions from last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
    
    new_questions_result = (
        client.table("quiz_questions")
        .select("*")
        .eq("is_active", True)
        .gte("created_at", thirty_days_ago)
        .order("created_at", desc=True)
        .limit(500)
        .execute()
    )
    new_pool = [q for q in (new_questions_result.data or []) if str(q.get("id")) not in exclude_ids]
    random.shuffle(new_pool)
    new_questions = new_pool[:new_questions_count]
    
    # Fetch wrong answers if user is authenticated
    wrong_questions = []
    if user_id:
        wrong_pool = _get_user_wrong_questions(client, user_id, wrong_questions_count * 2, thirty_days_ago)
        wrong_pool = [q for q in (wrong_pool or []) if str(q.get("id")) not in exclude_ids]
        random.shuffle(wrong_pool)
        wrong_questions = wrong_pool[:wrong_questions_count]
    
    # If no wrong questions or not enough, fill with new questions
    if len(wrong_questions) < wrong_questions_count:
        needed = wrong_questions_count - len(wrong_questions)
        additional_new = (
            client.table("quiz_questions")
            .select("*")
            .eq("is_active", True)
            .gte("created_at", thirty_days_ago)
            .order("created_at", desc=True)
            .limit(needed)
            .execute()
        )
        add_pool = [q for q in (additional_new.data or []) if str(q.get("id")) not in exclude_ids]
        random.shuffle(add_pool)
        new_questions.extend(add_pool[:needed])
    
    # Combine and randomize
    combined = new_questions + wrong_questions
    randomized_questions = _shuffle_and_map_questions(combined)[:50]

    # Exhaustion fallback
    if len(randomized_questions) < 50:
        remaining = 50 - len(randomized_questions)
        filler_res = (
            client.table("quiz_questions")
            .select("*")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(600)
            .execute()
        )
        already_ids = {str(q.get("id")) for q in randomized_questions}
        filler_pool = [q for q in (filler_res.data or []) if str(q.get("id")) not in already_ids]
        filler = _shuffle_and_map_questions(filler_pool)[:remaining]
        randomized_questions.extend(filler)

    return QuizResponse(
        questions=randomized_questions,
        quiz_type="monthly",
        total_questions=len(randomized_questions)
    )


@app.post("/quiz/submit", response_model=QuizResult)
def submit_quiz_attempt(submission: QuizSubmission):
    """
    Submit quiz results and store in database
    """
    client = supabase_client()
    user_id = _get_user_id_from_auth()
    
    if len(submission.questions) != len(submission.user_answers):
        raise HTTPException(status_code=400, detail="Questions and answers count mismatch")
    
    # Get correct answers for all questions
    correct_answers = []
    for question_id in submission.questions:
        result = (
            client.table("quiz_questions")
            .select("correct_answer")
            .eq("id", question_id)
            .execute()
        )
        if result.data:
            correct_answers.append(result.data[0]["correct_answer"])
        else:
            correct_answers.append(-1)  # Mark as invalid
    
    # Calculate score
    score = 0
    for i, user_answer in enumerate(submission.user_answers):
        if user_answer == correct_answers[i]:
            score += 1
    
    percentage_score = int((score / len(submission.questions)) * 100) if submission.questions else 0
    
    # Store quiz attempt
    attempt_data = {
        "user_id": user_id,
        "quiz_type": submission.quiz_type,
        "questions": submission.questions,
        "user_answers": submission.user_answers,
        "correct_answers": correct_answers,
        "score": percentage_score,
        "total_questions": len(submission.questions),
        "time_spent": submission.time_spent,
        "completed_at": datetime.now().isoformat()
    }
    
    attempt_result = client.table("quiz_attempts").insert(attempt_data).execute()
    attempt_id = attempt_result.data[0]["id"] if attempt_result.data else None
    
    # Store individual question performance
    if user_id and attempt_id:
        for i, question_id in enumerate(submission.questions):
            is_correct = submission.user_answers[i] == correct_answers[i]
            performance_data = {
                "user_id": user_id,
                "question_id": question_id,
                "is_correct": is_correct,
                "quiz_attempt_id": attempt_id
            }
            client.table("user_question_performance").insert(performance_data).execute()
    
    return QuizResult(
        quiz_attempt_id=attempt_id or "",
        score=percentage_score,
        total_questions=len(submission.questions),
        correct_answers=correct_answers,
        user_answers=submission.user_answers
    )


@app.get("/quiz/attempts/{quiz_type}", response_model=QuizAttemptInfo)
def get_quiz_attempt_info(quiz_type: str):
    """
    Get user's remaining quiz attempts for a specific quiz type
    """
    user_id = _get_user_id_from_auth()
    
    if quiz_type not in ["daily", "weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="Invalid quiz type")
    
    attempt_info = _check_quiz_attempt_limits(user_id, quiz_type)
    
    return QuizAttemptInfo(
        quiz_type=quiz_type,
        remaining_attempts=attempt_info["remaining_attempts"],
        max_attempts=attempt_info["max_attempts"],
        next_reset=attempt_info["next_reset"],
        can_attempt=attempt_info["can_attempt"]
    )


@app.post("/quiz/ad-unlock/{quiz_type}", response_model=AdUnlockResponse)
def unlock_ad_attempt(quiz_type: str):
    """
    Unlock an additional quiz attempt via rewarded ad
    """
    user_id = _get_user_id_from_auth()
    
    # Allow guest users to unlock attempts (they'll be tracked locally in the app)
    if not user_id:
        # For guest users, return success without backend tracking
        return AdUnlockResponse(
            success=True,
            message="Ad unlock successful for guest user",
            remaining_attempts=1,  # Guest users get 1 additional attempt
            max_attempts=3,  # Default max for guests
            next_reset="2024-01-01T00:00:00Z"  # Placeholder
        )
    
    if quiz_type not in ["daily", "weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="Invalid quiz type")
    
    # Check current attempt limits
    attempt_info = _check_quiz_attempt_limits(user_id, quiz_type)
    
    # Check if user can unlock more attempts (not exceeding max)
    if attempt_info["remaining_attempts"] >= attempt_info["max_attempts"]:
        return AdUnlockResponse(
            success=False,
            message="Maximum attempts already reached for this period",
            remaining_attempts=attempt_info["remaining_attempts"],
            max_attempts=attempt_info["max_attempts"],
            next_reset=attempt_info["next_reset"]
        )
    
    client = supabase_client()
    now = datetime.now()
    
    try:
        # Record the ad unlock
        ad_unlock_data = {
            "user_id": user_id,
            "quiz_type": quiz_type,
            "unlock_date": now.isoformat()
        }
        
        result = client.table("ad_attempts").insert(ad_unlock_data).execute()
        
        if result.data:
            # Get updated attempt info
            updated_attempt_info = _check_quiz_attempt_limits(user_id, quiz_type)
            
            return AdUnlockResponse(
                success=True,
                message="Successfully unlocked an additional attempt!",
                remaining_attempts=updated_attempt_info["remaining_attempts"],
                max_attempts=updated_attempt_info["max_attempts"],
                next_reset=updated_attempt_info["next_reset"]
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to record ad unlock")
            
    except Exception as e:
        print(f"Error recording ad unlock: {e}")
        raise HTTPException(status_code=500, detail="Failed to process ad unlock")


# Riddle endpoints
@app.get("/riddles/latest", response_model=RiddleResponse)
def get_latest_riddle_endpoint():
    """
    Get the most recent riddle for the app.
    Returns the latest riddle or a message if none found.
    """
    try:
        riddle_data = get_latest_riddle()
        
        if riddle_data:
            riddle = Riddle(
                id=riddle_data["id"],
                question=riddle_data["question"],
                answer=riddle_data["answer"],
                explanation=riddle_data.get("explanation"),
                created_at=riddle_data["created_at"]
            )
            return RiddleResponse(
                riddle=riddle,
                message="Latest riddle retrieved successfully"
            )
        else:
            return RiddleResponse(
                riddle=None,
                message="No riddles found. Please try again later."
            )
            
    except Exception as e:
        print(f"❌ Error fetching latest riddle: {e}")
        return RiddleResponse(
            riddle=None,
            message="Error retrieving riddle. Please try again later."
        )


# Helper functions
def _get_user_id_from_auth() -> Optional[str]:
    """
    Extract user ID from authentication context
    """
    # This would typically come from FastAPI's Depends or request context
    # For now, return None (anonymous user)
    return None


def _get_period_start(quiz_type: str, now: Optional[datetime] = None) -> datetime:
    if not now:
        now = datetime.now()
    if quiz_type == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if quiz_type == "weekly":
        today = now.date()
        monday = today - timedelta(days=today.weekday())
        return datetime.combine(monday, datetime.min.time())
    if quiz_type == "monthly":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _get_excluded_question_ids(client, user_id: Optional[str], quiz_type: str) -> set:
    """Return set of question IDs attempted by user within the active period for quiz_type."""
    if not user_id:
        return set()
    try:
        start = _get_period_start(quiz_type)
        res = (
            client.table("quiz_attempts")
            .select("questions")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("created_at", start.isoformat())
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        exclude = set()
        for row in res.data or []:
            for qid in row.get("questions", []) or []:
                exclude.add(str(qid))
        return exclude
    except Exception:
        return set()


def _shuffle_and_map_questions(items: List[dict]) -> List[dict]:
    """Shuffle list and randomize options per question."""
    arr = items[:]
    random.shuffle(arr)
    return [_randomize_question_options(q) for q in arr]


def _check_quiz_attempt_limits(user_id: str, quiz_type: str) -> dict:
    """
    Check if user has remaining attempts for a quiz type
    Returns dict with remaining_attempts, max_attempts, and next_reset info
    """
    if not user_id:
        # Anonymous users get 1 free attempt
        return {
            "remaining_attempts": 1,
            "max_attempts": 1,
            "next_reset": None,
            "can_attempt": True
        }
    
    client = supabase_client()
    now = datetime.now()
    
    # Define attempt limits based on quiz type
    if quiz_type == "daily":
        max_attempts = 3
        # Count attempts from today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        result = (
            client.table("quiz_attempts")
            .select("id")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("completed_at", today_start.isoformat())
            .execute()
        )
        used_attempts = len(result.data or [])
        next_reset = (today_start + timedelta(days=1)).isoformat()
        
    elif quiz_type == "weekly":
        max_attempts = 5
        # Count attempts from this week (Monday 00:00)
        today = now.date()
        monday = today - timedelta(days=today.weekday())
        week_start = datetime.combine(monday, datetime.min.time())
        result = (
            client.table("quiz_attempts")
            .select("id")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("completed_at", week_start.isoformat())
            .execute()
        )
        used_attempts = len(result.data or [])
        next_monday = monday + timedelta(days=7)
        next_reset = datetime.combine(next_monday, datetime.min.time()).isoformat()
        
    elif quiz_type == "monthly":
        max_attempts = 5
        # Count attempts from this month
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = (
            client.table("quiz_attempts")
            .select("id")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("completed_at", month_start.isoformat())
            .execute()
        )
        used_attempts = len(result.data or [])
        # Next month's first day
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        next_reset = next_month.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    
    else:
        # Default for unknown quiz types
        return {
            "remaining_attempts": 1,
            "max_attempts": 1,
            "next_reset": None,
            "can_attempt": True
        }
    
    # Get ad-unlocked attempts
    ad_attempts = _get_user_ad_attempts(user_id, quiz_type)
    
    # Calculate remaining attempts including ad unlocks
    remaining_attempts = max(0, max_attempts - used_attempts + ad_attempts)
    
    return {
        "remaining_attempts": remaining_attempts,
        "max_attempts": max_attempts,
        "next_reset": next_reset,
        "can_attempt": remaining_attempts > 0
    }


def _get_user_ad_attempts(user_id: str, quiz_type: str) -> int:
    """
    Get additional attempts unlocked via rewarded ads
    Query the ad_attempts table to count ad unlocks for this user and quiz type
    """
    if not user_id:
        return 0
    
    client = supabase_client()
    now = datetime.now()
    
    try:
        # Define time period based on quiz type
        if quiz_type == "daily":
            # Count ad unlocks from today
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            result = (
                client.table("ad_attempts")
                .select("id")
                .eq("user_id", user_id)
                .eq("quiz_type", quiz_type)
                .gte("unlock_date", today_start.isoformat())
                .execute()
            )
        elif quiz_type == "weekly":
            # Count ad unlocks from this week (Monday 00:00)
            today = now.date()
            monday = today - timedelta(days=today.weekday())
            week_start = datetime.combine(monday, datetime.min.time())
            result = (
                client.table("ad_attempts")
                .select("id")
                .eq("user_id", user_id)
                .eq("quiz_type", quiz_type)
                .gte("unlock_date", week_start.isoformat())
                .execute()
            )
        elif quiz_type == "monthly":
            # Count ad unlocks from this month
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            result = (
                client.table("ad_attempts")
                .select("id")
                .eq("user_id", user_id)
                .eq("quiz_type", quiz_type)
                .gte("unlock_date", month_start.isoformat())
                .execute()
            )
        else:
            return 0
        
        return len(result.data or [])
        
    except Exception as e:
        print(f"Error fetching ad attempts: {e}")
        return 0


def _get_user_wrong_questions(client, user_id: str, limit: int, since_date: str) -> List[dict]:
    """
    Get user's wrong answers since specific date
    """
    try:
        result = (
            client.table("user_question_performance")
            .select("question_id")
            .eq("user_id", user_id)
            .eq("is_correct", False)
            .gte("attempted_at", since_date)
            .order("attempted_at", desc=True)
            .limit(limit)
            .execute()
        )
        
        question_ids = [item["question_id"] for item in result.data or []]
        
        if not question_ids:
            return []
        
        # Get full question details
        questions_result = (
            client.table("quiz_questions")
            .select("*")
            .in_("id", question_ids)
            .eq("is_active", True)
            .execute()
        )
        
        return questions_result.data or []
        
    except Exception as e:
        print(f"Error fetching wrong questions: {e}")
        return []


def _randomize_question_options(question: dict) -> dict:
    """
    Randomize question options while maintaining correct answer mapping
    """
    if not question.get("options"):
        return question
    
    options = question["options"]
    correct_index = question["correct_answer"]
    
    # Convert to list if it's JSONB
    if isinstance(options, str):
        options = json.loads(options)
    
    # Create mapping of original indices to randomized indices
    original_indices = list(range(len(options)))
    random.shuffle(original_indices)
    
    # Create new options list
    new_options = [options[i] for i in original_indices]
    
    # Find new correct answer index
    new_correct_index = original_indices.index(correct_index)
    
    # Return updated question
    randomized_question = question.copy()
    randomized_question["options"] = new_options
    randomized_question["correct_answer"] = new_correct_index
    
    return randomized_question