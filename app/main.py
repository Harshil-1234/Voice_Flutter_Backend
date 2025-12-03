import os
import hashlib
import time
import random
import re
import codecs
import base64
from typing import List, Optional, Tuple
import feedparser
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Query, Path, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import json
from pydantic import BaseModel
from pathlib import Path as PPath
from dotenv import load_dotenv
from datetime import datetime, timedelta
from transformers import pipeline
import trafilatura
from googlenewsdecoder import new_decoderv1
from riddle_generator import generate_daily_riddle, get_latest_riddle, check_today_riddle_exists




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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Default batch size set to 20 per requirements
SUMMARIZE_BATCH_SIZE = int(os.getenv("SUMMARIZE_BATCH_SIZE", "20"))

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- NEW RSS INGESTION CONFIG ---
RSS_TOPIC_IDS = {
    "technology": "TECHNOLOGY",
    "science": "SCIENCE",
    "world": "WORLD",
    "business": "BUSINESS",
    "health": "HEALTH",
    "entertainment": "ENTERTAINMENT",
    "sports": "SPORTS",
    "politics": "POLITICS",
    "general": "NATION",
}

# Categories to be fetched once with a global country code (e.g., US)
GLOBAL_CATEGORIES = ["technology", "science", "world", "business", "health", "entertainment"]

# Categories to be fetched for each active country
LOCAL_CATEGORIES = ["general", "politics", "sports"]

# List of active countries to fetch local news for
# Expanded list of active countries for local category fetches
# Include major regions so local feeds include UK, CA, DE, JP, RU, FR, AU, BR, IT, ZA
ACTIVE_COUNTRIES = [
    "IN",  # India
    "US",  # United States
    "GB",  # United Kingdom
    
]

RSS_URL_PATTERN = "https://news.google.com/rss/headlines/section/topic/{topic_id}?hl=en-{country_code}&gl={country_code}&ceid={country_code}:en"
# --- END NEW RSS INGESTION CONFIG ---

# --- NEW SEARCH-BASED CATEGORIES ---
SEARCH_CATEGORIES = {
    "ai": "Artificial Intelligence OR AI OR Machine Learning",
    "esports": "Esports OR Gaming News OR Video Games",
    "animation": "Anime OR Manga OR Webtoon OR Crunchyroll",
    "crypto": "Cryptocurrency OR Bitcoin OR Ethereum OR Web3",
    "space": "SpaceX OR NASA OR ISRO OR Astronomy OR Space Exploration",
    "environment": "Environment OR Climate Change OR Nature",
}
# --- END NEW SEARCH-BASED CATEGORIES ---


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


def resolve_redirect_url(url: str, timeout: int = 10) -> str:
    """
    Resolve redirect chains (e.g., Google News redirects) to get the final URL.
    Uses requests.head() to follow redirects without downloading the full page.
    Returns the final URL or original URL if resolution fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        final_url = response.url
        if final_url != url:
            print(f"🔗 Resolved redirect: {url[:80]}... → {final_url[:80]}...")
        return final_url
    except Exception as e:
        print(f"⚠️ Redirect resolution failed for {url}: {e}")
        return url


def extract_with_trafilatura(url: str) -> Optional[str]:
    """
    Extract full article text using trafilatura with proper headers and configuration.
    Returns cleaned text or None if extraction fails or is too short.
    """
    try:
        # Use requests to fetch HTML (handles headers & redirects reliably)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        resp.raise_for_status()
        html = resp.text
        final_url = resp.url
        # Pass raw HTML string to trafilatura.extract
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text:
            # Log scraped length and resolved URL for debugging
            print(f"🔎 Trafilatura extracted {len(text)} chars from {final_url}")
            return text
        return None
    except Exception as e:
        print(f"⚠️ Trafilatura extraction failed for {url}: {e}")
        return None


def extract_with_beautifulsoup_fallback(url: str) -> Optional[str]:
    """
    Fallback extraction using requests + BeautifulSoup for when trafilatura fails.
    Extracts all paragraph text and joins them.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract all paragraph text
        paragraphs = soup.find_all(['p', 'article', 'div', 'section'])
        texts = []
        for para in paragraphs:
            text = para.get_text(separator="\n", strip=True)
            if text and len(text) > 20:  # Skip very short fragments
                texts.append(text)
        
        combined_text = "\n\n".join(texts)
        return combined_text if combined_text else None
    except Exception as e:
        print(f"⚠️ BeautifulSoup fallback failed for {url}: {e}")
        return None


def extract_hero_image(html_content: str) -> Optional[str]:
    """
    Extract the main article image (hero image) from HTML.
    Searches for Open Graph (og:image) or Twitter Card image meta tags.
    
    Priority:
    1. Open Graph image (og:image) - standard for FB/WhatsApp/LinkedIn
    2. Twitter Card image (twitter:image) - fallback for Twitter
    
    Returns: URL to the hero image or None if not found
    """
    try:
        if not html_content:
            return None
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Priority 1: Open Graph Image (og:image)
        og_image_tag = soup.find('meta', property='og:image')
        if og_image_tag and og_image_tag.get('content'):
            image_url = og_image_tag['content'].strip()
            if image_url:
                print(f"🖼️ [extract_hero_image] Found og:image: {image_url[:80]}")
                return image_url
        
        # Priority 2: Twitter Card Image (twitter:image)
        twitter_image_tag = soup.find('meta', name='twitter:image')
        if twitter_image_tag and twitter_image_tag.get('content'):
            image_url = twitter_image_tag['content'].strip()
            if image_url:
                print(f"🖼️ [extract_hero_image] Found twitter:image: {image_url[:80]}")
                return image_url
        
        # No image found
        print(f"⚠️ [extract_hero_image] No og:image or twitter:image found")
        return None
        
    except Exception as e:
        print(f"❌ [extract_hero_image] Error extracting image: {e}")
        return None


def extract_youtube_thumbnail(youtube_url: str) -> Optional[str]:
    """
    Extract YouTube video thumbnail URL from a YouTube link.
    Constructs the high-quality thumbnail: https://img.youtube.com/vi/{VIDEO_ID}/hqdefault.jpg
    """
    try:
        # Extract video ID from various YouTube URL formats
        video_id = None
        
        if "youtube.com/watch?v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        
        if video_id:
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            print(f"🎬 [extract_youtube_thumbnail] Generated thumbnail: {thumbnail_url}")
            return thumbnail_url
    except Exception as e:
        print(f"⚠️ [extract_youtube_thumbnail] Error: {e}")
    
    return None


def fetch_article_content(url: str) -> Tuple[Optional[str], str, bool, Optional[str]]:
    """
    Specialized Google News resolver with googlenewsdecoder library.
    
    Strategy:
    1) Decode Google URL using new_decoderv1 (handles Protobuf/Base64 encoding)
    2) Validate: ensure decoded URL is NOT a Google domain
    3) Check for YouTube: if video, mark is_video=True and extract thumbnail
    4) Scrape: use trafilatura.fetch_url() → trafilatura.extract() and extract_hero_image()
    
    Returns: (extracted_text or None, final_url, is_video, image_url or None)
    """
    print(f"\n🚀 [fetch_article_content] Starting with Google URL: {url[:80]}")
    final_url = url
    image_url = None
    
    # STEP 1: Decode the Google URL using the googlenewsdecoder library
    try:
        decoded_data = new_decoderv1(url)
        if decoded_data.get("status"):
            final_url = decoded_data["decoded_url"]
            print(f"✅ [googlenewsdecoder] Successfully decoded to: {final_url[:80]}")
        else:
            print(f"⚠️ [googlenewsdecoder] Decode failed (status=False). Keeping original URL.")
    except Exception as e:
        print(f"❌ [googlenewsdecoder] Decoder error: {e}. Keeping original URL.")

    # STEP 2: Safety Check - If still stuck on Google, ABORT immediately
    if "news.google.com" in final_url or "google.com" in final_url:
        print(f"⏭️ Skipping: Could not resolve from Google domain: {final_url[:80]}")
        return None, final_url, False, None

    # STEP 3: Video Detection (YouTube) - Extract thumbnail instead of scraping
    if "youtube.com" in final_url.lower() or "youtu.be" in final_url.lower():
        print(f"📹 [fetch_article_content] YouTube video detected: {final_url[:80]}")
        thumbnail = extract_youtube_thumbnail(final_url)
        return None, final_url, True, thumbnail

    # STEP 4: Scrape the actual publisher URL with trafilatura
    try:
        print(f"📄 [fetch_article_content] Fetching real article from: {final_url[:80]}")
        
        # Use trafilatura.fetch_url to handle headers, redirects, and encoding automatically
        downloaded = trafilatura.fetch_url(final_url)
        if not downloaded:
            print(f"⚠️ [trafilatura] fetch_url returned None/empty for {final_url[:80]}")
            return None, final_url, False, None
        
        # Extract text from the downloaded HTML
        text = trafilatura.extract(downloaded)
        
        # Extract hero image from the HTML (NEW)
        image_url = extract_hero_image(downloaded)
        
        # Basic validation: ensure we have sufficient text
        if text and len(text) > 200:
            print(f"✅ [fetch_article_content] Extracted {len(text)} chars from {final_url[:80]}")
            return text, final_url, False, image_url
        else:
            text_len = len(text) if text else 0
            print(f"⚠️ [fetch_article_content] Text too short ({text_len} chars) from {final_url[:80]}")
            return None, final_url, False, image_url
            
    except Exception as e:
        print(f"❌ [fetch_article_content] Scrape/extraction failed for {final_url[:80]}: {e}")
        return None, final_url, False, None


def summarize_text_if_possible(content, titles=None):
    """
    Summarize text using BART if length is sufficient.
    - If `content` is a single string → returns a single summary (string or None).
    - If `content` is a list of strings → returns a list of summaries aligned with input.
    Titles are optional, mainly used for debugging/logging.
    Skips articles with less than 600 characters as they are not proper news articles.
    Truncates input text to 2000 characters for performance.
    """
    try:
        # Truncate content before summarization
        if isinstance(content, str):
            content = content[:2000]
        elif isinstance(content, list):
            content = [c[:2000] for c in content]

        # Case 1: Single string
        if isinstance(content, str):
            # Skip articles shorter than 100 characters - not proper news articles
            if not content or len(content) < 100:
                return None
            if len(content.split()) < 30:
                return None

            summary = summarizer(
                content,
                max_length=100,
                min_length=30,
                do_sample=False,
                num_beams=6,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
            return summary[0]["summary_text"]

        # Case 2: List of strings
        elif isinstance(content, list):
            results = []
            for idx, text in enumerate(content):
                # Skip articles shorter than 100 characters - not proper news articles
                if not text or len(text) < 100:
                    results.append(None)
                    if titles and idx < len(titles):
                        print(f"⏭️ Skipped short article (len={len(text) if text else 0}): {titles[idx][:80]}")
                    continue
                if len(text.split()) < 30:
                    results.append(None)
                    continue
                summary = summarizer(
                    text,
                    max_length=100,
                    min_length=30,
                    do_sample=False,
                    num_beams=6,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True
                )
                results.append(summary[0]["summary_text"])

                # optional: log which title got summarized
                if titles and idx < len(titles):
                    snippet = summary[0]["summary_text"][:120].replace("\n", " ")
                    print(f"✅ Summarized: {titles[idx]} → {snippet}...")

            return results

        else:
            return None

    except Exception as e:
        print(f"Summarization error: {e}")
        if isinstance(content, str):
            return None
        elif isinstance(content, list):
            return [None] * len(content)


def call_groq_summarize(titles: List[str], texts: List[str]) -> List[str]:
    """
    Summarize multiple articles in one Groq API call. Returns a list of concise
    2–3 sentence paragraphs in the same order as inputs. If Groq is unavailable,
    returns empty list.
    """
    if not GROQ_API_KEY:
        return []
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        # System prompt: concise 2–3 sentence paragraphs, no bullet points
        system_prompt = (
            "You are a concise summarization assistant. Summarize news articles into clear 2–3 sentence paragraphs. "
            "No bullet points. Return ONLY the summary text, never include the original article content."
        )

        # Build a single user message containing all articles, separated by a delimiter
        # Ask model to return one paragraph per article, in order, separated by the same delimiter
        delimiter = "\n\n===\n\n"
        parts: List[str] = []
        for idx, (t, x) in enumerate(zip(titles, texts), start=1):
            safe_t = t or ""
            safe_x = x or ""
            parts.append(f"Article {idx}:\nTitle: {safe_t}\nContent: {safe_x[:2000]}") # Truncate input
        user_instruction = (
            "Summarize the following news articles. Return exactly one paragraph per article, "
            "in the same order, separated by the delimiter '" + delimiter.strip() + "'. "
            "Each summary must be 2–3 sentences (50–60 words). Do not use bullet points. "
            "IMPORTANT: Return ONLY the summary, not the original article content."
        )
        user_content = user_instruction + "\n\n" + delimiter.join(parts)

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            # Scale max tokens with batch size to stay within limits
            "max_tokens": max(1024, min(4096, max(1, len(titles)) * 80)),
        }

        # Exponential backoff for rate limit handling (429)
        retries = 3
        backoffs = [10, 20, 30]
        attempt = 0
        while True:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not msg:
                    return []
                result = msg.strip()
                # Split by delimiter into summaries; if counts mismatch, best-effort alignment
                chunks = [c.strip() for c in result.split(delimiter) if c.strip()]
                # Ensure length matches inputs; pad or trim if needed
                if len(chunks) < len(titles):
                    chunks.extend([""] * (len(titles) - len(chunks)))
                elif len(chunks) > len(titles):
                    chunks = chunks[: len(titles)]
                return chunks
            elif resp.status_code == 429 and attempt < retries:
                wait_s = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                attempt += 1
                print(f"Groq rate limit reached (429). Retrying in {wait_s}s (attempt {attempt}/{retries}).")
                time.sleep(wait_s)
                continue
            else:
                print(f"Groq summarize error {resp.status_code}: {resp.text[:200]}")
                break
    except Exception as e:
        print(f"Groq summarize exception: {e}")
    return []


def _summarize_in_batches(articles: List[dict]) -> Tuple[int, int]:
    """
    Summarize articles in batches using Groq or local model.
    Saves each summary individually to Supabase.
    Returns (num_batches, num_summarized).
    """
    to_sum = []
    for a in articles:
        title = a.get("title") or ""
        # Use the 'content' field which now holds the full text
        full_text = a.get("content", "")

        # Only add articles with at least 600 characters for summarization
        if full_text and len(full_text) >= 600:
            to_sum.append((a, title, full_text))
        elif full_text:
            print(f"⏭️ Skipping short article for summarization (len={len(full_text)}): {title[:80]}")

    if not to_sum:
        print("⚠️ No articles with sufficient text available for summarization.")
        return (0, 0)

    batch_size = 20
    batches = 0
    summarized_count = 0
    client = supabase_client() 

    for i in range(0, len(to_sum), batch_size):
        batch = to_sum[i : i + batch_size]
        titles = [t for (_, t, _) in batch]
        texts = [x for (_, _, x) in batch]

        # Summarize this batch
        summaries = summarize_text_if_possible(texts, titles)
        batches += 1

        # Align summaries and update Supabase per-article
        for (article, title, _), summary_text in zip(batch, summaries):
            if not summary_text:
                continue

            try:
                url_hash = md5_lower(article["url"])
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
    all_categories = list(RSS_TOPIC_IDS.keys()) + list(SEARCH_CATEGORIES.keys())
    for c in all_categories:
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
        # The 'content' field now holds the full scraped text.
        # Truncate to 2000 chars before sending to summarizer.
        full_text = (item.get("content") or "")[:2000]
        if full_text:
            try:
                print(f"Summary input source: content field (len={len(full_text)}) for title='{(item.get('title') or '')[:80]}'")
            except Exception:
                pass
            return full_text

        fallback = item.get("description") or ""
        try:
            src = "description" if item.get("description") else "none"
            print(f"Summary input source: {src} (len={len(fallback)}) for title='{(item.get('title') or '')[:80]}'")
        except Exception:
            pass
        return fallback

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(build_text, items))

    # Filter out articles with less than 600 characters for BART
    filtered_items = []
    filtered_titles = []
    filtered_texts = []
    
    for item, txt in zip(items, results):
        if txt and len(txt) >= 600:
            filtered_items.append(item)
            filtered_titles.append(item.get("title") or "")
            filtered_texts.append(txt)
        else:
            title = item.get("title") or ""
            print(f"⏭️ Skipping short article for summarization (len={len(txt) if txt else 0}): {title[:80]}")
    
    return filtered_titles, filtered_texts, filtered_items


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

        # Prepare texts (returns filtered items)
        titles, texts, filtered_items = _prepare_texts(combined)

        # Mark short articles (that were filtered out) as not needing summarization
        filtered_ids = {item.get("id") for item in filtered_items}
        for item in combined:
            if item.get("id") not in filtered_ids:
                # This article was filtered out (too short), mark it as not needing summarization
                try:
                    client.table("articles").update({
                        "summarization_needed": False,
                        "updated_at": "now()"
                    }).eq("id", item["id"]).execute()
                except Exception as e:
                    print(f"⚠️ Error marking short article as not needing summarization: {e}")

        # Summarize in one go (model batches over all categories)
        summaries = summarize_text_if_possible(texts, titles) or []

        # Persist results (use filtered_items which aligns with summaries)
        for item, summary_text in zip(filtered_items, summaries):
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

# --- START NEW INGESTION LOGIC ---

def fetch_rss_feed(category: str, country_code: str) -> List[dict]:
    """Fetches and parses a Google News RSS feed."""
    topic_id = RSS_TOPIC_IDS.get(category.lower())
    if not topic_id:
        return []
    
    url = RSS_URL_PATTERN.format(topic_id=topic_id, country_code=country_code)
    try:
        print(f"📡 Fetching RSS feed: {url}")
        feed = feedparser.parse(url)
        if feed.bozo:
            print(f"⚠️ Warning: Malformed feed from {url}. Reason: {feed.bozo_exception}")
        return feed.entries
    except Exception as e:
        print(f"❌ Error fetching or parsing RSS feed {url}: {e}")
        return []

def process_rss_entry(entry: dict, category: str, country: str) -> Optional[dict]:
    """Processes a single RSS entry into a dictionary for database insertion."""
    try:
        link = entry.get("link")
        if not link:
            return None

        # Extract clean description from summary HTML
        soup = BeautifulSoup(entry.get("summary", ""), "html.parser")
        description = soup.get_text(separator="\n", strip=True)

        # Attempt to decode Google News redirect to the real publisher URL (no network calls)
        final_link = link
        try:
            dec = None
            try:
                dec = new_decoderv1(link)
            except Exception as e:
                print(f"⚠️ googlenewsdecoder error for {link}: {e}")
            if dec and dec.get("status") and dec.get("decoded_url"):
                final_link = dec.get("decoded_url") or link
                print(f"🔗 Decoded RSS link to: {final_link}")
        except Exception as e:
            print(f"⚠️ Error during decode for {link}: {e}")

        is_video = False
        if final_link and ("youtube.com" in final_link.lower() or "youtu.be" in final_link.lower()):
            is_video = True

        return {
            "url": final_link,
            "original_rss_link": link,
            "title": entry.get("title", "No Title"),
            "description": description,
            "source": entry.get("source", {}).get("title"),
            "published_at": entry.get("published"),
            "category": category,
            "country": country,
            "summarization_needed": False, # Default to false, updated by logic below
            "is_video": is_video,
            "content": None, # Will be filled if scraped
            "summary": None, # Will be filled by logic below
        }
    except Exception as e:
        print(f"Error processing RSS entry: {e}")
        return None

def smart_ingest_all_categories():
    """
    Main function to fetch news from Google News RSS feeds, process, and store them.
    This replaces the old NewsAPI-based ingestion.
    """
    print("🚀 Starting smart ingestion cycle...")
    client = supabase_client()
    
    all_new_articles = []

    # 1. Fetch Global Categories
    print("--- Fetching Global Categories (Country: GLOBAL) ---")
    for category in GLOBAL_CATEGORIES:
        entries = fetch_rss_feed(category, "US")
        processed_entries = [process_rss_entry(e, category, "GLOBAL") for e in entries[:5]]
        all_new_articles.extend([p for p in processed_entries if p])
        time.sleep(2)

    # 2. Fetch Search-based Niche Categories
    print("--- Fetching Search-based Niche Categories (Country: GLOBAL) ---")
    for category, query in SEARCH_CATEGORIES.items():
        try:
            # URL-encode the query by replacing spaces with '+'
            encoded_query = query.replace(" ", "+")
            # Using "US" as the reference country for these global searches
            country_code = "US"
            search_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-{country_code}&gl={country_code}&ceid={country_code}:en"
            
            print(f"   → Fetching niche category '{category}': {search_url[:120]}...")
            feed = feedparser.parse(search_url)
            entries = feed.entries if hasattr(feed, 'entries') else []
            print(f"   → Found {len(entries)} entries for '{category}'")
            
            processed_entries = [process_rss_entry(e, category, "GLOBAL") for e in entries[:5]]
            all_new_articles.extend([p for p in processed_entries if p])
            time.sleep(2)
        except Exception as e:
            print(f"   ⚠️ Niche category fetch failed for '{category}': {e}")

    # 3. Fetch Local Categories for each active country
    for country_code in ACTIVE_COUNTRIES:
        print(f"--- Fetching Local Categories for Country: {country_code} ---")
        for category in LOCAL_CATEGORIES:
            entries = []
            
            # SPECIAL HANDLING: Sports category gets dual-fetch (local + global)
            if category == "sports":
                print(f"⚽ [Sports] Performing dual-fetch: Local + International")
                
                # Fetch 1: Local Sports (Cricket, local tournaments, etc.)
                local_entries = fetch_rss_feed(category, country_code)
                print(f"   → Local Sports entries: {len(local_entries)}")
                
                # Fetch 2: Global search for international sports (Football, F1, Tennis)
                try:
                    global_query = "Football OR Soccer OR Premier League OR Champions League OR F1 OR Tennis"
                    encoded_query = global_query.replace(" ", "+")
                    global_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-{country_code.lower()}&gl={country_code}&ceid={country_code}:en"
                    print(f"   → Fetching global search: {global_url[:80]}...")
                    global_feed = feedparser.parse(global_url)
                    global_entries = global_feed.entries if hasattr(global_feed, 'entries') else []
                    print(f"   → Global Sports entries: {len(global_entries)}")
                except Exception as e:
                    print(f"   ⚠️ Global sports fetch failed: {e}")
                    global_entries = []
                
                # Merge: Take top 4 from local + top 4 from global
                entries = local_entries[:4] + global_entries[:4]
                print(f"   ✅ Combined entries (local + global): {len(entries)}")
            else:
                # Standard fetch for non-sports categories
                entries = fetch_rss_feed(category, country_code)
            
            processed_entries = [process_rss_entry(e, category, country_code) for e in entries[:5]]
            all_new_articles.extend([p for p in processed_entries if p])

    if not all_new_articles:
        print("No new articles found in this cycle.")
        return

    # 3. Deduplication
    print(f"Found {len(all_new_articles)} potential articles. Checking for duplicates...")
    existing_urls_hashes = set()
    try:
        # Fetch all url_hash values from the table. This might be slow on very large tables.
        # An alternative is to check one by one, but that's many queries.
        # For now, this is a reasonable trade-off.
        result = client.table("articles").select("url_hash").execute()
        existing_urls_hashes = {item['url_hash'] for item in result.data}
    except Exception as e:
        print(f"⚠️ Could not fetch existing URLs for deduplication: {e}")

    unique_articles = []
    for article in all_new_articles:
        url_hash = md5_lower(article["url"])
        if url_hash not in existing_urls_hashes:
            article["url_hash"] = url_hash
            unique_articles.append(article)
            existing_urls_hashes.add(url_hash) # Add to set to handle duplicates within the same batch

    print(f"Found {len(unique_articles)} unique articles to process.")
    if not unique_articles:
        return

    # 4. Processing and Saving Core Article Data (in parallel)
    def process_and_scrape(article: dict):
        """
        Fetches full content, resolves final URL, and saves the initial article
        data to Supabase. It flags the article if it needs summarization later.
        """
        client = supabase_client()
        url = article["url"]
        
        try:
            text, real_url, is_video, image_url = fetch_article_content(url)
        except Exception as e:
            print(f"⚠️ Error during fetch_article_content for {url}: {e}")
            text, real_url, is_video, image_url = None, url, False, None

        # Prepare a dictionary with the data we intend to save
        article_data = article.copy()

        # Update article URL to resolved URL and clean it
        if real_url:
            article_data["url"] = real_url.split('?')[0]
        
        # Store extracted image
        if image_url:
            article_data["image_url"] = image_url

        # Handle video content
        if is_video:
            article_data["is_video"] = True
            article_data["summarization_needed"] = False
            article_data["summary"] = article_data.get("description")
            article_data["content"] = ""
            print(f"📹 Detected video, skipping summarization: {article_data.get('title')}")
        else:
            # Store full content and decide if summarization is needed
            final_content = (text if text and len(text) > 500 else (article_data.get("description") or ""))
            article_data["content"] = final_content
            
            if final_content and len(final_content) >= 600:
                article_data["summarization_needed"] = True
                print(f"📝 Article queued for summarization: {article_data.get('title')}")
            else:
                article_data["summarization_needed"] = False
                article_data["summary"] = article_data.get("description") # Fallback for short content
                print(f"📝 Text too short, using description as summary: {article_data.get('title')}")

        # Immediately upsert the article to the database
        try:
            # From supabase-schema.sql, these are the valid columns.
            # url_hash is generated by the DB, so we don't send it.
            # Fields like 'original_rss_link' are excluded.
            row_to_insert = {
                "url": article_data.get("url"),
                "title": article_data.get("title"),
                "description": article_data.get("description"),
                "content": article_data.get("content"),
                "source": article_data.get("source"),
                "image_url": article_data.get("image_url"),
                "category": article_data.get("category"),
                "country": article_data.get("country"),
                "is_video": article_data.get("is_video", False),
                "published_at": article_data.get("published_at"),
                "summary": article_data.get("summary"),
                "summarized": article_data.get("summarized", False),
                "summarization_needed": article_data.get("summarization_needed", True),
            }

            # Filter out keys with None values to avoid inserting NULLs unnecessarily
            row_to_insert = {k: v for k, v in row_to_insert.items() if v is not None}
            
            # Using on_conflict='title' as url_hash is generated and cannot be in ON CONFLICT.
            # This matches the behavior of other insert operations in this file.
            client.table("articles").upsert(row_to_insert, on_conflict="title").execute()
            print(f"✅ Saved initial data for: {article_data.get('title')[:60]}")
        except Exception as e:
            print(f"❌ DB insert/update failed for {article_data.get('title')}: {e}")

    # Run the scraping and saving process in parallel for each unique article
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_and_scrape, unique_articles)

    print("🏁 Smart ingestion cycle complete. Summaries will be processed by the background worker.")

# --- END NEW INGESTION LOGIC ---


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


from fastapi import Depends, Header

# --- START AUTH HELPERS ---

# Dependency to get user from JWT in Authorization header
def _get_authenticated_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
        
    try:
        # Use the standard client (with anon key) to validate the user's JWT
        client = supabase_client()
        user_response = client.auth.get_user(token)
        return user_response.user
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token or user not found")

# --- END AUTH HELPERS ---

@app.delete("/users/me", status_code=204)
def delete_user(user = Depends(_get_authenticated_user)):
    """
    Deletes the currently authenticated user's account.
    This is a protected endpoint that requires a valid JWT.
    It uses the service_role key to perform the admin deletion.
    """
    if not user or not user.id:
        raise HTTPException(status_code=400, detail="Could not identify user to delete")
    
    try:
        # Create a new client with service_role key for admin operations
        admin_client = supabase_client()
        
        print(f"🛡️ Admin action: Deleting user with ID: {user.id}")
        
        # Perform the deletion
        admin_client.auth.admin.delete_user(user.id)
        
        print(f"✅ Successfully deleted user with ID: {user.id}")
        
        # Return a 204 No Content response, which is appropriate for a successful DELETE
        return

    except Exception as e:
        print(f"❌ Error deleting user {user.id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting the account")

@app.get("/health")
def health():
    return {"status": "ok"}


def _fetch_prioritized_articles(
    client,
    base_query,
    page: int,
    page_size: int,
):
    # Only fetch summarized articles to ensure a summary is always available.
    offset = (page - 1) * page_size
    summarized_query = (
        base_query.eq("summarized", True)
        .order("created_at", desc=True)
        .range(offset, offset + page_size - 1)
    )
    summarized_res = summarized_query.execute()
    items = summarized_res.data or []

    # The fallback to fetch non-summarized articles has been removed
    # to guarantee that only articles with summaries are sent to the client.

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
                # Ensure the 'content' field for the API response contains the summary,
                # so the app displays the summary instead of the full article text.
                content=r.get("summary"),
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


# Configure scheduler with UTC timezone for consistency
scheduler = BackgroundScheduler(timezone="UTC")


@app.on_event("startup")
def schedule_jobs():
    # Run news ingestion immediately on startup, then every 15 minutes
    scheduler.add_job(smart_ingest_all_categories, "interval", minutes=15, id="ingest_news", replace_existing=True)
    
    # Add periodic round-robin summarizer (keeps summaries flowing across categories)
    scheduler.add_job(
        summarize_pending_round_robin,
        "interval",
        minutes=3,
        id="summarize_rr",
        replace_existing=True,
        kwargs={"per_category_limit": 2, "max_cycles": 10}
    )

    # Run quiz generation daily at 2 AM UTC
    scheduler.add_job(generate_daily_quiz_questions, "cron", hour=2, minute=0, id="generate_quiz", replace_existing=True, timezone="UTC")
    
    # Run riddle generation daily at 11:59 PM UTC (before midnight so it's ready when app refreshes)
    def scheduled_riddle_generation():
        """Wrapper function for scheduled riddle generation with error handling and logging"""
        try:
            print(f"🕛 [SCHEDULER] Running scheduled daily riddle generation at {datetime.now().isoformat()} UTC")
            result = generate_daily_riddle()
            if result:
                print(f"✅ [SCHEDULER] Successfully generated riddle: {result.get('id', 'unknown')}")
            else:
                print(f"⚠️ [SCHEDULER] Riddle generation returned None - may have failed or already exists")
        except Exception as e:
            print(f"❌ [SCHEDULER] Error in scheduled riddle generation: {e}")
            import traceback
            traceback.print_exc()
    
    scheduler.add_job(scheduled_riddle_generation, "cron", hour=23, minute=59, id="generate_riddle", replace_existing=True, timezone="UTC")
    
    scheduler.start()
    
    # Kick off initial runs asynchronously
    try:
        import threading
        # Start news ingestion
        threading.Thread(target=smart_ingest_all_categories, daemon=True).start()
        
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
    # Filter across title/description/content, but only for summarized articles
    res = (
        client.table("articles")
        .select("*")
        .eq("summarized", True)
        .or_(f"title.ilike.{pattern},description.ilike.{pattern},content.ilike.{pattern}")
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
                content=r.get("summary"), # Use summary for content
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
            # Only summarize if content exists and is at least 100 characters
            if a.content and len(a.content) >= 100:
                summary = summarize_text_if_possible(a.content, a.title)
            elif a.content:
                print(f"⏭️ Skipping short article in ingest (len={len(a.content)}): {a.title[:80]}")
            # Clean the URL by stripping query parameters to keep DB tidy
            clean_url = (a.url or "").split('?')[0]
            row = {
                "url": clean_url,
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
            # Upsert by title to rely on UNIQUE(title) constraint and update existing rows
            client.table("articles").upsert(row, on_conflict="title").execute()
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


@app.get("/.well-known/assetlinks.json")
def get_assetlinks():
    """
    Android App Links verification file.
    Replace YOUR_SHA256_FINGERPRINT with your app's actual SHA-256 fingerprint.
    Get it using: keytool -list -v -keystore ~/.android/debug.keystore -alias androiddebugkey
    """
    return [{
        "relation": ["delegate_permission/common.handle_all_urls"],
        "target": {
            "namespace": "android_app",
            "package_name": "com.readdio.app",
            "sha256_cert_fingerprints": [
                "YOUR_SHA256_FINGERPRINT_HERE"  # TODO: Replace with actual fingerprint
            ]
        }
    }]

@app.get("/.well-known/apple-app-site-association")
def get_apple_app_site_association():
    """
    iOS Universal Links verification file.
    Replace TEAM_ID with your Apple Developer Team ID.
    """
    return {
        "applinks": {
            "apps": [],
            "details": [
                {
                    "appID": "TEAM_ID.com.readdio.app",  # TODO: Replace TEAM_ID with actual Team ID
                    "paths": ["/article/*"]
                }
            ]
        }
    }

@app.get("/article/{encoded_url:path}")
def redirect_to_article(encoded_url: str, request):
    """
    Redirect endpoint for article deep links.
    Accepts base64-encoded article URL and redirects to app deep link.
    Falls back to Play Store/App Store if app is not installed.
    """
    try:
        import base64
        
        # Decode the base64 URL (URL-safe base64)
        # Replace URL-safe characters back to standard base64
        base64_url = encoded_url.replace('-', '+').replace('_', '/')
        # Add padding if needed
        padding = (4 - len(base64_url) % 4) % 4
        padded = base64_url + '=' * padding
        
        # Decode base64 to get original article URL (for reference, not used in fallback)
        decoded_bytes = base64.b64decode(padded)
        article_url = decoded_bytes.decode('utf-8')
        
        # Create deep link for the app
        deep_link = f"readdio://article/{encoded_url}"
        
        # Detect user agent to determine platform
        user_agent = request.headers.get("user-agent", "").lower()
        is_android = "android" in user_agent
        is_ios = "iphone" in user_agent or "ipad" in user_agent or "ipod" in user_agent
        
        # App Store URLs
        play_store_url = "https://play.google.com/store/apps/details?id=com.readdio.app"  # Package name: com.readdio.app
        app_store_url = "https://apps.apple.com/app/readdio/id0000000000"  # TODO: Replace with actual App Store ID when app is published
        
        # Determine which store to redirect to
        store_url = app_store_url if is_ios else play_store_url
        
        # Return HTML page with JavaScript to try app deep link, then fallback to store
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Open in Readdio</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-align: center;
                    padding: 20px;
                }}
                .container {{
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 500px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    margin: 0 0 20px 0;
                    font-size: 32px;
                }}
                p {{
                    margin: 10px 0;
                    font-size: 16px;
                    opacity: 0.9;
                }}
                .button {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 12px 24px;
                    background: white;
                    color: #667eea;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: bold;
                    transition: transform 0.2s;
                }}
                .button:hover {{
                    transform: scale(1.05);
                }}
                .spinner {{
                    border: 3px solid rgba(255, 255, 255, 0.3);
                    border-top: 3px solid white;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📰 Open in Readdio</h1>
                <p>Opening article in the Readdio app...</p>
                <div class="spinner"></div>
                <p style="font-size: 14px; opacity: 0.7; margin-top: 20px;">
                    Don't have the app? <a href="{store_url}" style="color: white; text-decoration: underline; font-weight: bold;">Download from {"App Store" if is_ios else "Play Store"}</a>
                </p>
            </div>
            <script>
                var appOpened = false;
                var startTime = Date.now();
                
                // Function to check if app opened (page becomes hidden)
                function checkAppOpened() {{
                    if (document.hidden || document.webkitHidden) {{
                        appOpened = true;
                        return true;
                    }}
                    return false;
                }}
                
                // Listen for page visibility changes
                document.addEventListener('visibilitychange', function() {{
                    if (document.hidden) {{
                        appOpened = true;
                    }}
                }});
                
                // Try to open the app deep link
                window.location.href = "{deep_link}";
                
                // Check if app opened immediately
                setTimeout(function() {{
                    if (!checkAppOpened()) {{
                        // App didn't open, redirect to store
                        window.location.href = "{store_url}";
                    }}
                }}, 1500);
                
                // Additional check after a longer delay
                setTimeout(function() {{
                    if (!appOpened && !document.hidden) {{
                        window.location.href = "{store_url}";
                    }}
                }}, 2500);
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        print(f"❌ Error processing article redirect: {e}")
        # Return error page
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Error - Readdio</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                    background: #f5f5f5;
                    text-align: center;
                    padding: 20px;
                }
                .container {
                    background: white;
                    border-radius: 12px;
                    padding: 40px;
                    max-width: 400px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                h1 { color: #333; }
                p { color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚠️ Error</h1>
                <p>Unable to process this link. Please try again.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)


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
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("completed_at", today_start.isoformat())
            .execute()
        )
        used_attempts = result.count
        next_reset = (today_start + timedelta(days=1)).isoformat()
        
    elif quiz_type == "weekly":
        max_attempts = 5
        # Count attempts from this week (Monday 00:00)
        today = now.date()
        monday = today - timedelta(days=today.weekday())
        week_start = datetime.combine(monday, datetime.min.time())
        result = (
            client.table("quiz_attempts")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("completed_at", week_start.isoformat())
            .execute()
        )
        used_attempts = result.count
        next_monday = monday + timedelta(days=7)
        next_reset = datetime.combine(next_monday, datetime.min.time()).isoformat()
        
    elif quiz_type == "monthly":
        max_attempts = 5
        # Count attempts from this month (1st day 00:00)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = (
            client.table("quiz_attempts")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("quiz_type", quiz_type)
            .gte("completed_at", month_start.isoformat())
            .execute()
        )
        used_attempts = result.count
        # Next reset is the first day of the next month
        next_month_start = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        next_reset = next_month_start.isoformat()
        
    else:
        return {"remaining_attempts": 0, "max_attempts": 0, "next_reset": None, "can_attempt": False}

    # Check for ad unlocks
    ad_unlock_result = (
        client.table("ad_attempts")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("quiz_type", quiz_type)
        .gte("unlock_date", _get_period_start(quiz_type, now).isoformat())
        .execute()
    )
    ad_unlocks = ad_unlock_result.count
    
    # Total attempts available = 1 free + ad unlocks
    total_available = 1 + ad_unlocks
    remaining = max(0, total_available - used_attempts)
    
    return {
        "remaining_attempts": remaining,
        "max_attempts": max_attempts,
        "next_reset": next_reset,
        "can_attempt": remaining > 0
    }

def _get_user_wrong_questions(client, user_id: str, limit: int, since: str) -> List[dict]:
    """Fetch questions the user answered incorrectly."""
    try:
        res = (
            client.table("user_question_performance")
            .select("question_id")
            .eq("user_id", user_id)
            .eq("is_correct", False)
            .gte("attempted_at", since)
            .order("attempted_at", desc=True)
            .limit(limit)
            .execute()
        )
        q_ids = [row["question_id"] for row in res.data or []]
        if not q_ids:
            return []
        
        # Fetch full question details
        questions_res = client.table("quiz_questions").select("*").in_("id", q_ids).execute()
        return questions_res.data or []
    except Exception as e:
        print(f"Error fetching wrong answers: {e}")
        return []

def _randomize_question_options(question: dict) -> dict:
    """Randomize the order of options and update the correct_answer index."""
    try:
        options = json.loads(question.get("options", "[]"))
        correct_index = question.get("correct_answer", 0)
        
        if not isinstance(options, list) or correct_index >= len(options):
            return question

        correct_option_text = options[correct_index]
        
        random.shuffle(options)
        
        new_correct_index = options.index(correct_option_text)
        
        question["options"] = json.dumps(options)
        question["correct_answer"] = new_correct_index
        return question
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Error randomizing options for question {question.get('id')}: {e}")
        return question