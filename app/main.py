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
import trafilatura
from googlenewsdecoder import new_decoderv1
from riddle_generator import generate_daily_riddle, get_latest_riddle, check_today_riddle_exists
from LocalLLMService import get_local_llm_service
from debate_service import router as debate_router
import firebase_admin
from firebase_admin import credentials, messaging




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

# Initialize LocalLLMService (Gemma-2-2b-it)
try:
    llm_service = get_local_llm_service()
    print("✅ LocalLLMService initialized successfully")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize LocalLLMService")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()
    print("⚠️ WARNING: Falling back to None - articles will NOT be summarized!")
    llm_service = None

# Initialize Firebase Admin SDK
try:
    # Use the specific service account file provided by the user
    FB_SVC_PATH = os.path.join(os.path.dirname(__file__), "readdio-firebase-adminsdk-fbsvc-559cbc2036.json")
    if os.path.exists(FB_SVC_PATH):
        if not firebase_admin._apps:
            cred = credentials.Certificate(FB_SVC_PATH)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized successfully")
        else:
            print("ℹ️ Firebase Admin SDK already initialized, reusing existing app")
    else:
        print(f"⚠️ Firebase service account not found at {FB_SVC_PATH}. FCM disabled.")
except Exception as e:
    print(f"❌ Failed to initialize Firebase Admin SDK: {e}")

# Create FastAPI app before any route decorators are declared.
app = FastAPI()

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

class ProfileUpdate(BaseModel):
    user_id: str
    display_name: str

def is_valid_display_name(name: str) -> Tuple[bool, str]:
    if not name or len(name.strip()) < 3:
        return False, "Display name must be at least 3 characters."
    
    if name.strip().isdigit():
        return False, "Display name cannot consist of only numbers."
    
    # Very basic profanity check 
    bad_words = ["admin", "root", "moderator", "fuck", "shit", "bitch"]
    name_lower = name.lower()
    for word in bad_words:
        if word in name_lower:
             return False, "Display name contains inappropriate language."

    return True, ""

@app.post("/user/update_display_name")
def update_display_name(req: ProfileUpdate):
    cleaned_name = req.display_name.strip()
    is_valid, error_msg = is_valid_display_name(cleaned_name)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        client = supabase_client()

        try:
            # Preferred schema: display_name + full_name.
            client.table("profiles").update({
                "display_name": cleaned_name,
                "full_name": cleaned_name
            }).eq("id", req.user_id).execute()
        except Exception as update_err:
            err_text = str(update_err).lower()
            missing_display_col = ("display_name" in err_text) and (
                "column" in err_text or "schema cache" in err_text
            )
            if missing_display_col:
                # Legacy schema fallback: update full_name only.
                client.table("profiles").update({
                    "full_name": cleaned_name
                }).eq("id", req.user_id).execute()
            else:
                raise update_err

        return {"message": "Display name updated successfully"}
    except Exception as e:
        err_text = str(e).lower()
        if "duplicate key value" in err_text or "unique constraint" in err_text:
            raise HTTPException(status_code=409, detail="Display name is already taken.")
        print(f"❌ Error updating display name: {e}")
        raise HTTPException(status_code=500, detail="Failed to update display name")

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
            print(f"🔗 Resolved redirect: {str(url)[:80]}... → {str(final_url)[:80]}...")
        return final_url
    except Exception as e:
        print(f"⚠️ Redirect resolution failed for {str(url)[:80]}: {e}")
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
    safe_url = str(url)
    print(f"\n🚀 [fetch_article_content] Starting with Google URL: {safe_url[:80]}")
    final_url = safe_url
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
    low_url = final_url.lower()
    if "youtube.com" in low_url or "youtu.be" in low_url:
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
    Summarize text using LocalLLMService (Gemma-2-2b-it).
    Returns summary + UPSC relevance + tags.
    - If `content` is a single string → returns a single Dict or None.
    - If `content` is a list of strings → returns a list of Dicts aligned with input.
    Titles are optional, mainly used for debugging/logging.
    No longer skips short articles; sends everything to the local model.
    """
    if llm_service is None:
        print("❌ CRITICAL: LocalLLMService is None - model not loaded!")
        print("   Check startup logs for initialization errors.")
        print("   Skipping summarization for this batch.")
        if isinstance(content, str):
            return None
        content_len = len(content) if isinstance(content, list) else 0
        return [None] * content_len
    
    try:
        # Case 1: Single string
        if isinstance(content, str):
            if not content or not content.strip():
                return None
            
            result = llm_service.analyze_article(content)
            if result:
                return result
            return None
        
        # Case 2: List of strings
        elif isinstance(content, list):
            results = []
            for idx, text in enumerate(content):
                if not text or not text.strip():
                    results.append(None)
                    if titles and idx < len(titles):
                        print(f"⏭️ Skipped empty article: {titles[idx][:80]}")
                    continue
                
                result = llm_service.analyze_article(text)
                if result:
                    results.append(result)
                    # Log summary for debug
                    if titles and idx < len(titles):
                        summary = result.get("summary", "")[:80].replace("\n", " ")
                        relevant = result.get("upsc_relevant", False)
                        tags = result.get("tags", [])
                        print(f"✅ Analyzed: {titles[idx][:60]} → Relevant: {relevant}, Tags: {tags}")
                else:
                    results.append(None)
            
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
    DEPRECATED: This function is no longer used.
    All summarization is now handled by LocalLLMService (Gemma-2-2b-it).
    
    This function remains for backward compatibility but should NOT be called.
    Returns empty list immediately to prevent any accidental Groq API usage.
    """
    print("⚠️ WARNING: call_groq_summarize() called but is DISABLED!")
    print("   Summarization is now exclusively using LocalLLMService (Gemma-2-2b-it).")
    print("   Returning empty list to prevent Groq API usage.")
    return []


def _summarize_in_batches(articles: List[dict]) -> Tuple[int, int]:
    """
    Summarize articles in batches using LocalLLMService.
    Saves each summary individually to Supabase along with UPSC tags.
    Only processes articles with content > 600 characters.
    Returns (num_batches, num_summarized).
    """
    to_sum = []
    for a in articles:
        title = a.get("title") or ""
        # Use the 'content' field which now holds the full text
        full_text = a.get("content", "")

        # GEMMA FILTER: Only take articles with more than 600 characters
        if full_text and len(full_text.strip()) > 600:
            to_sum.append((a, title, full_text))
        elif full_text and len(full_text.strip()) <= 600:
            print(f"⏭️ Rejecting short article ({len(full_text.strip())} chars): {title[:80]}")
        elif not full_text:
            print(f"⏭️ Skipping empty article: {title[:80]}")

    if not to_sum:
        print("⚠️ No articles with >600 characters available for summarization.")
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
        for article_data, result in zip(batch, summaries):
            article, title, _ = article_data
            if result is None:
                continue

            try:
                # Extract fields from LocalLLMService result
                summary_text = result.get("summary", "")
                upsc_relevant = result.get("upsc_relevant", False)
                tags = result.get("tags", [])

                # FIX: Force tags to None if not relevant
                if not upsc_relevant:
                    tags = None
                # Also force None if the list is empty (to avoid empty badges in UI)
                elif isinstance(tags, list) and not tags:
                    tags = None

                if not summary_text:
                    continue

                # Build update dict
                update_dict = {
                    "summary": summary_text,
                    "summarized": True,
                    "summarization_needed": False,
                    "updated_at": "now()"
                }

                # Add UPSC-specific fields if they exist in DB
                if upsc_relevant is not None:
                    update_dict["upsc_relevant"] = upsc_relevant
                if tags is not None:
                    update_dict["tags"] = json.dumps(tags)
                else:
                    # Explicitly set NULL in DB to clear any old tags
                    update_dict["tags"] = None
                
                url_hash = md5_lower(article["url"])
                client.table("articles").update(update_dict).eq("url_hash", url_hash).execute()

                summarized_count += 1
                tag_str = ", ".join(tags) if tags else "N/A"
                print(f"✅ Saved summary for: {title[:80]} | Relevant: {upsc_relevant} | Tags: {tag_str}")
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

    # No longer filter by 600 characters - LocalLLMService processes everything
    filtered_items = []
    filtered_titles = []
    filtered_texts = []
    
    for item, txt in zip(items, results):
        if txt and txt.strip():
            filtered_items.append(item)
            filtered_titles.append(item.get("title") or "")
            filtered_texts.append(txt)
        else:
            title = item.get("title") or ""
            print(f"⏭️ Skipping empty article: {title[:80]}")
    
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

        # With LocalLLMService, we process everything, so no need to mark short articles
        # All articles in combined should be in filtered_items

        # Summarize in one go (model processes all texts)
        summaries = summarize_text_if_possible(texts, titles) or []

        # Persist results (use filtered_items which aligns with summaries)
        for item, result in zip(filtered_items, summaries):
            if not result:
                continue
            try:
                summary_text = result.get("summary", "")
                upsc_relevant = result.get("upsc_relevant", False)
                tags = result.get("tags")
                
                if not summary_text:
                    continue
                
                update_dict = {
                    "summary": summary_text,
                    "summarized": True,
                    "summarization_needed": False,
                    "updated_at": "now()"
                }
                
                if upsc_relevant is not None:
                    update_dict["upsc_relevant"] = upsc_relevant
                if tags is not None:
                    update_dict["tags"] = json.dumps(tags)
                
                client.table("articles").update(update_dict).eq("id", item["id"]).execute()
                total_updated += 1
            except Exception as e:
                print(f"Save summary error for id={item.get('id')}: {e}")

    print(f"✅ Round‑robin summarization complete: cycles={cycles}, updated={total_updated}")

def clean_existing_quiz_options():
    print("🧹 Starting one-time cleanup of Quiz Options...")
    client = supabase_client()
    
    # 1. Fetch all questions
    # We page through them to avoid timeouts if you have thousands
    page = 0
    page_size = 100
    total_cleaned = 0
    
    while True:
        try:
            start = page * page_size
            end = start + page_size - 1
            
            res = client.table("quiz_questions").select("id, options").range(start, end).execute()
            questions = res.data or []
            
            if not questions:
                break # Done
            
            print(f"Processing batch {page + 1} ({len(questions)} questions)...")
            
            for q in questions:
                q_id = q['id']
                raw_options = q['options']
                
                # Parse JSON if it's a string, otherwise use as list
                if isinstance(raw_options, str):
                    try:
                        options_list = json.loads(raw_options)
                    except:
                        print(f"⚠️ Skipping invalid JSON for {q_id}")
                        continue
                elif isinstance(raw_options, list):
                    options_list = raw_options
                else:
                    continue

                # 2. Clean the Prefixes (The Magic Part)
                # Regex looks for: Start -> (Optional Space) -> (Letter/Number) -> (Dot/Bracket) -> (Space)
                # Examples handled: "A) Text", "1. Text", "(a) Text", "A. Text"
                new_options = []
                changed = False
                
                for opt in options_list:
                    # Clean the string
                    cleaned = re.sub(r'^[\s\(]*[A-Za-z0-9][\)\.\-]\s+', '', opt).strip()
                    new_options.append(cleaned)
                    
                    if cleaned != opt:
                        changed = True
                
                # 3. Update DB only if changes were made
                if changed:
                    # Save back as JSON string to be safe
                    client.table("quiz_questions").update({
                        "options": json.dumps(new_options) 
                    }).eq("id", q_id).execute()
                    total_cleaned += 1
            
            page += 1
            
        except Exception as e:
            print(f"❌ Error in cleanup loop: {e}")
            break
            
    print(f"✅ Cleanup Complete. Fixed {total_cleaned} questions.")

# --- START NEW INGESTION LOGIC ---

def sanitize_text(text: Optional[str]) -> Optional[str]:
    """
    Sanitizes text by checking for emptiness, whitespace-only, and minimum length.
    Returns None if text is invalid, otherwise returns the original stripped text.
    """
    if text is None:
        return None
    s_text = text.strip()
    if not s_text or len(s_text) < 30:
        return None
    return s_text


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

        # Extract clean description from summary HTML, removing related coverage links.
        summary_html = entry.get("summary", "")
        soup = BeautifulSoup(summary_html, "html.parser")

        # The related links are usually at the end in a list. Decompose them.
        for tag in soup.find_all(['ul', 'ol', 'table']):
            tag.decompose()
            
        # The main description is what's left.
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
        data to Supabase. It implements strict data integrity checks.
        """
        client = supabase_client()
        url = article["url"]

        try:
            scraped_text, real_url, is_video, image_url = fetch_article_content(url)
        except Exception as e:
            print(f"⚠️ Error during fetch_article_content for {url}: {e}")
            scraped_text, real_url, is_video, image_url = None, url, False, None

        # --- Start Refactored Logic ---
        
        # 1. Sanitize scraped text and RSS description
        content = sanitize_text(scraped_text)
        desc = sanitize_text(article.get("description"))

        # 2. CRITICAL CHECK: If both content and description are invalid, abort.
        if content is None and desc is None:
            print(f"⏭️ Skipping article (no content): {article.get('title', 'No Title')[:80]}")
            return # DO NOT SAVE

        article_data = article.copy()

        # Update URL and image
        if real_url:
            article_data["url"] = real_url.split('?')[0]
        if image_url:
            article_data["image_url"] = image_url
        
        # Handle videos separately
        if is_video:
            article_data["is_video"] = True
            article_data["content"] = None # No content for videos
            article_data["summary"] = desc # Use sanitized description if available
            article_data["summarized"] = desc is not None
            article_data["summarization_needed"] = False
            print(f"📹 Detected video, saving with description as summary: {article.get('title')}")
        else:
            # 3. Flag Logic for regular articles (with LocalLLMService, require 600+ characters)
            summary = None
            summarization_needed = False

            # If we have content, check if it meets the 600-character minimum for Gemma
            if content:
                if len(content.strip()) > 600:
                    # Content is sufficient - queue for summarization
                    summarization_needed = True
                    article_data["content"] = content
                    print(f"📝 Article has {len(content.strip())} chars, queued for LocalLLM summarization: {article.get('title')}")
                else:
                    # Content is too short - reject and don't save
                    print(f"⏭️ Rejecting article ({len(content.strip())} chars < 600): {article.get('title')}")
                    return
            
            # If no content but have description, check its length
            elif desc:
                if len(desc.strip()) > 600:
                    # Description is sufficient - use it
                    summary = desc
                    article_data["content"] = desc
                    print(f"📝 Using long description as summary ({len(desc.strip())} chars): {article.get('title')}")
                else:
                    # Description too short - reject and don't save
                    print(f"⏭️ Rejecting article (description {len(desc.strip())} chars < 600): {article.get('title')}")
                    return
            else:
                # No content and no description - nothing to work with
                print(f"⏭️ Skipping article (no content or description): {article.get('title')}")
                return


            # Final flag assignment
            article_data["summary"] = summary
            article_data["summarized"] = summary is not None
            article_data["summarization_needed"] = summarization_needed

        # --- End Refactored Logic ---

        # Immediately upsert the article to the database
        try:
            # Handle tags: convert empty list or non-relevant to None so DB stores NULL
            tags = article_data.get("tags")
            upsc_relevant = article_data.get("upsc_relevant", False)
            if not upsc_relevant or not tags:
                tags_processed = None
            else:
                tags_processed = tags

            row_to_insert = {
                "url": article_data.get("url"),
                "title": article_data.get("title"),
                "description": article_data.get("description"), # Keep original description
                "content": article_data.get("content"),
                "source": article_data.get("source"),
                "image_url": article_data.get("image_url"),
                "category": article_data.get("category"),
                "country": article_data.get("country"),
                "is_video": article_data.get("is_video", False),
                "published_at": article_data.get("published_at"),
                "summary": article_data.get("summary"),
                "summarized": article_data.get("summarized", False),
                "summarization_needed": article_data.get("summarization_needed", False),
                # include tags explicitly (may be None to force NULL in DB)
                "tags": (json.dumps(tags_processed) if tags_processed is not None else None),
            }

            # Keep all non-None values, but keep 'tags' even if it's None (to write NULL)
            row_to_insert = {k: v for k, v in row_to_insert.items() if v is not None or k == "tags"}
            
            client.table("articles").upsert(row_to_insert, on_conflict="title").execute()
            print(f"✅ Saved initial data for: {article_data.get('title', 'No Title')[:60]}")
        except Exception as e:
            print(f"❌ DB insert/update failed for {article_data.get('title', 'No Title')}: {e}")

    # Run the scraping and saving process in parallel for each unique article
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_and_scrape, unique_articles)

    print("🏁 Smart ingestion cycle complete. Summaries will be processed by the background worker.")

# --- END NEW INGESTION LOGIC ---


def fetch_recent_summarized_articles(limit: int = 50) -> List[dict]:
    """
    Fetch a batch of summarized articles that have not yet been used for a quiz.
    - Fetches articles where summarized is TRUE and quiz_generated is FALSE.
    - Orders by newest first to get the most recent content.
    - Filters for UPSC-relevant categories.
    """
    client = supabase_client()
    
    # Categories relevant for UPSC quiz generation
    relevant_categories = [
        "politics", "world", "business", "science", "environment", "general"
    ]
    
    try:
        result = (
            client.table("articles")
            .select("title, summary, category") # Select only needed columns
            .eq("summarized", True)
            .eq("quiz_generated", False) # Only fetch articles not yet processed
            .in_("category", relevant_categories)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"Error fetching summarized articles for quiz: {e}")
        return []


def call_groq_generate_quiz_questions(articles: List[dict]) -> List[dict]:
    """
    Generate UPSC-style quiz questions from summarized articles using Groq API.
    Returns a list of question objects with all required fields.
    """
    # New behavior: call Groq once per article, send title + first 1500 chars of summary,
    # expect a single JSON object per response, and only keep results where is_relevant==True.
    if not GROQ_API_KEY or not articles:
        return []

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # New system prompt (single-article)
    system_prompt = (
        "You are a Senior Question Setter for the UPSC Civil Services Exam (Prelims). "
        "Your task is to analyze a single news article and generate a high-quality, syllabus-relevant MCQ.\n\n"
        "### PHASE 1: RELEVANCE FILTER\n"
        "Analyze the news. Return 'is_relevant': false IF:\n"
        "1. Political/Partisan (Rallies, allegations).\n"
        "2. Irrelevant Foreign News (US/UK domestic issues) UNLESS involving India/UN/Climate.\n"
        "3. Trivial/Sports/Entertainment (Cricket scores, movie awards).\n"
        "4. Corporate (Stock prices, quarterly results).\n\n"
        "### PHASE 2: FORMAT SELECTION (If Relevant)\n"
        "Select the best format:\n"
        "**TYPE A: Standard Statements** (2-3 statements). Logic: Swap facts/ministries to create traps.\n"
        "**TYPE B: Pair Matching** (3 pairs, e.g., Place:Country). Logic: Mismatch 1-2 pairs.\n"
        "**TYPE C: Direct/Term** (Fact-based). Logic: 4 confusing options.\n\n"
        "### PHASE 3: JSON OUTPUT\n"
        "Return a SINGLE JSON object:\n"
        "{\n"
        "  \"is_relevant\": boolean,\n"
        "  \"topic\": \"String (Polity/Economy/Env/SciTech/IR)\",\n"
        "  \"question_type\": \"String\",\n"
        "  \"question_text\": \"String (Full question with numbered lines/pairs)\",\n"
        "  \"options\": [\"A\", \"B\", \"C\", \"D\"],\n"
        "  \"correct_option_index\": Integer (0-3),\n"
        "  \"explanation\": \"String (Explain why wrong options are wrong)\"\n"
        "}"
    )

    results: List[dict] = []

    # Per-article request loop
    for idx, article in enumerate(articles):
        try:
            title = article.get("title", "") or ""
            summary = (article.get("summary") or "")[:1500]

            user_content = f"Title: {title}\nSummary: {summary}"

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.25,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            }

            # Basic retry/backoff for per-article call
            retries = 3
            backoffs = [5, 10, 20]
            attempt = 0

            while True:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=120,
                )

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        # Expect a single JSON object
                        parsed = json.loads(content)
                        if not isinstance(parsed, dict):
                            print(f"Groq response for '{title[:60]}' is not a JSON object. Skipping.")
                            break

                        # If the model explicitly says it's not relevant, skip
                        if parsed.get("is_relevant") is False:
                            print(f"Skipped (not relevant): {title}")
                            break

                        # Only keep relevant items
                        if parsed.get("is_relevant") is True:
                            # Ensure the returned dict contains expected fields; inject source_article_title
                            parsed["source_article_title"] = title
                            results.append(parsed)
                        break

                    except Exception as e:
                        print(f"Groq parse error for '{title[:60]}': {e}")
                        break

                elif resp.status_code == 429 and attempt < retries:
                    wait_s = backoffs[attempt]
                    attempt += 1
                    print(f"Rate limited on Groq call for '{title[:60]}', retrying in {wait_s}s...")
                    time.sleep(wait_s)
                    continue
                else:
                    print(f"Groq error {resp.status_code} for '{title[:60]}': {resp.text[:200]}")
                    break

            # Sleep 4 seconds between requests to respect TPM limit
            time.sleep(4)

        except Exception as e:
            print(f"Exception processing article '{article.get('title','')[:60]}': {e}")
            # Continue to next article on any exception
            continue

    return results


def insert_quiz_questions(questions: List[dict]) -> int:
    """Insert generated quiz questions into the database."""
    if not questions:
        return 0
    
    client = supabase_client()
    
    # Prepare questions for insertion
    rows = []
    for q in questions:
        try:
            # Map the new JSON fields to DB columns. The model now returns:
            # - 'correct_option_index' (int 0-3)
            # - 'options' (list)
            # - 'source_article_title' should be present (injected by caller)
            
            # Clean prefixes from options (A), B), 1., etc.)
            raw_options = q.get("options", [])
            cleaned_options = []
            if isinstance(raw_options, list):
                for opt in raw_options:
                    try:
                        opt_str = opt if isinstance(opt, str) else str(opt)
                        # Remove prefixes like "A)", "1.", "(a)", "B-", etc. from the start
                        cleaned = re.sub(r'^[\s\(]*[A-Da-d1-4][\)\.\-]\s*', '', opt_str).strip()
                        cleaned_options.append(cleaned)
                    except Exception:
                        cleaned_options.append(opt)
            
            row = {
                "question_text": q.get("question_text", ""),
                "options": json.dumps(cleaned_options),
                "correct_answer": q.get("correct_option_index", q.get("correct_option", 0)),
                "explanation": q.get("explanation", ""),
                "topic": (q.get("topic") or "General").title(),
                "difficulty": (q.get("difficulty") or "medium").lower(),
                "source_article_id": q.get("source_article_title", ""),
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Validate required fields: text, 4 options, explanation and topic
            try:
                options_list = json.loads(row["options"])
            except Exception:
                options_list = []

            if (row["question_text"] and options_list and len(options_list) == 4 and row["explanation"] and row["topic"]):
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
    """
    Main function to generate daily UPSC-style quiz questions using a 'Fill the Bucket' loop.
    It iteratively fetches batches of unprocessed articles and generates questions until
    a target number is met or a safety limit is reached.
    """
    print("🚀 Starting 'Fill the Bucket' daily quiz generation...")
    client = supabase_client()

    target_questions = 60
    total_generated = 0
    max_articles_to_scan = 1000
    articles_scanned = 0
    
    # Use a set to keep track of titles processed in this run to avoid reprocessing
    processed_titles_in_run = set()

    while total_generated < target_questions and articles_scanned < max_articles_to_scan:
        print(f"--- Iteration: {total_generated}/{target_questions} questions generated, {articles_scanned}/{max_articles_to_scan} articles scanned ---")

        # 1. Fetch a new batch of unprocessed articles (increased batch size)
        articles_batch = fetch_recent_summarized_articles(limit=100)

        # Filter out any articles we might have already seen in this session (edge case)
        articles_to_process = [a for a in articles_batch if a.get('title') not in processed_titles_in_run]

        if not articles_to_process:
            print("✅ No more unprocessed articles in the database. Halting.")
            break

        titles_in_batch = [a.get('title') for a in articles_to_process if a.get('title')]
        articles_scanned += len(articles_to_process)
        processed_titles_in_run.update(titles_in_batch)

        print(f"ℹ️ Fetched {len(articles_to_process)} new articles to process...")

        # 2. Generate questions from the batch (in smaller sub-batches for Groq)
        groq_batch_size = 20
        generated_questions_in_batch = []
        for i in range(0, len(articles_to_process), groq_batch_size):
            sub_batch = articles_to_process[i:i + groq_batch_size]
            generated = call_groq_generate_quiz_questions(sub_batch)
            if generated:
                generated_questions_in_batch.extend(generated)

        # 3. Insert new questions into DB
        if generated_questions_in_batch:
            inserted_count = insert_quiz_questions(generated_questions_in_batch)
            total_generated += inserted_count
            print(f"👍 Generated and inserted {inserted_count} new questions. Total so far: {total_generated}")
        else:
            print("🤔 No relevant questions generated in this batch.")

        # 4. Mark the fetched articles as processed immediately
        if titles_in_batch:
            try:
                print(f"🔔 Marking {len(titles_in_batch)} articles as processed...")
                client.table("articles").update({
                    "quiz_generated": True,
                    "updated_at": "now()"
                }).in_("title", titles_in_batch).execute()
            except Exception as e:
                print(f"❌ Error marking articles as processed: {e}")

        # 5. Sleep if we are going to continue
        if total_generated < target_questions and articles_scanned < max_articles_to_scan and articles_to_process:
            print("⏳ Waiting 3 seconds before next iteration...")
            time.sleep(3)

    print(f"🏁 Daily quiz generation complete. Total questions generated: {total_generated}")


def generate_daily_debate_topic():
    print("⚖️ [DEBATE] Generating new daily debate topic...")
    client = supabase_client()
    try:
        # Fetch Top 5 Indian Politics articles
        res = client.table("articles").select("title, summary, id").eq("category", "politics").eq("country", "IN").eq("summarized", True).order("created_at", desc=True).limit(5).execute()
        articles = res.data or []
        if not articles:
            print("⚠️ [DEBATE] Not enough IN politics articles to generate a debate.")
            return

        combined_text = "\n".join([f"- {a['title']}: {a['summary']}" for a in articles])
        article_ids = [str(a['id']) for a in articles]

        # Call Local LLM
        from LocalLLMService import generate_debate_topic
        topic_data = generate_debate_topic(combined_text)

        if "statement" in topic_data:
            client.table("debate_topics").insert({
                "statement": topic_data["statement"],
                "context": topic_data.get("context", ""),
                "status": "upcoming",
                "related_article_ids": article_ids
            }).execute()
            print(f"✅ [DEBATE] Generated upcoming debate: {topic_data['statement']}")

    except Exception as e:
        print(f"❌ [DEBATE] Error generating topic: {e}")

def manage_debate_lifecycle():
    print("⏳ [DEBATE] Managing debate lifecycles...")
    client = supabase_client()
    try:
        now_iso = datetime.now().isoformat()
        
        # 1. Check for expired active debates
        expired_res = client.table("debate_topics").select("id, statement").eq("status", "active").lte("end_time", now_iso).execute()
        expired_debates = expired_res.data or []

        for debate in expired_debates:
            topic_id = debate["id"]
            statement = debate["statement"]
            print(f"🔄 [DEBATE] Concluding topic ID: {topic_id}")
            
            # Aggregate stats
            support_count = 0
            oppose_count = 0
            
            votes_res = client.table("user_votes").select("id, side, argument_text, ai_score").eq("topic_id", topic_id).execute()
            votes = votes_res.data or []
            
            support_args = []
            oppose_args = []
            
            for v in votes:
                if v["side"] == "support":
                    support_count += 1
                    if v.get("ai_score", 0) >= 7: support_args.append(v)
                elif v["side"] == "oppose":
                    oppose_count += 1
                    if v.get("ai_score", 0) >= 7: oppose_args.append(v)
            
            winning_side = "support" if support_count >= oppose_count else "oppose"
            
            # Fetch top 5 arguments for the winning side
            top_args = sorted(support_args if winning_side == "support" else oppose_args, key=lambda x: x.get("ai_score", 0), reverse=True)[:5]
            top_args_text = "\n".join([f"- {a['argument_text']}" for a in top_args]) if top_args else "No strong arguments provided."

            # Call AI to summarize
            ai_conclusion = "The debate ended with equal participation but lacked strong logical arguments to summarize."
            if top_args:
                from LocalLLMService import conclude_debate
                ai_conclusion = conclude_debate(statement, winning_side, top_args_text)
            
            # Update DB
            client.table("debate_topics").update({
                "status": "completed",
                "winning_side": winning_side,
                "ai_conclusion": ai_conclusion
            }).eq("id", topic_id).execute()
            print(f"✅ [DEBATE] Concluded topic ID {topic_id}. Winner: {winning_side}")

        # 2. Check if we need to activate an upcoming debate
        active_res = client.table("debate_topics").select("id", count="exact").eq("status", "active").execute()
        if active_res.count == 0:
            # Find the oldest upcoming debate
            upcoming_res = client.table("debate_topics").select("id").eq("status", "upcoming").order("created_at").limit(1).execute()
            if upcoming_res.data:
                next_id = upcoming_res.data[0]["id"]
                start_time = datetime.now()
                end_time = start_time + timedelta(hours=24)
                
                client.table("debate_topics").update({
                    "status": "active",
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }).eq("id", next_id).execute()
                print(f"🚀 [DEBATE] Activated new debate topic ID: {next_id} for 24 hours.")
            else:
                # No upcoming debates, trigger generation immediately
                generate_daily_debate_topic()

    except Exception as e:
        print(f"❌ [DEBATE] Lifecycle error: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(debate_router)


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


@app.post("/admin/send-news-notification")
def trigger_news_notification():
    """Manually trigger the daily news FCM notification."""
    try:
        send_daily_news_notification()
        return {"status": "sent", "message": "Daily news notification triggered."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/admin/send-quiz-notification")
def trigger_quiz_notification():
    """Manually trigger the daily quiz FCM notification."""
    try:
        send_daily_quiz_notification()
        return {"status": "sent", "message": "Daily quiz notification triggered."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/admin/scheduler-status")
def get_scheduler_status():
    """
    Check if the APScheduler is running and list all registered jobs with their next run times.
    Use this to verify that notification cron jobs are correctly registered and will fire.
    """
    try:
        jobs_info = []
        for job in scheduler.get_jobs():
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time) if job.next_run_time else "PAUSED/NONE",
                "trigger": str(job.trigger),
            })
        return {
            "scheduler_running": scheduler.running,
            "total_jobs": len(jobs_info),
            "current_time_utc": datetime.utcnow().isoformat(),
            "jobs": jobs_info,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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


def heal_database_flags():
    """
    Corrects inconsistent article flags in the database on startup.
    Sets summarized = TRUE and summarization_needed = FALSE for articles
    that have a valid summary but are incorrectly flagged.
    """
    print("🩹 [DB HEALING] Triggering server-side healing via RPC 'heal_article_flags'...")
    try:
        client = supabase_client()
        # Call the Postgres function (RPC) which performs the healing server-side
        res = client.rpc('heal_article_flags', {}).execute()

        # Basic success/failure logging. The supabase client response may include `error` or `data` attributes.
        if getattr(res, 'error', None):
            print(f"❌ [DB HEALING] RPC returned an error: {res.error}")
        else:
            print("✅ [DB HEALING] Database healing triggered successfully.")
            # Optionally log returned data if present
            if getattr(res, 'data', None):
                try:
                    print(f"🩹 [DB HEALING] RPC result: {res.data}")
                except Exception:
                    pass

    except Exception as e:
        print(f"❌ [DB HEALING] An error occurred while calling heal_article_flags RPC: {e}")


def send_daily_news_notification():
    """
    Fetches the top news article and sends a push notification to users
    via the 'daily_news' FCM topic.
    """
    print(f"🔔 [FCM] Preparing daily NEWS notification at {datetime.now().isoformat()} UTC")
    
    try:
        # 1. Fetch content for the notification
        client = supabase_client()
        res = (
            client.table("articles")
            .select("title, summary")
            .eq("summarized", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        title = "Daily Briefing Ready! 📰"
        body = "Check out today's top stories and stay ahead."
        
        if res.data and len(res.data) > 0:
            article = res.data[0]
            if article.get("title"):
                title = "Daily Briefing"
                body = article["title"]
        
        # 2. Construct the message
        # IMPORTANT: channel_id must match the channel created in the Flutter app
        # (high_importance_channel) so the notification uses max importance on Android 8+.
        # Note: channel_id goes inside AndroidNotification, NOT AndroidConfig.
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    icon='notification_icon',
                    color='#FF6B35',
                    channel_id='high_importance_channel',
                ),
            ),
            data={
                "route": "/home",
            },
            topic='daily_news',
        )
        
        # 3. Send the message
        response = messaging.send(message)
        print(f"✅ [FCM] Successfully sent daily NEWS notification: {response}")
        
    except Exception as e:
        print(f"❌ [FCM] Error sending daily NEWS notification: {e}")


def send_daily_quiz_notification():
    """
    Sends a push notification to users via the 'daily_quiz' FCM topic.
    """
    print(f"🔔 [FCM] Preparing daily QUIZ notification at {datetime.now().isoformat()} UTC")
    
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title="Daily Quiz Ready! 🧠",
                body="Test your knowledge with today's questions.",
            ),
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    icon='notification_icon',
                    color='#FF6B35',
                    channel_id='high_importance_channel',
                ),
            ),
            data={
                "route": "/quiz",
            },
            topic='daily_quiz',
        )
        
        response = messaging.send(message)
        print(f"✅ [FCM] Successfully sent daily QUIZ notification: {response}")
        
    except Exception as e:
        print(f"❌ [FCM] Error sending daily QUIZ notification: {e}")



@app.on_event("startup")
def schedule_jobs():
    # Heal inconsistent DB flags on startup
    heal_database_flags()

    # Run news ingestion immediately on startup, then every 15 minutes
    scheduler.add_job(smart_ingest_all_categories, "interval", minutes=15, id="ingest_news", replace_existing=True)


    # Add periodic round-robin summarizer (keeps summaries flowing across categories)
    scheduler.add_job(
        summarize_pending_round_robin,
        "interval",
        minutes=10,  # Run every 10 minutes instead of 3 for Llama-3
        id="summarize_rr",
        replace_existing=True,
        kwargs={"per_category_limit": 1, "max_cycles": 5}
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
    
    # Add Daily News push notification jobs (9:00 AM and 7:00 PM IST)
    # 9:00 AM IST = 03:30 UTC
    # 7:00 PM IST = 13:30 UTC
    # Wrapped in error-logging wrappers so failures are visible in GCP logs
    def _scheduled_news_notification(label=""):
        """Wrapper with full error logging for scheduled news FCM push."""
        try:
            print(f"\n{'='*60}")
            print(f"🔔 [SCHEDULER] Running scheduled NEWS notification ({label}) at {datetime.utcnow().isoformat()} UTC")
            print(f"{'='*60}")
            send_daily_news_notification()
            print(f"✅ [SCHEDULER] Scheduled NEWS notification ({label}) completed successfully")
        except Exception as e:
            print(f"❌ [SCHEDULER] FAILED scheduled NEWS notification ({label}): {e}")
            import traceback
            traceback.print_exc()

    def _scheduled_quiz_notification():
        """Wrapper with full error logging for scheduled quiz FCM push."""
        try:
            print(f"\n{'='*60}")
            print(f"🔔 [SCHEDULER] Running scheduled QUIZ notification at {datetime.utcnow().isoformat()} UTC")
            print(f"{'='*60}")
            send_daily_quiz_notification()
            print(f"✅ [SCHEDULER] Scheduled QUIZ notification completed successfully")
        except Exception as e:
            print(f"❌ [SCHEDULER] FAILED scheduled QUIZ notification: {e}")
            import traceback
            traceback.print_exc()

    scheduler.add_job(
        lambda: _scheduled_news_notification("morning"),
        "cron", hour=3, minute=30, id="push_notify_news_morning", replace_existing=True, timezone="UTC"
    )
    scheduler.add_job(
        lambda: _scheduled_news_notification("evening"),
        "cron", hour=13, minute=30, id="push_notify_news_evening", replace_existing=True, timezone="UTC"
    )

    # Add Daily Quiz push notification job (9:30 AM IST = 04:00 UTC)
    scheduler.add_job(
        _scheduled_quiz_notification,
        "cron", hour=4, minute=0, id="push_notify_quiz", replace_existing=True, timezone="UTC"
    )
    
    scheduler.start()

    # Log all registered jobs so we can verify in GCP logs
    print(f"\n{'='*60}")
    print(f"📋 [SCHEDULER] All registered jobs ({len(scheduler.get_jobs())} total):")
    for job in scheduler.get_jobs():
        print(f"   • {job.id}: next_run={job.next_run_time}, trigger={job.trigger}")
    print(f"{'='*60}\n")
    
    # 1. Run immediate round-robin and cleanup
    try:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        # Reduced limits for Llama-3 at startup
        executor.submit(summarize_pending_round_robin, per_category_limit=1, max_cycles=5)
        executor.submit(clean_existing_quiz_options)
        print("✅ Submitted async summarization & DB cleanup tasks at startup")
    except Exception as e:
        print(f"⚠️ Error starting async tasks: {e}")

    if not scheduler.running:
        try:
            scheduler.start()
            print("🕒 APScheduler started successfully")
        except Exception as e:
            print(f"❌ Failed to start APScheduler: {e}")

    # Register scheduled jobs
    try:
        # Debates
        scheduler.add_job(generate_daily_debate_topic, 'cron', hour=1, minute=0, id='gen_debate')
        scheduler.add_job(manage_debate_lifecycle, 'interval', minutes=10, id='manage_debate')
        
        # Existing
        scheduler.add_job(smart_ingest_all_categories, "interval", hours=4, id="smart_ingest_job")
        # Notice: removed duplicate summarize_round_robin_job since we have summarize_rr above
        scheduler.add_job(generate_daily_quiz_questions, "cron", hour=2, minute=0, id="daily_quiz_gen_job")
        scheduler.add_job(send_daily_news_notification, 'cron', hour=8, minute=0, id="daily_news_push")
        scheduler.add_job(generate_daily_riddle, 'cron', hour=3, minute=0, id="daily_riddle_gen")
        scheduler.add_job(send_daily_quiz_notification, 'cron', hour=12, minute=0, id="daily_quiz_push")
        
        print("📋 Scheduled jobs configured successfully")
    except Exception as e:
        print(f"❌ Error configuring scheduled jobs: {e}")


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
            content = a.content
            title = a.title or "Untitled"
            # Only summarize if content exists and is at least 100 characters
            if content and len(content) >= 100:
                summary = summarize_text_if_possible(content, title)
            elif content:
                print(f"⏭️ Skipping short article in ingest (len={len(content)}): {title[:80]}")
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
    arr = list(items)
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
        
        if not isinstance(options, list) or not options:
            return question

        # Normalize correct_index to int and clamp
        try:
            correct_index = int(correct_index or 0)
        except Exception:
            correct_index = 0
        if correct_index < 0 or correct_index >= len(options):
            correct_index = 0

        # 1. CLEAN PREFIXES (Remove 'A) ', 'b. ', '1)', '(a)', 'A-' etc. at start)
        cleaned_options: List[str] = []
        for opt in options:
            try:
                opt_str = opt if isinstance(opt, str) else str(opt)
            except Exception:
                opt_str = ""

            # Regex to remove "A)", "1.", "(a)", "A-" from start (as requested)
            cleaned = re.sub(r'^[\s\(]*[A-Da-d1-4][\)\.\-]\s*', '', opt_str).strip()
            cleaned_options.append(cleaned)

        # Track the correct option text after cleaning
        try:
            correct_option_text = cleaned_options[correct_index]
        except Exception:
            correct_option_text = None

        # 2. SHUFFLE cleaned options
        random.shuffle(cleaned_options)

        # 3. RE-INDEX: find where the correct option landed
        if correct_option_text is not None and correct_option_text in cleaned_options:
            new_correct_index = cleaned_options.index(correct_option_text)
        else:
            # fallback: keep same index if within bounds, otherwise 0
            new_correct_index = min(correct_index, len(cleaned_options) - 1)

        question["options"] = json.dumps(cleaned_options)
        question["correct_answer"] = new_correct_index
        return question
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Error randomizing options for question {question.get('id')}: {e}")
        return question



