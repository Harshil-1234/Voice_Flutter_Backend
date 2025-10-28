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
from transformers import pipeline
import trafilatura
from .riddle_generator import generate_daily_riddle, get_latest_riddle, check_today_riddle_exists


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
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
# Default batch size set to 20 per requirements
SUMMARIZE_BATCH_SIZE = int(os.getenv("SUMMARIZE_BATCH_SIZE", "20"))

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

def summarize_text_if_possible(content, titles=None):
    """
    Summarize text using BART if length is sufficient.
    - If `content` is a single string → returns a single summary (string or None).
    - If `content` is a list of strings → returns a list of summaries aligned with input.
    Titles are optional, mainly used for debugging/logging.
    """
    try:
        # Case 1: Single string
        if isinstance(content, str):
            if not content or len(content.split()) < 30:
                return None

            summary = summarizer(
                content,
                max_length=50,
                min_length=30,
                do_sample=False
            )
            return summary[0]["summary_text"]

        # Case 2: List of strings
        elif isinstance(content, list):
            results = []
            for idx, text in enumerate(content):
                if not text or len(text.split()) < 30:
                    results.append(None)
                    continue
                summary = summarizer(
                    text,
                    max_length=60,
                    min_length=45,
                    do_sample=False
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


# def call_groq_summarize(titles: List[str], texts: List[str]) -> List[str]:
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
            parts.append(f"Article {idx}:\nTitle: {safe_t}\nContent: {safe_x}")
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
    # if not GROQ_API_KEY:
    #     return (0, 0)
    to_sum = []
    for a in articles:
        title = a.get("title") or ""
        url = a.get("url")
        full_text = ""

        # Try fetching full article text
        if url:
            full_text = fetch_full_article_text(url)
            full_text = full_text[:500]  # limit length

        # Fallback to short content/description
        if not full_text:
            full_text = a.get("content") or a.get("description") or ""

        if full_text:
            to_sum.append((a, title, full_text))

    if not to_sum:
        print("⚠️ No articles with text available for summarization.")
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
                # url_hash = hashlib.md5(article["url"].lower().encode()).hexdigest()
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
        full = fetch_full_article_text(url)[:500] if url else ""
        if not full:
            full = item.get("content") or item.get("description") or ""
        return full

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
    
    # Conversation Intelligence - contextual chat feature
    # Periodic cleanup for conversations older than 10 minutes
    scheduler.add_job(
        cleanup_old_conversations,
        "interval",
        minutes=2,
        id="cleanup_conversations",
        replace_existing=True,
        kwargs={"max_age_minutes": 10}
    )
    
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


# Conversation Intelligence - contextual chat feature
class ContextChatRequest(BaseModel):
    article_id: str
    user_id: Optional[str] = None
    message: str
    conversation_id: Optional[str] = None  # optional existing context row id


# Conversation Intelligence - contextual chat feature
@app.post("/context/chat")
def contextual_chat(req: ContextChatRequest):
    """
    Handle article-context chat without modifying general chat flow.
    - Fetch article (title + summary/content)
    - Load prior conversation messages from conversation_context (if any)
    - Call Groq with combined context
    - Append new user+assistant turns back to conversation_context
    Returns: { reply, conversation_id }
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")

    client = supabase_client()

    # 1) Fetch article basics
    try:
        ares = (
            client.table("articles")
            .select("id,title,summary,content,description")
            .eq("id", req.article_id)
            .limit(1)
            .execute()
        )
        if not ares.data:
            raise HTTPException(status_code=404, detail="Article not found")
        art = ares.data[0]
        article_title = art.get("title") or ""
        article_context = (
            art.get("summary")
            or art.get("content")
            or art.get("description")
            or ""
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"ContextChat: article fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch article")

    # 2) Get existing conversation (by provided id, or by user+article)
    conv_row = None
    try:
        if req.conversation_id:
            cres = (
                client.table("conversation_context")
                .select("id,messages,user_id,article_id,created_at")
                .eq("id", req.conversation_id)
                .limit(1)
                .execute()
            )
            conv_row = (cres.data or [None])[0]
        else:
            # find the most recent conversation for this user+article (user_id may be null)
            q = (
                client.table("conversation_context")
                .select("id,messages,user_id,article_id,created_at")
                .eq("article_id", req.article_id)
                .order("created_at", desc=True)
                .limit(1)
            )
            if req.user_id:
                q = q.eq("user_id", req.user_id)
            else:
                q = q.is_("user_id", None)
            cres = q.execute()
            conv_row = (cres.data or [None])[0]
    except Exception as e:
        print(f"ContextChat: conversation fetch error: {e}")
        conv_row = None

    # Build message history for Groq
    history = []
    if conv_row and conv_row.get("messages"):
        try:
            history = conv_row["messages"] or []
        except Exception:
            history = []

    # 3) Call Groq with context
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "You are a helpful news assistant. Answer concisely using the provided article context. "
            "If the user's question goes beyond the article, answer based on general knowledge "
            "without hallucinating specifics. Keep answers brief and clear."
        )

        # Compose messages: system + article context + prior turns + user message
        groq_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Article Title: {article_title}\n\n"
                    f"Article Context:\n{article_context}\n\n"
                    "Use this article context to answer questions."
                ),
            },
        ]

        # Append prior turns (truncate to last ~10 turns to control length)
        prior = history[-10:] if len(history) > 10 else history
        for m in prior:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            if role in ("user", "assistant") and content:
                groq_messages.append({"role": role, "content": content})

        # Current user message
        groq_messages.append({"role": "user", "content": req.message})

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": groq_messages,
            "temperature": 0.2,
            "max_tokens": 512,
        }

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"ContextChat: Groq error {resp.status_code}: {resp.text[:200]}")
            raise HTTPException(status_code=502, detail="AI service error")
        data = resp.json()
        reply = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
            or ""
        ).strip()
        if not reply:
            reply = "Sorry, I couldn't generate a response."
    except HTTPException:
        raise
    except Exception as e:
        print(f"ContextChat: Groq call exception: {e}")
        raise HTTPException(status_code=502, detail="AI service error")

    # 4) Persist conversation turns
    try:
        new_messages = history + [
            {"role": "user", "content": req.message, "ts": datetime.now().isoformat()},
            {"role": "assistant", "content": reply, "ts": datetime.now().isoformat()},
        ]

        if conv_row:
            upd = (
                client.table("conversation_context")
                .update({"messages": new_messages})
                .eq("id", conv_row["id"])  # type: ignore
                .execute()
            )
            conv_id = conv_row["id"]
        else:
            ins = (
                client.table("conversation_context")
                .insert({
                    "article_id": req.article_id,
                    "user_id": req.user_id,
                    "messages": new_messages,
                })
                .execute()
            )
            conv_id = (ins.data or [{}])[0].get("id")
    except Exception as e:
        print(f"ContextChat: persist error: {e}")
        # Do not fail the response if storage fails
        conv_id = conv_row["id"] if conv_row else None

    return {"reply": reply, "conversation_id": conv_id}


# Conversation Intelligence - contextual chat feature
def cleanup_old_conversations(max_age_minutes: int = 10):
    """Delete conversation_context rows older than max_age_minutes."""
    try:
        client = supabase_client()
        cutoff = (datetime.utcnow() - timedelta(minutes=max_age_minutes)).isoformat()
        client.table("conversation_context").delete().lt("created_at", cutoff).execute()
    except Exception as e:
        print(f"⚠️ cleanup_old_conversations error: {e}")


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