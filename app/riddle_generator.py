import os
import json
import requests
import time
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
from supabase import create_client, Client


def supabase_client() -> Client:
    """Get Supabase client with service role key."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase env vars missing: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_existing_riddles(limit: int = 100) -> Set[str]:
    """
    Fetch existing riddles from the database to check for duplicates.
    Returns a set of riddle questions (normalized for comparison).
    """
    try:
        client = supabase_client()
        
        result = (
            client.table("riddles")
            .select("question, answer")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        
        # Create a set of normalized questions for duplicate checking
        existing = set()
        for riddle in (result.data or []):
            # Normalize: lowercase and strip whitespace
            question = riddle.get("question", "").lower().strip()
            if question:
                existing.add(question)
        
        return existing
        
    except Exception as e:
        print(f"⚠️ Error fetching existing riddles: {e}")
        return set()


def is_one_word(text: str) -> bool:
    """
    Check if the answer is exactly one word (allowing hyphens and apostrophes in compound words).
    """
    if not text:
        return False
    
    # Remove leading/trailing whitespace and punctuation
    cleaned = text.strip().rstrip('.,!?;:')
    
    # Split by whitespace
    words = cleaned.split()
    
    # Should be exactly one word
    return len(words) == 1


def generate_daily_riddle() -> Optional[Dict[str, Any]]:
    """
    Generate a daily riddle using Groq API and store it in Supabase.
    Ensures:
    - No duplicate riddles (checks against existing riddles)
    - Answer is exactly 1 word
    - Difficulty is high
    
    Returns the generated riddle data or None if generation fails.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY missing; skipping riddle generation")
        return None
    
    # Fetch existing riddles to avoid duplicates
    existing_riddles = get_existing_riddles(limit=200)
    print(f"📋 Checking against {len(existing_riddles)} existing riddles to avoid duplicates")
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # System prompt for generating high-difficulty riddles with 1-word answers
        system_prompt = (
            "You are a master riddler known for wit, clever wordplay, and 'Aha!' moments. "
            "Generate a riddle that is challenging but logically sound. "
            "Do not generate nonsense. The riddle must describe something common (an object, "
            "concept, or element) in a poetic or metaphorical way.\n\n"
            "CRITICAL RULES:\n"
            "1. The answer MUST be exactly ONE WORD. No multi-word answers.\n"
            "2. The difficulty should be 'Medium-Hard'. It should require lateral thinking but MUST be solvable.\n"
            "3. FAIRNESS TEST: When the user hears the answer, it must make perfect sense. If the connection is too vague, reject it.\n"
            "4. Style: Use rhyme, metaphor, or personification.\n\n"
            "BAD EXAMPLE (Too Vague): 'I am big and small.' (Answer: A box? A shadow? Nonsense.)\n"
            "GOOD EXAMPLE (Witty): 'I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?' (Answer: 'Map')\n\n"
            "Return ONLY a valid JSON object:\n"
            "- question: string (The riddle text)\n"
            "- answer: string (One word, lowercase, singular form if possible)\n"
            "- explanation: string (A brief, fun sentence explaining the clues)"
        )
        
        # Generate up to 5 attempts to find a unique riddle with 1-word answer
        max_generation_attempts = 5
        generation_attempt = 0
        
        while generation_attempt < max_generation_attempts:
            generation_attempt += 1
            
            user_content = (
                "Generate one HIGH DIFFICULTY riddle that requires deep reasoning or lateral thinking. "
                "The answer MUST be exactly ONE WORD (no phrases, no multiple words). "
                "Make it intellectually stimulating and thought-provoking. "
                "Ensure the difficulty is HIGH - challenging and not easily solvable. "
                "Return the response as a JSON object with question, answer, and explanation fields."
            )
            
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.8,  # Higher creativity for variety
                "max_tokens": 512,
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
                    timeout=60,
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    try:
                        # Parse the JSON response
                        riddle_data = json.loads(content)
                        
                        # Validate required fields
                        if not all(key in riddle_data for key in ["question", "answer", "explanation"]):
                            print(f"❌ Attempt {generation_attempt}: Invalid riddle response: missing required fields")
                            break
                        
                        question = riddle_data["question"].strip()
                        answer = riddle_data["answer"].strip()
                        
                        # Validate answer is exactly 1 word
                        if not is_one_word(answer):
                            print(f"⚠️ Attempt {generation_attempt}: Answer is not one word: '{answer}'")
                            print(f"   Retrying to generate a riddle with 1-word answer...")
                            break
                        
                        # Check for duplicates (normalize for comparison)
                        question_normalized = question.lower().strip()
                        
                        if question_normalized in existing_riddles:
                            print(f"⚠️ Attempt {generation_attempt}: Duplicate question detected")
                            print(f"   Retrying to generate a unique riddle...")
                            break
                        
                        # All validations passed - store in Supabase
                        client = supabase_client()
                        # Use UTC for consistency with scheduler
                        from datetime import timezone
                        riddle_row = {
                            "question": question,
                            "answer": answer,
                            "explanation": riddle_data["explanation"],
                            "created_at": datetime.now(timezone.utc).isoformat()
                        }
                        
                        result = client.table("riddles").insert(riddle_row).execute()
                        
                        if result.data:
                            print(f"✅ Successfully generated and stored daily riddle (attempt {generation_attempt})")
                            print(f"   Question: {question[:100]}...")
                            print(f"   Answer: {answer} (1 word ✓)")
                            print(f"   Difficulty: High ✓")
                            return result.data[0]
                        else:
                            print("❌ Failed to store riddle in database")
                            return None
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ Attempt {generation_attempt}: JSON parse error: {e}")
                        print(f"Response content: {content[:200]}")
                        break
                        
                elif resp.status_code == 429 and attempt < retries:
                    wait_s = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                    attempt += 1
                    print(f"⏳ Groq rate limit reached (429). Retrying in {wait_s}s (attempt {attempt}/{retries}).")
                    time.sleep(wait_s)
                    continue
                else:
                    print(f"❌ Groq riddle generation error {resp.status_code}: {resp.text[:200]}")
                    break
            
            # If we got here, the generation attempt failed validation
            # Wait a bit before retrying to increase variety
            if generation_attempt < max_generation_attempts:
                time.sleep(2)
        
        print(f"❌ Failed to generate a valid unique riddle after {max_generation_attempts} attempts")
        return None
                
    except Exception as e:
        print(f"❌ Riddle generation exception: {e}")
    
    return None


def get_latest_riddle() -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent riddle from the database.
    Returns the latest riddle or None if not found.
    """
    try:
        client = supabase_client()
        
        result = (
            client.table("riddles")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if result.data:
            return result.data[0]
        else:
            return None
            
    except Exception as e:
        print(f"❌ Error fetching latest riddle: {e}")
        return None


def check_today_riddle_exists() -> bool:
    """
    Check if a riddle already exists for today (UTC).
    Returns True if today's riddle exists, False otherwise.
    """
    try:
        client = supabase_client()
        # Use UTC for consistency with scheduler
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        today_utc = now_utc.date()
        tomorrow_utc = today_utc + timedelta(days=1)
        
        # Query for riddles created today (UTC)
        result = (
            client.table("riddles")
            .select("id")
            .gte("created_at", today_utc.isoformat())
            .lt("created_at", tomorrow_utc.isoformat())
            .limit(1)
            .execute()
        )
        
        exists = len(result.data or []) > 0
        if exists:
            print(f"✅ Riddle already exists for today (UTC): {today_utc.isoformat()}")
        else:
            print(f"ℹ️ No riddle found for today (UTC): {today_utc.isoformat()}")
        
        return exists
        
    except Exception as e:
        print(f"❌ Error checking today's riddle: {e}")
        import traceback
        traceback.print_exc()
        return False
