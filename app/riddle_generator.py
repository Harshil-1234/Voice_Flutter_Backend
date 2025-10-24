import os
import json
import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime
from supabase import create_client, Client


def supabase_client() -> Client:
    """Get Supabase client with service role key."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase env vars missing: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
    
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def generate_daily_riddle() -> Optional[Dict[str, Any]]:
    """
    Generate a daily riddle using Groq API and store it in Supabase.
    Returns the generated riddle data or None if generation fails.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY missing; skipping riddle generation")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # System prompt for generating challenging riddles
        system_prompt = (
            "You are an expert riddle creator. Generate a short but challenging riddle "
            "that requires deep reasoning or lateral thinking. The riddle should be "
            "thought-provoking and not easily solvable without careful consideration. "
            "Make it engaging and intellectually stimulating.\n\n"
            "Example style: 'I am the parent of civilizations, the killer of heroes. "
            "Without me, you cannot survive, but embrace me too much, and you'll cease "
            "to thrive. What am I?'\n\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            "- question: string (the riddle question)\n"
            "- answer: string (the correct answer)\n"
            "- explanation: string (brief explanation of why this is the answer)"
        )
        
        user_content = (
            "Generate one challenging riddle that requires deep reasoning or lateral thinking. "
            "Make it intellectually stimulating and thought-provoking. "
            "Return the response as a JSON object with question, answer, and explanation fields."
        )
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.7,  # Higher creativity for riddles
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
                        print("❌ Invalid riddle response: missing required fields")
                        return None
                    
                    # Store in Supabase
                    client = supabase_client()
                    riddle_row = {
                        "question": riddle_data["question"],
                        "answer": riddle_data["answer"],
                        "explanation": riddle_data["explanation"],
                        "created_at": datetime.now().isoformat()
                    }
                    
                    result = client.table("riddles").insert(riddle_row).execute()
                    
                    if result.data:
                        print(f"✅ Successfully generated and stored daily riddle")
                        print(f"Question: {riddle_data['question'][:100]}...")
                        return result.data[0]
                    else:
                        print("❌ Failed to store riddle in database")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error: {e}")
                    print(f"Response content: {content[:200]}")
                    return None
                    
            elif resp.status_code == 429 and attempt < retries:
                wait_s = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                attempt += 1
                print(f"⏳ Groq rate limit reached (429). Retrying in {wait_s}s (attempt {attempt}/{retries}).")
                time.sleep(wait_s)
                continue
            else:
                print(f"❌ Groq riddle generation error {resp.status_code}: {resp.text[:200]}")
                break
                
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
    Check if a riddle already exists for today.
    Returns True if today's riddle exists, False otherwise.
    """
    try:
        client = supabase_client()
        today = datetime.now().date()
        
        result = (
            client.table("riddles")
            .select("id")
            .gte("created_at", today.isoformat())
            .lt("created_at", (today.replace(day=today.day + 1) if today.day < 28 else today.replace(month=today.month + 1, day=1)).isoformat())
            .limit(1)
            .execute()
        )
        
        return len(result.data or []) > 0
        
    except Exception as e:
        print(f"❌ Error checking today's riddle: {e}")
        return False
