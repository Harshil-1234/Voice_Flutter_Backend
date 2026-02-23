from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from LocalLLMService import judge_argument

# Initialize Router
router = APIRouter(prefix="/debate", tags=["debate"])

class VoteRequest(BaseModel):
    user_id: str
    topic_id: str
    side: str
    argument_text: str

class VoteResponse(BaseModel):
    vote_id: str
    ai_score: int | None = None
    ai_feedback: str
    key_factors: list[str]
    xp_earned: int
    new_title: str | None = None
    ai_evaluated: bool = True

class UnlockRequest(BaseModel):
    vote_id: str

class UnlockResponse(BaseModel):
    detailed_analysis: str

# Gamification Constants
XP_BASE_VOTE = 10
XP_MULTIPLIER_SCORE = 5

def calculate_new_title(total_xp: int) -> str:
    if total_xp >= 5000:
        return 'Logic Lord'
    elif total_xp >= 1000:
        return 'Senior Pundit'
    elif total_xp >= 500:
        return 'Analyst'
    elif total_xp >= 100:
        return 'Observer'
    return 'Rookie'

def _calculate_and_apply_xp(client, user_id: str, score: int) -> tuple[int, str | None]:
    xp_earned = XP_BASE_VOTE + (score * XP_MULTIPLIER_SCORE)

    profile_res = client.table("profiles").select("debate_xp", "current_title").eq("id", user_id).execute()
    current_profile = profile_res.data[0] if profile_res.data else {"debate_xp": 0, "current_title": "Rookie"}
    current_xp = int(current_profile.get("debate_xp", 0) or 0)

    new_total_xp = current_xp + xp_earned
    expected_title = calculate_new_title(new_total_xp)
    new_title_to_return = None

    update_data = {"debate_xp": new_total_xp}
    if expected_title != current_profile.get("current_title", "Rookie"):
        new_title_to_return = expected_title
        update_data["current_title"] = expected_title

    client.table("profiles").update(update_data).eq("id", user_id).execute()
    return xp_earned, new_title_to_return

def _evaluate_vote_in_background(vote_id: str, user_id: str, statement: str, argument_text: str):
    from main import supabase_client

    client = supabase_client()
    ai_analysis = judge_argument(statement, argument_text)

    try:
        score = int(ai_analysis.get("score", 5))
    except Exception:
        score = 5
    score = max(1, min(10, score))

    ai_feedback = ai_analysis.get("feedback", "AI analysis completed.")
    detailed = ai_analysis.get("detailed_analysis", "No detailed analysis available.")

    client.table("user_votes").update({
        "ai_score": score,
        "ai_feedback": ai_feedback,
        "hidden_feedback": detailed,
        "ai_evaluated": True
    }).eq("id", vote_id).execute()

    # XP is awarded after AI scoring finishes.
    _calculate_and_apply_xp(client, user_id, score)

@router.post("/vote", response_model=VoteResponse)
async def submit_vote(vote: VoteRequest, background_tasks: BackgroundTasks):
    from main import supabase_client
    client = supabase_client()
    clean_argument = (vote.argument_text or "").strip()

    # Fast path: no argument means no AI call.
    if not clean_argument:
        vote_data = {
            "user_id": vote.user_id,
            "topic_id": vote.topic_id,
            "side": vote.side,
            "argument_text": "",
            "ai_score": 1,
            "ai_feedback": "Vote counted. No argument provided.",
            "hidden_feedback": "To participate fully in the debate and earn a higher logic score, provide a written argument.",
            "ai_evaluated": False
        }
        insert_res = client.table("user_votes").insert(vote_data).execute()
        vote_id = insert_res.data[0]["id"] if insert_res.data else ""

        xp_earned, new_title_to_return = _calculate_and_apply_xp(client, vote.user_id, 1)
        return VoteResponse(
            vote_id=vote_id,
            ai_score=1,
            ai_feedback="Vote counted. No argument provided.",
            key_factors=["Missing reasoning"],
            xp_earned=xp_earned,
            new_title=new_title_to_return,
            ai_evaluated=False
        )

    # Non-empty argument: insert pending vote and return immediately.
    topic_res = client.table("debate_topics").select("statement").eq("id", vote.topic_id).execute()
    if not topic_res.data:
        raise HTTPException(status_code=404, detail="Topic not found")
    statement = topic_res.data[0]["statement"]

    vote_data = {
        "user_id": vote.user_id,
        "topic_id": vote.topic_id,
        "side": vote.side,
        "argument_text": clean_argument,
        "ai_score": None,
        "ai_feedback": "AI is judging your logic...",
        "hidden_feedback": None,
        "ai_evaluated": False
    }
    insert_res = client.table("user_votes").insert(vote_data).execute()
    vote_id = insert_res.data[0]["id"] if insert_res.data else ""

    background_tasks.add_task(
        _evaluate_vote_in_background,
        vote_id,
        vote.user_id,
        statement,
        clean_argument,
    )

    return VoteResponse(
        vote_id=vote_id,
        ai_score=None,
        ai_feedback="Vote cast! AI is judging your logic...",
        key_factors=[],
        xp_earned=0,
        new_title=None,
        ai_evaluated=False
    )

@router.post("/unlock_feedback", response_model=UnlockResponse)
async def unlock_feedback(req: UnlockRequest):
    # Retrieve from DB after ad watched
    # supabase = get_supabase_client()
    # vote_record = supabase.table("user_votes").select("hidden_feedback").eq("id", req.vote_id).execute().data
    # if not vote_record:
    #     raise HTTPException(status_code=404, detail="Vote not found")
    
    
    mock_hidden_feedback = "Your argument lacked primary economic data regarding inflation, focusing purely on anecdotal social issues which weakens the logical structure."
    
    return UnlockResponse(detailed_analysis=mock_hidden_feedback)

@router.get("/user_stats/{user_id}")
async def get_user_stats(user_id: str):
    from main import supabase_client
    client = supabase_client()
    
    # 1. Fetch from Profiles
    profile_res = client.table("profiles").select("debate_xp", "current_title").eq("id", user_id).execute()
    profile = profile_res.data[0] if profile_res.data else {"debate_xp": 0, "current_title": "Rookie"}
    
    current_xp = profile.get("debate_xp", 0)
    current_title = profile.get("current_title", "Rookie")
    
    # 2. Fetch from User_votes
    votes_res = client.table("user_votes").select("ai_score").eq("user_id", user_id).execute()
    votes = votes_res.data or []
    
    votes_cast = len(votes)
    avg_score = 0.0
    if votes_cast > 0:
        avg_score = sum([v.get("ai_score", 0) for v in votes]) / votes_cast
    
    next_level_xp = 100
    if current_title == "Observer": next_level_xp = 500
    elif current_title == "Analyst": next_level_xp = 1000
    elif current_title == "Senior Pundit": next_level_xp = 5000
    elif current_title == "Logic Lord": next_level_xp = 5000 # Max Level
    
    return {
        "current_xp": current_xp,
        "next_level_xp": next_level_xp,
        "current_title": current_title,
        "votes_cast": votes_cast,
        "avg_logic_score": round(avg_score, 1)
    }

@router.get("/current/{user_id}")
async def get_current_debate(user_id: str):
    from main import supabase_client
    client = supabase_client()
    
    # 1. Fetch the active debate with schema fallbacks.
    active_debates = []
    try:
        res = client.table("debate_topics").select("*").eq("status", "active").order("created_at", desc=True).limit(1).execute()
        active_debates = res.data or []
    except Exception:
        active_debates = []

    if not active_debates:
        # Legacy schema fallback: is_active boolean.
        try:
            res = client.table("debate_topics").select("*").eq("is_active", True).order("created_at", desc=True).limit(1).execute()
            active_debates = res.data or []
        except Exception:
            active_debates = []

    if not active_debates:
        # Lifecycle fallback: show upcoming when no active exists yet.
        try:
            res = client.table("debate_topics").select("*").eq("status", "upcoming").order("created_at", desc=True).limit(1).execute()
            active_debates = res.data or []
        except Exception:
            active_debates = []
    
    if not active_debates:
        # Check if we should fallback to showing the last completed one or a "Coming Soon" state
        # For now return None so frontend shows "No active debate"
        return None

    active_debate = active_debates[0]

    # 2. Check if user has voted
    vote_res = client.table("user_votes").select("id, side").eq("topic_id", active_debate["id"]).eq("user_id", user_id).execute()
    user_votes = vote_res.data or []
    has_voted = len(user_votes) > 0

    # 3. Fetch community stats
    # Aggregating count and scores for community meter
    all_votes_res = client.table("user_votes").select("side, ai_score, argument_text").eq("topic_id", active_debate["id"]).execute()
    all_votes = all_votes_res.data or []
    
    support_score = 0
    oppose_score = 0
    support_args = []
    oppose_args = []
    
    for v in all_votes:
        if v["side"] == "support":
            support_score += 1
            if v.get("ai_score", 0) >= 7:
                support_args.append({"text": v["argument_text"], "score": v["ai_score"]})
        else:
            oppose_score += 1
            if v.get("ai_score", 0) >= 7:
                oppose_args.append({"text": v["argument_text"], "score": v["ai_score"]})

    # Sort and pick top 3 for each side
    top_support_args = sorted(support_args, key=lambda x: x["score"], reverse=True)[:3]
    top_oppose_args = sorted(oppose_args, key=lambda x: x["score"], reverse=True)[:3]

    if not has_voted:
        return {
            "debate": active_debate,
            "has_voted": False,
            "stats": None # Mask stats for blind voting
        }
    else:
        return {
            "debate": active_debate,
            "has_voted": True,
            "stats": {
                "support_score": support_score,
                "oppose_score": oppose_score,
                "top_support_args": top_support_args,
                "top_oppose_args": top_oppose_args
            }
        }

@router.get("/history/{user_id}")
async def get_debate_history(user_id: str):
    from main import supabase_client
    client = supabase_client()
    
    res = client.table("debate_topics").select("*").eq("status", "completed").order("end_time", desc=True).limit(10).execute()
    return res.data or []

