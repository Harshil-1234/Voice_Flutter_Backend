from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import json
import os
from datetime import datetime

from LocalLLMService import judge_argument

# Initialize Router
router = APIRouter(prefix="/debate", tags=["debate"])

class VoteRequest(BaseModel):
    user_id: str
    topic_id: str
    side: str
    argument_text: str

class VoteResponse(BaseModel):
    score: int
    feedback: str
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

@router.post("/vote", response_model=VoteResponse)
async def submit_vote(vote: VoteRequest):
    # Mocking statement for standalone function
    statement = "Artificial Intelligence will eventually replace most human jobs." 
    
    # 1. AI Judge Argument (using Local LLM)
    ai_evaluated = True
    if not vote.argument_text or len(vote.argument_text.strip()) < 5:
        # Edge case: No real argument provided. Save resources by skipping LLM.
        ai_evaluated = False
        ai_analysis = {
            "score": 1,
            "feedback": "Vote counted. No argument provided.",
            "key_factors": ["Missing reasoning"],
            "detailed_analysis": "To participate fully in the debate and earn a higher logic score, you must provide a written argument defending your position."
        }
    else:
        ai_analysis = judge_argument(statement, vote.argument_text)
    
    # 2. Gamification Math
    score = ai_analysis.get("score", 1)
    xp_earned = XP_BASE_VOTE + (score * XP_MULTIPLIER_SCORE)
    
    # 3. DB Transactions (Mocked for now)
    # supabase = get_supabase_client()
    
    # a. Store Vote and hidden feedback
    # vote_data = {
    #     "user_id": vote.user_id,
    #     ...
    #     "ai_score": score,
    #     "ai_feedback": ai_analysis["feedback"],
    #     "hidden_feedback": ai_analysis.get("detailed_analysis", "No detailed analysis available.")
    # }
    # supabase.table("user_votes").insert(vote_data).execute()
    
    # b. Fetch current XP and Update Profile
    # current_profile = supabase.table("profiles").select("debate_xp", "current_title").eq("id", vote.user_id).execute().data[0]
    # new_total_xp = current_profile['debate_xp'] + xp_earned
    # expected_title = calculate_new_title(new_total_xp)
    # new_title_to_return = None
    # if expected_title != current_profile['current_title']:
    #      new_title_to_return = expected_title
    #      supabase.table("profiles").update({"debate_xp": new_total_xp, "current_title": expected_title}).eq("id", vote.user_id).execute()
    # else:
    #      supabase.table("profiles").update({"debate_xp": new_total_xp}).eq("id", vote.user_id).execute()
    
    new_title_to_return = "Observer" # Mock level up
    
    return VoteResponse(
        score=score,
        feedback=ai_analysis["feedback"],
        key_factors=ai_analysis.get("key_factors", []),
        xp_earned=xp_earned,
        new_title=new_title_to_return,
        ai_evaluated=ai_evaluated
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
    # Mocking Database query since supabase is commented out
    # 1. Fetch from Profiles
    current_xp = 0 # Default if no votes
    current_title = "Rookie"
    
    # 2. Fetch from User_votes
    votes_cast = 0
    avg_score = 0.0
    
    # We will simulate that the user actually hasn't voted as per standard request:
    # "i havent even casted a single vote and my profile screen shows i have casted 24"
    
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
        "avg_logic_score": avg_score
    }

@router.get("/current/{user_id}")
async def get_current_debate(user_id: str):
    # Mocking Database query
    # 1. Fetch the active debate
    # active_debate = supabase.table("debate_topics").select("*").eq("status", "active").execute().data
    active_debate = {
        "id": "daily_1",
        "statement": "India should ban all algorithmic social media feeds.",
        "context": "Recent discussions highlight the negative impact of algorithmic feeds on mental health and political polarization in India.",
        "related_article_ids": ["article_1", "article_2"],
        "end_time": (datetime.now().timestamp() + 7200) * 1000 # Ends in 2 hours
    }

    # 2. Check if user has voted
    # user_vote = supabase.table("user_votes").select("id").eq("topic_id", active_debate["id"]).eq("user_id", user_id).execute().data
    has_voted = False # Mocking that user hasn't voted yet

    # 3. Fetch community stats (Mocked)
    support_score = 650
    oppose_score = 350
    top_support_args = [
        {"id": "arg1", "text": "Algorithms create echo chambers that harm democratic discourse.", "score": 9},
        {"id": "arg2", "text": "They prioritize engagement over truth, leading to misinformation.", "score": 8}
    ]
    top_oppose_args = [
        {"id": "arg3", "text": "Banning algorithms would cripple the tech industry and user experience.", "score": 8},
        {"id": "arg4", "text": "Users should have the choice to use algorithmic feeds or chronological ones.", "score": 7}
    ]

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
    # Mocking Database query
    # history = supabase.table("debate_topics").select("*").eq("status", "completed").order("end_time", desc=True).limit(10).execute().data
    
    mock_history = [
        {
            "id": "past_1",
            "statement": "The implementation of Universal Basic Income will resolve modern poverty.",
            "status": "completed",
            "winning_side": "oppose",
            "ai_conclusion": "The opposing side presented stronger logical points regarding inflation risks and funding mechanisms, while the supporting side relied heavily on ethical appeals without addressing the economic feasibility."
        },
        {
            "id": "past_2",
            "statement": "Space exploration budgets should be reallocated to combat climate change on Earth.",
            "status": "completed",
            "winning_side": "support",
            "ai_conclusion": "Support arguments effectively highlighted the immediate existential threat of climate change compared to the long-term, speculative benefits of space exploration."
        }
    ]
    return mock_history

