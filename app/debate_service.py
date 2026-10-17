from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import datetime
import random
import re

from LocalLLMService import judge_argument

# Initialize Router
router = APIRouter(prefix="/debate", tags=["debate"])

class VoteRequest(BaseModel):
    # Kept optional for older clients. The authenticated bearer token is always
    # the source of truth for the user id.
    user_id: str | None = None
    topic_id: str
    side: str
    argument_text: str
    is_anonymous: bool = False

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
    grant_access: bool = True

class UnlockResponse(BaseModel):
    detailed_analysis: str

class CreateRoomRequest(BaseModel):
    statement: str
    context: str = ""
    duration_minutes: int = 60

class JoinRoomRequest(BaseModel):
    code: str

# Gamification Constants
XP_BASE_VOTE = 10
XP_MULTIPLIER_SCORE = 5
PRIVATE_ROOM_DAILY_LIMIT = 1
PRIVATE_ROOM_ACTIVE_LIMIT = 3
JUDGE_PRIORITY = {"official": 3, "public": 2, "private": 1}
JOIN_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_failed_join_attempts: dict[str, list[datetime.datetime]] = {}

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


def _authenticated_user_id(request: Request) -> str:
    """Validate the Supabase bearer token and return its user id."""
    authorization = request.headers.get("authorization", "")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="A valid bearer token is required")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="A valid bearer token is required")

    try:
        from main import supabase_client

        user = supabase_client().auth.get_user(token).user
        user_id = getattr(user, "id", None)
        if not user_id:
            raise ValueError("Token did not resolve to a user")
        return str(user_id)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[DEBATE] Authentication failed: {exc}")
        raise HTTPException(status_code=401, detail="Invalid or expired session")

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

def _new_join_code(client) -> str:
    for _ in range(12):
        code = "".join(random.choice(JOIN_ALPHABET) for _ in range(6))
        if not client.table("debate_topics").select("id").eq("join_code", code).limit(1).execute().data:
            return code
    raise HTTPException(status_code=503, detail="Could not generate a room code; please retry")


def _insert_private_room(client, statement: str, context: str, user_id: str,
                         now: datetime.datetime, end: datetime.datetime,
                         duration_minutes: int) -> dict:
    """Insert a room with a collision-safe, user-visible join code.

    The code is protected by a database unique constraint.  The short retry
    loop handles the (rare) race where two requests select the same candidate
    code before either insert commits.
    """
    for _ in range(5):
        code = _new_join_code(client)
        try:
            result = client.table("debate_topics").insert({
                "statement": statement,
                "context": context,
                "kind": "private",
                "creator_id": user_id,
                "join_code": code,
                "duration_minutes": duration_minutes,
                "max_participants": 50,
                # The host is inserted below, so reflect that immediately.
                "participant_count": 1,
                "moderation_status": "approved",
                "status": "active",
                "start_time": now.isoformat(),
                "end_time": end.isoformat(),
            }).execute()
            if result.data:
                return result.data[0]
        except Exception as exc:
            # Retry only for a code collision; other database errors should be
            # surfaced instead of creating a room with an unknown state.
            if "duplicate" not in str(exc).lower() and "unique" not in str(exc).lower():
                raise
    raise HTTPException(status_code=503, detail="Could not generate a room code; please retry")

def _validate_room_statement(statement: str) -> str:
    text = re.sub(r"\s+", " ", (statement or "").strip())
    if len(text) < 12 or len(text) > 160 or text.isdigit():
        raise HTTPException(status_code=400, detail="Use a debatable statement between 12 and 160 characters")
    blocked = ("fuck", "shit", "bitch", "kill yourself")
    if any(term in text.lower() for term in blocked):
        raise HTTPException(status_code=400, detail="The room statement violates community rules")
    return text

@router.post("/rooms")
async def create_private_room(room: CreateRoomRequest, request: Request):
    from main import supabase_client
    client, user_id = supabase_client(), _authenticated_user_id(request)
    statement = _validate_room_statement(room.statement)
    context = (room.context or "").strip()
    if len(context) > 500 or room.duration_minutes not in {60, 360, 1440, 4320, 10080}:
        raise HTTPException(status_code=400, detail="Invalid room context or duration")
    profile = client.table("profiles").select("display_name").eq("id", user_id).limit(1).execute().data
    if not profile or not (profile[0].get("display_name") or "").strip():
        raise HTTPException(status_code=403, detail="Set a validated display name before creating a room")
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    made_today = client.table("room_creations").select("id", count="exact").eq("user_id", user_id).gte("created_at", today).execute().count or 0
    active = client.table("debate_topics").select("id", count="exact").eq("creator_id", user_id).eq("kind", "private").eq("status", "active").execute().count or 0
    if made_today >= PRIVATE_ROOM_DAILY_LIMIT or active >= PRIVATE_ROOM_ACTIVE_LIMIT:
        raise HTTPException(status_code=429, detail="Your private-room creation limit has been reached")
    end = now + datetime.timedelta(minutes=room.duration_minutes)
    inserted = _insert_private_room(client, statement, context, user_id, now, end, room.duration_minutes)
    try:
        client.table("debate_participants").insert({"topic_id": inserted["id"], "user_id": user_id, "role": "host"}).execute()
        client.table("room_creations").insert({"user_id": user_id, "topic_id": inserted["id"]}).execute()
    except Exception:
        # Do not leave an active room that has no host/metering record if a
        # dependent insert fails after the topic was created.
        try:
            client.table("debate_topics").delete().eq("id", inserted["id"]).execute()
        finally:
            raise
    return {"topic": inserted, "join_code": inserted["join_code"]}

@router.post("/rooms/join")
async def join_private_room(body: JoinRoomRequest, request: Request):
    from main import supabase_client
    client, user_id = supabase_client(), _authenticated_user_id(request)
    now = datetime.datetime.now(datetime.timezone.utc)
    attempts = [t for t in _failed_join_attempts.get(user_id, []) if (now - t).total_seconds() < 600]
    if len(attempts) >= 5: raise HTTPException(status_code=429, detail="Too many invalid code attempts; try again later")
    code = re.sub(r"\s+", "", body.code.upper())
    rows = client.table("debate_topics").select("id,status,max_participants,participant_count").eq("join_code", code).limit(1).execute().data
    if not rows:
        _failed_join_attempts[user_id] = attempts + [now]; raise HTTPException(status_code=404, detail="Room code not found")
    topic = rows[0]
    if topic.get("status") != "active": raise HTTPException(status_code=409, detail="This room is no longer active")
    if int(topic.get("participant_count") or 0) >= int(topic.get("max_participants") or 50): raise HTTPException(status_code=409, detail="This room is full")
    existing = client.table("debate_participants").select("user_id").eq("topic_id", topic["id"]).eq("user_id", user_id).execute().data
    if not existing:
        client.table("debate_participants").insert({"topic_id": topic["id"], "user_id": user_id}).execute()
        client.table("debate_topics").update({"participant_count": int(topic.get("participant_count") or 0) + 1}).eq("id", topic["id"]).execute()
    return {"topic_id": topic["id"]}

@router.get("/hub")
async def get_hub(request: Request):
    """Phase-1 hub: official debate plus rooms the authenticated user joined."""
    from main import supabase_client
    client, user_id = supabase_client(), _authenticated_user_id(request)
    official = client.table("debate_topics").select("*").eq("kind", "official").eq("status", "active").order("created_at", desc=True).limit(1).execute().data
    memberships = client.table("debate_participants").select("topic_id").eq("user_id", user_id).execute().data or []
    ids = [row["topic_id"] for row in memberships]
    rooms = client.table("debate_topics").select("*").in_("id", ids).order("status").order("end_time").execute().data if ids else []
    return {"official": official[0] if official else None, "your_rooms": rooms}

@router.post("/rooms/{topic_id}/close")
async def close_room(topic_id: str, request: Request):
    from main import supabase_client
    client, user_id = supabase_client(), _authenticated_user_id(request)
    rows = client.table("debate_topics").select("creator_id,kind,status").eq("id", topic_id).limit(1).execute().data
    if not rows or rows[0].get("kind") != "private": raise HTTPException(status_code=404, detail="Private room not found")
    if rows[0].get("creator_id") != user_id: raise HTTPException(status_code=403, detail="Only the host can close this room")
    client.table("debate_topics").update({"end_time": datetime.datetime.now(datetime.timezone.utc).isoformat()}).eq("id", topic_id).execute()
    return {"ok": True}

def _evaluate_vote_in_background(vote_id: str, user_id: str, statement: str, argument_text: str):
    from main import supabase_client

    client = supabase_client()
    try:
        ai_analysis = judge_argument(statement, argument_text) or {}
        print(f"[VOTE] Judging vote_id={vote_id} user={user_id} statement_len={len(statement)} arg_len={len(argument_text)}")

        try:
            score = int(ai_analysis.get("score", 5))
        except (TypeError, ValueError):
            score = 5
        score = max(1, min(10, score))

        ai_feedback = str(ai_analysis.get("feedback") or "AI analysis completed.")
        detailed = str(ai_analysis.get("detailed_analysis") or "No detailed analysis available.")
        raw_factors = ai_analysis.get("key_factors") or ai_analysis.get("tags") or []
        key_factors = [str(item).strip()[:80] for item in raw_factors if str(item).strip()][:5]

        update_res = client.table("user_votes").update({
            "ai_score": score,
            "ai_feedback": ai_feedback,
            "hidden_feedback": detailed,
            "ai_tags": key_factors,
            "ai_evaluated": True
        }).eq("id", vote_id).eq("ai_evaluated", False).execute()

        # Award XP only once, after a pending vote was actually transitioned.
        if update_res.data:
            _calculate_and_apply_xp(client, user_id, score)
        print(f"[VOTE] Judgment complete vote_id={vote_id} score={score}")
    except Exception as exc:
        # Never leave the scorecard in a permanent judging state when the model
        # or a transient database operation fails.
        print(f"[VOTE] Judgment failed vote_id={vote_id}: {exc}")
        try:
            fallback_res = client.table("user_votes").update({
                "ai_score": 5,
                "ai_feedback": "AI analysis is temporarily unavailable; a standard logic score was applied.",
                "hidden_feedback": "The detailed analysis could not be generated right now. Please use the feedback above as a starting point.",
                "ai_tags": ["Analysis unavailable"],
                "ai_evaluated": True,
            }).eq("id", vote_id).eq("ai_evaluated", False).execute()
            if fallback_res.data:
                _calculate_and_apply_xp(client, user_id, 5)
        except Exception as fallback_exc:
            print(f"[VOTE] Could not settle failed judgment vote_id={vote_id}: {fallback_exc}")

@router.post("/vote", response_model=VoteResponse)
async def submit_vote(vote: VoteRequest, background_tasks: BackgroundTasks, request: Request):
    from main import supabase_client
    client = supabase_client()
    user_id = _authenticated_user_id(request)
    clean_argument = (vote.argument_text or "").strip()

    if vote.side not in {"support", "oppose"}:
        raise HTTPException(status_code=400, detail="Side must be either support or oppose")
    if len(clean_argument) > 2000:
        raise HTTPException(status_code=400, detail="Arguments must be 2,000 characters or fewer")

    # A user can participate once in each daily topic. This keeps the blind
    # reveal meaningful and prevents duplicate XP awards.
    existing_vote = client.table("user_votes").select("id").eq("topic_id", vote.topic_id).eq("user_id", user_id).limit(1).execute()
    if existing_vote.data:
        raise HTTPException(status_code=409, detail="You have already voted in this debate")

    topic_res = client.table("debate_topics").select("statement,status,end_time,kind,creator_id").eq("id", vote.topic_id).limit(1).execute()
    if not topic_res.data:
        raise HTTPException(status_code=404, detail="Topic not found")
    topic = topic_res.data[0]
    if topic.get("status") != "active":
        raise HTTPException(status_code=409, detail="This debate is not accepting votes")
    if topic.get("kind") == "private":
        participant = client.table("debate_participants").select("user_id").eq("topic_id", vote.topic_id).eq("user_id", user_id).limit(1).execute().data
        if not participant:
            raise HTTPException(status_code=403, detail="Join this private room with its code before voting")
    raw_end_time = topic.get("end_time")
    if raw_end_time:
        try:
            end_time = datetime.datetime.fromisoformat(
                str(raw_end_time).replace("Z", "+00:00")
            )
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=datetime.timezone.utc)
            if end_time <= datetime.datetime.now(datetime.timezone.utc):
                raise HTTPException(status_code=409, detail="This debate has ended")
        except ValueError:
            # Do not block an otherwise active debate over a malformed legacy
            # timestamp; the lifecycle job will repair its state.
            pass
    statement = str(topic.get("statement") or "").strip()
    if not statement:
        raise HTTPException(status_code=409, detail="This debate has no valid statement")

    # Fast path: no argument means no AI call.
    if not clean_argument:
        vote_data = {
            "user_id": user_id,
            "topic_id": vote.topic_id,
            "side": vote.side,
            "argument_text": "",
            "ai_score": 1,
            "ai_feedback": "Vote counted. No argument provided.",
            "hidden_feedback": "To participate fully in the debate and earn a higher logic score, provide a written argument.",
            "ai_evaluated": True,
            "ai_tags": ["Missing reasoning"],
            "is_anonymous": vote.is_anonymous
        }
        insert_res = client.table("user_votes").insert(vote_data).execute()
        vote_id = insert_res.data[0]["id"] if insert_res.data else ""
        print(f"[VOTE] Received (no-argument) vote_id={vote_id} topic={vote.topic_id} side={vote.side} user={user_id} anonymous={vote.is_anonymous}")

        xp_earned, new_title_to_return = _calculate_and_apply_xp(client, user_id, 1)
        return VoteResponse(
            vote_id=vote_id,
            ai_score=1,
            ai_feedback="Vote counted. No argument provided.",
            key_factors=["Missing reasoning"],
            xp_earned=xp_earned,
            new_title=new_title_to_return,
            ai_evaluated=True
        )

    # Non-empty argument: insert pending vote and return immediately.
    vote_data = {
        "user_id": user_id,
        "topic_id": vote.topic_id,
        "side": vote.side,
        "argument_text": clean_argument,
        "ai_score": None,
        "ai_feedback": "AI is judging your logic...",
        "hidden_feedback": None,
        "ai_evaluated": False,
        "is_anonymous": vote.is_anonymous
    }
    insert_res = client.table("user_votes").insert(vote_data).execute()
    vote_id = insert_res.data[0]["id"] if insert_res.data else ""
    print(f"[VOTE] Received vote_id={vote_id} topic={vote.topic_id} side={vote.side} user={user_id} anonymous={vote.is_anonymous} arg_len={len(clean_argument)}")

    # The worker consumes this serial queue; requests never invoke Llama.
    client.table("judge_jobs").insert({
        "vote_id": vote_id,
        "topic_id": vote.topic_id,
        "priority": JUDGE_PRIORITY.get(topic.get("kind"), 1),
    }).execute()

    return VoteResponse(
        vote_id=vote_id,
        ai_score=None,
        ai_feedback="Vote cast! AI is judging your logic...",
        key_factors=[],
        xp_earned=0,
        new_title=None,
        ai_evaluated=False
    )

def process_judge_queue_once() -> None:
    """Run one queued Llama job. Schedule this frequently in the API process."""
    from main import supabase_client
    client = supabase_client()
    # Recover jobs left in `running` by a worker crash.  This prevents an
    # otherwise valid argument from waiting forever for AI feedback.
    stale_before = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=10)).isoformat()
    client.table("judge_jobs").update({"status": "queued"}).eq("status", "running").lt("started_at", stale_before).execute()
    jobs = client.table("judge_jobs").select("id,vote_id,topic_id,attempts").eq("status", "queued").order("priority", desc=True).order("created_at").limit(1).execute().data or []
    if not jobs: return
    job = jobs[0]
    client.table("judge_jobs").update({"status": "running", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "attempts": int(job.get("attempts") or 0) + 1}).eq("id", job["id"]).execute()
    try:
        vote = client.table("user_votes").select("user_id,argument_text").eq("id", job["vote_id"]).single().execute().data
        topic = client.table("debate_topics").select("statement").eq("id", job["topic_id"]).single().execute().data
        _evaluate_vote_in_background(job["vote_id"], vote["user_id"], topic["statement"], vote["argument_text"])
        client.table("judge_jobs").update({"status": "done"}).eq("id", job["id"]).execute()
    except Exception as exc:
        next_status = "failed" if int(job.get("attempts") or 0) >= 1 else "queued"
        client.table("judge_jobs").update({"status": next_status}).eq("id", job["id"]).execute()
        print(f"[DEBATE] judge queue job failed: {exc}")

@router.post("/unlock_feedback", response_model=UnlockResponse)
async def unlock_feedback(req: UnlockRequest, request: Request):
    """Record a rewarded/fail-open unlock and return only the owner's critique."""
    from main import supabase_client

    user_id = _authenticated_user_id(request)
    client = supabase_client()
    vote_res = client.table("user_votes").select("id, hidden_feedback").eq("id", req.vote_id).eq("user_id", user_id).limit(1).execute()
    if not vote_res.data:
        raise HTTPException(status_code=404, detail="Vote not found")
    if not req.grant_access:
        raise HTTPException(status_code=403, detail="Detailed analysis requires a rewarded unlock")

    # This is intentionally idempotent: an interrupted client can retry after
    # earning the reward without seeing a duplicate-key failure.
    existing = client.table("ad_attempts").select("id").eq("user_id", user_id).eq("reference_id", req.vote_id).eq("type", "debate_feedback").limit(1).execute()
    if not existing.data:
        client.table("ad_attempts").insert({
            "user_id": user_id,
            "type": "debate_feedback",
            "reference_id": req.vote_id,
            "unlock_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }).execute()

    detailed = str(vote_res.data[0].get("hidden_feedback") or "Detailed analysis is still being prepared. Please check back shortly.")
    return UnlockResponse(detailed_analysis=detailed)

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
    support_vote_count = 0
    oppose_vote_count = 0
    support_args = []
    oppose_args = []
    
    for v in all_votes:
        score_val = v.get("ai_score") or 0
        if v["side"] == "support":
            support_vote_count += 1
            support_score += score_val
            if score_val >= 7:
                support_args.append({"text": v["argument_text"], "score": score_val})
        elif v["side"] == "oppose":
            oppose_vote_count += 1
            oppose_score += score_val
            if score_val >= 7:
                oppose_args.append({"text": v["argument_text"], "score": score_val})

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
                "support_vote_count": support_vote_count,
                "oppose_vote_count": oppose_vote_count,
                "top_support_args": top_support_args,
                "top_oppose_args": top_oppose_args
            }
        }

@router.get("/history/{user_id}")
async def get_debate_history(user_id: str):
    from main import supabase_client
    client = supabase_client()
    
    res = client.table("debate_topics").select("*").eq("status", "completed").order("end_time", desc=True).limit(10).execute()
    history = res.data or []
    if not history:
        return []

    topic_ids = [str(row.get("id")) for row in history if row.get("id")]
    counts_by_topic: dict[str, dict[str, int]] = {}
    if topic_ids:
        try:
            votes_res = client.table("user_votes").select("topic_id, side").in_("topic_id", topic_ids).execute()
            for vote in (votes_res.data or []):
                topic_id = str(vote.get("topic_id") or "")
                side = str(vote.get("side") or "")
                if not topic_id:
                    continue
                if topic_id not in counts_by_topic:
                    counts_by_topic[topic_id] = {"support_vote_count": 0, "oppose_vote_count": 0}
                if side == "support":
                    counts_by_topic[topic_id]["support_vote_count"] += 1
                elif side == "oppose":
                    counts_by_topic[topic_id]["oppose_vote_count"] += 1
        except Exception as e:
            print(f"[DEBATE] Failed to load history vote counts: {e}")

    for row in history:
        topic_id = str(row.get("id") or "")
        topic_counts = counts_by_topic.get(topic_id, {"support_vote_count": 0, "oppose_vote_count": 0})
        row["support_vote_count"] = topic_counts["support_vote_count"]
        row["oppose_vote_count"] = topic_counts["oppose_vote_count"]
    return history

@router.post("/conclude/{topic_id}")
async def conclude_debate(topic_id: str):
    from main import supabase_client
    client = supabase_client()
    
    # 1. Fetch all votes for this topic
    votes_res = client.table("user_votes").select("side, ai_score, argument_text").eq("topic_id", topic_id).execute()
    votes = votes_res.data or []
    
    support_sum = sum([v.get("ai_score") or 0 for v in votes if v["side"] == "support"])
    oppose_sum = sum([v.get("ai_score") or 0 for v in votes if v["side"] == "oppose"])
    support_vote_count = len([v for v in votes if v.get("side") == "support"])
    oppose_vote_count = len([v for v in votes if v.get("side") == "oppose"])
    
    # 2. Determine winner
    winning_side = "draw"
    if support_sum > oppose_sum:
        winning_side = "support"
    elif oppose_sum > support_sum:
        winning_side = "oppose"

    # 3. Build AI conclusion with winner + strongest arguments.
    topic_res = client.table("debate_topics").select("statement").eq("id", topic_id).limit(1).execute()
    statement = (topic_res.data or [{}])[0].get("statement", "")

    if winning_side == "support":
        candidate_votes = [v for v in votes if v.get("side") == "support"]
    elif winning_side == "oppose":
        candidate_votes = [v for v in votes if v.get("side") == "oppose"]
    else:
        candidate_votes = votes

    top_votes = sorted(
        [v for v in candidate_votes if (v.get("argument_text") or "").strip()],
        key=lambda v: int(v.get("ai_score") or 0),
        reverse=True,
    )[:5]
    top_args_text = (
        "\n".join([f"- {(v.get('argument_text') or '').strip()}" for v in top_votes])
        if top_votes
        else "No strong arguments provided."
    )
    ai_conclusion = (
        "The debate ended in a draw on total logic points, with both sides presenting comparable reasoning."
        if winning_side == "draw"
        else "The winning side presented stronger logical points and practical evidence."
    )
    if statement:
        try:
            from LocalLLMService import conclude_debate as llm_conclude_debate
            ai_conclusion = llm_conclude_debate(statement, winning_side, top_args_text)
        except Exception as e:
            print(f"[DEBATE] LLM conclusion failed for topic {topic_id}: {e}")
        
    # 4. Update topic
    update_data = {
        "status": "completed",
        "support_score": support_sum,
        "oppose_score": oppose_sum,
        "winning_side": winning_side,
        "ai_conclusion": ai_conclusion,
        "end_time": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    client.table("debate_topics").update(update_data).eq("id", topic_id).execute()
    
    return {
        "status": "success",
        "topic_id": topic_id,
        "support_score": support_sum,
        "oppose_score": oppose_sum,
        "support_vote_count": support_vote_count,
        "oppose_vote_count": oppose_vote_count,
        "ai_conclusion": ai_conclusion,
        "winner": winning_side
    }
