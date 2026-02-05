# Gemma-2-2b-it LocalLLM Migration Summary

## Changes Made

### 1. **requirements.txt** 
- ❌ Removed: `torch==2.8.0`, `transformers==4.57.1`
- ✅ Added: `llama-cpp-python==0.2.90`
- **Rationale**: Reduces dependencies (~1.8GB savings), runs entirely on CPU, supports GGUF quantized models.

### 2. **app/LocalLLMService.py** ✅ (NEW)
Encapsulates Gemma-2-2b-it for local inference with:
- **Singleton Pattern**: Loads model once; reused across requests
- **Auto-Download**: Fetches `gemma-2-2b-it-Q4_K_M.gguf` from HuggingFace (bartowski/gemma-2-2b-it-GGUF) on first run
- **analyze_article(text)**: Returns `Dict`:
  ```json
  {
    "summary": "60-word concise summary",
    "upsc_relevant": true/false,
    "tags": ["GS-1", "Polity"] or null
  }
  ```
- **System Prompt**: Instructs model to:
  - Write 60-word summaries (factual, no opinions)
  - Classify UPSC relevance (GS-1, GS-2, GS-3, GS-4)
  - Return JSON with proper structure
  - Mark Sports/Entertainment/Politics as irrelevant (tags: null)

### 3. **app/main.py** ✅ (UPDATED)

#### Imports
- ❌ Removed: `from transformers import pipeline`
- ✅ Added: `from LocalLLMService import get_local_llm_service`
- ✅ Replaced: `summarizer = pipeline(...)` with `llm_service = get_local_llm_service()`

#### `summarize_text_if_possible()` (REFACTORED)
- **Old**: BART model, skipped articles < 100 words, returned single string
- **New**: LocalLLMService, processes ALL articles, returns Dict with summary + tags
- **Behavior**:
  ```python
  result = {
      "summary": "...",
      "upsc_relevant": bool,
      "tags": list or null
  }
  ```

#### `_summarize_in_batches()` (UPDATED)
- **Removed**: 600-character minimum check
- **Updated**: Save logic to handle Dict response, store `tags` to DB as JSON

#### `_prepare_texts()` (UPDATED)
- **Removed**: 600-character filter
- **Updated**: All non-empty articles pass through

#### `summarize_pending_round_robin()` (UPDATED)
- **Removed**: Short article marking logic
- **Updated**: Handle Dict results, save `upsc_relevant` and `tags` fields

#### `process_and_scrape()` (UPDATED)
- **Removed**: 600-character threshold (`if content and len(content) > 600:`)
- **New Logic**:
  - If `content` exists → queue for summarization (`summarization_needed = True`)
  - If no `content` but `description` exists → use as fallback summary
  - If neither exist → skip article

---

## Database Schema Expectations

Your `articles` table should have these columns (if not, add them):
- `summary` (TEXT) — 60-word summary
- `upsc_relevant` (BOOLEAN) — Classification result
- `tags` (JSONB/TEXT) — Array of tags like `["GS-2", "Polity"]` or `null`

**Migration Script** (if needed):
```sql
ALTER TABLE articles ADD COLUMN upsc_relevant BOOLEAN DEFAULT NULL;
ALTER TABLE articles ADD COLUMN tags JSONB DEFAULT NULL;
```

---

## Workflow

### Original (BART)
```
Raw Text → Filter (>600 chars) → BART Summarization → String Summary → Save
```

### New (Gemma-2-2b-it)
```
Raw Text → NO FILTER → LocalLLMService → {summary, upsc_relevant, tags} → Save All
```

---

## First Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **First Inference**:
   - On startup, `get_local_llm_service()` is called
   - Model (~1.3GB) is downloaded from HuggingFace → `app/models/` directory
   - Model loads into memory (CPU; ~4-5GB RAM)
   - Ready for inference

3. **Test**:
   ```bash
   python test_local_llm.py
   ```

---

## Performance Notes

- **Speed**: Gemma-2-2b-it is ~4-6 tokens/sec on CPU (reasonable for 60-word summaries)
- **Latency**: ~10-15 seconds per article (acceptable for batch operations)
- **Memory**: ~4-5GB RAM (singleton pattern keeps one instance)
- **Batching**: Process multiple articles in `_prepare_texts()` loop; summarize sequentially

---

## Rollback (if needed)

If issues arise, to return to BART:
1. Restore old `requirements.txt` (add back `torch`, `transformers`)
2. Revert `main.py` changes from git
3. Reinstall: `pip install torch transformers`

---

## Next Steps (Optional)

- **Optimize Prompting**: Fine-tune system prompt for better UPSC classification
- **Cache Model**: Pre-download GGUF model to avoid first-run delay
- **Parallel Processing**: Use `ThreadPoolExecutor` in `LocalLLMService` for CPU-bound summarization
- **Monitoring**: Log inference times per article to track performance

---

**Status**: ✅ Complete. Ready for testing and deployment.
