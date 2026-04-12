import re
import threading
from typing import Optional
from PyPDF2 import PdfReader
from transformers import TextIteratorStreamer


# =========================
# CHUNKING
# =========================
def chunk_text(text, size=220):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(re.findall(r"\S+", text)) * 1.3))

# =========================
# RETRIEVAL (LIGHTWEIGHT)
# =========================
def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_relevant_chunks(query, chunks, top_k=1):
    # Fast lexical overlap scoring to avoid embedding overhead.
    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]
    ranked = sorted(
        chunks,
        key=lambda chunk: len(query_tokens.intersection(_tokenize(chunk))),
        reverse=True,
    )
    return ranked[:top_k]

# =========================
# TASK DETECTION
# =========================
def detect_task(query, file):
    return "document" if file else "qa"

# =========================
# CLEAN OUTPUT
# =========================
def clean_output(text):
    text = re.sub(r"\s+", " ", text).strip()
    # OCR cleanup: normalize separators and repeated punctuation/noise.
    text = re.sub(r"[|_]{2,}", " ", text)
    text = re.sub(r"[-]{3,}", " ", text)
    text = re.sub(r"\s*([,:;])\s*", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Strip common prompt/formatting artifacts instead of dropping full output.
    text = re.sub(r"\b(question|answer|summary)\s*:\s*", " ", text, flags=re.IGNORECASE)
    if "possible rewrite" in text.lower() or "exercise" in text.lower():
        return "Not enough clean information available."
    lines = [ln.strip(" -•\t") for ln in text.split("\n") if ln.strip()]
    text = " ".join(lines)
    # Drop very noisy OCR-style fragments that are mostly numbers.
    if re.fullmatch(r"[\d\s,./\-:()]+", text):
        return "Not enough clean information available."
    if len(text.split()) < 5:
        return "Not enough information available."
    # If there are many OCR-like uppercase tokens, normalize for readability.
    words = text.split()
    if words:
        uppercase_ratio = sum(1 for w in words if len(w) > 2 and w.isupper()) / len(words)
        if uppercase_ratio > 0.35:
            text = text.capitalize()

    text = re.sub(r"(name of entity|amt\(rs\)|particular amt\(rs\)).*$", "", text, flags=re.IGNORECASE).strip()
    if len(text.split()) < 5:
        return "Not enough clean information available."
    return text.strip()

# =========================
# GENERATION
# =========================
def run_model(model, context, question, task="qa", max_tokens=60, context_limit=800):
    context = context[:context_limit]
    if task == "qa":
        prompt = f"""
Answer clearly in 2-3 sentences.

Context:
{context}

Question:
{question}

Answer:
"""
    else:
        prompt = f"""
Write a clear 2-3 sentence summary of the document below. Include the main topic, key entities, and important details. Do not use bullet points or labels.

Document:
{context}

Summary:
"""
    output = model(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.2
    )[0]["generated_text"]

    result = output.replace(prompt, "").strip()
    for stop in ["Question:", "Answer:", "Summary:"]:
        result = result.split(stop)[0]
    return clean_output(result)


def build_prompt(context: str, question: str, task: str, strict: bool = False) -> str:
    if task == "qa":
        if strict:
            return f"""
Answer in 2-4 complete sentences.
Do not include labels like Question/Answer.
Do not repeat the prompt.

Context:
{context}

Question:
{question}

Final response:
"""
        return f"""
Answer clearly in 2-3 sentences.

Context:
{context}

Question:
{question}

Answer:
"""
    if strict:
        return f"""
Provide a concise summary in complete sentences only.
Do not include labels or bullet formatting.

Document:
{context}

Final summary:
"""
    return f"""
Write a clear 2-3 sentence summary of the document below. Include the main topic, key entities, and important details. Do not use bullet points or labels.

Document:
{context}

Summary:
"""


def run_model_with_retry(model, context, question, task="qa", max_tokens=60, context_limit=800):
    # First pass (default prompt style)
    res = run_model(model, context, question, task, max_tokens=max_tokens, context_limit=context_limit)
    if (
        res != "Not enough clean information available."
        and res != "Not enough information available."
        and len(res.split()) >= 10
    ):
        return res

    # Retry once with stricter prompt instructions and slightly higher token budget.
    context = context[:context_limit]
    retry_prompt = build_prompt(context, question, task, strict=True)
    output = model(
        retry_prompt,
        max_new_tokens=min(max_tokens + 40, 240),
        do_sample=False,
        temperature=0.2
    )[0]["generated_text"]
    result = output.replace(retry_prompt, "").strip()
    for stop in ["Question:", "Answer:", "Summary:", "Final response:", "Final summary:"]:
        result = result.split(stop)[0]
    cleaned = clean_output(result)

   
    if len(cleaned.split()) > len(res.split()):
        return cleaned
    return res


def get_mode_config(mode: str) -> dict:
    """
    compare_fast: low latency, partial summary friendly
    compare_full: better coverage for full-document summaries
    """
    normalized = (mode or "compare_fast").strip().lower()
    if normalized == "compare":
        normalized = "compare_fast"
    if normalized == "cascade":
        normalized = "cascade_fast"
    if normalized == "cascade_full":
        return {
            "mode": "cascade_full",
            "strategy": "cascade",
            "task": "document",
            "chunk_size": 170,
            "top_k": 4,
            "short_text_limit": 2200,
            "context_limit": 2400,
            "max_tokens": 180,
            "confidence_threshold": 0.78,
        }
    if normalized == "cascade_fast":
        return {
            "mode": "cascade_fast",
            "strategy": "cascade",
            "task": "qa",
            "chunk_size": 220,
            "top_k": 1,
            "short_text_limit": 1200,
            "context_limit": 800,
            "max_tokens": 60,
            "confidence_threshold": 0.82,
        }
    if normalized == "compare_full":
        return {
            "mode": "compare_full",
            "strategy": "compare",
            "task": "document",
            "chunk_size": 170,
            "top_k": 4,
            "short_text_limit": 2200,
            "context_limit": 2400,
            "max_tokens": 180,
        }
    return {
        "mode": "compare_fast",
        "strategy": "compare",
        "task": "qa",
        "chunk_size": 220,
        "top_k": 1,
        "short_text_limit": 1200,
        "context_limit": 800,
        "max_tokens": 120,
    }


def score_confidence(context: str, question: str, response: str, task: str) -> float:
    # Return 0 immediately for known failure strings.
    if "Not enough" in response:
        return 0.0

    ctx_tokens = _tokenize(context)
    q_tokens = _tokenize(question)
    resp_tokens = _tokenize(response)
    if not resp_tokens:
        return 0.0

    context_overlap = len(ctx_tokens.intersection(resp_tokens)) / len(resp_tokens)
    question_overlap = len(q_tokens.intersection(resp_tokens)) / max(len(q_tokens), 1)
    if task == "qa":
        overlap = (0.55 * context_overlap) + (0.45 * question_overlap)
    else:
        overlap = (0.8 * context_overlap) + (0.2 * question_overlap)

    words = response.split()
    length_bonus = 0.1 if len(words) >= 20 else 0.0
    penalty = 0.0

    # Mild penalty for truncated endings.
    bad_tail_tokens = {"is", "are", "was", "were", "to", "of", "for", "at", "in"}
    tail = response.strip()
    last_word = words[-1].lower() if words else ""
    if tail.endswith((":", "-", ",", ";")) or last_word in bad_tail_tokens:
        penalty += 0.08

    # Mild penalty for very short outputs.
    if len(words) < 12:
        penalty += 0.05

    raw_score = overlap + length_bonus - penalty
    return round(max(0.0, min(0.95, raw_score)), 2)


def detect_truncation(response: str) -> bool:
    text = response.strip()
    if not text:
        return True
    words = text.split()
    if not words:
        return True
    last_word = re.sub(r"[^a-zA-Z]", "", words[-1]).lower()
    # Typical chopped stems from token limits.
    chopped_stems = ("dis", "incl", "cont", "acc", "fin", "stat", "doc", "info")
    if any(last_word.startswith(stem) for stem in chopped_stems):
        return True
    # Not ending with sentence punctuation often means the output is incomplete.
    if not text.endswith((".", "!", "?")):
        return True
    return False


def extract_pdf_text(reader: PdfReader, page_start: Optional[int], page_end: Optional[int]) -> str:
    total_pages = len(reader.pages)
    if total_pages == 0:
        return ""

    # Page inputs are 1-indexed for user convenience.
    start = page_start if page_start and page_start > 0 else 1
    end = page_end if page_end and page_end > 0 else total_pages
    if start > end:
        start, end = end, start
    start = max(1, min(start, total_pages))
    end = max(1, min(end, total_pages))

    text = ""
    for idx in range(start - 1, end):
        page_text = reader.pages[idx].extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# =========================
# REAL-TIME STREAMING
# =========================
def stream_model(model, context: str, question: str, task: str, max_tokens: int = 120, context_limit: int = 800):
    """
    Yields tokens one by one using TextIteratorStreamer.
    Runs model inference in a background thread so the main thread can stream.
    """
    context = context[:context_limit]
    if task == "qa":
        prompt = f"""Answer clearly in 2-3 sentences.

Context:
{context}

Question:
{question}

Answer:
"""
    else:
        prompt = f"""Write a clear 2-3 sentence summary of the document below. Include the main topic, key entities, and important details. Do not use bullet points or labels.

Document:
{context}

Summary:
"""
    tokenizer = model.tokenizer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": False,
        "temperature": 0.2,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()
