from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio

from PyPDF2 import PdfReader
from typing import Optional

from models import QueryBody
from llm_loader import model1, model2
from utils import (
    chunk_text,
    estimate_tokens,
    retrieve_relevant_chunks,
    run_model_with_retry,
    get_mode_config,
    score_confidence,
    detect_truncation,
    extract_pdf_text,
    _tokenize,
    stream_model,
)

app = FastAPI()

# =========================
# CORS (Frontend Connection)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROOT
# =========================
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "API is running"

# =========================
# REST API (NON-STREAMING)
# =========================
def run_comparison(text: str, question: str, mode: str, task: str):
    text = text.strip()
    cfg = get_mode_config(mode)

    # -------------------------
    # RAG
    # -------------------------
    chunks = chunk_text(text, size=cfg["chunk_size"])
    # For short text, skip retrieval entirely (faster + cleaner).
    if len(text) <= cfg["short_text_limit"]:
        relevant_chunks = [text]
    else:
        relevant_chunks = retrieve_relevant_chunks(question, chunks, top_k=cfg["top_k"])
    context = " ".join(relevant_chunks)

    # -------------------------
    # MODEL RUN
    # -------------------------
    start1 = time.time()
    res1 = run_model_with_retry(
        model1,
        context,
        question,
        task,
        max_tokens=cfg["max_tokens"],
        context_limit=cfg["context_limit"],
    )
    end1 = time.time()
    latency1 = round(end1 - start1, 2)

    # Lightweight relevance proxy via token overlap with context.
    rel1 = score_confidence(context, question, res1, task)
    trunc1 = detect_truncation(res1)
    if trunc1:
        rel1 = round(max(0.0, rel1 - 0.08), 2)

    response_payload = {
        "mode": cfg["mode"],
        "strategy": cfg["strategy"],
        "retrieved_context_preview": context[:350],
        "cascade_used": cfg["strategy"] == "cascade",
        "escalated_to_phi2": False,
        "tinyllama_confidence": rel1,
        "results": [
            {
                "model": "TinyLlama",
                "response": res1,
                "latency": latency1,
                "relevance": rel1,
                "truncation_detected": trunc1,
                "token_usage_estimate": estimate_tokens(res1)
            }
        ]
    }
    if cfg["strategy"] == "cascade":
        response_payload["confidence_threshold"] = cfg.get("confidence_threshold")

    should_escalate = (
        cfg["strategy"] == "compare"
        or rel1 < cfg.get("confidence_threshold", 1.0)
        or (cfg["strategy"] == "cascade" and trunc1)
    )
    if should_escalate:
        start2 = time.time()
        res2 = run_model_with_retry(
            model2,
            context,
            question,
            task,
            max_tokens=cfg["max_tokens"],
            context_limit=cfg["context_limit"],
        )
        end2 = time.time()
        latency2 = round(end2 - start2, 2)
        rel2 = score_confidence(context, question, res2, task)
        trunc2 = detect_truncation(res2)
        if trunc2:
            rel2 = round(max(0.0, rel2 - 0.08), 2)
        r1_tokens = _tokenize(res1)
        r2_tokens = _tokenize(res2)
        union = r1_tokens.union(r2_tokens)
        # If Phi-2 failed entirely, don't penalize agreement — use TinyLlama as winner.
        if "Not enough" in res2:
            response_payload["escalated_to_phi2"] = True
            response_payload["model_agreement"] = rel1
            response_payload["best_model"] = "TinyLlama"
            response_payload["final_answer"] = res1
            response_payload["results"].append(
                {
                    "model": "Phi-2",
                    "response": res2,
                    "latency": latency2,
                    "relevance": 0.0,
                    "truncation_detected": trunc2,
                    "token_usage_estimate": estimate_tokens(res2)
                }
            )
            return response_payload
        # If one model failed, use the winner's relevance as the agreement score.
        if "Not enough" in res1:
            agree = rel2
        else:
            agree = round(len(r1_tokens.intersection(r2_tokens)) / max(len(union), 1), 2)

        best_model = "TinyLlama" if rel1 >= rel2 else "Phi-2"
        winner_text = res1 if best_model == "TinyLlama" else res2
        response_payload["escalated_to_phi2"] = True
        response_payload["model_agreement"] = agree
        response_payload["best_model"] = best_model
        response_payload["final_answer"] = winner_text
        response_payload["results"].append(
            {
                "model": "Phi-2",
                "response": res2,
                "latency": latency2,
                "relevance": rel2,
                "truncation_detected": trunc2,
                "token_usage_estimate": estimate_tokens(res2)
            }
        )
        return response_payload

    response_payload["model_agreement"] = 1.0
    response_payload["best_model"] = "TinyLlama"
    response_payload["final_answer"] = res1
    return response_payload


@app.post("/query")
async def query_endpoint(body: QueryBody):
    try:
        if not body.query or not body.query.strip():
            return {"error": "Provide query"}
        mode = body.mode or "compare_fast"
        # Query endpoint should always use QA prompting.
        return run_comparison(body.query, body.query, mode, task="qa")
    except Exception as e:
        return {"error": str(e)}


@app.post("/query-file")
async def query_file_endpoint(
    file: UploadFile = File(...),
    query: Optional[str] = Form(default=None),
    mode: str = Form(default="compare_fast"),
    page_start: Optional[int] = Form(default=None),
    page_end: Optional[int] = Form(default=None),
):
    try:
        if page_start is not None and page_start < 1:
            return {"error": "page_start must be >= 1"}
        if page_end is not None and page_end < 1:
            return {"error": "page_end must be >= 1"}

        if file.filename.endswith(".txt"):
            text = (await file.read()).decode("utf-8")
        elif file.filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            text = extract_pdf_text(reader, page_start, page_end)
        else:
            return {"error": "Only .txt and .pdf supported"}

        if not text.strip():
            return {"error": "No extractable text found in selected file/page range"}

        question = query if query and query.strip() else "Summarize the document"
        # For files: QA if user asked a question, otherwise document summary.
        task = "qa" if query and query.strip() else "document"
        return run_comparison(text, question, mode, task=task)
    except Exception as e:
        return {"error": str(e)}

# =========================
# WEBSOCKET STREAMING (Real)
# =========================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()
    query = data.get("query", "")
    mode = data.get("mode", "compare_fast")
    task = "qa"

    cfg = get_mode_config(mode)
    chunks = chunk_text(query, size=cfg["chunk_size"])
    context = query if len(query) <= cfg["short_text_limit"] else " ".join(
        retrieve_relevant_chunks(query, chunks, top_k=cfg["top_k"])
    )

    results = []

    for model_name, model in [("TinyLlama", model1), ("Phi-2", model2)]:
        await websocket.send_json({"type": "start", "model": model_name})
        partial = ""
        start = asyncio.get_event_loop().time()

        try:
            for token in stream_model(model, context, query, task, max_tokens=cfg["max_tokens"], context_limit=cfg["context_limit"]):
                partial += token
                await websocket.send_json({
                    "type": "token",
                    "model": model_name,
                    "token": token,
                    "partial": partial
                })
                await asyncio.sleep(0)  # yield control to event loop
        except Exception as e:
            await websocket.send_json({"type": "error", "model": model_name, "message": str(e)})
            partial = "Error during generation."

        latency = round(asyncio.get_event_loop().time() - start, 2)
        relevance = score_confidence(context, query, partial, task)
        results.append({
            "model": model_name,
            "response": partial,
            "latency": latency,
            "relevance": relevance,
        })

        await websocket.send_json({"type": "done", "model": model_name, "latency": latency, "relevance": relevance})

        # Cascade: skip Phi-2 if TinyLlama was confident enough
        if cfg["strategy"] == "cascade" and results[0]["relevance"] >= cfg.get("confidence_threshold", 0.82):
            break

    best = max(results, key=lambda r: r["relevance"])
    r1_tokens = _tokenize(results[0]["response"]) if len(results) > 0 else set()
    r2_tokens = _tokenize(results[1]["response"]) if len(results) > 1 else set()
    union = r1_tokens.union(r2_tokens)
    agreement = round(len(r1_tokens.intersection(r2_tokens)) / max(len(union), 1), 2) if union else 1.0

    await websocket.send_json({
        "type": "final",
        "results": results,
        "best_model": best["model"],
        "final_answer": best["response"],
        "model_agreement": agreement,
        "mode": cfg["mode"],
    })

