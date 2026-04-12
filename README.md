# LLM Comparator (FastAPI + React-ready)

A web backend for side-by-side comparison of multiple LLM responses with confidence-based cascading.

## High-Level Design

![Architecture](./architecture_diagram.png)

## Implementation Details

- **Framework:** `FastAPI` for API development and request handling.
- **Models:** `TinyLlama` and `Phi-2` via `transformers` pipelines.
- **Routing strategies:**
  - `compare_fast` / `compare_full`
  - `cascade_fast` / `cascade_full`
- **Why this approach:**
  - compare modes give transparent side-by-side quality checks.
  - cascade modes reduce compute by escalating only when confidence is low.
  - lightweight lexical retrieval and token estimation keep runtime simple and cost-aware.

## API Endpoints

- `GET /` -> health string (`API is running`)
- `POST /query` (JSON)
  - body:
    ```json
    {
      "query": "Explain this text",
      "mode": "cascade_fast"
    }
    ```
- `POST /query-file` (multipart/form-data)
  - fields:
    - `file` (`.txt` or `.pdf`)
    - `query` (optional)
    - `mode` (optional, default `compare_fast`)
    - `page_start` (optional, 1-indexed)
    - `page_end` (optional, 1-indexed)

## Build and Run

1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start backend:
   ```bash
   uvicorn backend.main:app --reload
   ```
4. Open Swagger:
   - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Testing Checklist

- `GET /` returns `"API is running"`.
- `/query` works in `compare_fast`, `compare_full`, `cascade_fast`, `cascade_full`.
- `/query-file` works with `.txt` and `.pdf`.
- PDF page-range extraction works with `page_start`/`page_end` — testable via Swagger at `/docs`; frontend UI for page range is a planned enhancement.
- Cascade only escalates on low confidence or truncation.
- Outputs include latency and token usage estimates.

## Demo Videos

Demo videos are available here:

👉 [Watch Demo Videos](https://drive.google.com/drive/folders/1C_eWlLXLWGYGn7Cbd24D8E7vbuP89U-I?usp=sharing)

## Pros

- **No external API calls** - both models run fully locally, so there are no token costs or data privacy concerns.
- **Cascade mode is compute-efficient** - Phi-2 only runs when TinyLlama's confidence is below threshold, saving significant time on straightforward queries.
- **Transparent scoring** - every response includes latency, relevance, truncation flag, and token estimate, so you can see exactly why one model was picked over the other.
- **PDF page-range support** - you can target specific sections of a document instead of feeding the entire file, which keeps context clean and inference fast.
- **Retry logic**  if a model produces a weak or incomplete answer, it automatically retries with a stricter prompt before giving up.
- **Lexical RAG without embeddings** - chunk retrieval uses token overlap scoring, which is fast and requires no vector database or embedding model.
- **Real-time token streaming** - the `/ws` WebSocket endpoint streams actual model tokens as they are generated using `TextIteratorStreamer`, giving the frontend live output.

## Future Enhancements

- **GPU support** - add `device_map="auto"` and CUDA detection so the models automatically use a GPU when available, reducing latency from minutes to seconds.
- **Swap or extend models** - the architecture already supports adding more pipelines (e.g. Qwen, Mistral); a model registry config would make this plug-and-play without touching core logic.
- **Semantic similarity scoring** - replace lexical overlap with a lightweight embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`) for more accurate relevance and agreement scores.
- **Quantization** - load models in 4-bit or 8-bit via `bitsandbytes` to cut memory usage and speed up CPU inference significantly.
- **Caching** - cache responses for repeated queries using a simple hash-based store to avoid re-running expensive inference.
- **Frontend integration** - connect the React frontend fully to all four modes with real-time progress indicators, side-by-side diff highlighting, and page range input for PDF uploads.

## Security Notes

- Credentials should be stored in environment variables only.
- `.env` is excluded via `.gitignore`.
- Avoid committing sensitive data (API keys, PII, personal emails).

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.
