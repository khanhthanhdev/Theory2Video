## Goal

Help a coding AI become productive quickly in the Theory2Video (Theory2Manim) codebase. Focus on runtime entry points, data flows, conventions, and quick checks to validate edits.

## High-level architecture (big picture)

- Frontend: `streamlit_app.py` — Streamlit UI used to submit video generation jobs and view progress. Jobs are persisted to `jobs/job_history.json`.
- Core pipeline: `generate_video.py` — orchestrates planning, code generation, scene rendering and final combining into MP4 (see `EnhancedVideoGenerator`, `VideoGenerationConfig`).
- RAG / embeddings: `create_embeddings.py` + `src/rag/` — creates a Chroma vector DB for Manim docs, used when `use_rag=True`.
- Model & provider config: `src/utils/model_registry.py` and `llm_config/` — central place for provider API env names, supported models, and UI mapping.
- Helper tools: `mllm_tools/` — lightweight wrappers for different LLM providers (Gemini, OpenAI, etc.).
- Persistent artifacts: `models/` (onnx & voices), `output/` (generated videos), `jobs/` (job history), `thumbnails/`.

Why: the project separates UI, orchestration, retrieval (RAG), and provider abstractions. Changes that touch the pipeline should respect the async job lifecycle and job-state persistence in `jobs/job_history.json`.

## Important entrypoints

- Local dev UI: run `streamlit run streamlit_app.py` (the Dockerfile already uses this). Streamlit expects `Config` values from `src/config/config.py`.
- Embedding creation (build-time): `python create_embeddings.py` — builds Chroma DB at `Config.CHROMA_DB_PATH`.
- Quick check: `python main.py` prints a trivial greeting; not used in CI.

## Key files and what to look for

- `streamlit_app.py` — job submission, job lifecycle, API-key handling (does not persist API keys), progress callback mapping, and `EnhancedVideoGenerator` initialization. Examples:
  - Jobs persisted: `jobs/job_history.json`
  - Provider-specific API env var is resolved via `src.utils.model_registry.get_providers_config()` and used as `PROVIDERS_CFG[provider]['api_key_env']`.

- `generate_video.py` — the orchestration core. Look for `VideoGenerationConfig` fields (e.g., `use_rag`, `embedding_model`, concurrency limits) when changing defaults or behavior.

- `create_embeddings.py` — creates sample docs and a vector store for RAG. Useful for offline testing of retrieval.

- `src/config/config.py` — centralized configuration constants like `OUTPUT_DIR`, `CHROMA_DB_PATH`, `MANIM_DOCS_PATH`, `EMBEDDING_MODEL`. Edit with care; many modules import this.

- `mllm_tools/` — implement provider adapters (Gemini, OpenAI, etc.). When adding providers, add adapter here and update `model_registry`.

## Project-specific conventions

- Do not persist raw API keys. The UI stores only whether an API key was supplied; keys are applied to environment variables at runtime (see `submit_job` and `_provider_env_var`).
- Jobs are immutable by design aside from status/metadata. Update the job by loading, modifying fields, then calling `save_jobs()` which writes a temp file and replaces atomically.
- Output files are organized under `output/<sanitized_topic>/`. Use `_find_output_file` in `streamlit_app.py` as the canonical lookup.

## Common developer workflows & quick checks

- Start dev UI: `streamlit run streamlit_app.py --server.port 7860`
- Create embeddings locally: `python create_embeddings.py` (this creates sample docs if no Manim docs found and writes to `Config.CHROMA_DB_PATH`).
- Build/test in Docker: the included `Dockerfile` installs system deps (TeX, ffmpeg, sox) and downloads models into `models/` at build time. It runs `python create_embeddings.py` during image build.

Quick validation after code edits

- Lint/format: follow existing repo style (mostly simple Python). Run unit tests if added.
- Smoke test UI: start Streamlit and submit a small example (use the Example button in the UI) and verify a job reaches at least the planning stage in `jobs/job_history.json`.

## Integration points and external dependencies

- LLM providers: controlled via `mllm_tools/` and `src/utils/model_registry.py`. API key env var names are provider-specific; the UI will set the env var at runtime for a single job.
- Models: ONNX and voice assets live in `models/` and are downloaded by the `Dockerfile` during build. Local dev can place `kokoro-quant.onnx` and `voices.bin` into `models/`.
- System tools: `ffmpeg` is used for thumbnail extraction and video combining; `dvisvgm` / TeX are required by Manim rendering (installed in Dockerfile).

## Examples to guide edits

- Add a provider: implement adapter in `mllm_tools/<provider>.py`, register provider/models in `src/utils/model_registry.py`, add display & env var handling via `llm_config/` if needed.
- Change embedding model: update `Config.EMBEDDING_MODEL` in `src/config/config.py` and re-run `python create_embeddings.py`.

## Things NOT to change lightly

- `jobs/job_history.json` format and atomic save semantics. Tests and UI assume the temp-write-and-rename behavior in `save_jobs()`.
- `VideoGenerationConfig` field semantics; renaming or removing fields requires updating `generate_video.py` and any callers (e.g., `streamlit_app.py`).

## Where to look when debugging

- UI errors: Streamlit logs printed to console where you run `streamlit run` and `gradio_app.log` (if present) may contain traces.
- Pipeline errors: `generate_video.py` will bubble exceptions back to job state; inspect `jobs/job_history.json` for message/stack traces. Use local runs of the generator in a controlled script to reproduce.

## Final notes

- This file is intentionally short. If you want more detail on any aspect (e.g., how RAG is wired, or the exact fields of `VideoGenerationConfig`), tell me which area and I'll expand with targeted examples and tests.

---
Please review for accuracy and tell me any missing context (CI commands, local dataset locations, or provider account notes) and I'll iterate.
