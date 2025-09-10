# Repository Guidelines

## Project Structure & Module Organization
- `src/` — application code (`config/`, `core/`, `rag/`, `utils/`).
- `gradio_app.py` — main UI server; orchestrates generation pipeline.
- `generate_video.py` — video planning/render pipeline and configs.
- `tests/` — pytest suites (e.g., `tests/core/test_*.py`).
- `templates/` — contributor/plan templates; `scripts/` — project helpers.
- `data/`, `models/`, `output/`, `thumbnails/`, `jobs/` — runtime assets and results.
- `specs/` — feature specs per branch; `llm_config/`, `task_generator/` — LLM assets.

## Build, Test, and Development Commands
- Setup venv + deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run UI locally: `python gradio_app.py` (serves on port 7860 by default)
- Tests: `python -m pytest tests/ -q` (supports `pytest-asyncio`)
- Format/Lint: `black . && isort . && flake8 .`
- Docker (local): `docker-compose up --build` or `docker build -t theory2video .`
- Pipeline eval (optional): `python evaluate.py`
- Feature scaffolding: `scripts/create-new-feature.sh "my feature"` → `specs/NNN-my-feature/`

## Coding Style & Naming Conventions
- Python 3.11+; 4‑space indentation; prefer type hints and concise functions.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Keep side effects behind `if __name__ == "__main__":` in new scripts.
- Tools: format with `black`, sort imports with `isort`, lint with `flake8`.

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio`). Place tests under `tests/<area>/test_*.py`.
- Write unit tests for new/changed logic in `src/core` and planners; mock external I/O.
- Run: `python -m pytest -q`; filter with `-k substring`; add fixtures as needed.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (≤50 chars), descriptive body. Example: `core: refactor planner stage updates (#123)`.
- Reference issues (`#ID`) and affected modules (`core`, `ui`, `docs`).
- PRs: include summary, rationale, test steps, linked issues, and screenshots/logs for UI changes. Note any `.env`/config changes.

## Security & Configuration Tips
- Use `.env` for secrets; do not commit keys. Review `.gitignore`/`.dockerignore`.
- Large/ephemeral outputs (`output/`, `jobs/`, `thumbnails/`) should not be versioned.

## Agent-Specific Tips
- Use `scripts/setup-plan.sh` and `scripts/check-task-prerequisites.sh` on feature branches named like `NNN-feature-name`.
- Keep specs in `specs/NNN-feature-name/` (`plan.md`, `spec.md`); update agent context via `scripts/update-agent-context.sh`.
