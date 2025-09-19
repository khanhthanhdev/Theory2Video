import os
import re
import json
import time
import uuid
import asyncio
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

import base64
import mimetypes
import streamlit as st

from src.config.config import Config
from generate_video import EnhancedVideoGenerator, VideoGenerationConfig, allowed_models, default_model
from src.utils.model_registry import get_providers_config

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
local_css("assets/style.css")
# ----------------------------
# Page config and light styling
# ----------------------------
st.set_page_config(page_title="Theory2Manim • Demo", layout="wide")


def _image_to_data_uri(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _rerun():
    """Compatibility rerun across Streamlit versions."""
    try:
        # Newer Streamlit versions
        if hasattr(st, "rerun"):
            st.rerun()
            return
        # Older experimental API
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass


def build_theme_css(mode: str = "System") -> str:
    light_vars = """:root{ --bg:#f8fafc; --card:#ffffff; --border:#e5e7eb; --text:#111827; --muted:#6b7280; --primary:#2563eb; --progressFrom:#60a5fa; --progressTo:#2563eb; }"""
    dark_vars = """:root{ --bg:#0b1220; --card:#0f172a; --border:#1f2937; --text:#e5e7eb; --muted:#9aa4b2; --primary:#3b82f6; --progressFrom:#1d4ed8; --progressTo:#60a5fa; }"""

    if mode == "Light":
        var_block = light_vars
    elif mode == "Dark":
        var_block = dark_vars
    else:  # System
        var_block = light_vars + "@media (prefers-color-scheme: dark){" + dark_vars + "}"

    # Invert filter class for single-logo setups (when no dark logo provided)
    if mode == "Dark":
        invert_rule = ".invert-dark{ filter: invert(1) hue-rotate(180deg) contrast(0.9) brightness(0.9); }"
    elif mode == "Light":
        invert_rule = ".invert-dark{ filter: none; }"
    else:
        invert_rule = ".invert-dark{ filter: none; } @media (prefers-color-scheme: dark){ .invert-dark{ filter: invert(1) hue-rotate(180deg) contrast(0.9) brightness(0.9); } }"

    base_css = f"""
    <style>
    {var_block}
    body {{ background: var(--bg); color: var(--text); }}
    .app-header {{ position:sticky; top:0; z-index:10; background:var(--card); border-bottom:1px solid var(--border); }}
    .app-shell {{ max-width: 1152px; margin: 0 auto; padding: 12px 8px; }}
    .brand {{ display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:-0.01em; color: var(--text); }}
    .brand .dot {{ width:10px; height:10px; border-radius:999px; background:var(--primary); display:inline-block; box-shadow:0 0 0 4px rgba(59,130,246,.15); }}
    .brand .logo-img {{ height:24px; width:auto; display:inline-block; }}
    .pill {{ font-size:12px; color:var(--text); background:transparent; border:1px solid var(--border); padding:2px 8px; border-radius:999px; opacity:.9; }}
    .hero h1 {{ margin:0; font-weight:800; letter-spacing:-0.02em; }}
    .hero p {{ margin: 6px 0 0 0; color: var(--muted); }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; box-shadow:0 8px 24px rgba(0,0,0,0.25); padding:18px; }}
    .muted {{ color: var(--muted); }}
    .progress-wrap {{ margin-top:8px; }}
    input[type="text"], textarea {{ font-size: 16px !important; padding: 12px 14px !important; border-radius: 10px !important; color: var(--text) !important; background: var(--card) !important; }}
    input[type="text"]{{ height: 52px !important; }}
    .stButton > button {{ width: 100%; padding: 0.8rem 1rem; border-radius: 10px; font-weight: 700; }}
    @media (max-width: 1100px){{ .app-shell {{ padding: 8px 12px; }} }}
    @media (max-width: 900px){{
      div[data-testid="stHorizontalBlock"]{{ flex-direction: column !important; gap: 1rem !important; }}
      div[data-testid="column"]{{ width: 100% !important; }}
    }}
    {invert_rule}
    </style>
    """
    return base_css



# ----------------------------
# Job storage helpers (shared file)
# ----------------------------
JOB_STORE_PATH = "jobs/job_history.json"
JOBS_LOCK = threading.Lock()


def load_jobs() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(JOB_STORE_PATH):
        return {}
    try:
        with open(JOB_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_jobs(data: Dict[str, Dict[str, Any]]):
    os.makedirs(os.path.dirname(JOB_STORE_PATH), exist_ok=True)
    tmp = JOB_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, JOB_STORE_PATH)


def _persist_job(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with JOBS_LOCK:
        data = load_jobs()
        data[job_id] = payload
        save_jobs(data)
        return payload


def _mutate_job(job_id: str, mutator) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        data = load_jobs()
        job = data.get(job_id)
        if not job:
            return None
        mutator(job)
        save_jobs(data)
        return job


CANCEL_EVENTS: Dict[str, threading.Event] = {}
CANCEL_EVENTS_LOCK = threading.Lock()


def _register_cancel_event(job_id: str) -> threading.Event:
    with CANCEL_EVENTS_LOCK:
        event = threading.Event()
        CANCEL_EVENTS[job_id] = event
        return event


def _get_cancel_event(job_id: str) -> Optional[threading.Event]:
    with CANCEL_EVENTS_LOCK:
        return CANCEL_EVENTS.get(job_id)


def _clear_cancel_event(job_id: str):
    with CANCEL_EVENTS_LOCK:
        CANCEL_EVENTS.pop(job_id, None)


PROVIDERS_CFG = get_providers_config()


# ----------------------------
# Pipeline helpers
# ----------------------------
DEFAULT_MODEL = default_model if isinstance(default_model, str) else (
    allowed_models[0] if isinstance(allowed_models, list) and allowed_models else "gemini/gemini-2.5-pro"
)


def validate_model(model_name: str) -> bool:
    return isinstance(model_name, str) and "/" in model_name and model_name in allowed_models


def init_video_generator(model: str, temperature: float, quality: str = "medium") -> EnhancedVideoGenerator:
    cfg = VideoGenerationConfig(
        planner_model=model,
        scene_model=model,
        helper_model=model,
        temperature=float(temperature),
        output_dir=Config.OUTPUT_DIR,
        verbose=True,
        use_rag=False,
        use_context_learning=False,
        context_learning_path=Config.CONTEXT_LEARNING_PATH,
        chroma_db_path=Config.CHROMA_DB_PATH,
        manim_docs_path=Config.MANIM_DOCS_PATH,
        embedding_model=Config.EMBEDDING_MODEL,
        use_visual_fix_code=False,
        use_langfuse=True,
        max_scene_concurrency=5,
        max_topic_concurrency=2,
        max_concurrent_renders=5,
        max_retries=3,
        enable_caching=True,
        default_quality=quality,
        use_gpu_acceleration=False,
        preview_mode=False,
    )
    return EnhancedVideoGenerator(cfg)


def _find_output_file(topic: str, output_dir: str) -> Optional[str]:
    file_prefix = re.sub(r"[^a-z0-9_]+", "_", topic.lower())
    combined = os.path.join(output_dir, file_prefix, f"{file_prefix}_combined.mp4")
    if os.path.exists(combined):
        return combined
    scene_dir = os.path.join(output_dir, file_prefix)
    if os.path.exists(scene_dir):
        for root, _, files in os.walk(scene_dir):
            for f in files:
                if f.endswith(".mp4"):
                    return os.path.join(root, f)
    return None


def _provider_env_var(model_name: str) -> Optional[str]:
    try:
        provider = (model_name or "").split("/", 1)[0].lower()
        return PROVIDERS_CFG.get(provider, {}).get("api_key_env")
    except Exception:
        return None


async def _run_pipeline(job_id: str, topic: str, description: str, model: str, temperature: float, quality: str, api_key: Optional[str]):
    cancel_event = _get_cancel_event(job_id)

    def _update(updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return _mutate_job(job_id, lambda job: job.update(updates))

    try:
        if _update({
            "status": "initializing",
            "stage": "initializing",
            "progress": 5,
            "message": "Initializing video generator...",
            "start_time": datetime.now().isoformat(),
        }) is None:
            return

        try:
            env_var = _provider_env_var(model)
            if api_key and env_var:
                os.environ[env_var] = api_key
            vg = init_video_generator(model, temperature, quality)
            _update({"message": "Video generator ready", "progress": 10})
        except Exception as e:
            _update({"status": "failed", "message": f"Initialization error: {e}"})
            return

        def on_progress(stage: str, sub_progress: int = 0, msg: Optional[str] = None):
            if cancel_event and cancel_event.is_set():
                return

            base_map = {
                "planning": 20,
                "implementation_planning": 30,
                "code_generation": 45,
                "scene_rendering": 75,
                "video_combining": 90,
                "finalizing": 95,
            }

            def _apply(job: Dict[str, Any]):
                base = base_map.get(stage, job.get("progress", 10))
                total = int(min(99, base + (sub_progress or 0) // 2))
                job.update({
                    "status": stage,
                    "stage": stage,
                    "progress": total,
                    "message": msg or stage.replace("_", " ").title(),
                    "last_updated": datetime.now().isoformat(),
                })

            _mutate_job(job_id, _apply)

        if hasattr(vg, "set_progress_callback"):
            vg.set_progress_callback(on_progress)

        pipeline_task = asyncio.create_task(
            vg.generate_video_pipeline(
                topic=topic,
                description=description,
                only_plan=False,
                specific_scenes=None,
            )
        )
        pipeline_result: Optional[Dict[str, Any]] = None

        while True:
            if cancel_event and cancel_event.is_set():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    raise

            if pipeline_task.done():
                pipeline_result = await pipeline_task
                break

            await asyncio.sleep(0.5)

        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError

        if _update({
            "status": "finalizing",
            "stage": "finalizing",
            "progress": 95,
            "message": "Finalizing output...",
        }) is None:
            return

        output_file = None
        if pipeline_result and isinstance(pipeline_result, dict):
            output_file = pipeline_result.get("output_file")
        if not output_file:
            output_file = _find_output_file(topic, Config.OUTPUT_DIR)
        if not output_file:
            _update({"status": "failed", "message": "No video output was generated"})
            return

        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError

        thumb = os.path.join("thumbnails", f"{job_id}.jpg")
        try:
            import subprocess

            res = subprocess.run(
                ["ffmpeg", "-i", output_file, "-ss", "00:00:03", "-frames:v", "1", thumb],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                thumb = None
        except Exception:
            thumb = None

        _update({
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "message": "Video generation completed",
            "output_file": output_file,
            "thumbnail": thumb,
            "end_time": datetime.now().isoformat(),
        })

    except asyncio.CancelledError:
        _update({
            "status": "cancelled",
            "stage": "cancelled",
            "message": "Job cancelled by user",
            "last_updated": datetime.now().isoformat(),
        })
    except Exception as e:
        print("Pipeline error:", e)
        print(traceback.format_exc())
        _update({
            "status": "failed",
            "message": f"Unexpected error: {e}",
            "last_updated": datetime.now().isoformat(),
        })
    finally:
        _clear_cancel_event(job_id)


def _start_async(job_id: str, topic: str, description: str, model: str, temperature: float, quality: str, api_key: Optional[str]):
    def _runner():
        asyncio.run(_run_pipeline(job_id, topic, description, model, temperature, quality, api_key))
    threading.Thread(target=_runner, daemon=True).start()


def submit_job(topic: str, description: str, model: str, temperature: float, quality: str, api_key: Optional[str]):
    if not topic or len(topic.strip()) < 3:
        return None, "Please enter a topic (min 3 chars)."
    if not description or len(description.strip()) < 10:
        return None, "Please add a more detailed description (>= 10 chars)."
    if not validate_model(model):
        return None, f"Invalid model. Choose from: {allowed_models}"

    jid = str(uuid.uuid4())
    job_payload = {
        "id": jid,
        "topic": topic.strip(),
        "description": description.strip(),
        "status": "pending",
        "stage": "pending",
        "progress": 0,
        "message": "Queued...",
        "start_time": datetime.now().isoformat(),
        # Do not persist raw API keys to disk; only store model params
        "params": {"model": model, "temperature": float(temperature), "quality": quality, "has_api_key": bool(api_key)},
    }
    _persist_job(jid, job_payload)
    _register_cancel_event(jid)
    _start_async(jid, topic.strip(), description.strip(), model, float(temperature), quality, api_key)
    return jid, "Job submitted!"


def get_status(job_id: Optional[str]):
    data = load_jobs()
    if not job_id or job_id not in data:
        return None, 0, "No active job.", "inactive"
    j = data[job_id]
    status = j.get("status", "unknown")
    progress = int(j.get("progress", 0))
    msg = j.get("message", status.title())
    return j, progress, msg, status


def cancel_job(job_id: str):
    if not job_id:
        return "No active job."
    event = _get_cancel_event(job_id)
    if event:
        event.set()

    updated = _mutate_job(job_id, lambda job: job.update({
        "status": "cancelled",
        "stage": "cancelled",
        "message": "Job cancelled by user",
        "last_updated": datetime.now().isoformat(),
    }))

    if updated is None:
        return "Job not found."
    return "Job cancelled."


# ----------------------------
# UI
# ----------------------------
def resolved_theme(mode: str) -> str:
    if mode == "System":
        try:
            base = st.get_option("theme.base")
            return "Dark" if str(base).lower() == "dark" else "Light"
        except Exception:
            return "Light"
    return mode

# Sidebar navigation and config
page = st.sidebar.radio("Navigation", ["Generate", "Job History"], index=0, key="page")
st.sidebar.subheader("Appearance")
theme_mode = st.sidebar.selectbox("Theme", ["System", "Light", "Dark"], index=0, key="theme_mode")

# Inject theme-aware CSS
st.markdown(build_theme_css(theme_mode), unsafe_allow_html=True)

# Logo handling (optional): set env LOGO_LIGHT / LOGO_DARK to file paths
logo_light = _image_to_data_uri(os.getenv("LOGO_LIGHT"))
logo_dark = _image_to_data_uri(os.getenv("LOGO_DARK"))
current = resolved_theme(theme_mode)
logo_uri = logo_dark if current == "Dark" and logo_dark else logo_light
logo_class = "logo-img" + (" invert-dark" if (logo_uri and not logo_dark) else "")
logo_html = f"<img class='{logo_class}' src='{logo_uri}' alt='logo'/>" if logo_uri else "<span class='dot'></span>"

st.markdown(
    f"""
    <div class='app-header'>
      <div class='app-shell'>
        <div class='brand'>{logo_html} Theory2Manim <span class='pill'>Demo</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

# Pre-apply example prefill values on rerun before widgets mount
if "prefill" in st.session_state and not st.session_state.get("prefill_applied"):
    pf = st.session_state["prefill"]
    st.session_state["topic"] = pf.get("topic", "")
    st.session_state["description"] = pf.get("description", "")
    st.session_state["model"] = pf.get("model", DEFAULT_MODEL)
    st.session_state["temperature"] = pf.get("temperature", 0.7)
    st.session_state["quality"] = pf.get("quality", "medium")
    st.session_state["prefill_applied"] = True

with st.sidebar:
    st.subheader("Configuration")
    # Model select honors prefilled state
    _model_default = st.session_state.get("model", DEFAULT_MODEL)
    _model_index = allowed_models.index(_model_default) if _model_default in allowed_models else (
        allowed_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in allowed_models else 0
    )
    model = st.selectbox("Model", options=allowed_models, index=_model_index, key="model")
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, step=0.1, value=st.session_state.get("temperature", 0.7), key="temperature")
    _qualities = ["preview", "low", "medium", "high"]
    _q_default = st.session_state.get("quality", "medium")
    _q_index = _qualities.index(_q_default) if _q_default in _qualities else _qualities.index("medium")
    quality = st.selectbox("Quality", options=_qualities, index=_q_index, key="quality")
    # Provider API key input
    provider = (model or "").split("/", 1)[0].lower() if model else None
    env_name = PROVIDERS_CFG.get(provider, {}).get("api_key_env") if provider else None
    api_key_help = f"Will use environment {env_name} if provided here; leave blank to use existing environment." if env_name else "Enter API key if required by the selected provider."
    api_key = st.text_input("API Key", type="password", help=api_key_help, key="api_key")
    # Quick status hint
    if env_name:
        if api_key:
            st.caption(f"Using provided key for {provider}.")
        elif os.getenv(env_name):
            st.caption(f"Detected {env_name} in environment.")
        else:
            st.caption(f"No {env_name} set. Provide a key above.")


if page == "Generate":
    # 1. Header Section
    st.markdown("<div class='hero-section'>", unsafe_allow_html=True)
    st.markdown("# Transform Theory into Visual Learning")
    st.markdown("Generate educational math and science videos with AI-powered animations")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 2. Input Section
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("### Create Your Video")
    
    # Responsive input layout
    input_col1, input_col2 = st.columns([3, 1], gap="medium")
    
    with input_col1:
        topic = st.text_input(
            "Video Topic", 
            placeholder="e.g., Binary Search, Eigenvalues, Fourier Transform",
            key="topic",
            label_visibility="collapsed"
        )
        description = st.text_area(
            "Description",
            placeholder="What should the video teach? Include target audience and key concepts to cover...",
            height=120,
            key="description",
            label_visibility="collapsed"
        )
    
    with input_col2:
        # Generate button with loading state
        job_id = st.session_state.get("job_id")
        job, progress, message, status = get_status(job_id)
        
        if status in ("pending", "initializing", "planning", "implementation_planning", "code_generation", "scene_rendering", "video_combining", "finalizing"):
            st.button("🔄 Generating...", disabled=True, use_container_width=True)
        else:
            if st.button("✨ Generate Video", use_container_width=True, type="primary"):
                jid, info = submit_job(topic, description, model, temperature, quality, api_key)
                if jid:
                    st.session_state["job_id"] = jid
                st.session_state["info_msg"] = info
                _rerun()
        
        # Secondary actions
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("📝 Example", use_container_width=True, help="Load example content"):
                st.session_state["prefill"] = {
                    "topic": "Binary Search Algorithm",
                    "description": "Create a visual explanation of binary search for computer science students, showing step-by-step execution and time complexity analysis.",
                    "model": DEFAULT_MODEL,
                    "temperature": 0.6,
                    "quality": "medium",
                }
                st.session_state["prefill_applied"] = False
                _rerun()
        
        with action_cols[1]:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all inputs"):
                st.session_state.pop("prefill", None)
                st.session_state.pop("job_id", None)
                st.session_state["info_msg"] = "Ready to create your video."
                _rerun()
        
        with action_cols[2]:
            if st.button("⏹️ Stop", use_container_width=True, help="Cancel current generation"):
                msg = cancel_job(st.session_state.get("job_id"))
                st.session_state["info_msg"] = msg
                _rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Status/Result Area
    st.markdown("<div class='status-section'>", unsafe_allow_html=True)
    
    if job_id:
        if status == "completed" and job and job.get("output_file") and os.path.exists(job["output_file"]):
            # Success state - prominent video display
            st.success("🎉 Video generation completed successfully!")
            
            # Video player with download
            video_col1, video_col2 = st.columns([4, 1])
            with video_col1:
                st.video(job["output_file"])
            with video_col2:
                st.markdown("### Download")
                try:
                    with open(job["output_file"], "rb") as f:
                        st.download_button(
                            "⬇️ Download MP4",
                            f,
                            file_name=os.path.basename(job["output_file"]),
                            mime="video/mp4",
                            use_container_width=True
                        )
                except Exception:
                    st.error("Download unavailable")
                
                st.markdown("---")
                st.markdown("**Generation Details:**")
                st.text(f"Job ID: {job_id[:8]}...")
                if job.get("start_time"):
                    try:
                        start_dt = datetime.fromisoformat(job["start_time"])
                        st.text(f"Created: {start_dt.strftime('%H:%M:%S')}")
                    except:
                        pass
        
        elif status in ("pending", "initializing", "planning", "implementation_planning", "code_generation", "scene_rendering", "video_combining", "finalizing"):
            # Loading state
            st.info(f"🔄 {message}")
            progress_bar = st.progress(progress / 100 if progress else 0)
            
            # Auto-refresh during generation
            auto_refresh_placeholder = st.empty()
            with auto_refresh_placeholder:
                if st.checkbox("🔄 Auto-refresh", value=True, help="Automatically update progress"):
                    time.sleep(3)
                    _rerun()
        
        elif status == "failed":
            # Error state
            st.error(f"❌ Generation failed: {message}")
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.pop("job_id", None)
                _rerun()
        
        elif status == "cancelled":
            # Cancelled state  
            st.warning("⏹️ Generation was cancelled")
    
    else:
        # Initial state
        info_msg = st.session_state.get("info_msg", "")
        if info_msg and info_msg != "Ready to create your video.":
            if "error" in info_msg.lower() or "failed" in info_msg.lower():
                st.error(info_msg)
            elif "success" in info_msg.lower() or "submitted" in info_msg.lower():
                st.success(info_msg)
            else:
                st.info(info_msg)
    
    # Clear prefill marker after render
    if st.session_state.get("prefill_applied"):
        st.session_state.pop("prefill", None)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Demo Section
    st.markdown("<div class='demo-section'>", unsafe_allow_html=True)
    st.markdown("## Demo Videos")
    st.markdown("Explore examples of AI-generated educational content")
    
    # Demo videos grid (3x2)
    demo_videos = [
        {"file": "demo_1.mp4", "title": "Binary Search Algorithm", "description": "Step-by-step visualization of binary search"},
        {"file": "demo_2.mp4", "title": "Fourier Transform", "description": "Understanding frequency domain transformations"},
        {"file": "demo_3.mp4", "title": "Neural Networks", "description": "How artificial neurons learn patterns"},
        {"file": "demo_4.mp4", "title": "Calculus Derivatives", "description": "Geometric interpretation of derivatives"},
        {"file": "demo_5.mp4", "title": "Quantum States", "description": "Visualization of quantum superposition"},
        {"file": "demo_6.mp4", "title": "Graph Algorithms", "description": "Pathfinding and graph traversal methods"}
    ]
    
    # Create responsive grid
    for i in range(0, len(demo_videos), 3):
        cols = st.columns(3, gap="medium")
        for j, col in enumerate(cols):
            if i + j < len(demo_videos):
                demo = demo_videos[i + j]
                video_path = f"public/{demo['file']}"
                
                with col:
                    if os.path.exists(video_path):
                        with st.container():
                            st.markdown(f"**{demo['title']}**")
                            st.video(video_path)
                            st.caption(demo['description'])
                    else:
                        st.info(f"Demo video {demo['file']} not found")
    
    st.markdown("</div>", unsafe_allow_html=True)


def list_jobs_sorted() -> Dict[str, Dict[str, Any]]:
    data = load_jobs()
    # Sort by start_time desc
    items = list(data.items())
    def key_fn(kv):
        _, j = kv
        ts = j.get("start_time", "")
        return ts
    items.sort(key=key_fn, reverse=True)
    return dict(items)


def rerun_job(job_id: str):
    data = load_jobs()
    if job_id not in data:
        st.warning("Job not found.")
        return
    j = data[job_id]
    params = j.get("params", {})
    jid, info = submit_job(
        j.get("topic", ""),
        j.get("description", ""),
        params.get("model", DEFAULT_MODEL),
        float(params.get("temperature", 0.7)),
        params.get("quality", "medium"),
        st.session_state.get("api_key")
    )
    if jid:
        st.session_state["job_id"] = jid
        st.session_state["page"] = "Generate"
        st.success("Re-queued with same settings.")
        _rerun()


def load_to_generate(job_id: str):
    data = load_jobs()
    if job_id not in data:
        st.warning("Job not found.")
        return
    j = data[job_id]
    params = j.get("params", {})
    st.session_state["prefill"] = {
        "topic": j.get("topic", ""),
        "description": j.get("description", ""),
        "model": params.get("model", DEFAULT_MODEL),
        "temperature": float(params.get("temperature", 0.7)),
        "quality": params.get("quality", "medium"),
    }
    st.session_state["prefill_applied"] = False
    st.session_state["page"] = "Generate"
    _rerun()


if page == "Job History":
    st.markdown("### Job History")
    data = list_jobs_sorted()
    if not data:
        st.info("No jobs yet.")
    else:
        # Summary metrics
        total = len(data)
        completed = sum(1 for j in data.values() if j.get("status") == "completed")
        failed = sum(1 for j in data.values() if j.get("status") == "failed")
        running = sum(1 for j in data.values() if j.get("status") in ["pending", "initializing", "planning", "implementation_planning", "code_generation", "scene_rendering", "video_combining", "finalizing"]) 
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", total)
        m2.metric("Completed", completed)
        m3.metric("Running", running)
        m4.metric("Failed", failed)

        # List
        import pandas as pd
        rows = []
        for jid, j in data.items():
            start = j.get("start_time", "")
            try:
                dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                start_fmt = dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                start_fmt = start
            rows.append({
                "ID": jid[:8] + "...",
                "Job ID": jid,
                "Topic": j.get("topic", "")[:60],
                "Status": j.get("status", "unknown").title(),
                "Progress": j.get("progress", 0),
                "Start Time": start_fmt,
                "Message": j.get("message", "")[:100],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df[["ID", "Topic", "Status", "Progress", "Start Time", "Message"]], width='stretch', hide_index=True)

        # Selection
        ids = [r["Job ID"] for r in rows]
        selected = st.selectbox("Select a job to view details", options=ids, format_func=lambda x: f"{x[:8]}... - {data[x].get('topic','')} ({data[x].get('status','').title()} {data[x].get('progress',0)}%)")

        if selected:
            j = data[selected]
            st.subheader("Details")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write("Topic:", j.get("topic", ""))
                st.write("Description:", j.get("description", ""))
                st.write("Message:", j.get("message", ""))
            with c2:
                st.write("Status:", j.get("status", "").title())
                st.write("Progress:", j.get("progress", 0))
                st.write("Start:", j.get("start_time", ""))
                st.write("End:", j.get("end_time", ""))

            if j.get("status") == "completed" and j.get("output_file") and os.path.exists(j["output_file"]):
                st.video(j["output_file"])
                try:
                    with open(j["output_file"], "rb") as f:
                        st.download_button("Download Video", f, file_name=os.path.basename(j["output_file"]))
                except Exception:
                    pass

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("Re-run Job", width='stretch'):
                    rerun_job(selected)
            with bcol2:
                if st.button("Load to Generate", width='stretch'):
                    load_to_generate(selected)

st.markdown("</div>", unsafe_allow_html=True)
