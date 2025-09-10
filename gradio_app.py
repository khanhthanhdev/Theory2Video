import os
import gradio as gr
import asyncio
import uuid
import threading
import subprocess
import shutil
from datetime import datetime
import logging
import traceback
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from mllm_tools.litellm import LiteLLMWrapper
from src.config.config import Config
from generate_video import EnhancedVideoGenerator, VideoGenerationConfig, allowed_models, default_model
from provider import provider_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gradio_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("thumbnails", exist_ok=True)
os.makedirs("jobs", exist_ok=True)  # Ensure jobs directory exists

# Job persistence and management (SOLID: single responsibility for storage)
class JobStore:
    """Simple persistent store for job metadata."""
    def __init__(self, store_path: str = "jobs/job_history.json") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, obj: Any) -> Any:
        """Recursively remove sensitive keys from nested dicts/lists before persisting."""
        SENSITIVE_KEYS = {
            'api_key', 'openai_api_key', 'gemini_api_key', 'google_api_key',
            'azure_openai_key', 'hf_token', 'huggingface_token',
            'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token'
        }
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.lower() in SENSITIVE_KEYS:
                    # Drop sensitive entries entirely
                    continue
                cleaned[k] = self._sanitize(v)
            return cleaned
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        if not self.store_path.exists():
            return {}
        try:
            with self.store_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Sanitize any secrets found and rewrite file if it changed
                sanitized = self._sanitize(data)
                if sanitized != data:
                    try:
                        self.save_all(sanitized)
                    except Exception:
                        # If save fails, still return sanitized data
                        pass
                return sanitized
            return {}
        except Exception as e:
            logger.error(f"Failed to load job history: {e}")
            return {}

    def save_all(self, jobs: Dict[str, Dict[str, Any]]) -> None:
        try:
            # Ensure the parent directory exists
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.store_path.with_suffix('.tmp')
            # Sanitize before writing to disk
            to_write = self._sanitize(jobs)
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(to_write, f, indent=2)
            tmp_path.replace(self.store_path)
        except Exception as e:
            logger.error(f"Failed to save job history: {e}")

    def upsert(self, job_id: str, job_data: Dict[str, Any], jobs: Dict[str, Dict[str, Any]]) -> None:
        jobs[job_id] = job_data
        self.save_all(jobs)

    def delete(self, job_id: str, jobs: Dict[str, Dict[str, Any]]) -> None:
        if job_id in jobs:
            del jobs[job_id]
            self.save_all(jobs)


class ProgressTracker:
    """Enhanced progress tracking with detailed stages and real-time updates."""
    
    def __init__(self, job_id: str, job_store: JobStore):
        self.job_id = job_id
        self.job_store = job_store
        self.stages = {
            'initializing': {'progress': 5, 'description': 'Initializing video generator'},
            'planning': {'progress': 15, 'description': 'Generating scene outline'},
            'implementation_planning': {'progress': 25, 'description': 'Creating implementation plans'},
            'code_generation': {'progress': 40, 'description': 'Generating Manim code'},
            'scene_rendering': {'progress': 70, 'description': 'Rendering video scenes'},
            'video_combining': {'progress': 90, 'description': 'Combining scene videos'},
            'finalizing': {'progress': 95, 'description': 'Creating thumbnails and finalizing'},
            'completed': {'progress': 100, 'description': 'Video generation completed'}
        }
        self.current_stage = 'initializing'
        self.sub_progress = 0  # Progress within current stage
        
    def update_stage(self, stage: str, sub_progress: int = 0, custom_message: str = None):
        """Update the current stage and progress."""
        if stage in self.stages:
            self.current_stage = stage
            self.sub_progress = max(0, min(100, sub_progress))
            
            # Calculate overall progress
            base_progress = self.stages[stage]['progress']
            if stage != 'completed':
                # Add sub-progress within the stage
                next_stages = list(self.stages.keys())
                current_idx = next_stages.index(stage)
                if current_idx < len(next_stages) - 1:
                    next_stage = next_stages[current_idx + 1]
                    stage_range = self.stages[next_stage]['progress'] - base_progress
                    overall_progress = base_progress + (stage_range * sub_progress / 100)
                else:
                    overall_progress = base_progress
            else:
                overall_progress = 100
            
            # Create message
            message = custom_message or self.stages[stage]['description']
            if sub_progress > 0 and stage != 'completed':
                message = f"{message} ({sub_progress}%)"
            
            # Update job status
            self._update_job_status(overall_progress, message, stage)
            
    def update_scene_progress(self, current_scene: int, total_scenes: int, scene_stage: str = "rendering"):
        """Update progress for scene-based operations."""
        if total_scenes > 0:
            scene_progress = (current_scene - 1) / total_scenes * 100
            message = f"Processing scene {current_scene} of {total_scenes}"
            self.update_stage('scene_rendering', scene_progress, message)
    
    def _update_job_status(self, progress: float, message: str, stage: str):
        """Update the job status in storage."""
        if self.job_id in job_status:
            job_status[self.job_id].update({
                'progress': int(progress),
                'message': message,
                'stage': stage,
                'last_updated': datetime.now().isoformat()
            })
            self.job_store.save_all(job_status)
            logger.info(f"Job {self.job_id} progress: {int(progress)}% - {message}")
    
    def set_error(self, error_message: str, stack_trace: str = None):
        """Set job as failed with error information."""
        job_status[self.job_id].update({
            'status': 'failed',
            'error': error_message,
            'stack_trace': stack_trace,
            'message': f'Error: {error_message[:100]}...' if len(error_message) > 100 else f'Error: {error_message}',
            'last_updated': datetime.now().isoformat()
        })
        self.job_store.save_all(job_status)
    
    def set_completed(self, output_info: Dict[str, Any]):
        """Set job as completed with output information."""
        job_status[self.job_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Video generation completed',
            'stage': 'completed',
            'last_updated': datetime.now().isoformat(),
            **output_info
        })
        self.job_store.save_all(job_status)


job_store = JobStore()
job_status: Dict[str, Dict[str, Any]] = job_store.load_all()

# Default model setting from central registry
DEFAULT_MODEL = default_model if isinstance(default_model, str) else (
    allowed_models[0] if isinstance(allowed_models, list) and allowed_models else "openai/gpt-4o"
)

def cancel_job(job_id):
    """Cancel a running job."""
    if job_id and job_id in job_status:
        if job_status[job_id]['status'] in ['pending', 'initializing', 'planning', 'running']:
            job_status[job_id]['status'] = 'cancelled'
            job_status[job_id]['message'] = 'Job cancelled by user'
            job_store.save_all(job_status)
            return f"Job {job_id} has been cancelled"
    return "Job not found or cannot be cancelled"

def delete_job(job_id):
    """Delete a job from history."""
    if job_id and job_id in job_status:
        # Remove output files if they exist
        job = job_status[job_id]
        if job.get('output_file') and os.path.exists(job['output_file']):
            try:
                # Remove the entire output directory for this job
                output_dir = os.path.dirname(job['output_file'])
                shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error removing output files: {e}")
        
        # Remove thumbnail
        if job.get('thumbnail') and os.path.exists(job['thumbnail']):
            try:
                os.remove(job['thumbnail'])
            except Exception as e:
                logger.error(f"Error removing thumbnail: {e}")
        
        # Remove from job status and persist
        del job_status[job_id]
        job_store.save_all(job_status)
        return f"Job {job_id} deleted successfully"
    return "Job not found"

def get_job_statistics():
    """Get statistics about jobs."""
    total_jobs = len(job_status)
    completed_jobs = sum(1 for job in job_status.values() if job.get('status') == 'completed')
    failed_jobs = sum(1 for job in job_status.values() if job.get('status') == 'failed')
    running_jobs = sum(1 for job in job_status.values() if job.get('status') in ['pending', 'initializing', 'planning', 'running'])
    
    return {
        'total': total_jobs,
        'completed': completed_jobs,
        'failed': failed_jobs,
        'running': running_jobs
    }

def validate_model_availability(model_name: str) -> bool:
    """Validate that the model is available and properly formatted."""
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Check if model is in allowed list
    if model_name not in allowed_models:
        logger.warning(f"Model {model_name} is not in the allowed models list")
        return False
    
    # Basic format validation (should be provider/model)
    if "/" not in model_name:
        logger.warning(f"Model {model_name} should be in format 'provider/model'")
        return False
    
    return True

def init_video_generator(params):
    """Initialize the EnhancedVideoGenerator with the given parameters."""
    model_name = params.get('model', DEFAULT_MODEL)
    # Allow scene/helper models to default to planner model
    scene_model_name = params.get('scene_model') or model_name
    helper_model_name = params.get('helper_model') or model_name
    
    # Validate models
    for model, model_type in [(model_name, 'planner'), (scene_model_name, 'scene'), (helper_model_name, 'helper')]:
        if not validate_model_availability(model):
            raise ValueError(f"Invalid or unavailable {model_type} model: {model}. Available models: {allowed_models}")
    
    verbose = params.get('verbose', True)  # Set verbose to True by default for better debugging
    max_scene_concurrency = params.get('max_scene_concurrency', 1)
    
    # Create configuration for the enhanced video generator
    config = VideoGenerationConfig(
        planner_model=model_name,
        scene_model=scene_model_name,
        helper_model=helper_model_name,
        temperature=float(params.get('temperature', 0.7)),
        output_dir=params.get('output_dir', Config.OUTPUT_DIR),
        verbose=verbose,
        use_rag=params.get('use_rag', False),
        use_context_learning=params.get('use_context_learning', False),
        context_learning_path=params.get('context_learning_path', Config.CONTEXT_LEARNING_PATH),
        chroma_db_path=params.get('chroma_db_path', Config.CHROMA_DB_PATH),
        manim_docs_path=params.get('manim_docs_path', Config.MANIM_DOCS_PATH),
        embedding_model=params.get('embedding_model', Config.EMBEDDING_MODEL),
        use_visual_fix_code=params.get('use_visual_fix_code', True),  # Enable visual fix code by default
        use_langfuse=params.get('use_langfuse', False),
        max_scene_concurrency=max_scene_concurrency,
        max_topic_concurrency=int(params.get('max_topic_concurrency', 1)),
        max_concurrent_renders=int(params.get('max_concurrent_renders', 4)),
        max_retries=params.get('max_retries', 3),
        enable_caching=bool(params.get('enable_caching', True)),
        default_quality=params.get('quality', 'medium'),
        use_gpu_acceleration=bool(params.get('use_gpu_acceleration', False)),
        preview_mode=bool(params.get('preview_mode', False))
    )
    
    # Initialize EnhancedVideoGenerator
    video_generator = EnhancedVideoGenerator(config)
    
    return video_generator

async def process_video_generation(job_id, params):
    """Process video generation asynchronously with detailed progress tracking."""
    progress_tracker = ProgressTracker(job_id, job_store)
    
    try:
        # Initialize job
        job_status[job_id].update({
            'status': 'initializing',
            'progress': 0,
            'message': 'Starting video generation...',
            'params': params,
            'start_time': datetime.now().isoformat()
        })
        progress_tracker.update_stage('initializing', 0, 'Validating configuration...')
        
        # Initialize video generator with better error handling
        try:
            video_generator = init_video_generator(params)
            progress_tracker.update_stage('initializing', 50, 'Video generator initialized successfully')
        except ValueError as e:
            progress_tracker.set_error(f"Configuration error: {str(e)}")
            return
        except Exception as e:
            progress_tracker.set_error(f"Initialization error: {str(e)}")
            return
        
        # Extract parameters
        topic = params.get('topic')
        description = params.get('description')
        only_plan = params.get('only_plan', False)
        specific_scenes = params.get('scenes')
        
        # Parse scenes if provided as a string
        if isinstance(specific_scenes, str):
            s = specific_scenes.strip()
            if s:
                try:
                    specific_scenes = [int(n) for n in re.findall(r"\d+", s)]
                except Exception:
                    specific_scenes = None
            else:
                specific_scenes = None
        
        # Create progress callback for video generator
        def progress_callback(stage: str, progress: int = 0, message: str = None):
            progress_tracker.update_stage(stage, progress, message)
        
        # Log job start
        logger.info(f"Starting job {job_id} for topic: {topic}")
        progress_tracker.update_stage('planning', 0, 'Starting video generation pipeline...')
        
        start_time = datetime.now()
        
        # Run video generation with progress tracking
        try:
            # Add progress callback to video generator if it supports it
            if hasattr(video_generator, 'set_progress_callback'):
                video_generator.set_progress_callback(progress_callback)
            
            await video_generator.generate_video_pipeline(
                topic=topic,
                description=description,
                only_plan=only_plan,
                specific_scenes=specific_scenes
            )
                
            logger.info(f"Video generation pipeline completed for job {job_id}")
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error in video generation pipeline for job {job_id}: {error_msg}")
            logger.error(stack_trace)
            progress_tracker.set_error(error_msg, stack_trace)
            return
        
        # Start finalizing stage
        progress_tracker.update_stage('finalizing', 0, 'Processing output files...')
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Get output file path
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        output_file = os.path.join(
            params.get('output_dir', Config.OUTPUT_DIR),
            file_prefix,
            f"{file_prefix}_combined.mp4"
        )
        
        # Check if output file actually exists
        if not os.path.exists(output_file):
            alternative_output = None
            # Look for any MP4 files that might have been generated
            scene_dir = os.path.join(params.get('output_dir', Config.OUTPUT_DIR), file_prefix)
            if os.path.exists(scene_dir):
                for root, dirs, files in os.walk(scene_dir):
                    for file in files:
                        if file.endswith('.mp4'):
                            alternative_output = os.path.join(root, file)
                            logger.info(f"Combined video not found, but found alternative: {alternative_output}")
                            break
                    if alternative_output:
                        break
            
            if alternative_output:
                output_file = alternative_output
            else:
                logger.error(f"No video output file found for job {job_id}")
                progress_tracker.set_error("No video output was generated. Check Manim execution logs.")
                return
        
        progress_tracker.update_stage('finalizing', 30, 'Creating thumbnail...')
        
        # Create a thumbnail from the video if it exists
        thumbnail_path = None
        if os.path.exists(output_file):
            thumbnail_path = os.path.join("thumbnails", f"{job_id}.jpg")
            try:
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', output_file, 
                    '-ss', '00:00:05', '-frames:v', '1', 
                    thumbnail_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Error creating thumbnail: {result.stderr}")
                    thumbnail_path = None
            except Exception as e:
                logger.error(f"Error creating thumbnail: {str(e)}")
                thumbnail_path = None
        
        progress_tracker.update_stage('finalizing', 70, 'Collecting scene snapshots...')
        
        # Get scene snapshots
        scene_snapshots = []
        scene_dir = os.path.join(params.get('output_dir', Config.OUTPUT_DIR), file_prefix)
        if os.path.exists(scene_dir):
            for i in range(1, 10):  # Check up to 10 possible scenes
                scene_snapshot_dir = os.path.join(scene_dir, f"scene{i}")
                if os.path.exists(scene_snapshot_dir):
                    img_files = [f for f in os.listdir(scene_snapshot_dir) if f.endswith('.png')]
                    if img_files:
                        img_path = os.path.join(scene_snapshot_dir, img_files[-1])  # Get the last image
                        scene_snapshots.append(img_path)
        
        # Complete the job with all output information
        output_info = {
            'output_file': output_file if os.path.exists(output_file) else None,
            'processing_time': processing_time,
            'thumbnail': thumbnail_path,
            'scene_snapshots': scene_snapshots,
            'end_time': end_time.isoformat()
        }
        
        progress_tracker.set_completed(output_info)
        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
        
    except Exception as e:
        # Handle any unexpected exceptions
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in job {job_id}: {error_msg}\n{stack_trace}")
        
        # Use progress tracker if available, otherwise fall back to direct update
        if 'progress_tracker' in locals():
            progress_tracker.set_error(error_msg, stack_trace)
        else:
            job_status[job_id].update({
                'status': 'failed',
                'error': error_msg,
                'stack_trace': stack_trace,
                'message': f'Error: {error_msg[:100]}...' if len(error_msg) > 100 else f'Error: {error_msg}'
            })
            job_store.save_all(job_status)

def start_async_job(job_id, params):
    """Start an async job in a separate thread."""
    def run_async():
        asyncio.run(process_video_generation(job_id, params))
    
    thread = threading.Thread(target=run_async)
    thread.daemon = True
    thread.start()
    return thread

def submit_job(topic, description,
               model, scene_model, helper_model,
               max_retries, use_rag, use_visual_fix_code, temperature, use_context_learning,
               verbose, use_langfuse,
               max_scene_concurrency, max_topic_concurrency, max_concurrent_renders,
               quality, use_gpu_acceleration, preview_mode, enable_caching,
               only_plan, scenes,
               output_dir, chroma_db_path, manim_docs_path, context_learning_path, embedding_model,
               api_key):
    """Submit a new video generation job."""
    # Input validation
    if not topic.strip():
        return "Error: Topic is required", None, gr.update(visible=False)
    
    if not description.strip():
        return "Error: Description is required", None, gr.update(visible=False)
    
    if len(topic.strip()) < 3:
        return "Error: Topic must be at least 3 characters long", None, gr.update(visible=False)
    
    if len(description.strip()) < 10:
        return "Error: Description must be at least 10 characters long", None, gr.update(visible=False)
    
    # Normalize optional scene/helper models
    scene_model = None if (not scene_model or scene_model == "(same as planner)") else scene_model
    helper_model = None if (not helper_model or helper_model == "(same as planner)") else helper_model
    
    # Determine provider from model (format: provider/model)
    provider_prefix = None
    if isinstance(model, str) and "/" in model:
        provider_prefix = model.split("/", 1)[0].lower()
    
    # Validate credentials depending on provider
    if provider_prefix in (None, "openai", "gemini"):
        if not api_key or not api_key.strip():
            return "Error: Please enter your API key for the selected provider", None, gr.update(visible=False)
        # Set appropriate environment variable so the backend uses the key provided
        try:
            key = api_key.strip()
            if provider_prefix in (None, "openai"):
                os.environ["OPENAI_API_KEY"] = key
            elif provider_prefix == "gemini":
                # LiteLLM + wrappers read either GEMINI_API_KEY or GOOGLE_API_KEY
                os.environ["GEMINI_API_KEY"] = key
                os.environ["GOOGLE_API_KEY"] = key
        except Exception as _:
            # Non-fatal; env may already be set
            pass
    elif provider_prefix == "bedrock":
        # For Bedrock, ensure AWS credentials are available in environment
        missing = [
            env for env in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
            if not os.getenv(env)
        ]
        # Region can be provided as AWS_REGION or AWS_DEFAULT_REGION
        if not (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")):
            missing.append("AWS_REGION or AWS_DEFAULT_REGION")
        if missing:
            return (
                f"Error: Missing AWS credentials for Bedrock: {', '.join(missing)}. "
                "Set them in your environment or .env file.",
                None,
                gr.update(visible=False)
            )
    
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_status[job_id] = {
            'id': job_id,
            'status': 'pending',
            'topic': topic,
            'description': description,
            'model': model,
            'start_time': datetime.now().isoformat(),
            'progress': 0,
            'message': 'Job submitted, waiting to start...'
        }
        
        # Prepare parameters with default configuration
        params = {
            'topic': topic,
            'description': description,
            'model': model,
            'scene_model': scene_model,
            'helper_model': helper_model,
            'max_retries': max_retries,
            'use_rag': use_rag,
            'use_visual_fix_code': use_visual_fix_code,
            'temperature': temperature,
            'use_context_learning': use_context_learning,
            'verbose': verbose,
            'use_langfuse': use_langfuse,
            'max_scene_concurrency': max_scene_concurrency,
            'max_topic_concurrency': max_topic_concurrency,
            'max_concurrent_renders': max_concurrent_renders,
            'quality': quality,
            'use_gpu_acceleration': use_gpu_acceleration,
            'preview_mode': preview_mode,
            'enable_caching': enable_caching,
            'only_plan': only_plan,
            'scenes': scenes,
            'output_dir': output_dir or Config.OUTPUT_DIR,
            'chroma_db_path': chroma_db_path or Config.CHROMA_DB_PATH,
            'manim_docs_path': manim_docs_path or Config.MANIM_DOCS_PATH,
            'context_learning_path': context_learning_path or Config.CONTEXT_LEARNING_PATH,
            'embedding_model': embedding_model or Config.EMBEDDING_MODEL,
            # Persist the provider inferred from the model
            'provider': provider_prefix or 'openai'
        }
        
        # Start job asynchronously
        start_async_job(job_id, params)
        
        # Persist immediately after submission
        job_store.save_all(job_status)
        return f"Job submitted successfully. Job ID: {job_id}", job_id, gr.update(visible=True)
    
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        return f"Error: {str(e)}", None, gr.update(visible=False)

def check_job_status(job_id):
    """Check the status of a job."""
    if not job_id or job_id not in job_status:
        return {"status": "not_found", "message": "Job not found"}
    
    return job_status[job_id]

def auto_refresh_status(job_id):
    """Auto-refresh status for active jobs with detailed progress information."""
    if not job_id or job_id not in job_status:
        return None, "Job not found", None, 0, gr.update(visible=False)
    
    job = job_status[job_id]
    status = job.get('status', 'unknown')
    message = job.get('message', 'No status message')
    stage = job.get('stage', 'unknown')
    progress = job.get('progress', 0)
    
    # Format a more detailed status message
    detailed_message = f"Status: {status.title()}"
    if stage and stage != status:
        detailed_message += f" | Stage: {stage.replace('_', ' ').title()}"
    if 'last_updated' in job:
        try:
            last_update = datetime.fromisoformat(job['last_updated'])
            time_ago = (datetime.now() - last_update).total_seconds()
            if time_ago < 60:
                detailed_message += f" | Updated: {int(time_ago)}s ago"
            elif time_ago < 3600:
                detailed_message += f" | Updated: {int(time_ago//60)}m ago"
        except:
            pass
    
    # Show cancel button for active jobs
    show_cancel = status in ['pending', 'initializing', 'planning', 'implementation_planning', 
                           'code_generation', 'scene_rendering', 'video_combining', 'finalizing']
    
    return job.get('output_file'), message, detailed_message, progress, gr.update(visible=show_cancel)

def get_video_details(job_id):
    """Get details of a completed video job."""
    if not job_id or job_id not in job_status:
        return None, None, None, [], "Job not found"
    
    job = job_status[job_id]
    
    if job['status'] != 'completed':
        return None, None, None, [], f"Video not ready. Current status: {job['status']}"
    
    # Get video path, processing time, thumbnail and scene snapshots
    video_path = job.get('output_file')
    processing_time = job.get('processing_time', 0)
    thumbnail = job.get('thumbnail')
    scene_snapshots = job.get('scene_snapshots', [])
    
    if not video_path or not os.path.exists(video_path):
        return None, None, None, [], "Video file not found"
    
    return video_path, processing_time, thumbnail, scene_snapshots, None

def get_job_list():
    """Get a list of all jobs."""
    job_list = []
    for job_id, job in job_status.items():
        job_list.append({
            'id': job_id,
            'topic': job.get('topic', 'Unknown'),
            'status': job.get('status', 'unknown'),
            'start_time': job.get('start_time', ''),
            'progress': job.get('progress', 0),
            'message': job.get('message', '')
        })
    
    # Sort by start time, most recent first
    job_list.sort(key=lambda x: x.get('start_time', ''), reverse=True)
    return job_list

def format_status_message(job):
    """Format status message for display."""
    if not job:
        return "No job selected"
    
    status = job.get('status', 'unknown')
    progress = job.get('progress', 0)
    message = job.get('message', '')
    
    return f"Status: {status.title()} ({progress}%)\n{message}"

def update_status_display(job_id):
    """Update the status display for a job."""
    if not job_id:
        return ("No job selected", 
                gr.update(value=None), 
                gr.update(visible=False), 
                gr.update(visible=False), 
                gr.update(value=[]),
                gr.update(visible=False),
                gr.update(visible=False))
    
    job = check_job_status(job_id)
    status_message = format_status_message(job)
    
    # Check if the job is completed to show the video
    if job.get('status') == 'completed' and job.get('output_file') and os.path.exists(job.get('output_file')):
        video_path = job.get('output_file')
        video_vis = True
        thumbnail = job.get('thumbnail')
        scene_snapshots = job.get('scene_snapshots', [])
        processing_time = job.get('processing_time', 0)
        
        return (status_message, 
                gr.update(value=video_path), 
                gr.update(visible=video_vis), 
                gr.update(visible=thumbnail is not None, value=thumbnail), 
                gr.update(value=scene_snapshots),
                gr.update(visible=True, value=f"Processing Time: {processing_time:.2f} seconds"),
                gr.update(visible=job.get('status') in ['pending', 'initializing', 'planning', 'running']))
    
    return (status_message, 
            gr.update(value=None), 
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(value=[]),
            gr.update(visible=False),
            gr.update(visible=job.get('status') in ['pending', 'initializing', 'planning', 'running']))

# Create Gradio interface
with gr.Blocks(
    title="Theory2Manim Video Generator", 
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ),
    css="""
    .main-header {
        text-align: center;
        background: #f5f7fb;
        color: #111827;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .status-card {
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        background: #f8f9fa;
    }
    .metric-card {
        border: 1px solid #e1e5e9;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        background: white;
    }
    .job-actions {
        gap: 0.5rem;
    }
    /* Center the main generated video and keep it prominent */
    .centered-video-row {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    #main-video {
        max-width: 1100px;
        width: 100%;
        margin: 0 auto;
    }
    #main-video video {
        width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    /* Center any video placed inside a column with this class */
    .center-video-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
    }
    /* Job details video centering */
    #job-video {
        max-width: 900px;
        width: 100%;
        margin: 0 auto;
    }
    #job-video video {
        width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    """
) as app:
    
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="main-header">
                    <h1 style="margin:0;">Theory2Manim Video Generator</h1>
                    <p style="margin:0.5rem 0 0 0; color:#4b5563;">Generate educational videos from structured planning and Manim rendering.</p>
                </div>
            """)
            gr.Markdown(
                "Note: Video generation may take 10–15 minutes per request and can be resource-intensive.",
            )
    
    # Summary
    with gr.Row():
        stats_total = gr.Textbox(label="Total Jobs", interactive=False, scale=1)
        stats_completed = gr.Textbox(label="Completed", interactive=False, scale=1)
        stats_running = gr.Textbox(label="Running", interactive=False, scale=1)
        stats_failed = gr.Textbox(label="Failed", interactive=False, scale=1)
    
    with gr.Tab("Generate Video"):
        # Two-column layout: Left = Settings/Models, Right = Input/Output
        with gr.Row():
            # Left Column: Settings and Models
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Settings")
                    with gr.Row():
                        max_retries_input = gr.Slider(
                            label="Max Retries",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            info="Retry attempts for failures"
                        )
                        quality_dropdown = gr.Dropdown(
                            label="Quality",
                            choices=['preview', 'low', 'medium', 'high', 'production'],
                            value='medium',
                            interactive=True
                        )
                    with gr.Row():
                        max_scene_concurrency_input = gr.Slider(
                            label="Scene Concurrency",
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1
                        )
                        max_topic_concurrency_input = gr.Slider(
                            label="Topic Concurrency",
                            minimum=1,
                            maximum=3,
                            value=1,
                            step=1
                        )
                        max_concurrent_renders_input = gr.Slider(
                            label="Render Workers",
                            minimum=1,
                            maximum=8,
                            value=4,
                            step=1
                        )
                    with gr.Row():
                        use_rag_input = gr.Checkbox(label="Use RAG", value=False)
                        use_visual_fix_code_input = gr.Checkbox(label="Visual Fix", value=True)
                        use_context_learning_input = gr.Checkbox(label="Context Learning", value=False)
                        verbose_input = gr.Checkbox(label="Verbose", value=True)
                        use_langfuse_input = gr.Checkbox(label="Langfuse", value=False)
                    with gr.Row():
                        use_gpu_input = gr.Checkbox(label="GPU", value=False)
                        preview_mode_input = gr.Checkbox(label="Preview Mode", value=False)
                        enable_caching_input = gr.Checkbox(label="Caching", value=True)
                with gr.Group():
                    gr.Markdown("### Models")
                    model_dropdown = gr.Dropdown(
                        label="Planner Model",
                        choices=allowed_models,
                        value=DEFAULT_MODEL,
                        interactive=True
                    )
                    scene_model_dropdown = gr.Dropdown(
                        label="Scene Model (optional)",
                        choices=["(same as planner)"] + allowed_models,
                        value="(same as planner)",
                        interactive=True
                    )
                    helper_model_dropdown = gr.Dropdown(
                        label="Helper Model (optional)",
                        choices=["(same as planner)"] + allowed_models,
                        value="(same as planner)",
                        interactive=True
                    )
                    temperature_input = gr.Slider(
                        label="Creativity (Temperature)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1
                    )
                    api_key_input = gr.Textbox(
                        label="API Key (if required)",
                        placeholder="Enter API key for OpenAI/Gemini (Bedrock uses AWS creds)",
                        type="password",
                        value="",
                        interactive=True
                    )
                with gr.Group():
                    gr.Markdown("### Paths & Data")
                    with gr.Row():
                        output_dir_input = gr.Textbox(label="Output Directory", value=Config.OUTPUT_DIR)
                        chroma_db_path_input = gr.Textbox(label="ChromaDB Path", value=Config.CHROMA_DB_PATH)
                    with gr.Row():
                        manim_docs_path_input = gr.Textbox(label="Manim Docs Path", value=Config.MANIM_DOCS_PATH)
                        context_learning_path_input = gr.Textbox(label="Context Learning Path", value=Config.CONTEXT_LEARNING_PATH)
                    embedding_model_input = gr.Textbox(label="Embedding Model", value=Config.EMBEDDING_MODEL)

            # Right Column: Input and Output Video
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### Input")
                    topic_input = gr.Textbox(
                        label="Topic",
                        placeholder="e.g., Fourier Transform, Calculus Derivatives, Quantum Mechanics"
                    )
                    description_input = gr.Textbox(
                        label="Detailed Description",
                        placeholder="Provide a comprehensive description...",
                        lines=6
                    )
                    with gr.Row():
                        only_plan_input = gr.Checkbox(label="Plan Only (no render)", value=False)
                        scenes_input = gr.Textbox(label="Specific Scenes (e.g., 1,3,5)", placeholder="Leave empty for all scenes")
                    with gr.Row():
                        submit_btn = gr.Button("Generate Video", variant="primary", size="lg")
                        clear_form_btn = gr.Button("Clear", variant="secondary")
                    result_text = gr.Textbox(label="Status", interactive=False)
                    job_id_output = gr.Textbox(label="Job ID", visible=False)

                with gr.Group():
                    gr.Markdown("### Output Video")
                    with gr.Column(visible=False) as status_container:
                        with gr.Group():
                            with gr.Row():
                                with gr.Column(scale=3):
                                    status_text = gr.Textbox(label="Current Status", interactive=False, elem_classes=["status-card"])
                                    processing_time_text = gr.Textbox(label="Processing Information", visible=False, interactive=False)
                                with gr.Column(scale=1):
                                    with gr.Group():
                                        refresh_btn = gr.Button("Refresh Status", variant="secondary")
                                        live_updates_checkbox = gr.Checkbox(label="Auto-refresh (5s)", value=False)
                                        cancel_btn = gr.Button("Cancel Job", variant="stop", visible=False)
                            with gr.Row(elem_classes=["centered-video-row"]):
                                video_output = gr.Video(
                                    label="Generated Video",
                                    interactive=False,
                                    visible=False,
                                    show_download_button=True,
                                    height=480,
                                    elem_id="main-video"
                                )
                            with gr.Row():
                                with gr.Column(scale=1):
                                    thumbnail_preview = gr.Image(label="Video Thumbnail", visible=False, height=200)
                                with gr.Column(scale=2):
                                    scene_gallery = gr.Gallery(
                                        label="Scene Previews",
                                        columns=2,
                                        object_fit="contain",
                                        height=400,
                                        show_download_button=True
                                    )
    
    with gr.Tab("Job History & Management"):
        # Job list table (full width)
        jobs_table = gr.Dataframe(
            headers=["ID", "Topic", "Status", "Progress (%)", "Start Time", "Message"],
            datatype=["str", "str", "str", "number", "str", "str"],
            interactive=False,
            label=None,
            wrap=True,
            elem_classes=["job-history-table"]
        )
        # Action buttons (horizontal row, full width)
        with gr.Row():
            select_job_btn = gr.Button("View Details", variant="primary", size="sm")
            load_to_form_btn = gr.Button("Load to Form", variant="secondary", size="sm")
            rerun_job_btn = gr.Button("Re-run Job", variant="primary", size="sm")
            delete_job_btn = gr.Button("Delete", variant="stop", size="sm")
            download_job_btn = gr.Button("Download", variant="secondary", size="sm")
            refresh_jobs_btn = gr.Button("Refresh List", variant="secondary", size="sm")
            clear_completed_btn = gr.Button("Clear Completed", variant="secondary", size="sm")
            clear_all_btn = gr.Button("Clear All", variant="stop", size="sm")
        selected_job_id = gr.Textbox(label="Selected Job ID", visible=False)
        # Job details viewer (full width, below buttons)
        with gr.Group(elem_classes=["job-details-panel"]):
            gr.Markdown("""
            <div style='font-size:1.2em; font-weight:600; margin-bottom:0.5em;'>
                <span style='color:#3b82f6'>Job Details</span>
            </div>
            """)
            close_details_btn = gr.Button("Back to Job List", variant="secondary", size="sm", visible=False)
            job_details_container = gr.Column(visible=False)
            with job_details_container:
                with gr.Row():
                    with gr.Column(scale=2):
                        job_topic_display = gr.Textbox(label="Topic", interactive=False)
                        job_description_display = gr.Textbox(label="Description", interactive=False, lines=3)
                        job_model_display = gr.Textbox(label="Model Used", interactive=False)
                    with gr.Column(scale=1):
                        job_status_display = gr.Textbox(label="Status", interactive=False)
                        job_progress_display = gr.Number(label="Progress (%)", interactive=False)
                        job_start_time_display = gr.Textbox(label="Start Time", interactive=False)
                with gr.Row():
                    job_processing_time_display = gr.Textbox(label="Processing Time", interactive=False)
                    job_message_display = gr.Textbox(label="Current Message", interactive=False)
                with gr.Column(visible=False, elem_classes=["center-video-col"]) as job_video_container:
                    gr.Markdown("### Generated Video")
                    job_video_player = gr.Video(
                        label="Video Output", 
                        interactive=False,
                        show_download_button=True,
                        height=360,
                        elem_id="job-video"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            job_thumbnail_display = gr.Image(
                                label="Thumbnail", 
                                height=150,
                                interactive=False
                            )
                        with gr.Column(scale=2):
                            job_scene_gallery = gr.Gallery(
                                label="Scene Previews", 
                                columns=3, 
                                object_fit="contain", 
                                height=150,
                                show_download_button=True
                            )
                with gr.Column(visible=False) as job_error_container:
                    gr.Markdown("### Error Details")
                    job_error_display = gr.Textbox(
                        label="Error Message", 
                        interactive=False, 
                        lines=3
                    )
                    job_stack_trace_display = gr.Textbox(
                        label="Stack Trace", 
                        interactive=False, 
                        lines=5,
                        max_lines=10
                    )
            no_job_selected = gr.Markdown(
            """
            <div style='padding:2em 0;text-align:center;color:#888;'>
                <b>No Job Selected</b><br>
                Select a job from the list to view its details.
            </div>
            """,
            visible=True
            )

    with gr.Tab("Help & Documentation"):
        gr.Markdown("""
        ## How to Use Theory2Manim
        
        ### Step 1: Content Planning
        - Topic: Enter a clear, specific topic (e.g., "Linear Algebra: Matrix Multiplication").
        - Description: Provide detailed context about what you want covered:
          - Target audience level (beginner, intermediate, advanced)
          - Specific concepts to include
          - Examples or applications to demonstrate
          - Preferred video length or depth
        
        ### Step 2: Model & API Configuration
        - Enter your API key (if required by your selected provider)
        - Choose the model(s) to use for planning and scene generation
        
        ### Step 3: Advanced Settings
        - Temperature: 0.3–0.5 for factual content; 0.7–0.9 for more creative content
        - Retrieval Augmented Generation (RAG): Enable for topics requiring documentation context
        - Visual Code Fixing: Recommended for better video quality
        - Context Learning: Use previous successful videos as examples
        
        ### Step 4: Monitor Progress
        - Use the Job History tab to monitor all video generation tasks
        - Use Refresh Status to get updates during processing
        - Cancel jobs if needed during processing
        
        ### Step 5: Review Results
        - Preview generated videos directly in the interface
        - View scene breakdowns and thumbnails
        - Download videos for offline use
        
        ## Tips for Best Results
        1. Be specific: detailed descriptions lead to better videos
        2. Start simple: try basic topics first to understand the system
        3. Use examples: mention specific examples you want included
        4. Set context: specify the educational level and background needed
        5. Review settings: adjust temperature and models based on your content type
        
        ## Troubleshooting
        - Job appears stuck: cancel and resubmit with different settings
        - Poor visual quality: increase temperature or enable Visual Code Fixing
        - Missing content: provide more detailed descriptions
        - Errors: enable verbose logging and check the status messages
        """)
    
    # Event handlers with improved functionality
    def clear_form():
        return (
            "", "",  # topic, description
            DEFAULT_MODEL, "(same as planner)", "(same as planner)",
            "",  # api key
            0.7,  # temperature
            3, 1, 1, 4,  # retries, scene conc, topic conc, render workers
            False, True, False, True, False,  # use_rag, visual_fix, context_learning, verbose, langfuse
            'medium', False, False, True,  # quality, gpu, preview, caching
            False, "",  # only_plan, scenes
            Config.OUTPUT_DIR, Config.CHROMA_DB_PATH, Config.MANIM_DOCS_PATH, Config.CONTEXT_LEARNING_PATH, Config.EMBEDDING_MODEL,
            "Form cleared."
        )
    
    def update_stats():
        stats = get_job_statistics()
        return (f"{stats['total']}", 
                f"{stats['completed']}", 
                f"{stats['running']}", 
                f"{stats['failed']}")
    
    def clear_completed_jobs():
        completed_jobs = [job_id for job_id, job in job_status.items() 
                         if job.get('status') == 'completed']
        for job_id in completed_jobs:
            delete_job(job_id)
        return f"Cleared {len(completed_jobs)} completed jobs"
    
    def clear_all_jobs():
        count = len(job_status)
        job_status.clear()
        return f"Cleared all {count} jobs"
    
    # Connect simplified event handlers - no model selection needed
    
    # Auto-refresh functionality with conditional updating
    auto_refresh_timer = gr.Timer(5.0, active=False)  # 5-second timer, inactive by default
    
    clear_form_btn.click(
        fn=clear_form,
        outputs=[
            topic_input, description_input,
            model_dropdown, scene_model_dropdown, helper_model_dropdown,
            api_key_input, temperature_input,
            max_retries_input, max_scene_concurrency_input, max_topic_concurrency_input, max_concurrent_renders_input,
            use_rag_input, use_visual_fix_code_input, use_context_learning_input, verbose_input, use_langfuse_input,
            quality_dropdown, use_gpu_input, preview_mode_input, enable_caching_input,
            only_plan_input, scenes_input,
            output_dir_input, chroma_db_path_input, manim_docs_path_input, context_learning_path_input, embedding_model_input,
            result_text
        ]
    )
    
    submit_btn.click(
        fn=submit_job,
        inputs=[
            topic_input, description_input,
            model_dropdown, scene_model_dropdown, helper_model_dropdown,
            max_retries_input,
            use_rag_input, use_visual_fix_code_input, temperature_input, use_context_learning_input,
            verbose_input, use_langfuse_input,
            max_scene_concurrency_input, max_topic_concurrency_input, max_concurrent_renders_input,
            quality_dropdown, use_gpu_input, preview_mode_input, enable_caching_input,
            only_plan_input, scenes_input,
            output_dir_input, chroma_db_path_input, manim_docs_path_input, context_learning_path_input, embedding_model_input,
            api_key_input
        ],
        outputs=[result_text, job_id_output, status_container]
    ).then(
        fn=lambda job_id: (True, gr.Timer(active=True)) if job_id else (False, gr.Timer(active=False)),  # Enable auto-refresh for new jobs
        inputs=[job_id_output],
        outputs=[live_updates_checkbox, auto_refresh_timer]
    ).then(
        fn=auto_refresh_status,
        inputs=[job_id_output],
        outputs=[video_output, status_text, processing_time_text, gr.Number(visible=False), cancel_btn]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    refresh_btn.click(
        fn=auto_refresh_status,
        inputs=[job_id_output],
        outputs=[video_output, status_text, processing_time_text, gr.Number(visible=False), cancel_btn]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    # Toggle auto-refresh timer based on checkbox
    live_updates_checkbox.change(
        fn=lambda enabled: gr.Timer(active=enabled),
        inputs=[live_updates_checkbox],
        outputs=[auto_refresh_timer]
    )
    
    # Auto-refresh status when timer ticks (only if job is active)
    def conditional_auto_refresh(job_id):
        if not job_id or job_id not in job_status:
            return gr.Timer(active=False)  # Stop timer if no job
        
        job = job_status[job_id]
        status = job.get('status', '')
        
        # Keep refreshing for active jobs
        is_active = status in ['pending', 'initializing', 'planning', 'implementation_planning', 
                              'code_generation', 'scene_rendering', 'video_combining', 'finalizing']
        
        # Return auto-refresh data and timer status
        video_file, message, detailed_message, progress, cancel_visible = auto_refresh_status(job_id)
        
        return (video_file, message, detailed_message, progress, cancel_visible, 
                gr.Timer(active=is_active))
    
    auto_refresh_timer.tick(
        fn=conditional_auto_refresh,
        inputs=[job_id_output],
        outputs=[video_output, status_text, processing_time_text, gr.Number(visible=False), 
                cancel_btn, auto_refresh_timer]
    )
    
    cancel_btn.click(
        fn=cancel_job,
        inputs=[job_id_output],
        outputs=[result_text]
    ).then(
        fn=update_status_display,
        inputs=[job_id_output],
        outputs=[status_text, video_output, video_output, thumbnail_preview, scene_gallery, processing_time_text, cancel_btn]
    )
    
    # Job history tab functions

    def load_job_list():
        jobs = get_job_list()
        rows = []
        for job in jobs:
            start_time = job.get('start_time', '')
            if start_time:
                try:
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = start_time
            else:
                formatted_time = 'Unknown'
            rows.append([
                job['id'][:8] + '...',
                job['topic'][:50] + ('...' if len(job['topic']) > 50 else ''),
                job['status'].title(),
                job['progress'],
                formatted_time,
                job['message'][:100] + ('...' if len(job['message']) > 100 else '')
            ])
        return rows

    def select_job(evt: gr.EventData):
        if not evt or not hasattr(evt, 'index') or not evt.index:
            # No job selected
            return "", "No job selected", gr.update(visible=False)
        selected_row = evt.index[0]
        jobs = get_job_list()
        if selected_row < len(jobs):
            # Job selected
            return jobs[selected_row]['id'], f"Selected job: {jobs[selected_row]['topic']}", gr.update(visible=True)
        return "", "No job selected", gr.update(visible=False)

    def load_job_to_form(job_id: str):
        """Load a previous job's parameters back into the Generate form."""
        if not job_id or job_id not in job_status:
            return (
                "", "",
                DEFAULT_MODEL, "(same as planner)", "(same as planner)",
                "",
                0.7,
                3, 1, 1, 4,
                False, True, False, True, False,
                'medium', False, False, True,
                False, "",
                Config.OUTPUT_DIR, Config.CHROMA_DB_PATH, Config.MANIM_DOCS_PATH, Config.CONTEXT_LEARNING_PATH, Config.EMBEDDING_MODEL
            )
        params = job_status[job_id].get('params', {})
        def getp(key, default=None):
            return params.get(key, default)
        model = getp('model', DEFAULT_MODEL)
        scene_model = getp('scene_model') or "(same as planner)"
        helper_model = getp('helper_model') or "(same as planner)"
        scenes_val = getp('scenes', "")
        # normalize scenes list to csv string
        if isinstance(scenes_val, list):
            scenes_val = ",".join(str(s) for s in scenes_val)
        return (
            job_status[job_id].get('topic', ''), job_status[job_id].get('description', ''),
            model, scene_model, helper_model,
            "",  # api key intentionally blank
            float(getp('temperature', 0.7)),
            int(getp('max_retries', 3)), int(getp('max_scene_concurrency', 1)), int(getp('max_topic_concurrency', 1)), int(getp('max_concurrent_renders', 4)),
            bool(getp('use_rag', False)), bool(getp('use_visual_fix_code', True)), bool(getp('use_context_learning', False)), bool(getp('verbose', True)), bool(getp('use_langfuse', False)),
            getp('quality', 'medium'), bool(getp('use_gpu_acceleration', False)), bool(getp('preview_mode', False)), bool(getp('enable_caching', True)),
            bool(getp('only_plan', False)), scenes_val,
            getp('output_dir', Config.OUTPUT_DIR), getp('chroma_db_path', Config.CHROMA_DB_PATH), getp('manim_docs_path', Config.MANIM_DOCS_PATH), getp('context_learning_path', Config.CONTEXT_LEARNING_PATH), getp('embedding_model', Config.EMBEDDING_MODEL)
        )

    def requeue_job(job_id: str):
        """Re-run a previous job with the same parameters."""
        if not job_id or job_id not in job_status:
            return "Error: Job not found", None, gr.update(visible=False)
        params = job_status[job_id].get('params')
        if not params:
            return "Error: No parameters found for this job", None, gr.update(visible=False)
        try:
            new_job_id = str(uuid.uuid4())
            job_status[new_job_id] = {
                'id': new_job_id,
                'status': 'pending',
                'topic': job_status[job_id].get('topic', ''),
                'description': job_status[job_id].get('description', ''),
                'model': params.get('model', DEFAULT_MODEL),
                'start_time': datetime.now().isoformat(),
                'progress': 0,
                'message': 'Job submitted, waiting to start...'
            }
            job_store.save_all(job_status)
            start_async_job(new_job_id, params)
            return f"Job re-submitted. New Job ID: {new_job_id}", new_job_id, gr.update(visible=True)
        except Exception as e:
            logger.error(f"Error re-queuing job {job_id}: {e}")
            return f"Error: {str(e)}", None, gr.update(visible=False)
    
    def back_to_job_list():
        # Show job list, hide details
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def view_job_details(job_id):
        """View details of a selected job."""
        if not job_id or job_id not in job_status:
            # Return 17 outputs, all hidden or empty
            return (
                gr.update(visible=False),  # job_details_container
                gr.update(visible=True),   # no_job_selected
                "", "", "", "", 0, "", "", "",  # topic, desc, model, status, progress, start, proc_time, msg
                gr.update(visible=False),  # job_video_container
                gr.update(visible=False, value=None),  # job_video_player
                gr.update(visible=False, value=None),  # job_thumbnail_display
                gr.update(visible=False, value=[]),    # job_scene_gallery
                gr.update(visible=False),  # job_error_container
                gr.update(visible=False, value=""),   # job_error_display
                gr.update(visible=False, value="")    # job_stack_trace_display
            )
        job = job_status[job_id]
        # Format start time
        start_time = job.get('start_time', '')
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = start_time
        else:
            formatted_time = 'Unknown'
        
        # Video and error visibility
        is_completed = job.get('status') == 'completed'
        is_failed = job.get('status') == 'failed'
        # Always return 17 outputs in order
        return (
            gr.update(visible=True),  # job_details_container
            gr.update(visible=False),  # no_job_selected
            job.get('topic', ''),
            job.get('description', ''),
            job.get('model', ''),
            gr.update(value=job.get('status', '').title()),  # status_display
            job.get('progress', 0),
            formatted_time,
            job.get('processing_time', ''),
            job.get('message', ''),
            gr.update(visible=is_completed),  # job_video_container
            gr.update(visible=is_completed, value=job.get('output_file') if is_completed else None),  # job_video_player
            gr.update(visible=is_completed and job.get('thumbnail') is not None, value=job.get('thumbnail') if is_completed else None),  # job_thumbnail_display
            gr.update(visible=is_completed, value=job.get('scene_snapshots', []) if is_completed else []),  # job_scene_gallery
            gr.update(visible=is_failed),  # job_error_container
            gr.update(visible=is_failed, value=job.get('error', '') if is_failed else ""),  # job_error_display
            gr.update(visible=is_failed, value=job.get('stack_trace', '') if is_failed else "")  # job_stack_trace_display
        )

    def delete_selected_job(job_id):
        """Delete the selected job and update the UI."""
        if not job_id or job_id not in job_status:
            return "Job not found", None, gr.update(visible=False)
        
        # Delete the job
        result = delete_job(job_id)
        
        # Update job list
        jobs = get_job_list()
        
        # Refresh job table
        return result, gr.update(value=load_job_list()), gr.update(visible=False)

    def download_job_results(job_id):
        """Download the results of a job."""
        if not job_id or job_id not in job_status:
            return "Job not found", None
        
        job = job_status[job_id]
        output_file = job.get('output_file')
        
        if not output_file or not os.path.exists(output_file):
            return "Output file not found", None
        
        return "Download started", output_file
    
    # Connect job history tab event handlers
    refresh_jobs_btn.click(
        fn=load_job_list,
        outputs=[jobs_table]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    jobs_table.select(
        fn=select_job,
        outputs=[selected_job_id, result_text, close_details_btn]
    )
    
    select_job_btn.click(
        fn=view_job_details,
        inputs=[selected_job_id],
        outputs=[
            job_details_container, no_job_selected,
            job_topic_display, job_description_display, job_model_display,
            job_status_display, job_progress_display, job_start_time_display,
            job_processing_time_display, job_message_display,
            job_video_container, job_video_player, job_thumbnail_display, job_scene_gallery,
            job_error_container, job_error_display, job_stack_trace_display
        ]
    )
    
    close_details_btn.click(
        fn=back_to_job_list,
        outputs=[job_details_container, no_job_selected, close_details_btn]
    )
    
    load_to_form_btn.click(
        fn=load_job_to_form,
        inputs=[selected_job_id],
        outputs=[
            topic_input, description_input,
            model_dropdown, scene_model_dropdown, helper_model_dropdown,
            api_key_input, temperature_input,
            max_retries_input, max_scene_concurrency_input, max_topic_concurrency_input, max_concurrent_renders_input,
            use_rag_input, use_visual_fix_code_input, use_context_learning_input, verbose_input, use_langfuse_input,
            quality_dropdown, use_gpu_input, preview_mode_input, enable_caching_input,
            only_plan_input, scenes_input,
            output_dir_input, chroma_db_path_input, manim_docs_path_input, context_learning_path_input, embedding_model_input,
        ]
    )

    rerun_job_btn.click(
        fn=requeue_job,
        inputs=[selected_job_id],
        outputs=[result_text, job_id_output, status_container]
    ).then(
        fn=update_status_display,
        inputs=[job_id_output],
        outputs=[status_text, video_output, video_output, thumbnail_preview, scene_gallery, processing_time_text, cancel_btn]
    )

    download_job_btn.click(
        fn=download_job_results,
        inputs=[selected_job_id],
        outputs=[result_text]
    )
    
    delete_job_btn.click(
        fn=delete_selected_job,
        inputs=[selected_job_id],
        outputs=[result_text, selected_job_id]
    ).then(
        fn=load_job_list,
        outputs=[jobs_table]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    clear_completed_btn.click(
        fn=clear_completed_jobs,
        outputs=[result_text]
    ).then(
        fn=load_job_list,
        outputs=[jobs_table]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    clear_all_btn.click(
        fn=clear_all_jobs,
        outputs=[result_text]
    ).then(
        fn=load_job_list,
        outputs=[jobs_table]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    # Set up polling for status updates
    app.load(
        fn=load_job_list,
        outputs=[jobs_table]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    # Load on app start
    def on_app_start():
        if not os.path.exists("thumbnails"):
            os.makedirs("thumbnails", exist_ok=True)
        return "Welcome to Theory2Manim Video Generator."
    
    app.load(
        fn=on_app_start,
        outputs=[result_text]
    )




if __name__ == "__main__":
    import os
    app.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False
    )
