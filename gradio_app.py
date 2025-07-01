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
from typing import Dict, List, Optional

from mllm_tools.litellm import LiteLLMWrapper
from src.config.config import Config
from generate_video import EnhancedVideoGenerator, VideoGenerationConfig, allowed_models
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

# Global dictionary to track job status
job_status = {}

# Default model setting - simplified to use GPT-4o-mini for all operations
DEFAULT_MODEL = "openai/gpt-4o-mini"

def cancel_job(job_id):
    """Cancel a running job."""
    if job_id and job_id in job_status:
        if job_status[job_id]['status'] in ['pending', 'initializing', 'planning', 'running']:
            job_status[job_id]['status'] = 'cancelled'
            job_status[job_id]['message'] = 'Job cancelled by user'
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
        
        # Remove from job status
        del job_status[job_id]
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

def init_video_generator(params):
    """Initialize the EnhancedVideoGenerator with the given parameters."""
    model_name = params.get('model', DEFAULT_MODEL)
    helper_model_name = params.get('helper_model', DEFAULT_MODEL)
    verbose = params.get('verbose', True)  # Set verbose to True by default for better debugging
    max_scene_concurrency = params.get('max_scene_concurrency', 1)
    
    # Create configuration for the enhanced video generator
    config = VideoGenerationConfig(
        planner_model=model_name,
        scene_model=model_name,
        helper_model=helper_model_name,
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
        max_retries=params.get('max_retries', 3)
    )
    
    # Initialize EnhancedVideoGenerator
    video_generator = EnhancedVideoGenerator(config)
    
    return video_generator

async def process_video_generation(job_id, params):
    """Process video generation asynchronously."""
    try:
        # Update job status
        job_status[job_id]['status'] = 'initializing'
        job_status[job_id]['progress'] = 5
        job_status[job_id]['message'] = 'Initializing video generator...'
        
        # Initialize video generator
        video_generator = init_video_generator(params)
        
        # Extract video generation parameters
        topic = params.get('topic')
        description = params.get('description')
        max_retries = int(params.get('max_retries', 3))
        only_plan = params.get('only_plan', False)
        
        # Log job start
        logger.info(f"Starting job {job_id} for topic: {topic}")
        job_status[job_id]['status'] = 'planning'
        job_status[job_id]['progress'] = 10
        job_status[job_id]['message'] = 'Planning video scenes...'
        
        # Generate video pipeline
        start_time = datetime.now()
        logger.info(f"Running generate_video_pipeline for topic: {topic}")
        
        # Create an event loop for the async process
        def update_progress_callback(progress, message):
            job_status[job_id]['progress'] = progress
            job_status[job_id]['message'] = message
            logger.info(f"Job {job_id} progress: {progress}% - {message}")
        
        # Start a background task to periodically update progress
        async def progress_update_task():
            stages = [
                (15, 'Creating scene outline...'),
                (25, 'Generating implementation plans...'),
                (35, 'Generating code for scenes...'),
                (45, 'Compiling Manim code...'),
                (60, 'Rendering scenes...'),
                (80, 'Combining videos...'),
                (90, 'Finalizing video...')
            ]
            
            for progress, message in stages:
                update_progress_callback(progress, message)
                await asyncio.sleep(5)  # Wait between updates
                
                # Stop updating if job is complete or failed
                if job_status[job_id]['status'] in ['completed', 'failed']:
                    break
        
        # Start progress update task
        progress_task = asyncio.create_task(progress_update_task())
        
        # Run the main video generation task with detailed logging
        try:
            logger.info(f"Starting video generation pipeline for job {job_id}")
            update_progress_callback(15, 'Starting video generation pipeline...')
            
            await video_generator.generate_video_pipeline(
                topic=topic,
                description=description,
                only_plan=only_plan
            )
                
            logger.info(f"Video generation pipeline completed for job {job_id}")
        except Exception as e:
            logger.error(f"Error in video generation pipeline for job {job_id}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Cancel progress update task
        if not progress_task.done():
            progress_task.cancel()
        
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
                raise Exception("No video output was generated. Check Manim execution logs.")
        
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
        
        # Update job status to completed
        job_status[job_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Video generation completed',
            'output_file': output_file if os.path.exists(output_file) else None,
            'processing_time': processing_time,
            'thumbnail': thumbnail_path,
            'scene_snapshots': scene_snapshots
        })
        
        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
        
    except Exception as e:
        # Handle exceptions
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error in job {job_id}: {error_msg}\n{stack_trace}")
        
        job_status[job_id].update({
            'status': 'failed',
            'error': error_msg,
            'stack_trace': stack_trace,
            'message': f'Error: {error_msg[:100]}...' if len(error_msg) > 100 else f'Error: {error_msg}'
        })

def start_async_job(job_id, params):
    """Start an async job in a separate thread."""
    def run_async():
        asyncio.run(process_video_generation(job_id, params))
    
    thread = threading.Thread(target=run_async)
    thread.daemon = True
    thread.start()
    return thread

def submit_job(topic, description, max_retries, use_rag, use_visual_fix_code, temperature, use_context_learning, verbose, max_scene_concurrency, api_key):
    """Submit a new video generation job."""
    # Input validation
    if not topic.strip():
        return "❌ Error: Topic is required", None, gr.update(visible=False)
    
    if not description.strip():
        return "❌ Error: Description is required", None, gr.update(visible=False)
    
    if len(topic.strip()) < 3:
        return "❌ Error: Topic must be at least 3 characters long", None, gr.update(visible=False)
    
    if len(description.strip()) < 10:
        return "❌ Error: Description must be at least 10 characters long", None, gr.update(visible=False)
    
    # Set default model
    model = DEFAULT_MODEL
    helper_model = DEFAULT_MODEL
    
    # Validate API key
    if not api_key or not api_key.strip():
        return "❌ Error: Please enter your OpenAI API key", None, gr.update(visible=False)
    
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
            'helper_model': helper_model,
            'max_retries': max_retries,
            'use_rag': use_rag,
            'use_visual_fix_code': use_visual_fix_code,
            'temperature': temperature,
            'use_context_learning': use_context_learning,
            'verbose': verbose,
            'max_scene_concurrency': max_scene_concurrency,
            'output_dir': Config.OUTPUT_DIR,
            # Use OpenAI as default provider
            'provider': 'openai',
            'api_key': api_key.strip()
        }
        
        # Start job asynchronously
        start_async_job(job_id, params)
        
        return f"✅ Job submitted successfully. Job ID: {job_id}", job_id, gr.update(visible=True)
    
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        return f"❌ Error: {str(e)}", None, gr.update(visible=False)

def check_job_status(job_id):
    """Check the status of a job."""
    if not job_id or job_id not in job_status:
        return {"status": "not_found", "message": "Job not found"}
    
    return job_status[job_id]

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
    
    status_emoji = {
        'pending': '⏳',
        'initializing': '🔄',
        'planning': '🧠',
        'running': '⚙️',
        'completed': '✅',
        'failed': '❌',
        'unknown': '❓'
    }.get(status, '❓')
    
    return f"{status_emoji} Status: {status.title()} ({progress}%)\n{message}"

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
                gr.update(visible=True, value=f"⏱️ Processing Time: {processing_time:.2f} seconds"),
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
    title="Theory2Manim 3blue1brown Video Style Generator", 
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ),
    css="""
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
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
    """
) as app:
    
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="main-header">
                    <h1>🎬 Theory2Manim 3blue1brown Video Style Generator</h1>
                    <p>Transform mathematical and scientific concepts into engaging educational videos</p>
                </div>
            """)
            gr.Markdown(
                "⚠️ **Note:** Video generation typically takes **10–15 minutes** per request. "
                "Each video may consume **700,000 to 1,000,000 tokens**. Please plan accordingly.",
                
            )
    
    # Statistics Dashboard
    with gr.Row():
        stats_total = gr.Textbox(label="📊 Total Jobs", interactive=False, scale=1)
        stats_completed = gr.Textbox(label="✅ Completed", interactive=False, scale=1)
        stats_running = gr.Textbox(label="⚙️ Running", interactive=False, scale=1)
        stats_failed = gr.Textbox(label="❌ Failed", interactive=False, scale=1)
    
    with gr.Tab("🎥 Generate Video"):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 📝 Content Configuration")
                    topic_input = gr.Textbox(
                        label="📚 Topic", 
                        placeholder="e.g., Fourier Transform, Calculus Derivatives, Quantum Mechanics",
                        info="Enter the main topic for your educational video"
                    )
                    description_input = gr.Textbox(
                        label="📋 Detailed Description", 
                        placeholder="Provide a comprehensive description of what you want the video to cover, including specific concepts, examples, and target audience level...",
                        lines=6,
                        info="The more detailed your description, the better the AI can generate relevant content"
                    )
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🔑 API Configuration")
                    
                    # Simple API key input
                    api_key_input = gr.Textbox(
                        label="🔑 OpenAI API Key",
                        placeholder="Enter your OpenAI API key",
                        type="password",
                        value="",
                        interactive=True,
                        info="Your OpenAI API key for GPT-4o-mini access"
                    )
                    
                    # Display current model setting
                    gr.Markdown(f"**🤖 AI Model:** Using {DEFAULT_MODEL} for all operations")
                    
                    # Temperature control
                    temperature_input = gr.Slider(
                        label="🌡️ Creativity (Temperature)", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1,
                        info="Lower = more focused, Higher = more creative"
                    )
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 🔧 Advanced Settings")
                    with gr.Row():
                        max_retries_input = gr.Slider(
                            label="🔄 Max Retries", 
                            minimum=1, 
                            maximum=10, 
                            value=3, 
                            step=1,
                            info="Number of retry attempts for failed operations"
                        )
                        max_scene_concurrency_input = gr.Slider(
                            label="⚡ Scene Concurrency", 
                            minimum=1, 
                            maximum=5, 
                            value=1, 
                            step=1,
                            info="Number of scenes to process simultaneously"
                        )
                    
                    with gr.Row():
                        use_rag_input = gr.Checkbox(
                            label="📚 Use RAG (Retrieval Augmented Generation)", 
                            value=False,
                            info="Enhance generation with relevant knowledge retrieval"
                        )
                        use_visual_fix_code_input = gr.Checkbox(
                            label="🎨 Use Visual Code Fixing", 
                            value=True,
                            info="Automatically fix visual rendering issues"
                        )
                        use_context_learning_input = gr.Checkbox(
                            label="🧠 Use Context Learning", 
                            value=False,
                            info="Learn from previous successful videos"
                        )
                        verbose_input = gr.Checkbox(
                            label="📝 Verbose Logging", 
                            value=True,
                            info="Enable detailed logging for debugging"
                        )
        
        with gr.Row():
            with gr.Column(scale=3):
                submit_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
            with gr.Column(scale=1):
                clear_form_btn = gr.Button("🧹 Clear Form", variant="secondary")
        
        result_text = gr.Textbox(label="📋 Status", interactive=False)
        job_id_output = gr.Textbox(label="Job ID", visible=False)
        
        with gr.Column(visible=False) as status_container:
            with gr.Group():
                gr.Markdown("### 📊 Job Progress")
                with gr.Row():
                    with gr.Column(scale=3):
                        status_text = gr.Textbox(label="Current Status", interactive=False, elem_classes=["status-card"])
                        processing_time_text = gr.Textbox(label="Processing Information", visible=False, interactive=False)
                    with gr.Column(scale=1):
                        with gr.Group():
                            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                            cancel_btn = gr.Button("⏹️ Cancel Job", variant="stop", visible=False)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        video_output = gr.Video(
                            label="🎬 Generated Video", 
                            interactive=False, 
                            visible=False,
                            show_download_button=True
                        )
                        thumbnail_preview = gr.Image(
                            label="🖼️ Video Thumbnail", 
                            visible=False,
                            height=200
                        )
                    
                    with gr.Column(scale=1):
                        scene_gallery = gr.Gallery(
                            label="🎨 Scene Previews", 
                            columns=2, 
                            object_fit="contain", 
                            height=400,
                            show_download_button=True
                        )
    
    with gr.Tab("📂 Job History & Management"):
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
            select_job_btn = gr.Button("👁️ View Details", variant="primary", size="sm")
            delete_job_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
            download_job_btn = gr.Button("💾 Download", variant="secondary", size="sm")
            refresh_jobs_btn = gr.Button("🔄 Refresh List", variant="secondary", size="sm")
            clear_completed_btn = gr.Button("🧹 Clear Completed", variant="secondary", size="sm")
            clear_all_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
        selected_job_id = gr.Textbox(label="Selected Job ID", visible=False)
        # Job details viewer (full width, below buttons)
        with gr.Group(elem_classes=["job-details-panel"]):
            gr.Markdown("""
            <div style='font-size:1.2em; font-weight:600; margin-bottom:0.5em;'>
                👁️ <span style='color:#3b82f6'>Job Details Viewer</span>
            </div>
            """)
            close_details_btn = gr.Button("⬅️ Back to Job List", variant="secondary", size="sm", visible=False)
            job_details_container = gr.Column(visible=False)
            with job_details_container:
                with gr.Row():
                    with gr.Column(scale=2):
                        job_topic_display = gr.Textbox(label="📚 Topic", interactive=False)
                        job_description_display = gr.Textbox(label="📝 Description", interactive=False, lines=3)
                        job_model_display = gr.Textbox(label="🤖 Model Used", interactive=False)
                    with gr.Column(scale=1):
                        job_status_display = gr.Textbox(label="📊 Status", interactive=False)
                        job_progress_display = gr.Number(label="📈 Progress (%)", interactive=False)
                        job_start_time_display = gr.Textbox(label="⏰ Start Time", interactive=False)
                with gr.Row():
                    job_processing_time_display = gr.Textbox(label="⏱️ Processing Time", interactive=False)
                    job_message_display = gr.Textbox(label="💬 Current Message", interactive=False)
                with gr.Column(visible=False) as job_video_container:
                    gr.Markdown("### 🎬 Generated Video")
                    job_video_player = gr.Video(
                        label="Video Output", 
                        interactive=False,
                        show_download_button=True,
                        height=300
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            job_thumbnail_display = gr.Image(
                                label="🖼️ Thumbnail", 
                                height=150,
                                interactive=False
                            )
                        with gr.Column(scale=2):
                            job_scene_gallery = gr.Gallery(
                                label="🎨 Scene Previews", 
                                columns=3, 
                                object_fit="contain", 
                                height=150,
                                show_download_button=True
                            )
                with gr.Column(visible=False) as job_error_container:
                    gr.Markdown("### ❌ Error Details")
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
                <b>📄 No Job Selected</b><br>
                Select a job from the list to view its details.
            </div>
            """,
            visible=True
            )

    with gr.Tab("ℹ️ Help & Documentation"):
        gr.Markdown("""
        ## 🎯 How to Use Theory2Manim
        
        ### 📝 Step 1: Content Planning
        - **Topic**: Enter a clear, specific topic (e.g., "Linear Algebra: Matrix Multiplication")
        - **Description**: Provide detailed context about what you want covered:
          - Target audience level (beginner, intermediate, advanced)
          - Specific concepts to include
          - Examples or applications to demonstrate
          - Preferred video length or depth
        
        ### 🔑 Step 2: API Configuration
        - **OpenAI API Key**: Enter your OpenAI API key for access
        - **AI Model**: Automatically uses GPT-4o-mini for optimal cost and performance
        
        ### ⚙️ Step 3: Advanced Settings
        - **Temperature**: 0.3-0.5 for factual content, 0.7-0.9 for creative explanations
        - **RAG**: Enable for topics requiring external knowledge
        - **Visual Code Fixing**: Recommended for better video quality
        - **Context Learning**: Use previous successful videos as examples
        
        ### 📊 Step 4: Monitor Progress
        - Check the **Job History** tab to monitor all your video generation tasks
        - Use **Refresh Status** to get real-time updates
        - **Cancel** jobs if needed during processing
        
        ### 🎬 Step 5: Review Results
        - Preview generated videos directly in the interface
        - View scene breakdowns and thumbnails
        - Download videos for offline use
        
        ## 💡 Tips for Best Results
        1. **Be Specific**: Detailed descriptions lead to better videos
        2. **Start Simple**: Try basic topics first to understand the system
        3. **Use Examples**: Mention specific examples you want included
        4. **Set Context**: Specify the educational level and background needed
        5. **Review Settings**: Adjust temperature and models based on your content type
        
        ## 🔧 Troubleshooting
        - **Job Stuck**: Try canceling and resubmitting with different settings
        - **Poor Quality**: Use higher temperature or enable Visual Code Fixing
        - **Missing Content**: Provide more detailed descriptions
        - **Errors**: Check the verbose logs in the status messages
        """)
    
    # Event handlers with improved functionality
    def clear_form():
        return ("", "", 0.7, False, True, False, True, 1, 1, "", "Form cleared! Ready for new input.")
    
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
    
    clear_form_btn.click(
        fn=clear_form,
        outputs=[topic_input, description_input, temperature_input, 
                use_rag_input, use_visual_fix_code_input, use_context_learning_input, 
                verbose_input, max_retries_input, max_scene_concurrency_input, api_key_input, result_text]
    )
    
    submit_btn.click(
        fn=submit_job,
        inputs=[
            topic_input, description_input, max_retries_input,
            use_rag_input, use_visual_fix_code_input, temperature_input, use_context_learning_input,
            verbose_input, max_scene_concurrency_input, api_key_input
        ],
        outputs=[result_text, job_id_output, status_container]
    ).then(
        fn=update_status_display,
        inputs=[job_id_output],
        outputs=[status_text, video_output, video_output, thumbnail_preview, scene_gallery, processing_time_text, cancel_btn]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
    )
    
    refresh_btn.click(
        fn=update_status_display,
        inputs=[job_id_output],
        outputs=[status_text, video_output, video_output, thumbnail_preview, scene_gallery, processing_time_text, cancel_btn]
    ).then(
        fn=update_stats,
        outputs=[stats_total, stats_completed, stats_running, stats_failed]
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
        return "🎬 Welcome to Theory2Manim Video Generator! Ready to create amazing educational videos."
    
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