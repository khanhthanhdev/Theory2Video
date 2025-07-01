import os
import json
import asyncio
import uuid
from typing import Union, List, Dict, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import argparse
import re
from dotenv import load_dotenv

from mllm_tools.litellm import LiteLLMWrapper
from mllm_tools.openrouter import OpenRouterWrapper
from src.core.video_planner import EnhancedVideoPlanner
from src.core.code_generator import CodeGenerator  # Use existing CodeGenerator
from src.core.video_renderer import VideoRenderer  # Use existing VideoRenderer
from src.utils.utils import extract_xml
from src.config.config import Config
from task_generator import get_banned_reasonings
from task_generator.prompts_raw import (_code_font_size, _code_disable, _code_limit, _prompt_manim_cheatsheet)

# Load configuration
load_dotenv(override=True)

# Load allowed models
allowed_models_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'allowed_models.json')
with open(allowed_models_path, 'r') as f:
    allowed_models_data = json.load(f)
    allowed_models = allowed_models_data.get("allowed_models", [])

@dataclass
class VideoGenerationConfig:
    """Configuration for video generation pipeline."""
    planner_model: str
    scene_model: Optional[str] = None
    helper_model: Optional[str] = None
    output_dir: str = "output"
    verbose: bool = False
    use_rag: bool = False
    use_context_learning: bool = False
    context_learning_path: str = "data/context_learning"
    chroma_db_path: str = "data/rag/chroma_db"
    manim_docs_path: str = "data/rag/manim_docs"
    embedding_model: str = "hf:ibm-granite/granite-embedding-30m-english"
    use_visual_fix_code: bool = False
    use_langfuse: bool = True
    max_scene_concurrency: int = 5
    max_topic_concurrency: int = 1
    max_retries: int = 5
    
    # Renderer optimizations
    enable_caching: bool = True
    default_quality: str = "medium"
    use_gpu_acceleration: bool = False
    preview_mode: bool = False
    max_concurrent_renders: int = 4

# Protocols for dependency injection (Interface Segregation Principle)
class ModelProvider(Protocol):
    """Protocol for AI model providers."""
    def __call__(self, prompt: str, **kwargs) -> str: ...

class PlannerInterface(Protocol):
    """Interface for video planners."""
    async def generate_scene_outline(self, topic: str, description: str, session_id: str) -> str: ...
    async def generate_scene_implementation_concurrently_enhanced(
        self, topic: str, description: str, plan: str, session_id: str
    ) -> List[str]: ...

class CodeGeneratorInterface(Protocol):
    """Interface for code generators."""
    def generate_manim_code(self, **kwargs) -> tuple: ...
    def fix_code_errors(self, **kwargs) -> tuple: ...
    def visual_self_reflection(self, **kwargs) -> tuple: ...

class RendererInterface(Protocol):
    """Interface for video renderers."""
    async def render_scene_optimized(self, **kwargs) -> tuple: ...
    async def combine_videos_optimized(self, topic: str, **kwargs) -> str: ...

# Factory for creating components (Factory Pattern)
class ComponentFactory:
    """Factory for creating video generation components."""
    
    @staticmethod
    def create_model(model_name: str, config: VideoGenerationConfig) -> ModelProvider:
        """Create AI model wrapper."""
        # Use OpenRouter wrapper for OpenRouter models
        if model_name.startswith('openrouter/'):
            return OpenRouterWrapper(
                model_name=model_name,
                temperature=0.7,
                print_cost=True,
                verbose=config.verbose,
                use_langfuse=config.use_langfuse
            )
        else:
            # Use LiteLLM wrapper for other models
            return LiteLLMWrapper(
                model_name=model_name,
                temperature=0.7,
                print_cost=True,
                verbose=config.verbose,
                use_langfuse=config.use_langfuse
            )
    
    @staticmethod
    def create_planner(planner_model: ModelProvider, helper_model: ModelProvider, 
                      config: VideoGenerationConfig, session_id: str) -> PlannerInterface:
        """Create video planner with enhanced capabilities."""
        return EnhancedVideoPlanner(
            planner_model=planner_model,
            helper_model=helper_model,
            output_dir=config.output_dir,
            print_response=config.verbose,
            use_context_learning=config.use_context_learning,
            context_learning_path=config.context_learning_path,
            use_rag=config.use_rag,
            session_id=session_id,
            chroma_db_path=config.chroma_db_path,
            manim_docs_path=config.manim_docs_path,
            embedding_model=config.embedding_model,
            use_langfuse=config.use_langfuse,
            max_scene_concurrency=config.max_scene_concurrency,
            max_step_concurrency=3,
            enable_caching=config.enable_caching
        )
    
    @staticmethod
    def create_code_generator(scene_model: ModelProvider, helper_model: ModelProvider,
                            config: VideoGenerationConfig, session_id: str) -> CodeGeneratorInterface:
        """Create code generator with existing implementation."""
        return CodeGenerator(  # Use existing CodeGenerator
            scene_model=scene_model,
            helper_model=helper_model,
            output_dir=config.output_dir,
            print_response=config.verbose,
            use_rag=config.use_rag,
            use_context_learning=config.use_context_learning,
            context_learning_path=config.context_learning_path,
            chroma_db_path=config.chroma_db_path,
            manim_docs_path=config.manim_docs_path,
            embedding_model=config.embedding_model,
            use_visual_fix_code=config.use_visual_fix_code,
            use_langfuse=config.use_langfuse,
            session_id=session_id
        )
    
    @staticmethod
    def create_renderer(config: VideoGenerationConfig) -> RendererInterface:
        """Create video renderer with existing implementation."""
        return VideoRenderer(  # Use existing VideoRenderer
            output_dir=config.output_dir,
            print_response=config.verbose,
            use_visual_fix_code=config.use_visual_fix_code
        )

# Enhanced VideoRenderer wrapper to add async methods
class AsyncVideoRendererWrapper:
    """Wrapper to add async functionality to existing VideoRenderer."""
    
    def __init__(self, renderer: VideoRenderer, config: VideoGenerationConfig):
        self.renderer = renderer
        self.config = config
        self.render_stats = {'cache_hits': 0, 'total_renders': 0}
    
    async def render_scene_optimized(self, **kwargs) -> tuple:
        """Async wrapper for scene rendering with intelligent error handling."""
        # Extract parameters
        code = kwargs.get('code')
        file_prefix = kwargs.get('file_prefix')
        curr_scene = kwargs.get('curr_scene')
        curr_version = kwargs.get('curr_version', 1)
        code_dir = kwargs.get('code_dir')
        media_dir = kwargs.get('media_dir')
        code_generator = kwargs.get('code_generator')
        scene_implementation = kwargs.get('scene_implementation')
        description = kwargs.get('description')
        scene_outline = kwargs.get('scene_outline')
        scene_trace_id = kwargs.get('scene_trace_id')
        topic = kwargs.get('topic')
        session_id = kwargs.get('session_id')
        
        # Use existing render_scene method with all parameters
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.renderer.render_scene,
            code,
            file_prefix,
            curr_scene,
            curr_version,
            code_dir,
            media_dir,
            False,  # use_visual_fix_code
            None,   # visual_self_reflection_func
            None,   # banned_reasonings
            scene_trace_id,
            topic,
            session_id,
            code_generator,
            scene_implementation,
            description,
            scene_outline
        )
        
        self.render_stats['total_renders'] += 1
        return result
    
    async def render_multiple_scenes_parallel(self, scene_configs: List[Dict], 
                                           max_concurrent: int = None) -> List[tuple]:
        """Render multiple scenes in parallel."""
        max_concurrent = max_concurrent or self.config.max_concurrent_renders
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def render_single_scene(config):
            async with semaphore:
                return await self.render_scene_optimized(**config)
        
        print(f"🚀 Starting parallel rendering of {len(scene_configs)} scenes (max concurrent: {max_concurrent})")
        
        tasks = [render_single_scene(config) for config in scene_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"📊 Render results: {successful}/{len(results)} scenes successful")
        
        return results
    
    async def combine_videos_optimized(self, topic: str, **kwargs) -> str:
        """Async wrapper for video combination."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.renderer.combine_videos,
            topic
        )

# Service classes (Single Responsibility Principle)
class SessionManager:
    """Manages session IDs for video generation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def load_or_create_session_id(self) -> str:
        """Load existing session ID or create new one."""
        session_file = os.path.join(self.output_dir, "session_id.txt")
        
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_id = f.read().strip()
                print(f"📋 Loaded existing session ID: {session_id}")
                return session_id
        
        session_id = str(uuid.uuid4())
        os.makedirs(self.output_dir, exist_ok=True)
        with open(session_file, 'w') as f:
            f.write(session_id)
        print(f"🆕 Created new session ID: {session_id}")
        return session_id
    
    def save_topic_session_id(self, topic: str, session_id: str) -> None:
        """Save session ID for specific topic."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        topic_dir = os.path.join(self.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)
        
        session_file = os.path.join(topic_dir, "session_id.txt")
        with open(session_file, 'w') as f:
            f.write(session_id)

class SceneAnalyzer:
    """Analyzes and manages scene information."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def load_implementation_plans(self, topic: str) -> Dict[int, Optional[str]]:
        """Load implementation plans for each scene."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        
        if not os.path.exists(scene_outline_path):
            return {}
        
        with open(scene_outline_path, "r") as f:
            scene_outline = f.read()
        
        scene_outline_content = extract_xml(scene_outline)
        scene_count = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
        
        implementation_plans = {}
        for i in range(1, scene_count + 1):
            plan_path = os.path.join(
                self.output_dir, file_prefix, f"scene{i}", 
                f"{file_prefix}_scene{i}_implementation_plan.txt"
            )
            if os.path.exists(plan_path):
                with open(plan_path, "r") as f:
                    implementation_plans[i] = f.read()
                print(f"📄 Found existing implementation plan for scene {i}")
            else:
                implementation_plans[i] = None
                print(f"❌ Missing implementation plan for scene {i}")
        
        return implementation_plans
    
    def analyze_scene_status(self, topic: str) -> Dict:
        """Analyze status of all scenes for a topic."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        
        # Check scene outline
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        has_scene_outline = os.path.exists(scene_outline_path)
        
        num_scenes = 0
        if has_scene_outline:
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            scene_outline_content = extract_xml(scene_outline)
            num_scenes = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
        
        # Analyze each scene
        scene_status = []
        implementation_plans = code_files = rendered_scenes = 0
        
        for i in range(1, num_scenes + 1):
            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i}")
            
            # Check implementation plan
            plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt")
            has_plan = os.path.exists(plan_path)
            if has_plan:
                implementation_plans += 1
            
            # Check code files
            code_dir = os.path.join(scene_dir, "code")
            has_code = os.path.exists(code_dir) and any(f.endswith('.py') for f in os.listdir(code_dir))
            if has_code:
                code_files += 1
            
            # Check rendered videos
            has_render = os.path.exists(os.path.join(scene_dir, "succ_rendered.txt"))
            if has_render:
                rendered_scenes += 1
            
            scene_status.append({
                'scene_number': i,
                'has_plan': has_plan,
                'has_code': has_code,
                'has_render': has_render
            })
        
        # Check combined video
        combined_video_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_combined.mp4")
        has_combined_video = os.path.exists(combined_video_path)
        
        return {
            'topic': topic,
            'has_scene_outline': has_scene_outline,
            'total_scenes': num_scenes,
            'implementation_plans': implementation_plans,
            'code_files': code_files,
            'rendered_scenes': rendered_scenes,
            'has_combined_video': has_combined_video,
            'scene_status': scene_status
        }

# Scene rendering wrapper for existing render_scene method
class SceneRenderingService:
    """Service for rendering individual scenes with existing VideoRenderer."""
    
    def __init__(self, renderer: VideoRenderer, code_generator: CodeGenerator, 
                 banned_reasonings: List[str], config: VideoGenerationConfig):
        self.renderer = renderer
        self.code_generator = code_generator
        self.banned_reasonings = banned_reasonings
        self.config = config
    
    async def render_scene_with_code_generation(self, topic: str, description: str,
                                              scene_outline: str, scene_implementation: str,
                                              scene_number: int, file_prefix: str,
                                              code_dir: str, media_dir: str,
                                              scene_trace_id: str, session_id: str) -> tuple:
        """Render a scene with code generation and error handling."""
        
        print(f"🎬 Processing scene {scene_number} for {topic}")
        
        try:
            # Step 1: Generate Manim code
            print(f"⚡ Generating code for scene {scene_number}")
            code, _ = self.code_generator.generate_manim_code(
                topic=topic,
                description=description,
                scene_outline=scene_outline,
                scene_implementation=scene_implementation,
                scene_number=scene_number,
                scene_trace_id=scene_trace_id,
                session_id=session_id
            )
            
            # Step 2: Render with intelligent error handling (single attempt - renderer handles retries and fixes)
            current_version = 1
            
            print(f"🎞️ Rendering scene {scene_number} with intelligent error handling")
            
            try:
                # Use existing render_scene method with enhanced error handling
                loop = asyncio.get_event_loop()
                result_code, error = await loop.run_in_executor(
                    None,
                    self.renderer.render_scene,
                    code,
                    file_prefix,
                    scene_number,
                    current_version,
                    code_dir,
                    media_dir,
                    False,  # use_visual_fix_code
                    None,   # visual_self_reflection_func
                    self.banned_reasonings,
                    scene_trace_id,
                    topic,
                    session_id,
                    self.code_generator,  # Pass code generator for intelligent error handling
                    scene_implementation,  # Pass implementation for context
                    description,  # Pass description for context
                    scene_outline  # Pass scene outline for context
                )
                
                if error is None:
                    # Success - mark as rendered
                    scene_dir = os.path.join(self.config.output_dir, file_prefix, f"scene{scene_number}")
                    success_file = os.path.join(scene_dir, "succ_rendered.txt")
                    with open(success_file, 'w') as f:
                        f.write(f"Successfully rendered with intelligent error handling")
                    
                    print(f"✅ Scene {scene_number} rendered successfully")
                    return result_code, None
                else:
                    # Error occurred even with intelligent retry/fix attempts
                    print(f"❌ Scene {scene_number} failed after intelligent error handling: {error}")
                    return result_code, error
                
            except Exception as e:
                print(f"❌ Exception during scene {scene_number} rendering: {e}")
                return code, str(e)
            
        except Exception as e:
            print(f"❌ Fatal error in scene {scene_number}: {e}")
            return None, str(e)

# Main Video Generator (Open/Closed Principle - extensible via composition)
class EnhancedVideoGenerator:
    """Enhanced video generator following SOLID principles."""
    
    def __init__(self, config: VideoGenerationConfig):
        self.config = config
        self.session_manager = SessionManager(config.output_dir)
        self.scene_analyzer = SceneAnalyzer(config.output_dir)
        self.banned_reasonings = get_banned_reasonings()
        
        # Initialize session
        self.session_id = self.session_manager.load_or_create_session_id()
        
        # Create AI models
        self.planner_model = ComponentFactory.create_model(config.planner_model, config)
        self.scene_model = ComponentFactory.create_model(
            config.scene_model or config.planner_model, config
        )
        self.helper_model = ComponentFactory.create_model(
            config.helper_model or config.planner_model, config
        )
        
        # Create components using dependency injection
        self.planner = ComponentFactory.create_planner(
            self.planner_model, self.helper_model, config, self.session_id
        )
        self.code_generator = ComponentFactory.create_code_generator(
            self.scene_model, self.helper_model, config, self.session_id
        )
        
        # Create renderer with async wrapper
        base_renderer = ComponentFactory.create_renderer(config)
        self.renderer = AsyncVideoRendererWrapper(base_renderer, config)
        
        # Create scene rendering service
        self.scene_service = SceneRenderingService(
            base_renderer, self.code_generator, self.banned_reasonings, config
        )
        
        # Concurrency control
        self.scene_semaphore = asyncio.Semaphore(config.max_scene_concurrency)
        
        print(f"🚀 Enhanced VideoGenerator initialized with:")
        print(f"   Planner: {config.planner_model}")
        print(f"   Scene: {config.scene_model or config.planner_model}")
        print(f"   Helper: {config.helper_model or config.planner_model}")
        print(f"   Max Scene Concurrency: {config.max_scene_concurrency}")
        print(f"   Caching: {'✅' if config.enable_caching else '❌'}")
        print(f"   GPU Acceleration: {'✅' if config.use_gpu_acceleration else '❌'}")

    async def generate_scene_outline(self, topic: str, description: str) -> str:
        """Generate scene outline for topic."""
        print(f"📝 Generating scene outline for: {topic}")
        return await self.planner.generate_scene_outline(topic, description, self.session_id)

    async def generate_video_pipeline(self, topic: str, description: str, 
                                    only_plan: bool = False, 
                                    specific_scenes: List[int] = None) -> None:
        """Complete video generation pipeline with enhanced performance."""
        
        print(f"🎬 Starting enhanced video pipeline for: {topic}")
        self.session_manager.save_topic_session_id(topic, self.session_id)
        
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        
        # Step 1: Load or generate scene outline
        scene_outline = await self._load_or_generate_outline(topic, description, file_prefix)
        
        # Step 2: Generate implementation plans
        implementation_plans = await self._generate_implementation_plans(
            topic, description, scene_outline, file_prefix, specific_scenes
        )
        
        if only_plan:
            print(f"📋 Plan-only mode completed for: {topic}")
            return
        
        # Step 3: Render scenes with optimization
        await self._render_scenes_optimized(
            topic, description, scene_outline, implementation_plans, file_prefix
        )
        
        # Step 4: Combine videos
        await self._combine_videos_optimized(topic)
        
        print(f"✅ Enhanced video pipeline completed for: {topic}")

    async def _load_or_generate_outline(self, topic: str, description: str, file_prefix: str) -> str:
        """Load existing outline or generate new one."""
        scene_outline_path = os.path.join(self.config.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        
        if os.path.exists(scene_outline_path):
            with open(scene_outline_path, "r") as f:
                scene_outline = f.read()
            print(f"📄 Loaded existing scene outline for: {topic}")
            
            # Detect plugins if RAG is enabled
            if self.config.use_rag and hasattr(self.planner, 'rag_integration'):
                plugins = self.planner.rag_integration.detect_relevant_plugins(topic, description)
                if plugins:
                    self.planner.rag_integration.set_relevant_plugins(plugins)
                    print(f"🔌 Detected relevant plugins: {plugins}")
        else:
            print(f"📝 Generating new scene outline for: {topic}")
            scene_outline = await self.planner.generate_scene_outline(topic, description, self.session_id)
            
            os.makedirs(os.path.join(self.config.output_dir, file_prefix), exist_ok=True)
            with open(scene_outline_path, "w") as f:
                f.write(scene_outline)
        
        return scene_outline

    async def _generate_implementation_plans(self, topic: str, description: str, 
                                           scene_outline: str, file_prefix: str,
                                           specific_scenes: List[int] = None) -> Dict[int, str]:
        """Generate missing implementation plans."""
        
        # First, ensure the topic directory exists
        topic_dir = os.path.join(self.config.output_dir, file_prefix)
        os.makedirs(topic_dir, exist_ok=True)
        
        try:
            implementation_plans_dict = self.scene_analyzer.load_implementation_plans(topic)
            
            if not implementation_plans_dict:
                print(f"No existing implementation plans found for {topic}. Generating all plans from scratch.")
                scene_outline_content = extract_xml(scene_outline)
                scene_count = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline_content))
                
                if scene_count == 0:
                    print(f"⚠️ Warning: No scenes found in scene outline. Check the regex pattern and scene outline format.")
                    print(f"Scene outline content: {scene_outline_content[:100]}...")
                
                print(f"Found {scene_count} scenes in the outline.")
                implementation_plans_dict = {i: None for i in range(1, scene_count + 1)}
            
            # Find missing scenes
            missing_scenes = [
                scene_num for scene_num, plan in implementation_plans_dict.items()
                if plan is None and (specific_scenes is None or scene_num in specific_scenes)
            ]
            
            if missing_scenes:
                print(f"📋 Generating implementation plans for scenes: {missing_scenes}")
                
                # Make sure scene directories exist for missing scenes
                for scene_num in missing_scenes:
                    scene_dir = os.path.join(topic_dir, f"scene{scene_num}")
                    os.makedirs(scene_dir, exist_ok=True)
                
                # Use enhanced concurrent generation if available
                if hasattr(self.planner, 'generate_scene_implementation_concurrently_enhanced'):
                    try:
                        all_plans = await self.planner.generate_scene_implementation_concurrently_enhanced(
                            topic, description, scene_outline, self.session_id
                        )
                        
                        if not all_plans:
                            print(f"❌ Error: No implementation plans were returned!")
                            return implementation_plans_dict
                            
                        # Update missing plans
                        updated_count = 0
                        for i, scene_num in enumerate(sorted(missing_scenes)):
                            if i < len(all_plans):
                                plan = all_plans[i]
                                if isinstance(plan, str) and plan.strip():
                                    implementation_plans_dict[scene_num] = plan
                                    updated_count += 1
                                else:
                                    print(f"⚠️ Warning: Empty or invalid plan for scene {scene_num}")
                        
                        print(f"✅ Generated {updated_count}/{len(missing_scenes)} implementation plans")
                        
                    except Exception as e:
                        print(f"❌ Error generating implementation plans: {str(e)}")
                        
                else:
                    # Fallback to sequential generation
                    print("⚠️ Using fallback sequential plan generation")
                    # Implement sequential generation if needed
            else:
                print("✅ All implementation plans already exist.")
                
            return implementation_plans_dict
            
        except Exception as e:
            print(f"❌ Fatal error in implementation plan generation: {str(e)}")
            raise

    async def _render_scenes_optimized(self, topic: str, description: str, 
                                     scene_outline: str, implementation_plans: Dict[int, str],
                                     file_prefix: str) -> None:
        """Render scenes with enhanced optimization."""
        
        # Determine which scenes need processing
        scenes_to_process = self._get_scenes_to_process(implementation_plans, file_prefix)
        
        if not scenes_to_process:
            print(f"✅ No scenes need processing for: {topic}")
            return
        
        print(f"🎬 Rendering {len(scenes_to_process)} scenes with optimization...")
        
        # Create render tasks
        render_tasks = []
        for scene_num, implementation_plan in scenes_to_process:
            task = self._create_scene_render_task(
                topic, description, scene_outline, implementation_plan,
                scene_num, file_prefix
            )
            render_tasks.append(task)
        
        # Execute with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_renders)
        
        async def execute_render_task(task):
            async with semaphore:
                return await task
        
        # Run all render tasks
        results = await asyncio.gather(
            *[execute_render_task(task) for task in render_tasks],
            return_exceptions=True
        )
        
        # Process results
        successful_renders = sum(1 for r in results if not isinstance(r, Exception) and r[1] is None)
        print(f"📊 Render results: {successful_renders}/{len(results)} scenes successful")

    def _get_scenes_to_process(self, implementation_plans: Dict[int, str], 
                             file_prefix: str) -> List[tuple]:
        """Determine which scenes need processing."""
        scenes_to_process = []
        
        for scene_num, implementation_plan in implementation_plans.items():
            if implementation_plan is None:
                continue
                
            scene_dir = os.path.join(self.config.output_dir, file_prefix, f"scene{scene_num}")
            
            # Check if scene already successfully rendered
            if not os.path.exists(os.path.join(scene_dir, "succ_rendered.txt")):
                scenes_to_process.append((scene_num, implementation_plan))
        
        return scenes_to_process

    def _create_scene_render_task(self, topic: str, description: str, scene_outline: str,
                                      implementation_plan: str, scene_num: int, file_prefix: str):
        """Create render task for a scene."""
        
        # Generate or load scene trace ID
        scene_dir = os.path.join(self.config.output_dir, file_prefix, f"scene{scene_num}")
        subplan_dir = os.path.join(scene_dir, "subplans")
        os.makedirs(subplan_dir, exist_ok=True)
        
        scene_trace_id_path = os.path.join(subplan_dir, "scene_trace_id.txt")
        try:
            with open(scene_trace_id_path, 'r') as f:
                scene_trace_id = f.read().strip()
        except FileNotFoundError:
            scene_trace_id = str(uuid.uuid4())
            with open(scene_trace_id_path, 'w') as f:
                f.write(scene_trace_id)
        
        # Create directories
        code_dir = os.path.join(scene_dir, "code")
        media_dir = os.path.join(self.config.output_dir, file_prefix, "media")
        os.makedirs(code_dir, exist_ok=True)
        
        # Return coroutine that will be awaited later
        return self.scene_service.render_scene_with_code_generation(
            topic=topic,
            description=description,
            scene_outline=scene_outline,
            scene_implementation=implementation_plan,
            scene_number=scene_num,
            file_prefix=file_prefix,
            code_dir=code_dir,
            media_dir=media_dir,
            scene_trace_id=scene_trace_id,
            session_id=self.session_id
        )

    async def _combine_videos_optimized(self, topic: str) -> None:
        """Combine videos with hardware acceleration."""
        print(f"🎞️ Combining videos for: {topic}")
        
        try:
            output_path = await self.renderer.combine_videos_optimized(
                topic, use_hardware_acceleration=self.config.use_gpu_acceleration
            )
            print(f"✅ Combined video saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error combining videos: {e}")

    async def process_multiple_topics(self, topics_data: List[Dict], 
                                    only_plan: bool = False,
                                    specific_scenes: List[int] = None) -> None:
        """Process multiple topics concurrently."""
        
        topic_semaphore = asyncio.Semaphore(self.config.max_topic_concurrency)
        
        async def process_single_topic(topic_data):
            async with topic_semaphore:
                topic = topic_data['theorem']
                description = topic_data['description']
                print(f"🎯 Processing topic: {topic}")
                
                try:
                    await self.generate_video_pipeline(
                        topic, description, only_plan=only_plan, 
                        specific_scenes=specific_scenes
                    )
                    print(f"✅ Completed topic: {topic}")
                except Exception as e:
                    print(f"❌ Error processing {topic}: {e}")
        
        tasks = [process_single_topic(topic_data) for topic_data in topics_data]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_status_summary(self, topics_data: List[Dict]) -> None:
        """Print comprehensive status summary."""
        print("\n📊 Comprehensive Status Summary")
        print("=" * 160)
        
        all_statuses = [
            self.scene_analyzer.analyze_scene_status(topic_data['theorem']) 
            for topic_data in topics_data
        ]
        
        # Print header
        print(f"{'Topic':<40} {'Outline':<8} {'Total':<8} {'Status (Plan/Code/Render)':<50} {'Combined':<10} {'Missing Components':<40}")
        print("-" * 160)
        
        # Print each topic status
        for status in all_statuses:
            scene_status_str = ""
            for scene in status['scene_status']:
                scene_str = (
                    ("P" if scene['has_plan'] else "-") +
                    ("C" if scene['has_code'] else "-") +
                    ("R" if scene['has_render'] else "-") + " "
                )
                scene_status_str += scene_str
            
            # Collect missing components
            missing_components = self._format_missing_components(status['scene_status'])
            
            print(f"{status['topic'][:37]+'...' if len(status['topic'])>37 else status['topic']:<40} "
                f"{'✓' if status['has_scene_outline'] else '✗':<8} "
                f"{status['total_scenes']:<8} "
                f"{scene_status_str[:47]+'...' if len(scene_status_str)>47 else scene_status_str:<50} "
                f"{'✓' if status['has_combined_video'] else '✗':<10} "
                f"{missing_components[:37]+'...' if len(missing_components)>37 else missing_components:<40}")
        
        # Print summary statistics
        self._print_summary_statistics(all_statuses, len(topics_data))

    def _format_missing_components(self, scene_status: List[Dict]) -> str:
        """Format missing components string."""
        missing_plans = [str(s['scene_number']) for s in scene_status if not s['has_plan']]
        missing_code = [str(s['scene_number']) for s in scene_status if not s['has_code']]
        missing_renders = [str(s['scene_number']) for s in scene_status if not s['has_render']]
        
        missing_str = []
        if missing_plans:
            missing_str.append(f"P:{','.join(missing_plans)}")
        if missing_code:
            missing_str.append(f"C:{','.join(missing_code)}")
        if missing_renders:
            missing_str.append(f"R:{','.join(missing_renders)}")
        
        return ' '.join(missing_str)

    def _print_summary_statistics(self, all_statuses: List[Dict], total_topics: int) -> None:
        """Print summary statistics."""
        total_scenes = sum(status['total_scenes'] for status in all_statuses)
        total_plans = sum(status['implementation_plans'] for status in all_statuses)
        total_code = sum(status['code_files'] for status in all_statuses)
        total_renders = sum(status['rendered_scenes'] for status in all_statuses)
        total_combined = sum(1 for status in all_statuses if status['has_combined_video'])
        
        print("\n📈 Summary Statistics:")
        print(f"   Total topics: {total_topics}")
        print(f"   Total scenes: {total_scenes}")
        print(f"   Completion rates:")
        print(f"     Plans: {total_plans}/{total_scenes} ({total_plans/max(1,total_scenes)*100:.1f}%)")
        print(f"     Code: {total_code}/{total_scenes} ({total_code/max(1,total_scenes)*100:.1f}%)")
        print(f"     Renders: {total_renders}/{total_scenes} ({total_renders/max(1,total_scenes)*100:.1f}%)")
        print(f"     Combined videos: {total_combined}/{total_topics} ({total_combined/max(1,total_topics)*100:.1f}%)")

# Command-line interface
class VideoGeneratorCLI:
    """Command-line interface for video generation."""
    
    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create argument parser with all options."""
        parser = argparse.ArgumentParser(description='Enhanced Manim Video Generator')
        
        # Model configuration
        parser.add_argument('--model', type=str, choices=allowed_models,
                          default='gemini/gemini-2.5-flash-preview-04-17', help='AI model to use')
        parser.add_argument('--scene_model', type=str, choices=allowed_models,
                          help='Specific model for scene generation')
        parser.add_argument('--helper_model', type=str, choices=allowed_models,
                          help='Helper model for additional tasks')
        
        # Input/Output
        parser.add_argument('--topic', type=str, help='Single topic to process')
        parser.add_argument('--context', type=str, help='Context for the topic')
        parser.add_argument('--theorems_path', type=str, help='Path to theorems JSON file')
        parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory')
        
        # Processing options
        parser.add_argument('--sample_size', type=int, help='Number of theorems to sample')
        parser.add_argument('--scenes', nargs='+', type=int, help='Specific scenes to process')
        parser.add_argument('--max_retries', type=int, default=5, help='Maximum retries for code generation')
        
        # Mode flags
        parser.add_argument('--only_plan', action='store_true', help='Only generate plans')
        parser.add_argument('--only_render', action='store_true', help='Only render scenes')
        parser.add_argument('--only_combine', action='store_true', help='Only combine videos')
        parser.add_argument('--check_status', action='store_true', help='Check status of all topics')
        
        # Performance options
        parser.add_argument('--max_scene_concurrency', type=int, default=5, help='Max concurrent scenes')
        parser.add_argument('--max_topic_concurrency', type=int, default=1, help='Max concurrent topics')
        parser.add_argument('--max_concurrent_renders', type=int, default=4, help='Max concurrent renders')
        parser.add_argument('--quality', choices=['preview', 'low', 'medium', 'high', 'production'],
                          default='medium', help='Render quality preset')
        
        # Feature flags
        parser.add_argument('--verbose', action='store_true', help='Verbose output')
        parser.add_argument('--use_rag', action='store_true', help='Use RAG')
        parser.add_argument('--use_context_learning', action='store_true', help='Use context learning')
        parser.add_argument('--use_visual_fix_code', action='store_true', help='Use visual code fixing')
        parser.add_argument('--use_langfuse', action='store_true', help='Enable Langfuse logging')
        parser.add_argument('--enable_caching', action='store_true', default=True, help='Enable caching')
        parser.add_argument('--use_gpu_acceleration', action='store_true', default=False, help='Use GPU acceleration')
        parser.add_argument('--preview_mode', action='store_true', help='Enable preview mode')
        
        # Paths
        parser.add_argument('--chroma_db_path', type=str, default=Config.CHROMA_DB_PATH, help='ChromaDB path')
        parser.add_argument('--manim_docs_path', type=str, default=Config.MANIM_DOCS_PATH, help='Manim docs path')
        parser.add_argument('--context_learning_path', type=str, default=Config.CONTEXT_LEARNING_PATH, help='Context learning path')
        parser.add_argument('--embedding_model', type=str, default=Config.EMBEDDING_MODEL, help='Embedding model')
        
        return parser
    
    @staticmethod
    def create_config_from_args(args) -> VideoGenerationConfig:
        """Create configuration from command-line arguments."""
        return VideoGenerationConfig(
            planner_model=args.model,
            scene_model=args.scene_model,
            helper_model=args.helper_model,
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_rag=args.use_rag,
            use_context_learning=args.use_context_learning,
            context_learning_path=args.context_learning_path,
            chroma_db_path=args.chroma_db_path,
            manim_docs_path=args.manim_docs_path,
            embedding_model=args.embedding_model,
            use_visual_fix_code=args.use_visual_fix_code,
            use_langfuse=args.use_langfuse,
            max_scene_concurrency=args.max_scene_concurrency,
            max_topic_concurrency=args.max_topic_concurrency,
            max_retries=args.max_retries,
            enable_caching=args.enable_caching,
            default_quality=args.quality,
            use_gpu_acceleration=args.use_gpu_acceleration,
            preview_mode=args.preview_mode,
            max_concurrent_renders=args.max_concurrent_renders
        )

async def main():
    """Enhanced main function with improved error handling and performance."""
    parser = VideoGeneratorCLI.create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = VideoGeneratorCLI.create_config_from_args(args)
    
    # Initialize enhanced video generator
    video_generator = EnhancedVideoGenerator(config)
    
    try:
        if args.theorems_path:
            await handle_multiple_topics(video_generator, args)
        elif args.topic and args.context:
            await handle_single_topic(video_generator, args)
        else:
            print("❌ Please provide either (--theorems_path) or (--topic and --context)")
            return
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise

async def handle_multiple_topics(video_generator: EnhancedVideoGenerator, args):
    """Handle processing of multiple topics."""
    with open(args.theorems_path, "r") as f:
        theorems = json.load(f)
    
    if args.sample_size:
        theorems = theorems[:args.sample_size]
    
    if args.check_status:
        video_generator.get_status_summary(theorems)
        return
    
    if args.only_combine:
        for theorem in theorems:
            await video_generator._combine_videos_optimized(theorem['theorem'])
    else:
        await video_generator.process_multiple_topics(
            theorems, 
            only_plan=args.only_plan,
            specific_scenes=args.scenes
        )

async def handle_single_topic(video_generator: EnhancedVideoGenerator, args):
    """Handle processing of single topic."""
    if args.only_combine:
        await video_generator._combine_videos_optimized(args.topic)
    else:
        await video_generator.generate_video_pipeline(
            args.topic,
            args.context,
            only_plan=args.only_plan
        )

if __name__ == "__main__":
    asyncio.run(main())