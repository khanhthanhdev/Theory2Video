import os
import re
import json
import glob
from typing import List, Optional, Dict, Tuple
import uuid
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import aiofiles

from mllm_tools.utils import _prepare_text_inputs
from src.utils.utils import extract_xml
from task_generator import (
    get_prompt_scene_plan,
    get_prompt_scene_vision_storyboard,
    get_prompt_scene_technical_implementation,
    get_prompt_scene_animation_narration,
    get_prompt_context_learning_scene_plan,
    get_prompt_context_learning_vision_storyboard,
    get_prompt_context_learning_technical_implementation,
    get_prompt_context_learning_animation_narration,
    get_prompt_context_learning_code
)
from src.rag.rag_integration import RAGIntegration

class EnhancedVideoPlanner:
    """Enhanced video planner with improved parallelization and performance."""
    
    def __init__(self, planner_model, helper_model=None, output_dir="output", 
                 print_response=False, use_context_learning=False, 
                 context_learning_path="data/context_learning", use_rag=False, 
                 session_id=None, chroma_db_path="data/rag/chroma_db", 
                 manim_docs_path="data/rag/manim_docs", 
                 embedding_model="text-embedding-ada-002", use_langfuse=True,
                 max_scene_concurrency=5, max_step_concurrency=3, enable_caching=True,
                 use_all_scenes_single_call: bool = False):
        
        self.planner_model = planner_model
        self.helper_model = helper_model if helper_model is not None else planner_model
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_context_learning = use_context_learning
        self.context_learning_path = context_learning_path
        self.use_rag = use_rag
        self.session_id = session_id
        self.enable_caching = enable_caching
        self.use_all_scenes_single_call = use_all_scenes_single_call
        
        # Enhanced concurrency control
        self.max_scene_concurrency = max_scene_concurrency
        self.max_step_concurrency = max_step_concurrency
        self.scene_semaphore = asyncio.Semaphore(max_scene_concurrency)
        self.step_semaphore = asyncio.Semaphore(max_step_concurrency)
        
        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Cache for prompts and examples
        self._context_cache = {}
        self._prompt_cache = {}
        
        # Initialize context examples with caching
        self._initialize_context_examples()
        
        # Initialize RAG with enhanced settings
        self.rag_integration = None
        self.relevant_plugins = []
        if use_rag:
            self.rag_integration = RAGIntegration(
                helper_model=helper_model,
                output_dir=output_dir,
                chroma_db_path=chroma_db_path,
                manim_docs_path=manim_docs_path,
                embedding_model=embedding_model,
                use_langfuse=use_langfuse,
                session_id=session_id
            )

    def _initialize_context_examples(self):
        """Initialize and cache context examples for faster access."""
        example_types = [
            'scene_plan', 'scene_vision_storyboard', 'technical_implementation',
            'scene_animation_narration', 'code'
        ]
        
        if self.use_context_learning:
            for example_type in example_types:
                self._context_cache[example_type] = self._load_context_examples(example_type)
        else:
            for example_type in example_types:
                self._context_cache[example_type] = None

    @lru_cache(maxsize=128)
    def _get_cached_prompt(self, prompt_type: str, *args) -> str:
        """Get cached prompt to avoid regeneration."""
        prompt_generators = {
            'scene_plan': get_prompt_scene_plan,
            'scene_vision_storyboard': get_prompt_scene_vision_storyboard,
            'scene_technical_implementation': get_prompt_scene_technical_implementation,
            'scene_animation_narration': get_prompt_scene_animation_narration
        }
        
        generator = prompt_generators.get(prompt_type)
        if generator:
            return generator(*args)
        return ""

    async def _async_file_write(self, file_path: str, content: str):
        """Asynchronous file writing for better performance."""
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)

    async def _async_file_read(self, file_path: str) -> Optional[str]:
        """Asynchronous file reading."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except FileNotFoundError:
            return None

    async def _ensure_directories(self, *paths):
        """Asynchronously ensure directories exist."""
        loop = asyncio.get_event_loop()
        for path in paths:
            await loop.run_in_executor(self.thread_pool, lambda p: os.makedirs(p, exist_ok=True), path)

    def _load_context_examples(self, example_type: str) -> str:
        """Load context learning examples with improved performance."""
        if example_type in self._context_cache:
            return self._context_cache[example_type]
            
        examples = []
        file_patterns = {
            'scene_plan': '*_scene_plan.txt',
            'scene_vision_storyboard': '*_scene_vision_storyboard.txt',
            'technical_implementation': '*_technical_implementation.txt',
            'scene_animation_narration': '*_scene_animation_narration.txt',
            'code': '*.py'
        }
        
        pattern = file_patterns.get(example_type)
        if not pattern:
            return None

        # Use glob for faster file discovery
        search_pattern = os.path.join(self.context_learning_path, "**", pattern)
        for example_file in glob.glob(search_pattern, recursive=True):
            try:
                with open(example_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    examples.append(f"# Example from {os.path.basename(example_file)}\n{content}\n")
            except Exception as e:
                print(f"Warning: Could not load example {example_file}: {e}")

        if examples:
            formatted_examples = self._format_examples(example_type, examples)
            self._context_cache[example_type] = formatted_examples
            return formatted_examples
        return None

    def _format_examples(self, example_type: str, examples: List[str]) -> str:
        """Format examples using the appropriate template."""
        templates = {
            'scene_plan': get_prompt_context_learning_scene_plan,
            'scene_vision_storyboard': get_prompt_context_learning_vision_storyboard,
            'technical_implementation': get_prompt_context_learning_technical_implementation,
            'scene_animation_narration': get_prompt_context_learning_animation_narration,
            'code': get_prompt_context_learning_code
        }
        
        template = templates.get(example_type)
        if template:
            return template(examples="\n".join(examples))
        return None

    async def _call_model_async(self, model, prompt: str, metadata: Dict) -> str:
        """Run a potentially blocking model call in a thread to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        def _invoke():
            return model(_prepare_text_inputs(prompt), metadata=metadata)
        response = await loop.run_in_executor(self.thread_pool, _invoke)
        if self.print_response:
            preview = response[:500] + ("..." if len(response) > 500 else "")
            print(f"[Model Response Preview] {metadata.get('generation_name','')}: {preview}")
        return response

    async def generate_scene_outline(self, topic: str, description: str, session_id: str) -> str:
        """Enhanced scene outline generation with async I/O."""
        start_time = time.time()
        
        # Detect relevant plugins upfront if RAG is enabled
        if self.use_rag and self.rag_integration:
            plugin_detection_task = asyncio.create_task(
                self._detect_plugins_async(topic, description)
            )
        
        # Prepare prompt with cached examples
        prompt = self._get_cached_prompt('scene_plan', topic, description)
        
        if self.use_context_learning and self._context_cache.get('scene_plan'):
            prompt += f"\n\nHere are some example scene plans for reference:\n{self._context_cache['scene_plan']}"

        # Wait for plugin detection if enabled
        if self.use_rag and self.rag_integration:
            self.relevant_plugins = await plugin_detection_task
            print(f"✅ Detected relevant plugins: {self.relevant_plugins}")

        # Generate plan using planner model
        response_text = await self._call_model_async(
            self.planner_model,
            prompt,
            metadata={
                "generation_name": "scene_outline", 
                "tags": [topic, "scene-outline"], 
                "session_id": session_id
            }
        )
        
        # Extract scene outline with improved error handling
        scene_outline = self._extract_scene_outline_robust(response_text)

        # Async file operations
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        output_dir = os.path.join(self.output_dir, file_prefix)
        await self._ensure_directories(output_dir)
        
        file_path = os.path.join(output_dir, f"{file_prefix}_scene_outline.txt")
        await self._async_file_write(file_path, scene_outline)
        
        elapsed_time = time.time() - start_time
        print(f"Scene outline generated in {elapsed_time:.2f}s - saved to {file_prefix}_scene_outline.txt")

        return scene_outline

    async def _detect_plugins_async(self, topic: str, description: str) -> List[str]:
        """Asynchronously detect relevant plugins."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: self.rag_integration.detect_relevant_plugins(topic, description) or []
        )

    async def _generate_scene_step_parallel(self, step_name: str, prompt_func, 
                                          scene_trace_id: str, topic: str, 
                                          scene_number: int, session_id: str, 
                                          output_path: str, *args) -> Tuple[str, str]:
        """Generate a single scene step with async operations."""
        async with self.step_semaphore:  # Control step-level concurrency
            
            # Check cache first if enabled
            if self.enable_caching:
                cached_content = await self._async_file_read(output_path)
                if cached_content:
                    print(f"Using cached {step_name} for scene {scene_number}")
                    return cached_content, output_path
            
            print(f"🚀 Generating {step_name} for scene {scene_number}")
            start_time = time.time()
            
            # Generate prompt
            prompt = prompt_func(*args)
            
            # Add context examples if available (mapped to known keys)
            example_key_map = {
                'scene_vision_storyboard': 'scene_vision_storyboard',
                'scene_technical_implementation': 'technical_implementation',
                'scene_animation_narration': 'scene_animation_narration'
            }
            cache_key = example_key_map.get(step_name)
            if cache_key and self._context_cache.get(cache_key):
                prompt += f"\n\nHere are some example {step_name}s:\n{self._context_cache[cache_key]}"
            
            # Add RAG context if enabled
            if self.use_rag and self.rag_integration:
                rag_queries = await self._generate_rag_queries_async(
                    step_name, args, scene_trace_id, topic, scene_number, session_id
                )
                
                if rag_queries:
                    retrieved_docs = self.rag_integration.get_relevant_docs(
                        rag_queries=rag_queries,
                        scene_trace_id=scene_trace_id,
                        topic=topic,
                        scene_number=scene_number
                    )
                    prompt += f"\n\n{retrieved_docs}"

            # Generate content
            response = await self._call_model_async(
                self.planner_model,
                prompt,
                metadata={
                    "generation_name": step_name,
                    "trace_id": scene_trace_id,
                    "tags": [topic, f"scene{scene_number}"],
                    "session_id": session_id
                }
            )
            
            # Extract content using step-specific patterns
            extraction_patterns = {
                'scene_vision_storyboard': r'(<SCENE_VISION_STORYBOARD_PLAN>.*?</SCENE_VISION_STORYBOARD_PLAN>)',
                'scene_technical_implementation': r'(<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>.*?</SCENE_TECHNICAL_IMPLEMENTATION_PLAN>)',
                'scene_animation_narration': r'(<SCENE_ANIMATION_NARRATION_PLAN>.*?</SCENE_ANIMATION_NARRATION_PLAN>)'
            }
            
            pattern = extraction_patterns.get(step_name)
            if pattern:
                match = re.search(pattern, response, re.DOTALL)
                content = match.group(1) if match else response
            else:
                content = response
            
            # Async file save
            await self._async_file_write(output_path, content)
            
            elapsed_time = time.time() - start_time
            print(f"{step_name} for scene {scene_number} completed in {elapsed_time:.2f}s")
            
            return content, output_path

    async def _generate_rag_queries_async(self, step_name: str, args: tuple, 
                                        scene_trace_id: str, topic: str, 
                                        scene_number: int, session_id: str) -> List[Dict]:
        """Generate RAG queries asynchronously based on step type."""
        query_generators = {
            'scene_vision_storyboard': self.rag_integration._generate_rag_queries_storyboard,
            'scene_technical_implementation': self.rag_integration._generate_rag_queries_technical,
            'scene_animation_narration': self.rag_integration._generate_rag_queries_narration
        }
        
        generator = query_generators.get(step_name)
        if not generator:
            return []
        
        # Map args to appropriate parameters based on step
        if step_name == 'scene_vision_storyboard':
            scene_plan = args[3] if len(args) > 3 else ""
            return generator(
                scene_plan=scene_plan,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id,
                relevant_plugins=self.relevant_plugins
            )
        elif step_name == 'scene_technical_implementation':
            storyboard = args[4] if len(args) > 4 else ""
            return generator(
                storyboard=storyboard,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id,
                relevant_plugins=self.relevant_plugins
            )
        elif step_name == 'scene_animation_narration':
            storyboard = args[4] if len(args) > 4 else ""
            return generator(
                storyboard=storyboard,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id,
                relevant_plugins=self.relevant_plugins
            )
        
        return []

    async def _generate_combined_rag_queries(self, scene_outline: str, scene_trace_id: str, topic: str, scene_number: int, session_id: str) -> List[Dict]:
        """
        Generates RAG queries for both storyboard and technical implementation in a single LLM call
        to reduce latency.
        """
        plugins_info = ""
        if self.relevant_plugins:
            plugins_info = f"\n        The following plugins have been identified as relevant: {self.relevant_plugins}.\n        You can generate queries related to these plugins.\n        "

        prompt = f"""
        You are an expert in Manim and video production. Based on the following scene outline, generate a list of search queries to find relevant documentation for creating both the visual storyboard and the technical implementation.
        {plugins_info}
        Scene Outline:
        {scene_outline}

        Provide a JSON array of query objects. Each object should have a "query" field. The queries should be concise and effective for a documentation search engine.

        Example format:
        ```json
        [
            {{"query": "Manim camera zoom"}},
            {{"query": "concept of derivative visualization"}},
            {{"query": "Manim Write text animation"}}
        ]
        ```
        Your response must contain only the JSON array inside a ```json block.
        """

        response = await self._call_model_async(
            self.helper_model,
            prompt,
            metadata={
                "generation_name": "rag_query_generation_combined",
                "trace_id": scene_trace_id,
                "tags": [topic, f"scene{scene_number}", "rag-query"],
                "session_id": session_id
            }
        )

        try:
            # Robustly extract JSON from response, handling markdown code blocks
            match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if match:
                json_part = match.group(1)
            else:
                # Fallback for raw JSON
                json_part = response[response.find('['):response.rfind(']')+1]
            
            if not json_part:
                return []

            queries = json.loads(json_part)
            return queries
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ Warning: Could not parse RAG queries for scene {scene_number}. Error: {e}. Proceeding without RAG for this part.")
            return []

    async def _generate_scene_implementation_single_enhanced(self, topic: str, description: str,
                                                           scene_outline_i: str, scene_number: int,
                                                           file_prefix: str, session_id: str,
                                                           scene_trace_id: str) -> str:
        """Generates all implementation plans for a single scene in one LLM call."""
        start_time = time.time()
        print(f"🚀 Starting scene {scene_number} implementation (Consolidated Call)")

        # Setup directories
        scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_number}")
        subplan_dir = os.path.join(scene_dir, "subplans")
        await self._ensure_directories(scene_dir, subplan_dir)

        # Save scene trace ID
        await self._async_file_write(os.path.join(subplan_dir, "scene_trace_id.txt"), scene_trace_id)

        # Check cache first if enabled
        combined_plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{scene_number}_implementation_plan.txt")
        if self.enable_caching:
            cached_content = await self._async_file_read(combined_plan_path)
            if cached_content:
                print(f"✅ Using cached implementation plan for scene {scene_number}")
                return cached_content

        # --- New Consolidated Prompt ---
        # This prompt asks for all three plans at once.
        prompt = f"""
        You are an expert in educational video production and Manim animation.
        Your task is to generate the complete implementation plan for a single scene in a video about '{topic}'.

        Video Description: {description}
        Scene Outline: {scene_outline_i}

        Please generate the following three plans enclosed in their respective XML tags:
        1.  <SCENE_VISION_STORYBOARD_PLAN>: A detailed visual storyboard.
        2.  <SCENE_TECHNICAL_IMPLEMENTATION_PLAN>: A technical plan for Manim code.
        3.  <SCENE_ANIMATION_NARRATION_PLAN>: The narration script for the scene.

        Ensure the response contains all three plans within a single block.
        """

        # Add RAG context if enabled
        if self.use_rag and self.rag_integration:
            # For a consolidated call, we generate RAG queries for all steps in one go
            all_queries = await self._generate_combined_rag_queries(
                scene_outline_i, scene_trace_id, topic, scene_number, session_id
            )
            if all_queries:
                retrieved_docs = self.rag_integration.get_relevant_docs(all_queries, scene_trace_id, topic, scene_number)
                prompt += f"\n\nHere is some context from relevant documentation:\n{retrieved_docs}"

        # --- Single LLM Call ---
        response = await self._call_model_async(
            self.planner_model,
            prompt,
            metadata={
                "generation_name": "scene_implementation_combined",
                "trace_id": scene_trace_id,
                "tags": [topic, f"scene{scene_number}"],
                "session_id": session_id
            }
        )

        # --- Extract and Save Individual Plans ---
        def extract_plan(tag, text):
            match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
            return match.group(1).strip() if match else f"<{tag}>Plan not generated.</{tag}>"

        vision_storyboard_content = extract_plan('SCENE_VISION_STORYBOARD_PLAN', response)
        technical_implementation_content = extract_plan('SCENE_TECHNICAL_IMPLEMENTATION_PLAN', response)
        animation_narration_content = extract_plan('SCENE_ANIMATION_NARRATION_PLAN', response)

        await self._async_file_write(os.path.join(subplan_dir, f"{file_prefix}_scene{scene_number}_vision_storyboard_plan.txt"), vision_storyboard_content)
        await self._async_file_write(os.path.join(subplan_dir, f"{file_prefix}_scene{scene_number}_technical_implementation_plan.txt"), technical_implementation_content)
        await self._async_file_write(os.path.join(subplan_dir, f"{file_prefix}_scene{scene_number}_animation_narration_plan.txt"), animation_narration_content)

        # --- Combine and Save Final Plan ---
        implementation_plan = (
            f"{vision_storyboard_content}\n\n"
            f"{technical_implementation_content}\n\n"
            f"{animation_narration_content}\n\n"
        )
        combined_content = f"# Scene {scene_number} Implementation Plan\n\n{implementation_plan}"
        await self._async_file_write(combined_plan_path, combined_content)
        print(f"✅ Saved implementation plan for scene {scene_number} to: {combined_plan_path}")

        elapsed_time = time.time() - start_time
        print(f"Scene {scene_number} implementation completed in {elapsed_time:.2f}s")

        return implementation_plan

    async def generate_scene_implementation_concurrently_enhanced(self, topic: str, description: str, 
                                                                plan: str, session_id: str) -> List[str]:
        """Enhanced concurrent scene implementation with better performance."""
        start_time = time.time()
        
        # If configured, handle all scenes in a single LLM call
        if self.use_all_scenes_single_call:
            return await self.generate_scene_implementation_all_in_one(topic, description, plan, session_id)
        
        # Extract scene information
        scene_outline = extract_xml(plan)
        scene_number = len(re.findall(r'<SCENE_(\d+)>', scene_outline))
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        
        print(f"Starting implementation generation for {scene_number} scenes with max concurrency: {self.max_scene_concurrency}")

        async def generate_single_scene_implementation(i):
            async with self.scene_semaphore:  # Control scene-level concurrency
                scene_regex = r'(<SCENE_{0}>.*?</SCENE_{0}>)'.format(i)
                scene_match = re.search(
                    scene_regex, 
                    scene_outline, 
                    re.DOTALL
                )
                if not scene_match:
                    print(f"❌ Error: Could not find scene {i} in scene outline. Regex pattern: {scene_regex}")
                    raise ValueError(f"Scene {i} not found in scene outline")
                scene_outline_i = scene_match.group(1)
                scene_trace_id = str(uuid.uuid4())
                
                return await self._generate_scene_implementation_single_enhanced(
                    topic, description, scene_outline_i, i, file_prefix, session_id, scene_trace_id
                )

        # Create tasks for all scenes
        tasks = [generate_single_scene_implementation(i + 1) for i in range(scene_number)]
        
        # Execute with progress tracking
        print(f"Executing {len(tasks)} scene implementation tasks...")
        try:
            all_scene_implementation_plans = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            successful_plans = []
            error_count = 0
            for i, result in enumerate(all_scene_implementation_plans):
                if isinstance(result, Exception):
                    print(f"❌ Error in scene {i+1}: {result}")
                    error_message = f"# Scene {i+1} - Error: {result}"
                    successful_plans.append(error_message)
                    
                    # Write error to file to maintain file structure even on failure
                    scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i+1}")
                    os.makedirs(scene_dir, exist_ok=True)
                    error_file_path = os.path.join(scene_dir, f"{file_prefix}_scene{i+1}_implementation_plan.txt")
                    try:
                        with open(error_file_path, 'w') as f:
                            f.write(error_message)
                    except Exception as e:
                        print(f"❌ Failed to write error file for scene {i+1}: {e}")
                    
                    error_count += 1
                else:
                    successful_plans.append(result)
                    print(f"✅ Successfully generated implementation plan for scene {i+1}")
            
            total_time = time.time() - start_time
            print(f"All scene implementations completed in {total_time:.2f}s")
            print(f" Average time per scene: {total_time/len(tasks):.2f}s")
            print(f" Success rate: {len(tasks) - error_count}/{len(tasks)} scenes ({(len(tasks) - error_count) / len(tasks) * 100:.1f}%)")
            
            if error_count > 0:
                print(f"⚠️ Warning: {error_count} scenes had errors during implementation plan generation")
                
        except Exception as e:
            print(f"❌ Fatal error during scene implementation tasks: {e}")
            raise
        
        return successful_plans

    async def generate_scene_implementation_all_in_one(self, topic: str, description: str,
                                                       plan: str, session_id: str) -> List[str]:
        """
        Generate implementation plans for ALL scenes in a single LLM call, then
        parse and save per-scene outputs to mirror the separate-call behavior.
        Returns a list of per-scene implementation plan strings (vision + technical + narration concatenated).
        """
        start_time = time.time()
        scene_outline = extract_xml(plan)
        scene_count = len(re.findall(r'<SCENE_(\d+)>', scene_outline))
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())

        if scene_count == 0:
            print("❌ No scenes found in scene outline.")
            return []

        # If everything is already cached, skip model call
        if self.enable_caching:
            all_exist = True
            for i in range(1, scene_count + 1):
                combined_plan_path = os.path.join(self.output_dir, file_prefix, f"scene{i}", f"{file_prefix}_scene{i}_implementation_plan.txt")
                if not os.path.exists(combined_plan_path):
                    all_exist = False
                    break
            if all_exist:
                print("✅ Using cached implementation plans for all scenes")
                results: List[str] = []
                for i in range(1, scene_count + 1):
                    combined_plan_path = os.path.join(self.output_dir, file_prefix, f"scene{i}", f"{file_prefix}_scene{i}_implementation_plan.txt")
                    content = await self._async_file_read(combined_plan_path)
                    results.append(content or "")
                return results

        # Build a single prompt for all scenes
        # Construct a detailed template to steer output depth
        example_block = "\n".join([
            (
                f"    <SCENE_{i}>\n"
                f"        <SCENE_VISION_STORYBOARD_PLAN>\n"
                f"[SCENE_VISION]\n"
                f"- Scene overview, key takeaway, and visual learning objectives using specific Manim classes (MathTex, Tex, Axes, VGroup, Shapes).\n"
                f"- Explain how visuals support learning; adhere to safe area (0.5 units) and spacing (0.3 units).\n"
                f"\n[STORYBOARD]\n"
                f"- Visual Flow & Pacing: sequence of animations with types (Create, Write, FadeIn, Transform, etc.) and run_time.\n"
                f"- Sub-scene Breakdown (>= 3). For each sub-scene include:\n"
                f"  * Visual Element (primary object).\n"
                f"  * Animation steps (>= 4) with explicit run_time.\n"
                f"  * Relative positioning only (.next_to/.align_to/.shift) with buff >= 0.3; no absolute coordinates.\n"
                f"  * Color/style, and explicit notes on maintaining safe area and spacing.\n"
                f"        </SCENE_VISION_STORYBOARD_PLAN>\n"
                f"\n        <SCENE_TECHNICAL_IMPLEMENTATION_PLAN>\n"
                f"0. Dependencies: manim only (+ allowed plugins), no external assets.\n"
                f"1. Manim Object Selection & Configuration: list all objects with content, font sizes, colors, shape dims; Tex vs MathTex rules and LaTeX notes.\n"
                f"2. VGroup Structure & Hierarchy: parent-child groupings and internal spacing >= 0.3.\n"
                f"3. Spatial Positioning Strategy: reference objects/edges, methods, and buff values; checkpoints to avoid text overflow; font size guidance.\n"
                f"4. Animation Methods & Lifecycle: Create/Write/Transform/Uncreate/FadeIn/FadeOut; run_time, lag_ratio, Wait buffers.\n"
                f"5. Code Structure & Reusability: helper functions, construct flow, and comments referencing docs.\n"
                f"Mandatory Safety Checks: safe area 0.5, spacing 0.3, Wait() buffers.\n"
                f"        </SCENE_TECHNICAL_IMPLEMENTATION_PLAN>\n"
                f"\n        <SCENE_ANIMATION_NARRATION_PLAN>\n"
                f"[ANIMATION_STRATEGY]\n"
                f"- Pedagogical animation plan; VGroup transitions and element animations with run_time; explain learning purpose.\n"
                f"- Scene flow with pedagogical pacing; Wait() with durations and reasons.\n"
                f"\n[NARRATION]\n"
                f"- Full narration script with embedded animation timing cues; smooth continuity with previous/next scenes.\n"
                f"        </SCENE_ANIMATION_NARRATION_PLAN>\n"
                f"    </SCENE_{i}>\n"
            )
            for i in range(1, scene_count + 1)
        ])

        prompt = f"""
You are an expert in educational video production and Manim animation.
Task: Generate complete, highly detailed implementation plans for ALL scenes of '{topic}'.

Video Description: {description}
Scene Outline:
{scene_outline}

Global Rules (Apply to every scene):
- Safe area margins: 0.5 units on all sides; all objects must remain within.
- Minimum spacing: 0.3 units between all objects/VGroups (edge-to-edge).
- Positioning: only relative methods (.next_to/.align_to/.shift) with explicit buff values; no absolute coordinates.
- Text: MathTex for math; Tex for other text; use \\text{{}} inside MathTex if mixing.
- Every animation must include run_time; use Wait() buffers with a pedagogical reason.
- No external assets; only Manim (and explicitly allowed plugins if clearly justified).

Respond ONLY with the XML block below (no extra text before/after):
```xml
<SCENE_IMPLEMENTATION_PLANS>
{example_block}
</SCENE_IMPLEMENTATION_PLANS>
```
"""

        # Optionally include trimmed context-learning examples to anchor the expected depth
        if self.use_context_learning:
            def _trim(s: Optional[str], n: int = 1200) -> str:
                if not s:
                    return ""
                return s[:n]
            examples = []
            if self._context_cache.get('scene_vision_storyboard'):
                examples.append("Example storyboard detail:\n" + _trim(self._context_cache['scene_vision_storyboard']))
            if self._context_cache.get('technical_implementation'):
                examples.append("Example technical implementation detail:\n" + _trim(self._context_cache['technical_implementation']))
            if self._context_cache.get('scene_animation_narration'):
                examples.append("Example animation+narration detail:\n" + _trim(self._context_cache['scene_animation_narration']))
            if examples:
                prompt += "\n\nHere are brief examples to illustrate the expected level of detail (do not copy; adapt to this topic):\n" + "\n\n".join(examples)

        # Optionally add a small combined RAG context once
        if self.use_rag and self.rag_integration:
            # Generate queries once using the entire outline
            all_queries = await self._generate_combined_rag_queries(
                scene_outline, str(uuid.uuid4()), topic, 0, session_id
            )
            if all_queries:
                retrieved_docs = self.rag_integration.get_relevant_docs(
                    rag_queries=all_queries,
                    scene_trace_id=str(uuid.uuid4()),
                    topic=topic,
                    scene_number=0
                )
                prompt += f"\n\nHere is some context from relevant documentation:\n{retrieved_docs}"

        response = await self._call_model_async(
            self.planner_model,
            prompt,
            metadata={
                "generation_name": "scene_implementation_all_in_one",
                "tags": [topic, "all-scenes"],
                "session_id": session_id
            }
        )

        # Extract XML block
        xml_match = re.search(r'```xml\s*\n(.*?)\n```', response, re.DOTALL)
        if xml_match:
            response_xml = xml_match.group(1)
        else:
            # Fallback: try to find the wrapper or use whole response
            wrap_match = re.search(r'(<SCENE_IMPLEMENTATION_PLANS>.*?</SCENE_IMPLEMENTATION_PLANS>)', response, re.DOTALL)
            response_xml = wrap_match.group(1) if wrap_match else response

        # For each scene, extract and save
        def _extract_inner(tag: str, text: str) -> str:
            m = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
            return m.group(1).strip() if m else f"<{tag}>Plan not generated.</{tag}>"

        per_scene_plans: List[str] = []
        for i in range(1, scene_count + 1):
            scene_regex = r'(<SCENE_{0}>.*?</SCENE_{0}>)'.format(i)
            scene_match = re.search(scene_regex, response_xml, re.DOTALL)
            if not scene_match:
                print(f"❌ Error: Combined response missing SCENE_{i} block")
                per_scene_plans.append(f"# Scene {i} - Error: Missing scene block in combined response")
                continue

            scene_block = scene_match.group(1)
            vision_storyboard_content = _extract_inner('SCENE_VISION_STORYBOARD_PLAN', scene_block)
            technical_implementation_content = _extract_inner('SCENE_TECHNICAL_IMPLEMENTATION_PLAN', scene_block)
            animation_narration_content = _extract_inner('SCENE_ANIMATION_NARRATION_PLAN', scene_block)

            scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i}")
            subplan_dir = os.path.join(scene_dir, "subplans")
            await self._ensure_directories(scene_dir, subplan_dir)

            # Save trace id for each scene
            scene_trace_id = str(uuid.uuid4())
            await self._async_file_write(os.path.join(subplan_dir, "scene_trace_id.txt"), scene_trace_id)

            # Save individual subplans
            await self._async_file_write(os.path.join(subplan_dir, f"{file_prefix}_scene{i}_vision_storyboard_plan.txt"), vision_storyboard_content)
            await self._async_file_write(os.path.join(subplan_dir, f"{file_prefix}_scene{i}_technical_implementation_plan.txt"), technical_implementation_content)
            await self._async_file_write(os.path.join(subplan_dir, f"{file_prefix}_scene{i}_animation_narration_plan.txt"), animation_narration_content)

            # Save combined per-scene plan
            implementation_plan = (
                f"{vision_storyboard_content}\n\n"
                f"{technical_implementation_content}\n\n"
                f"{animation_narration_content}\n\n"
            )
            combined_plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{i}_implementation_plan.txt")
            combined_content = f"# Scene {i} Implementation Plan\n\n{implementation_plan}"
            await self._async_file_write(combined_plan_path, combined_content)

            per_scene_plans.append(implementation_plan)

        elapsed = time.time() - start_time
        print(f"✅ All-in-one implementation generation completed in {elapsed:.2f}s for {scene_count} scenes")
        return per_scene_plans

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        self.thread_pool.shutdown(wait=True)

    # Legacy method compatibility
    async def generate_scene_implementation_concurrently(self, topic: str, description: str, 
                                                       plan: str, session_id: str, 
                                                       scene_semaphore=None) -> List[str]:
        """Legacy compatibility method - redirects to enhanced version."""
        if scene_semaphore:
            self.scene_semaphore = scene_semaphore
        return await self.generate_scene_implementation_concurrently_enhanced(
            topic, description, plan, session_id
        )

    def _extract_scene_outline_robust(self, response_text: str) -> str:
        """
        Robust extraction of scene outline that handles various XML format issues.
        
        This method addresses common problems:
        1. XML wrapped in markdown code blocks
        2. Missing closing tags
        3. Malformed XML structure
        4. Extra text before/after XML
        """
        import re
        
        # First try: Look for XML wrapped in markdown code blocks
        markdown_xml_pattern = r'```xml\s*\n(<SCENE_OUTLINE>.*?</SCENE_OUTLINE>)\s*\n```'
        markdown_match = re.search(markdown_xml_pattern, response_text, re.DOTALL)
        if markdown_match:
            xml_content = markdown_match.group(1)
            return self._validate_and_fix_xml(xml_content)
        
        # Second try: Look for direct XML tags
        direct_xml_pattern = r'(<SCENE_OUTLINE>.*?</SCENE_OUTLINE>)'
        direct_match = re.search(direct_xml_pattern, response_text, re.DOTALL)
        if direct_match:
            xml_content = direct_match.group(1)
            return self._validate_and_fix_xml(xml_content)
        
        # Third try: Look for incomplete XML and attempt to fix
        incomplete_pattern = r'<SCENE_OUTLINE>(.*?)(?:</SCENE_OUTLINE>|$)'
        incomplete_match = re.search(incomplete_pattern, response_text, re.DOTALL)
        if incomplete_match:
            xml_content = incomplete_match.group(1)
            # Add missing closing tag if needed
            full_xml = f"<SCENE_OUTLINE>{xml_content}</SCENE_OUTLINE>"
            return self._validate_and_fix_xml(full_xml)
        
        # If no XML structure found, return the entire response but warn
        print("⚠️ Warning: No valid XML structure found in LLM response. Using full response.")
        print("Response preview:", response_text[:200] + "..." if len(response_text) > 200 else response_text)
        return response_text
    
    def _validate_and_fix_xml(self, xml_content: str) -> str:
        """
        Validate and fix common XML issues in scene outlines.
        """
        import re
        
        # Check for unclosed scene tags
        scene_pattern = r'<SCENE_(\d+)>'
        scene_matches = re.findall(scene_pattern, xml_content)
        
        fixed_content = xml_content
        
        for scene_num in scene_matches:
            # Check if this scene has a proper closing tag
            open_tag = f"<SCENE_{scene_num}>"
            close_tag = f"</SCENE_{scene_num}>"
            
            # Find the position of this scene's opening tag
            open_pos = fixed_content.find(open_tag)
            if open_pos == -1:
                continue
                
            # Find the next scene's opening tag (if any)
            next_scene_pattern = f"<SCENE_{int(scene_num) + 1}>"
            next_scene_pos = fixed_content.find(next_scene_pattern, open_pos)
            
            # Check if there's a closing tag before the next scene
            close_pos = fixed_content.find(close_tag, open_pos)
            
            if close_pos == -1 or (next_scene_pos != -1 and close_pos > next_scene_pos):
                # Missing or misplaced closing tag
                if next_scene_pos != -1:
                    # Insert closing tag before next scene
                    insert_pos = next_scene_pos
                    while insert_pos > 0 and fixed_content[insert_pos - 1] in ' \n\t':
                        insert_pos -= 1
                    fixed_content = (fixed_content[:insert_pos] + 
                                   f"\n    {close_tag}\n\n    " + 
                                   fixed_content[insert_pos:])
                else:
                    # Insert closing tag at the end
                    end_outline_pos = fixed_content.find("</SCENE_OUTLINE>")
                    if end_outline_pos != -1:
                        fixed_content = (fixed_content[:end_outline_pos] + 
                                       f"\n    {close_tag}\n" + 
                                       fixed_content[end_outline_pos:])
                    else:
                        fixed_content += f"\n    {close_tag}"
                
                print(f"🔧 Fixed missing closing tag for SCENE_{scene_num}")
        
        # Ensure proper SCENE_OUTLINE structure
        if not fixed_content.strip().startswith("<SCENE_OUTLINE>"):
            fixed_content = f"<SCENE_OUTLINE>\n{fixed_content}"
        
        if not fixed_content.strip().endswith("</SCENE_OUTLINE>"):
            fixed_content = f"{fixed_content}\n</SCENE_OUTLINE>"
        
        return fixed_content

# Update class alias for backward compatibility
VideoPlanner = EnhancedVideoPlanner
