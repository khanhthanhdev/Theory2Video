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
                 max_scene_concurrency=5, max_step_concurrency=3, enable_caching=True):
        
        self.planner_model = planner_model
        self.helper_model = helper_model if helper_model is not None else planner_model
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_context_learning = use_context_learning
        self.context_learning_path = context_learning_path
        self.use_rag = use_rag
        self.session_id = session_id
        self.enable_caching = enable_caching
        
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

    async def _async_file_read(self, file_path: str) -> str:
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
        response_text = self.planner_model(
            _prepare_text_inputs(prompt),
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
            
            # Add context examples if available
            example_type = step_name.replace('_plan', '').replace('scene_', '')
            if self._context_cache.get(example_type):
                prompt += f"\n\nHere are some example {step_name}s:\n{self._context_cache[example_type]}"
            
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
            response = self.planner_model(
                _prepare_text_inputs(prompt),
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

    async def _generate_scene_implementation_single_enhanced(self, topic: str, description: str, 
                                                           scene_outline_i: str, scene_number: int, 
                                                           file_prefix: str, session_id: str, 
                                                           scene_trace_id: str) -> str:
        """Enhanced single scene implementation with parallel steps."""
        start_time = time.time()
        print(f"Starting scene {scene_number} implementation (parallel processing)")
        
        # Setup directories
        scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_number}")
        subplan_dir = os.path.join(scene_dir, "subplans")
        await self._ensure_directories(scene_dir, subplan_dir)

        # Save scene trace ID
        trace_id_file = os.path.join(subplan_dir, "scene_trace_id.txt")
        await self._async_file_write(trace_id_file, scene_trace_id)

        # Define all steps with their configurations
        steps_config = [
            {
                'name': 'scene_vision_storyboard',
                'prompt_func': get_prompt_scene_vision_storyboard,
                'args': (scene_number, topic, description, scene_outline_i, self.relevant_plugins),
                'output_path': os.path.join(subplan_dir, f"{file_prefix}_scene{scene_number}_vision_storyboard_plan.txt")
            }
        ]

        # Execute Step 1: Vision Storyboard (sequential dependency)
        vision_storyboard_content, _ = await self._generate_scene_step_parallel(
            steps_config[0]['name'],
            steps_config[0]['prompt_func'],
            scene_trace_id,
            topic,
            scene_number,
            session_id,
            steps_config[0]['output_path'],
            *steps_config[0]['args']
        )

        # Prepare Step 2 and 3 for parallel execution (both depend on Step 1)
        remaining_steps = [
            {
                'name': 'scene_technical_implementation',
                'prompt_func': get_prompt_scene_technical_implementation,
                'args': (scene_number, topic, description, scene_outline_i, vision_storyboard_content, self.relevant_plugins),
                'output_path': os.path.join(subplan_dir, f"{file_prefix}_scene{scene_number}_technical_implementation_plan.txt")
            },
            {
                'name': 'scene_animation_narration',
                'prompt_func': get_prompt_scene_animation_narration,
                'args': (scene_number, topic, description, scene_outline_i, vision_storyboard_content, None, self.relevant_plugins),
                'output_path': os.path.join(subplan_dir, f"{file_prefix}_scene{scene_number}_animation_narration_plan.txt")
            }
        ]

        # Execute Steps 2 and 3 in parallel
        parallel_tasks = []
        for step_config in remaining_steps:
            task = asyncio.create_task(
                self._generate_scene_step_parallel(
                    step_config['name'],
                    step_config['prompt_func'],
                    scene_trace_id,
                    topic,
                    scene_number,
                    session_id,
                    step_config['output_path'],
                    *step_config['args']
                )
            )
            parallel_tasks.append(task)

        # Wait for parallel tasks to complete
        parallel_results = await asyncio.gather(*parallel_tasks)
        technical_implementation_content = parallel_results[0][0]
        animation_narration_content = parallel_results[1][0]

        # Update animation narration args with technical implementation and regenerate if needed
        if technical_implementation_content:
            updated_animation_args = (
                scene_number, topic, description, scene_outline_i, 
                vision_storyboard_content, technical_implementation_content, self.relevant_plugins
            )
            
            animation_narration_content, _ = await self._generate_scene_step_parallel(
                'scene_animation_narration',
                get_prompt_scene_animation_narration,
                scene_trace_id,
                topic,
                scene_number,
                session_id,
                remaining_steps[1]['output_path'],
                *updated_animation_args
            )

        # Combine all implementation plans
        implementation_plan = (
            f"{vision_storyboard_content}\n\n"
            f"{technical_implementation_content}\n\n"
            f"{animation_narration_content}\n\n"
        )

        # Ensure scene directory exists (just to be extra safe)
        scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{scene_number}")
        await self._ensure_directories(scene_dir)
        
        # Save combined implementation plan
        combined_plan_path = os.path.join(scene_dir, f"{file_prefix}_scene{scene_number}_implementation_plan.txt")
        combined_content = f"# Scene {scene_number} Implementation Plan\n\n{implementation_plan}"
        
        try:
            await self._async_file_write(combined_plan_path, combined_content)
            print(f"✅ Saved implementation plan for scene {scene_number} to: {combined_plan_path}")
        except Exception as e:
            print(f"❌ Error saving implementation plan for scene {scene_number}: {e}")
            raise

        elapsed_time = time.time() - start_time
        print(f"Scene {scene_number} implementation completed in {elapsed_time:.2f}s")

        return implementation_plan

    async def generate_scene_implementation_concurrently_enhanced(self, topic: str, description: str, 
                                                                plan: str, session_id: str) -> List[str]:
        """Enhanced concurrent scene implementation with better performance."""
        start_time = time.time()
        
        # Extract scene information
        scene_outline = extract_xml(plan)
        scene_number = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline))
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