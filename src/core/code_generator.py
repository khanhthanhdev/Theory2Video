import os
import re
import json
import logging
import glob
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
from PIL import Image

from src.utils.utils import extract_json
from mllm_tools.utils import _prepare_text_inputs, _extract_code, _prepare_text_image_inputs
from mllm_tools.gemini import GeminiWrapper
from mllm_tools.vertex_ai import VertexAIWrapper
from task_generator import (
    get_prompt_code_generation,
    get_prompt_fix_error,
    get_prompt_visual_fix_error,
    get_banned_reasonings,
    get_prompt_rag_query_generation_fix_error,
    get_prompt_context_learning_code,
    get_prompt_rag_query_generation_code
)
from task_generator.prompts_raw import (
    _code_font_size,
    _code_disable,
    _code_limit,
    _prompt_manim_cheatsheet
)
from src.rag.vector_store import RAGVectorStore

# Configuration constants
DEFAULT_MAX_RETRIES = 10
DEFAULT_RAG_K_VALUE = 2
CACHE_FILE_ENCODING = 'utf-8'
CODE_PATTERN = r"```python(.*)```"
JSON_PATTERN = r'```json(.*)```'

# Set up logging
logger = logging.getLogger(__name__)

class CodeGenerator:
    """A class for generating and managing Manim code with improved error handling and maintainability."""

    def __init__(
        self, 
        scene_model: Any, 
        helper_model: Any, 
        output_dir: str = "output", 
        print_response: bool = False, 
        use_rag: bool = False, 
        use_context_learning: bool = False, 
        context_learning_path: str = "data/context_learning", 
        chroma_db_path: str = "rag/chroma_db", 
        manim_docs_path: str = "rag/manim_docs", 
        embedding_model: str = "azure/text-embedding-3-large", 
        use_visual_fix_code: bool = False, 
        use_langfuse: bool = True, 
        session_id: Optional[str] = None
    ) -> None:
        """Initialize the CodeGenerator.

        Args:
            scene_model: The model used for scene generation
            helper_model: The model used for helper tasks
            output_dir (str, optional): Directory for output files. Defaults to "output".
            print_response (bool, optional): Whether to print model responses. Defaults to False.
            use_rag (bool, optional): Whether to use RAG. Defaults to False.
            use_context_learning (bool, optional): Whether to use context learning. Defaults to False.
            context_learning_path (str, optional): Path to context learning examples. Defaults to "data/context_learning".
            chroma_db_path (str, optional): Path to ChromaDB. Defaults to "rag/chroma_db".
            manim_docs_path (str, optional): Path to Manim docs. Defaults to "rag/manim_docs".
            embedding_model (str, optional): Name of embedding model. Defaults to "azure/text-embedding-3-large".
            use_visual_fix_code (bool, optional): Whether to use visual code fixing. Defaults to False.
            use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True.
            session_id (str, optional): Session identifier. Defaults to None.
        """
        self.scene_model = scene_model
        self.helper_model = helper_model
        self.output_dir = Path(output_dir)
        self.print_response = print_response
        self.use_rag = use_rag
        self.use_context_learning = use_context_learning
        self.context_learning_path = Path(context_learning_path)
        self.manim_docs_path = Path(manim_docs_path)
        self.use_visual_fix_code = use_visual_fix_code
        self.session_id = session_id
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load context examples and banned reasonings
        self.context_examples = self._load_context_examples() if use_context_learning else None
        self.banned_reasonings = self._load_banned_reasonings()
        
        # Initialize RAG vector store if enabled
        self.vector_store = self._initialize_vector_store(
            chroma_db_path, embedding_model, use_langfuse
        ) if use_rag else None
        
        logger.info(f"CodeGenerator initialized with RAG: {use_rag}, Context Learning: {use_context_learning}")

    def _load_banned_reasonings(self) -> List[str]:
        """Load banned reasonings with error handling."""
        try:
            return get_banned_reasonings()
        except Exception as e:
            logger.warning(f"Failed to load banned reasonings: {e}")
            return []

    def _initialize_vector_store(self, chroma_db_path: str, embedding_model: str, use_langfuse: bool) -> Optional[RAGVectorStore]:
        """Initialize RAG vector store with error handling."""
        try:
            return RAGVectorStore(
                chroma_db_path=chroma_db_path,
                manim_docs_path=str(self.manim_docs_path),
                embedding_model=embedding_model,
                session_id=self.session_id,
                use_langfuse=use_langfuse
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG vector store: {e}")
            return None

    def _load_context_examples(self) -> Optional[str]:
        """Load all context learning examples from the specified directory.

        Returns:
            Optional[str]: Formatted context learning examples, or None if no examples found.
        """
        if not self.context_learning_path.exists():
            logger.warning(f"Context learning path does not exist: {self.context_learning_path}")
            return None
            
        examples = []
        pattern = str(self.context_learning_path / "**" / "*.py")
        
        try:
            for example_file in glob.glob(pattern, recursive=True):
                example_path = Path(example_file)
                try:
                    with example_path.open('r', encoding=CACHE_FILE_ENCODING) as f:
                        content = f.read()
                        examples.append(f"# Example from {example_path.name}\n{content}\n")
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to read example file {example_file}: {e}")
                    continue

            if examples:
                formatted_examples = get_prompt_context_learning_code(
                    examples="\n".join(examples)
                )
                logger.info(f"Loaded {len(examples)} context learning examples")
                return formatted_examples
                
        except Exception as e:
            logger.error(f"Error loading context examples: {e}")
            
        return None

    def _create_cache_directory(self, topic: str, scene_number: int, cache_type: str = "rag_cache") -> Path:
        """Create and return cache directory path."""
        sanitized_topic = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        cache_dir = self.output_dir / sanitized_topic / f"scene{scene_number}" / cache_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _load_cached_queries(self, cache_file: Path) -> Optional[List[str]]:
        """Load cached queries from file with error handling."""
        if not cache_file.exists():
            return None
            
        try:
            with cache_file.open('r', encoding=CACHE_FILE_ENCODING) as f:
                cached_queries = json.load(f)
                logger.debug(f"Loaded cached queries from {cache_file}")
                return cached_queries
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cached queries from {cache_file}: {e}")
            return None

    def _save_queries_to_cache(self, queries: List[str], cache_file: Path) -> None:
        """Save queries to cache file with error handling."""
        try:
            with cache_file.open('w', encoding=CACHE_FILE_ENCODING) as f:
                json.dump(queries, f, indent=2)
                logger.debug(f"Saved queries to cache: {cache_file}")
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save queries to cache {cache_file}: {e}")

    def _extract_json_from_response(self, response: str, error_context: str = "") -> List[str]:
        """Extract and parse JSON from model response with improved error handling."""
        # Try to extract JSON from code blocks first
        json_match = re.search(JSON_PATTERN, response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Fallback: clean the response and try direct parsing
            json_text = response.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError when parsing {error_context}: {e}")
            logger.error(f"Response text was: {response[:500]}...")
            return []

    def _generate_rag_queries_code(
        self, 
        implementation: str, 
        scene_trace_id: Optional[str] = None, 
        topic: Optional[str] = None, 
        scene_number: Optional[int] = None, 
        session_id: Optional[str] = None, 
        relevant_plugins: List[str] = None
    ) -> List[str]:
        """Generate RAG queries from the implementation plan.

        Args:
            implementation: The implementation plan text
            scene_trace_id: Trace ID for the scene
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier
            relevant_plugins: List of relevant plugins

        Returns:
            List of generated RAG queries
        """
        if relevant_plugins is None:
            relevant_plugins = []
            
        if not topic or scene_number is None:
            logger.warning("Missing topic or scene_number for RAG query generation")
            return []

        # Setup cache
        cache_dir = self._create_cache_directory(topic, scene_number)
        cache_file = cache_dir / "rag_queries_code.json"

        # Try to load from cache
        cached_queries = self._load_cached_queries(cache_file)
        if cached_queries is not None:
            logger.info(f"Using cached RAG queries for {topic}_scene{scene_number}")
            return cached_queries

        # Generate new queries
        try:
            plugins_text = ", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
            prompt = get_prompt_rag_query_generation_code(implementation, plugins_text)

            response = self.helper_model(
                _prepare_text_inputs(prompt),
                metadata={
                    "generation_name": "rag_query_generation", 
                    "trace_id": scene_trace_id, 
                    "tags": [topic, f"scene{scene_number}"], 
                    "session_id": session_id
                }
            )

            logger.debug(f"RAG queries response: {response[:200]}...")
            queries = self._extract_json_from_response(response, "RAG queries for code generation")
            
            # Cache the queries
            if queries:
                self._save_queries_to_cache(queries, cache_file)
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating RAG queries for code: {e}")
            return []

    def _generate_rag_queries_error_fix(
        self, 
        error: str, 
        code: str, 
        scene_trace_id: Optional[str] = None, 
        topic: Optional[str] = None, 
        scene_number: Optional[int] = None, 
        session_id: Optional[str] = None, 
        relevant_plugins: List[str] = None
    ) -> List[str]:
        """Generate RAG queries for fixing code errors.

        Args:
            error: The error message to fix
            code: The code containing the error
            scene_trace_id: Trace ID for the scene
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier
            relevant_plugins: List of relevant plugins

        Returns:
            List of generated RAG queries for error fixing
        """
        if relevant_plugins is None:
            relevant_plugins = []
            
        if not topic or scene_number is None:
            logger.warning("Missing topic or scene_number for RAG error fix query generation")
            return []

        # Setup cache
        cache_dir = self._create_cache_directory(topic, scene_number)
        cache_file = cache_dir / "rag_queries_error_fix.json"

        # Try to load from cache
        cached_queries = self._load_cached_queries(cache_file)
        if cached_queries is not None:
            logger.info(f"Using cached RAG error fix queries for {topic}_scene{scene_number}")
            return cached_queries

        # Generate new queries for error fix
        try:
            plugins_text = ", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
            prompt = get_prompt_rag_query_generation_fix_error(
                error=error,
                code=code,
                relevant_plugins=plugins_text
            )

            response = self.helper_model(
                _prepare_text_inputs(prompt),
                metadata={
                    "generation_name": "rag-query-generation-fix-error", 
                    "trace_id": scene_trace_id, 
                    "tags": [topic, f"scene{scene_number}"], 
                    "session_id": session_id
                }
            )

            queries = self._extract_json_from_response(response, "RAG queries for error fix")
            
            # Cache the queries
            if queries:
                self._save_queries_to_cache(queries, cache_file)
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating RAG queries for error fix: {e}")
            return []

    def _extract_code_with_retries(
        self, 
        response_text: str, 
        pattern: str = CODE_PATTERN, 
        generation_name: Optional[str] = None, 
        trace_id: Optional[str] = None, 
        session_id: Optional[str] = None, 
        max_retries: int = DEFAULT_MAX_RETRIES
    ) -> str:
        """Extract code from response text with retry logic.

        Args:
            response_text: The text containing code to extract
            pattern: Regex pattern for extracting code
            generation_name: Name of generation step
            trace_id: Trace identifier
            session_id: Session identifier
            max_retries: Maximum number of retries

        Returns:
            The extracted code

        Raises:
            ValueError: If code extraction fails after max retries
        """
        retry_prompt_template = """
        Please extract the Python code in the correct format using the pattern: {pattern}. 
        You MUST NOT include any other text or comments. 
        You MUST return the exact same code as in the previous response, NO CONTENT EDITING is allowed.
        Previous response: 
        {response_text}
        """

        for attempt in range(max_retries):
            try:
                code_match = re.search(pattern, response_text, re.DOTALL)
                if code_match:
                    extracted_code = code_match.group(1).strip()
                    logger.debug(f"Successfully extracted code on attempt {attempt + 1}")
                    return extracted_code
                
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}: Failed to extract code pattern. Retrying...")
                    
                    # Regenerate response with a more explicit prompt
                    retry_prompt = retry_prompt_template.format(
                        pattern=pattern, 
                        response_text=response_text[:1000]  # Limit response length
                    )
                    
                    response_text = self.scene_model(
                        _prepare_text_inputs(retry_prompt),
                        metadata={
                            "generation_name": f"{generation_name}_format_retry_{attempt + 1}",
                            "trace_id": trace_id,
                            "session_id": session_id
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Error during code extraction attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break
        
        raise ValueError(f"Failed to extract code pattern after {max_retries} attempts. Pattern: {pattern}")

    def _prepare_additional_context(self, additional_context: Union[str, List[str], None]) -> List[str]:
        """Prepare additional context for code generation."""
        if additional_context is None:
            return []
        elif isinstance(additional_context, str):
            return [additional_context]
        return additional_context.copy()

    def _retrieve_rag_context(
        self, 
        rag_queries: List[str], 
        scene_trace_id: Optional[str], 
        topic: str, 
        scene_number: int
    ) -> Optional[str]:
        """Retrieve context from RAG vector store."""
        if not self.vector_store or not rag_queries:
            return None
            
        try:
            return self.vector_store.find_relevant_docs(
                queries=rag_queries,
                k=DEFAULT_RAG_K_VALUE,
                trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number
            )
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return None

    def generate_manim_code(
        self,
        topic: str,
        description: str,                            
        scene_outline: str,
        scene_implementation: str,
        scene_number: int,
        additional_context: Union[str, List[str], None] = None,
        scene_trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        rag_queries_cache: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """Generate Manim code from video plan.

        Args:
            topic: Topic of the scene
            description: Description of the scene
            scene_outline: Outline of the scene
            scene_implementation: Implementation details
            scene_number: Scene number
            additional_context: Additional context
            scene_trace_id: Trace identifier
            session_id: Session identifier
            rag_queries_cache: Cache for RAG queries (deprecated, use file cache)

        Returns:
            Tuple of generated code and response text

        Raises:
            ValueError: If code generation fails
        """
        try:
            # Prepare additional context
            context_list = self._prepare_additional_context(additional_context)

            # Add context learning examples if enabled
            if self.use_context_learning and self.context_examples:
                context_list.append(self.context_examples)

            # Add RAG context if enabled
            if self.use_rag:
                rag_queries = self._generate_rag_queries_code(
                    implementation=scene_implementation,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id or self.session_id
                )

                rag_context = self._retrieve_rag_context(
                    rag_queries, scene_trace_id, topic, scene_number
                )
                
                if rag_context:
                    context_list.append(rag_context)

            # Generate prompt
            prompt = get_prompt_code_generation(
                scene_outline=scene_outline,
                scene_implementation=scene_implementation,
                topic=topic,
                description=description,
                scene_number=scene_number,
                additional_context=context_list if context_list else None
            )

            # Generate code using model
            response_text = self.scene_model(
                _prepare_text_inputs(prompt),
                metadata={
                    "generation_name": "code_generation", 
                    "trace_id": scene_trace_id, 
                    "tags": [topic, f"scene{scene_number}"], 
                    "session_id": session_id or self.session_id
                }
            )

            # Extract code with retries
            code = self._extract_code_with_retries(
                response_text,
                CODE_PATTERN,
                generation_name="code_generation",
                trace_id=scene_trace_id,
                session_id=session_id or self.session_id
            )
            
            logger.info(f"Successfully generated code for {topic} scene {scene_number}")
            return code, response_text
            
        except Exception as e:
            logger.error(f"Error generating Manim code for {topic} scene {scene_number}: {e}")
            raise ValueError(f"Code generation failed: {e}") from e

    def fix_code_errors(
        self, 
        implementation_plan: str, 
        code: str, 
        error: str, 
        scene_trace_id: str, 
        topic: str, 
        scene_number: int, 
        session_id: str, 
        rag_queries_cache: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """Fix errors in generated Manim code.

        Args:
            implementation_plan: Original implementation plan
            code: Code containing errors
            error: Error message to fix
            scene_trace_id: Trace identifier
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier
            rag_queries_cache: Cache for RAG queries (deprecated, use file cache)

        Returns:
            Tuple of fixed code and response text

        Raises:
            ValueError: If code fixing fails
        """
        try:
            # Start with base error fix prompt
            additional_context = None
            
            # Add RAG context if enabled
            if self.use_rag:
                rag_queries = self._generate_rag_queries_error_fix(
                    error=error,
                    code=code,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id
                )
                
                rag_context = self._retrieve_rag_context(
                    rag_queries, scene_trace_id, topic, scene_number
                )
                
                if rag_context:
                    additional_context = rag_context

            # Generate prompt (with or without RAG context)
            if additional_context:
                prompt = get_prompt_fix_error(
                    implementation_plan=implementation_plan, 
                    manim_code=code, 
                    error=error, 
                    additional_context=additional_context
                )
            else:
                prompt = get_prompt_fix_error(
                    implementation_plan=implementation_plan, 
                    manim_code=code, 
                    error=error
                )

            # Get fixed code from model
            response_text = self.scene_model(
                _prepare_text_inputs(prompt),
                metadata={
                    "generation_name": "code_fix_error", 
                    "trace_id": scene_trace_id, 
                    "tags": [topic, f"scene{scene_number}"], 
                    "session_id": session_id
                }
            )

            # Extract fixed code with retries
            fixed_code = self._extract_code_with_retries(
                response_text,
                CODE_PATTERN,
                generation_name="code_fix_error",
                trace_id=scene_trace_id,
                session_id=session_id
            )
            
            logger.info(f"Successfully fixed code errors for {topic} scene {scene_number}")
            return fixed_code, response_text
            
        except Exception as e:
            logger.error(f"Error fixing code for {topic} scene {scene_number}: {e}")
            raise ValueError(f"Code error fixing failed: {e}") from e

    def visual_self_reflection(
        self, 
        code: str, 
        media_path: Union[str, Image.Image], 
        scene_trace_id: str, 
        topic: str, 
        scene_number: int, 
        session_id: str
    ) -> Tuple[str, str]:
        """Use snapshot image or mp4 video to fix code.

        Args:
            code: Code to fix
            media_path: Path to media file or PIL Image
            scene_trace_id: Trace identifier
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier

        Returns:
            Tuple of fixed code and response text

        Raises:
            ValueError: If visual self-reflection fails
            FileNotFoundError: If media file doesn't exist
        """
        try:
            # Validate media input
            if isinstance(media_path, str):
                media_file = Path(media_path)
                if not media_file.exists():
                    raise FileNotFoundError(f"Media file not found: {media_path}")
            
            # Determine if we're dealing with video or image
            is_video = isinstance(media_path, str) and media_path.lower().endswith('.mp4')
            
            # Load prompt template
            prompt_file = Path('task_generator/prompts_raw/prompt_visual_self_reflection.txt')
            if not prompt_file.exists():
                logger.warning(f"Visual self-reflection prompt file not found: {prompt_file}")
                # Fallback prompt
                prompt_template = """
                Analyze the visual output and the provided code. Fix any issues you notice in the code.
                
                Code:
                {code}
                """
            else:
                with prompt_file.open('r', encoding=CACHE_FILE_ENCODING) as f:
                    prompt_template = f.read()
            
            # Format prompt
            prompt = prompt_template.format(code=code)
            
            # Prepare input based on media type and model capabilities
            if is_video and isinstance(self.scene_model, (GeminiWrapper, VertexAIWrapper)):
                # For video with Gemini models
                messages = [
                    {"type": "text", "content": prompt},
                    {"type": "video", "content": str(media_path)}
                ]
            else:
                # For images or non-Gemini models
                if isinstance(media_path, str):
                    media = Image.open(media_path)
                else:
                    media = media_path
                messages = [
                    {"type": "text", "content": prompt},
                    {"type": "image", "content": media}
                ]
            
            # Get model response
            response_text = self.scene_model(
                messages,
                metadata={
                    "generation_name": "visual_self_reflection",
                    "trace_id": scene_trace_id,
                    "tags": [topic, f"scene{scene_number}"],
                    "session_id": session_id
                }
            )
            
            # Extract code with retries
            fixed_code = self._extract_code_with_retries(
                response_text,
                CODE_PATTERN,
                generation_name="visual_self_reflection",
                trace_id=scene_trace_id,
                session_id=session_id
            )
            
            logger.info(f"Successfully completed visual self-reflection for {topic} scene {scene_number}")
            return fixed_code, response_text
            
        except Exception as e:
            logger.error(f"Error in visual self-reflection for {topic} scene {scene_number}: {e}")
            raise ValueError(f"Visual self-reflection failed: {e}") from e

    def enhanced_visual_self_reflection(
        self, 
        code: str, 
        media_path: Union[str, Image.Image], 
        scene_trace_id: str, 
        topic: str, 
        scene_number: int, 
        session_id: str,
        implementation_plan: Optional[str] = None
    ) -> Tuple[str, str]:
        """Enhanced visual self-reflection using VLM for detailed error detection.

        This method specifically focuses on detecting and fixing:
        - Element overlap and collision
        - Out-of-bounds positioning
        - Spatial boundary violations
        - Poor visual arrangement
        - Educational effectiveness issues

        Args:
            code: Code to analyze and fix
            media_path: Path to media file or PIL Image
            scene_trace_id: Trace identifier
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier
            implementation_plan: Optional implementation plan for context

        Returns:
            Tuple of fixed code and response text

        Raises:
            ValueError: If enhanced visual analysis fails
            FileNotFoundError: If media file doesn't exist
        """
        try:
            # Validate media input
            if isinstance(media_path, str):
                media_file = Path(media_path)
                if not media_file.exists():
                    raise FileNotFoundError(f"Media file not found: {media_path}")
            
            # Determine if we're dealing with video or image
            is_video = isinstance(media_path, str) and media_path.lower().endswith('.mp4')
            
            # Load enhanced visual analysis prompt
            enhanced_prompt_file = Path('task_generator/prompts_raw/prompt_enhanced_visual_self_reflection.txt')
            if enhanced_prompt_file.exists():
                with enhanced_prompt_file.open('r', encoding=CACHE_FILE_ENCODING) as f:
                    prompt_template = f.read()
            else:
                # Fallback to original prompt if enhanced version not found
                logger.warning("Enhanced visual self-reflection prompt not found, using fallback")
                prompt_template = self._get_fallback_visual_prompt()
            
            # Format prompt with implementation plan and code
            prompt = prompt_template.format(
                implementation=implementation_plan or "No implementation plan provided",
                code=code
            )
            
            # Prepare input based on media type and model capabilities
            if is_video and isinstance(self.scene_model, (GeminiWrapper, VertexAIWrapper)):
                # For video with Gemini/Vertex AI models
                messages = [
                    {"type": "text", "content": prompt},
                    {"type": "video", "content": str(media_path)}
                ]
            else:
                # For images or non-Gemini models
                if isinstance(media_path, str):
                    media = Image.open(media_path)
                else:
                    media = media_path
                messages = [
                    {"type": "text", "content": prompt},
                    {"type": "image", "content": media}
                ]
            
            # Get enhanced VLM analysis response
            response_text = self.scene_model(
                messages,
                metadata={
                    "generation_name": "enhanced_visual_self_reflection",
                    "trace_id": scene_trace_id,
                    "tags": [topic, f"scene{scene_number}", "visual_error_detection"],
                    "session_id": session_id
                }
            )
            
            # Parse response for visual analysis results
            if "<LGTM>" in response_text or response_text.strip() == "<LGTM>":
                logger.info(f"Enhanced visual analysis passed for {topic} scene {scene_number}")
                return code, response_text
            
            # Extract improved code if visual issues were found
            fixed_code = self._extract_visual_fix_code(response_text, scene_trace_id, session_id)
            
            logger.info(f"Enhanced visual self-reflection completed with fixes for {topic} scene {scene_number}")
            return fixed_code, response_text
            
        except Exception as e:
            logger.error(f"Error in enhanced visual self-reflection for {topic} scene {scene_number}: {e}")
            # Fallback to original visual_self_reflection if enhanced version fails
            logger.info("Falling back to original visual_self_reflection method")
            return self.visual_self_reflection(
                code, media_path, scene_trace_id, topic, scene_number, session_id
            )

    def _extract_visual_fix_code(
        self, 
        response_text: str, 
        scene_trace_id: Optional[str] = None, 
        session_id: Optional[str] = None
    ) -> str:
        """Extract code from enhanced visual analysis response.

        Args:
            response_text: The VLM response containing visual analysis
            scene_trace_id: Trace identifier
            session_id: Session identifier

        Returns:
            The extracted and fixed code

        Raises:
            ValueError: If code extraction fails
        """
        # Try to extract code from <improved_code> tags first
        improved_code_pattern = r'<improved_code>\s*```python\s*(.*?)\s*```\s*</improved_code>'
        code_match = re.search(improved_code_pattern, response_text, re.DOTALL)
        
        if code_match:
            extracted_code = code_match.group(1).strip()
            logger.debug("Successfully extracted code from <improved_code> tags")
            return extracted_code
        
        # Fallback to standard code extraction
        return self._extract_code_with_retries(
            response_text,
            CODE_PATTERN,
            generation_name="enhanced_visual_fix",
            trace_id=scene_trace_id,
            session_id=session_id
        )

    def _get_fallback_visual_prompt(self) -> str:
        """Get fallback visual analysis prompt if enhanced version is not available."""
        return """
        Analyze the visual output and the provided code for the following issues:
        
        1. **Element Overlap:** Check for overlapping text, shapes, or mathematical expressions
        2. **Out-of-Bounds Objects:** Identify elements outside the visible frame
        3. **Spacing Issues:** Verify minimum 0.3 unit spacing between elements
        4. **Safe Area Compliance:** Ensure 0.5 unit margins from frame edges
        5. **Educational Clarity:** Assess if arrangement supports learning objectives
        
        Implementation Plan: {implementation}
        
        Code to analyze:
        {code}
        
        If issues are found, provide fixed code. If no issues, return "<LGTM>".
        
        <improved_code>
        ```python
        [Fixed code here]
        ```
        </improved_code>
        """

    def detect_visual_errors(
        self, 
        media_path: Union[str, Image.Image],
        scene_trace_id: Optional[str] = None,
        topic: Optional[str] = None,
        scene_number: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect visual errors using VLM without code modification.

        This method provides detailed visual error analysis without attempting to fix code.
        Useful for validation and quality assessment.

        Args:
            media_path: Path to media file or PIL Image
            scene_trace_id: Trace identifier
            topic: Topic of the scene
            scene_number: Scene number
            session_id: Session identifier

        Returns:
            Dictionary containing visual error analysis results

        Raises:
            ValueError: If visual error detection fails
            FileNotFoundError: If media file doesn't exist
        """
        try:
            # Validate media input
            if isinstance(media_path, str):
                media_file = Path(media_path)
                if not media_file.exists():
                    raise FileNotFoundError(f"Media file not found: {media_path}")
            
            # Create analysis prompt
            analysis_prompt = """
            You are an expert visual quality analyst. Analyze this Manim-generated frame/video for:
            
            1. **Element Overlap Detection:**
               - Text overlapping with shapes or other text
               - Mathematical expressions colliding
               - Unintentional object occlusion
            
            2. **Spatial Boundary Issues:**
               - Objects extending beyond frame boundaries
               - Violations of safe area margins (0.5 units from edges)
               - Insufficient spacing between elements (minimum 0.3 units)
            
            3. **Visual Quality Assessment:**
               - Overall composition balance
               - Readability of text elements
               - Educational effectiveness of arrangement
            
            Provide your analysis in the following format:
            
            **VISUAL ERROR ANALYSIS:**
            - Overlap Issues: [List any overlapping elements]
            - Boundary Violations: [List out-of-bounds elements]
            - Spacing Problems: [List spacing violations]
            - Quality Issues: [List other visual problems]
            
            **SEVERITY ASSESSMENT:**
            - Critical Errors: [Issues that severely impact readability]
            - Major Errors: [Issues that noticeably reduce quality]
            - Minor Errors: [Issues that slightly affect visual appeal]
            
            **OVERALL RATING:** [Excellent/Good/Fair/Poor]
            """
            
            # Determine media type and prepare input
            is_video = isinstance(media_path, str) and media_path.lower().endswith('.mp4')
            
            if is_video and isinstance(self.scene_model, (GeminiWrapper, VertexAIWrapper)):
                messages = [
                    {"type": "text", "content": analysis_prompt},
                    {"type": "video", "content": str(media_path)}
                ]
            else:
                if isinstance(media_path, str):
                    media = Image.open(media_path)
                else:
                    media = media_path
                messages = [
                    {"type": "text", "content": analysis_prompt},
                    {"type": "image", "content": media}
                ]
            
            # Get analysis response
            response_text = self.scene_model(
                messages,
                metadata={
                    "generation_name": "visual_error_detection",
                    "trace_id": scene_trace_id,
                    "tags": [topic or "unknown", f"scene{scene_number or 0}", "quality_analysis"],
                    "session_id": session_id or self.session_id
                }
            )
            
            # Parse response into structured results
            analysis_results = self._parse_visual_analysis(response_text)
            
            logger.info(f"Visual error detection completed for scene {scene_number or 'unknown'}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in visual error detection: {e}")
            raise ValueError(f"Visual error detection failed: {e}") from e

    def _parse_visual_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse visual analysis response into structured data.

        Args:
            response_text: Raw response from VLM

        Returns:
            Structured analysis results
        """
        results = {
            "overlap_issues": [],
            "boundary_violations": [],
            "spacing_problems": [],
            "quality_issues": [],
            "critical_errors": [],
            "major_errors": [],
            "minor_errors": [],
            "overall_rating": "Unknown",
            "raw_analysis": response_text
        }
        
        try:
            # Extract different sections using regex patterns
            overlap_match = re.search(r'Overlap Issues:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if overlap_match:
                results["overlap_issues"] = [item.strip() for item in overlap_match.group(1).split('\n') if item.strip()]
            
            boundary_match = re.search(r'Boundary Violations:\s*(.*?)(?=\n-|\n\*\*|$)', response_text, re.DOTALL)
            if boundary_match:
                results["boundary_violations"] = [item.strip() for item in boundary_match.group(1).split('\n') if item.strip()]
            
            rating_match = re.search(r'OVERALL RATING.*?:\s*([A-Za-z]+)', response_text)
            if rating_match:
                results["overall_rating"] = rating_match.group(1)
            
        except Exception as e:
            logger.warning(f"Error parsing visual analysis: {e}")
        
        return results