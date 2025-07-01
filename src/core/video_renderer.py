import os
import re
import subprocess
import asyncio
import concurrent.futures
from PIL import Image
from typing import Optional, List, Union, Dict
import traceback
import sys
import time
import json
import hashlib
from pathlib import Path
import shutil
import tempfile

try:
    import ffmpeg
except ImportError:
    print("Warning: ffmpeg-python not installed. Video combination features will be limited.")
    ffmpeg = None

from src.core.parse_video import (
    get_images_from_video,
    image_with_most_non_black_space
)


class OptimizedVideoRenderer:
    """Enhanced video renderer with significant performance optimizations."""

    def __init__(self, output_dir="output", print_response=False, use_visual_fix_code=False,
                 max_concurrent_renders=4, enable_caching=True, default_quality="medium",
                 use_gpu_acceleration=False, preview_mode=False):
        """Initialize the enhanced VideoRenderer.

        Args:
            output_dir (str): Directory for output files
            print_response (bool): Whether to print responses
            use_visual_fix_code (bool): Whether to use visual fix code
            max_concurrent_renders (int): Maximum concurrent render processes
            enable_caching (bool): Enable intelligent caching system
            default_quality (str): Default render quality (low/medium/high/preview)
            use_gpu_acceleration (bool): Use GPU acceleration if available
            preview_mode (bool): Enable preview mode for faster development
        """
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_visual_fix_code = use_visual_fix_code
        self.max_concurrent_renders = max_concurrent_renders
        self.enable_caching = enable_caching
        self.default_quality = default_quality
        self.use_gpu_acceleration = use_gpu_acceleration
        self.preview_mode = preview_mode
        
        # Performance monitoring
        self.render_stats = {
            'total_renders': 0,
            'cache_hits': 0,
            'total_time': 0,
            'average_time': 0
        }
        
        # Quality presets for faster rendering
        self.quality_presets = {
            'preview': {'flag': '-ql', 'fps': 15, 'resolution': '480p'},
            'low': {'flag': '-ql', 'fps': 15, 'resolution': '480p'},
            'medium': {'flag': '-qm', 'fps': 30, 'resolution': '720p'},
            'high': {'flag': '-qh', 'fps': 60, 'resolution': '1080p'},
            'production': {'flag': '-qp', 'fps': 60, 'resolution': '1440p'}
        }
        
        # Cache directory for rendered scenes
        self.cache_dir = os.path.join(output_dir, '.render_cache')
        if enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Thread pool for concurrent operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_renders)

    def _get_code_hash(self, code: str) -> str:
        """Generate hash for code to enable caching."""
        return hashlib.md5(code.encode()).hexdigest()

    def _get_cache_path(self, code_hash: str, quality: str) -> str:
        """Get cache file path for given code hash and quality."""
        return os.path.join(self.cache_dir, f"{code_hash}_{quality}.mp4")

    def _is_cached(self, code: str, quality: str) -> Optional[str]:
        """Check if rendered video exists in cache."""
        if not self.enable_caching:
            return None
        
        code_hash = self._get_code_hash(code)
        cache_path = self._get_cache_path(code_hash, quality)
        
        if os.path.exists(cache_path):
            print(f"Cache hit for code hash {code_hash[:8]}...")
            self.render_stats['cache_hits'] += 1
            return cache_path
        return None

    def _save_to_cache(self, code: str, quality: str, video_path: str):
        """Save rendered video to cache."""
        if not self.enable_caching or not os.path.exists(video_path):
            return
        
        code_hash = self._get_code_hash(code)
        cache_path = self._get_cache_path(code_hash, quality)
        
        try:
            shutil.copy2(video_path, cache_path)
            print(f"Cached render for hash {code_hash[:8]}...")
        except Exception as e:
            print(f"Warning: Could not cache render: {e}")

    async def render_scene_optimized(self, code: str, file_prefix: str, curr_scene: int, 
                                   curr_version: int, code_dir: str, media_dir: str, 
                                   quality: str = None, max_retries: int = 3, 
                                   use_visual_fix_code=False, visual_self_reflection_func=None, 
                                   banned_reasonings=None, scene_trace_id=None, topic=None, 
                                   session_id=None, code_generator=None, 
                                   scene_implementation=None, description=None, 
                                   scene_outline=None) -> tuple:
        """Optimized scene rendering with intelligent error handling and code generation fixes."""
        
        start_time = time.time()
        quality = quality or self.default_quality
        current_code = code
        
        # Check cache first
        cached_video = self._is_cached(current_code, quality)
        if cached_video:
            # Copy cached video to expected location
            expected_path = self._get_expected_video_path(file_prefix, curr_scene, curr_version, media_dir)
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            shutil.copy2(cached_video, expected_path)
            
            elapsed = time.time() - start_time
            print(f"Scene {curr_scene} rendered from cache in {elapsed:.2f}s")
            return current_code, None

        # Optimize manim command for speed
        file_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
        
        # Write optimized code file
        await self._write_code_file_async(file_path, current_code)
        
        # Build optimized manim command
        manim_cmd = self._build_optimized_command(file_path, media_dir, quality)
        
        retries = 0
        while retries < max_retries:
            try:
                print(f"🎬 Rendering scene {curr_scene} (quality: {quality}, attempt: {retries + 1})")
                
                # Execute manim with optimizations
                result = await asyncio.to_thread(
                    self._run_manim_optimized,
                    manim_cmd,
                    file_path
                )

                if result.returncode != 0:
                    raise Exception(result.stderr)

                # Find the rendered video
                video_path = self._find_rendered_video(file_prefix, curr_scene, curr_version, media_dir)
                
                # Save to cache
                self._save_to_cache(current_code, quality, video_path)

                # Visual fix code processing
                if use_visual_fix_code and visual_self_reflection_func and banned_reasonings:
                    current_code = await self._process_visual_fix(
                        current_code, video_path, file_prefix, curr_scene, curr_version,
                        code_dir, visual_self_reflection_func, banned_reasonings,
                        scene_trace_id, topic, session_id
                    )

                elapsed = time.time() - start_time
                self.render_stats['total_renders'] += 1
                self.render_stats['total_time'] += elapsed
                self.render_stats['average_time'] = self.render_stats['total_time'] / self.render_stats['total_renders']
                
                print(f"Scene {curr_scene} rendered successfully in {elapsed:.2f}s")
                print(f"Average render time: {self.render_stats['average_time']:.2f}s")
                
                return current_code, None

            except Exception as e:
                print(f"Render attempt {retries + 1} failed: {e}")
                
                # Save error log
                error_log_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_error_{retries}.log")
                await self._write_error_log_async(error_log_path, str(e), retries)
                
                # Instead of blind retry, try to fix the code if we have a code generator
                if code_generator and scene_implementation and retries < max_retries - 1:
                    print(f"🔧 Attempting to fix code using CodeGenerator (attempt {retries + 1})")
                    try:
                        fixed_code, fix_log = code_generator.fix_code_errors(
                            implementation_plan=scene_implementation,
                            code=current_code,
                            error=str(e),
                            scene_trace_id=scene_trace_id,
                            topic=topic,
                            scene_number=curr_scene,
                            session_id=session_id
                        )
                        
                        if fixed_code and fixed_code != current_code:
                            print(f"✨ Code fix generated, updating for next attempt")
                            current_code = fixed_code
                            curr_version += 1
                            
                            # Update file path and write fixed code
                            file_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
                            await self._write_code_file_async(file_path, current_code)
                            
                            # Update manim command for new file
                            manim_cmd = self._build_optimized_command(file_path, media_dir, quality)
                            
                            # Log the fix
                            fix_log_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_fix_log.txt")
                            await self._write_error_log_async(fix_log_path, fix_log or "Code fix applied", 0)
                        else:
                            print(f"⚠️ Code generator returned same or empty code, doing standard retry")
                    except Exception as fix_error:
                        print(f"❌ Code fix attempt failed: {fix_error}")
                        # Fall back to standard retry behavior
                
                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    return current_code, str(e)

        return current_code, f"Failed after {max_retries} attempts"

    def _build_optimized_command(self, file_path: str, media_dir: str, quality: str) -> List[str]:
        """Build optimized manim command with performance flags."""
        quality_preset = self.quality_presets.get(quality, self.quality_presets['medium'])
        
        cmd = [
            "manim",
            "render",
            quality_preset['flag'],  # Quality setting
            file_path,
            "--media_dir", media_dir,
            "--fps", str(quality_preset['fps'])
        ]
        
        # Add caching option (only disable if needed)
        if not self.enable_caching:
            cmd.append("--disable_caching")
        
        # Add GPU acceleration if available and enabled
        if self.use_gpu_acceleration:
            cmd.extend(["--renderer", "opengl"])
        
        # Preview mode optimizations
        if self.preview_mode or quality == 'preview':
            cmd.extend([
                "--save_last_frame",  # Only render final frame for quick preview
                "--write_to_movie"    # Skip unnecessary file operations
            ])
        
        return cmd

    def _run_manim_optimized(self, cmd: List[str], file_path: str) -> subprocess.CompletedProcess:
        """Run manim command with optimizations."""
        env = os.environ.copy()
        
        # Optimize environment for performance
        env.update({
            'MANIM_DISABLE_CACHING': 'false' if self.enable_caching else 'true',
            'MANIM_VERBOSITY': 'WARNING',  # Reduce log verbosity
            'OMP_NUM_THREADS': str(os.cpu_count()),  # Use all CPU cores
            'MANIM_RENDERER_TIMEOUT': '300'  # 5 minute timeout
        })
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )

    async def _write_code_file_async(self, file_path: str, code: str):
        """Asynchronously write code file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Add optimization hints to the code
        optimized_code = self._optimize_code_for_rendering(code)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(optimized_code)

    def _optimize_code_for_rendering(self, code: str) -> str:
        """Add optimization hints to Manim code."""
        optimizations = [
            "",
            "# Manim rendering optimizations",
            "from manim import config",
            "config.frame_rate = 30  # Balanced frame rate",
            "config.pixel_height = 720  # Optimized resolution",
            "config.pixel_width = 1280",
            ""
        ]
        
        # Find the end of manim imports specifically
        lines = code.split('\n')
        manim_import_end = 0
        
        for i, line in enumerate(lines):
            # Look for manim-related imports
            if (line.strip().startswith('from manim') or 
                line.strip().startswith('import manim') or
                line.strip().startswith('from manim_')):
                manim_import_end = i + 1
        
        # If no manim imports found, look for the end of all imports
        if manim_import_end == 0:
            for i, line in enumerate(lines):
                if (line.strip().startswith(('from ', 'import ')) and 
                    not line.strip().startswith('#')):
                    manim_import_end = i + 1
        
        # Insert optimization code after manim imports
        lines[manim_import_end:manim_import_end] = optimizations
        
        return '\n'.join(lines)

    async def _write_error_log_async(self, file_path: str, error: str, attempt: int):
        """Asynchronously write error log."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_content = f"[{timestamp}] Attempt {attempt + 1}: {error}\n"
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(log_content)

    def _get_expected_video_path(self, file_prefix: str, scene: int, version: int, media_dir: str) -> str:
        """Get expected path for rendered video."""
        return os.path.join(
            media_dir, "videos", f"{file_prefix}_scene{scene}_v{version}", 
            "1080p60", f"{file_prefix}_scene{scene}_v{version}.mp4"
        )

    def _find_rendered_video(self, file_prefix: str, scene: int, version: int, media_dir: str) -> str:
        """Find the rendered video file."""
        video_dir = os.path.join(media_dir, "videos", f"{file_prefix}_scene{scene}_v{version}")
        
        # Look in quality-specific subdirectories
        for quality_dir in ["1080p60", "720p30", "480p15"]:
            search_dir = os.path.join(video_dir, quality_dir)
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.mp4'):
                        return os.path.join(search_dir, file)
        
        raise FileNotFoundError(f"No rendered video found for scene {scene} version {version}")

    async def _process_visual_fix(self, code: str, video_path: str, file_prefix: str, 
                                scene: int, version: int, code_dir: str, 
                                visual_self_reflection_func, banned_reasonings: List[str],
                                scene_trace_id: str, topic: str, session_id: str) -> str:
        """Process visual fix code with optimization."""
        
        # For Gemini/Vertex AI models, pass the video directly
        if hasattr(self, 'scene_model') and self.scene_model.model_name.startswith(('gemini/', 'vertex_ai/')):
            media_input = video_path
        else:
            # For other models, create optimized snapshot
            media_input = await self._create_optimized_snapshot(topic, scene, version)
                
        new_code, log = visual_self_reflection_func(
            code, media_input, scene_trace_id=scene_trace_id,
            topic=topic, scene_number=scene, session_id=session_id
        )

        # Save visual fix log
        log_path = os.path.join(code_dir, f"{file_prefix}_scene{scene}_v{version}_vfix_log.txt")
        await self._write_error_log_async(log_path, log, 0)

        # Check for termination markers
        if "<LGTM>" in new_code or any(word in new_code for word in banned_reasonings):
            return code

        # Save updated code
        new_version = version + 1
        new_code_path = os.path.join(code_dir, f"{file_prefix}_scene{scene}_v{new_version}.py")
        await self._write_code_file_async(new_code_path, new_code)
        print(f"Visual fix code saved to scene{scene}/code/{file_prefix}_scene{scene}_v{new_version}.py")
        
        return new_code

    async def render_multiple_scenes_parallel(self, scene_configs: List[Dict], 
                                           max_concurrent: int = None) -> List[tuple]:
        """Render multiple scenes in parallel with optimized resource management."""
        
        max_concurrent = max_concurrent or self.max_concurrent_renders
        print(f"Starting parallel rendering of {len(scene_configs)} scenes (max concurrent: {max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def render_single_scene(config):
            async with semaphore:
                return await self.render_scene_optimized(**config)
        
        start_time = time.time()
        
        # Execute all renders concurrently
        tasks = [render_single_scene(config) for config in scene_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if not isinstance(r, Exception) and r[1] is None)
        
        print(f"Parallel rendering completed in {elapsed:.2f}s")
        print(f"Success rate: {successful}/{len(scene_configs)} scenes")
        print(f"Cache hit rate: {self.render_stats['cache_hits']}/{self.render_stats['total_renders']} ({self.render_stats['cache_hits']/max(1,self.render_stats['total_renders'])*100:.1f}%)")
        
        return results

    async def _create_optimized_snapshot(self, topic: str, scene_number: int, 
                                       version_number: int) -> Image.Image:
        """Create optimized snapshot with async processing."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        video_folder_path = os.path.join(
            self.output_dir, file_prefix, "media", "videos", 
            f"{file_prefix}_scene{scene_number}_v{version_number}", "1080p60"
        )
        
        # Find video file
        video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
        if not video_files:
            raise FileNotFoundError(f"No mp4 files found in {video_folder_path}")
        
        video_path = os.path.join(video_folder_path, video_files[0])
        
        # Create snapshot asynchronously
        return await asyncio.to_thread(
            lambda: image_with_most_non_black_space(
                get_images_from_video(video_path), 
                return_type="image"
            )
        )

    async def combine_videos_optimized(self, topic: str, use_hardware_acceleration: bool = False) -> str:
        """Optimized video combination with hardware acceleration and parallel processing."""
        
        start_time = time.time()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        
        print(f"🎬 Starting optimized video combination for topic: {topic}")
        print(f"🖥️ GPU Acceleration: {'Enabled' if use_hardware_acceleration else 'Disabled (CPU only)'}")
        
        # Prepare paths
        video_output_dir = os.path.join(self.output_dir, file_prefix)
        output_video_path = os.path.join(video_output_dir, f"{file_prefix}_combined.mp4")
        output_srt_path = os.path.join(video_output_dir, f"{file_prefix}_combined.srt")
        
        # Check if already exists
        if os.path.exists(output_video_path):
            print(f"Combined video already exists at {output_video_path}")
            return output_video_path
        
        # Get scene information
        scene_videos, scene_subtitles = await self._gather_scene_files_async(file_prefix)
        
        if not scene_videos:
            raise ValueError("No scene videos found to combine")
        
        print(f"📹 Found {len(scene_videos)} scene videos to combine")
        
        try:
            if ffmpeg is None:
                print("⚠️ ffmpeg-python not available, using direct FFmpeg fallback...")
                fallback_output = await self._fallback_video_combination(scene_videos, output_video_path)
                print(f"✅ Direct FFmpeg combination successful: {fallback_output}")
                return fallback_output
            
            # Analyze videos in parallel
            print("🔍 Analyzing video properties...")
            analysis_tasks = [
                asyncio.to_thread(self._analyze_video, video) 
                for video in scene_videos
            ]
            video_info = await asyncio.gather(*analysis_tasks)
            
            has_audio = [info['has_audio'] for info in video_info]
            print(f"🎵 Audio tracks found: {sum(has_audio)}/{len(scene_videos)} videos")
            
            # Build optimized ffmpeg command
            if any(has_audio):
                print("🎵 Combining videos with audio tracks...")
                await self._combine_with_audio_optimized(
                    scene_videos, video_info, output_video_path, use_hardware_acceleration
                )
            else:
                print("🔇 Combining videos without audio...")
                await self._combine_without_audio_optimized(
                    scene_videos, output_video_path, use_hardware_acceleration
                )
            
            # Verify the output file was created and is valid
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video was not created: {output_video_path}")
            
            # Check if the video file is valid
            file_size = os.path.getsize(output_video_path)
            if file_size < 1024:  # Less than 1KB is probably invalid
                raise ValueError(f"Output video file seems invalid (size: {file_size} bytes)")
            
            print(f"✅ Video file created successfully (size: {file_size / (1024*1024):.2f} MB)")
            
            # Combine subtitles if available
            if scene_subtitles:
                print("📝 Combining subtitles...")
                await self._combine_subtitles_async(scene_subtitles, scene_videos, output_srt_path)
            
            elapsed = time.time() - start_time
            print(f"🎉 Video combination completed in {elapsed:.2f}s")
            print(f"📁 Output: {output_video_path}")
            
            return output_video_path
            
        except Exception as e:
            print(f"❌ Error in optimized video combination: {e}")
            print("🔧 Attempting fallback video combination...")
            
            # Fallback to simple concatenation
            try:
                fallback_output = await self._fallback_video_combination(scene_videos, output_video_path)
                print(f"✅ Fallback combination successful: {fallback_output}")
                return fallback_output
            except Exception as fallback_error:
                print(f"❌ Fallback combination also failed: {fallback_error}")
                traceback.print_exc()
                raise

    async def _gather_scene_files_async(self, file_prefix: str) -> tuple:
        """Asynchronously gather scene video and subtitle files."""
        search_path = os.path.join(self.output_dir, file_prefix, "media", "videos")
        
        # Get scene count
        scene_outline_path = os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt")
        with open(scene_outline_path) as f:
            plan = f.read()
        
        scene_outline_match = re.search(r'(<SCENE_OUTLINE>.*?</SCENE_OUTLINE>)', plan, re.DOTALL)
        if not scene_outline_match:
            print(f"No scene outline found in plan: {plan[:200]}...")
            return []
        scene_outline = scene_outline_match.group(1)
        scene_count = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline))
        
        # Find scene files in parallel
        tasks = [
            asyncio.to_thread(self._find_scene_files, search_path, file_prefix, scene_num)
            for scene_num in range(1, scene_count + 1)
        ]
        
        results = await asyncio.gather(*tasks)
        
        scene_videos = []
        scene_subtitles = []
        
        for video, subtitle in results:
            if video:
                scene_videos.append(video)
                scene_subtitles.append(subtitle)
        
        return scene_videos, scene_subtitles

    def _find_scene_files(self, search_path: str, file_prefix: str, scene_num: int) -> tuple:
        """Find video and subtitle files for a specific scene."""
        scene_folders = []
        for root, dirs, files in os.walk(search_path):
            for dir in dirs:
                if dir.startswith(f"{file_prefix}_scene{scene_num}"):
                    scene_folders.append(os.path.join(root, dir))
        
        if not scene_folders:
            return None, None
        
        # Get latest version
        scene_folders.sort(key=lambda f: int(f.split("_v")[-1]) if "_v" in f else 0)
        folder = scene_folders[-1]
        
        video_file = None
        subtitle_file = None
        
        quality_dirs = ["1080p60", "720p30", "480p15"]
        for quality_dir in quality_dirs:
            quality_path = os.path.join(folder, quality_dir)
            if os.path.exists(quality_path):
                for filename in os.listdir(quality_path):
                    if filename.endswith('.mp4') and not video_file:
                        video_file = os.path.join(quality_path, filename)
                    elif filename.endswith('.srt') and not subtitle_file:
                        subtitle_file = os.path.join(quality_path, filename)
                break
        
        return video_file, subtitle_file

    def _analyze_video(self, video_path: str) -> Dict:
        """Analyze video properties for optimization."""
        if ffmpeg is None:
            # Fallback analysis using direct FFmpeg probe
            import subprocess
            import json
            
            try:
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_streams',
                    video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                probe_data = json.loads(result.stdout)
                
                video_stream = next(stream for stream in probe_data['streams'] if stream['codec_type'] == 'video')
                audio_streams = [stream for stream in probe_data['streams'] if stream['codec_type'] == 'audio']
                
                return {
                    'path': video_path,
                    'duration': float(video_stream.get('duration', 0)),
                    'has_audio': len(audio_streams) > 0,
                    'width': int(video_stream.get('width', 1920)),
                    'height': int(video_stream.get('height', 1080)),
                    'fps': eval(video_stream.get('avg_frame_rate', '30/1'))
                }
            except Exception as e:
                print(f"Warning: Could not analyze video {video_path}: {e}")
                # Return default values
                return {
                    'path': video_path,
                    'duration': 10.0,  # Default duration
                    'has_audio': False,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30
                }
        
        probe = ffmpeg.probe(video_path)
        video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        
        return {
            'path': video_path,
            'duration': float(video_stream['duration']),
            'has_audio': len(audio_streams) > 0,
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['avg_frame_rate'])
        }

    async def _combine_with_audio_optimized(self, scene_videos: List[str], video_info: List[Dict], 
                                          output_path: str, use_hardware_acceleration: bool):
        """Combine videos with audio using hardware acceleration."""
        import ffmpeg
        
        streams = []
        for video_path, info in zip(scene_videos, video_info):
            input_vid = ffmpeg.input(video_path)
            
            if info['has_audio']:
                streams.extend([input_vid['v'], input_vid['a']])
            else:
                # Add silent audio
                silent_audio = ffmpeg.input(
                    f'anullsrc=channel_layout=stereo:sample_rate=44100',
                    f='lavfi', t=info['duration']
                )['a']
                streams.extend([input_vid['v'], silent_audio])
        
        # Build optimized encoding options for maximum compatibility
        encode_options = {
            'c:v': 'libx264',      # Use libx264 for maximum compatibility
            'c:a': 'aac',          # AAC audio codec
            'preset': 'medium',    # Balanced preset for good quality/speed
            'crf': '23',           # Good quality/speed balance
            'pix_fmt': 'yuv420p',  # Pixel format for maximum compatibility
            'movflags': '+faststart',  # Enable fast start for web playback
            'r': '30',             # Set frame rate to 30fps
            'threads': '0',        # Use all available threads
            'profile:v': 'high',   # H.264 profile for better compatibility
            'level': '4.0'         # H.264 level for broad device support
        }
        
        # Only use hardware acceleration if explicitly requested and working
        if use_hardware_acceleration:
            try:
                # Test if NVENC is available by creating a simple test
                test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1', 
                           '-c:v', 'h264_nvenc', '-f', 'null', '-']
                test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                
                if test_result.returncode == 0:
                    encode_options.update({
                        'c:v': 'h264_nvenc',
                        'preset': 'fast',      # NVENC preset
                        'profile:v': 'high',
                        'level': '4.0',
                        'rc': 'constqp',       # Constant quality mode
                        'qp': '23'            # Quality parameter
                    })
                    print("✅ Using NVIDIA hardware acceleration")
                else:
                    print("⚠️ NVIDIA hardware acceleration not available, using CPU encoding")
            except Exception as e:
                print(f"⚠️ Hardware acceleration test failed: {e}, using CPU encoding")
        
        concat = ffmpeg.concat(*streams, v=1, a=1, unsafe=True)
        
        # Run with progress monitoring
        process = (
            concat
            .output(output_path, **encode_options)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        await self._monitor_ffmpeg_progress(process, "audio combination")

    async def _combine_without_audio_optimized(self, scene_videos: List[str], 
                                             output_path: str, use_hardware_acceleration: bool):
        """Combine videos without audio using hardware acceleration."""
        import ffmpeg
        
        streams = [ffmpeg.input(video)['v'] for video in scene_videos]
        
        # Build encoding options for maximum compatibility
        encode_options = {
            'c:v': 'libx264',      # Use libx264 for maximum compatibility
            'preset': 'medium',    # Balanced preset
            'crf': '20',           # Good quality
            'pix_fmt': 'yuv420p',  # Pixel format for maximum compatibility
            'movflags': '+faststart',  # Enable fast start
            'r': '30',             # Set frame rate to 30fps
            'threads': '0',        # Use all available threads
            'profile:v': 'high',   # H.264 profile
            'level': '4.0'         # H.264 level
        }
        
        # Test hardware acceleration availability
        if use_hardware_acceleration:
            try:
                # Test if NVENC is available
                test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1', 
                           '-c:v', 'h264_nvenc', '-f', 'null', '-']
                test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                
                if test_result.returncode == 0:
                    encode_options.update({
                        'c:v': 'h264_nvenc',
                        'preset': 'fast',
                        'profile:v': 'high',
                        'level': '4.0',
                        'rc': 'constqp',
                        'qp': '20'
                    })
                    print("✅ Using NVIDIA hardware acceleration for video-only combination")
                else:
                    print("⚠️ NVIDIA hardware acceleration not available, using CPU encoding")
            except Exception as e:
                print(f"⚠️ Hardware acceleration test failed: {e}, using CPU encoding")
        
        concat = ffmpeg.concat(*streams, v=1, unsafe=True)
        
        process = (
            concat
            .output(output_path, **encode_options)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        await self._monitor_ffmpeg_progress(process, "video combination")

    async def _monitor_ffmpeg_progress(self, process, operation_name: str):
        """Monitor FFmpeg progress asynchronously."""
        print(f"Starting {operation_name}...")
        
        while True:
            line = await asyncio.to_thread(process.stdout.readline)
            if not line:
                break
            
            line = line.decode('utf-8')
            if 'frame=' in line:
                # Extract progress information
                frame_match = re.search(r'frame=\s*(\d+)', line)
                time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
                
                if frame_match and time_match:
                    frame = frame_match.group(1)
                    time_str = time_match.group(1)
                    print(f"\r⚡ Processing: frame={frame}, time={time_str}", end='', flush=True)
        
        stdout, stderr = await asyncio.to_thread(process.communicate)
        print(f"\n{operation_name} completed!")
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode('utf-8')}")

    async def _combine_subtitles_async(self, scene_subtitles: List[str], 
                                     scene_videos: List[str], output_path: str):
        """Combine subtitles asynchronously."""
        
        def combine_subtitles():
            with open(output_path, 'w', encoding='utf-8') as outfile:
                current_time_offset = 0
                subtitle_index = 1

                for srt_file, video_file in zip(scene_subtitles, scene_videos):
                    if srt_file is None:
                        continue

                    with open(srt_file, 'r', encoding='utf-8') as infile:
                        lines = infile.readlines()
                        i = 0
                        while i < len(lines):
                            line = lines[i].strip()
                            if line.isdigit():
                                outfile.write(f"{subtitle_index}\n")
                                subtitle_index += 1
                                i += 1

                                time_line = lines[i].strip()
                                start_time, end_time = time_line.split(' --> ')

                                def adjust_time(time_str, offset):
                                    h, m, s = time_str.replace(',', '.').split(':')
                                    total_seconds = float(h) * 3600 + float(m) * 60 + float(s) + offset
                                    h = int(total_seconds // 3600)
                                    m = int((total_seconds % 3600) // 60)
                                    s = total_seconds % 60
                                    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

                                new_start = adjust_time(start_time, current_time_offset)
                                new_end = adjust_time(end_time, current_time_offset)
                                outfile.write(f"{new_start} --> {new_end}\n")
                                i += 1

                                while i < len(lines) and lines[i].strip():
                                    outfile.write(lines[i])
                                    i += 1
                                outfile.write('\n')
                            else:
                                i += 1

                    # Update time offset
                    import ffmpeg
                    probe = ffmpeg.probe(video_file)
                    duration = float(probe['streams'][0]['duration'])
                    current_time_offset += duration

        await asyncio.to_thread(combine_subtitles)
        print(f"Subtitles combined to {output_path}")

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        return {
            **self.render_stats,
            'cache_hit_rate': self.render_stats['cache_hits'] / max(1, self.render_stats['total_renders']),
            'cache_enabled': self.enable_caching,
            'concurrent_renders': self.max_concurrent_renders
        }

    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cache files."""
        if not self.enable_caching:
            return
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            if os.path.getmtime(file_path) < current_time - max_age_seconds:
                os.remove(file_path)
                print(f"Removed old cache file: {file}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.executor.shutdown(wait=True)

    def render_scene(self, code: str, file_prefix: str, curr_scene: int, 
                    curr_version: int, code_dir: str, media_dir: str, 
                    use_visual_fix_code=False, visual_self_reflection_func=None, 
                    banned_reasonings=None, scene_trace_id=None, topic=None, 
                    session_id=None, code_generator=None, scene_implementation=None,
                    description=None, scene_outline=None) -> tuple:
        """Legacy render_scene method for backward compatibility."""
        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.render_scene_optimized(
                    code=code,
                    file_prefix=file_prefix,
                    curr_scene=curr_scene,
                    curr_version=curr_version,
                    code_dir=code_dir,
                    media_dir=media_dir,
                    use_visual_fix_code=use_visual_fix_code,
                    visual_self_reflection_func=visual_self_reflection_func,
                    banned_reasonings=banned_reasonings,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    session_id=session_id,
                    code_generator=code_generator,
                    scene_implementation=scene_implementation,
                    description=description,
                    scene_outline=scene_outline
                )
            )
            return result
        finally:
            loop.close()

    def combine_videos(self, topic: str) -> str:
        """Legacy combine_videos method for backward compatibility."""
        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.combine_videos_optimized(topic=topic)
            )
            return result
        finally:
            loop.close()

    async def _fallback_video_combination(self, scene_videos: List[str], output_path: str) -> str:
        """Simple fallback video combination using direct FFmpeg commands."""
        
        print("🔧 Using fallback video combination method...")
        
        # Create a temporary file list for concat demuxer
        temp_dir = tempfile.mkdtemp()
        file_list_path = os.path.join(temp_dir, "file_list.txt")
        
        try:
            # Write file list for concat demuxer
            with open(file_list_path, 'w') as f:
                for video in scene_videos:
                    # Ensure proper path format for concat demuxer
                    video_path = os.path.abspath(video).replace('\\', '/')
                    f.write(f"file '{video_path}'\n")
            
            print(f"📝 Created file list: {file_list_path}")
            print(f"🎬 Combining {len(scene_videos)} videos using direct FFmpeg...")
            
            # Use direct FFmpeg command for maximum compatibility
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                '-crf', '25',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output file
                output_path
            ]
            
            print(f"🔧 Running command: {' '.join(cmd)}")
            
            # Run the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor progress
            async def read_stderr():
                stderr_output = []
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8').strip()
                    stderr_output.append(line_str)
                    
                    if 'frame=' in line_str:
                        frame_match = re.search(r'frame=\s*(\d+)', line_str)
                        time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line_str)
                        
                        if frame_match and time_match:
                            frame = frame_match.group(1)
                            time_str = time_match.group(1)
                            print(f"\r🔧 Fallback processing: frame={frame}, time={time_str}", end='', flush=True)
                
                return stderr_output
            
            # Wait for completion
            stderr_task = asyncio.create_task(read_stderr())
            await process.wait()
            stderr_output = await stderr_task
            
            print(f"\n🔧 Fallback combination completed!")
            
            if process.returncode != 0:
                error_msg = '\n'.join(stderr_output)
                print(f"❌ FFmpeg error output:\n{error_msg}")
                raise Exception(f"Direct FFmpeg command failed with return code {process.returncode}")
            
            # Verify output
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Fallback output video was not created: {output_path}")
            
            file_size = os.path.getsize(output_path)
            if file_size < 1024:
                raise ValueError(f"Fallback output video file seems invalid (size: {file_size} bytes)")
            
            print(f"✅ Fallback video created successfully (size: {file_size / (1024*1024):.2f} MB)")
            return output_path
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(file_list_path):
                    os.remove(file_list_path)
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"⚠️ Could not clean up temp files: {e}")

# Backward compatibility alias
VideoRenderer = OptimizedVideoRenderer