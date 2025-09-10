import os
import re
import sys
import asyncio
import subprocess
import types
from pathlib import Path

import pytest


class MinimalPlanner:
    async def generate_scene_outline(self, topic: str, description: str, session_id: str) -> str:
        # Return a minimal valid outline with one scene
        return (
            "<SCENE_OUTLINE>\n"
            "<SCENE_1>\n"
            "Scene Title: Test Scene\n"
            "Scene Purpose: Demonstrate pipeline\n"
            "Scene Description: Simple\n"
            "Scene Layout: Centered\n"
            "</SCENE_1>\n"
            "</SCENE_OUTLINE>\n"
        )

    async def generate_scene_implementation_concurrently_enhanced(self, topic: str, description: str, plan: str, session_id: str):
        # Return a simple one-scene implementation plan
        return [
            """
<SCENE_VISION_STORYBOARD_PLAN>
Storyboard: Draw a circle.
</SCENE_VISION_STORYBOARD_PLAN>

<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>
Use a single Scene; add a Circle and a short wait.
</SCENE_TECHNICAL_IMPLEMENTATION_PLAN>

<SCENE_ANIMATION_NARRATION_PLAN>
No narration required.
</SCENE_ANIMATION_NARRATION_PLAN>
"""
        ]


@pytest.mark.asyncio
async def test_end_to_end_pipeline_creates_combined_video(tmp_path: Path, monkeypatch):
    # Arrange
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inject a fake litellm module before importing generate_video to avoid heavy deps
    fake_litellm = types.ModuleType('litellm')
    def _fake_completion(*args, **kwargs):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])
    def _fake_completion_cost(*args, **kwargs):
        return 0.0
    fake_litellm.completion = _fake_completion
    fake_litellm.completion_cost = _fake_completion_cost
    fake_litellm.success_callback = []
    fake_litellm.failure_callback = []
    sys.modules['litellm'] = fake_litellm

    # Import after stubbing litellm
    from generate_video import EnhancedVideoGenerator, VideoGenerationConfig

    config = VideoGenerationConfig(
        planner_model="dummy/model",
        scene_model="dummy/model",
        helper_model="dummy/model",
        output_dir=str(out_dir),
        enable_caching=True,
        preview_mode=False,
        use_visual_fix_code=False,
        max_concurrent_renders=1,
        verbose=False,
    )

    gen = EnhancedVideoGenerator(config)

    # Inject minimal planner (avoid network/LLM)
    gen.planner = MinimalPlanner()

    # Stub code generation to return trivial Manim code (we won't actually execute manim)
    def fake_generate_manim_code(**kwargs):
        code = (
            "from manim import *\n\n"
            "class Scene1(Scene):\n"
            "    def construct(self):\n"
            "        self.add(Circle())\n"
            "        self.wait(0.1)\n"
        )
        return code, "ok"

    gen.code_generator.generate_manim_code = fake_generate_manim_code  # type: ignore

    # Monkeypatch renderer's low-level manim runner to simulate success and create a dummy video file
    base_renderer = gen.scene_service.renderer

    def fake_run_manim(cmd, file_path):
        # Extract media_dir from cmd
        media_dir = None
        if "--media_dir" in cmd:
            idx = cmd.index("--media_dir")
            media_dir = cmd[idx + 1]
        assert media_dir, "--media_dir not found in command"

        # Derive file_prefix, scene, version from file_path name
        # e.g., mytopic_scene1_v1.py
        stem = Path(file_path).stem
        m = re.match(r"(.+)_scene(\d+)_v(\d+)$", stem)
        assert m, f"Unexpected file name format: {stem}"
        file_prefix, scene_no, version_no = m.group(1), int(m.group(2)), int(m.group(3))

        # Create expected output path: media_dir/videos/{file_prefix}_scene{n}_v{v}/1080p60/<file>.mp4
        video_dir = Path(media_dir) / "videos" / f"{file_prefix}_scene{scene_no}_v{version_no}" / "1080p60"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / f"{file_prefix}_scene{scene_no}_v{version_no}.mp4").write_bytes(b"00")

        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(base_renderer, "_run_manim_optimized", fake_run_manim)

    # Monkeypatch combine to avoid ffmpeg and just write a dummy combined file
    async def fake_combine_videos_optimized(topic: str, use_hardware_acceleration: bool = False) -> str:
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        out_path = Path(config.output_dir) / file_prefix / f"{file_prefix}_combined.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"00")
        return str(out_path)

    monkeypatch.setattr(gen.renderer, "combine_videos_optimized", fake_combine_videos_optimized)

    # Act
    topic = "Integration Test"
    description = "Ensure full pipeline wiring"
    await gen.generate_video_pipeline(topic, description)

    # Assert
    file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
    topic_dir = out_dir / file_prefix
    assert (topic_dir / f"{file_prefix}_scene_outline.txt").exists()
    assert (topic_dir / "scene1" / "succ_rendered.txt").exists()
    assert (topic_dir / f"{file_prefix}_combined.mp4").exists()
