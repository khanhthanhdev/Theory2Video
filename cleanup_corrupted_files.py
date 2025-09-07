#!/usr/bin/env python3
"""
Cleanup script to remove corrupted cache files and scene outlines that contain error messages.
"""

import os
import glob
import re
from pathlib import Path

def is_corrupted_file(file_path: str) -> bool:
    """Check if a file contains error messages instead of valid content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common error patterns
        error_indicators = [
            "Error:",
            "Exception:",
            "Traceback",
            "litellm.NotFoundError",
            "VertexAIException",
            '"error"',
            '"code": 404',
            "models/gemini-",  # Check for model-specific errors
            "is not found for API",
            "preview-04-17"  # Old model name
        ]
        
        for indicator in error_indicators:
            if indicator in content:
                return True
        
        return False
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return False

def cleanup_corrupted_files():
    """Remove corrupted scene outline and cache files."""
    output_dir = "output"
    removed_count = 0
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    # Find all scene outline files
    scene_outline_pattern = os.path.join(output_dir, "**", "*_scene_outline.txt")
    scene_outline_files = glob.glob(scene_outline_pattern, recursive=True)
    
    print(f"Found {len(scene_outline_files)} scene outline files to check")
    
    for file_path in scene_outline_files:
        if is_corrupted_file(file_path):
            print(f"Removing corrupted file: {file_path}")
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    # Also check for other text files that might contain errors
    other_text_patterns = [
        os.path.join(output_dir, "**", "*_implementation_plan.txt"),
        os.path.join(output_dir, "**", "*_vision_storyboard.txt"),
        os.path.join(output_dir, "**", "*_technical_implementation.txt"),
    ]
    
    for pattern in other_text_patterns:
        text_files = glob.glob(pattern, recursive=True)
        for file_path in text_files:
            if is_corrupted_file(file_path):
                print(f"Removing corrupted file: {file_path}")
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    print(f"✅ Cleanup complete. Removed {removed_count} corrupted files.")
    
    if removed_count > 0:
        print("\nNote: The next video generation will create fresh files without errors.")

if __name__ == "__main__":
    cleanup_corrupted_files()
