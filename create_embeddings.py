#!/usr/bin/env python3
"""
Script to create embeddings and populate the vector store with Manim documentation.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.rag.vector_store import EnhancedRAGVectorStore as RAGVectorStore
from src.config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_manim_docs():
    """Download Manim documentation if not available locally."""
    print("Checking for Manim documentation...")
    
    manim_docs_path = Config.MANIM_DOCS_PATH
    os.makedirs(manim_docs_path, exist_ok=True)
    
    # Check if we have any documentation files
    doc_files = []
    for ext in ['.py', '.md', '.rst', '.txt']:
        doc_files.extend(list(Path(manim_docs_path).rglob(f'*{ext}')))
    
    if len(doc_files) == 0:
        print("No documentation files found. Creating sample documentation...")
        create_sample_manim_docs(manim_docs_path)
    else:
        print(f"Found {len(doc_files)} documentation files")

def create_sample_manim_docs(docs_path: str):
    """Create sample Manim documentation for testing."""
    
    # Create core documentation
    core_docs = {
        "circle.py": '''
"""Circle mobject in Manim.

The Circle class is one of the basic geometric shapes in Manim.
It can be used to create circular objects that can be animated.

Example:
    circle = Circle(radius=1, color=BLUE)
    self.add(circle)
    self.play(Create(circle))

Attributes:
    radius (float): The radius of the circle
    color (str): The color of the circle
    stroke_width (float): The width of the circle's outline
"""

from manim import *

class Circle(Arc):
    \"\"\"A circle mobject.
    
    Parameters
    ----------
    radius : float
        The radius of the circle.
    color : str
        The color of the circle.
    \"\"\"
    
    def __init__(self, radius=1, **kwargs):
        super().__init__(start_angle=0, angle=TAU, radius=radius, **kwargs)
        
    def animate_creation(self):
        \"\"\"Animate the creation of the circle.\"\"\"
        return Create(self)
        
    def move_to_position(self, position):
        \"\"\"Move circle to a specific position.\"\"\"
        return self.move_to(position)
''',
        
        "text.py": '''
"""Text mobjects in Manim.

Text and Tex classes for displaying text and mathematical expressions.

Example:
    text = Text("Hello World", font_size=48)
    equation = MathTex(r"E = mc^2")
    
    self.play(Write(text))
    self.play(Transform(text, equation))
"""

from manim import *

class Text(SVGMobject):
    \"\"\"Text mobject for displaying regular text.
    
    Parameters
    ----------
    text : str
        The text to display
    font_size : int
        Size of the font
    color : str
        Color of the text
    \"\"\"
    
    def __init__(self, text, font_size=48, **kwargs):
        self.text = text
        self.font_size = font_size
        super().__init__(**kwargs)
        
class MathTex(SVGMobject):
    \"\"\"Mathematical text using LaTeX.
    
    Parameters
    ----------
    tex_string : str
        LaTeX string for mathematical expression
    \"\"\"
    
    def __init__(self, tex_string, **kwargs):
        self.tex_string = tex_string
        super().__init__(**kwargs)
''',

        "animation.py": '''
"""Animation classes in Manim.

Core animation classes for creating smooth transitions and movements.

Example:
    circle = Circle()
    
    # Basic animations
    self.play(Create(circle))
    self.play(FadeIn(circle))
    self.play(Transform(circle, square))
"""

from manim import *

class Create(Animation):
    \"\"\"Animation that creates a mobject by drawing it.
    
    Parameters
    ----------
    mobject : Mobject
        The mobject to create
    run_time : float
        Duration of the animation
    \"\"\"
    
    def __init__(self, mobject, run_time=1, **kwargs):
        super().__init__(mobject, run_time=run_time, **kwargs)

class Transform(Animation):
    \"\"\"Transform one mobject into another.
    
    Parameters
    ----------
    mobject : Mobject
        Source mobject
    target_mobject : Mobject
        Target mobject to transform into
    \"\"\"
    
    def __init__(self, mobject, target_mobject, **kwargs):
        self.target_mobject = target_mobject
        super().__init__(mobject, **kwargs)

class FadeIn(Animation):
    \"\"\"Fade in animation.\"\"\"
    pass

class FadeOut(Animation):
    \"\"\"Fade out animation.\"\"\"
    pass
''',

        "scene.py": '''
"""Scene class - the foundation of Manim animations.

Every Manim animation is built using Scene classes.

Example:
    class MyScene(Scene):
        def construct(self):
            circle = Circle()
            self.play(Create(circle))
            self.wait(1)
"""

from manim import *

class Scene:
    \"\"\"Base class for all Manim scenes.
    
    The Scene class provides the foundation for creating animations.
    Override the construct() method to define your animation.
    \"\"\"
    
    def __init__(self):
        self.mobjects = []
        
    def construct(self):
        \"\"\"Override this method to create your animation.\"\"\"
        pass
        
    def add(self, *mobjects):
        \"\"\"Add mobjects to the scene.\"\"\"
        for mobject in mobjects:
            self.mobjects.append(mobject)
            
    def play(self, *animations, run_time=1):
        \"\"\"Play animations.
        
        Parameters
        ----------
        animations : Animation
            Animations to play
        run_time : float
            Duration to run animations
        \"\"\"
        for animation in animations:
            animation.run_time = run_time
            
    def wait(self, duration=1):
        \"\"\"Wait for specified duration.\"\"\"
        time.sleep(duration)
''',

        "plotting.py": '''
"""Plotting utilities in Manim.

Classes for creating mathematical plots and graphs.

Example:
    axes = Axes(x_range=[-3, 3], y_range=[-2, 2])
    graph = axes.plot(lambda x: x**2, color=BLUE)
    
    self.play(Create(axes))
    self.play(Create(graph))
"""

from manim import *

class Axes(VGroup):
    \"\"\"Coordinate axes for plotting.
    
    Parameters
    ----------
    x_range : list
        Range for x-axis [min, max, step]
    y_range : list  
        Range for y-axis [min, max, step]
    \"\"\"
    
    def __init__(self, x_range=[-1, 1, 1], y_range=[-1, 1, 1], **kwargs):
        self.x_range = x_range
        self.y_range = y_range
        super().__init__(**kwargs)
        
    def plot(self, function, color=WHITE, **kwargs):
        \"\"\"Plot a mathematical function.
        
        Parameters
        ----------
        function : callable
            Function to plot
        color : str
            Color of the plot
        \"\"\"
        return ParametricFunction(function, color=color, **kwargs)
        
    def get_graph(self, function, **kwargs):
        \"\"\"Get graph of function.\"\"\"
        return self.plot(function, **kwargs)

class NumberPlane(Axes):
    \"\"\"A coordinate plane with grid lines.\"\"\"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
'''
    }
    
    # Create the core documentation files
    for filename, content in core_docs.items():
        file_path = os.path.join(docs_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Create a README file
    readme_content = '''# Manim Documentation

This directory contains documentation for Manim (Mathematical Animation Engine).

## Core Classes

- **Circle**: Create circular shapes and animations
- **Text/MathTex**: Display text and mathematical expressions  
- **Scene**: Base class for all animations
- **Animation**: Classes for creating smooth transitions
- **Axes/NumberPlane**: Plotting and coordinate systems

## Basic Usage

```python
from manim import *

class MyScene(Scene):
    def construct(self):
        # Create objects
        circle = Circle(radius=1, color=BLUE)
        text = Text("Hello Manim!")
        
        # Animate
        self.play(Create(circle))
        self.play(Write(text))
        self.wait(1)
```

For more information, visit: https://docs.manim.community/
'''
    
    readme_path = os.path.join(docs_path, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created {len(core_docs) + 1} sample documentation files")

def create_vector_store_with_progress():
    """Create and populate the vector store with progress tracking."""
    
    print("Initializing Enhanced RAG Vector Store...")
    print(f"ChromaDB Path: {Config.CHROMA_DB_PATH}")
    print(f"Manim Docs Path: {Config.MANIM_DOCS_PATH}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"Expected Embedding Dimensions: 384")
    
    # Ensure ChromaDB directory exists
    os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Create vector store - this will automatically process documents and create embeddings
        vector_store = RAGVectorStore(
            chroma_db_path=Config.CHROMA_DB_PATH,
            manim_docs_path=Config.MANIM_DOCS_PATH,
            embedding_model=Config.EMBEDDING_MODEL,
            session_id="embedding_creation",
            use_langfuse=False
        )
        
        print("Vector store initialized successfully!")
        
        # Check document count
        if hasattr(vector_store, 'core_vector_store') and vector_store.core_vector_store:
            try:
                collection = vector_store.core_vector_store._collection
                doc_count = collection.count()
                print(f"Total documents in vector store: {doc_count}")
                
                if doc_count > 0:
                    # Test the embedding dimensions
                    embedding_function = vector_store._get_embedding_function()
                    test_embedding = embedding_function.embed_query("test query")
                    print(f"Embedding dimensions: {len(test_embedding)}")
                    
                    # Get sample documents
                    sample_docs = collection.peek(limit=3)
                    print(f"Sample document IDs: {sample_docs.get('ids', [])[:3]}")
                    
                    # Test search functionality
                    print("\nTesting search functionality...")
                    results = vector_store.core_vector_store.similarity_search_with_relevance_scores(
                        query="How to create a circle?",
                        k=3,
                        score_threshold=0.0
                    )

                    print(f"Search results: {len(results)} documents found")
                    for i, (doc, score) in enumerate(results):
                        print(f"   [{i+1}] Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")
                        print(f"       Content: {doc.page_content[:100]}...")
                else:
                    print("Warning: No documents found in vector store")
                    
            except Exception as e:
                print(f"Error checking document count: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"\nVector store creation completed in {elapsed_time:.2f} seconds")

        return vector_store
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_embeddings():
    """Verify that embeddings were created successfully."""
    
    print("\nVerifying embeddings...")
    
    try:
        # Load the vector store
        vector_store = RAGVectorStore(
            chroma_db_path=Config.CHROMA_DB_PATH,
            manim_docs_path=Config.MANIM_DOCS_PATH,
            embedding_model=Config.EMBEDDING_MODEL,
            session_id="verification",
            use_langfuse=False
        )
        
        # Test queries
        test_queries = [
            "How to create a circle in Manim?",
            "Text animation examples",
            "Mathematical plotting with axes",
            "Scene construction basics"
        ]
        
        print("Testing search with various queries:")
        
        for query in test_queries:
            results = vector_store.core_vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=2,
                score_threshold=0.0
            )
            
            print(f"\nQuery: '{query}'")
            print(f"Results: {len(results)} documents")
            
            for i, (doc, score) in enumerate(results):
                print(f"  [{i+1}] Score: {score:.4f}")
                print(f"      Source: {doc.metadata.get('source', 'unknown')}")
                print(f"      Content: {doc.page_content[:80]}...")
        
        print("\nEmbedding verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"Embedding verification failed: {e}")
        return False

def main():
    """Main function to create embeddings."""
    
    print("MANIM RAG EMBEDDING CREATION")
    print("="*50)
    
    # Step 1: Download/create documentation
    download_manim_docs()
    
    # Step 2: Create vector store and embeddings
    vector_store = create_vector_store_with_progress()
    
    if vector_store is None:
        print("❌ Failed to create vector store. Exiting.")
        return False
    
    # Step 3: Verify embeddings
    success = verify_embeddings()
    
    if success:
        print("\nSUCCESS! Embeddings created and verified.")
        print(f" ChromaDB location: {Config.CHROMA_DB_PATH}")
        print("You can now run the RAG system tests.")
        print("\nNext steps:")
        print("  1. Run: python test_rag_system.py")
        print("  2. Run: python debug_retrieval.py")
    else:
        print("\n FAILED! Something went wrong during verification.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
