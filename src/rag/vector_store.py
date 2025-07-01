import json
import os
import ast
from typing import List, Dict, Tuple, Optional
import uuid
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import Language
from langchain_core.embeddings import Embeddings
import statistics
import tiktoken
from tqdm import tqdm
from langfuse import Langfuse
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

from mllm_tools.utils import _prepare_text_inputs
from task_generator import get_prompt_detect_plugins

class CodeAwareTextSplitter:
    """Enhanced text splitter that understands code structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_python_file(self, content: str, metadata: dict) -> List[Document]:
        """Split Python files preserving code structure."""
        documents = []
        
        try:
            tree = ast.parse(content)
            
            # Extract classes and functions with their docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get the source code segment
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 20)
                    
                    lines = content.split('\n')
                    code_segment = '\n'.join(lines[start_line-1:end_line])
                    
                    # Extract docstring
                    docstring = ast.get_docstring(node) or ""
                    
                    # Create enhanced content
                    enhanced_content = f"""
Type: {"Class" if isinstance(node, ast.ClassDef) else "Function"}
Name: {node.name}
Docstring: {docstring}

Code:
```python
{code_segment}
```
                    """.strip()
                    
                    # Enhanced metadata
                    enhanced_metadata = {
                        **metadata,
                        'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                        'name': node.name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'has_docstring': bool(docstring),
                        'docstring': docstring[:200] + "..." if len(docstring) > 200 else docstring
                    }
                    
                    documents.append(Document(
                        page_content=enhanced_content,
                        metadata=enhanced_metadata
                    ))
            
            # Also create chunks for imports and module-level code
            imports_and_constants = self._extract_imports_and_constants(content)
            if imports_and_constants:
                documents.append(Document(
                    page_content=f"Module-level imports and constants:\n\n{imports_and_constants}",
                    metadata={**metadata, 'type': 'module_level', 'name': 'imports_constants'}
                ))
                
        except SyntaxError:
            # Fallback to regular text splitting for invalid Python
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            documents = splitter.split_documents([Document(page_content=content, metadata=metadata)])
        
        return documents
    
    def split_markdown_file(self, content: str, metadata: dict) -> List[Document]:
        """Split Markdown files preserving structure."""
        documents = []
        
        # Split by headers while preserving hierarchy
        sections = self._split_by_headers(content)
        
        for section in sections:
            # Extract code blocks
            code_blocks = self._extract_code_blocks(section['content'])
            
            # Create document for text content
            text_content = self._remove_code_blocks(section['content'])
            if text_content.strip():
                enhanced_metadata = {
                    **metadata,
                    'type': 'markdown_section',
                    'header': section['header'],
                    'level': section['level'],
                    'has_code_blocks': len(code_blocks) > 0
                }
                
                documents.append(Document(
                    page_content=f"Header: {section['header']}\n\n{text_content}",
                    metadata=enhanced_metadata
                ))
            
            # Create separate documents for code blocks
            for i, code_block in enumerate(code_blocks):
                enhanced_metadata = {
                    **metadata,
                    'type': 'code_block',
                    'language': code_block['language'],
                    'in_section': section['header'],
                    'block_index': i
                }
                
                documents.append(Document(
                    page_content=f"Code example in '{section['header']}':\n\n```{code_block['language']}\n{code_block['code']}\n```",
                    metadata=enhanced_metadata
                ))
        
        return documents
    
    def _extract_imports_and_constants(self, content: str) -> str:
        """Extract imports and module-level constants."""
        lines = content.split('\n')
        relevant_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                (stripped and not stripped.startswith('def ') and 
                 not stripped.startswith('class ') and
                 not stripped.startswith('#') and
                 '=' in stripped and stripped.split('=')[0].strip().isupper())):
                relevant_lines.append(line)
        
        return '\n'.join(relevant_lines)
    
    def _split_by_headers(self, content: str) -> List[Dict]:
        """Split markdown content by headers."""
        sections = []
        lines = content.split('\n')
        current_section = {'header': 'Introduction', 'level': 0, 'content': ''}
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {'header': header, 'level': level, 'content': ''}
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2)
            code_blocks.append({'language': language, 'code': code})
        
        return code_blocks
    
    def _remove_code_blocks(self, content: str) -> str:
        """Remove code blocks from content."""
        pattern = r'```\w*\n.*?\n```'
        return re.sub(pattern, '', content, flags=re.DOTALL)

class EnhancedRAGVectorStore:
    """Enhanced RAG vector store with improved code understanding."""
    
    def __init__(self, 
                 chroma_db_path: str = "chroma_db",
                 manim_docs_path: str = "rag/manim_docs",
                 embedding_model: str = "hf:ibm-granite/granite-embedding-30m-english",
                 trace_id: str = None,
                 session_id: str = None,
                 use_langfuse: bool = True,
                 helper_model = None):
        self.chroma_db_path = chroma_db_path
        self.manim_docs_path = manim_docs_path
        self.embedding_model = embedding_model
        self.trace_id = trace_id
        self.session_id = session_id
        self.use_langfuse = use_langfuse
        self.helper_model = helper_model
        self.enc = tiktoken.encoding_for_model("gpt-4")
        self.plugin_stores = {}
        self.code_splitter = CodeAwareTextSplitter()
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """Enhanced vector store creation with better document processing."""
        print("Creating enhanced vector store with code-aware processing...")
        core_path = os.path.join(self.chroma_db_path, "manim_core_enhanced")
        
        if os.path.exists(core_path):
            print("Loading existing enhanced ChromaDB...")
            self.core_vector_store = Chroma(
                collection_name="manim_core_enhanced",
                persist_directory=core_path,
                embedding_function=self._get_embedding_function()
            )
        else:
            print("Creating new enhanced ChromaDB...")
            self.core_vector_store = self._create_enhanced_core_store()
        
        # Process plugins with enhanced splitting
        plugin_docs_path = os.path.join(self.manim_docs_path, "plugin_docs")
        if os.path.exists(plugin_docs_path):
            for plugin_name in os.listdir(plugin_docs_path):
                plugin_store_path = os.path.join(self.chroma_db_path, f"manim_plugin_{plugin_name}_enhanced")
                if os.path.exists(plugin_store_path):
                    print(f"Loading existing enhanced plugin store: {plugin_name}")
                    self.plugin_stores[plugin_name] = Chroma(
                        collection_name=f"manim_plugin_{plugin_name}_enhanced",
                        persist_directory=plugin_store_path,
                        embedding_function=self._get_embedding_function()
                    )
                else:
                    print(f"Creating new enhanced plugin store: {plugin_name}")
                    plugin_path = os.path.join(plugin_docs_path, plugin_name)
                    if os.path.isdir(plugin_path):
                        plugin_store = Chroma(
                            collection_name=f"manim_plugin_{plugin_name}_enhanced",
                            embedding_function=self._get_embedding_function(),
                            persist_directory=plugin_store_path
                        )
                        plugin_docs = self._process_documentation_folder_enhanced(plugin_path)
                        if plugin_docs:
                            self._add_documents_to_store(plugin_store, plugin_docs, plugin_name)
                        self.plugin_stores[plugin_name] = plugin_store
        
        return self.core_vector_store

    def _get_embedding_function(self) -> Embeddings:
        """Enhanced embedding function with better model selection."""
        if self.embedding_model.startswith('hf:'):
            model_name = self.embedding_model[3:]
            print(f"Using HuggingFaceEmbeddings with model: {model_name}")
            
            # Use better models for code understanding
            if 'code' not in model_name.lower():
                print("Consider using a code-specific embedding model like 'microsoft/codebert-base'")
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            raise ValueError("Only HuggingFace embeddings are supported in this configuration.")

    def _create_enhanced_core_store(self):
        """Create enhanced core store with better document processing."""
        core_vector_store = Chroma(
            collection_name="manim_core_enhanced",
            embedding_function=self._get_embedding_function(),
            persist_directory=os.path.join(self.chroma_db_path, "manim_core_enhanced")
        )
        
        core_docs = self._process_documentation_folder_enhanced(
            os.path.join(self.manim_docs_path, "manim_core")
        )
        if core_docs:
            self._add_documents_to_store(core_vector_store, core_docs, "manim_core_enhanced")
        
        return core_vector_store

    def _process_documentation_folder_enhanced(self, folder_path: str) -> List[Document]:
        """Enhanced document processing with code-aware splitting."""
        all_docs = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.md', '.py')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        base_metadata = {
                            'source': file_path,
                            'filename': file,
                            'file_type': 'python' if file.endswith('.py') else 'markdown',
                            'relative_path': os.path.relpath(file_path, folder_path)
                        }
                        
                        if file.endswith('.py'):
                            docs = self.code_splitter.split_python_file(content, base_metadata)
                        else:  # .md files
                            docs = self.code_splitter.split_markdown_file(content, base_metadata)
                        
                        # Add source prefix to content
                        for doc in docs:
                            doc.page_content = f"Source: {file_path}\nType: {doc.metadata.get('type', 'unknown')}\n\n{doc.page_content}"
                        
                        all_docs.extend(docs)
                        
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
        
        print(f"Processed {len(all_docs)} enhanced document chunks from {folder_path}")
        return all_docs

    def _add_documents_to_store(self, vector_store: Chroma, documents: List[Document], store_name: str):
        """Enhanced document addition with better batching."""
        print(f"Adding {len(documents)} enhanced documents to {store_name} store")
        
        # Group documents by type for better organization
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        print(f"Document types distribution: {dict((k, len(v)) for k, v in doc_types.items())}")
        
        # Calculate token statistics
        token_lengths = [len(self.enc.encode(doc.page_content)) for doc in documents]
        print(f"Token length statistics for {store_name}: "
              f"Min: {min(token_lengths)}, Max: {max(token_lengths)}, "
              f"Mean: {sum(token_lengths) / len(token_lengths):.1f}, "
              f"Median: {statistics.median(token_lengths):.1f}")
        
        batch_size = 10
        for i in tqdm(range(0, len(documents), batch_size), desc=f"Processing {store_name} enhanced batches"):
            batch_docs = documents[i:i + batch_size]
            batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        
        vector_store.persist()

    def find_relevant_docs(self, queries: List[Dict], k: int = 5, trace_id: str = None, topic: str = None, scene_number: int = None) -> str:
        """Find relevant documents - compatibility method that calls the enhanced version."""
        return self.find_relevant_docs_enhanced(queries, k, trace_id, topic, scene_number)

    def find_relevant_docs_enhanced(self, queries: List[Dict], k: int = 5, trace_id: str = None, topic: str = None, scene_number: int = None) -> str:
        """Enhanced document retrieval with type-aware search."""
        # Separate queries by intent
        code_queries = [q for q in queries if any(keyword in q["query"].lower() 
                       for keyword in ["function", "class", "method", "import", "code", "implementation"])]
        concept_queries = [q for q in queries if q not in code_queries]
        
        all_results = []
        
        # Search with different strategies for different query types
        for query in code_queries:
            results = self._search_with_filters(
                query["query"], 
                k=k, 
                filter_metadata={'type': ['function', 'class', 'code_block']},
                boost_code=True
            )
            all_results.extend(results)
        
        for query in concept_queries:
            results = self._search_with_filters(
                query["query"], 
                k=k, 
                filter_metadata={'type': ['markdown_section', 'module_level']},
                boost_code=False
            )
            all_results.extend(results)
        
        # Remove duplicates and format results
        unique_results = self._remove_duplicates(all_results)
        return self._format_results(unique_results)
    
    def _search_with_filters(self, query: str, k: int, filter_metadata: Dict = None, boost_code: bool = False) -> List[Dict]:
        """Search with metadata filters and result boosting."""
        # This is a simplified version - in practice, you'd implement proper filtering
        core_results = self.core_vector_store.similarity_search_with_relevance_scores(
            query=query, k=k, score_threshold=0.3
        )
        
        formatted_results = []
        for result in core_results:
            doc, score = result
            # Boost scores for code-related results if needed
            if boost_code and doc.metadata.get('type') in ['function', 'class', 'code_block']:
                score *= 1.2
            
            formatted_results.append({
                "query": query,
                "source": doc.metadata['source'],
                "content": doc.page_content,
                "score": score,
                "type": doc.metadata.get('type', 'unknown'),
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            content_hash = hash(result['content'][:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                unique_results.append(result)
                seen_content.add(content_hash)
        
        return unique_results[:10]  # Return top 10 unique results
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format results with enhanced presentation."""
        if not results:
            return "No relevant documentation found."
        
        formatted = "## Relevant Documentation\n\n"
        
        # Group by type
        by_type = {}
        for result in results:
            result_type = result['type']
            if result_type not in by_type:
                by_type[result_type] = []
            by_type[result_type].append(result)
        
        for result_type, type_results in by_type.items():
            formatted += f"### {result_type.replace('_', ' ').title()} Documentation\n\n"
            
            for result in type_results:
                formatted += f"**Source:** {result['source']}\n"
                formatted += f"**Relevance Score:** {result['score']:.3f}\n"
                formatted += f"**Content:**\n```\n{result['content'][:500]}...\n```\n\n"
        
        return formatted

# Update the existing RAGVectorStore class alias for backward compatibility
RAGVectorStore = EnhancedRAGVectorStore