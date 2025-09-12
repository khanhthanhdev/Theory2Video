import json
import re
import os
try:
    from pylatexenc.latexencode import utf8tolatex, UnicodeToLatexEncoder
except:
    print("Warning: Missing pylatexenc, please do pip install pylatexenc")

def _print_response(response_type: str, theorem_name: str, content: str, separator: str = "=" * 50) -> None:
    """Print formatted responses from the video generation process.

    Prints a formatted response with separators and headers for readability.

    Args:
        response_type (str): Type of response (e.g., 'Scene Plan', 'Implementation Plan')
        theorem_name (str): Name of the theorem being processed
        content (str): The content to print
        separator (str, optional): Separator string for visual distinction. Defaults to 50 equals signs.

    Returns:
        None
    """
    print(f"\n{separator}")
    print(f"{response_type} for {theorem_name}:")
    print(f"{separator}\n")
    print(content)
    print(f"\n{separator}")

def _extract_code(response_text: str) -> str:
    """Extract code blocks from a text response.

    Extracts Python code blocks delimited by ```python markers. If no code blocks are found,
    returns the entire response text.

    Args:
        response_text (str): The text response containing code blocks

    Returns:
        str: The extracted code blocks joined by newlines, or the full response if no blocks found
    """
    code = ""
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        code = "\n\n".join(code_blocks)
    elif "```" not in response_text: # if no code block, return the whole response
        code = response_text
    return code 

def extract_json(response: str):
    """Extract and parse JSON content from a text response.

    Strategy (best-effort, robust):
    1) Try json.loads on the whole response.
    2) Try ```json fenced block.
    3) Try generic ``` fenced block.
    4) Try to locate the first JSON object/array substring by bracket matching and parse that.

    Returns dict or list on success; [] on failure (backward compatible falsy value).
    """
    def _try_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # 1) Direct
    data = _try_load(response)
    if data is not None:
        return data

    # 2) ```json fenced
    match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        data = _try_load(match.group(1))
        if data is not None:
            return data

    # 3) generic fenced
    match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        data = _try_load(match.group(1))
        if data is not None:
            return data

    # 4) Bracket matching for object or array
    def _extract_first_json_substring(s: str) -> str:
        # Find first '{' or '[' and try to capture matching close using a simple stack
        start = None
        opener = None
        for i, ch in enumerate(s):
            if ch == '{' or ch == '[':
                start = i
                opener = ch
                break
        if start is None:
            return None
        closer = '}' if opener == '{' else ']'
        depth = 0
        in_str = False
        escape = False
        for j in range(start, len(s)):
            c = s[j]
            if in_str:
                if escape:
                    escape = False
                elif c == '\\':
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return s[start:j+1]
        return None

    candidate = _extract_first_json_substring(response)
    if candidate:
        data = _try_load(candidate)
        if data is not None:
            return data

    if os.environ.get("T2V_DEBUG_JSON") == "1":
        print("Warning: Failed to extract valid JSON content from response (showing first 200 chars):", response[:200])
    return []

def parse_batched_scenes(response_text: str, key_field: str, expected_scene_order=None):
    """Parse batched per-scene outputs from a single LLM response.

    - Tries to parse strict JSON with shape {"scenes": [{"scene_number": int, key_field: str}, ...]}
    - Accepts aliases for scene id: scene_number | scene | n
    - If JSON parse fails, falls back to regex tag extraction for known plan tags and
      aligns blocks with expected_scene_order if provided.

    Args:
        response_text: Raw LLM response
        key_field: One of 'vision_storyboard', 'technical_implementation', 'animation_narration'
        expected_scene_order: Optional list of scene numbers indicating intended order

    Returns:
        Dict[int, str]: Mapping from scene_number -> content
    """
    mapping = {}
    data = extract_json(response_text)
    if isinstance(data, dict) and isinstance(data.get('scenes'), list):
        for item in data['scenes']:
            sn = item.get('scene_number') or item.get('scene') or item.get('n')
            content = item.get(key_field)
            if isinstance(sn, int) and isinstance(content, str) and content.strip():
                mapping[sn] = content
    if mapping:
        return mapping

    # Fallback to tag-based extraction
    tag_patterns = {
        'vision_storyboard': r'(<SCENE_VISION_STORYBOARD_PLAN>.*?</SCENE_VISION_STORYBOARD_PLAN>)',
        'technical_implementation': r'(<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>.*?</SCENE_TECHNICAL_IMPLEMENTATION_PLAN>)',
        'animation_narration': r'(<SCENE_ANIMATION_NARRATION_PLAN>.*?</SCENE_ANIMATION_NARRATION_PLAN>)'
    }
    pattern = tag_patterns.get(key_field)
    if pattern:
        blocks = re.findall(pattern, response_text, re.DOTALL)
        if blocks:
            if os.environ.get("T2V_DEBUG_JSON") == "1":
                print(f"Info: Tag-based fallback extracted {len(blocks)} blocks for {key_field}")
            if expected_scene_order:
                for idx, sn in enumerate(expected_scene_order):
                    if idx < len(blocks):
                        mapping[sn] = blocks[idx]
            else:
                # Assign sequentially starting from 1
                for i, block in enumerate(blocks, start=1):
                    mapping[i] = block
    return mapping

def _fix_unicode_to_latex(text: str, parse_unicode: bool = True) -> str:
    """Convert Unicode symbols to LaTeX source code.

    Converts Unicode subscripts and superscripts to LaTeX format, with optional full Unicode parsing.

    Args:
        text (str): The text containing Unicode symbols to convert
        parse_unicode (bool, optional): Whether to perform full Unicode to LaTeX conversion. Defaults to True.

    Returns:
        str: The text with Unicode symbols converted to LaTeX format
    """
    # Map of unicode subscripts to latex format
    subscripts = {
        "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
        "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
        "₊": "_+", "₋": "_-"
    }
    # Map of unicode superscripts to latex format  
    superscripts = {
        "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
        "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
        "⁺": "^+", "⁻": "^-"
    }

    for unicode_char, latex_format in {**subscripts, **superscripts}.items():
        text = text.replace(unicode_char, latex_format)

    if parse_unicode:
        text = utf8tolatex(text)

    return text

def extract_xml(response: str) -> str:
    """Extract XML content from a text response.

    Extracts XML content between ```xml markers. Returns the full response if no XML blocks found.

    Args:
        response (str): The text response containing XML content

    Returns:
        str: The extracted XML content, or the full response if no XML blocks found
    """
    try:
        match = re.search(r'```xml\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return response
    except Exception:
        return response
