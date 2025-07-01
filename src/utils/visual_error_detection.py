"""
Visual Error Detection Utilities for Manim Code Analysis

This module provides utilities for detecting and analyzing visual errors in Manim animations,
specifically focusing on element overlap, positioning issues, and spatial constraint violations.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Visual error detection patterns
VISUAL_ERROR_PATTERNS = {
    'overlap_keywords': [
        'overlap', 'overlapping', 'collision', 'colliding', 'obscured', 'hidden',
        'blocked', 'covering', 'covered', 'behind', 'on top of'
    ],
    'boundary_keywords': [
        'out of bounds', 'outside frame', 'clipped', 'cut off', 'beyond edge',
        'outside safe area', 'margin violation', 'boundary violation'
    ],
    'spacing_keywords': [
        'too close', 'insufficient spacing', 'cramped', 'crowded', 'bunched up',
        'spacing violation', 'minimum distance', 'tight spacing'
    ],
    'positioning_keywords': [
        'misaligned', 'mispositioned', 'wrong position', 'incorrect placement',
        'poor arrangement', 'bad layout', 'disorganized'
    ]
}

# Critical visual issues that require immediate fixing
CRITICAL_VISUAL_ISSUES = [
    'text completely obscured',
    'formula unreadable',
    'important element hidden',
    'content outside frame',
    'major overlap',
    'critical positioning error'
]

# Safe area and spacing constraints (Manim units)
VISUAL_CONSTRAINTS = {
    'safe_area_margin': 0.5,  # Units from frame edge
    'minimum_spacing': 0.3,   # Units between elements
    'frame_width': 14.22,     # Manim frame width
    'frame_height': 8.0,      # Manim frame height
    'center_x': 0.0,          # Frame center X
    'center_y': 0.0,          # Frame center Y
    'x_bounds': (-7.0, 7.0),  # Safe X coordinate range
    'y_bounds': (-4.0, 4.0)   # Safe Y coordinate range
}

class VisualErrorDetector:
    """Utility class for detecting and categorizing visual errors in VLM responses."""
    
    def __init__(self):
        self.error_patterns = VISUAL_ERROR_PATTERNS
        self.critical_issues = CRITICAL_VISUAL_ISSUES
        self.constraints = VISUAL_CONSTRAINTS
    
    def detect_error_types(self, analysis_text: str) -> Dict[str, List[str]]:
        """
        Detect different types of visual errors from VLM analysis text.
        
        Args:
            analysis_text: Raw text from VLM visual analysis
            
        Returns:
            Dictionary categorizing detected errors by type
        """
        errors = {
            'overlap_errors': [],
            'boundary_errors': [],
            'spacing_errors': [],
            'positioning_errors': [],
            'critical_errors': []
        }
        
        analysis_lower = analysis_text.lower()
        
        # Check for overlap errors
        for keyword in self.error_patterns['overlap_keywords']:
            if keyword in analysis_lower:
                errors['overlap_errors'].append(self._extract_error_context(analysis_text, keyword))
        
        # Check for boundary errors
        for keyword in self.error_patterns['boundary_keywords']:
            if keyword in analysis_lower:
                errors['boundary_errors'].append(self._extract_error_context(analysis_text, keyword))
        
        # Check for spacing errors
        for keyword in self.error_patterns['spacing_keywords']:
            if keyword in analysis_lower:
                errors['spacing_errors'].append(self._extract_error_context(analysis_text, keyword))
        
        # Check for positioning errors
        for keyword in self.error_patterns['positioning_keywords']:
            if keyword in analysis_lower:
                errors['positioning_errors'].append(self._extract_error_context(analysis_text, keyword))
        
        # Check for critical issues
        for issue in self.critical_issues:
            if issue in analysis_lower:
                errors['critical_errors'].append(self._extract_error_context(analysis_text, issue))
        
        # Remove empty entries and duplicates
        for error_type in errors:
            errors[error_type] = list(set([e for e in errors[error_type] if e]))
        
        return errors
    
    def _extract_error_context(self, text: str, keyword: str, context_length: int = 100) -> str:
        """
        Extract context around a detected error keyword.
        
        Args:
            text: Full analysis text
            keyword: Error keyword found
            context_length: Characters to include around keyword
            
        Returns:
            Context string around the error keyword
        """
        try:
            # Find keyword position (case insensitive)
            lower_text = text.lower()
            keyword_pos = lower_text.find(keyword.lower())
            
            if keyword_pos == -1:
                return keyword
            
            # Extract context around keyword
            start = max(0, keyword_pos - context_length // 2)
            end = min(len(text), keyword_pos + len(keyword) + context_length // 2)
            
            context = text[start:end].strip()
            
            # Clean up context
            context = re.sub(r'\s+', ' ', context)
            
            return context
        except Exception as e:
            logger.warning(f"Error extracting context for keyword '{keyword}': {e}")
            return keyword
    
    def categorize_severity(self, errors: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Categorize the severity of detected visual errors.
        
        Args:
            errors: Dictionary of detected errors by type
            
        Returns:
            Dictionary mapping error types to severity levels
        """
        severity_map = {}
        
        # Critical errors are always high severity
        if errors['critical_errors']:
            severity_map['critical'] = 'HIGH'
        
        # Overlap errors can vary in severity
        if errors['overlap_errors']:
            # Check if any overlap errors mention important elements
            important_keywords = ['text', 'formula', 'equation', 'title', 'label']
            has_important_overlap = any(
                any(keyword in error.lower() for keyword in important_keywords)
                for error in errors['overlap_errors']
            )
            severity_map['overlap'] = 'HIGH' if has_important_overlap else 'MEDIUM'
        
        # Boundary errors are typically medium to high severity
        if errors['boundary_errors']:
            severity_map['boundary'] = 'MEDIUM'
        
        # Spacing errors are usually low to medium severity
        if errors['spacing_errors']:
            severity_map['spacing'] = 'LOW'
        
        # Positioning errors vary based on context
        if errors['positioning_errors']:
            severity_map['positioning'] = 'MEDIUM'
        
        return severity_map
    
    def generate_fix_suggestions(self, errors: Dict[str, List[str]]) -> List[str]:
        """
        Generate specific code fix suggestions based on detected errors.
        
        Args:
            errors: Dictionary of detected errors by type
            
        Returns:
            List of specific fix suggestions
        """
        suggestions = []
        
        if errors['overlap_errors']:
            suggestions.extend([
                "Use `.next_to()` method to position elements relative to each other with proper spacing",
                "Apply `buff` parameter in positioning methods to ensure minimum 0.3 unit spacing",
                "Reorganize elements into VGroups for better spatial management",
                "Use `bring_to_front()` or `bring_to_back()` to manage z-order layering"
            ])
        
        if errors['boundary_errors']:
            suggestions.extend([
                "Ensure all elements are positioned within safe area bounds (-7 to 7 for X, -4 to 4 for Y)",
                "Use `move_to(ORIGIN)` and then apply relative positioning to keep elements centered",
                "Check element sizes and scale them down if they extend beyond frame boundaries",
                "Apply safe area margins of 0.5 units from frame edges"
            ])
        
        if errors['spacing_errors']:
            suggestions.extend([
                "Use `buff=0.3` or higher in `.next_to()` methods for proper spacing",
                "Apply `.shift()` method to adjust element positions for better spacing",
                "Consider using `.arrange()` method for VGroups to maintain consistent spacing",
                "Verify minimum 0.3 unit spacing between all visual elements"
            ])
        
        if errors['positioning_errors']:
            suggestions.extend([
                "Use relative positioning methods exclusively: `.next_to()`, `.align_to()`, `.shift()`",
                "Position elements relative to ORIGIN, other objects, or scene margins",
                "Ensure logical flow and visual hierarchy in element arrangement",
                "Group related elements using VGroup for coordinated positioning"
            ])
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def validate_manim_constraints(self, code: str) -> Dict[str, List[str]]:
        """
        Validate Manim code against spatial constraints.
        
        Args:
            code: Manim code to validate
            
        Returns:
            Dictionary of constraint violations found in code
        """
        violations = {
            'absolute_coordinates': [],
            'unsafe_positioning': [],
            'missing_spacing': [],
            'out_of_bounds': []
        }
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for absolute coordinates (potential issues)
            if re.search(r'move_to\s*\(\s*[-+]?\d+\.?\d*\s*,\s*[-+]?\d+\.?\d*', line):
                violations['absolute_coordinates'].append(f"Line {i}: {line.strip()}")
            
            # Check for potentially unsafe positioning
            if re.search(r'shift\s*\(\s*[^)]*[5-9]\d*', line):
                violations['unsafe_positioning'].append(f"Line {i}: Large shift detected - {line.strip()}")
            
            # Check for missing buff parameters in next_to calls
            if 'next_to' in line and 'buff' not in line:
                violations['missing_spacing'].append(f"Line {i}: Missing buff parameter - {line.strip()}")
            
            # Check for coordinates that might be out of bounds
            coord_matches = re.findall(r'[-+]?\d+\.?\d*', line)
            for coord in coord_matches:
                try:
                    val = float(coord)
                    if abs(val) > 10:  # Potentially problematic large coordinates
                        violations['out_of_bounds'].append(f"Line {i}: Large coordinate {val} - {line.strip()}")
                except ValueError:
                    continue
        
        return violations


def create_visual_fix_context(
    errors: Dict[str, List[str]], 
    suggestions: List[str], 
    constraints: Dict[str, Any]
) -> str:
    """
    Create a formatted context string for visual fix operations.
    
    Args:
        errors: Detected visual errors
        suggestions: Fix suggestions
        constraints: Visual constraints to enforce
        
    Returns:
        Formatted context string for LLM prompt
    """
    context_parts = []
    
    if any(errors.values()):
        context_parts.append("**DETECTED VISUAL ERRORS:**")
        
        for error_type, error_list in errors.items():
            if error_list:
                error_type_formatted = error_type.replace('_', ' ').title()
                context_parts.append(f"\n{error_type_formatted}:")
                for error in error_list:
                    context_parts.append(f"  - {error}")
    
    if suggestions:
        context_parts.append("\n\n**RECOMMENDED FIXES:**")
        for i, suggestion in enumerate(suggestions, 1):
            context_parts.append(f"{i}. {suggestion}")
    
    context_parts.append("\n\n**SPATIAL CONSTRAINTS TO ENFORCE:**")
    context_parts.append(f"- Safe area margin: {constraints['safe_area_margin']} units from edges")
    context_parts.append(f"- Minimum spacing: {constraints['minimum_spacing']} units between elements")
    context_parts.append(f"- X coordinate bounds: {constraints['x_bounds']}")
    context_parts.append(f"- Y coordinate bounds: {constraints['y_bounds']}")
    
    return '\n'.join(context_parts)


# Export main utilities
__all__ = [
    'VisualErrorDetector',
    'VISUAL_ERROR_PATTERNS',
    'CRITICAL_VISUAL_ISSUES', 
    'VISUAL_CONSTRAINTS',
    'create_visual_fix_context'
]
