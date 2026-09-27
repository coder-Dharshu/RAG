"""
Backward compatibility proxy for decomposer.py -> query_analyzer.py.
"""

from typing import List, Dict, Any
from query_analyzer import decompose_and_analyze


def decompose(question: str, use_llm: bool = True) -> List[Dict[str, Any]]:
    return decompose_and_analyze(question, use_llm=use_llm)
