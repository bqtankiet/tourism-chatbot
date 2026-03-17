from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class ChatResponse:
    answer: str
    query: str
    context: str
    results: List[Tuple[Any, float]]
    token_usage: Dict[str, int]
    debug: Dict[str, Any]

    def to_dict(self):
        return asdict(self)
