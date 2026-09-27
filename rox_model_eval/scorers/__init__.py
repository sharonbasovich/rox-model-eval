from collections.abc import Callable

from ..types import RunOutput, ScoreBreakdown, Suite, Task
from .agent_session import score_agent_session
from .data_ops import score_data_ops
from .drafting import score_drafting
from .extraction import score_extraction, score_record
from .grounded_qa import score_grounded_qa
from .long_context import score_long_context
from .ranking import score_ranking
from .research import score_research
from .safety import score_safety
from .tool_calling import score_tool_calling

Scorer = Callable[[Suite, Task, RunOutput], ScoreBreakdown]

SCORERS: dict[str, Scorer] = {
    "extraction": score_extraction,
    "research": score_research,
    "drafting": score_drafting,
    "ranking": score_ranking,
    "grounded_qa": score_grounded_qa,
    "tool_calling": score_tool_calling,
    "long_context": score_long_context,
    "safety": score_safety,
    "agent_session": score_agent_session,
    "data_ops": score_data_ops,
}

__all__ = ["SCORERS", "Scorer", "score_record"]
