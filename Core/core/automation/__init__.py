"""
Automation sub-package.

Owns the NamedAutomation data model and the AutomationLibrary, which
stores named automations consisting of a name and an ordered sequence of
steps, and retrieves them by exact name for execution via the
Execution_Engine.

Also provides the question-paper analysis built-in automation (Req 18)
and the document-humanization built-in automation (Req 19).

Design reference: Automation_Library + Execution_Engine.
Requirements: 17.1, 17.3, 18, 19.
"""

from .models import NamedAutomation
from .automation_library import AutomationLibrary, AutomationNotFoundError
from .screen_agent import ScreenAgent, AgentResult, AgentStep
from .question_paper_analysis import (
    AUTOMATION_NAME as QUESTION_PAPER_AUTOMATION_NAME,
    AnalysisContext,
    TopicExtractor,
    QuestionPaperAnalyzer,
    run_question_paper_analysis,
    register_builtin_automation as register_question_paper_automation,
)
from .document_humanizer import (
    AUTOMATION_NAME as DOCUMENT_HUMANIZATION_AUTOMATION_NAME,
    DocumentToken,
    LaTeXParseError,
    LaTeXParser,
    HumanizationContext,
    HumanizationResult,
    ProseHumanizer,
    DocumentHumanizer,
    segment_prose,
    humanize_segments,
    run_document_humanization,
    register_builtin_automation as register_document_humanization_automation,
    PROSE_TOKEN,
    MARKUP_TOKEN,
)

__all__ = [
    "NamedAutomation",
    "AutomationLibrary",
    "AutomationNotFoundError",
    # Agentic screen control
    "ScreenAgent",
    "AgentResult",
    "AgentStep",
    # Question-paper analysis (Req 18)
    "QUESTION_PAPER_AUTOMATION_NAME",
    "AnalysisContext",
    "TopicExtractor",
    "QuestionPaperAnalyzer",
    "run_question_paper_analysis",
    "register_question_paper_automation",
    # Document humanization (Req 19)
    "DOCUMENT_HUMANIZATION_AUTOMATION_NAME",
    "DocumentToken",
    "LaTeXParseError",
    "LaTeXParser",
    "HumanizationContext",
    "HumanizationResult",
    "ProseHumanizer",
    "DocumentHumanizer",
    "segment_prose",
    "humanize_segments",
    "run_document_humanization",
    "register_document_humanization_automation",
    "PROSE_TOKEN",
    "MARKUP_TOKEN",
]
