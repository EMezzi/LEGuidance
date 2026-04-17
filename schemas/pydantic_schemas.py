from pydantic import BaseModel, Field
from typing import Literal, Optional, List

# Criteria extraction

# ---- Enums / literals ----
EntityType = Literal[
    "person", "organization", "team", "award", "league", "event", "work", "place", "other"
]

ConstraintKind = Literal[
    "relation",  # e.g., "played for", "won", "born in"
    "award",  # award name constraint
    "league",  # league/division constraint
    "season",  # season constraint (if you also keep it as a constraint)
    "time",  # time window constraint, non-season date/year too
    "role",  # role/title constraint
    "qualifier",  # "career statistics", "during", "when looking at", etc.
    "domain",  # domain-specific context if needed
    "other"
]

ConstraintTopic = Literal[
    "Film",
    "Transportation",
    "Video games",
    "Industry",
    "Theater",
    "Television",
    "Music",
    "Geography",
    "Literature",
    "History",
    "Economy",
    "Sports",
    "Science",
    "Politics",
    "Buildings",
    "Other"
]


class QuestionTopic(BaseModel):
    question_topic: ConstraintTopic = Field(..., description="Topic of the question")


class AnswerSubject(BaseModel):
    expected_answer_type_specific: str = Field(..., description="Specific type of the expected answer")
    expected_answer_type_general: str = Field(..., description="Generic type of the expected answer")


class Target(BaseModel):
    text: str = Field(..., description="Main entity the question is about.")
    # type: #EntityType = Field(..., description="Type of the target entity.")
    type: str = Field(default=None, description="Type of target entity.")


class Constraint(BaseModel):
    kind: ConstraintKind = Field(..., description="Constraint category.")
    evidence: str = Field(..., description="Exact phrase from the question.")
    normalized: str = Field(..., description="Cleaned/normalized version of evidence.")


class TimeConstraint(BaseModel):
    label: str = Field(..., description="Time phrase from the question (or normalized label).")
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    start_date: Optional[str] = Field(default=None, description="ISO-8601 if explicit (YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, description="ISO-8601 if explicit (YYYY-MM-DD).")


class Alias(BaseModel):
    text: str
    reason: str  # "typo", "nickname", "alternative spelling", "abbreviation", ...


class DistinctionCriteria(BaseModel):
    topic: QuestionTopic
    expected_answer_type: AnswerSubject
    expected_cardinality: Literal["single", "multiple", "unknown"] = "single"

    target: Target
    asked_property: str = Field(..., description="Short predicate describing what is asked (e.g., 'team played for').")

    constraints: List[Constraint] = Field(default_factory=list)
    time_constraints: List[TimeConstraint] = Field(default_factory=list)
    aliases: List[Alias] = Field(default_factory=list)

    rewritten_question: str = Field(..., description="Minimal rewrite preserving meaning.")


# Modality decision
class ModalityDecision(BaseModel):
    modalities: Literal[
        "image",
        "text",
        "table",
        "image_text",
        "image_table",
        "text_table",
    ] = Field(..., description="Predicted modality combination")


class YesNoQuestion(BaseModel):
    is_yes_no: bool = Field(..., description="Whether the question is a yes or no question")
    confidence: float = Field(..., ge=0, le=1)


class IsComparison(BaseModel):
    is_comparison: bool = Field(..., description="Whether the question requires a comparison")
    num_elements: int = Field(..., ge=0, description="Number of elements/entities that must be compared")
    confidence: float = Field(..., ge=0, le=1)


class IsGraphical(BaseModel):
    is_graphical: bool = Field(..., description="Whether you need to analyse a graphical element to answer a question")
    confidence: float = Field(..., ge=0, le=1)


# Logical Entropy
class AnswerContainsCriteria(BaseModel):
    answer: str = Field(..., description="Answer yes or no to whether the data contains the criteria")


class TableDescription(BaseModel):
    description: str = Field(..., descripton="High level description of the table")


class ParagraphExtraction(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Whether the paragraph contains information relevant to the question."
    )

    evidence: Optional[str] = Field(
        None,
        description=(
            "A minimal, precise textual description of the paragraph that directly "
            "connects it to the question. Must only include explicitly stated information. "
            "Should be None if is_relevant is False."
        )
    )


class ImageExtraction(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Whether the image contains information relevant to the question."
    )

    evidence: Optional[str] = Field(
        None,
        description=(
            "A minimal, precise textual description of the image that directly "
            "connects it to the question. Must only include explicitly stated information. "
            "Should be None if is_relevant is False."
        )
    )


class TableRowExtraction(BaseModel):
    is_relevant: bool = Field(
        ...,
        description="Whether the table row contains information relevant to the question."
    )

    evidence: Optional[str] = Field(
        None,
        description=(
            "A minimal, precise textual description of the table row that directly "
            "connects it to the question. Must only include explicitly stated information. "
            "Should be None if is_relevant is False."
        )
    )


class ImageContainsAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the image describes an entity matching the expected answer type")
    entity: str = Field(..., description="The answer for the question extracted from the image")
    match_level: Literal["specific", "general", "none"]
    confidence: float = Field(..., ge=0, le=1)


class ParagraphContainsAnswer(BaseModel):
    contains: bool = Field(...,
                           description="Whether the paragraph contains an element matching the expected answer type")
    entity: str = Field(...,
                        description="The exact text span from the paragraph that matches the expected answer type, or NONE")
    confidence: float = Field(..., ge=0, le=1)


class RowContainsAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the row contains an element matching the expected answer type")
    entity: str = Field(...,
                        description="The exact text span from the row cell that matches the expected answer type, or NONE")
    confidence: float = Field(..., ge=0, le=1)


class DPAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the given data contains answer to the question")
    entity: str = Field(..., description="The answer to the question if present otherwise NONE")
    confidence: float = Field(..., ge=0, le=1)


class CoTAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the given data contains answer to the question")
    reasoning: bool = Field(..., description="The reasoning to arrive at the final answer")
    entity: str = Field(..., description="The answer to the question if present otherwise NONE")
    confidence: float = Field(..., ge=0, le=1)


class PPAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the given data contains answer to the question")
    plan: str = Field(..., description="The high-level strategy (Phase 1)")
    execution: str = Field(..., description="Step-by-step extraction and logic (Phase 2)")
    entity: str = Field(..., description="The final concise answer or NONE if answer is not found")
    confidence: float = Field(..., ge=0, le=1)
