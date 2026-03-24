from typing import Literal, Optional
from pydantic import BaseModel, Field


class ModalityDecision(BaseModel):
    modalities: Literal[
        "image",
        "text",
        "table",
        "image_text",
        "image_table",
        "text_table",
    ] = Field(..., description="Predicted modality combination")


class AnswerContainsCriteria(BaseModel):
    answer: str = Field(..., description="Answer yes or no to whether the data contains the criteria")


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


class ImageContainsAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the image describes an entity matching the expected answer type")
    entity: str = Field(..., description="The answer for the question extracted from the image")
    match_level: Literal["specific", "general", "none"]
    confidence: float = Field(..., ge=0, le=1)


class RestrictionCriterias(BaseModel):
    entity: str = Field(..., description="The element that allowed to insert the element in the positive partition")
    confidence: float = Field(..., ge=0, le=1)


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


class BridgeElement(BaseModel):
    implicit_bridge_required: bool = Field(
        ...,
        description="Whether an implicit bridge element is required to answer the question"
    )
    target_type: str = Field(
        ...,
        description="Semantic category of the missing element"
    )
    condition: str = Field(
        ...,
        description="Constraint the element must satisfy"
    )
    evidence_type: Literal["text", "image", "table"] = Field(
        ...,
        description="Type of evidence under analysis"
    )
    evidence_contains_information: bool = Field(
        ...,
        description="Whether the evidence under analysis contains the implicit bridge element"
    )
    extracted_information: Optional[str] = Field(
        None,
        description="The actual information extracted from the evidence (or None if not present)"
    )
    evidence_span: Optional[str] = Field(
        None,
        description="Minimal supporting span from the evidence (text, table cell, or visual description), or None"
    )


class TableDescription(BaseModel):
    description: str = Field(..., descripton="High level description of the table")
