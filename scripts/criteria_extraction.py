from __future__ import annotations
from openai import OpenAI
from dotenv import load_dotenv

import json
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field

# ---- Enums / literals ----

AnswerType = Literal[
    "year", "date", "city", "state", "country", "person", "organization",
    "number", "percentage", "currency", "duration", "title", "other"
]

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


# ---- Core models ----

class AnswerSubject(BaseModel):
    answer_class: AnswerType = Field(
        ...,
        description="Minimal generic type of the expected answer."
    )


class Target(BaseModel):
    text: str = Field(..., description="Main entity the question is about.")
    # type: #EntityType = Field(..., description="Type of the target entity.")
    type: str = Field(default=None, description="Type fo target entity.")


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
    answer_class: AnswerSubject
    expected_cardinality: Literal["single", "multiple", "unknown"] = "single"

    target: Target
    asked_property: str = Field(..., description="Short predicate describing what is asked (e.g., 'team played for').")

    constraints: List[Constraint] = Field(default_factory=list)
    time_constraints: List[TimeConstraint] = Field(default_factory=list)
    aliases: List[Alias] = Field(default_factory=list)

    rewritten_question: str = Field(..., description="Minimal rewrite preserving meaning.")


def save_json_file(json_object, file_name: str, question_text: str, criteria_extraction_dir, iteration):
    os.makedirs(f"{criteria_extraction_dir}_{iteration}", exist_ok=True)
    path = os.path.join(f"{criteria_extraction_dir}_{iteration}", file_name)

    # Ensure .json extension
    if not path.endswith(".json"):
        path += ".json"

    # Pydantic model -> dict
    if hasattr(json_object, "model_dump"):
        print("uao")
        payload = json_object.model_dump()
        payload["original_question"] = question_text
    else:
        print("Show me")
        payload = json_object
        payload["original_question"] = question_text

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def extract_criterias(question_text, question, criteria_extraction_dir, iteration):
    response = client.responses.parse(
        model="gpt-5.2",
        input=[
            {
                "role": "system",
                "content": """You extract structured query frames from natural-language questions. Return ONLY valid 
                JSON that matches the provided schema. Do not use outside knowledge; only what is in the question."""
            },
            {
                "role": "user",
                "content": f"""
                            Return DistinctionCriteria for the question below.

                            High-level goals:
                            - Identify the answer type (answer_class.answer_class) as the minimal generic type:
                              Use ONE of: year, date, city, state, country, person, organization, number, percentage, currency, duration, title, other.
                              Do NOT include named entities or extra words.
                            
                            - Identify the target: the main entity the question is about (usually a person, organization, award, etc).
                            
                            - asked_property: write a short predicate describing what is being requested about the target
                              Examples: "year won award", "team played for", "birthplace", "population", "date founded".
                            
                            - constraints: extract atomic, independently-checkable conditions the correct answer must satisfy.
                              Prefer multiple short constraints instead of one long one.
                              Each constraint must include:
                                - kind: one of relation, award, league, season, time, role, qualifier, domain, other
                                - evidence: exact phrase from the question
                                - normalized: normalized/cleaned version of that phrase
                            
                            - time_constraints:
                              If any explicit season/year/date range is present, add a TimeConstraint.
                              Normalize seasons like "2014-15 season" -> start_year=2014, end_year=2015, label="2014-15 season".
                            
                            - aliases:
                              Add alternative spellings or suspected typos only if strongly implied by the question text (otherwise empty).
                            
                            - rewritten_question:
                              Rewrite the question by using the extracted information. Remove fluff while keep the meaning. 
                              
                            Rules:
                            - Do NOT restate the full question in constraints.
                            - Do NOT add facts that are not explicitly in the question.
                            - expected_cardinality: "single" if the question asks for one specific thing, else "multiple" or "unknown".
                            
                            Question: {question_text}
                            """
            }
        ],
        text_format=DistinctionCriteria,
    )

    distinction_criterias = response.output_parsed
    print("see: ", dir(distinction_criterias))
    print("See: ", distinction_criterias)
    save_json_file(distinction_criterias, question, question_text, criteria_extraction_dir, iteration)


if __name__ == '__main__':
    dc = DistinctionCriteria(
        answer_class=AnswerSubject(answer_class="year"),
        target=Target(text="Paris"),  # ← depends on your Target model
        asked_property="capital of",
        rewritten_question="What is the capital of France?"
    )

    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")

    client = OpenAI(api_key=OPENAI_KEY)

    # The loop is to repeat the criteria extraction multiple times
    for iteration in range(0, 1):
        for i, question in enumerate(sorted(os.listdir(QUESTIONS_MULTIMODALQA_TRAINING))[:10]):
            print("Question: ", i, question)
            json_question = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))
            question_text = json_question["question"]
            print("Question text: ", question_text)

            extract_criterias(question_text, question, CRITERIA_EXTRACTION_DIR, iteration)
