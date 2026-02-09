from __future__ import annotations
from miscellaneous.prompt import system_prompt_criteria, user_prompt_criteria
from miscellaneous.json_schemas import json_schema_extraction_criteria

import json
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
import random

# ---- Enums / literals ----

# AnswerType = Literal[
#    "year", "date", "city", "state", "country", "person", "organization",
#    "number", "percentage", "currency", "duration", "title", "other"
# ]

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


# ---- Core models ----


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


def save_json_file(json_object, file_name, question_text, criteria_extraction_dir, model, iteration):
    os.makedirs(f"{criteria_extraction_dir}/{model}/iteration_{iteration}", exist_ok=True)
    path = os.path.join(f"{criteria_extraction_dir}/{model}/iteration_{iteration}", file_name)

    print(path)

    # Ensure .json extension
    if not path.endswith(".json"):
        path += ".json"

    # Case 1: Pydantic model -> dict
    if hasattr(json_object, "model_dump"):
        payload = json_object.model_dump()

    # Case 2: String input (possibly JSON string)
    elif isinstance(json_object, str):
        try:
            payload = json.loads(json_object)  # parse JSON string into dict/list
        except json.JSONDecodeError:
            # fallback: save as raw text inside JSON
            payload = {"text": json_object}
    else:
        payload = json_object

    # Add original question
    if isinstance(payload, dict):
        payload["original_question"] = question_text
    else:
        payload = {"data": payload, "original_question": question_text}

    with open(path, "w", encoding="utf-8") as f:
        print("Final dump")
        print(payload)
        json.dump(payload, f, ensure_ascii=False, indent=4)



def extract_criterias(model, client, question_text, question, criteria_extraction_dir, iteration):
    # Add tho the prompt the generation of a set that contains the entire information brought by the sentence.

    if model == "gpt-5.2":
        response = client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt_criteria
                },
                {
                    "role": "user",
                    "content": user_prompt_criteria.format(question_text=question_text)
                }
            ],
            text_format=DistinctionCriteria,
        )

        distinction_criterias = response.output_parsed
        print("see: ", dir(distinction_criterias))
        print("See: ", distinction_criterias)
        save_json_file(distinction_criterias, question, question_text, criteria_extraction_dir, model, iteration)

    elif model == "claude-sonnet-4-5":
        print("Ci piace ancora di più")
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt_criteria,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt_criteria.format(question_text=question_text)
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": json_schema_extraction_criteria
                }
            }
        )

        distinction_criterias = response.content[0].text
        save_json_file(distinction_criterias, question, question_text, criteria_extraction_dir, model, iteration)

    elif model == "deepseek-r1:8b":
        print("Eccoci qua")
        """
        configs = {
            "max_tokens": 200,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "personal_answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "country": {"type": "string"},
                            "capital": {"type": "string"}
                        },
                        "required": ["country", "capital"],
                        "additionalProperties": False
                    }
                }
            }
        }
        """

        configs = {
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extracted_criterias",
                    "schema": json_schema_extraction_criteria
                }
            }
        }

        print("Vediamo questo bel prompt")
        print(user_prompt_criteria.format(question_text=question_text))

        prompt_parameters = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt_criteria},
                {"role": "user", "content": user_prompt_criteria.format(question_text=question_text)}
            ],
        }

        prompt_parameters.update(configs)

        response = client.chat.completions.create(**prompt_parameters)
        distinction_criterias = response.choices[0].message.content

        print(f"Response: {type(distinction_criterias), distinction_criterias}\n\n")
        print(f"Usage stats: {response.usage}")

        save_json_file(distinction_criterias, question, question_text, criteria_extraction_dir, model, iteration)


def extract_criterias_main(model, client):

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")

    selected_questions = random.sample(os.listdir(QUESTIONS_MULTIMODALQA_TRAINING), 1000)

    # The loop is to repeat the criteria extraction multiple times
    for iteration in range(0, 1):
        for i, question in enumerate(sorted(os.listdir(QUESTIONS_MULTIMODALQA_TRAINING))[:1]):
            print("Question: ", i, question)
            json_question = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))
            question_text = json_question["question"]
            answer_text = json_question["answers"][0]["answer"]
            print("Question text: ", question_text)
            print("Answer text: ", answer_text)

            extract_criterias(model, client, question_text, question, CRITERIA_EXTRACTION_DIR, iteration)
