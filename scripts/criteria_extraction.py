from __future__ import annotations
from miscellaneous.prompt import system_prompt_criteria, user_prompt_criteria
from miscellaneous.json_schemas import json_schema_extraction_criteria
import pickle as pk
from collections import Counter

import json
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
import random
from dotenv import load_dotenv

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


def substitute(questions_multimodal_qa, criteria_extraction_dir, model, selected_files):
    present_files = os.listdir(f"{criteria_extraction_dir}/{model}/iteration_0")
    print("Present files: ", len(sorted(present_files)))
    print("Selected files: ", len(sorted(selected_files)))

    used_indices = set()

    missing = []

    for j, present_file in enumerate(present_files):
        print("File: ", j)
        if present_file not in selected_files:

            present_json = json.load(open(os.path.join(questions_multimodal_qa, present_file), "rb"))
            modalities_present = tuple(sorted(present_json["metadata"]["modalities"]))

            substituted = False

            for i, chosen_file in enumerate(selected_files):

                if i in used_indices:
                    continue

                chosen_json = json.load(open(os.path.join(questions_multimodal_qa, chosen_file), "rb"))
                modalities_chosen = tuple(sorted(chosen_json["metadata"]["modalities"]))

                if modalities_present == modalities_chosen:
                    selected_files[i] = present_file
                    used_indices.add(i)
                    substituted = True
                    break

            if not substituted:
                missing.append(present_file)

    dups = [f for f, c in Counter(selected_files).items() if c > 1]
    print("Duplicates in selected_files:", len(dups))

    print("present unique:", len(set(present_files)))
    print("selected unique:", len(set(selected_files)))

    missing_final = set(present_files) - set(selected_files)
    extra_final = set(selected_files) - set(present_files)

    print("missing_final:", len(missing_final))
    print("extra_final:", len(extra_final))

    print("missing sample:", list(missing_final)[:10])
    print("extra sample:", list(extra_final)[:10])

    intersection = set(present_files).intersection(set(selected_files))
    print("intersection: ", intersection)
    print("len intersection: ", len(intersection))

    print("Still missing:", missing)
    print("Subset check:", set(present_files).issubset(set(selected_files)))


def check_modality_frequency(question_multimodal_dir, selected_questions):
    d = {}

    for question in selected_questions:
        json_question = json.load(open(os.path.join(question_multimodal_dir, question), "rb"))
        modality = tuple(sorted(json_question["metadata"]["modalities"]))

        if modality not in d:
            d[modality] = 1
        else:
            d[modality] += 1

    print("Number of questions for each modality between the chosen ones")
    print(d)


def select_questions(model, client):
    load_dotenv()

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")

    # Select questions to extract criterias from
    questions_by_modality = json.load(open("../dataset/question_by_modality.json", "rb"))

    selected_questions = []

    n = 909

    # First we select all the questions from table - table
    temporary = random.sample(list(questions_by_modality["multimodal"]["('table', 'table')"].keys()), n)
    selected_questions.extend(temporary)

    # Here we select the questions from
    temporary = random.sample(list(questions_by_modality["multimodal"]["('table', 'text')"].keys()), n)
    selected_questions.extend(temporary)

    temporary = random.sample(list(questions_by_modality["multimodal"]["('image', 'table')"].keys()), n)
    selected_questions.extend(temporary)

    temporary = random.sample(list(questions_by_modality["multimodal"]["('image', 'text')"].keys()), n)
    selected_questions.extend(temporary)

    # Now we randomly select question with unimodal
    temporary = random.sample(list(questions_by_modality["unimodal"]["table"].keys()), n)
    selected_questions.extend(temporary)

    temporary = random.sample(list(questions_by_modality["unimodal"]["text"].keys()), n)
    selected_questions.extend(temporary)

    temporary = random.sample(list(questions_by_modality["unimodal"]["image"].keys()), n)
    selected_questions.extend(temporary)

    check_modality_frequency(QUESTIONS_MULTIMODALQA_TRAINING, selected_questions)
    substitute(QUESTIONS_MULTIMODALQA_TRAINING, CRITERIA_EXTRACTION_DIR, model, selected_questions)
    check_modality_frequency(QUESTIONS_MULTIMODALQA_TRAINING, selected_questions)

    with open("selected_questions.pk", "wb") as file:
        pk.dump(selected_questions, file)


def extract_criterias_main(model, client):
    load_dotenv()

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")

    with open("selected_questions.pk", "rb") as file:
        selected_questions = pk.load(file)

    # The loop is to repeat the criteria extraction multiple times
    for iteration in range(0, 1):
        for i, question in enumerate(sorted(selected_questions)):
            print("Question: ", i, question)
            if question in os.listdir(f"{CRITERIA_EXTRACTION_DIR}/{model}/iteration_0"):
                print("Already present in here")
                json_object = json.load(
                    open(os.path.join(f"{CRITERIA_EXTRACTION_DIR}/{model}/iteration_0", question), "rb"))
                if "answer_class" in json_object or len(
                        list(json_object["expected_answer_type"].keys())) != 2 or "topic" not in json_object:
                    print("Present but with old features")
                    json_question = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))
                    question_text = json_question["question"]
                    extract_criterias(model, client, question_text, question, CRITERIA_EXTRACTION_DIR, iteration)
            else:
                print("Not present between criterias. So let's recompute it.")
                json_question = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))
                question_text = json_question["question"]
                extract_criterias(model, client, question_text, question, CRITERIA_EXTRACTION_DIR, iteration)


if __name__ == "__main__":
    random.seed(42)
    select_questions("gpt-5.2", None)
