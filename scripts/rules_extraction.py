import os
import ast
import time
import json
import random
import base64
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class Rule(BaseModel):
    rule_id: str = Field(
        ...,
        description="Unique identifier for the rule"
    )

    condition: str = Field(
        ...,
        description="IF-part of the rule, expressed as a human-readable logical condition over linguistic or semantic features"
    )

    predicted_modalities: List[str] = Field(
        ...,
        description="List of modalities required when the rule fires (e.g. ['text'], ['image'], ['table', 'text'])"
    )

    linguistic_triggers: List[str] = Field(
        ...,
        description="Syntactic or semantic features that activate the rule (e.g. 'visual attribute query', 'table filtering')"
    )

    rationale: str = Field(
        ...,
        description="Explanation of why text alone is insufficient to answer the question"
    )

    example_questions: Optional[List[str]] = Field(
        default=None,
        description="Optional example questions that are covered by this rule"
    )


class RuleSet(BaseModel):
    rules: List[Rule] = Field(
        ...,
        description="Ordered list of high-precision modality selection rules"
    )

    fallback_rule: Rule = Field(
        ...,
        description="Default rule applied when no other rule matches"
    )


def dataset_creation(path_questions):
    """This method is used to create the batches to then train the model with self-refinement for rule generation"""

    json_questions = json.load(open(path_questions, "rb"))

    dataset = []

    for key in json_questions.keys():
        for modality in json_questions[key].keys():
            for question in json_questions[key][modality].keys():
                text_question = json_questions[key][modality][question]["question"]
                dataset.append({text_question: modality})

    random.shuffle(dataset)

    return dataset


class RuleExtractor:
    def __init__(self, key, dataset):
        self.key = key
        self.client = OpenAI(api_key=self.key)
        self.dataset = dataset

    def batches_generation(self):

        dataset = self.dataset.copy()
        random.shuffle(dataset)
        unused = dataset

        batches = []
        k = 25

        while len(unused) >= k:
            selection = random.sample(unused, k)
            for item in selection:
                unused.remove(item)

            batches.append(selection)

        return batches

    def rule_initialization(self, first_batch):

        dataset_string = ""

        for i, question in enumerate(first_batch):
            key, value = next(iter(question.items()))
            dataset_string += f"{i}.\nquestion: {key}\ngold_modality: {value}\n\n"

        print(dataset_string)

        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are designing an interpretable rule-based system to predict which modality (or modalities) are required to answer a question.
                    
                    You are given questions and their gold modality labels.
                    A modality is required only if the information needed to answer the question cannot be reliably obtained from text alone.
                    
                    Your task is to induce general, human-interpretable rules that predict the required modality based on: 
                        - syntactic structure (e.g., question type, main verbs, dependency relations)
                        - semantic content (e.g., perceptual properties, events, evidence type)
                    
                    Instructions
                        1. Do NOT use dataset-specific keywords or surface cues unless absolutely necessary.
                        2. Rules must generalize across paraphrases. 
                        3. Express each rule in IF–THEN form.
                        4. Each rule should state why the modality is required.
                        5. Prefer high-precision rules over high coverage.
                    
                    For each rule, output: 
                        - Rules: IF … THEN required modality = …
                        - Linguistic trigger: syntactic or semantic feature(s)
                        - Rationale: why text alone is insufficient
                        - Example questions (optional)
                        - Data
                    
                    Task
                    Identify recurring linguistic and semantic patterns that determine modality.
                    Propose a minimal set of rules that explain these patterns.
                    Include a default fallback rule for cases not covered.
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"""Below is an initial batch of labeled questions from a multimodal question-answering dataset.

                                        Each example consists of:
                                        - question: the natural-language question
                                        - gold_modality: the modality or modalities required to answer it correctly
                                        
                                        These examples are provided for RULE INDUCTION.
                                        Do NOT attempt to answer the questions.
                                        
                                        Allowed modalities:
                                        - text
                                        - image
                                        - table
                                        - image+table
                                        - table+text
                                        - table+table
                                        
                                        Dataset:
                                        {dataset_string}
                                    """
                        },
                    ]
                }
            ],
            text_format=RuleSet,
        )

        initialization_rules = response.output_parsed
        return initialization_rules

    def extract_rules(self):
        batches = self.batches_generation()

        initialization_rules = self.rule_initialization(batches[0])

        if hasattr(initialization_rules, "model_dump"):
            initialization_rules = initialization_rules.model_dump()

            with open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/results/extracted_rules/initialization_rules.json", "w") as json_file:
                json.dump(initialization_rules, json_file, indent=4)


if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    IMAGE_DIR = os.getenv("IMAGE_DIR")
    TEXT_DIR = os.getenv("TEXT_DIR")
    TABLE_DIR = os.getenv("TABLE_DIR")

    ASSOCIATION_DIR = os.getenv("ASSOCIATION_DIR")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")
    ANSWERS_DIR_TRAINING = os.getenv("ANSWERS_DIR_TRAINING")

    dataset = dataset_creation("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/question_by_modality.json")
    a = RuleExtractor(OPENAI_KEY, dataset)

    a.extract_rules()
