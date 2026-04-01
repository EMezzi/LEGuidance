from __future__ import annotations

import os
import json
import random
import pickle as pk
from collections import Counter

from prompts.le.prompt_criteria_extraction import system_prompt_criteria, user_prompt_criteria
from schemas.pydantic_schemas import *
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from functions_extract_criterias import *


class CriteriasAgent:
    def __init__(self, openai_client, bedrock_client, path_criterias, modalities):
        self.openai_client = openai_client
        self.bedrock_client = bedrock_client
        self.path_criterias = path_criterias
        self.modalities = modalities

    def extract_criterias(self, model, question_text, question, criteria_extraction_dir, iteration):
        # Add tho the prompt the generation of a set that contains the entire information brought by the sentence.

        if model == "gpt-5.2":
            extract_criterias_gpt(model, self.openai_client, question_text, question, criteria_extraction_dir,
                                  iteration)
        elif (model == "global.amazon.nova-2-lite-v1:0" or model == "mistral.mistral-large-3-675b-instruct" or
              model == "moonshotai.kimi-k2.5" or model == "nvidia.nemotron-nano-12b-v2" or
              model == "qwen.qwen3-vl-235b-a22b" or model == "us.anthropic.claude-sonnet-4-6"):
            extract_criterias_amazon(model, self.bedrock_client, question_text, question, criteria_extraction_dir,
                                     iteration)


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


def extract_criterias_main(criterias_agent, model, client):
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
                    criterias_agent.extract_criterias(model, client, question_text, question, CRITERIA_EXTRACTION_DIR,
                                                      iteration)
            else:
                print("Not present between criterias. So let's recompute it.")
                json_question = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))
                question_text = json_question["question"]
                criterias_agent.extract_criterias(model, client, question_text, question, CRITERIA_EXTRACTION_DIR,
                                                  iteration)


if __name__ == "__main__":
    # random.seed(42)
    # select_questions("gpt-5.2", None)

    print(criteria_tool)
