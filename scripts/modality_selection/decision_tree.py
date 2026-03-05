from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

import os
import json
from dotenv import load_dotenv
import pandas as pd
import pickle as pk

import random

from collections import Counter


def build_dataset(selected_questions):
    questions = {}
    for i, question in enumerate(selected_questions):
        criteria_json = json.load(open(os.path.join("/Users/emanuelemezzi/PycharmProjects/DatasetAnalysis/iteration_0",
                                                    question), "rb"))

        question_json = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))

        questions[question] = {}
        data = {}

        data["original_question"] = criteria_json["original_question"]
        data["question_length"] = len(criteria_json["original_question"])
        data["question_topic"] = criteria_json["topic"]["question_topic"]
        data["question_answer_class_specific"] = criteria_json["expected_answer_type"]["expected_answer_type_specific"]
        data["question_answer_class_general"] = criteria_json["expected_answer_type"]["expected_answer_type_general"]
        data["modality"] = question_json["metadata"]["modalities"]

        questions[question] = data

    dataset = pd.DataFrame(questions).T

    with open("tree_dataset.pkl", "wb") as file:
        pk.dump(dataset, file)


if __name__ == "__main__":
    load_dotenv()
    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")

    random.seed(42)
    selected_questions = pk.load(open("/Users/emanuelemezzi/PycharmProjects/LEGuidance/scripts/selected_questions.pk", "rb"))

    build_dataset(selected_questions)

