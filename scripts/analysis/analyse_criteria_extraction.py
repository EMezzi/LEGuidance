from __future__ import annotations
from dotenv import load_dotenv
import json
import os
from sentence_transformers import CrossEncoder
from nltk.translate.bleu_score import sentence_bleu
import torch

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
QUESTIONS_MULTIMODALQA = os.getenv("QUESTIONS_MULTIMODALQA")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", activation_fn=torch.nn.Sigmoid())


def semantic_coverage(sentence1, sentence2):
    return model.predict([(sentence1, sentence2)])


def entity_coverage(sentence1, sentence2):
    return sentence_bleu([sentence1.split()], sentence2.split(), weights=(1, 0, 0, 0))


def coverage(sentence1, sentence2):
    print("The entity coverage is: ", entity_coverage(sentence1, sentence2))
    print("The semantic coverage is: ", semantic_coverage(sentence1, sentence2))


def recreate_question(json_object):
    question = ""

    question += json_object["target"]["text"]
    question += f" {json_object['asked_property']}"

    for constraint in json_object["constraints"]:
        question += f" {constraint['evidence']}"

    return question


if __name__ == '__main__':

    for json_file in os.listdir("../criteria_extraction/"):
        json_object = json.load(open(os.path.join("../criteria_extraction/", json_file), "rb"))

        original_question = json_object["original_question"]
        rewritten_question = json_object["rewritten_question"]
        # Sentences
        print("Original question: ", original_question)
        print("Rewritten question: ", rewritten_question)

        # Progressively calculate the embedding, when adding information
        coverage(original_question.lower(), rewritten_question.lower())

        # Coverage with recreated question
        print("Let's see for the recreated question")
        recreated_question = recreate_question(json_object)
        print("Recreated question: ", recreated_question)
        coverage(original_question.lower(), recreated_question.lower())
        print("\n")