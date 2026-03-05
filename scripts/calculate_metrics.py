import os
import ast
import json
import pandas as pd
import pickle as pk
from dotenv import load_dotenv
import matplotlib.pyplot as plt


def plot_results(responses):
    modalities = list(responses.keys())
    accuracy = [
        responses[m]['correct'] / (responses[m]['correct'] + responses[m]['incorrect']) * 100
        for m in modalities
    ]

    # Plot
    plt.figure()
    plt.bar(modalities, accuracy)

    plt.xlabel("Modality")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Modality")

    plt.savefig('../figures/results_unimodal.png')
    plt.show()

def dataset_build(dataset, type):
    if type == "unimodal":
        image_rows = dataset[dataset["modality"].astype(str) == "['image']"].sample(10, random_state=42)
        text_rows = dataset[dataset["modality"].astype(str) == "['text']"].sample(10, random_state=42)
        table_rows = dataset[dataset["modality"].astype(str) == "['table']"].sample(10, random_state=42)

        test_dataset = pd.concat([image_rows, text_rows, table_rows]) \
            .sample(frac=1, random_state=42) \
            .reset_index(drop=False)

        return test_dataset

    elif type == "multimodal":
        image_text_rows = dataset[dataset["modality"].astype(str) == "['image', 'text']"].sample(7, random_state=42)
        text_image_rows = dataset[dataset["modality"].astype(str) == "['text', 'image']"].sample(7, random_state=42)
        image_table_rows = dataset[dataset["modality"].astype(str) == "['image', 'table']"].sample(7, random_state=42)
        table_image_rows = dataset[dataset["modality"].astype(str) == "['table', 'image']"].sample(7, random_state=42)

        text_table_rows = dataset[dataset["modality"].astype(str) == "['text', 'table']"].sample(7, random_state=42)
        table_text_rows = dataset[dataset["modality"].astype(str) == "['table', 'text']"].sample(7, random_state=42)

        table_table_rows = dataset[dataset["modality"].astype(str) == "['table', 'table']"].sample(7, random_state=42)

        test_dataset = pd.concat(
            [image_text_rows, text_image_rows, image_table_rows, table_image_rows,
             text_table_rows, table_text_rows, table_table_rows]) \
            .sample(frac=1, random_state=42) \
            .reset_index(drop=False)

        return test_dataset


if __name__ == "__main__":
    load_dotenv()

    models = ["gpt-5.2"]  # "claude-sonnet-4-5"]

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    ANSWERS_DIR_TRAINING = os.getenv("ANSWERS_DIR_TRAINING")

    dataset = pk.load(open("./modality_selection/tree_dataset.pkl", "rb"))

    unimodal_multimodal = "unimodal"

    questions_dataset = dataset_build(dataset, unimodal_multimodal)
    print(questions_dataset)
    questions_list = questions_dataset["index"].to_list()

    responses = {'image': {"correct": 0, "incorrect": 0}, 'text': {"correct": 0, "incorrect": 0},
                 'table': {"correct": 0, "incorrect": 0}}

    """
    responses = {'image_text': {"correct": 0, "incorrect": 0}, "text_image": {"correct": 0, "incorrect": 0},
                 'image_table': {"correct": 0, "incorrect": 0}, "table_image": {"correct": 0, "incorrect": 0},
                 'text_table': {"correct": 0, "incorrect": 0}, 'table_text': {"correct": 0, "incorrect": 0},
                 'table_table': {"correct": 0, "incorrect": 0}}
    """

    for model in models:
        print(f"Model: {model}")
        for i, q in questions_dataset.iterrows():
            question = q["index"]
            modality = q["modality"][0]
            question_text = q["original_question"]
            print("Question: ", question, "Modality: ", modality)
            if question in os.listdir(ANSWERS_DIR_TRAINING + f"/{model}/{unimodal_multimodal}"):

                gt = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))

                if len(gt["answers"]) > 1:
                    answers_gt = set([el["answer"].lower() for el in gt["answers"]])
                else:
                    answers_gt = set(gt["answers"][0]["answer"].lower().split(', '))

                model_response = json.load(open(os.path.join(ANSWERS_DIR_TRAINING + f"/{model}/{unimodal_multimodal}", question), "rb"))["final_answer"]

                if isinstance(model_response, str):
                    model_response = set(model_response.lower().split(', '))
                    model_response = {response.replace("and ", "") for response in model_response}
                elif isinstance(model_response, list):
                    model_response = set(model_response)
                    model_response = {response.lower().replace("and ", "") for response in model_response}

                if answers_gt == model_response:
                    responses[modality]["correct"] += 1
                elif len(answers_gt) == 1 and len(model_response) == 1:
                    el_gt, el_model = answers_gt.pop(), model_response.pop()
                    if el_gt in el_model or el_model in el_gt:
                        responses[modality]["correct"] += 1
                    else:
                        print(f"Question text: ", question_text)
                        print(f"Ground truth: {el_gt}. Model response: {el_model}. Incorrect.")
                        responses[modality]["incorrect"] += 1
                else:
                    print(f"Question text: ", question_text)
                    print(f"Ground truth: {answers_gt}. Model response: {model_response}. Incorrect.")
                    responses[modality]["incorrect"] += 1

            else:
                print(f"Question text: ", question_text)
                print("Not yet answered")
                responses[modality]["incorrect"] += 1

    plot_results(responses)

