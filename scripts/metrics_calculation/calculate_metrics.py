import os
import ast
import json
import numpy as np
import pandas as pd
import pickle as pk
from dotenv import load_dotenv
import matplotlib.pyplot as plt

iteration = "iteration_1"


def plot_results(responses, iteration):

    modalities = list(responses.keys())
    exact_matching = [responses[m] for m in modalities]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.bar(modalities, exact_matching, color='skyblue')

    plt.xlabel("Modality")
    plt.ylabel("Exact matching (%)")
    plt.title("Exact matching per modality")

    # Set proper y-ticks (0 to 1 with step 0.1)
    plt.yticks(np.arange(0, 1.1, 0.1))  # include 1.0

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'../figures/results_{iteration}.png')
    plt.show()


def dataset_build(dataset):
    image_rows = dataset[dataset["modality"].astype(str) == "['image']"].sample(20, random_state=42)
    text_rows = dataset[dataset["modality"].astype(str) == "['text']"].sample(20, random_state=42)
    table_rows = dataset[dataset["modality"].astype(str) == "['table']"].sample(20, random_state=42)

    image_text_rows = dataset[dataset["modality"].astype(str) == "['image', 'text']"].sample(20, random_state=42)
    text_image_rows = dataset[dataset["modality"].astype(str) == "['text', 'image']"].sample(20, random_state=42)
    image_table_rows = dataset[dataset["modality"].astype(str) == "['image', 'table']"].sample(20, random_state=42)
    table_image_rows = dataset[dataset["modality"].astype(str) == "['table', 'image']"].sample(20, random_state=42)

    text_table_rows = dataset[dataset["modality"].astype(str) == "['text', 'table']"].sample(20, random_state=42)
    table_text_rows = dataset[dataset["modality"].astype(str) == "['table', 'text']"].sample(20, random_state=42)

    table_table_rows = dataset[dataset["modality"].astype(str) == "['table', 'table']"].sample(20, random_state=42)

    test_dataset = pd.concat(
        [image_rows, text_rows, table_rows, image_text_rows, text_image_rows, image_table_rows, table_image_rows,
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

    questions_dataset = dataset_build(dataset)
    questions_list = questions_dataset["index"].to_list()

    responses = {'image': {"correct": 0, "incorrect": 0}, 'text': {"correct": 0, "incorrect": 0},
                 'table': {"correct": 0, "incorrect": 0},
                 'image_text': {"correct": 0, "incorrect": 0}, 'text_image': {"correct": 0, "incorrect": 0},
                 'image_table': {"correct": 0, "incorrect": 0}, 'table_image': {"correct": 0, "incorrect": 0},
                 'text_table': {"correct": 0, "incorrect": 0}, 'table_text': {"correct": 0, "incorrect": 0},
                 'table_table': {"correct": 0, "incorrect": 0}}

    wrong = {'image': [],
             'table': [],
             'text': [],
             'image_text': [],
             'text_image': [],
             'image_table': [],
             'table_image': [],
             'text_table': [],
             'table_text': [],
             'table_table': []}

    for model in models:
        print(f"Model: {model}")
        for i, q in questions_dataset.head(100).iterrows():
            question = q["index"]
            modality = '_'.join(q["modality"])
            question_text = q["original_question"]
            print("Question: ", question, "Modality: ", modality)

            if '_' in modality:
                unimodal_multimodal = "multimodal"
            else:
                unimodal_multimodal = "unimodal"

            if question in os.listdir(f"{ANSWERS_DIR_TRAINING}/{iteration}/{model}/{unimodal_multimodal}"):

                gt = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))

                if len(gt["answers"]) > 1:
                    answers_gt = set([el["answer"].lower() for el in gt["answers"]])
                else:
                    answers_gt = set(gt["answers"][0]["answer"].lower().split(', '))

                model_response = \
                    json.load(
                        open(
                            os.path.join(f"{ANSWERS_DIR_TRAINING}/{iteration}/{model}/{unimodal_multimodal}", question),
                            "rb"))[
                        "final_answer"]

                # if isinstance(model_response, str):
                #    model_response = set(model_response.lower().split(', '))
                #    model_response = {response.replace("and ", "") for response in model_response}
                if isinstance(model_response, list):
                    model_response = set(model_response)
                    model_response = [m.lower() for m in model_response if m != 'NONE']
                    model_response = {response.lower().replace("and ", "") for response in model_response}

                print(f"Question text: {question_text}")
                print(f"Answers gt: {answers_gt}")
                print(f"Model response: {model_response}")

                if answers_gt == model_response:
                    responses[modality]["correct"] += 1

                if isinstance(model_response, str):
                    el_gt, el_model = answers_gt.pop(), model_response

                    if el_gt in el_model or el_model in el_gt:
                        # print("Correct 2")
                        responses[modality]["correct"] += 1
                    else:
                        print("Not correct 1")
                        wrong[modality].append(question)
                        responses[modality]["incorrect"] += 1

                elif len(answers_gt) == 1 and len(model_response) == 1:
                    el_gt, el_model = answers_gt.pop(), model_response.pop()
                    if el_gt in el_model or el_model in el_gt:
                        # print("Correct 2")
                        responses[modality]["correct"] += 1
                    else:
                        print("Not correct 2")
                        wrong[modality].append(question)
                        responses[modality]["incorrect"] += 1

                elif len(answers_gt) == 1 and len(model_response) > 1:
                    model_response = {m for m in model_response if m != 'None'}
                    el_gt, el_model = answers_gt.pop(), model_response.pop()
                    if el_gt in el_model or el_model in el_gt:
                        # print("Correct 3")
                        responses[modality]["correct"] += 1
                    else:
                        print("Not correct 3")
                        wrong[modality].append(question)
                        responses[modality]["incorrect"] += 1

                elif len(answers_gt) > 1 and len(model_response) > 1:
                    el_gt, el_model = ' '.join(answers_gt), ' '.join(model_response)
                    if el_gt in el_model or el_model in el_gt:
                        # print("Correct 4")
                        responses[modality]["correct"] += 1
                    else:
                        if answers_gt.issubset(model_response) or model_response.issubset(answers_gt):
                            responses[modality]["correct"] += 1
                        else:
                            print("Not correct 4")
                            wrong[modality].append(question)
                            responses[modality]["incorrect"] += 1

                elif len(answers_gt) > 1 and len(model_response) == 1:
                    print("ci siamo o no?")
                    el_gt, el_model = ', '.join(sorted(answers_gt)), model_response.pop()
                    print(el_gt)
                    print(el_model)
                    if el_gt in el_model or el_model in el_gt:
                        responses[modality]["correct"] += 1
                    else:
                        el_model = ', '.join(sorted(el_model.split(', ')))
                        if el_gt == el_model or el_model in el_gt:
                            responses[modality]["correct"] += 1
                        else:
                            print("Not correct 5")
                            wrong[modality].append(question)
                            responses[modality]["incorrect"] += 1

                else:
                    print("Not correct 6")
                    wrong[modality].append(question)
                    responses[modality]["incorrect"] += 1

            else:
                print("Not correct 7")
                print("Not yet answered")
                wrong[modality].append(question)
                responses[modality]["incorrect"] += 1

    responses = {key: round(responses[key]["correct"] / (responses[key]["correct"] + responses[key]["incorrect"]), 2)
                 for key in responses}

    print(f"Responses exact matching: {responses}")

    unimodal_keys = ['image', 'text', 'table', 'table_table']
    multimodal_keys = [k for k in responses if k not in unimodal_keys]

    # Compute averages
    unimodal_avg = sum(responses[k] for k in unimodal_keys) / len(unimodal_keys)
    multimodal_avg = sum(responses[k] for k in multimodal_keys) / len(multimodal_keys)

    print(f"Unimodal average: {unimodal_avg:.2f}")
    print(f"Multimodal average: {multimodal_avg:.2f}")
    print(f"Total average: {np.mean([unimodal_avg, multimodal_avg])}")

    print(responses)
    plot_results(responses, iteration)

    pk.dump(wrong, open(f"wrong_{iteration}.pk", "wb"))
