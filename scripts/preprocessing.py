import json
import os
import ast
import seaborn
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def preprocess_questions():
    new_dict = {}
    for modality in MODALITIES:
        if modality == "image":
            print("Start modality image")
            new_dict["image"] = {}
            for json_image in sorted(os.listdir(IMAGE_DIR)):
                object_image = json.load(open(os.path.join(IMAGE_DIR, json_image), "rb"))
                object_image["json"] = json_image
                new_dict["image"][object_image["id"]] = object_image

            with open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/all_data.json", "w") as json_file:
                json.dump(new_dict, json_file, indent=4)

            print("Finish modality image")

        elif modality == "text":
            print("Start modality text")
            new_dict["text"] = {}
            for json_text in sorted(os.listdir(TEXT_DIR)):
                object_text = json.load(open(os.path.join(TEXT_DIR, json_text), "rb"))
                object_text["json"] = json_text
                new_dict["text"][object_text["id"]] = object_text

            with open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/all_data.json", "w") as json_file:
                json.dump(new_dict, json_file, indent=4)
            print("Finish modality text")

        elif modality == "tables":
            print("Start modality tables")
            new_dict["table"] = {}
            for json_table in sorted(os.listdir(TABLE_DIR)):
                object_table = json.load(open(os.path.join(TABLE_DIR, json_table), "rb"))
                object_table["json"] = json_table
                new_dict["table"][object_table["id"]] = object_table

            with open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/all_data.json", "w") as json_file:
                json.dump(new_dict, json_file, indent=4)

            print("Finish modality tables")


def separate_unimodal_multimodal():
    """In this function we separate the unimodal and the multimodal question"""
    modality_question = {"unimodal": {}, "multimodal": {}}

    for question in sorted(os.listdir(QUESTIONS_MULTIMODALQA_TRAINING)):
        print("Question: ", question)
        json_question = json.load(open(os.path.join(QUESTIONS_MULTIMODALQA_TRAINING, question), "rb"))

        modalities = json_question["metadata"]["modalities"].copy()

        if len(modalities) == 1:
            if modalities[0] not in modality_question["unimodal"]:
                modality_question["unimodal"][modalities[0]] = {}

            modality_question["unimodal"][modalities[0]][question] = {
                "question": json_question["question"],
            }

        elif len(modalities) > 1:
            modalities = str(tuple(sorted(modalities)))
            if modalities not in modality_question["multimodal"]:
                modality_question["multimodal"][modalities] = {}

            modality_question["multimodal"][modalities][question] = {
                "question": json_question["question"],
                "intermediate_answers": json_question["metadata"]["intermediate_answers"]
            }

    print(modality_question)

    with open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/question_by_modality.json", "w") as json_file:
        json.dump(modality_question, json_file, indent=4)


def unimodal_multimodal_statistics():
    questions_by_modality = json.load(
        open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/question_by_modality.json", "rb"))

    # Unimodal vs multimodal
    total_unimodal = sum(len(v) for v in questions_by_modality["unimodal"].values())
    total_multimodal = sum(len(v) for v in questions_by_modality["multimodal"].values())

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    axes[0].bar(['Unimodal', 'Multimodal'], [total_unimodal, total_multimodal])
    axes[0].set_title('Total Questions')
    axes[0].set_ylabel('Count')

    # Distribution of unimodal categories
    unimodal_labels = list(questions_by_modality["unimodal"].keys())
    unimodal_counts = [len(v) for v in questions_by_modality["unimodal"].values()]

    # Plot 3: unimodal distribution
    axes[1].bar(unimodal_labels, unimodal_counts)
    axes[1].set_title("Unimodal Distribution")

    # Distribution of multimodal categories
    multimodal_labels = list(questions_by_modality["multimodal"].keys())
    multimodal_counts = [len(v) for v in questions_by_modality["multimodal"].values()]

    print(multimodal_counts)

    axes[2].bar(multimodal_labels, multimodal_counts)
    axes[2].set_title("Multimodal Distribution")
    axes[2].tick_params(axis="x", rotation=30)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.xticks(rotation=30, ha="right")

    plt.savefig("dataset_statistics.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    load_dotenv()

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")

    IMAGE_DIR = os.getenv("IMAGE_DIR")
    TEXT_DIR = os.getenv("TEXT_DIR")
    TABLE_DIR = os.getenv("TABLE_DIR")

    # preprocess_questions()

    """Separation of the sentences by modality"""
    # separate_unimodal_multimodal()
    # questions_by_modality = json.load(open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/question_by_modality.json", "rb"))

    unimodal_multimodal_statistics()
