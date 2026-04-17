import os
import json
import time
import base64
import random
import imghdr
import pandas as pd
from dotenv import load_dotenv

"""Function to build the dataset"""

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

def import_directories(dataset, setting, approach):
    if dataset == "multimodalqa":
        IMAGE_DIR = os.getenv("IMAGE_DIR")
        TEXT_DIR = os.getenv("TEXT_DIR")
        TABLE_DIR = os.getenv("TABLE_DIR")
        FINAL_DATASET_IMAGES = os.getenv("FINAL_DATASET_IMAGES")

        ASSOCIATION_DIR = os.getenv(f"ASSOCIATION_MULTIMODALQA_{setting.upper()}")
        QUESTIONS_DIR = os.getenv(f"QUESTIONS_MULTIMODALQA_{setting.upper()}")
        CRITERIA_DIR = os.getenv(f"CRITERIA_MULTIMODALQA_{setting.upper()}")

        ANSWERS_DIR = "dp"
        if approach == "dp":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_VALIDATION_DP")
        elif approach == "cot":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_VALIDATION_COT")
        elif approach == "pp":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_VALIDATION_PP")
        elif approach == "le":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_{setting.upper()}")

        return IMAGE_DIR, TEXT_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ASSOCIATION_DIR, QUESTIONS_DIR, CRITERIA_DIR, ANSWERS_DIR

    elif dataset == "manymodalqa":
        QUESTIONS_DIR = os.getenv(f"QUESTIONS_MANYMODALQA_{setting.upper()}")
        IMAGE_DIR = os.getenv(f"QUESTIONS_MANYMODALQA_IMAGES")

        CRITERIA_DIR = os.getenv(f"CRITERIA_MANYMODALQA_{setting.upper()}")
        ANSWERS_DIR = os.getenv(f"ANSWERS_MANYMODALQA_{setting.upper()}")

        return QUESTIONS_DIR, IMAGE_DIR, CRITERIA_DIR, ANSWERS_DIR


"""Utils functions"""


def get_image_format(image_path, image_bytes):
    detected_type = imghdr.what(None, h=image_bytes)

    if detected_type:
        # Bedrock expects 'jpeg', but imghdr might return 'jpg'
        ext = detected_type.lower().replace('jpg', 'jpeg')
    else:
        # Fallback to filename extension ONLY if binary detection fails
        ext = image_path.split('.')[-1].lower().replace('jpg', 'jpeg')

    # 3. FINAL SAFETY CHECK
    if ext not in ['png', 'jpeg', 'gif', 'webp']:
        ext = 'png'  # Safe default

    return ext


def get_question_data(question_dir, question):
    question_json = json.load(open(os.path.join(question_dir, question), "rb"))
    question_text = question_json["question"]

    images_doc_ids = question_json["metadata"]["image_doc_ids"]
    text_doc_ids = question_json["metadata"]["text_doc_ids"]
    table_id = question_json["metadata"]["table_id"]

    return {"question_text": question_text, "image_doc_ids": images_doc_ids, "text_doc_ids": text_doc_ids,
            "table_id": table_id}


def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(make_hashable(x) for x in obj)
    else:
        return obj


def detect_media_type_from_bytes(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("Unknown image format (not png/jpg/gif/webp)")


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_question_files(association_dir, question):
    question_json = json.load(open(os.path.join(association_dir, question), "rb"))

    image_set = question_json["image_set"]
    text_set = question_json["text_set"]
    table_set = question_json["table_set"]

    return {"image_set": image_set, "text_set": text_set, "table_set": table_set}


def get_questions(questions_dir):
    """
    dataset = pk.load(open("./modality_selection/tree_dataset.pkl", "rb"))

    questions_dataset = dataset_build(dataset)
    questions_list = questions_dataset.to_dict(orient="records")
    questions_list = [question for question in questions_list]

    return questions_list
    """

    random.seed(42)

    questions = os.listdir(questions_dir)
    questions_list = random.sample(questions, len(questions))

    return questions_list


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        if isinstance(row, list):
            flat_list.extend(row)

    return flat_list


def average_modality_le(partitions):
    d = {}
    for modality, items in partitions.items():
        le_values = [item['le'] for item in items if "le" in item and item['le'] > 0]
        if le_values:
            d[modality] = sum(le_values) / len(le_values)
        else:

            d[modality] = None

    return d


"""Functions to create set of images, paragraphs, and table"""


def create_image_set(question_data, all_data_images):
    print("Create image set")
    images = question_data["image_doc_ids"]
    print("Images are: ", images)
    answer_set_images = []

    for image in images:
        answer_set_images.append(all_data_images[image])

    return answer_set_images


def create_text_set(question_data, all_data_text):
    print("Create text set")
    texts = question_data["text_doc_ids"]
    answer_set_text = [all_data_text[text] for text in texts]

    return answer_set_text


def create_table_set(question_data, all_data_table):
    print("Create table set")
    if isinstance(question_data["table_id"], str):
        tables = [question_data["table_id"]]
    elif isinstance(question_data["table_id"], list):
        tables = question_data["table_id"]
    else:
        tables = None

    print("Tables: ", tables)

    # Since we do not want to mix rows from different tables
    answer_set_tables = []

    for table in tables:
        table_json = all_data_table[table]

        d = {table: None}
        answer_set_rows = []
        column_names = [el["column_name"] for el in table_json["table"]["header"]]
        # Here we extract the rows of the table
        for row in table_json["table"]["table_rows"]:
            new_row = [{**cell, "header": header} for cell, header in zip(row, column_names)]
            answer_set_rows.append(new_row)

        d[table] = answer_set_rows
        d["json"] = table_json["json"]
        answer_set_tables.append(d)

    return answer_set_tables


def create_connection(question, question_data, association_dir):
    start_time = time.time()

    all_data_json_object = json.load(
        open("../dataset/all_data.json", "rb"))

    text_set = create_text_set(question_data, all_data_json_object["text"])
    print("Text set: ", text_set)

    image_set = create_image_set(question_data, all_data_json_object["image"])
    print("Image set: ", image_set)

    table_set = create_table_set(question_data, all_data_json_object["table"])
    print("Table set: ", table_set)

    print("--- %s seconds ---" % (time.time() - start_time))

    d = {"image_set": image_set, "text_set": text_set, "table_set": table_set}

    # Association between question and json files of the images, texts, and tables
    with open(os.path.join(association_dir, question), "w") as json_file:
        json.dump(d, json_file, indent=4)


def create_association_qa(questions_dir, association_dir):
    print(len(os.listdir(questions_dir)))
    for i, question in enumerate(sorted(os.listdir(questions_dir))):
        if question not in os.listdir(association_dir):
            print(f"Question: {i}, {question}")
            question_data = get_question_data(questions_dir, question)
            print("Question data: ", question_data)
            create_connection(question, question_data, association_dir)


def save_json_file(json_object, file_name, question_text, criteria_extraction_dir, model):
    os.makedirs(f"{criteria_extraction_dir}/{model}/", exist_ok=True)
    path = os.path.join(f"{criteria_extraction_dir}/{model}/", file_name)

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





if __name__ == '__main__':
    load_dotenv()

    QUESTIONS_MULTIMODALQA_VALIDATION = os.getenv("QUESTIONS_MULTIMODALQA_VALIDATION")
    ASSOCIATION_DIR_VALIDATION = os.getenv("ASSOCIATION_MULTIMODALQA_VALIDATION")

    create_association_qa(QUESTIONS_MULTIMODALQA_VALIDATION, ASSOCIATION_DIR_VALIDATION)
