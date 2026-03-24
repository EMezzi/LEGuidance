import os
import json
import time
import base64
import pandas as pd
import pickle as pk

"""Utils functions"""


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
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
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


def get_questions():
    dataset = pk.load(open("./modality_selection/tree_dataset.pkl", "rb"))

    questions_dataset = dataset_build(dataset)
    questions_list = questions_dataset.to_dict(orient="records")
    questions_list = [question for question in questions_list if question["index"] in ['question_14103.json']]

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
        open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/all_data.json", "rb"))

    table_set = create_table_set(question_data, all_data_json_object["table"])
    print("Table set: ", table_set)

    image_set = create_image_set(question_data, all_data_json_object["image"])
    print("Image set: ", image_set)

    text_set = create_text_set(question_data, all_data_json_object["text"])
    print("Text set: ", text_set)

    print("--- %s seconds ---" % (time.time() - start_time))

    d = {"image_set": image_set, "text_set": text_set, "table_set": table_set}

    # Association between question and json files of the images, texts, and tables
    with open(os.path.join(association_dir, question), "w") as json_file:
        json.dump(d, json_file, indent=4)


def create_association_qa(questions_dir, association_dir):
    for question in sorted(os.listdir(questions_dir)):
        if question not in os.listdir(association_dir):
            print("Question: ", question)
            question_data = get_question_data(questions_dir, question)
            print("Question data: ", question_data)
            create_connection(question, question_data, association_dir)


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
