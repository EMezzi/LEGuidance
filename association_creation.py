import os
import csv
import json
import argparse
import logging
from io import StringIO
from dotenv import load_dotenv

from utils.utilities import (
    create_association_qa,
    import_directories,
    get_questions
)


# ----------------------------
# Logging
# ----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["multimodalqa", "manymodalqa", "all"])

    parser.add_argument("--setting", type=str, default="validation")

    return parser.parse_args()


# ----------------------------
# CSV → Table
# ----------------------------
def csv_to_table_dict(raw_string):
    decoded = raw_string.encode().decode("unicode_escape")
    rows = list(csv.reader(StringIO(decoded)))

    if not rows:
        return []

    headers = rows[0]
    data_rows = rows[1:]

    table_rows = []

    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue

        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))

        dict_row = [
            {
                "header": headers[i],
                "text": row[i],
                "links": []
            }
            for i in range(len(headers))
        ]

        table_rows.append(dict_row)

    return table_rows


# ----------------------------
# ManyModalQA: create question files
# ----------------------------
def create_manymodal_dataset(questions_dir):
    source_file = "../manymodalqa/ManyModalQAData/official_aaai_split_dev_data.json"

    logging.info(f"Loading raw dataset from {source_file}")

    questions = json.load(open(source_file, "rb"))

    for q in questions:
        out_path = os.path.join(questions_dir, f"{q['id']}.json")
        json.dump(q, open(out_path, "w"), indent=4)

    logging.info(f"Saved {len(questions)} question files")


# ----------------------------
# ManyModalQA: split modalities
# ----------------------------
def create_single_files_manymodalqa(
    questions,
    questions_dir,
    text_dir,
    image_dir,
    table_dir,
    association_dir
):
    for question in questions:
        question_path = os.path.join(questions_dir, question)
        question_json = json.load(open(question_path, "rb"))

        qid = question_json["id"]

        association_dict = {
            "text_set": None,
            "image_set": None,
            "table_set": None
        }

        # TEXT
        text_el = {
            "title": "",
            "url": "",
            "id": qid,
            "text": question_json.get("text", "")
        }

        json.dump(text_el, open(os.path.join(text_dir, question), "w"), indent=4)
        text_el["json"] = f"{qid}.json"
        association_dict["text_set"] = [text_el]

        # IMAGE
        image_data = question_json.get("image")
        image_el = {
            "title": image_data["caption"] if image_data else "",
            "url": image_data["url"] if image_data else "",
            "id": qid,
            "path": f"{qid}.png" if image_data else ""
        }

        json.dump(image_el, open(os.path.join(image_dir, question), "w"), indent=4)
        image_el["json"] = f"{qid}.json"
        association_dict["image_set"] = [image_el]

        # TABLE
        table_data = question_json.get("table")
        table_rows = csv_to_table_dict(table_data) if table_data else []

        table_el = {
            "title": "",
            "url": "",
            "id": qid,
            "table": {
                "table_rows": table_rows,
                "table_name": ""
            }
        }

        json.dump(table_el, open(os.path.join(table_dir, question), "w"), indent=4)

        association_dict["table_set"] = [{
            qid: table_rows,
            "json": f"{qid}.json"
        }]

        json.dump(
            association_dict,
            open(os.path.join(association_dir, f"{qid}.json"), "w"),
            indent=4
        )

    logging.info(f"Processed {len(questions)} questions")


# ----------------------------
# Load directories
# ----------------------------
def load_data(dataset, setting):
    dirs = import_directories(dataset, setting, "le")

    (
        IMAGE_DIR,
        TEXT_DIR,
        TABLE_DIR,
        FINAL_DATASET_IMAGES,
        ASSOCIATION_DIR,
        QUESTIONS_DIR,
        CRITERIA_DIR,
        ANSWERS_DIR,
    ) = dirs

    questions = get_questions(dataset, QUESTIONS_DIR)

    return {
        "IMAGE_DIR": IMAGE_DIR,
        "TEXT_DIR": TEXT_DIR,
        "TABLE_DIR": TABLE_DIR,
        "ASSOCIATION_DIR": ASSOCIATION_DIR,
        "QUESTIONS_DIR": QUESTIONS_DIR,
        "questions": questions,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    load_dotenv()

    args = parse_args()

    datasets = ["multimodalqa", "manymodalqa"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        logging.info(f"Processing dataset: {dataset}")

        data = load_data(dataset, args.setting)

        if dataset == "multimodalqa":
            create_association_qa(
                data["QUESTIONS_DIR"],
                data["ASSOCIATION_DIR"]
            )

        elif dataset == "manymodalqa":
            create_manymodal_dataset(data["QUESTIONS_DIR"])

            # reload questions after creation
            data["questions"] = get_questions(dataset, data["QUESTIONS_DIR"])

            create_single_files_manymodalqa(
                data["questions"],
                data["QUESTIONS_DIR"],
                data["TEXT_DIR"],
                data["IMAGE_DIR"],
                data["TABLE_DIR"],
                data["ASSOCIATION_DIR"],
            )


if __name__ == "__main__":
    main()