import os
import ast
import time
import json
import random
import base64
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AnswerContainsCriteria(BaseModel):
    answer: str = Field(..., description="Answer yes or not to whether the data contains the criteria")


def get_question_data(question_dir, question):
    question_json = json.load(open(os.path.join(question_dir, question), "rb"))
    question_text = question_json["question"]

    images_doc_ids = question_json["metadata"]["image_doc_ids"]
    text_doc_ids = question_json["metadata"]["text_doc_ids"]
    table_id = question_json["metadata"]["table_id"]

    return {"question_text": question_text, "image_doc_ids": images_doc_ids, "text_doc_ids": text_doc_ids,
            "table_id": table_id}


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_question_files(association_dir, question):
    question_json = json.load(open(os.path.join(association_dir, question), "rb"))

    image_set = question_json["image_set"]
    text_set = question_json["text_set"]
    table_set = question_json["table_set"]

    return {"image_set": image_set, "text_set": text_set, "table_set": table_set}


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        if isinstance(row, list):
            flat_list.extend(row)

    return flat_list


def create_association_qa(questions_dir, image_dir, text_dir, table_dir, association_dir):
    for question in sorted(os.listdir(questions_dir)):
        if question not in os.listdir(association_dir):
            print("Question: ", question)
            question_data = get_question_data(questions_dir, question)
            print("Question data: ", question_data)
            a.create_connection(question, question_data, image_dir, text_dir, table_dir, association_dir)


def answer_qa(questions_dir, association_dir, image_dir, text_dir, table_dir, answers_dir, modalities, mode):
    for question in sorted(os.listdir(questions_dir))[:5]:
        print("Question: ", question)
        # if question not in os.listdir(os.path.join(answers_dir, mode)):
        question_data = get_question_data(questions_dir, question)
        question_files = get_question_files(association_dir, question)
        a.answer_question(question, question_data, question_files, image_dir, text_dir, table_dir,
                          os.path.join(answers_dir, mode), modalities, mode)


class Agent:
    def __init__(self, key, path_criterias, modalities):
        self.key = key
        self.client = OpenAI(api_key=self.key)
        self.path_criterias = path_criterias
        self.modalities = modalities

    def create_connection(self, question, question_data, image_dir, text_dir, table_dir, association_dir):
        start_time = time.time()

        all_data_json_object = json.load(open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/all_data.json", "rb"))

        table_set = self.create_table_set(question_data, all_data_json_object["table"])
        print("Table set: ", table_set)

        image_set = self.create_image_set(question_data, all_data_json_object["image"])
        print("Image set: ", image_set)

        text_set = self.create_text_set(question_data, all_data_json_object["text"])
        print("Text set: ", text_set)

        print("--- %s seconds ---" % (time.time() - start_time))

        d = {"image_set": image_set, "text_set": text_set, "table_set": table_set}

        # Association between question and json files of the images, texts, and tables
        with open(os.path.join(association_dir, question), "w") as json_file:
            json.dump(d, json_file, indent=4)

    def read_criterias(self, question):
        criteria_object = json.load(open(os.path.join(self.path_criterias, question), "rb"))

        criterias = [criteria_object["target"]["text"], criteria_object["asked_property"]]

        for constraint in criteria_object["constraints"]:
            criterias.append(constraint['evidence'])

        return criterias

    def create_image_set(self, question_data, all_data_images):
        print("Create image set")
        images = question_data["image_doc_ids"]
        print("Images are: ", images)
        answer_set_images = []

        for image in images:
            answer_set_images.append(all_data_images[image])

        """
        for file in os.listdir(image_dir):
            image_json = json.load(open(os.path.join(image_dir, file), "rb"))
            # Check if one of the images is in the file opened
            if image_json["id"] in images:
                answer_set_images.append({file: image_json})
        """

        return answer_set_images

    def create_text_set(self, question_data, all_data_text):
        print("Create text set")
        texts = question_data["text_doc_ids"]
        answer_set_text = []

        for text in texts:
            answer_set_text.append(all_data_text[text])

        """
        for file in os.listdir(text_dir):
            text_json = json.load(open(os.path.join(text_dir, file), "rb"))
            if text_json["id"] in texts:
                answer_set_text.append({file: text_json})
        """

        return answer_set_text

    def create_table_set(self, question_data, all_data_table):
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

        """
        for file in os.listdir(table_dir):
            table_json = json.load(open(os.path.join(table_dir, file), "rb"))
            if table_json["id"] in tables:
                # for table in tables:
                # if table == table_json["id"]:
                d = {file: None}
                answer_set_rows = []
                column_names = [el["column_name"] for el in table_json["table"]["header"]]
                # Here we extract the rows of the table
                for row in table_json["table"]["table_rows"]:
                    new_row = [{**cell, "header": header} for cell, header in zip(row, column_names)]
                    answer_set_rows.append(new_row)

                d[file] = answer_set_rows
                answer_set_tables.append(d)

        return answer_set_tables
        """

    def analyse_image(self, criteria, metadata, image64):
        print(type(criteria), criteria)
        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are a visual recognition assistant. Your task is to determine whether the given image contains 
                    a visual element, object, structure, or concept that matches the provided criteria, even if the criteria is not written as text. 
                    Respond only with 'yes' or 'no'."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Does the image with title '{metadata}' contain something that can be described as: '{criteria}'?"
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image64}",
                        },
                    ]
                }
            ],
            text_format=AnswerContainsCriteria,
        )

        found = response.output_parsed
        print(found)
        return found

    def analyse_text(self, criteria, paragraph):
        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You check whether specific pieces of text (criterias) are contained in this paragraphs. Return yes or no"""
                },
                {
                    "role": "user",
                    "content": f"""
                                Return AnswerContainsCriteria from the question below. 
                                
                                This is the criteria to be found {criteria}. 
                                This is the text in which the criteria has to be found {paragraph}. 
                                """
                }
            ],
            text_format=AnswerContainsCriteria,
        )

        found = response.output_parsed
        return found["answer"]

    def analyse_table(self):
        pass

    def fill_criterias(self, question: str, question_data: dict, question_files: dict, image_dir: str, text_dir: str,
                       table_dir: str,
                       modalities: list, mode: str):

        criterias = self.read_criterias(question)
        n_criterias = len(criterias)

        print("The question files are: ", question_files)

        # This dictionary contains the association between the modality and the criterias found for that modality
        d = {key: [] for key in modalities}

        # This dictionary will contain answers for both unimodal and multimodal

        # In this case we try to give the answer by using only one modality. We try this for every modality.
        if mode == "unimodal":
            if os.path.exists(os.path.join(
                    "/Users/emanuelemezzi/PycharmProjects/multimodalqa/results/QA_Answers/json_files/unimodal/",
                    question)):
                json_answer = json.load(open(os.path.join(
                    "/Users/emanuelemezzi/PycharmProjects/multimodalqa/results/QA_Answers/json_files/unimodal/",
                    question)))

                for mod in modalities:
                    if mod not in json_answer:
                        json_answer[mode][mod] = {}
                        for criteria in criterias:
                            json_answer[mode][mod]["criteria"] = criteria
                            print("Criteria: ", criteria)
                            # self.analyse_image(criteria, image)
                            if mod == "image":
                                image_set = question_files['image_set']
                                for image in image_set[:2]:
                                    for key in image.keys():
                                        json_answer[mode][mod]["image"] = key
                                        image_path = os.path.join(
                                            "/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                                            image[key]["path"])
                                        metadata = image[key]["title"]
                                        image_base64 = encode_image(image_path)
                                        answer = self.analyse_image(criteria, metadata, image_base64)
                                        json_answer[mode][mod]["answer"] = answer

            else:
                answer_dict = {}
                answer_dict[mode] = {}
                for mod in modalities:
                    if mod not in answer_dict:
                        answer_dict[mode][mod] = {}
                        for criteria in criterias:
                            answer_dict[mode][mod][criteria] = []
                            print("Criteria: ", criteria)
                            # self.analyse_image(criteria, image)
                            if mod == "image":
                                image_set = question_files['image_set']
                                for image in image_set[:2]:
                                    for key in image.keys():
                                        d = {"image": key}
                                        # answer_dict[mode][mod][criteria]["image"] = key
                                        image_path = os.path.join(
                                            "/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                                            image[key]["path"])
                                        metadata = image[key]["title"]
                                        image_base64 = encode_image(image_path)
                                        answer = self.analyse_image(criteria, metadata, image_base64).answer
                                        print(answer, type(answer))
                                        d["answer"] = answer
                                        answer_dict[mode][mod][criteria].append(d)

                with open(os.path.join(f"{ANSWERS_DIR_TRAINING}/{mode}", question), "w") as json_file:
                    json.dump(answer_dict, json_file, indent=4)

                    """
                    if mod == "text":
                        text_set = question_files['text_set']
                        for text in text_set[:2]:
                            for key in text.keys():
                                self.analyse_text(criteria, text[key]["text"])
                    """

                    """
                    if mod == "tables":
                        pass
                    """


        # In this other case we give the answer by using more than one modality
        elif mode == "multimodal":
            pass
            # Lower bound: only one modality, and upper bound by selecting one modality onlu
            # Upper bound: brute force
            # And with this process we show the association between the question and the type of modality to answer it (single one of fusion)
            # Check weather the additional modality is noise
            while n_criterias > 0:
                for modality in modalities:
                    if modality == "images":
                        images = question_data["images_doc_ids"]
                        for image in images:
                            for criteria in criterias:
                                self.analyse_image(criteria, os.path.join(image_dir, image))

                    elif modality == "text":
                        pass

                    elif modality == "tables":
                        pass

    def partition_calculation(self, criterias, modalities):
        pass

    def logical_entropy(self, partition):
        if len(partition) == 1:
            le = 1
        else:
            n_elements = sum([len(partition[key]) for key in partition])
            p = 1 / n_elements

            for key in partition:
                print(len(partition[key]), partition[key])

            cumulative_prob = sum([pow(p * len(partition[key]), 2) for key in partition])

            le = 1 - cumulative_prob

        return le

    def answer_question(self, question, question_data, question_files, image_dir, text_dir, table_dir, answer_dir,
                        modalities, mode):

        print("Let's answer the question")
        print(question_data)
        self.fill_criterias(question, question_data, question_files, image_dir, text_dir, table_dir, modalities, mode)

        """
        d = {"answer": None, "entropy_level": random.random()}
        with open(os.path.join(answer_dir, question), "w") as json_file:
            json.dump(d, json_file, indent=4)
        """


if __name__ == '__main__':
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    IMAGE_DIR = os.getenv("IMAGE_DIR")
    TEXT_DIR = os.getenv("TEXT_DIR")
    TABLE_DIR = os.getenv("TABLE_DIR")

    ASSOCIATION_DIR = os.getenv("ASSOCIATION_DIR")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")
    ANSWERS_DIR_TRAINING = os.getenv("ANSWERS_DIR_TRAINING")

    a = Agent(OPENAI_KEY, os.path.join(CRITERIA_EXTRACTION_DIR, "iteration_0"), MODALITIES)

    create_association_qa(QUESTIONS_MULTIMODALQA_TRAINING, IMAGE_DIR, TEXT_DIR, TABLE_DIR, ASSOCIATION_DIR)

    """
    mode = input("You want to try to answer with multiple modalities? Answer yes/no: ")

    if mode == "yes":
        answer_qa(QUESTIONS_MULTIMODALQA_TRAINING, ASSOCIATION_DIR, IMAGE_DIR, TEXT_DIR, TABLE_DIR,
                  ANSWERS_DIR_TRAINING, MODALITIES, "multimodal")
    else:
        answer_qa(QUESTIONS_MULTIMODALQA_TRAINING, ASSOCIATION_DIR, IMAGE_DIR, TEXT_DIR, TABLE_DIR,
                  ANSWERS_DIR_TRAINING, MODALITIES, "unimodal")
    """

    """
    log_e = a.logical_entropy({'filling': {'A', 'B', 'C'}, 'not_filling': {'D', 'E', 'F'}})
    print("Logical entropy is: ", log_e)
    log_e = a.logical_entropy({'filling': {'A'}, 'not_filling': {'B', 'C', 'D', 'E', 'F'}})
    print("Logical entropy is: ", log_e)
    """
