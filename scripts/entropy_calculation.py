import os
import ast
import time
import json
import random
import base64
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List


class AnswerContainsCriteria(BaseModel):
    answer: str = Field(..., description="Answer yes or no to whether the data contains the criteria")


class QuestionRespectRule(BaseModel):
    rule_id: str = Field(...,
                         description="ID of the rule that best matches the question; fallback rule is used if no specific rule applies")


class TableDescription(BaseModel):
    description: str = Field(..., descripton="High level description of the table")


class AnswerSetResponse(BaseModel):
    answer_set: List[str] = Field(..., description="Answer set for that type data modality")


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


def answer_qa(questions_dir, association_dir, image_dir, text_dir, table_dir, answers_dir, modalities):
    for question in sorted(os.listdir(questions_dir))[:1]:
        print("Question: ", question)
        # if question not in os.listdir(os.path.join(answers_dir, mode)):
        question_data = get_question_data(questions_dir, question)
        question_files = get_question_files(association_dir, question)
        a.answer_question(question, question_data, question_files, image_dir, text_dir, table_dir,
                          answers_dir, modalities)


class Agent:
    def __init__(self, key, path_criterias, modalities):
        self.key = key
        self.client = OpenAI(api_key=self.key)
        self.path_criterias = path_criterias
        self.modalities = modalities

    def create_connection(self, question, question_data, image_dir, text_dir, table_dir, association_dir):
        start_time = time.time()

        all_data_json_object = json.load(
            open("/Users/emanuelemezzi/PycharmProjects/multimodalqa/dataset/all_data.json", "rb"))

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

        criterias = [criteria_object["answer_class"]["answer_class"],
                     criteria_object["target"]["text"],
                     criteria_object["asked_property"]]

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

        return answer_set_images

    def create_text_set(self, question_data, all_data_text):
        print("Create text set")
        texts = question_data["text_doc_ids"]
        answer_set_text = []

        for text in texts:
            answer_set_text.append(all_data_text[text])

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

    def analyse_image(self, criteria, metadata, image64):
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
        return found.answer

    def analyse_text(self, criteria, paragraph, metadata):
        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are a text analysis assistant. Your task is to determine whether the given text 
                    contains a concept, statement, fact, or idea that matches the provided criteria, even if the wording differs
                    or the criteria is only implied. Respond only with 'yes' or 'no'.
                    """
                },
                {
                    "role": "user",
                    "content": f"""Does the following text with title {metadata} contain something that can be described as: '{criteria}'?
                    Text: '''{paragraph}'''
                    """
                }
            ],
            text_format=AnswerContainsCriteria,
        )

        found = response.output_parsed
        return found.answer

    def table_general_understanding(self, metadata, columns):
        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are a data understanding assistant. Your task is to infer the general topic and 
                    purpose of a table based only on its metadata and column names. Do NOT invent specific data values or row-level details.
                    Produce a concise, high-level description that applies to all rows.
                    """
                },
                {
                    "role": "user",
                    "content": f"""Table title / metadata: '''{metadata}'''
                    
                    Column names: '''{columns}'''
                    
                    Generate a short description of what this table is about.
                    """
                }
            ],
            text_format=TableDescription,
        )

        found = response.output_parsed
        return found.description

    def analyse_table_row(self, criteria, row, metadata, description):
        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are a table analysis assistant. 
                    Your task is to determine whether a single table row contains data that matches the provided 
                    criteria, either explicitly or implicitly.
                    Consider the meaning of the entire row, not just exact wording.
                    Respond only with 'yes' or 'no'.
                    """
                },
                {
                    "role": "user",
                    "content": f"""Does the following row of the table with title {metadata} and description {description}
                    contain something that can be described as: '{criteria}'?
                    Table row: '''{row}'''
                    """
                }
            ],
            text_format=AnswerContainsCriteria,
        )

        found = response.output_parsed
        return found.answer

    def create_answer_set(self, target_answer, elements_type, elements):
        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are an answer set extraction assistant.
                    Your task is to extract all values belonging to a specified answer category from a collection of elements.
                    
                    Only extract values that are explicitly present or can be unambiguously inferred from the provided data.
                    Do NOT invent values or use general world knowledge.
                    Return a deduplicated list.
                    """
                },
                {
                    "role": "user",
                    "content": f"""Target answer category: '''{target_answer}'''
                    Elements from which to extract the answer set ({elements_type}): '''{elements}'''
                    
                    Extract all values that belong to the target answer category.
                    """
                }
            ],
            text_format=AnswerSetResponse,
        )

        found = response.output_parsed
        return found.answer_set

    def decide_modality(self, question):

        rules_json = json.load(
            open("/Users/emanuelemezzi/PycharmProjects/LEGuidance/results/extracted_rules/initialization_rules.json",
                 "rb"))

        print(rules_json.keys())

        rules = [{f"R{i}": {"condition": rule["condition"], "predicted_modalities": rule["predicted_modalities"]}} for
                 i, rule in enumerate(rules_json["rules"])]

        print(rules)
        rules.append({
            f"R{len(rules_json['rules'])}": {
                "condition": rules_json["fallback_rule"]["condition"],
                "predicted_modalities": rules_json["fallback_rule"]["predicted_modalities"],
            }
        })

        print(rules)

        rule_conditions_string = ""

        for i, rule in enumerate(rules):
            rule_conditions_string += f"\t{i}. R{i}: {rule[f'R{i}']['condition']}\n"

        print(rule_conditions_string)

        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are a rule-matching assistant.
                        Your task is to determine which ONE rule condition is best satisfied by the given question.

                        Definitions:
                        - Each rule describes a situation based on the type of information required to answer a question.
                        - A rule is satisfied if the information required to answer the question matches the situation described in the rule condition.
                        - The LAST rule in the list is a fallback rule and MUST be selected if no other rule is a strong match.
                        
                        Instructions:
                        - Compare the question against EACH rule condition independently.
                        - Select EXACTLY ONE rule.
                        - Prefer the rule that best captures the core information need of the question.
                        - If no specific rule clearly applies, select the fallback rule (last rule).
                        - Do NOT invent new rules.
                        - Do NOT answer the question itself.
                        - Base your decision on semantic intent, not surface keywords.
                        """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"""Question: "{question}"
                            
                            Rule conditions:
                                {rule_conditions_string}
                        
                            Task: 
                                - Which rule condition is most strongly satisfied by the question? 
                                - If none apply, answer NONE.
                            """
                        }
                    ]
                }
            ],
            text_format=QuestionRespectRule,
        )

        rule_id = response.output_parsed.rule_id
        print("The rule id is: ", rule_id)

        for rule in rules:
            if rule_id in rule:
                return rule[rule_id]["predicted_modalities"]

    def fill_criterias(self, question: str, question_data: dict, question_files: dict,
                       image_dir: str, text_dir: str, table_dir: str, modalities: list, starting_modality: str):

        answer_set_image, answer_set_text, answer_set_table = [], [], []

        # Here we extract the criterias to search for in the modalities
        answer_class, *criterias = self.read_criterias(question)
        n_criterias = len(criterias)

        partitions = {"image": [], "text": [], "table": []}

        # Lower bound: only one modality, and upper bound by selecting one modality only
        # Upper bound: brute force And with this process we show the association between the question and the type of
        # modality to answer it (single one of fusion)
        # Check weather the additional modality is noise
        if starting_modality == "image":
            for criteria in criterias:
                print("Current criteria is: ", criteria)
                correct_elements = []
                image_set = question_files['image_set']
                for image in image_set:
                    metadata = image["title"]
                    image_base64 = encode_image(
                        os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                                     image["path"]))
                    answer = self.analyse_image(criteria, metadata, image_base64)
                    if answer.lower() == "yes":
                        correct_elements.append(image)

                partition = self.create_partition(image_set, correct_elements, criteria)
                partitions["image"].append(partition)

        elif starting_modality == "text":
            for criteria in criterias:
                print("Current criteria is: ", criteria)
                correct_elements = []
                text_set = question_files['text_set']
                print("The text set is: ", text_set)
                for text in text_set:
                    print("The text is: ", text)
                    metadata = text["title"]
                    answer = self.analyse_text(criteria, text["text"], metadata)

                    if answer.lower() == "yes":
                        correct_elements.append(text)

                partition = self.create_partition(text_set, correct_elements, criteria)
                partitions["text"].append(partition)

        elif starting_modality == "table":
            table_set = question_files["table_set"]
            table = table_set[0].copy()

            entire_table = json.load(open(os.path.join(table_dir, table['json']), 'rb'))
            title = entire_table['title']
            table_columns = [element["column_name"] for element in entire_table["table"]["header"]]
            rows = table[list(table.keys())[0]]

            table_description = self.table_general_understanding(title, table_columns)

            answer_set_table = self.create_answer_set(answer_class, "table_rows", rows)
            print("The answer set is: ", answer_set_table)

            for criteria in criterias:
                print("Current criteria is: ", criteria)
                correct_elements = []

                for row in rows:
                    answer = self.analyse_table_row(criteria, row, title, table_description)
                    if answer.lower() == "yes":
                        correct_elements.append(row)

                partition = self.create_partition(rows, correct_elements, criteria)

                partitions["table"].append(partition)

                filling = partition['splitting']['filling']
                not_filling = partition['splitting']['not_filling']

                print(partition['splitting']['filling'], len(filling))
                print(partition['splitting']['not_filling'], len(not_filling))

        # Check the partition with the lowest logical entropy (le > 0 and le < 1)
        with open("/Users/emanuelemezzi/PycharmProjects/LEGuidance/results/partitions_created/partitions.json",
                  "w") as json_file:
            json.dump(partitions, json_file, indent=4)

        return answer_class, answer_set_image, answer_set_text, answer_set_table

    def create_partition(self, all_data, selected_data, criteria):

        not_filling = [item for item in all_data if item not in selected_data]

        set_splitting = {'filling': selected_data, 'not_filling': not_filling}
        le = self.logical_entropy(set_splitting)
        partition = {'splitting': set_splitting, 'le': le, 'criteria': criteria}

        return partition

    def logical_entropy(self, partition):
        if len(partition) == 1:
            le = 1
        else:
            n_elements = sum([len(partition[key]) for key in partition])
            p = 1 / n_elements

            cumulative_prob = sum([pow(p * len(partition[key]), 2) for key in partition])

            le = 1 - cumulative_prob

        return le if le > 0 or le < 1 else 1

    def return_final_answer(self, answer_class, answer_set_image, answer_set_text, answer_set_table):
        answer_sets = {"image": answer_set_image, "text": answer_set_text, "table": answer_set_table}

        partitions = json.load(open(
            os.path.join("/Users/emanuelemezzi/PycharmProjects/LEGuidance/results/partitions_created",
                         "partitions.json"), "rb"))

        # The partition must bring the modality from which they were created. This way is it possible to
        fillings = [{"modality": modality, "i": i, "filling": partition["splitting"]["filling"], "le": partition["le"],
                     "criteria": partition["criteria"]} for modality in partitions.keys() for i, partition in
                    enumerate(partitions[modality])]

        # Minimum logical entropy among those
        min_le = min(x["le"] for x in fillings)

        # Minimum filling length among those
        min_filling_len = min(len(x["filling"]) for x in fillings if x["le"] == min_le)

        # Get ALL entries matching both
        min_le_filling = [x for x in fillings if x["le"] == min_le and len(x["filling"]) == min_filling_len]

        # Count the amounts of criterias that where able to isolate the element in the set
        criterias_filled = {}
        for element in min_le_filling:
            criterias_filled[element['i']] = {"criterias_filled": 1, "modality": element["modality"]}
            for filling in fillings:
                if element["criteria"] != filling["criteria"]:
                    for l in element["filling"]:
                        print("New criteria: ", filling["criteria"])
                        if l in filling["filling"]:
                            criterias_filled[element["i"]]["criterias_filled"] += 1

        print("Criterias filled")
        print(criterias_filled)

        # Find the index that has the maximum number of criterias filled
        modality = criterias_filled[max(criterias_filled, key=lambda k: criterias_filled[k]['criterias_filled'])][
            "modality"]

        print("The modality of election is: ", modality)

        answer_set = answer_sets[modality]

        print("The answer set is: ", answer_set)

        final_answer = None
        for filling in min_le_filling:
            for row in filling['filling']:
                for cell in row:
                    if answer_class == cell["header"].lower() and cell["text"] in answer_set:
                        final_answer = cell["text"]

        return final_answer

    def answer_question(self, question, question_data, question_files, image_dir, text_dir, table_dir, answer_dir,
                        modalities):

        print("Let's answer the question")
        print(question_data)

        # Starting modality: here we select the modality with which to start. The starting modality can also be the
        # finishing modality.
        starting_modality = self.decide_modality(question)
        print(starting_modality)

        answer_class, answer_set_image, answer_set_text, answer_set_table = "year", [], [], ["2012", "2018", "2019"]

        """
        answer_class, answer_set_image, answer_set_text, answer_set_table = self.fill_criterias(question, question_data,
                                                                                                question_files,
                                                                                                image_dir, text_dir,
                                                                                                table_dir, modalities,
                                                                                                starting_modality)
        """

        final_answer = self.return_final_answer(answer_class, answer_set_image, answer_set_text, answer_set_table)

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

    # create_association_qa(QUESTIONS_MULTIMODALQA_TRAINING, IMAGE_DIR, TEXT_DIR, TABLE_DIR, ASSOCIATION_DIR)

    answer_qa(QUESTIONS_MULTIMODALQA_TRAINING, ASSOCIATION_DIR, IMAGE_DIR, TEXT_DIR, TABLE_DIR,
              ANSWERS_DIR_TRAINING, MODALITIES)

    """
    log_e = a.logical_entropy({'filling': {'A', 'B', 'C'}, 'not_filling': {'D', 'E', 'F'}})
    print("Logical entropy is: ", log_e)
    log_e = a.logical_entropy({'filling': {'A'}, 'not_filling': {'B', 'C', 'D', 'E', 'F'}})
    print("Logical entropy is: ", log_e)
    """
