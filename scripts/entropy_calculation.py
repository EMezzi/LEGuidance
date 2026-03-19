import os
import time
import json
import base64
import numpy as np

from miscellaneous.pydantic_schemas import *

from miscellaneous.prompt import (
    system_prompt_image, user_prompt_image_text_input, user_prompt_image_image_input,
    system_prompt_text, user_prompt_text,
    system_prompt_table, user_prompt_table,
    system_prompt_row, user_prompt_row,
)

from miscellaneous.json_schemas import json_schema_check_criteria, json_schema_table_description

iteration = 'iteration_1'


def get_question_data(question_dir, question):
    question_json = json.load(open(os.path.join(question_dir, question), "rb"))
    question_text = question_json["question"]

    images_doc_ids = question_json["metadata"]["image_doc_ids"]
    text_doc_ids = question_json["metadata"]["text_doc_ids"]
    table_id = question_json["metadata"]["table_id"]

    return {"question_text": question_text, "image_doc_ids": images_doc_ids, "text_doc_ids": text_doc_ids,
            "table_id": table_id}


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


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        if isinstance(row, list):
            flat_list.extend(row)

    return flat_list


def create_association_qa(a, questions_dir, association_dir, image_dir, text_dir, table_dir):
    for question in sorted(os.listdir(questions_dir)):
        if question not in os.listdir(association_dir):
            print("Question: ", question)
            question_data = get_question_data(questions_dir, question)
            print("Question data: ", question_data)
            a.create_connection(question, question_data, image_dir, text_dir, table_dir, association_dir)


def average_modality_le(partitions):
    d = {}
    for modality, items in partitions.items():
        le_values = [item['le'] for item in items if "le" in item and item['le'] > 0]
        if le_values:
            d[modality] = sum(le_values) / len(le_values)
        else:
            d[modality] = None

    return d


class Agent:
    def __init__(self, client, path_criterias, modalities):
        self.client = client
        self.path_criterias = path_criterias
        self.modalities = modalities

    def create_connection(self, question, question_data, association_dir):
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

    def read_criterias(self, question, unimodal):
        criteria_object = json.load(open(os.path.join(self.path_criterias, question), "rb"))

        if unimodal:
            criterias = [criteria_object["expected_answer_type"]["expected_answer_type_specific"],
                         criteria_object["expected_answer_type"]["expected_answer_type_general"],
                         criteria_object["rewritten_question"]]

            for constraint in criteria_object["constraints"]:
                criterias.append(constraint['evidence'])

            return criterias

        else:
            print("ciao come stai")
            # Expected answer type will be used also as criteria for the splitting.
            criterias = [criteria_object["expected_answer_type"]["expected_answer_type_specific"],
                         criteria_object["expected_answer_type"]["expected_answer_type_general"],
                         criteria_object["rewritten_question"],
                         criteria_object["expected_answer_type"]["expected_answer_type_specific"],
                         criteria_object["rewritten_question"],
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
        answer_set_text = [all_data_text[text] for text in texts]

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

    def analyse_image(self, model, criteria, metadata, image):
        """This method checks whether an image can be separated from the others based on a specific criteria."""

        if model == "gpt-5.2":
            image64 = encode_image(
                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                             image["path"]))

            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_image
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt_image_text_input.format(metadata=metadata, criteria=criteria),
                            },
                            {
                                "type": "input_image",
                                "image_url": user_prompt_image_image_input.format(image64=image64),
                            },
                        ]
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

        elif model == "claude-sonnet-4-5":
            print("Image is: ", image["path"])

            with open(os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                                   image["path"]), "rb") as f:
                image_bytes = f.read()

            media_type = detect_media_type_from_bytes(image_bytes)

            image64 = encode_image(
                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                             image["path"]))  # base64.standard_b64encode(image_bytes).decode("utf-8")

            response = self.client.messages.create(
                model=model,
                system=system_prompt_image,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt_image_text_input.format(metadata=metadata, criteria=criteria),
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image64,
                                }
                            },
                        ]
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema_check_criteria
                    }
                }
            )

            found = json.loads(response.content[0].text)
            return found["answer"]

    def analyse_image_restricting_criteria(self, model, criteria, metadata, image_path):
        if model == "gpt-5.2":
            image64 = encode_image(
                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                             image_path))

            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a visual reasoning assistant.

Your task is to determine whether a given IMAGE or its METADATA/TITLE contains visual evidence of any facts expressed in the CRITERIA.

Rules:
1) Only use what is directly visible in the image and the provided metadata/title text.
2) Do NOT use external knowledge or assumptions.
3) The image only needs to contain evidence of **any of the provided facts**, not the full original criteria.
4) If at least one fact is present in the image or metadata, respond "yes". Otherwise, respond "no".

Respond ONLY with "yes" or "no"."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"""
Given:
- Image TITLE/METADATA: '{metadata}'
- CRITERIA: {criteria}

Task:
Determine whether the image or its metadata/title contains visual evidence of any facts expressed in the criteria.

Answer "yes" if at least one fact is present, otherwise answer "no".
""",
                            },
                            {
                                "type": "input_image",
                                "image_url": user_prompt_image_image_input.format(image64=image64),
                            },
                        ]
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def analyse_text_bridge_element(self, model, question_text, title, criteria, text):
        print("variables")
        print(question_text)
        print(criteria)
        print(text)

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a reasoning assistant.

You are given: 
- A QUESTION
- Two paragraphs: PARAGRAPH 1 AND PARAGRAPH 2

Your task is to determine whether any entity, concept, or fact mentioned in PARAGRAPH 1 is also present in PARAGRAPH 2.

Rules:
1) Use only the text provided.
2) Do NOT require full fact matching.
3) If any shared entity, name, or concept (e.g., "Manchester United") appears in both paragraphs, respond "yes".

Respond ONLY with "yes" or "no"."""
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
                        {question_text}
                        
TITLE PARAGRAPH 1: 
{title}

PARAGRAPH 1: 
{criteria}

PARAGRAPH 2:
{text}

Task:
Determine whether any entity, concept, or fact mentioned in PARAGRAPH 1 is also present in PARAGRAPH 2..

Answer "yes" or "no"."""
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def analyse_text_restricting_criteria(self, model, criteria, metadata, text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a reasoning assistant.

Your task is to determine whether any of the concepts contained in CRITERIA are also contained in PARAGRAPH or its METADATA/TITLE.

Rules:
1) Use only information explicitly stated in CRITERIA, the PARAGRAPH and its TITLE.
2) Do NOT use external knowledge, assumptions, or inference beyond what is written.
4) If at least one fact is explicitly supported by the PARAGRAPH, or METADATA/TITLE, respond "yes".

Respond ONLY with "yes" or "no"."""
                    },
                    {
                        "role": "user",
                        "content": f"""
            TITLE/METADATA: 
            {metadata}
            
            PARAGRAPH:
            {text}
            
            CRITERIA: 
            {criteria}

            Task:
            Determine whether the content of the PARAGRAPH or METADATA/TITLE contains any concept expressed in the CRITERIA.

            Answer "yes" or "no"."""
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def analyse_table_row_restricting_criteria(self, model, criteria, row, title, name, description):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a reasoning assistant.

Your task it to determine whether any of the concepts contained in CRITERIA also contained in TABLE ROW.

Rules:
1) Use only information explicitly stated in the table row.
2) Do NOT use external knowledge, assumptions, or inference beyond what is written.
3) If at least one fact is explicitly supported by the row, title, or description, respond "yes".

Respond ONLY with "yes" or "no"."""
                    },
                    {
                        "role": "user",
                        "content": f"""
TABLE ROW:
{row}

CRITERIA:
{criteria}

Task:
Determine whether the content of the TABLE ROW contains any concept expressed in the CRITERIA.

Answer "yes" or "no".
"""
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def extract_restricting_criteria_text(self, model, question_text, title, text, characteristics):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a reasoning module for a multimodal question answering system.
                        
Your task is to identify implicit intermediate information required to answer a question and determine whether the provided text paragraph contains that information.
A question may require an entity or value that is not explicitly stated but must be inferred from external evidence. This missing element is called an implicit bridge element.

You are given:
- The QUESTION
- The PARAGRAPH TITLE
- The CHARACTERISTICS previously checked that distinguish this PARAGRAPH from the others
- The PARAGRAPH TEXT

Step 1 — Question analysis
Determine whether answering the question requires identifying an intermediate element not explicitly given.

If such an element exists, extract:

Target_type:
The semantic category of the missing element. Do NOT output a specific instance.

Condition:
The property or constraint the missing element must satisfy according to the question.

Step 2 — Evidence inspection
You will receive a candidate text paragraph.
Determine whether it contains information that satisfies the condition considering also the CHARACTERISTICS that helped distinguishing the PARAGRAPH from the others in the previous steps.

Step 3 — Evidence extraction
If the paragraph contains the required information, extract the entity/value and the minimal supporting text.

Guidelines:
- Only extract information explicitly present in the paragraph.
- Do not infer unsupported facts.
- Be concise and precise.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
{question_text} 
        
PARAGRAPH TITLE:
{title}
 
CHARACTERISTICS: 
{characteristics}

PARAGRAPH TEXT:
{text}

Provide the relevant information according to the instructions.
"""
                    },
                ],
                text_format=BridgeElement,
            )

            bridge_element = response.output_parsed
            print(bridge_element)
            return bridge_element.extracted_information

    def extract_restricting_criteria_image(self, model, question_text, image_title, image_path):

        image64 = encode_image(
            os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                         image_path))

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a reasoning module for a multimodal question answering system.
                        
        Your task is to extract relevant information from this IMAGE in relation to the QUESTION received. 

        You are given:
        - The QUESTION
        - The IMAGE TITLE
        - The IMAGE
        
        Step 1 - Question analysis: 
        - Given the QUESTION, the IMAGE TITLE, and the IMAGE check if the information in the IMAGE is related to the QUESTION.
                
        Step 2 - Evidence extraction: 
        - If the IMAGE contains information related to the QUESTION, create a short textual description of the IMAGE taking into account the QUESTION. The description must be minimal and contain the entity/value that connects the IMAGE to the QUESTION.

        Guidelines:
        - In the description only include information explicitly present in the image.
        - Do not infer unsupported facts.
        - Be concise and precise.
        """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"""QUESTION:
        {question_text}

        IMAGE TITLE:
        {image_title}

        Provide the relevant information according to the instructions.
        """,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"""data:image/jpeg;base64,{image64}""",
                            },
                        ]
                    }
                ],
                text_format=ImageExtraction,
            )

            bridge_element = response.output_parsed
            print(bridge_element)
            return bridge_element.evidence

    def extract_restricting_criteria_table_row(self, model, question_text, document_title, table_name,
                                               table_description, table_row, characteristics):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a reasoning module for a multimodal question answering system.
                        
                Your task is to extract relevant information from this TABLE_ROW in relation to the QUESTION received. 
                
                You are given: 
                - The QUESTION
                - The DOCUMENT TITLE of the document containing the table
                - The TABLE NAME
                - The TABLE description
                - The TABLE ROW (with all cell values)
                    
                Step 1 - Question analysis: 
                - Given the QUESTION, the TABLE NAME, the TABLE DESCRIPTION, and the TABLE ROW check if the information in the TABLE ROW is related to the QUESTION
                
                Step 2 - Evidence extraction: 
                - If the TABLE ROW contains information related to the QUESTION, create a short textual description of the TABLE ROW taking into account the QUESTION. The description must be minimal and contain the entity/value that connects the TABLE_ROW to the QUESTION

        Guidelines:
        - In the description only include information explicitly present in the table row.
        - Do not infer unsupported facts.
        - Be concise and precise.
                """
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
        {question_text}

        DOCUMENT TITLE:
        {document_title}

        TABLE NAME:
        {table_name}

        TABLE DESCRIPTION: 
        {table_description}

        TABLE ROW: 
        {table_row}

        Provide the relevant information according to the instructions.
        """
                    },
                ],
                text_format=TableRowExtraction,
            )

            bridge_element = response.output_parsed
            print(bridge_element)
            return bridge_element.evidence

    def analyse_text(self, model, criteria, paragraph, metadata):
        """This method checks whether a text can be separated from the others based on a specific criteria."""

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_text
                    },
                    {
                        "role": "user",
                        "content": user_prompt_text.format(metadata=metadata, criteria=criteria, paragraph=paragraph)
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

        elif model == "claude-sonnet-4-5":
            response = self.client.messages.create(
                model=model,
                system=system_prompt_text,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt_text.format(metadata=metadata, criteria=criteria, paragraph=paragraph)
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema_check_criteria
                    }
                }
            )
            print(response.content[0].text)
            found = json.loads(response.content[0].text)
            return found["answer"]

    def table_general_understanding(self, model, table_title, table_name, columns):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_table
                    },
                    {
                        "role": "user",
                        "content": user_prompt_table.format(table_title=table_title, table_name=table_name,
                                                            columns=columns)
                    }
                ],
                text_format=TableDescription,
            )

            found = response.output_parsed
            return found.description

        elif model == "claude-sonnet-4-5":
            response = self.client.messages.create(
                model=model,
                system=system_prompt_table,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt_table.format(table_title=table_title, table_name=table_name,
                                                            columns=columns)
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema_table_description
                    }
                }
            )

            found = json.loads(response.content[0].text)
            return found["description"]

    def analyse_table_row(self, model, criteria, row, metadata, description):
        """This method checks whether a table row can be separated from the others based on a specific criteria."""

        # Split the search in the table in two phases. First you only do the splitting with the metadata, and then
        # you try to do the partitioning with the rows

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_row
                    },
                    {
                        "role": "user",
                        "content": user_prompt_row.format(metadata=metadata, description=description, criteria=criteria,
                                                          row=row)
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

        elif model == "claude-sonnet-4-5":
            response = self.client.messages.create(
                model=model,
                system=system_prompt_row,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt_row.format(metadata=metadata, description=description, criteria=criteria,
                                                          row=row)
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema_check_criteria
                    }
                }
            )

            found = json.loads(response.content[0].text)
            return found["answer"]

    def decide_modality_llm(self, question, images, texts, table, table_dir, final_dataset_images):
        """This method decides the modality by showing the data to the LLM"""

        images_text = "\n\n".join([
            f"{i + 1}. Title: {img['title']}"
            for i, img in enumerate(images)
        ])

        image_inputs = []
        for img in images:
            image64 = encode_image(os.path.join(final_dataset_images, img["path"]))
            image_inputs.append({
                "title": img["title"],
                "image_url": f"data:image/jpeg;base64,{image64}"
            })

        paragraphs_text = "\n\n".join([
            f"{i + 1}. Title: {p['title']}\nContent: {p['text']}"
            for i, p in enumerate(texts)
        ])

        json_table = json.load(open(os.path.join(table_dir, table["json"]), "rb"))

        tables_text = f"""
            Table Title: {json_table["title"]}
            Table name: {json_table["table"]["table_name"]}
            Content: {json_table["table"]}
            """

        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": """You are a multimodal modality-selection classifier.

        You will be given:
        - A question
        - IMAGES (titles + image content)
        - TEXT paragraphs (titles + content)
        - TABLES (titles + table content)

        Your job is to decide which modality combination is required to answer the question.

        IMPORTANT:
        The output must be EXACTLY ONE of the following labels:
        - image          (answer can be derived from image alone)
        - text           (answer can be derived from text alone)
        - table          (answer can be derived from table alone)
        - image_text     (answer requires combining IMAGE + TEXT)
        - image_table    (answer requires combining IMAGE + TABLE)
        - text_table     (answer requires combining TEXT + TABLE)

        Rules:
        - Use ONLY the provided data.
        - Do NOT use outside knowledge or your knowledge.
        - Do NOT guess.
        - Pick the best matching label.
        - Output ONLY the label (no explanation).
        """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"""
        Question:
        {question}

        IMAGES:
        {images_text}

        TEXT paragraphs:
        {paragraphs_text}

        TABLES:
        {tables_text}

        Which modality or modality combination is required? Return only one label.
        """
                        },
                        *[
                            {"type": "input_image", "image_url": img["image_url"]}
                            for img in image_inputs
                        ]
                    ]
                }
            ],
            text_format=ModalityDecision,
        )

        return response.output_parsed.modalities

    def decide_modality_reduced_data(self, question, remaining_modalities, images, texts, table, table_dir,
                                     final_dataset_images):
        """In this method we want to find the modality in case the ones chosen were not enough"""
        # print("Final dataset images: ", final_dataset_images)
        # print("Question is: ", question)

        images_text, image_inputs, paragraphs_text, tables_text = None, None, None, None
        for modality in remaining_modalities:
            if modality == "image":
                images_text = "\n\n".join([
                    f"{i + 1}. Title: {img['title']}"
                    for i, img in enumerate(images)
                ])

                image_inputs = []
                for img in images:
                    image64 = encode_image(os.path.join(final_dataset_images, img["path"]))
                    image_inputs.append({
                        "title": img["title"],
                        "image_url": f"data:image/jpeg;base64,{image64}"
                    })

            elif modality == "text":
                paragraphs_text = "\n\n".join([
                    f"{i + 1}. Title: {p['title']}\nContent: {p['text']}"
                    for i, p in enumerate(texts)
                ])

            elif modality == "table":
                json_table = json.load(open(os.path.join(table_dir, table["json"]), "rb"))

                tables_text = f"""Table Title: {json_table["title"]} \nTable name: {json_table["table"]["table_name"]} \nContent: {json_table["table"]}"""

        content_sections = []

        if images_text is not None:
            content_sections.append(f"""IMAGES: \n{images_text}""")

        if paragraphs_text is not None:
            content_sections.append(f"""TEXT paragraphs: \n{paragraphs_text}""")

        if tables_text is not None:
            content_sections.append(f"""TABLES: \n{tables_text}""")

        available_content = "\n\n".join(content_sections)

        response = self.client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": f"""You are a multimodal modality-selection classifier in recovery mode.

A previous attempt to answer the question using a certain modality has failed.

Your task is to decide which of the REMAINING candidate modalities is most appropriate to answer the question.

You will be given:
- A question
- The available data (images, text paragraphs, tables)
- The modality that was already attempted and failed
- A restricted set of candidate modalities to choose from

IMPORTANT:
- You must choose ONLY from the provided candidate modalities: {remaining_modalities}.
- Do NOT select the previously attempted modality.
- Use ONLY the provided data.
- Do NOT use external knowledge.
- Do NOT guess.
- Select the modality that is most likely required to answer the question.
- Output ONLY the selected label (no explanation).
"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"""
                Question:
                {question}

                {available_content}

                Which modality or modality combination is required? Return only one label.
                """
                        },
                        *(
                            [
                                {"type": "input_image", "image_url": img["image_url"]}
                                for img in image_inputs
                            ]
                            if image_inputs is not None else []
                        )
                    ]
                }
            ],
            text_format=ModalityDecision,
        )

        return response.output_parsed.modalities

    def fill_criteria_images(self, model, criterias, question_files, partitions):
        """This method checks whether the splitting is possible for each criterion for images."""

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []
            image_set = question_files['image_set']
            for image in image_set:
                # print(f"Image: {image}")
                metadata = image["title"]
                answer = self.analyse_image_restricting_criteria(model, criteria, metadata, image["path"])
                # answer = self.analyse_image(model, criteria, metadata, image)

                if answer.lower() == "yes":
                    correct_elements.append(image)

            partition = self.create_partition(image_set, correct_elements, criteria)
            partitions["image"].append(partition)

    def fill_criteria_text(self, model, criterias, question_files, partitions):
        """This method checks whether the splitting is possible for each criterion for texts."""

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []
            text_set = question_files['text_set']
            for text in text_set:
                # print(f"Text: {text}")
                metadata = text["title"]
                answer = self.analyse_text_restricting_criteria(model, criteria, text["text"], metadata)
                # answer = self.analyse_text(model, criteria, text["text"], metadata)

                if answer.lower() == "yes":
                    correct_elements.append(text)

            partition = self.create_partition(text_set, correct_elements, criteria)
            partitions["text"].append(partition)

            # filling = partition['splitting']['filling']
            # not_filling = partition['splitting']['not_filling']

            # print(partition['splitting']['filling'], len(filling))
            # print(partition['splitting']['not_filling'], len(not_filling))

    def fill_criteria_table(self, model, criterias, question_files, table_dir, partitions):
        """This method checks whether the splitting is possible for each criterion for table rows."""
        table_set = question_files["table_set"]
        table = table_set[0].copy()

        entire_table = json.load(open(os.path.join(table_dir, table['json']), 'rb'))
        table_title = entire_table['title']
        table_name = entire_table['table']['table_name']

        print("URL table: ", entire_table['url'])
        table_columns = [element["column_name"] for element in entire_table["table"]["header"]]
        rows = table[list(table.keys())[0]]

        table_description = self.table_general_understanding(model, table_title, table_name, table_columns)
        print(f"Table description: {table_description}")

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []

            for row in rows:
                # print(f"Row: {row}")
                answer = self.analyse_table_row_restricting_criteria(model, criteria, row, table_title, table_name,
                                                                     table_description)
                # answer = self.analyse_table_row(model, criteria, row, table_title, table_description)
                if answer.lower() == "yes":
                    correct_elements.append(row)

            partition = self.create_partition(rows, correct_elements, criteria)
            partition["table_understanding"] = table_description
            partition["table_title"] = table_title
            partition["table_name"] = table_name

            partitions["table"].append(partition)

            # filling = partition['splitting']['filling']
            # not_filling = partition['splitting']['not_filling']

            # print("Parition filling: ", partition['splitting']['filling'], len(filling))
            # print("Partition not filling: ", partition['splitting']['not_filling'], len(not_filling))

    def check_answer_in_row(self, model, answer_class_specific, row, question_text, table_description,
                            conditional_criteria):

        print("Check it:")

        if conditional_criteria:
            if model == "gpt-5.2":
                response = self.client.responses.parse(
                    model="gpt-5.2",
                    input=[
                        {
                            "role": "system",
                            "content": """You are an evidence verifier for a multimodal question answering system.
    
    Your goal is to determine whether the TABLE ROW contains the answer value to the QUESTION.
    
    You are given:
    - QUESTION
    - The value the question asks for (EXPECTED ANSWER TYPE)
    - TABLE DESCRIPTION
    - OPTIONAL ADDITIONAL CONTEXTUAL INFORMATION
    - TABLE ROW
    - The answer value
    
    Important principles:
    
    1. A QUESTION may contain multiple constraints.
    2. The ADDITIONAL CONTEXTUAL INFORMATION may already satisfy some of those constraints.
    3. The TABLE ROW does NOT need to satisfy every constraint.
    4. The TABLE ROW only needs to contain the final answer value.
    
    Procedure:
    
    Step 1: Identify which constraints are already satisfied by the contextual information.
    Step 2: Check whether the TABLE ROW contains the answer value.
    Step 3: Verify that the value is consistent with the question when combined with the contextual information.
    
    Decision rules:
    
    Return TRUE if:
    - the answer value appears explicitly in the TABLE ROW, and
    - the value satisfies the QUESTION when combined with the contextual information.
    
    Return FALSE if:
    - the answer value does not appear in the row.
    
    Extraction rules:
    
    If contains = TRUE:
    - Extract the exact value from the TABLE ROW.
    - Copy the value exactly as written.
    
    If contains = FALSE:
    - answer = NONE
    
    Output format:
    
    contains: TRUE or FALSE
    answer: <exact cell value or NONE>
    """
                        },
                        {
                            "role": "user",
                            "content": f"""QUESTION:
    {question_text}
    
    EXPECTED ANSWER TYPE:
    {answer_class_specific}
    
    TABLE DESCRIPTION:
    {table_description}
    
    OPTIONAL ADDITIONAL CONTEXTUAL INFORMATION:
    {conditional_criteria}
    
    TABLE ROW:
    {row}
    """
                        }
                    ],
                    text_format=RowContainsAnswer,
                )

                found = response.output_parsed
                return found.contains, found.entity, found.confidence

        else:
            if model == "gpt-5.2":
                response = self.client.responses.parse(
                    model="gpt-5.2",
                    input=[
                        {
                            "role": "system",
                            "content": """You are a semantic evidence verifier for a multimodal question answering system.

            Your task is to determine whether a TABLE ROW or TABLE ROWS contains a value that directly answers the QUESTION.

            You are given:
            - The QUESTION
            - The EXPECTED ANSWER TYPE
            - The TABLE DESCRIPTION
            - The TABLE ROW or TABLE ROWS (with all cell values)

            This is NOT keyword matching.
            You must perform semantic reasoning.

            Guidelines:
            - Consider synonyms and contextual meaning.
            - Use the table description to interpret the row correctly.
            - Only return TRUE if one of the row's values directly answers the question.
            - If the row contains related information but no direct answer, return FALSE.
            - If uncertain, return FALSE.

            Extraction rules:
            - If contains = TRUE:
                - Extract the exact cell value from the row.
                - Do not paraphrase.
                - Do not normalize.
                - Copy the value exactly as written.
            - If contains = FALSE:
                - answer must be "NONE".

            Base your decision strictly on the provided row content.
            """
                        },
                        {
                            "role": "user",
                            "content": f"""QUESTION:
            {question_text}

            EXPECTED ANSWER TYPE:
            {answer_class_specific}

            TABLE DESCRIPTION:
            {table_description}

            TABLE ROW:
            {row}
            """
                        }
                    ],
                    text_format=RowContainsAnswer,
                )

                found = response.output_parsed
                return found.contains, found.entity, found.confidence

    def check_answer_in_paragraph(self, model, answer_class, question_text, paragraph_text, contextual_information):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a semantic evidence verifier for a multimodal question answering system.
                        
        You are given:
        - A QUESTION 
        - A PARAGRAPH
        - An EXPECTED ANSWER TYPE
        - CONTEXTUAL INFORMATION

        Your task is to determine whether a PARAGRAPH contains an entity that completely or partially matches the EXPECTED ANSWER TYPE required by a QUESTION.

        Rules:
        1. This is NOT keyword matching. You must perform semantic reasoning.
        2. Consider synonyms, paraphrases, and implicit mentions.
        3. You may use the CONTEXTUAL INFORMATION to resolve references in the QUESTION.
        4. Only return TRUE if the PARAGRAPH contains a specific entity that could fully or partially answer the question. 
           Partial answers are acceptable if they are relevant.
        5. If the paragraph only provides related context but no actual answer entity, return FALSE.
        6. If uncertain, return FALSE.

        Extraction rules:
        - If contains = TRUE, extract the exact text span from the paragraph.
        - Do not paraphrase.
        - Do not normalize.
        - Copy the entity exactly as written.
        - If contains = FALSE, entity must be "NONE".

        Base your decision strictly on the provided text and contextual information.
        """
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
        {question_text}

        EXPECTED ANSWER TYPE:
        {answer_class}

        PARAGRAPH:
        {paragraph_text}
        
        CONTEXTUAL INFORMATION:
        {contextual_information}
        """
                    }
                ],
                text_format=ParagraphContainsAnswer,
            )

            found = response.output_parsed
            return found.contains, found.entity, found.confidence

    def check_answer_in_caption(self, model, answer_class_specific, answer_class_general, question_text, caption_text):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a semantic evidence verifier for a multimodal question answering system.

Your task is to determine whether a CAPTION contains an entity that matches the EXPECTED ANSWER TYPE required by a QUESTION.

Two answer types are provided:
- SPECIFIC ANSWER TYPE (more precise)
- GENERAL ANSWER TYPE (broader category)

You must analyze ONLY the provided caption text.
Do NOT infer beyond what is explicitly stated in the caption.

Matching logic:
1. First determine whether the caption contains an entity that satisfies the SPECIFIC answer type.
2. If not, determine whether it contains an entity that satisfies the GENERAL answer type.
3. If neither is satisfied, return contains = FALSE.

Guidelines:
- Consider synonyms and paraphrases.
- The caption must explicitly describe the entity.
- Do not infer roles, professions, or properties unless directly stated.
- If uncertain, return FALSE.
- Prefer precision over recall.

Extraction rules:
- If contains = TRUE, extract the exact text span from the caption.
- Do not paraphrase.
- Copy the entity exactly as written.
- If no match, entity must be "NONE".

Return:
- match_level = "specific" if specific type matched
- match_level = "general" if only general type matched
- match_level = "none" if no match
"""
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
{question_text}

SPECIFIC ANSWER TYPE:
{answer_class_specific}

GENERAL ANSWER TYPE:
{answer_class_general}

CAPTION:
{caption_text}
"""
                    }
                ],
                text_format=ImageContainsAnswer,
            )

            found = response.output_parsed
            return found.contains, found.entity, found.match_level, found.confidence

    def check_answer_caption_image(self, model, answer_class_specific, answer_class_general, question_text,
                                   caption_text, image_input):

        if model == "gpt-5.2":
            image64 = encode_image(
                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                             image_input))

            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a multimodal evidence verifier for a question answering system.

    Your task is to determine whether either the CAPTION or the IMAGE contains an entity 
    that matches the EXPECTED ANSWER TYPE required by a QUESTION.

    Two answer types are provided:
    - SPECIFIC ANSWER TYPE (more precise)
    - GENERAL ANSWER TYPE (broader category)

    You are given:
    - A QUESTION
    - A CAPTION
    - An IMAGE

    You must evaluate both sources of evidence.

    Matching logic:

    1. First check whether the CAPTION explicitly contains an entity 
       that satisfies the SPECIFIC answer type.
    2. If not, check whether the IMAGE contains a visually identifiable entity 
       that satisfies the SPECIFIC answer type.
    3. If still not satisfied, repeat steps 1–2 for the GENERAL answer type.
    4. If neither modality satisfies either answer type, return contains = FALSE.

    Guidelines:

    - Caption evidence must be explicitly stated text.
    - Image evidence must be visually observable.
    - Do NOT infer beyond what is written in the caption.
    - Do NOT rely on external knowledge beyond what is visually observable.
    - Do NOT guess identities unless clearly recognizable or explicitly written.
    - Do NOT assume roles or properties unless directly supported.
    - If uncertain, return contains = FALSE.

    Extraction rules:

    - If evidence comes from the CAPTION, extract the exact text span.
    - If evidence comes from the IMAGE, provide a concise visual description.
    - If no match, entity must be "NONE".
    
    Critical instructions for the IMAGE:

    - Examine all parts of the image carefully, including background, corners, and small or partially obscured objects.
    - Pay attention to subtle details such as tiny text, small objects, fine patterns, colors, positions, or partially visible entities.
    - Do NOT rely only on large, obvious objects. Small details may be crucial to answer the question.
    - Only return TRUE for contains if the entity is clearly present in the image.
    - If uncertain, return FALSE.

    Return:
    - contains: TRUE or FALSE
    - entity: extracted entity or "NONE"
    - match_level: "specific" | "general" | "none"
    - confidence: float between 0 and 1
    """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"""QUESTION:
    {question_text}

    SPECIFIC ANSWER TYPE:
    {answer_class_specific}

    GENERAL ANSWER TYPE:
    {answer_class_general}

    CAPTION:
    {caption_text}
    """
                            },
                            {
                                "type": "input_image",
                                "image_url": f"""data:image/jpeg;base64,{image64}""",
                            },
                        ],
                    },
                ],
                text_format=ImageContainsAnswer,
            )

            found = response.output_parsed
            return found.contains, found.entity, found.match_level, found.confidence

    def create_unimodal_partitions(self, model, question, question_files, table_dir, modality, criterias):
        """This method checks whether images, text, and table rows, can be split based on the distinction criterias."""

        unimodal_partition_path = os.path.join(
            f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")

        if os.path.exists(unimodal_partition_path):
            partitions = json.load(open(unimodal_partition_path, "rb"))
        else:
            partitions = {"image": [], "text": [], "table": []}

        filled_modalities = [key for key, value in partitions.items() if value]

        print(f"MODALITIES ALREADY WITH PARTITIONS CREATED: {filled_modalities}")

        if modality not in filled_modalities:
            if modality == "image":
                self.fill_criteria_images(model, criterias, question_files, partitions)

            elif modality == "text":
                self.fill_criteria_text(model, criterias, question_files, partitions)

            elif modality == "table":
                self.fill_criteria_table(model, criterias, question_files, table_dir, partitions)

            # Check the partition with the lowest logical entropy (le > 0 and le < 1)
            with open(f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}",
                      "w") as json_file:
                json.dump(partitions, json_file, indent=4)

    def create_partition(self, all_data, selected_data, criteria):
        """This method creates the partition."""

        not_filling = [item for item in all_data if item not in selected_data]

        set_splitting = {'filling': selected_data, 'not_filling': not_filling}
        le = self.logical_entropy(set_splitting)
        partition = {'splitting': set_splitting, 'le': le, 'criteria': criteria}

        return partition

    def yesnoquestion(self, model, question_text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a question type classifier.

Your task is to determine whether a QUESTION can be correctly answered with only "yes" or "no".

Definition:
A YES/NO question is one whose complete and correct answer is either:
- "yes"
- "no"

Rules:
- If the question requires a name, number, date, list, explanation, or description, or some other term it is NOT a yes/no question.
- Do not rely only on the first word; evaluate the meaning.
- If uncertain, return FALSE.

Return:
- is_yes_no = TRUE if the question can be answered strictly with yes or no.
- is_yes_no = FALSE otherwise.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
                        {question_text}
                        """
                    }
                ],
                text_format=YesNoQuestion,
            )

            found = response.output_parsed
            return found.is_yes_no, found.confidence

    def logical_entropy(self, partition):
        """This method calculates the logical entropy, which depends on how the elements were split."""
        if len(partition) == 1:
            le = 1
        else:
            n_elements = sum([len(partition[key]) for key in partition])
            p = 1 / n_elements
            cumulative_prob = sum([pow(p * len(partition[key]), 2) for key in partition])
            le = 1 - cumulative_prob

        return le if le > 0 or le < 1 else 1

    def decide_answer_modality(self, model, question_text, tied_modalities):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a modality selection agent in a multimodal system.

Given a QUESTION and a list of AVAILABLE MODALITIES Your task is to decide which modality is more appropriate to answer the given QUESTION when the AVAILABLE MODALITIES are tied in confidence score.

Decision Rules:
1. Choose the modality that provides the most reliable and direct evidence for answering the question.
4. If both could work, choose the modality that minimizes ambiguity and hallucination risk.
5. Output ONLY one of the words in the list of available modalities.
6. Do not explain your reasoning.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION: {question_text}

AVAILABLE MODALITIES tied in confidence:
{tied_modalities}

Which modality should be used to answer the question?
Return only one between the {tied_modalities}.
"""
                    }
                ],
                text_format=ModalityDecision,
            )

            found = response.output_parsed
            return found.modalities

    def extract_partitions(self, model, question, mod_chosen, question_text, partitions_path):
        """In this question we check whether it is possible to return an answer with only one modality"""
        # We state that if the predicted modality was one and if there is a partition with minimum logical entropy
        # then we can try to verify whether we can give an answer

        partitions = json.load(
            open(os.path.join(f"../results/partitions/{partitions_path}/{iteration}/{model}", f"partitions_{question}"),
                 "rb"))

        if mod_chosen:
            print("We are in mod chosen case")
            fillings = [
                {"modality": mod_chosen, "i": i, "filling": partition["splitting"]["filling"], "le": partition["le"],
                 "criteria": partition["criteria"]} for i, partition in
                enumerate(partitions[mod_chosen]) if partition["le"] > 0]

            final_answer = None

            if not fillings:
                return None, partitions, None, None

            # Minimum logical entropy among those. Minimum filling length among those. Get ALL entries matching both
            min_le = min(x["le"] for x in fillings)
            min_filling_len = min(len(x["filling"]) for x in fillings if x["le"] == min_le)
            min_le_filling = [x for x in fillings if x["le"] == min_le and len(x["filling"]) == min_filling_len]

            # Count the amounts of criterias that where able to isolate the element in the set
            criterias_filled = {}
            for element in min_le_filling:
                # Check if the elements fill considering the
                criterias_filled[(element['i'], element['modality'])] = {"criterias_filled": 1,
                                                                         "modality": element["modality"]}
                for filling in fillings:
                    if element["criteria"] != filling["criteria"]:
                        for l in element["filling"]:
                            if l in filling["filling"]:
                                criterias_filled[(element["i"], element['modality'])]["criterias_filled"] += 1

            print(f"Minimum modality: {mod_chosen}. Min le filling: {min_le_filling}.")
            return mod_chosen, partitions, min_le_filling, final_answer

        if not mod_chosen:

            average_le = average_modality_le(partitions)
            filtered = {k: v for k, v in average_le.items() if v is not None}
            max_value = max(filtered.values())
            max_modalities = [k for k, v in filtered.items() if v == max_value]

            print("Max modalities: ", max_modalities)

            # The partition must bring the modality from which they were created.
            fillings = [
                {"modality": modality, "i": i, "filling": partition["splitting"]["filling"], "le": partition["le"],
                 "criteria": partition["criteria"]} for modality in partitions.keys() for i, partition in
                enumerate(partitions[modality]) if partition["le"] > 0]

            final_answer = None

            if not fillings:
                return None, partitions, None, None

            # Minimum logical entropy among those. Minimum filling length among those. Get ALL entries matching both
            min_le = min(x["le"] for x in fillings)
            min_filling_len = min(len(x["filling"]) for x in fillings if x["le"] == min_le)
            min_le_filling = [x for x in fillings if x["le"] == min_le and len(x["filling"]) == min_filling_len]

            # Count the amounts of criterias that where able to isolate the element in the set
            criterias_filled = {}
            for element in min_le_filling:
                # Check if the elements fill considering the
                criterias_filled[(element['i'], element['modality'])] = {"criterias_filled": 1,
                                                                         "modality": element["modality"]}
                for filling in fillings:
                    if element["criteria"] != filling["criteria"]:
                        for l in element["filling"]:
                            if l in filling["filling"]:
                                criterias_filled[(element["i"], element['modality'])]["criterias_filled"] += 1

            if len(max_modalities) > 1:
                print("the LLMs decides")
                modality = self.decide_answer_modality(model, question_text, max_modalities)
            else:
                # Find the index that has the maximum number of criterias filled
                print("The LLM does not decide")
                modality = criterias_filled[max(criterias_filled,
                                                key=lambda k: criterias_filled[k]['criterias_filled'])]["modality"]

            print(f"Minimum modality: {modality}. Min le filling: {min_le_filling}.")
            return modality, partitions, min_le_filling, final_answer

    def iscomparison(self, model, question_text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a question classifier that determines whether a question requires a comparison to be answered.

        Task:
        Return TRUE if answering the question requires evaluating two or more entities relative to each other
        (e.g., differences, similarities, ranking, better/worse, larger/smaller).

        Return FALSE if the question only asks for:
        - a fact about one entity
        - a definition or explanation
        - a location, time, or property of a single item
        - a yes/no question about a single item

        Additional task:
        If the question requires a comparison, determine the number of entities that must be compared.

        Important rules:
        - Do not rely only on keywords like "which", "better", or "difference".
        - Focus on whether the answer must compare multiple entities.
        - If the question can be answered without comparing entities, return FALSE.
        - If uncertain, return FALSE.

        Output rules:
        - is_comparison = TRUE if the question requires comparing entities.
        - is_comparison = FALSE otherwise.
        - num_elements = the number of entities that must be compared.
        - If the question is not a comparison, num_elements = 0.
        - confidence = confidence score between 0 and 1.
        """
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
        {question_text}
        """
                    }
                ],
                text_format=IsComparison,
            )

            found = response.output_parsed
            return found.is_comparison, found.num_elements, found.confidence

    def isgraphical(self, model, question_text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a classifier that determines whether answering a question requires analyzing a graphical element.

Task:
Return TRUE if the question requires interpreting or identifying something based on a visual/graphical element.

Return FALSE if the question can be answered using only text or general knowledge.

Guidelines:
- TRUE if the question refers to:
  - images, pictures, diagrams, maps, charts, graphs
  - visual features (e.g., colors, shapes, layout, positions, race of a person through skin color)
  - descriptions of covers, logos, symbols, or scenes
  - phrases like "shown in the image", "in the picture", "based on the chart"

- FALSE if:
  - the question is purely factual, textual, or conceptual

Important:
- If uncertain, return FALSE.

Output:
- is_graphical: TRUE or FALSE
- confidence: confidence score between 0 and 1
"""
                    },
                    {
                        "role": "user",
                        "content": f"""QUESTION:
{question_text}
"""
                    }
                ],
                text_format=IsGraphical,
            )

            found = response.output_parsed
            return found.is_graphical, found.confidence

    def find_final_answer(self, model, question, question_text, answer_class_specific, answer_class_general,
                          answers_dir, modality, partitions, min_le_filling, iscomparison, num_elements):

        print(f"The question text is: {question_text}")

        if not any(d.get("filling") for d in min_le_filling):
            print("Fillings are empty")
            yesnoquestion, confidence = self.yesnoquestion(model, question_text)
            print(yesnoquestion)
            if yesnoquestion:
                final_answer = "no"
                answer_dict = {"final_answer": final_answer}
                json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

                return final_answer

        final_answer = None
        if modality == "image":
            semantic_checks = []

            for filling in min_le_filling:
                if filling['modality'] == modality:
                    for image in filling['filling']:
                        semantic_checks.append(self.check_answer_caption_image(model, answer_class_specific.lower(),
                                                                               answer_class_general.lower(),
                                                                               question_text, image['title'],
                                                                               image['path']))

            print("Semantic checks: ", semantic_checks)
            sorted_results = sorted(semantic_checks, key=lambda x: (x[0], x[3]), reverse=True)
            print("Sorted results: ", sorted_results)
            # final_answer = [sorted_result[1] for sorted_result in sorted_results if sorted_result[3]]

            high_conf = [r[1] for r in sorted_results if r[3] > 0.90]
            if high_conf:
                final_answer = high_conf
            else:
                final_answer = [sorted_results[0][1]] if sorted_results else []

        elif modality == "text":
            print(f"Let's give answer from text: {question_text}")

            semantic_checks = []
            for filling in min_le_filling:
                if 'old_criteria' in filling['criteria']:
                    contextual_information = [filling['criteria']['old_criteria'],
                                              filling['criteria']['conditional_criteria']]
                else:
                    contextual_information = [filling['criteria']]
                # print(f"This is the criterias: {filling['criteria']}")
                for paragraph in filling['filling']:
                    semantic_checks.append(
                        self.check_answer_in_paragraph(model,
                                                       answer_class_general.lower(),
                                                       question_text,
                                                       paragraph,
                                                       contextual_information))

            print("Semantic checks: ", semantic_checks)
            sorted_results = sorted(semantic_checks, key=lambda x: (x[0], x[2]), reverse=True)
            print("Sorted results: ", sorted_results)

            high_conf = [r[1] for r in sorted_results if r[2] > 0.90]
            if high_conf:
                final_answer = high_conf
            else:
                final_answer = [sorted_results[0][1]] if sorted_results else []
            # final_answer = sorted_results[0][1]

        elif modality == "table":
            table_description = partitions["table"][0]["table_understanding"]
            semantic_checks = []

            print(partitions["table"][0]["criteria"])

            if "conditional_criteria" in partitions["table"][0]["criteria"]:
                conditional_criteria = partitions["table"][0]["criteria"]["conditional_criteria"]
            else:
                conditional_criteria = None

            print(f"Conditional criteria for answer: {conditional_criteria}")

            yesnoquestion, confidence = self.yesnoquestion(model, question_text)
            print(yesnoquestion)
            if yesnoquestion:
                final_answer = "yes"
                answer_dict = {"final_answer": final_answer}
                json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

                return final_answer

            elif iscomparison:
                print("Siamo qui")
                if len(min_le_filling) >= num_elements:
                    print("True")
                    semantic_checks.append(self.check_answer_in_row(model,
                                                                    answer_class_specific.lower(),
                                                                    min_le_filling,
                                                                    question_text,
                                                                    table_description,
                                                                    conditional_criteria))
                else:
                    semantic_checks.append((False, "NONE", 1.00))
            else:
                for filling in min_le_filling:
                    for row in filling['filling']:
                        print("Eccoci qui")
                        print(row)
                        print(question_text)
                        print(table_description)
                        semantic_checks.append(self.check_answer_in_row(model,
                                                                        answer_class_specific.lower(),
                                                                        row,
                                                                        question_text,
                                                                        table_description,
                                                                        conditional_criteria))

            sorted_results = sorted(semantic_checks, key=lambda x: (x[0], x[2]), reverse=True)
            print(f"Sorted results: {sorted_results}")

            high_conf = [r[1] for r in sorted_results if r[2] > 0.90]
            print(high_conf)
            if high_conf:
                final_answer = high_conf
            else:
                final_answer = [sorted_results[0][1]] if sorted_results else []

        answer_dict = {"final_answer": final_answer}
        json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

        if final_answer[0] == 'NONE':
            return None
        else:
            return final_answer

    def create_multi_hop_partitions(self, model, question, question_text, answer_class):
        """With this method we generate multi hop partitions within the same modality. In this case we check what are
        the criterias, within the same modality, that allows to diminish the logical entropy"""
        unimodal_partitions_path = os.path.join(
            f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

        multihop_partitions_path = os.path.join(
            f"../results/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")
        if os.path.exists(multihop_partitions_path):
            multihop_partitions = json.load(open(multihop_partitions_path, "rb"))
        else:
            multihop_partitions = {"image": [], "text": [], "table": []}

        # You have to check for which one has not been done
        non_empty_modalities = {}
        for main_key, items in multihop_partitions.items():
            non_empty_modalities[main_key] = any(
                item.get("conditional_modalities") == main_key
                for item in items
            )

        print("Non empty modalities: ", non_empty_modalities)

        # We first find the criteria with the lowest logical entropy for each modality
        results_per_modality = {}
        for modality, items in unimodal_partitions.items():
            if not non_empty_modalities[modality]:
                # Filter out items with le == 0
                nonzero_items = [item for item in items if item["le"] != 0]

                if not nonzero_items:  # If all items are 0 or no items exist
                    results_per_modality[modality] = None
                    continue

                best_item = min(nonzero_items, key=lambda x: x["le"])

                print("best item is: ", best_item)

                # Apply your restricting_criteria function
                if modality == "text":
                    restricting_criterias = self.extract_restricting_criteria_text(
                        model,
                        question_text,
                        best_item["splitting"]["filling"]['title'],  # the filling items
                        best_item["splitting"]["filling"]['text'],
                        best_item["criteria"]  # the criteria for this partition
                    )

                elif modality == "image":
                    restricting_criterias = self.extract_restricting_criteria_image(
                        model,
                        question_text,
                        best_item["splitting"]["filling"]['title'],  # the filling items
                        best_item["splitting"]["filling"]['path'],
                        best_item["criteria"]  # the criteria for this partition
                    )

                elif modality == "table":
                    restricting_criterias = self.extract_restricting_criteria_table_row(
                        model,
                        question_text,
                        best_item["splitting"]["filling"],
                        None,
                        None,
                        None,
                        None
                    )

                # Store results
                results_per_modality[modality] = {
                    "best_item": best_item["splitting"]["filling"],
                    "criteria": best_item["criteria"],
                    "restricting_criterias": restricting_criterias
                }

        for modality in results_per_modality.keys():
            print(f"Results per {modality.upper()}: {results_per_modality[modality]}")

        # For each modality, having the criteria that we have extracted from the element with the lowest le
        # we want to check whether there is an intersection with any other element of the same modality.
        # Of course, they must be different elements, otherwise it does not make any sense.
        for modality, items in unimodal_partitions.items():
            if not non_empty_modalities[modality]:
                print(f"Modality is: {modality}")
                if results_per_modality[modality]:
                    best_item = results_per_modality[modality]["best_item"]
                    print(f"Best item: {best_item}")
                    restricting_criterias = results_per_modality[modality]["restricting_criterias"]
                    for partition in unimodal_partitions[modality]:
                        if not partition["splitting"]["filling"]:
                            continue

                        correct_elements = []
                        # This contains the images that fill the previous criteria
                        modality_set = partition["splitting"]["filling"]
                        print(f"Modality set: {modality_set}")

                        l1 = {make_hashable(d) for d in best_item}
                        l2 = {make_hashable(d) for d in modality_set}

                        if l1 != l2:
                            print("They are different")
                            print(l1)
                            print(l2)

                            for element in modality_set:
                                if modality == "image":
                                    answer = self.analyse_image_restricting_criteria(model,
                                                                                     restricting_criterias,
                                                                                     element["title"],
                                                                                     element["path"])
                                elif modality == "text":
                                    answer = self.analyse_text_restricting_criteria(model,
                                                                                    restricting_criterias,
                                                                                    element["title"],
                                                                                    element["text"])
                                elif modality == "table":
                                    association_json = json.load(open(os.path.join(
                                        "/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/association",
                                        question), "rb"))
                                    table = association_json["table_set"][0].copy()

                                    json_table = json.load(open(
                                        os.path.join(
                                            "/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/tables",
                                            table["json"]), "rb"))
                                    answer = self.analyse_table_row_restricting_criteria(model, restricting_criterias,
                                                                                         element,
                                                                                         json_table["title"],
                                                                                         json_table["table"][
                                                                                             "table_name"],
                                                                                         partition[
                                                                                             "table_understanding"])

                                if answer.lower() == "yes":
                                    correct_elements.append(element)

                            # When calculating the partition, to the modality set we also want to add the elements that did not
                            # respect the criteria as the beginning so, the not filling ones
                            all_elements = partition["splitting"]["filling"] + partition["splitting"]["not_filling"]

                            new_partition = self.create_partition(all_elements, correct_elements,
                                                                  restricting_criterias[0])
                            conditional_criteria = new_partition["criteria"]
                            if modality == "table":
                                new_partition["table_understanding"] = partition["table_understanding"]

                            new_partition["criteria"] = {"old_criteria": partition["criteria"],
                                                         "conditional_criteria": conditional_criteria}
                            new_partition["conditional_modalities"] = modality

                            multihop_partitions[modality].append(new_partition)

                        else:
                            print("Same elements we do not create a double partition.")

        json.dump(multihop_partitions, open(multihop_partitions_path, "w"), indent=4)
        return multihop_partitions

    def create_multi(self, model, question, question_text, multimodal_partitions, unimodal_partition,
                     conditional_modality, conditioned_modality, restricting_criterias):

        correct_elements = []
        modality_set = unimodal_partition["splitting"]["filling"]

        for element in modality_set:
            if conditioned_modality == "image":
                answer = self.analyse_image_restricting_criteria(model,
                                                                 restricting_criterias,
                                                                 element["title"],
                                                                 element["path"])
            elif conditioned_modality == "text":
                print("Ci siamo con il testo")
                print(f"Il testo è: {element}")
                answer = self.analyse_text_bridge_element(model,
                                                          question_text,
                                                          element["title"],
                                                          restricting_criterias,
                                                          element["text"])

                print(f"The answer is: {answer}")
            elif conditioned_modality == "table":
                print("Ci siamo con la tabella")
                print(f"The row is: {element}")

                association_json = json.load(open(
                    os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/association",
                                 question), "rb"))
                table = association_json["table_set"][0].copy()

                json_table = json.load(open(
                    os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/tables",
                                 table["json"]), "rb"))
                answer = self.analyse_table_row_restricting_criteria(model,
                                                                     restricting_criterias,
                                                                     element,
                                                                     json_table["title"],
                                                                     json_table["table"]["table_name"],
                                                                     unimodal_partition["table_understanding"])

                print(f"The answer is: {answer}")

            if answer.lower() == "yes":
                correct_elements.append(element)

        # When calculating the partition, to the modality set we also want to add the elements that did not respect
        # the criteria as the beginning so, the not filling ones
        all_elements = unimodal_partition["splitting"]["filling"] + unimodal_partition["splitting"]["not_filling"]

        new_partition = self.create_partition(all_elements, correct_elements, restricting_criterias)
        conditional_criteria = new_partition["criteria"]
        if conditioned_modality == "table":
            new_partition["table_understanding"] = unimodal_partition["table_understanding"]

        new_partition["criteria"] = {"old_criteria": unimodal_partition["criteria"],
                                     "conditional_criteria": conditional_criteria}
        new_partition["conditional_modalities"] = conditional_modality

        multimodal_partitions[conditioned_modality].append(new_partition)

    def return_restricting_criteria(self, model, question_text, unimodal_partitions, conditional_modality):
        print("Let's first extract the restricting criteria")
        print(f"The conditional modality is: {conditional_modality}")

        # Here we take all the partitions with logical entropy greater than 0
        valid_partitions_conditional_modality = [p for p in unimodal_partitions[conditional_modality] if p["le"] > 0 or
                                                 p['splitting']['filling']]

        if not valid_partitions_conditional_modality:
            return []

        # Here we have to create a dictionary that maps the elements with the criterias respected

        fillings_conditional_modality = [item for p in valid_partitions_conditional_modality for item in
                                         p.get("splitting", {}).get("filling", [])]
        unique_filling = list(
            {json.dumps(item, sort_keys=True): item for item in fillings_conditional_modality}.values())

        unique_filling = {i: filling for i, filling in enumerate(unique_filling)}

        element_mapping = {}
        for i, filling in unique_filling.items():
            element_mapping[i] = {'el_filling': filling, 'criterias': []}
            for partition in valid_partitions_conditional_modality:
                if filling in partition["splitting"]["filling"]:
                    element_mapping[i]['criterias'].append(partition['criteria'])

        for id in element_mapping.keys():
            print("El filling: ", element_mapping[id]['el_filling'])
            print("El criterias: ", element_mapping[id]['criterias'])

        """
        filling_el_criterias = {}
        for partition in valid_partitions_conditional_modality:
            for el in partition["splitting"]["filling"]

        criterias_conditional_modality = [p["criteria"] for p in valid_partitions_conditional_modality]

        fillings_conditional_modality = [item for p in valid_partitions_conditional_modality for item in
                                         p.get("splitting", {}).get("filling", [])]
        unique_filling = list(
            {json.dumps(item, sort_keys=True): item for item in fillings_conditional_modality}.values())
        fillings_conditional_modality = unique_filling
        """

        restricting_criterias = []

        if conditional_modality == "text":
            print("Vai col testo motherfucker")
            # for filling in fillings_conditional_modality:
            for id in element_mapping.keys():
                restricting_criterias.append(self.extract_restricting_criteria_text(model, question_text,
                                                                                    element_mapping[id]['el_filling'][
                                                                                        'title'],
                                                                                    element_mapping[id]['el_filling'][
                                                                                        'text'],
                                                                                    element_mapping[id]['criterias']))
        elif conditional_modality == "image":
            print("Vai col image motherfucker")
            # for filling in fillings_conditional_modality:
            for id in element_mapping.keys():
                restricting_criterias.append(self.extract_restricting_criteria_image(model, question_text,
                                                                                     element_mapping[id]['el_filling'][
                                                                                         'title'],
                                                                                     element_mapping[id]['el_filling'][
                                                                                         'path']))
        elif conditional_modality == "table":
            document_title = valid_partitions_conditional_modality[0]["table_title"]
            table_name = valid_partitions_conditional_modality[0]["table_name"]
            table_description = valid_partitions_conditional_modality[0]["table_understanding"]

            print("Vai col table motherfucker")
            # for filling in fillings_conditional_modality:
            for id in element_mapping.keys():
                print(f"id: {id}")
                restricting_criterias.append(self.extract_restricting_criteria_table_row(model,
                                                                                         question_text,
                                                                                         document_title,
                                                                                         table_name,
                                                                                         table_description,
                                                                                         element_mapping[id][
                                                                                             'el_filling'],
                                                                                         element_mapping[id][
                                                                                             'criterias']))

        """
        for filling in fillings_conditional_modality:
            restricting_criterias.append(self.extract_restricting_criteria(model, question_text, filling,
                                                                           criterias_conditional_modality, el_type,
                                                                           final_answer_class))
        """

        return restricting_criterias

    def create_multimodal_partitions(self, model, question, question_text, final_answer_class):
        print("Salve buonasera come state?")
        unimodal_partitions_path = os.path.join(
            f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

        multimodal_partitions_path = os.path.join(
            f"../results/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")
        if os.path.exists(multimodal_partitions_path):
            multimodal_partitions = json.load(open(multimodal_partitions_path, "rb"))
        else:
            multimodal_partitions = {"image": [], "text": [], "table": []}

        # Check which modalities are non empty in the unimodal as between those we can do the conditioning
        non_empty_unimodal = [modality for modality in unimodal_partitions.keys() if unimodal_partitions[modality]]
        non_empty_unimodal_pairs = [(modality1, modality2) for modality1 in non_empty_unimodal for modality2 in
                                    non_empty_unimodal if modality1 != modality2]

        print(f"The pairs are: {non_empty_unimodal_pairs}")

        existing_combinations = set()
        for main_modality, items in multimodal_partitions.items():
            for el in items:
                cond = el.get("conditional_modalities")
                if cond is not None:
                    existing_combinations.add((main_modality, cond))

        print(f"Existing combinations: {existing_combinations}")

        for pair in non_empty_unimodal_pairs:
            print(f"Vediamo cosa c'è: {pair}")
            if pair not in existing_combinations:
                print(f"Non ci siamo proprio: {pair}")

                conditional_modality = pair[0]
                conditioned_modality = pair[1]

                print(f"Conditional modality: {conditional_modality}. Conditioned modality: {conditioned_modality}")

                restricting_criterias = self.return_restricting_criteria(model, question_text, unimodal_partitions,
                                                                         conditional_modality)

                restricting_criterias = [el for el in restricting_criterias if el is not None and el.lower() != 'none']
                print(f"The restricting criterias are: {restricting_criterias}")

                for unimodal_partition in unimodal_partitions[conditioned_modality]:
                    if not unimodal_partition["splitting"]["filling"]:
                        continue
                    else:
                        if restricting_criterias:
                            self.create_multi(model, question, question_text, multimodal_partitions, unimodal_partition,
                                              conditional_modality, conditioned_modality, restricting_criterias)
                        else:
                            pass

        return multimodal_partitions

    def relative_entropy_change(self, H1, H2):
        """
        Calculate the signed relative change from H1 to H2.

        Positive → increase (worse)
        Negative → decrease (better)
        Zero → no change
        """

        if H2 is None:
            return H1  # treat missing value as no change
        if H1 is None or H1 == 0:
            return -H2
        # if H1 == 0:
        #    return 0.0  # avoid division by zero

        return (H1 - H2) / H1

    def choose_unimodal_multimodal(self, unimodal_partitions_path, multimodal_partitions_path, modality_answers_given):
        """This method compares the entropy of unimodal partitions with the entropy of multimodal partitions and decides
        whether the final answer will be given looking at unimodal partitions of multimodal partitions."""

        print(f"modality answers given are: {modality_answers_given}")

        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))
        multimodal_partitions = json.load(open(multimodal_partitions_path, "rb"))

        # Here we gather the minimum element and le in unimodal
        average_modality_le_unimodal = average_modality_le(unimodal_partitions)
        print(f"Average modality le unimodal: {average_modality_le_unimodal}")

        if any(average_modality_le_unimodal[key] for key in average_modality_le_unimodal.keys()):
            min_modality_unimodal = min((k for k, v in average_modality_le_unimodal.items() if v is not None),
                                        key=lambda k: average_modality_le_unimodal[k])
            min_value_unimodal = average_modality_le_unimodal[min_modality_unimodal]
        else:
            min_modality_unimodal = None
            min_value_unimodal = None

        print(f"Min modality unimodal: {min_modality_unimodal}. Min value unimodal: {min_value_unimodal}")

        # Here we gather the minimum element and le in multimodal
        average_modality_le_multimodal = average_modality_le(multimodal_partitions)
        print(f"Average modality le multimodal: {average_modality_le_multimodal}")

        if any(average_modality_le_multimodal[key] for key in average_modality_le_multimodal.keys()):
            min_modality_multimodal = min((k for k, v in average_modality_le_multimodal.items() if v is not None),
                                          key=lambda k: average_modality_le_multimodal[k])
            min_value_multimodal = average_modality_le_multimodal[min_modality_multimodal]
        else:
            min_modality_multimodal = None
            min_value_multimodal = None

        print(f"Min modality multimodal: {min_modality_multimodal}. Min value multimodal: {min_value_multimodal}")

        if min_modality_multimodal is None:
            return "unimodal_partitions", None

        if any(average_modality_le_unimodal[k] is None and average_modality_le_multimodal.get(k) is not None for k in average_modality_le_unimodal):
            return "multimodal_partitions", [modality for modality in average_modality_le_multimodal if average_modality_le_unimodal.get(modality) is None and average_modality_le_multimodal.get(modality) is not None][0]

        # We check whether any key that in unimodal was not None than became None. This is to check the importance of
        # the order
        if any(average_modality_le_unimodal[k] is not None and average_modality_le_multimodal.get(k) is None
               for k in average_modality_le_unimodal):

            print("We are in this case")
            relative_changes = {}
            for mod in average_modality_le_unimodal.keys():
                if average_modality_le_unimodal[mod] and average_modality_le_multimodal[mod]:
                    relative_change = self.relative_entropy_change(average_modality_le_unimodal[mod],
                                                                   average_modality_le_multimodal[mod])

                    if relative_change > 0:
                        relative_changes[mod] = relative_change

            sorted_modalities = sorted(
                relative_changes.items(),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"Sorted modalities are: {sorted_modalities}")

            # Pick the best valid modality
            mod_max_relative_change = None
            for mod, change in sorted_modalities:
                print(f"Mod: {mod}")
                if mod in modality_answers_given:
                    mod_max_relative_change = mod
                    break

            print(f"The mod max is: {mod_max_relative_change}")
            if mod_max_relative_change is None:
                return "unimodal_partitions", None

            print(f"Modality with max relative change and relative change: {mod_max_relative_change}, "
                  f"{relative_changes[mod_max_relative_change]}")

            return "multimodal_partitions", mod_max_relative_change

        else:
            relative_changes = {}
            for mod in average_modality_le_unimodal.keys():
                if average_modality_le_unimodal[mod] and average_modality_le_multimodal[mod]:
                    relative_change = self.relative_entropy_change(average_modality_le_unimodal[mod],
                                                                   average_modality_le_multimodal[mod])

                    if relative_change > 0:
                        relative_changes[mod] = relative_change

            if relative_changes:
                sorted_modalities = sorted(
                    relative_changes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Pick the best valid modality
                mod_max_relative_change = None
                for mod, change in sorted_modalities:
                    if mod in modality_answers_given:
                        mod_max_relative_change = mod
                        break

                if mod_max_relative_change is None:
                    return "unimodal_partitions", None

                print(f"Modality with max relative change and relative change: {mod_max_relative_change}, "
                      f"{relative_changes[mod_max_relative_change]}")

                return "multimodal_partitions", mod_max_relative_change

            else:
                return "unimodal_partitions", None

        if min_modality_unimodal == min_modality_multimodal:
            if average_modality_le_unimodal[min_modality_unimodal] < average_modality_le_multimodal[
                min_modality_multimodal]:
                return "unimodal_partitions", None
            else:
                return "multimodal_partitions", None

        # Here we measure the average change of the logical entropy for the minimum modality in unimodal and for the
        # minimum modality in multimodal
        change_minimum_unimodal = self.relative_entropy_change(average_modality_le_unimodal[min_modality_unimodal],
                                                               average_modality_le_multimodal[min_modality_unimodal])

        change_minimum_multimodal = self.relative_entropy_change(average_modality_le_unimodal[min_modality_multimodal],
                                                                 average_modality_le_multimodal[
                                                                     min_modality_multimodal])

        print(change_minimum_unimodal, change_minimum_multimodal)

        # Case in which both of them start greater than 0 meaning that the filling was not empty
        if average_modality_le_unimodal[min_modality_unimodal] and average_modality_le_unimodal[
            min_modality_multimodal]:
            if (average_modality_le_unimodal[min_modality_unimodal] > 0 and
                    average_modality_le_unimodal[min_modality_multimodal] > 0):
                print("When unimodal they were both bigger than 0")
                # Case in which they both conclude greater than 0, meaning that the filling in the multimodal is not
                # empty for both of them (which means we did a combination within the same modality)

                # In case they are both not None
                if (average_modality_le_multimodal[min_modality_unimodal] and
                        average_modality_le_multimodal[min_modality_multimodal]):
                    print("ci siamo")
                    if (average_modality_le_multimodal[min_modality_unimodal] > 0 and
                            average_modality_le_multimodal[min_modality_multimodal] > 0):
                        print("E siamo anche qui")
                        if change_minimum_unimodal < change_minimum_multimodal:
                            return "unimodal_partitions", None
                        elif change_minimum_multimodal < change_minimum_unimodal:
                            return "multimodal_partitions", None
                    elif (average_modality_le_multimodal[min_modality_unimodal] == 0 and
                          average_modality_le_multimodal[min_modality_multimodal] > 0):
                        return "multimodal_partitions", None
                    elif average_modality_le_multimodal[min_modality_unimodal] > 0 and average_modality_le_multimodal[
                        min_modality_multimodal] == 0:
                        return "unimodal_partitions", None
                    else:
                        return "mannacc a miserj", None

                elif average_modality_le_multimodal[min_modality_unimodal] is None or average_modality_le_multimodal[
                    min_modality_multimodal] is None:
                    print("Uno dei due è Non")
                    if min_value_unimodal < min_value_multimodal:
                        return "unimodal_partitions", None
                    elif min_value_multimodal < min_value_unimodal:
                        return "multimodal_partitions", None
            else:
                print("What are we taking about. It is impossible")

        elif not average_modality_le_unimodal[min_modality_unimodal] or not average_modality_le_unimodal[
            min_modality_multimodal]:
            if min_value_unimodal < min_value_multimodal:
                return "unimodal_partitions", None
            elif min_value_multimodal < min_value_unimodal:
                return "multimodal_partitions", None
            else:
                return "unimodal_partitions", None

    def find_final_answer_boolean_table(self, model, question, question_text, partitions_path,
                                        election_modality, answers_dir):

        print(f"Chi te muort: {question_text}")
        modality, partitions, min_le_filling, final_answer = self.extract_partitions(model,
                                                                                     question,
                                                                                     "table",
                                                                                     question_text,
                                                                                     partitions_path)

        final_answer = None
        for partition in partitions[election_modality]:
            print(f'Criteria: {partition["criteria"]}')
            if partition["criteria"] == question_text:
                final_answer = "yes" if partition["splitting"]["filling"] else "no"
                break

        answer_dict = {"final_answer": final_answer}
        print(answer_dict)
        json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

        return final_answer

    def find_final_answer_boolean_two_steps_table(self, model, question, question_text, partitions_path,
                                                  election_modality, answers_dir, answer_class_specific, num_elements):

        modality, partitions, min_le_filling, final_answer = self.extract_partitions(model, question,
                                                                                     "table",
                                                                                     question_text,
                                                                                     partitions_path)

        print("Let's see what are we talking about")

        fillings_to_consider = []
        for partition in partitions[election_modality]:
            print(f'Criteria: {partition["criteria"]}')
            filling = partition["splitting"]["filling"]
            if len(filling) == num_elements:
                fillings_to_consider.append(filling)

        print(fillings_to_consider)

        table_description = partitions["table"][0]["table_understanding"]
        semantic_checks = []

        for filling in fillings_to_consider:
            semantic_checks.append(self.check_answer_in_row(model,
                                                            answer_class_specific.lower(),
                                                            filling,
                                                            question_text,
                                                            table_description,
                                                            None))

        sorted_results = sorted(semantic_checks, key=lambda x: (x[0], x[2]), reverse=True)
        print(f"Sorted results: {sorted_results}")

        high_conf = [r[1] for r in sorted_results if r[2] > 0.90]
        if high_conf:
            final_answer = high_conf
        else:
            final_answer = [sorted_results[0][1]] if sorted_results else []

        answer_dict = {"final_answer": final_answer}
        json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

        if final_answer[0] == 'NONE':
            return None
        else:
            return final_answer

    def find_final_answer_comparison_three_modalities(self, model, question, question_text, answers_dir,
                                                      answer_class_specific, num_elements):

        unimodal_partitions_path = os.path.join(
            f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))
        multimodal_partitions = self.create_multimodal_partitions(model, question, question_text,
                                                                  answer_class_specific.lower())

        table_description = unimodal_partitions["table"][0]["table_understanding"]

        # Now we have to fuse the rows isolated from the text and the rows isolated from the image
        fillings_to_consider = []
        criterias = {"initial_criteria": [], "conditional_criteria": []}
        for mod in multimodal_partitions.keys():
            if mod == "table":
                for partition in multimodal_partitions[mod]:
                    if partition["conditional_modalities"] == "text" or partition["conditional_modalities"] == "image":
                        fillings_to_consider.extend(partition["splitting"]["filling"])
                        criterias["initial_criteria"].extend(partition["criteria"]["old_criteria"])
                        criterias["conditional_criteria"].extend(partition["criteria"]["conditional_criteria"])

        criterias["initial_criteria"] = list(set(criterias["initial_criteria"]))
        criterias["conditional_criteria"] = list(set(criterias["conditional_criteria"]))

        seen = set()
        for row in fillings_to_consider:
            # Convert row into a canonical string
            row_key = json.dumps(row, sort_keys=True)
            if row_key not in seen:
               seen.add(row_key)

        # This should basically contain the rows of the table that have been isolated from the image and from the text
        fillings_to_consider = [json.loads(el) for el in seen]
        print(f"Fillings to consider are: {fillings_to_consider}")

        semantic_checks = self.check_answer_in_row(model, answer_class_specific.lower(), fillings_to_consider,
                                                   question_text, table_description, criterias)

        print(semantic_checks)
        semantic_checks = [semantic_checks]

        sorted_results = sorted(semantic_checks, key=lambda x: (x[0], x[2]), reverse=True)
        print(f"Sorted results: {sorted_results}")

        high_conf = [r[1] for r in sorted_results if r[2] > 0.90]
        if high_conf:
            final_answer = high_conf
        else:
            final_answer = [sorted_results[0][1]] if sorted_results else []

        answer_dict = {"final_answer": final_answer}
        json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

        if final_answer[0] == 'NONE':
            return None
        else:
            return final_answer

    def return_final_answer(self, model, question, question_files, question_text, rewritten_question_text,
                            priority_modalities, criterias, remaining_modalities, answer_class_specific,
                            answer_class_general, table_dir, final_dataset_images, answers_dir):

        final_answer = None

        unimodal_partitions_path = os.path.join(
            f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        multimodal_partitions_path = os.path.join(
            f"../results/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")

        # First we create the partitions, both in the unimodal case and also in the multi-hop unimodal case
        for modality in priority_modalities:
            print(f"CREATING UNIMODAL PARTITIONS for {modality.upper()}")
            print("************************************************************************")
            self.create_unimodal_partitions(model, question, question_files, table_dir, modality, criterias)
            print("************************************************************************")
            print(f"END CREATION")
            print("************************************************************************\n")

        # In case the predicted modality is table and the answer_class is boolean then we do not start the entire
        # analysis
        if len(priority_modalities) == 1 and priority_modalities[0] == "table":
            if answer_class_general == "boolean":
                print("WE do not even create the multi-hop. We just go with a single table")
                self.find_final_answer_boolean_table(model, question, rewritten_question_text,
                                                     "unimodal_partitions", priority_modalities[0],
                                                     answers_dir)

                return

            else:
                is_comparison, num_elements, confidence = self.iscomparison(model, question_text)
                if is_comparison:
                    print(f"num elements is {num_elements}")
                    print("comparison mood")
                    self.find_final_answer_boolean_two_steps_table(model, question, question_text,
                                                                   "unimodal_partitions", priority_modalities[0],
                                                                   answers_dir, answer_class_specific, num_elements)

                    return

        print(f"Question text: {question_text}")
        is_comparison, num_elements, confidence = self.iscomparison(model, question_text)

        print(f"Do we need a comparison? {is_comparison}, num_elements: {num_elements}")

        # We check whether a comparison is needed
        if is_comparison:
            # We check how many modalities have been selected
            if len(priority_modalities) == 2:
                # We check if analysing images is needed
                is_graphical, confidence = self.isgraphical(model, question_text)
                # If is_graphical is true and we have not yet inserted images in the modalities then we need to do
                # it, and we can proceed.
                print(f"Do we need a graphical? {is_graphical}")
                if is_graphical and 'image' not in priority_modalities:
                    self.create_unimodal_partitions(model, question, question_files, table_dir, "image", criterias)
                    self.find_final_answer_comparison_three_modalities(model, question, rewritten_question_text,
                                                                       answers_dir, answer_class_specific, num_elements)

        multimodal = len(priority_modalities) > 1
        modality_answers_given_multi = ["text", "table", "image"]
        modality_answers_given_uni = ["text", "table", "image"]

        while final_answer is None:

            if not multimodal:
                # Create multi-hop partitions only if the question is not directly unimodal
                print(f"CREATING UNIMODAL MULTI-HOP PARTITIONS")
                print("************************************************************************")
                # self.create_multi_hop_partitions(model, question, question_text, answer_class_specific)
                print("************************************************************************")
                print(f"END CREATION")
                print("************************************************************************\n")

                print("***************************************************************")
                print("WE START A UNIMODAL ANALYSIS")
                print("***************************************************************")
                if os.path.exists(multimodal_partitions_path):
                    partitions_path, mod_chosen = self.choose_unimodal_multimodal(unimodal_partitions_path,
                                                                                  multimodal_partitions_path,
                                                                                  modality_answers_given_uni)
                else:
                    partitions_path, mod_chosen = "unimodal_partitions", None

                print(f"The partitions path is: {partitions_path}")
                modality, partitions, min_le_filling, final_answer = self.extract_partitions(model,
                                                                                             question,
                                                                                             mod_chosen,
                                                                                             rewritten_question_text,
                                                                                             partitions_path)

                print("We are still in the moment of not being multimodal")
                print(f"Modality: {modality}")
                print(f"Partitions: {partitions}")
                print(f"Min le filling: ", min_le_filling)

                if final_answer is None:
                    print("Let's calculate the final answer")
                    final_answer = self.find_final_answer(model,
                                                          question,
                                                          rewritten_question_text,
                                                          answer_class_specific,
                                                          answer_class_general,
                                                          answers_dir,
                                                          modality,
                                                          partitions,
                                                          min_le_filling,
                                                          is_comparison,
                                                          num_elements)

                if final_answer is None:
                    print("We were not able to find an answer. We have to chose another modality and go multimodal")
                    # If there are still modalities to choose from we select one of them. For now, we select randomly.
                    if remaining_modalities:
                        decided_modality = self.decide_modality_reduced_data(question_text,
                                                                             remaining_modalities,
                                                                             question_files["image_set"],
                                                                             question_files["text_set"],
                                                                             question_files["table_set"][0],
                                                                             table_dir,
                                                                             final_dataset_images)

                        modalities_json = json.load(open(f"../results/modalities_predicted/{model}/{question}", "rb"))
                        max_step = max([int(step.split('_')[1]) for step in modalities_json.keys()])
                        modalities_json[f'step_{max_step + 1}'] = decided_modality

                        json.dump(modalities_json,
                                  open(os.path.join(f"../results/modalities_predicted/{model}/{question}"), "w"),
                                  indent=4)

                        remaining_modalities.remove(decided_modality)
                        self.create_unimodal_partitions(model, question, question_files, table_dir, decided_modality,
                                                        criterias)
                        # self.create_multi_hop_partitions(model, question, question_text)
                        multimodal = True

            elif multimodal:
                print("***************************************************************")
                print("WE START A MULTIMODAL ANALYSIS")
                print("***************************************************************")
                # Here we create the multimodal partitions
                multimodal_partitions_path = os.path.join(
                    f"../results/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")

                # if not os.path.exists(multimodal_partitions_path):
                print("************************************************************************")
                print(f"CREATING MULTIMODAL PARTITIONS")
                multimodal_partitions = self.create_multimodal_partitions(model, question, rewritten_question_text,
                                                                          answer_class_specific)
                json.dump(multimodal_partitions,
                          open(f"../results/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}",
                               "w"),
                          indent=4)
                print("************************************************************************\n")

                unimodal_partitions_path = os.path.join(
                    f"../results/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")

                # Here we choose whether to use unimodal or multimodal, depending from the value of the logical entropy
                print("***************************************************************")
                print("Unimodal or Multimodal?")
                print("***************************************************************")
                partitions_path, mod_chosen = self.choose_unimodal_multimodal(unimodal_partitions_path,
                                                                              multimodal_partitions_path,
                                                                              modality_answers_given_multi)
                print(f"Partitions path {partitions_path}")
                print("***************************************************************")
                print("Extracting the partitions")
                modality, partitions, min_le_filling, final_answer = self.extract_partitions(model,
                                                                                             question,
                                                                                             mod_chosen,
                                                                                             rewritten_question_text,
                                                                                             partitions_path)

                print(f"The mod chosen is: {modality}")

                modality_answers_given_multi.remove(modality)
                print("***************************************************************")

                if final_answer is None:
                    print("Let's give this multimodal final answer")
                    print("Min le filling: ", min_le_filling)
                    final_answer = self.find_final_answer(model,
                                                          question,
                                                          rewritten_question_text,
                                                          answer_class_specific,
                                                          answer_class_general,
                                                          answers_dir,
                                                          modality,
                                                          partitions,
                                                          min_le_filling,
                                                          is_comparison,
                                                          num_elements)

                    print("This is the final answer: ", final_answer)

                if final_answer is None:
                    print("We need to find another modality!")
                    if remaining_modalities:
                        if len(remaining_modalities) > 1:
                            decided_modality = self.decide_modality_reduced_data(question_text,
                                                                                 remaining_modalities,
                                                                                 question_files["image_set"],
                                                                                 question_files["text_set"],
                                                                                 question_files["table_set"][0],
                                                                                 table_dir,
                                                                                 final_dataset_images)
                        else:
                            decided_modality = remaining_modalities[0]

                        modalities_json = json.load(open(f"../results/modalities_predicted/{model}/{question}", "rb"))
                        max_step = max([int(step.split('_')[1]) for step in modalities_json.keys()])
                        modalities_json[f'step_{max_step + 1}'] = decided_modality

                        json.dump(modalities_json, open(f"../results/modalities_predicted/{model}/{question}", "w"),
                                  indent=4)

                        remaining_modalities.remove(decided_modality)
                        print("************************************************************************")
                        print(f"CREATING UNIMODAL PARTITIONS for {decided_modality.upper()}")
                        print("************************************************************************\n")
                        self.create_unimodal_partitions(model, question, question_files, table_dir, decided_modality,
                                                        criterias)

                        self.create_multimodal_partitions(model, question, question_text, answer_class_specific.lower())

                        if not remaining_modalities and is_comparison:
                            final_answer = self.find_final_answer_comparison_three_modalities(model, question,
                                                                                              rewritten_question_text,
                                                                               answers_dir, answer_class_specific,
                                                                               num_elements)

                        """
                        print("************************************************************************")
                        print(f"CREATING MULTI-HOP PARTITIONS for {decided_modality.upper()}")
                        print("************************************************************************\n")
                        self.create_multi_hop_partitions(model, question, question_text)
                        """

                    else:
                        return "NONE"

    def answer_question(self, model, question, question_data, question_files, table_dir, final_dataset_images,
                        answers_dir):
        """Method which calls the other methods to calculate the final answer."""

        # START MODALITY SELECTION
        all_modalities = ["image", "table", "text"]

        # Predict the modalities that must be used and create the file that contains them (if it does not exist)
        modalities_json = os.path.join(f"../results/modalities_predicted/{model}/{question}")

        if os.path.exists(modalities_json):
            print("************************************")
            print("Modality file already exists? YES")
            print("************************************\n")
            priority_modalities = json.load(open(f"../results/modalities_predicted/{model}/{question}", "rb"))
            priority_modalities = priority_modalities["step_1"]
        else:
            print("*************************************************")
            print("Modality file already exists? NO. We create it")
            print("*************************************************\n")
            priority_modalities = self.decide_modality_llm(question_data["question_text"],
                                                           question_files["image_set"],
                                                           question_files["text_set"],
                                                           question_files["table_set"][0],
                                                           table_dir, final_dataset_images).split('_')

            json.dump({"step_1": priority_modalities},
                      open(os.path.join(f"../results/modalities_predicted/{model}/{question}"), "w"), indent=4)

        remaining_modalities = [m for m in all_modalities if m not in priority_modalities]

        if len(priority_modalities) == 1:
            answer_class_specific, answer_class_general, rewritten_question_text, *criterias = self.read_criterias(
                question, False)
        else:
            answer_class_specific, answer_class_general, rewritten_question_text, *criterias = self.read_criterias(
                question, False)

        print("ANSWER CLASS - PRIORITY MODALITIES - CRITERIAS")
        print("*************************************************************************************")
        print(f"Answer class {{Specific: {answer_class_specific}, General:{answer_class_general}}}")
        print(f"The priority modalities are: {priority_modalities}")
        print(f"The remaining modalities are {remaining_modalities}")
        print("Criterias: ", criterias)
        print("*************************************************************************************\n")
        # END MODALITY SELECTION

        # ANSWER TO THE QUESTION
        self.return_final_answer(model, question, question_files, question_data["question_text"],
                                 rewritten_question_text,
                                 priority_modalities, criterias, remaining_modalities, answer_class_specific,
                                 answer_class_general, table_dir, final_dataset_images, answers_dir)
        # END ANSWER TO QUESTION


def answer_qa(model, agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images,
              answers_dir):
    os.makedirs(answers_dir, exist_ok=True)
    os.makedirs(f"../results/partitions/unimodal_partitions/{iteration}/{model}/", exist_ok=True)

    print("ciao")

    for i, q in enumerate(questions_list):
        question = q["index"]
        modalities = q["modality"]
        if len(modalities) == 1:
            unimodal_multimodal = "/unimodal/"
        else:
            unimodal_multimodal = "/multimodal/"

        # if question not in os.listdir(answers_dir + unimodal_multimodal):
        print("\nQUESTION JSON - QUESTION TEXT - TARGET MODALITIES")
        print("*************************************************************************************")
        print(f"Question number: {i}")
        print(f"Question json: {question}")
        print(f"Question text: {q['original_question']}")
        print(f"Target modalities: {modalities}")
        print("*************************************************************************************\n")
        question_data = get_question_data(questions_dir, question)
        question_files = get_question_files(association_dir, question)

        agent.answer_question(model, question, question_data, question_files, table_dir, final_dataset_images,
                              answers_dir + unimodal_multimodal)


def entropy_calculation_main(model, agent, questions_list, questions_dir, association_dir, table_dir,
                             final_dataset_images, answers_dir):
    # create_association_qa(agent, questions_dir, association_dir, image_dir, text_dir, table_dir)

    answer_qa(model, agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images,
              answers_dir)


def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(make_hashable(x) for x in obj)
    else:
        return obj
