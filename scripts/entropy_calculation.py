import os
import time
import json
import random
import base64
from typing import List, Literal

import numpy as np
from pydantic import BaseModel, Field

from miscellaneous.prompt import (
    system_prompt_image, user_prompt_image_text_input, user_prompt_image_image_input,
    system_prompt_text, user_prompt_text,
    system_prompt_table, user_prompt_table,
    system_prompt_row, user_prompt_row,
    system_prompt_answer_set, user_prompt_answer_set
)

from miscellaneous.json_schemas import json_schema_check_criteria, json_schema_answer_set, json_schema_table_description


class ModalityDecision(BaseModel):
    modalities: Literal[
        "image",
        "text",
        "table",
        "image_text",
        "image_table",
        "text_table",
    ] = Field(..., description="Predicted modality combination")


class AnswerContainsCriteria(BaseModel):
    answer: str = Field(..., description="Answer yes or no to whether the data contains the criteria")


class ParagraphContainsAnswer(BaseModel):
    contains: bool = Field(...,
                           description="Whether the paragraph contains an element matching the expected answer type")
    entity: str = Field(...,
                        description="The exact text span from the paragraph that matches the expected answer type, or NONE")
    confidence: float = Field(..., ge=0, le=1)


class RowContainsAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the row contains an element matching the expected answer type")
    entity: str = Field(...,
                        description="The exact text span from the row cell that matches the expected answer type, or NONE")
    confidence: float = Field(..., ge=0, le=1)


class YesNoQuestion(BaseModel):
    is_yes_no: bool = Field(..., description="Whether the question is a yes or no question")
    confidence: float = Field(..., ge=0, le=1)


class ImageContainsAnswer(BaseModel):
    contains: bool = Field(..., description="Whether the image describes an entity matching the expected answer type")
    entity: str = Field(..., description="The answer for the question extracted from the image")
    match_level: Literal["specific", "general", "none"]
    confidence: float = Field(..., ge=0, le=1)


class RestrictionCriterias(BaseModel):
    entity: str = Field(..., description="The element that allowed to insert the element in the positive partition")
    confidence: float = Field(..., ge=0, le=1)


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
            # Expected answer type will be used also as criteria for the splitting.
            criterias = [criteria_object["expected_answer_type"]["expected_answer_type_specific"],
                         criteria_object["expected_answer_type"]["expected_answer_type_general"],
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

    def analyse_text_restricting_criteria(self, model, criteria, metadata, text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a precise text reasoning assistant.

Your task is to determine whether the provided PARAGRAPH or its METADATA/TITLE explicitly support at least one fact from the given CRITERIA.

Rules:
1) Use only information explicitly stated in the paragraph or metadata/title.
2) Do NOT use external knowledge, assumptions, or inference beyond the text.
3) Only one supported fact is sufficient.
4) If at least one fact is explicitly supported, respond "yes".

Respond ONLY with "yes" or "no"."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"""
        Given:
        - TITLE/METADATA: {metadata}
        - PARAGRAPH: {text}
        - CRITERIA: {criteria}

        Task:
        Determine whether the paragraph or its metadata/title contain evidence of any facts expressed in the criteria.

        Answer "yes" or "no".
        """,
                            },
                        ]
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
                        "content": f"""You are a precise text reasoning assistant.

Your task is to determine whether a given TABLE ROW, together with the TABLE TITLE and TABLE DESCRIPTION, explicitly support at least one fact from the provided CRITERIA.

Rules:
1) Use only information explicitly stated in the table row, the table title, or the table description.
2) Do NOT use external knowledge, assumptions, or inference beyond what is written.
3) Only one explicitly supported fact is sufficient.
4) If at least one fact is explicitly supported by the row, title, or description, respond "yes".

Respond ONLY with "yes" or "no"."""
                    },
                    {
                        "role": "user",
                        "content": f"""TABLE TITLE:
    {title}
    
    TABLE NAME: 
    {name}

TABLE DESCRIPTION:
{description}

TABLE ROW:
{row}

CRITERIA:
{criteria}

Task:
Determine whether the content of the row or the title of the table or the description of the table contain evidence of any facts expressed in the criteria.

Answer "yes" or "no".
"""
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def extract_restricting_criteria(self, model, question, element, validated_criterias):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": f"""
You are a reasoning assistant that extracts structured information from a given element based on validated criteria.

Task:
- Extract the relevant piece of information from the element that directly answers or constrains the question.
- Use ONLY the information present in the element and the validated criteria.
- Do NOT add external knowledge, assumptions, or guesses.
- Provide the extracted information in a concise, structured format (e.g., key-value pairs or short factual statement) that can be used to refine other modalities.

Output:
- The extracted relevant piece of information only.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
Given:
1) Question: {question}
2) Element (text, image metadata, or other): {element}
3) List of criteria this element satisfies: {validated_criterias}

Provide the extracted relevant information according to the instructions.
"""
                    }
                ],
                text_format=RestrictionCriterias,
            )

            found = response.output_parsed
            return found.entity, found.confidence

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
        """This method checks whether the splitting is possible for each criteria for images."""

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []
            image_set = question_files['image_set']
            for image in image_set:
                # print(f"Image: {image}")
                metadata = image["title"]
                answer = self.analyse_image(model, criteria, metadata, image)

                if answer.lower() == "yes":
                    correct_elements.append(image)

            partition = self.create_partition(image_set, correct_elements, criteria)
            partitions["image"].append(partition)

    def fill_criteria_text(self, model, criterias, question_files, partitions):
        """This method checks whether the splitting is possible for each criteria for texts."""

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []
            text_set = question_files['text_set']
            for text in text_set:
                # print(f"Text: {text}")
                metadata = text["title"]
                answer = self.analyse_text(model, criteria, text["text"], metadata)

                if answer.lower() == "yes":
                    correct_elements.append(text)

            partition = self.create_partition(text_set, correct_elements, criteria)
            partitions["text"].append(partition)

            filling = partition['splitting']['filling']
            not_filling = partition['splitting']['not_filling']

            print(partition['splitting']['filling'], len(filling))
            print(partition['splitting']['not_filling'], len(not_filling))

    def fill_criteria_table(self, model, criterias, question_files, table_dir, partitions):
        """This method checks whether the splitting is possible for each criteria for table rows."""
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
                answer = self.analyse_table_row(model, criteria, row, table_title, table_description)
                if answer.lower() == "yes":
                    correct_elements.append(row)

            partition = self.create_partition(rows, correct_elements, criteria)
            partition["table_understanding"] = table_description
            partition["table_title"] = table_title

            partitions["table"].append(partition)

            filling = partition['splitting']['filling']
            not_filling = partition['splitting']['not_filling']

            print("Parition filling: ", partition['splitting']['filling'], len(filling))
            print("Partition not filling: ", partition['splitting']['not_filling'], len(not_filling))

    def check_answer_in_row(self, model, answer_class_specific, row, question_text, table_description):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a semantic evidence verifier for a multimodal question answering system.

Your task is to determine whether a TABLE ROW contains a value that directly answers the QUESTION.

You are given:
- The QUESTION
- The EXPECTED ANSWER TYPE
- The TABLE DESCRIPTION
- The TABLE ROW (with all cell values)

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

    def check_answer_in_paragraph(self, model, answer_class, question_text, paragraph_text):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": """You are a semantic evidence verifier for a multimodal question answering system.

        Your task is to determine whether a PARAGRAPH contains an entity that matches the EXPECTED ANSWER TYPE required by a QUESTION.

        This is NOT keyword matching.
        You must perform semantic reasoning.

        Guidelines:
        - Consider synonyms, paraphrases, and implicit mentions.
        - The paragraph does not need to explicitly repeat the expected answer type.
        - Only return TRUE if the paragraph contains a specific entity that could directly answer the question.
        - If the paragraph only provides related context but no actual answer entity, return FALSE.
        - If uncertain, return FALSE.

        Extraction rules:
        - If contains = TRUE, extract the exact text span from the paragraph.
        - Do not paraphrase.
        - Do not normalize.
        - Copy the entity exactly as written.
        - If contains = FALSE, entity must be "NONE".

        Base your decision strictly on the provided text.
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

        unimodal_partition_path = os.path.join(f"../results/partitions_created/{model}/partitions_{question}")

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
            with open(f"../results/partitions_created/{model}/partitions_{question}", "w") as json_file:
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

    def extract_partitions(self, model, question, question_text, partitions_path):
        """In this question we check whether it is possible to return an answer with only one modality"""
        # We state that if the predicted modality was one and if there is a partition with minimum logical entropy
        # then we can try to verify whether we can give an answer

        partitions = json.load(
            open(os.path.join(f"../results/{partitions_path}/{model}", f"partitions_{question}"), "rb"))

        average_le = average_modality_le(partitions)
        filtered = {k: v for k, v in average_le.items() if v is not None}
        max_value = max(filtered.values())
        max_modalities = [k for k, v in filtered.items() if v == max_value]

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
            modality = self.decide_answer_modality(model, question_text, max_modalities)
        else:
            # Find the index that has the maximum number of criterias filled
            modality = criterias_filled[max(criterias_filled,
                                            key=lambda k: criterias_filled[k]['criterias_filled'])]["modality"]

        print(f"Minimum modality: {modality}. Min le filling: {min_le_filling}.")
        return modality, partitions, min_le_filling, final_answer

    def find_final_answer(self, model, question, question_text, answer_class_specific, answer_class_general,
                          answers_dir, modality, partitions, min_le_filling):

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
            semantic_checks = []
            for filling in min_le_filling:
                for paragraph in filling['filling']:
                    semantic_checks.append(
                        self.check_answer_in_paragraph(model, answer_class_specific.lower(),
                                                       question_text,
                                                       paragraph))

            print("Semantic checks: ", semantic_checks)
            sorted_results = sorted(semantic_checks, key=lambda x: (x[0], x[2]), reverse=True)
            print("Sorted results: ", sorted_results)

            high_conf = [r[1] for r in sorted_results if r[2] > 0.90]
            if high_conf:
                final_answer = high_conf
            else:
                final_answer = [sorted_results[0][1]] if sorted_results else []
            #final_answer = sorted_results[0][1]

        elif modality == "table":
            table_description = partitions["table"][0]["table_understanding"]
            semantic_checks = []

            yesnoquestion, confidence = self.yesnoquestion(model, question_text)
            print(yesnoquestion)
            if yesnoquestion:
                final_answer = "yes"
                answer_dict = {"final_answer": final_answer}
                json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

                return final_answer

            else:
                for filling in min_le_filling:
                    for row in filling['filling']:
                        semantic_checks.append(self.check_answer_in_row(model, answer_class_specific.lower(),
                                                                        row,
                                                                        question_text,
                                                                        table_description))

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

    def create_multi_hop_partitions(self, model, question, question_text):
        """With this method we generate multi hop partitions within the same modality. In this case we check what are
        the criterias, within the same modality, that allows to diminish the logical entropy"""
        unimodal_partitions_path = os.path.join(f"../results/partitions_created/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

        multihop_partitions_path = os.path.join(f"../results/multimodal_partitions/{model}/partitions_{question}")
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
                restricting_criterias = self.extract_restricting_criteria(
                    model,
                    question_text,
                    best_item["splitting"]["filling"],  # the filling items
                    best_item["criteria"]  # the criteria for this partition
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
        # Of course they must be different elements, otherwise it does not make any sense.
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

    def create_multimodal_partitions(self, model, question, question_text):

        unimodal_partitions_path = os.path.join(f"../results/partitions_created/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

        multimodal_partitions_path = os.path.join(f"../results/multimodal_partitions/{model}/partitions_{question}")
        if os.path.exists(multimodal_partitions_path):
            multimodal_partitions = json.load(open(multimodal_partitions_path, "rb"))
        else:
            multimodal_partitions = {"image": [], "text": [], "table": []}

        """Let's find the modality with the lowest overall logical entropy between the unimodal partitions. 
        However, let's only consider, if there are, the modalities that have not been combined before."""
        average_le_modality = {}
        for modality, items in unimodal_partitions.items():
            le_values = [item['le'] for item in items if "le" in item and item['le'] > 0]
            if le_values:
                average_le_modality[modality] = sum(le_values) / len(le_values)
            else:
                average_le_modality[modality] = None

        print(average_le_modality)
        average_le_modality = {k: v for k, v in average_le_modality.items() if v is not None}
        min_modality = min(average_le_modality, key=average_le_modality.get)
        min_value = average_le_modality[min_modality]

        print(f"Modality with minimum LE: {min_modality}. Minimum LE value: {min_value}")
        multimodal_pairs = [(min_modality, mod) for mod in unimodal_partitions.keys() if unimodal_partitions[mod]
                            and min_modality != mod]
        print(f"Multimodal pairs are: {multimodal_pairs}")

        # Once you have calculated the modality that has the lowest logical entropy you create a set that contains the
        # possible elements with the partitions of that modality"""
        valid_partitions = [p for p in unimodal_partitions[min_modality] if p["le"] > 0]

        fillings_min_modality = {
            "filling": [item for p in valid_partitions for item in p.get("splitting", {}).get("filling", [])],
            "le": np.average([p["le"] for p in valid_partitions]) if valid_partitions else None,
            "criterias": [p["criteria"] for p in valid_partitions],
        }

        unique_filling = list(
            {json.dumps(item, sort_keys=True): item for item in fillings_min_modality["filling"]}.values())
        fillings_min_modality["filling"] = unique_filling

        # Once we have isolated the elements from the first modality (the one with the lowest le) we need to find an
        # intersection between those elements and the ones in the other modality The point is: how to find the
        # intersection? LLM?
        restricting_criterias = self.extract_restricting_criteria(model, question_text,
                                                                  fillings_min_modality["filling"],
                                                                  fillings_min_modality["criterias"])

        existing_combinations = set()
        for main_modality, items in multimodal_partitions.items():
            for el in items:
                cond = el.get("conditional_modalities")
                if cond is not None:
                    existing_combinations.add((main_modality, cond))

        print(f"The pairs of modality are {existing_combinations}")

        for multimodal_pair in multimodal_pairs:
            print(f"Multimodal pair: {multimodal_pair}")
            if multimodal_pair not in existing_combinations:
                modality_to_consider = multimodal_pair[1]
                print(f"Modality to consider now is: {modality_to_consider}")
                if not unimodal_partitions[modality_to_consider]:
                    continue

                for partition in unimodal_partitions[modality_to_consider]:
                    if not partition["splitting"]["filling"]:
                        continue

                    correct_elements = []
                    modality_set = partition["splitting"]["filling"]

                    for element in modality_set:
                        if modality_to_consider == "image":
                            answer = self.analyse_image_restricting_criteria(model, restricting_criterias,
                                                                             element["title"],
                                                                             element["path"])
                        elif modality_to_consider == "text":
                            answer = self.analyse_text_restricting_criteria(model, restricting_criterias,
                                                                            element["title"], element["text"])
                        elif modality_to_consider == "table":

                            association_json = json.load(open(
                                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/association",
                                             question), "rb"))
                            table = association_json["table_set"][0].copy()

                            json_table = json.load(open(
                                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/tables",
                                             table["json"]), "rb"))
                            answer = self.analyse_table_row_restricting_criteria(model, restricting_criterias,
                                                                                 element,
                                                                                 json_table["title"],
                                                                                 json_table["table"]["table_name"],
                                                                                 partition["table_understanding"])

                        if answer.lower() == "yes":
                            correct_elements.append(element)

                    # When calculating the partition, to the modality set we also want to add the elements that did not respect the criteria as the beginning
                    # so, the not filling ones
                    all_elements = partition["splitting"]["filling"] + partition["splitting"]["not_filling"]

                    new_partition = self.create_partition(all_elements, correct_elements, restricting_criterias[0])
                    conditional_criteria = new_partition["criteria"]
                    if modality_to_consider == "table":
                        new_partition["table_understanding"] = partition["table_understanding"]

                    new_partition["criteria"] = {"old_criteria": partition["criteria"],
                                                 "conditional_criteria": conditional_criteria}
                    new_partition["conditional_modalities"] = min_modality

                    multimodal_partitions[modality_to_consider].append(new_partition)

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
        #if H1 == 0:
        #    return 0.0  # avoid division by zero

        return (H2 - H1) / H1

    def choose_unimodal_multimodal(self, unimodal_partitions_path, multimodal_partitions_path):
        """This method compares the entropy of unimodal partitions with the entropy of multimodal partitions and decides
        whether the final answer will be given looking at unimodal partitions of multimodal partitions."""

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

        print(f"Min modality unimodal: {min_modality_multimodal}. Min value unimodal: {min_value_multimodal}")

        if min_modality_multimodal is None:
            return "partitions_created"

        if min_modality_unimodal == min_modality_multimodal:
            if average_modality_le_unimodal[min_modality_unimodal] < average_modality_le_multimodal[
                min_modality_multimodal]:
                return "partitions_created"
            else:
                return "multimodal_partitions"

        # Here we measure the average change of the logical entropy for the minimum modality in unimodal and for the
        # minimum modality in multimodal
        change_minimum_unimodal = self.relative_entropy_change(average_modality_le_unimodal[min_modality_unimodal],
                                                               average_modality_le_multimodal[min_modality_unimodal])

        change_minimum_multimodal = self.relative_entropy_change(average_modality_le_unimodal[min_modality_multimodal],
                                                                 average_modality_le_multimodal[min_modality_multimodal])

        print(change_minimum_unimodal, change_minimum_multimodal)

        # Case in which both of them start greater than 0 meaning that the filling was not empty
        if average_modality_le_unimodal[min_modality_unimodal] and average_modality_le_unimodal[min_modality_multimodal]:
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
                            return "partitions_created"
                        elif change_minimum_multimodal < change_minimum_unimodal:
                            return "multimodal_partitions"
                    elif (average_modality_le_multimodal[min_modality_unimodal] == 0 and
                          average_modality_le_multimodal[min_modality_multimodal] > 0):
                        return "multimodal_partitions"
                    elif average_modality_le_multimodal[min_modality_unimodal] > 0 and average_modality_le_multimodal[
                        min_modality_multimodal] == 0:
                        return "partitions_created"
                    else:
                        return "mannacc a miserj"

                elif average_modality_le_multimodal[min_modality_unimodal] is None or average_modality_le_multimodal[
                    min_modality_multimodal] is None:
                    print("Uno dei due è Non")
                    if min_value_unimodal < min_value_multimodal:
                        return "partitions_created"
                    elif min_value_multimodal < min_value_unimodal:
                        return "multimodal_partitions"
            else:
                print("What are we taking about. It is impossible")

        elif not average_modality_le_unimodal[min_modality_unimodal] or not average_modality_le_unimodal[min_modality_multimodal]:
            if min_value_unimodal < min_value_multimodal:
                return "partitions_created"
            elif min_value_multimodal < min_value_unimodal:
                return "multimodal_partitions"
            else:
                return "partitions_created"

    def return_final_answer(self, model, question, question_files, question_text, priority_modalities, criterias,
                            remaining_modalities, answer_class_specific, answer_class_general, table_dir,
                            final_dataset_images, answers_dir):

        unimodal_partitions_path = os.path.join(f"../results/partitions_created/{model}/partitions_{question}")
        multimodal_partitions_path = os.path.join(f"../results/multimodal_partitions/{model}/partitions_{question}")

        # First we create the partitions, both in the unimodal case and also in the multi-hop unimodal case
        for modality in priority_modalities:
            print(f"CREATING UNIMODAL PARTITIONS for {modality.upper()}")
            print("************************************************************************")
            self.create_unimodal_partitions(model, question, question_files, table_dir, modality, criterias)
            print("************************************************************************")
            print(f"END CREATION")
            print("************************************************************************\n")

        # for modality in priority_modalities:
        print(f"CREATING UNIMODAL MULTI-HOP PARTITIONS")
        print("************************************************************************")
        self.create_multi_hop_partitions(model, question, question_text)
        print("************************************************************************")
        print(f"END CREATION")
        print("************************************************************************\n")

        final_answer = None
        multimodal = len(priority_modalities) > 1

        while final_answer is None:

            if not multimodal:
                print("***************************************************************")
                print("WE START A UNIMODAL ANALYSIS")
                print("***************************************************************")
                partitions_path = self.choose_unimodal_multimodal(unimodal_partitions_path, multimodal_partitions_path)
                print(f"The partitions path is: {partitions_path}")
                modality, partitions, min_le_filling, final_answer = self.extract_partitions(model,
                                                                                             question,
                                                                                             question_text,
                                                                                             partitions_path)

                print("We are still in the moment of not being multimodal")
                print(f"Modality: {modality}")
                print(f"Partitions: {partitions}")
                print(f"Min le filling: ", min_le_filling)

                if final_answer is None:
                    print("Let's calculate the final answer")
                    final_answer = self.find_final_answer(model, question, question_text, answer_class_specific,
                                                          answer_class_general, answers_dir, modality, partitions,
                                                          min_le_filling)

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
                        self.create_unimodal_partitions(model, question, question_files, table_dir, decided_modality, criterias)
                        self.create_multi_hop_partitions(model, question, question_text)
                        multimodal = True

            elif multimodal:
                print("***************************************************************")
                print("WE START A MULTIMODAL ANALYSIS")
                print("***************************************************************")
                # Here we create the multimodal partitions
                multimodal_partitions_path = os.path.join(
                    f"../results/multimodal_partitions/{model}/partitions_{question}")

                # if not os.path.exists(multimodal_partitions_path):
                print("************************************************************************")
                print(f"CREATING MULTIMODAL PARTITIONS")
                multimodal_partitions = self.create_multimodal_partitions(model, question, question_text)
                json.dump(multimodal_partitions,
                          open(f"../results/multimodal_partitions/{model}/partitions_{question}", "w"),
                          indent=4)
                print("************************************************************************\n")

                unimodal_partitions_path = os.path.join(f"../results/partitions_created/{model}/partitions_{question}")

                # Here we choose whether to use unimodal or multimodal, depending from the value of the logical entropy
                print("***************************************************************")
                print("Unimodal or Multimodal?")
                print("***************************************************************")
                partitions_path = self.choose_unimodal_multimodal(unimodal_partitions_path, multimodal_partitions_path)
                print(f"Partitions path {partitions_path}")
                print("***************************************************************")
                print("Extracting the partitions")
                modality, partitions, min_le_filling, final_answer = self.extract_partitions(model,
                                                                                             question,
                                                                                             question_text,
                                                                                             partitions_path)
                print("***************************************************************")

                if final_answer is None:
                    print("Let's give this multimodal final answer")
                    print("Min le filling: ", min_le_filling)
                    final_answer = self.find_final_answer(model, question, question_text, answer_class_specific,
                                                          answer_class_general, answers_dir, modality, partitions,
                                                          min_le_filling)

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

                        print("************************************************************************")
                        print(f"CREATING MULTI-HOP PARTITIONS for {decided_modality.upper()}")
                        print("************************************************************************\n")
                        self.create_multi_hop_partitions(model, question, question_text)

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
            answer_class_specific, answer_class_general, *criterias = self.read_criterias(question, False)
        else:
            answer_class_specific, answer_class_general, *criterias = self.read_criterias(question, False)

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
                                 priority_modalities, criterias, remaining_modalities, answer_class_specific,
                                 answer_class_general, table_dir, final_dataset_images, answers_dir)
        # END ANSWER TO QUESTION


def answer_qa(model, agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images,
              answers_dir):
    os.makedirs(answers_dir, exist_ok=True)
    os.makedirs(f"../results/partitions_created/{model}/", exist_ok=True)

    for i, q in enumerate(questions_list):
        question = q["index"]
        modalities = q["modality"]
        if len(modalities) == 1:
            unimodal_multimodal = "/unimodal/"
        else:
            unimodal_multimodal = "/multimodal/"

        if question not in os.listdir(answers_dir + unimodal_multimodal):
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


def entropy_calculation_main(model, agent, questions_list, questions_dir, association_dir, image_dir, text_dir,
                             table_dir, final_dataset_images, answers_dir):
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
