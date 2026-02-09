import os
import time
import json
import random
import base64
from typing import List
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from miscellaneous.prompt import (
    system_prompt_image, user_prompt_image_text_input, user_prompt_image_image_input,
    system_prompt_text, user_prompt_text,
    system_prompt_table, user_prompt_table,
    system_prompt_row, user_prompt_row,
    system_prompt_answer_set, user_prompt_answer_set
)

from miscellaneous.json_schemas import json_schema_check_criteria, json_schema_answer_set, json_schema_table_description


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


def answer_qa(model, agent, questions_dir, association_dir, table_dir, answers_dir):
    os.makedirs(answers_dir, exist_ok=True)
    os.makedirs(f"../results/partitions_created/{model}/", exist_ok=True)

    for question in sorted(os.listdir(questions_dir))[:1]:
        print("Question: ", question)
        # if question not in os.listdir(os.path.join(answers_dir, mode)):
        question_data = get_question_data(questions_dir, question)
        question_files = get_question_files(association_dir, question)
        agent.answer_question(model, question, question_data, question_files, table_dir, answers_dir)


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

    def read_criterias(self, question):
        criteria_object = json.load(open(os.path.join(self.path_criterias, question), "rb"))

        criterias = [criteria_object["expected_answer_type"]["expected_answer_type_specific"],
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

    def table_general_understanding(self, model, metadata, columns):

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
                        "content": user_prompt_table.format(metadata=metadata, columns=columns)
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
                        "content": user_prompt_table.format(metadata=metadata, columns=columns)
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

        print("Metadata table: ", metadata)
        print("Description table: ", description)
        print("Row: ", row)

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

    def create_answer_set(self, model, target_answer, elements_type, elements):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_answer_set
                    },
                    {
                        "role": "user",
                        "content": user_prompt_answer_set.format(target_answer=target_answer,
                                                                 elements_type=elements_type, elements=elements)
                    }
                ],
                text_format=AnswerSetResponse,
            )

            found = response.output_parsed
            return found.answer_set

        elif model == "claude-sonnet-4-5":
            response = self.client.messages.create(
                model=model,
                system=system_prompt_answer_set,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt_answer_set.format(target_answer=target_answer,
                                                                 elements_type=elements_type, elements=elements)
                    }
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": json_schema_answer_set
                    }
                }
            )

            found = json.loads(response.content[0].text)
            print(found, type(found))
            return found["answer_set"]

    def decide_modality(self, question):
        """This method decides the modality from which to start."""

        rules_json = json.load(
            open("/Users/emanuelemezzi/PycharmProjects/LEGuidance/results/extracted_rules/initialization_rules.json",
                 "rb"))

        rules = [{f"R{i}": {"condition": rule["condition"], "predicted_modalities": rule["predicted_modalities"]}} for
                 i, rule in enumerate(rules_json["rules"])]

        rules.append({
            f"R{len(rules_json['rules'])}": {
                "condition": rules_json["fallback_rule"]["condition"],
                "predicted_modalities": rules_json["fallback_rule"]["predicted_modalities"],
            }
        })

        rule_conditions_string = ""

        for i, rule in enumerate(rules):
            rule_conditions_string += f"\t{i}. R{i}: {rule[f'R{i}']['condition']}\n"

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
                return rule[rule_id]["predicted_modalities"][0]

    def fill_criteria_images(self, model, criterias, question_files, partitions):
        """This method checks whether the splitting is possible for each criteria for images."""

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []
            image_set = question_files['image_set']
            print("The image set is: ", image_set)
            for image in image_set:
                print("The image is: ", image)
                metadata = image["title"]
                answer = self.analyse_image(model, criteria, metadata, image)

                if answer.lower() == "yes":
                    correct_elements.append(image)

            partition = self.create_partition(image_set, correct_elements, criteria)
            partitions["image"].append(partition)

            filling = partition['splitting']['filling']
            not_filling = partition['splitting']['not_filling']

            print(partition['splitting']['filling'], len(filling))
            print(partition['splitting']['not_filling'], len(not_filling))

    def fill_criteria_text(self, model, criterias, question_files, partitions):
        """This method checks whether the splitting is possible for each criteria for texts."""

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []
            text_set = question_files['text_set']
            print("The text set is: ", text_set)
            for text in text_set:
                print("The text is: ", text)
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
        title = entire_table['title']
        print("URL table: ", entire_table['url'])
        table_columns = [element["column_name"] for element in entire_table["table"]["header"]]
        rows = table[list(table.keys())[0]]

        table_description = self.table_general_understanding(model, title, table_columns)

        for criteria in criterias:
            print("Current criteria is: ", criteria)
            correct_elements = []

            for row in rows:
                answer = self.analyse_table_row(model, criteria, row, title, table_description)
                if answer.lower() == "yes":
                    correct_elements.append(row)

            partition = self.create_partition(rows, correct_elements, criteria)

            partitions["table"].append(partition)

            filling = partition['splitting']['filling']
            not_filling = partition['splitting']['not_filling']

            print(partition['splitting']['filling'], len(filling))
            print(partition['splitting']['not_filling'], len(not_filling))

    def fill_criterias(self, model, question: str, question_files: dict, table_dir: str, starting_modality: str,
                       criterias, partitions):
        """This method checks whether images, text, and table rows, can be split based on the distinction criterias."""

        # Here we extract the criterias to search for in the modalities
        # answer_class, *criterias = self.read_criterias(question)
        # n_criterias = len(criterias)

        # Lower bound: only one modality, and upper bound by selecting one modality only
        # Upper bound: brute force And with this process we show the association between the question and the type of
        # modality to answer it (single one of fusion)
        # Check weather the additional modality is noise
        if starting_modality == "image":
            self.fill_criteria_images(model, criterias, question_files, partitions)

        elif starting_modality == "text":
            self.fill_criteria_text(model, criterias, question_files, partitions)

        elif starting_modality == "table":
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

    def return_final_answer(self, model, question, answer_class, answers_dir):
        """This method allows to retrieve the final answer. It receives in input the answer_class,
        and the answer_sets."""
        # answer_sets = {"image": answer_set_image, "text": answer_set_text, "table": answer_set_table}

        partitions = json.load(
            open(os.path.join(f"../results/partitions_created/{model}", f"partitions_{question}"), "rb"))

        print("Partitions are: ", partitions)

        # The partition must bring the modality from which they were created. This way is it possible to
        fillings = [{"modality": modality, "i": i, "filling": partition["splitting"]["filling"], "le": partition["le"],
                     "criteria": partition["criteria"]} for modality in partitions.keys() for i, partition in
                    enumerate(partitions[modality]) if partition["le"] > 0]

        print("The fillings are: ", fillings)

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

        # answer_set = answer_sets[modality]
        # print("The answer set is: ", answer_set)

        embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
        answer_class_embedding = embeddings_model.encode(answer_class.lower())

        final_answer = None
        if modality == "table":
            for filling in min_le_filling:
                for row in filling['filling']:
                    for cell in row:
                        cell_header_embedding = embeddings_model.encode(cell["header"].lower())
                        # if answer_class == cell["header"].lower():
                        if embeddings_model.similarity(answer_class_embedding, cell_header_embedding) > 0.90:
                            final_answer = cell["text"]

                        # if answer_class == cell["header"].lower() and cell["text"] in answer_set:
                        #    final_answer = cell["text"]

        answer_dict = {"final_answer": final_answer}
        json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

        return final_answer

    def answer_question(self, model, question, question_data, question_files, table_dir, answers_dir):
        """Method which calls the other methods to calculate the final answer."""

        print(f"Let's answer the question {question}")
        print(f"This is the question data: {question_data}")

        answer_class, *criterias = self.read_criterias(question)

        print(f"This is the type of answer we want: {answer_class}")

        # START CREATE ANSWER SETS

        """
        # Answer set images
        image_set = question_files["image_set"]
        print("Image set: ", image_set)
        answer_set_image = self.create_answer_set(model, answer_class, "images", image_set)
        print("The image answer set is: ", answer_set_image)

        # Answer set text
        text_set = question_files["text_set"]
        answer_set_text = self.create_answer_set(model, answer_class, "paragraphs", text_set)
        print("The text answer set is: ", answer_set_text)

        # Answer set table
        table_set = question_files["table_set"]
        table = table_set[0].copy()
        rows = table[list(table.keys())[0]]

        answer_set_table = self.create_answer_set(model, answer_class, "table_rows", rows)
        print("The table answer set is: ", answer_set_table)
        # END ANSWER SET GENERATION
        """

        """
        # START MODALITY SELECTION
        modalities_to_try = ["image", "table", "text"]
        modality = "image"  # self.decide_modality(question)
        # END MODALITY SELECTION

        # START OF THE PARTITIONING HERE: CHOSE ANOTHER MODALITY UNTIL EACH ONE OF THEM HAS BEEN CHOSEN
        partitions = {"image": [], "text": [], "table": []}
        while modalities_to_try:
            print(f"The current modality is {modality.upper()}")
            self.fill_criterias(model, question, question_files, table_dir, modality, criterias, partitions)

            modalities_to_try.remove(modality)
            if modalities_to_try:
                modality = random.choice(list(modalities_to_try))
        """

        print("Ciao")
        # END OF THE PARTITIONING HERE

        # final_answer = self.return_final_answer(model, question, answer_class, answer_set_image, answer_set_text,
        #                                       answer_set_table, answers_dir)

        final_answer = self.return_final_answer(model, question, answer_class, answers_dir)

        print("The final answer is: ", final_answer)


def entropy_calculation_main(model, agent, modalities, questions_dir, association_dir, image_dir, text_dir, table_dir,
                             answers_dir):
    # create_association_qa(agent, questions_dir, association_dir, image_dir, text_dir, table_dir)

    answer_qa(model, agent, questions_dir, association_dir, table_dir, answers_dir)

    """
    log_e = a.logical_entropy({'filling': {'A', 'B', 'C'}, 'not_filling': {'D', 'E', 'F'}})
    print("Logical entropy is: ", log_e)
    log_e = a.logical_entropy({'filling': {'A'}, 'not_filling': {'B', 'C', 'D', 'E', 'F'}})
    print("Logical entropy is: ", log_e)
    """
