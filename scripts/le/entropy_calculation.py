from schemas.json_schemas import json_schema_check_criteria, json_schema_table_description
from schemas.pydantic_schemas import *
from utils.utilities import *

from prompts.le.prompt_modality_decision import *
from prompts.le.prompt_question_type import *
from prompts.le.prompt_analyse_criteria import *
from prompts.le.prompt_bridge_el_extraction import *
from prompts.le.prompt_bridge_modality import *
from prompts.le.prompt_check_answer import *

from functions_entropy_calculation import *

iteration = 'iteration_1'

os.chdir(os.path.dirname(__file__))


class LEAgent:
    def __init__(self, openai_client, bedrock_client, path_criterias, modalities):
        self.openai_client = openai_client
        self.bedrock_client = bedrock_client
        self.path_criterias = path_criterias
        self.modalities = modalities

    @staticmethod
    def data_preparation(texts, images, final_dataset_images, table, table_dir):
        paragraphs_text = "\n\n".join([
            f"{i + 1}. Title: {p['title']}\nContent: {p['text']}"
            for i, p in enumerate(texts)
        ])

        images_text = "\n\n".join([
            f"{i + 1}. Title: {img['title']}"
            for i, img in enumerate(images)
        ])

        # images_text = [f"{i + 1}. Title: {img['title']}" for i, img in enumerate(images)]

        image_inputs = []
        for img in images:
            image64 = encode_image(os.path.join(final_dataset_images, img["path"]))
            image_inputs.append({
                "title": img["title"],
                "image_url": f"{image64}",
                "image_ext": detect_media_type_from_bytes(
                    open(os.path.join(final_dataset_images, img["path"]), "rb").read())
            })

        json_table = json.load(open(os.path.join(table_dir, table["json"]), "rb"))

        tables_text = f"""Table Title: {json_table["title"]}
        Table name: {json_table["table"]["table_name"]}
        Content: {json_table["table"]}
        """

        return paragraphs_text, images_text, image_inputs, tables_text

    @staticmethod
    def decide_modality_llm(model, openai_client, bedrock_client, question, images, texts, table, table_dir,
                            final_dataset_images):
        """This method decides the modality by showing the data to the LLM"""

        paragraphs_text, images_text, images_inputs, tables_text = (
            LEAgent.data_preparation(texts, images, final_dataset_images, table, table_dir))

        if model == "global.amazon.nova-2-lite-v1:0":
            return decide_modality_llm_nova2(model, bedrock_client, system_prompt_modality, user_prompt_modality,
                                             question, images_text, images_inputs, paragraphs_text, tables_text)

        elif model == "gpt-5.2" or model == "mistral.mistral-large-3-675b-instruct":
            return decide_modality_llm_gpt(openai_client, system_prompt_modality, user_prompt_modality, question,
                                           images_text, images_inputs, paragraphs_text, tables_text)

        elif model == "moonshotai.kimi-k2.5":
            return decide_modality_llm_moonshot(model, bedrock_client, system_prompt_modality, user_prompt_modality,
                                                question, images_text, images_inputs, paragraphs_text, tables_text)

        elif model == "nvidia.nemotron-nano-12b-v2":
            return decide_modality_llm_moonshot(model, bedrock_client, system_prompt_modality, user_prompt_modality,
                                                question, images_text, images_inputs, paragraphs_text, tables_text)

        elif model == "qwen.qwen3-vl-235b-a22b":
            return decide_modality_llm_qwen(model, bedrock_client, system_prompt_modality, user_prompt_modality,
                                            question, images_text, images_inputs, paragraphs_text, tables_text)

        elif model == "us.anthropic.claude-sonnet-4-6":
            return decide_modality_llm_claude(model, bedrock_client, system_prompt_modality, user_prompt_modality,
                                              question, images_text, images_inputs, paragraphs_text, tables_text)

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

        response = self.openai_client.responses.parse(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": system_prompt_reduced_modality.format(remaining_modalities=remaining_modalities)
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt_reduced_modality.format(question=question,
                                                                        available_content=available_content)
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

    def yesnoquestion(self, model, openai_client, bedrock_client, question_text):
        if (model == "global.amazon.nova-2-lite-v1:0" or model == "mistral.mistral-large-3-675b-instruct" or
                model == "moonshotai.kimi-k2.5" or model == "nvidia.nemotron-nano-12b-v2" or
                model == "qwen.qwen3-vl-235b-a22b" or model == "us.anthropic.claude-sonnet-4-6"):
            return yesnoquestion_amazon(model, bedrock_client, system_prompt_bool_question, question_text)

        elif model == "gpt-5.2" or model:
            return yesnoquestion_gpt(openai_client, system_prompt_bool_question, question_text)

    def iscomparison(self, model, openai_client, bedrock_client, question_text):
        if (model == "global.amazon.nova-2-lite-v1:0" or model == "mistral.mistral-large-3-675b-instruct" or
                model == "moonshotai.kimi-k2.5" or model == "nvidia.nemotron-nano-12b-v2" or
                model == "qwen.qwen3-vl-235b-a22b" or model == "us.anthropic.claude-sonnet-4-6"):
            return iscomparison_amazon(model, bedrock_client, system_prompt_comparison_question, question_text)

        elif model == "gpt-5.2":
            return iscomparison_gpt(openai_client, system_prompt_comparison_question, question_text)

    def isgraphical(self, model, question_text):
        if (model == "global.amazon.nova-2-lite-v1:0" or model == "mistral.mistral-large-3-675b-instruct" or
                model == "moonshotai.kimi-k2.5" or model == "nvidia.nemotron-nano-12b-v2" or
                model == "qwen.qwen3-vl-235b-a22b" or model == "us.anthropic.claude-sonnet-4-6"):
            return isgraphical_amazon(model, self.bedrock_client, system_prompt_isgraphical_question, question_text)

        if model == "gpt-5.2":
            return isgraphical_gpt(model, self.openai_client, system_prompt_isgraphical_question, question_text)

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
                         criteria_object["rewritten_question"],
                         criteria_object["expected_answer_type"]["expected_answer_type_specific"],
                         criteria_object["rewritten_question"],
                         criteria_object["target"]["text"],
                         criteria_object["asked_property"]]

            for constraint in criteria_object["constraints"]:
                criterias.append(constraint['evidence'])

            return criterias

    def analyse_text_restricting_criteria(self, model, criteria, metadata, text):
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
                        "content": user_prompt_text.format(metadata=metadata, text=text, criteria=criteria)
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

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
                        "content": system_prompt_image
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt_image_text.format(metadata=metadata, criteria=criteria),
                            },
                            {
                                "type": "input_image",
                                "image_url": user_prompt_image_image.format(image64=image64),
                            },
                        ]
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

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
                        "content": user_prompt_table.format(table_title=table_title,
                                                            table_name=table_name,
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
                        "content": user_prompt_table.format(table_title=table_title,
                                                            table_name=table_name,
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

    def analyse_table_row_restricting_criteria(self, model, criteria, row):
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
                        "content": user_prompt_row.format(row=row, criteria=criteria)
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def create_unimodal_partitions(self, model, question, question_files, table_dir, modality, criterias, dataset,
                                   approach, setting):
        """This method checks whether images, text, and table rows, can be split based on the distinction criterias."""

        unimodal_partition_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")

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
            with open(
                    f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}",
                    "w") as json_file:
                json.dump(partitions, json_file, indent=4)

    @staticmethod
    def logical_entropy(partition):
        """This method calculates the logical entropy, which depends on how the elements were split."""
        if len(partition) == 1:
            le = 1
        else:
            n_elements = sum([len(partition[key]) for key in partition])
            p = 1 / n_elements
            cumulative_prob = sum([pow(p * len(partition[key]), 2) for key in partition])
            le = 1 - cumulative_prob

        return le if le > 0 or le < 1 else 1

    @staticmethod
    def create_partition(all_data, selected_data, criteria):
        """This method creates the partition."""

        not_filling = [item for item in all_data if item not in selected_data]

        set_splitting = {'filling': selected_data, 'not_filling': not_filling}
        le = LEAgent.logical_entropy(set_splitting)
        partition = {'splitting': set_splitting, 'le': le, 'criteria': criteria}

        return partition

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

            partition = LEAgent.create_partition(image_set, correct_elements, criteria)
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

            partition = LEAgent.create_partition(text_set, correct_elements, criteria)
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
                answer = self.analyse_table_row_restricting_criteria(model, criteria, row)

                # answer = self.analyse_table_row(model, criteria, row, table_title, table_description)
                if answer.lower() == "yes":
                    correct_elements.append(row)

            partition = LEAgent.create_partition(rows, correct_elements, criteria)
            partition["table_understanding"] = table_description
            partition["table_title"] = table_title
            partition["table_name"] = table_name

            partitions["table"].append(partition)

    def extract_restricting_criteria_text(self, model, question_text, title, text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_restricting_text
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_restricting_text.format(question_text=question_text, title=title,
                                                                     text=text)
                            }
                        ]
                    }
                ],
                text_format=ParagraphExtraction,
            )

            bridge_element = response.output_parsed
            print(bridge_element)
            return bridge_element.evidence

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
                        "content": system_restricting_image
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_restricting_image_text.format(question_text=question_text,
                                                                           image_title=image_title)
                            },
                            {
                                "type": "input_image",
                                "image_url": user_prompt_image_image.format(image64=image64)
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
                                               table_description, table_row):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_restricting_table_row
                    },
                    {
                        "role": "user",
                        "content": user_restricting_table_row.format(question_text=question_text,
                                                                     document_title=document_title,
                                                                     table_name=table_name,
                                                                     table_description=table_description,
                                                                     table_row=table_row)
                    },
                ],
                text_format=TableRowExtraction,
            )

            bridge_element = response.output_parsed
            print(bridge_element)
            return bridge_element.evidence

    def analyse_text_bridge_element(self, model, question_text, criteria, title, text):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_text_bridge
                    },
                    {
                        "role": "user",
                        "content": user_prompt_text_bridge.format(question_text=question_text, criteria=criteria,
                                                                  title=title, text=text)
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def analyse_image_bridge_element(self, model, question_text, criteria, image_title, image_path):
        if model == "gpt-5.2":
            image64 = encode_image(
                os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/final_dataset_images",
                             image_path))

            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_image_bridge
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_prompt_image_bridge_text.format(question_text=question_text,
                                                                             criteria=criteria,
                                                                             image_title=image_title)
                            },
                            {
                                "type": "input_image",
                                "image_url": user_prompt_image_image.format(image64=image64),
                            },
                        ]
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

    def analyse_table_row_bridge_criteria(self, model, question_text, criteria, table_row, table_name,
                                          table_description):
        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt_row_bridge
                    },
                    {
                        "role": "user",
                        "content": user_prompt_row_bridge.format(question_text=question_text, criteria=criteria,
                                                                 table_name=table_name,
                                                                 table_description=table_description,
                                                                 table_row=table_row)
                    }
                ],
                text_format=AnswerContainsCriteria,
            )

            found = response.output_parsed
            return found.answer

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

    def check_answer_in_paragraph(self, model, answer_class, question_text, paragraph_text, contextual_information):

        if model == "gpt-5.2":
            response = self.client.responses.parse(
                model="gpt-5.2",
                input=[
                    {
                        "role": "system",
                        "content": system_check_answer_text
                    },
                    {
                        "role": "user",
                        "content": user_check_answer_text.format(question_text=question_text, answer_class=answer_class,
                                                                 paragraph_text=paragraph_text,
                                                                 contextual_information=contextual_information)
                    }
                ],
                text_format=ParagraphContainsAnswer,
            )

            found = response.output_parsed
            return found.contains, found.entity, found.confidence

    def check_answer_image(self, model, answer_class_specific, answer_class_general, question_text,
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
                        "content": system_check_answer_image
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": user_check_answer_image.format(question_text=question_text,
                                                                       answer_class_specific=answer_class_specific,
                                                                       answer_class_general=answer_class_general,
                                                                       caption_text=caption_text)
                            },
                            {
                                "type": "input_image",
                                "image_url": user_prompt_image_image.format(image64=image64)
                            },
                        ],
                    },
                ],
                text_format=ImageContainsAnswer,
            )

            found = response.output_parsed
            return found.contains, found.entity, found.match_level, found.confidence

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
                            "content": system_check_answer_row_cond_criteria
                        },
                        {
                            "role": "user",
                            "content": user_check_answer_row_cond_criteria.format(question_text=question_text,
                                                                                  answer_class_specific=answer_class_specific,
                                                                                  table_description=table_description,
                                                                                  conditional_criteria=conditional_criteria,
                                                                                  row=row)
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
                            "content": system_check_answer_row
                        },
                        {
                            "role": "user",
                            "content": user_check_answer_row.format(question_text=question_text,
                                                                    answer_class_specific=answer_class_specific,
                                                                    table_description=table_description,
                                                                    row=row)
                        }
                    ],
                    text_format=RowContainsAnswer,
                )

                found = response.output_parsed
                return found.contains, found.entity, found.confidence

    @staticmethod
    def extract_partitions(model, question, mod_chosen, question_text, partitions_path, dataset, approach, setting):
        """In this question we check whether it is possible to return an answer with only one modality"""
        # We state that if the predicted modality was one and if there is a partition with minimum logical entropy
        # then we can try to verify whether we can give an answer

        partitions = json.load(
            open(os.path.join(
                f"../../results/{dataset}/{approach}/{setting}/partitions/{partitions_path}/{iteration}/{model}",
                f"partitions_{question}"), "rb"))

        if mod_chosen:
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
                        for el in element["filling"]:
                            if el in filling["filling"]:
                                criterias_filled[(element["i"], element['modality'])]["criterias_filled"] += 1

            print(f"Minimum modality: {mod_chosen}. Min le filling: {min_le_filling}.")

            return mod_chosen, partitions, min_le_filling, final_answer

        """
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
                        for el in element["filling"]:
                            if el in filling["filling"]:
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
        """

    def find_final_answer(self, model, question, question_text, answer_class_specific, answer_class_general,
                          answers_dir, modality, partitions, min_le_filling, iscomparison, num_elements):

        print(f"The question text is: {question_text}")

        if not any(d.get("filling") for d in min_le_filling):
            print("Fillings are empty")
            yesnoquestion, confidence = self.yesnoquestion(model, self.openai_client, self.bedrock_client,
                                                           question_text)
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
                        semantic_checks.append(self.check_answer_image(model, answer_class_specific.lower(),
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

            yesnoquestion, confidence = self.yesnoquestion(model, self.openai_client, self.bedrock_client,
                                                           question_text)
            print(yesnoquestion)
            if yesnoquestion:
                final_answer = "yes"
                answer_dict = {"final_answer": final_answer}
                json.dump(answer_dict, open(os.path.join(answers_dir, question), "w"), indent=4)

                return final_answer

            elif iscomparison:
                print("It is a comparison task")
                if len(min_le_filling) >= num_elements:
                    semantic_checks.append(self.check_answer_in_row(model,
                                                                    answer_class_specific.lower(),
                                                                    min_le_filling,
                                                                    question_text,
                                                                    table_description,
                                                                    conditional_criteria))
                else:
                    semantic_checks.append((False, "NONE", 1.00))
            else:
                print("It is not a comparison task and not a boolean question")
                for filling in min_le_filling:
                    for row in filling['filling']:
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

    def create_multi_hop_partitions(self, model, question, question_text, dataset, approach, setting):
        """With this method we generate multi hop partitions within the same modality. In this case we check what are
        the criterias, within the same modality, that allows to diminish the logical entropy"""
        unimodal_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

        multihop_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")
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

                restricting_criterias = []

                # Apply your restricting_criteria function
                if modality == "text":
                    restricting_criterias = self.extract_restricting_criteria_text(
                        model,
                        question_text,
                        best_item["splitting"]["filling"]['title'],  # the filling items
                        best_item["splitting"]["filling"]['text'])

                elif modality == "image":
                    restricting_criterias = self.extract_restricting_criteria_image(
                        model,
                        question_text,
                        best_item["splitting"]["filling"]['title'],  # the filling items
                        best_item["splitting"]["filling"]['path'])

                elif modality == "table":
                    restricting_criterias = self.extract_restricting_criteria_table_row(
                        model,
                        question_text,
                        best_item["splitting"]["filling"],
                        None,
                        None,
                        None)

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
                                answer = "no"
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
                                                                                         element)

                                if answer.lower() == "yes":
                                    correct_elements.append(element)

                            # When calculating the partition, to the modality set we also want to add the elements that did not
                            # respect the criteria as the beginning so, the not filling ones
                            all_elements = partition["splitting"]["filling"] + partition["splitting"]["not_filling"]

                            new_partition = LEAgent.create_partition(all_elements, correct_elements,
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

    def return_restricting_criteria(self, model, question, question_text, unimodal_partitions, conditional_modality,
                                    dataset, approach, setting):

        # Here we take all the partitions with logical entropy greater than 0
        valid_partitions_conditional_modality = [p for p in unimodal_partitions[conditional_modality] if p["le"] > 0 or
                                                 p['splitting']['filling']]

        if not valid_partitions_conditional_modality:
            return [], []

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

        old_criterias = []
        for el_id in element_mapping.keys():
            # print("El filling: ", element_mapping[id]['el_filling'])
            # print("El criterias: ", element_mapping[id]['criterias'])
            old_criterias.extend(element_mapping[el_id]['criterias'])

        if os.path.exists(
                f"../../results/{dataset}/{approach}/{setting}/restricting_criterias_extraction/{model}/{question}"):

            print("The restricting file already exist")
            restricting_criterias_json = (
                json.load(open(
                    f"../../results/{dataset}/{approach}/{setting}/restricting_criterias_extraction/{model}/{question}",
                    "rb")))

            if conditional_modality in restricting_criterias_json:
                print("The conditional modality already exists in it")
                return restricting_criterias_json[conditional_modality], old_criterias

        print("The restricting file does not exists")
        restricting_criterias = []

        if conditional_modality == "text":
            # for filling in fillings_conditional_modality:
            for el_id in element_mapping.keys():
                print(f"The element from which to extract the criteria is: {element_mapping[el_id]['el_filling']}")
                restricting_criterias.append(self.extract_restricting_criteria_text(model, question_text,
                                                                                    element_mapping[el_id][
                                                                                        'el_filling'][
                                                                                        'title'],
                                                                                    element_mapping[el_id][
                                                                                        'el_filling'][
                                                                                        'text']))
        elif conditional_modality == "image":
            # for filling in fillings_conditional_modality:
            for el_id in element_mapping.keys():
                print(f"The element from which to extract the criteria is: {element_mapping[el_id]['el_filling']}")
                restricting_criterias.append(self.extract_restricting_criteria_image(model, question_text,
                                                                                     element_mapping[el_id][
                                                                                         'el_filling'][
                                                                                         'title'],
                                                                                     element_mapping[el_id][
                                                                                         'el_filling'][
                                                                                         'path']))
        elif conditional_modality == "table":
            document_title = valid_partitions_conditional_modality[0]["table_title"]
            table_name = valid_partitions_conditional_modality[0]["table_name"]
            table_description = valid_partitions_conditional_modality[0]["table_understanding"]

            # for filling in fillings_conditional_modality:
            for el_id in element_mapping.keys():
                print(f"The element from which to extract the criteria is: {element_mapping[el_id]['el_filling']}")
                restricting_criterias.append(self.extract_restricting_criteria_table_row(model,
                                                                                         question_text,
                                                                                         document_title,
                                                                                         table_name,
                                                                                         table_description,
                                                                                         element_mapping[el_id][
                                                                                             'el_filling']))

        if os.path.exists(
                f"../../results/{dataset}/{approach}/{setting}/restricting_criterias_extraction/{question}"):
            restricting_criterias_json = json.load(open(
                f"../../results/{dataset}/{approach}/{setting}/restricting_criterias_extraction/{question}", "rb"))

            print("The file already exists but the conditional modality is new")
            restricting_criterias_json[conditional_modality] = restricting_criterias

        else:
            print("The file does not exists. We need to build it")
            restricting_criterias_json = {conditional_modality: restricting_criterias}

        json.dump(restricting_criterias_json, open(
            f"../../results/{dataset}/{approach}/{setting}/restricting_criterias_extraction/{question}",
            "w"), indent=4)

        return restricting_criterias, old_criterias

    def create_multi(self, model, question, question_text, multimodal_partitions, unimodal_partition,
                     conditional_modality, conditioned_modality, restricting_criterias):

        correct_elements = []
        modality_set = unimodal_partition["splitting"]["filling"]

        for element in modality_set:
            answer = "no"
            print(f"The conditioned element is: {element}")
            if conditioned_modality == "image":
                answer = self.analyse_image_bridge_element(model,
                                                           question_text,
                                                           restricting_criterias,
                                                           element["title"],
                                                           element["path"])

            elif conditioned_modality == "text":
                answer = self.analyse_text_bridge_element(model,
                                                          question_text,
                                                          restricting_criterias,
                                                          element["title"],
                                                          element["text"])

            elif conditioned_modality == "table":
                association_json = json.load(open(
                    os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/association", question),
                    "rb"))
                table = association_json["table_set"][0].copy()

                json_table = json.load(open(
                    os.path.join("/Users/emanuelemezzi/Desktop/datasetNIPS/multimodalqa_files/tables", table["json"]),
                    "rb"))
                answer = self.analyse_table_row_bridge_criteria(model,
                                                                question_text,
                                                                restricting_criterias,
                                                                element,
                                                                json_table["table"]["table_name"],
                                                                unimodal_partition["table_understanding"])

            print(f"The answer for the conditioned element is: {answer.lower()}")
            if answer.lower() == "yes":
                correct_elements.append(element)

        # When calculating the partition, to the modality set we also want to add the elements that did not respect
        # the criteria as the beginning so, the not filling ones
        all_elements = unimodal_partition["splitting"]["filling"] + unimodal_partition["splitting"]["not_filling"]

        new_partition = LEAgent.create_partition(all_elements, correct_elements, restricting_criterias)
        conditional_criteria = new_partition["criteria"]
        if conditioned_modality == "table":
            new_partition["table_understanding"] = unimodal_partition["table_understanding"]

        new_partition["criteria"] = {"old_criteria": unimodal_partition["criteria"],
                                     "conditional_criteria": conditional_criteria}
        new_partition["conditional_modalities"] = conditional_modality

        multimodal_partitions[conditioned_modality].append(new_partition)

    def create_multimodal_partitions(self, model, question, question_text, dataset, approach, setting):
        unimodal_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

        multimodal_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")
        if os.path.exists(multimodal_partitions_path):
            multimodal_partitions = json.load(open(multimodal_partitions_path, "rb"))
        else:
            multimodal_partitions = {"image": [], "text": [], "table": []}

        # Check which modalities are non-empty in the unimodal as between those we can do the conditioning
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
            if pair not in existing_combinations:
                print(f"Let's analyse {pair}")

                conditional_modality = pair[0]
                conditioned_modality = pair[1]

                print(f"Conditional modality: {conditional_modality}. Conditioned modality: {conditioned_modality}")

                restricting_criterias, old_criterias = self.return_restricting_criteria(model,
                                                                                        question,
                                                                                        question_text,
                                                                                        unimodal_partitions,
                                                                                        conditional_modality,
                                                                                        dataset, approach, setting)

                restricting_criterias = [el for el in restricting_criterias if el is not None and el.lower() != 'none']
                print(f"The restricting criterias are: {restricting_criterias}")
                print(f"The old criterias are: {old_criterias}")

                if restricting_criterias:
                    for unimodal_partition in unimodal_partitions[conditioned_modality]:
                        if unimodal_partition["splitting"]["filling"]:
                            # if not restricting_criterias:
                            #    restricting_criterias = old_criterias

                            # print(f"The old restricting criterias are: {restricting_criterias}")

                            self.create_multi(model, question, question_text, multimodal_partitions, unimodal_partition,
                                              conditional_modality, conditioned_modality, restricting_criterias)

        return multimodal_partitions

    @staticmethod
    def relative_entropy_change(h1, h2):
        """
        Calculate the signed relative change from H1 to H2.

        Positive → increase (worse)
        Negative → decrease (better)
        Zero → no change
        """

        if h2 is None:
            return h1  # treat missing value as no change
        if h1 is None or h1 == 0:
            return -h2
        # if H1 == 0:
        #    return 0.0  # avoid division by zero

        return (h1 - h2) / h1

    @staticmethod
    def find_min(average_le):
        min_modality, min_value = None, None
        if any(average_le[key] for key in average_le.keys()):
            min_modality = min((k for k, v in average_le.items() if v is not None), key=lambda k: average_le[k])
            min_value = average_le[min_modality]

        return min_modality, min_value

    @staticmethod
    def calculate_entropy_evolution(average_modality_le_unimodal, average_modality_le_multimodal,
                                    modality_answers_given):

        # min_modality_unimodal, min_value_unimodal = LEAgent.find_min(average_modality_le_unimodal)

        # min_modality_multimodal, min_value_multimodal = LEAgent.find_min(average_modality_le_multimodal)

        print(f"Average modality le unimodal: {average_modality_le_unimodal}, {average_modality_le_multimodal}")

        modality_answers_given["unimodal_entropies"].append(average_modality_le_unimodal)
        modality_answers_given["multimodal_entropies"].append(average_modality_le_multimodal)

        print(modality_answers_given)

        return average_modality_le_unimodal, average_modality_le_multimodal

    @staticmethod
    def values_different(v1, v2):
        # Case 1: one is None and the other is not None → True
        if (v1 is None) and v2 is not None:
            return True

        # Case 2: one is not None and the other is None -> False
        if (v1 is not None) and v2 is None:
            return False

        # Case 2: they are both None → SAME
        if v1 is None and v2 is None:
            return False

        # Case 4: normal fallback
        return v1 != v2

    @staticmethod
    def get_differences(prev, curr):
        diffs = []
        print(f"Prev is: {prev}")
        keys = set(prev.keys())

        for k in keys:
            if LEAgent.values_different(prev.get(k), curr.get(k)):
                diffs.append(k)

        return diffs

    @staticmethod
    def update_answers(state, prev, curr, uni_multi):
        for mod in LEAgent.get_differences(prev, curr):
            if state[uni_multi].get(mod) is False:
                state[uni_multi][mod] = True

    @staticmethod
    def choose_unimodal_multimodal(is_comparison, unimodal_partitions_path, multimodal_partitions_path,
                                   modality_answers_given):
        """This method compares the entropy of unimodal partitions with the entropy of multimodal partitions and decides
        whether the final answer will be given looking at unimodal partitions of multimodal partitions."""

        print(f"Modality answers given are: {modality_answers_given}")

        average_modality_le_unimodal = None

        if os.path.exists(unimodal_partitions_path):
            unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))

            # Here we gather the minimum element and le in unimodal
            average_modality_le_unimodal = average_modality_le(unimodal_partitions)
            print(f"Average modality le unimodal: {average_modality_le_unimodal}")

        if os.path.exists(multimodal_partitions_path):
            multimodal_partitions = json.load(open(multimodal_partitions_path, "rb"))

        else:
            valid = {mod: val for mod, val in average_modality_le_unimodal.items() if
                     val is not None and modality_answers_given["unimodal_answers"].get(mod)}

            min_mod = min(valid, key=valid.get)
            return "unimodal_partitions", min_mod

        average_modality_le_multimodal = average_modality_le(multimodal_partitions)
        print(f"Average modality le multimodal: {average_modality_le_multimodal}")

        print(f"Average le unimodal and multimodal: {average_modality_le_unimodal}, {average_modality_le_multimodal}")

        LEAgent.calculate_entropy_evolution(average_modality_le_unimodal,
                                            average_modality_le_multimodal,
                                            modality_answers_given)

        print(average_modality_le_multimodal)
        if len(modality_answers_given["unimodal_entropies"]) >= 2:
            LEAgent.update_answers(modality_answers_given,
                                   modality_answers_given["unimodal_entropies"][-2],
                                   modality_answers_given["unimodal_entropies"][-1],
                                   "unimodal_answers")

            print(f"Updated: {modality_answers_given}")

        if len(modality_answers_given["multimodal_entropies"]) >= 2:
            LEAgent.update_answers(modality_answers_given,
                                   modality_answers_given["multimodal_entropies"][-2],
                                   modality_answers_given["multimodal_entropies"][-1],
                                   "multimodal_answers")

            print(f"Updated: {modality_answers_given}")

        # In case there is a comparison we check first the ones for which there was a change in logical entropy
        if is_comparison:
            print("It's a comparison. We need to check the positive changes")
            relative_changes = {}
            for mod in average_modality_le_unimodal.keys():
                if average_modality_le_unimodal[mod] and average_modality_le_multimodal[mod]:
                    relative_change = LEAgent.relative_entropy_change(average_modality_le_unimodal[mod],
                                                                      average_modality_le_multimodal[mod])

                    if relative_change < 0:
                        relative_changes[mod] = relative_change

            print(f"Positive changes: {relative_changes}")
            if relative_changes:
                sorted_modalities = sorted(
                    relative_changes.items(),
                    key=lambda x: x[1],
                    reverse=False
                )

                # Pick the best valid modality
                mod_max_relative_change = None
                for mod, change in sorted_modalities:
                    if modality_answers_given["multimodal_answers"][mod]:
                        mod_max_relative_change = mod
                        break

                if mod_max_relative_change is None:
                    valid = {mod: val for mod, val in average_modality_le_unimodal.items() if
                             val is not None and modality_answers_given["unimodal_answers"].get(mod)}

                    if valid:
                        min_mod = min(valid, key=valid.get)
                        print(f"Unimodal mod min: {min_mod}")
                        return "unimodal_partitions", min_mod
                    else:
                        valid_multimodal = {mod: val for mod, val in average_modality_le_multimodal.items() if
                                            val is not None and modality_answers_given["multimodal_answers"].get(
                                                mod)}

                        if valid_multimodal:
                            min_mod = min(valid_multimodal, key=valid_multimodal.get)
                            print(f"Multimodal rule broken min: {min_mod}")
                            return "multimodal_partitions", min_mod
                        else:
                            return None, None

                print(f"Modality with max relative change and relative change: {mod_max_relative_change}, "
                      f"{relative_changes[mod_max_relative_change]}")

                return "multimodal_partitions", mod_max_relative_change

        # Here we check whether there is any modality that is not False and thus that can be used
        if any(v for v in modality_answers_given["unimodal_answers"].values()) or any(
                v for v in modality_answers_given["multimodal_answers"].values()):

            # We check the case in which any key that was None in unimodal became not None in multimodal.
            if any(average_modality_le_unimodal[k] is None and average_modality_le_multimodal.get(k) is not None for k
                   in average_modality_le_unimodal):

                list_multi = [modality for modality in average_modality_le_multimodal if
                              average_modality_le_unimodal.get(modality) is None and
                              average_modality_le_multimodal.get(modality) is not None]

                for mod in list_multi:
                    if modality_answers_given["multimodal_answers"][mod]:
                        return "multimodal_partitions", mod

            # We check whether any key that in unimodal was not None than became None. This is to check the
            # importance of the order.
            if any(average_modality_le_unimodal[k] is not None and average_modality_le_multimodal.get(k) is None
                   for k in average_modality_le_unimodal):

                print("We are in this case")
                relative_changes = {}
                for mod in average_modality_le_unimodal.keys():
                    if average_modality_le_unimodal[mod] and average_modality_le_multimodal[mod]:
                        relative_change = LEAgent.relative_entropy_change(average_modality_le_unimodal[mod],
                                                                          average_modality_le_multimodal[mod])

                        if relative_change >= 0:
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
                    if modality_answers_given["multimodal_answers"][mod]:
                        mod_max_relative_change = mod
                        break

                print(f"Multimodal mod min: {mod_max_relative_change}")
                if mod_max_relative_change is None:
                    valid = {mod: val for mod, val in average_modality_le_unimodal.items() if
                             val is not None and modality_answers_given["unimodal_answers"].get(mod)}

                    # Maybe here we have to break the rules
                    if valid:
                        min_mod = min(valid, key=valid.get)
                        print(f"Unimodal mod min: {min_mod}")
                        return "unimodal_partitions", min_mod
                    else:

                        valid_multimodal = {mod: val for mod, val in average_modality_le_multimodal.items() if
                                            val is not None and modality_answers_given["multimodal_answers"].get(mod)}

                        if valid_multimodal:
                            min_mod = min(valid_multimodal, key=valid_multimodal.get)
                            print(f"Multimodal rule broken min: {min_mod}")
                            return "multimodal_partitions", min_mod
                        else:
                            return None, None

                print(f"Modality with max relative change and relative change: {mod_max_relative_change}, "
                      f"{relative_changes[mod_max_relative_change]}")

                return "multimodal_partitions", mod_max_relative_change

            else:
                relative_changes = {}
                for mod in average_modality_le_unimodal.keys():
                    if average_modality_le_unimodal[mod] and average_modality_le_multimodal[mod]:
                        relative_change = LEAgent.relative_entropy_change(average_modality_le_unimodal[mod],
                                                                          average_modality_le_multimodal[mod])

                        if relative_change >= 0:
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
                        if modality_answers_given["multimodal_answers"][mod]:
                            mod_max_relative_change = mod
                            break

                    if mod_max_relative_change is None:
                        valid = {mod: val for mod, val in average_modality_le_unimodal.items() if
                                 val is not None and modality_answers_given["unimodal_answers"].get(mod)}

                        if valid:
                            min_mod = min(valid, key=valid.get)
                            print(f"Unimodal mod min: {min_mod}")
                            return "unimodal_partitions", min_mod
                        else:
                            valid_multimodal = {mod: val for mod, val in average_modality_le_multimodal.items() if
                                                val is not None and modality_answers_given["multimodal_answers"].get(
                                                    mod)}

                            if valid_multimodal:
                                min_mod = min(valid_multimodal, key=valid_multimodal.get)
                                print(f"Multimodal rule broken min: {min_mod}")
                                return "multimodal_partitions", min_mod
                            else:
                                return None, None

                    print(f"Modality with max relative change and relative change: {mod_max_relative_change}, "
                          f"{relative_changes[mod_max_relative_change]}")

                    return "multimodal_partitions", mod_max_relative_change

                else:
                    valid = {mod: val for mod, val in average_modality_le_unimodal.items() if
                             val is not None and modality_answers_given["unimodal_answers"].get(mod)}

                    if valid:
                        min_mod = min(valid, key=valid.get)
                        print(f"Unimodal mod min: {min_mod}")
                        return "unimodal_partitions", min_mod
                    else:
                        valid_multimodal = {mod: val for mod, val in average_modality_le_multimodal.items() if
                                            val is not None and modality_answers_given["multimodal_answers"].get(mod)}
                        if valid_multimodal:
                            min_mod = min(valid_multimodal, key=valid_multimodal.get)
                            print(f"Multimodal rule broken min: {min_mod}")
                            return "multimodal_partitions", min_mod
                        else:
                            return None, None

    @staticmethod
    def find_final_answer_boolean_table(model, question, question_text, partitions_path,
                                        election_modality, answers_dir, dataset, approach, setting):

        print(f"Final answer to be found in table (boolean question): {question_text}")
        modality, partitions, min_le_filling, final_answer = LEAgent.extract_partitions(model,
                                                                                        question,
                                                                                        "table",
                                                                                        question_text,
                                                                                        partitions_path,
                                                                                        dataset,
                                                                                        approach,
                                                                                        setting)

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
                                                  election_modality, answers_dir, answer_class_specific, num_elements,
                                                  dataset, approach, setting):

        modality, partitions, min_le_filling, final_answer = LEAgent.extract_partitions(model, question,
                                                                                        "table",
                                                                                        question_text,
                                                                                        partitions_path,
                                                                                        dataset,
                                                                                        approach,
                                                                                        setting)

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
                                                      answer_class_specific, dataset, approach, setting):

        unimodal_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        unimodal_partitions = json.load(open(unimodal_partitions_path, "rb"))
        multimodal_partitions = self.create_multimodal_partitions(model, question, question_text, dataset, approach,
                                                                  setting)

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
                            answer_class_general, table_dir, final_dataset_images, answers_dir, dataset, approach,
                            setting):

        final_answer = None

        unimodal_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}")
        multimodal_partitions_path = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")

        # First we create the partitions, both in the unimodal case and also in the multi-hop unimodal case
        for modality in priority_modalities:
            print(f"CREATING UNIMODAL PARTITIONS for {modality.upper()}")
            print("************************************************************************")
            self.create_unimodal_partitions(model, question, question_files, table_dir, modality, criterias, dataset,
                                            approach, setting)
            print("************************************************************************")
            print(f"END CREATION")
            print("************************************************************************\n")

        # In case the predicted modality is table and the answer_class is boolean then we do not start the entire
        # analysis
        if len(priority_modalities) == 1 and priority_modalities[0] == "table":
            if answer_class_general == "boolean":
                print("WE do not even create the multi-hop. We just go with a single table")
                LEAgent.find_final_answer_boolean_table(model, question, rewritten_question_text,
                                                        "unimodal_partitions", priority_modalities[0],
                                                        answers_dir, dataset, approach, setting)

                return

            else:
                is_comparison, num_elements, confidence = self.iscomparison(model, question_text)
                if is_comparison:
                    print(f"num elements is {num_elements}")
                    print("comparison mood")
                    self.find_final_answer_boolean_two_steps_table(model, question, question_text,
                                                                   "unimodal_partitions", priority_modalities[0],
                                                                   answers_dir, answer_class_specific, num_elements,
                                                                   dataset, approach, setting)

                    return

        is_comparison, num_elements, confidence = self.iscomparison(model, question_text)
        print(f"Do we need a comparison? {is_comparison}, num_elements: {num_elements}")

        # We check whether a comparison is needed
        if is_comparison:
            # We check how many modalities have been selected
            if len(priority_modalities) == 2:
                # We check if analysing images is needed
                is_graphical, confidence = self.isgraphical(model, question_text)
                # If is_graphical is true, and we have not yet inserted images in the modalities then we need to do
                # it, and we can proceed.
                print(f"Do we need a graphical? {is_graphical}")
                if is_graphical and 'image' not in priority_modalities:
                    self.create_unimodal_partitions(model, question, question_files, table_dir, "image", criterias,
                                                    dataset, approach, setting)
                    self.find_final_answer_comparison_three_modalities(model, question, rewritten_question_text,
                                                                       answers_dir, answer_class_specific, dataset,
                                                                       approach, setting)

        multimodal = len(priority_modalities) > 1

        # This dictionary keeps whether we gave answers with a certain modality in unimodal or multimodal setting
        # Moreover, it also keeps track of whether there were any changes in the unimodal average entropy and
        # multimodal average entropy
        modality_answers_given = {"unimodal_answers": {"text": True, "table": True, "image": True},
                                  "multimodal_answers": {"text": True, "table": True, "image": True},
                                  "unimodal_entropies": [],
                                  "multimodal_entropies": []}

        if multimodal:
            print("***************************************************************")
            print("WE START A MULTIMODAL ANALYSIS")
            print("***************************************************************")
            # Here we create the multimodal partitions
            multimodal_partitions_path = os.path.join(
                f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}")

            # if not os.path.exists(multimodal_partitions_path):
            print("************************************************************************")
            print(f"CREATING MULTIMODAL PARTITIONS")
            multimodal_partitions = self.create_multimodal_partitions(model, question, rewritten_question_text, dataset,
                                                                      approach, setting)

            json.dump(multimodal_partitions,
                      open(
                          f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}",
                          "w"), indent=4)
            print("************************************************************************\n")

        print(f"Final answer: {final_answer}")
        print(modality_answers_given["unimodal_answers"].values())
        print(modality_answers_given["multimodal_answers"].values())

        # This is the variable that will allow to know what happens at each step
        steps = {}

        i = 0
        while (final_answer is None and (any(modality_answers_given["unimodal_answers"].values()) or
                                         any(modality_answers_given["multimodal_answers"].values()))):

            print(f"Modality answers given: {modality_answers_given}")
            partitions_path, mod_chosen = LEAgent.choose_unimodal_multimodal(is_comparison, unimodal_partitions_path,
                                                                             multimodal_partitions_path,
                                                                             modality_answers_given)
            print(f"Modality answers given: {modality_answers_given}")

            if partitions_path is None:
                break

            print(f"Partitions path {partitions_path}")
            print("***************************************************************")
            print("Extracting the partitions")
            modality, partitions, min_le_filling, final_answer = LEAgent.extract_partitions(model,
                                                                                            question,
                                                                                            mod_chosen,
                                                                                            rewritten_question_text,
                                                                                            partitions_path,
                                                                                            dataset,
                                                                                            approach,
                                                                                            setting)

            print(f"The mod chosen is: {modality}")

            if "unimodal" in partitions_path:
                modality_answers_given["unimodal_answers"][modality] = False
            else:
                modality_answers_given["multimodal_answers"][modality] = False

            json.dump(modality_answers_given,
                      open(f"../../results/{dataset}/{approach}/{setting}/entropy_evolution/{model}/{question}", "w"),
                      indent=4)

            print(f"Modality answers given are now this: {modality_answers_given}")
            print("***************************************************************")

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

                steps[i] = {"unimodal_multimodal": partitions_path, "modality_chosen": mod_chosen,
                            "answer": final_answer}

                json.dump(steps,
                          open(f"../../results/{dataset}/{approach}/{setting}/steps/{model}/{question}", "w"),
                          indent=4)

                i += 1

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

                    modalities_json = json.load(
                        open(f"../../results/{dataset}/{approach}/{setting}/modalities_predicted/{model}/{question}",
                             "rb"))
                    max_step = max([int(step.split('_')[1]) for step in modalities_json.keys()])
                    modalities_json[f'step_{max_step + 1}'] = decided_modality

                    json.dump(modalities_json, open(
                        f"../../results/{dataset}/{approach}/{setting}/modalities_predicted/{model}/{question}", "w"),
                              indent=4)

                    remaining_modalities.remove(decided_modality)
                    print("************************************************************************")
                    print(f"CREATING UNIMODAL PARTITIONS for {decided_modality.upper()}")
                    print("************************************************************************\n")
                    self.create_unimodal_partitions(model, question, question_files, table_dir, decided_modality,
                                                    criterias, dataset, approach, setting)

                    print("And now we create multimodal partitions.")
                    multimodal_partitions = self.create_multimodal_partitions(model, question, question_text, dataset,
                                                                              approach, setting)

                    json.dump(multimodal_partitions, open(
                        f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}",
                        "w"),
                              indent=4)

                    unimodal_partitions = json.load(
                        open(
                            f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/partitions_{question}",
                            "rb"))
                    multimodal_partitions = json.load(
                        open(
                            f"../../results/{dataset}/{approach}/{setting}/partitions/multimodal_partitions/{iteration}/{model}/partitions_{question}",
                            "rb"))

                    average_modality_le_unimodal = average_modality_le(unimodal_partitions)
                    average_modality_le_multimodal = average_modality_le(multimodal_partitions)

                    LEAgent.calculate_entropy_evolution(average_modality_le_unimodal,
                                                        average_modality_le_multimodal,
                                                        modality_answers_given)

                    if not remaining_modalities and is_comparison:
                        i += 1
                        final_answer = self.find_final_answer_comparison_three_modalities(model, question,
                                                                                          rewritten_question_text,
                                                                                          answers_dir,
                                                                                          answer_class_specific,
                                                                                          dataset,
                                                                                          approach,
                                                                                          setting)

                        modality_answers_given["multimodal_answers"]["table"] = False

                        steps[i] = {"unimodal_multimodal": "multimodal_partitions",
                                    "modality_chosen": "table",
                                    "answer": final_answer}

                        json.dump(steps,
                                  open(f"../../results/{dataset}/{approach}/{setting}/steps/{model}/{question}", "w"),
                                  indent=4)

                        json.dump(modality_answers_given,
                                  open(
                                      f"../../results/{dataset}/{approach}/{setting}/entropy_evolution/{model}/{question}",
                                      "w"),
                                  indent=4)

    def answer_question(self, model, question, question_data, question_files, table_dir, final_dataset_images,
                        answers_dir, dataset, approach, setting):
        """Method which calls the other methods to calculate the final answer."""

        # START MODALITY SELECTION
        all_modalities = ["image", "table", "text"]

        # Predict the modalities that must be used and create the file that contains them (if it does not exist)
        modalities_json = os.path.join(
            f"../../results/{dataset}/{approach}/{setting}/modalities_predicted/{model}/{question}")

        print(f"Modalities json: {modalities_json}")

        if os.path.exists(modalities_json):
            print("************************************")
            print("Modality file already exists? YES")
            print("************************************\n")
            priority_modalities = json.load(
                open(f"../../results/{dataset}/{approach}/{setting}/modalities_predicted/{model}/{question}", "rb"))
            priority_modalities = priority_modalities["step_1"]
        else:
            print("*************************************************")
            print("Modality file already exists? NO. We create it")
            print("*************************************************\n")
            priority_modalities = LEAgent.decide_modality_llm(model,
                                                              self.openai_client,
                                                              self.bedrock_client,
                                                              question_data["question_text"],
                                                              question_files["image_set"],
                                                              question_files["text_set"],
                                                              question_files["table_set"][0],
                                                              table_dir, final_dataset_images).split('_')

            json.dump({"step_1": priority_modalities},
                      open(os.path.join(
                          f"../../results/{dataset}/{approach}/{setting}/modalities_predicted/{model}/{question}"),
                          "w"), indent=4)

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
                                 answer_class_general, table_dir, final_dataset_images, answers_dir, dataset, approach,
                                 setting)
        # END ANSWER TO QUESTION


def answer_qa(model, agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images,
              answers_dir, dataset, approach, setting):
    os.makedirs(answers_dir, exist_ok=True)
    os.makedirs(f"../../results/{dataset}/{approach}/{setting}/partitions/unimodal_partitions/{iteration}/{model}/",
                exist_ok=True)

    for i, question in enumerate(questions_list):
        json_question = json.load(open(os.path.join(questions_dir, question), "rb"))
        modalities = json_question["metadata"]["modalities"]

        if len(modalities) == 1:
            unimodal_multimodal = "/unimodal/"
        else:
            unimodal_multimodal = "/multimodal/"

        # if question not in os.listdir(answers_dir + unimodal_multimodal):
        print("\nQUESTION JSON - QUESTION TEXT - TARGET MODALITIES")
        print("*************************************************************************************")
        print(f"Question number: {i}")
        print(f"Question json: {question}")
        print(f"Question text: {json_question['question']}")
        print(f"Target modalities: {modalities}")
        print("*************************************************************************************\n")
        question_data = get_question_data(questions_dir, question)
        question_files = get_question_files(association_dir, question)

        agent.answer_question(model, question, question_data, question_files, table_dir, final_dataset_images,
                              answers_dir + unimodal_multimodal, dataset, approach, setting)


def entropy_calculation_main(model, agent, questions_list, questions_dir, association_dir, table_dir,
                             final_dataset_images, answers_dir, dataset, approach, setting):
    # create_association_qa(agent, questions_dir, association_dir, image_dir, text_dir, table_dir)

    answer_qa(model, agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images,
              answers_dir, dataset, approach, setting)


if __name__ == '__main__':
    if os.path.exists(f"../../results/multimodalqa/le/training/modalities_predicted/gpt-5.2/question_9312.json"):
        print("Cazzo")
