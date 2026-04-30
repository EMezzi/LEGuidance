import base64
import re
import os
import json
from utils.utilities import encode_image, get_image_format
from schemas.pydantic_schemas import *
import random
from schemas.tools import (modality_tool, yes_no_tool, is_comparison_tool, is_graphical_tool,
                           analyse_criteria_tool,
                           table_description_tool,
                           paragraph_extraction_tool, image_extraction_tool, table_row_extraction_tool,
                           image_contains_answer_tool, paragraph_contains_answer_tool, row_contains_answer_tool)


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                   FUNCTIONS FOR MODALITY DECISION                          │
# └────────────────────────────────────────────────────────────────────────────┘

def decide_modality_llm_amazon(model, bedrock_client, system_prompt_modality, user_prompt_modality, question_text,
                               images_text, images_inputs, paragraphs_text, tables_text, use_tool=True):
    try:
        content_list = []

        # 1. Standardized Image Preprocessing
        for img in images_inputs:
            raw_b64 = img['image_url']
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",")[1]

            image_bytes = base64.b64decode(raw_b64)
            # Standardize format: Bedrock Converse supports png, jpeg, gif, webp
            # img_format = img.get('image_ext', 'png').lower().replace('jpg', 'jpeg')

            content_list.append({
                "image": {
                    "format": img["image_ext"],
                    "source": {"bytes": image_bytes}
                }
            })

        # 2. Unified Text Prompt
        # Standardized to use 'question' - ensure your templates match this key
        formatted_text = user_prompt_modality.format(
            question=question_text,
            images_text=images_text,
            paragraphs_text=paragraphs_text,
            tables_text=tables_text
        )

        # Add JSON instructions if not using a tool (NVIDIA use-case)
        if not use_tool:
            allowed = "image | text | table | image_text | image_table | text_table"
            formatted_text += f"""

            ### Allowed Modalities:
            You MUST choose exactly one from this list: [{allowed}]

            ### Response Format:
            {{ "modalities": "selected_value" }}
            """

            system_prompt_modality += "\nRespond ONLY with a raw JSON object: {\"modalities\": string}"

        content_list.append({"text": formatted_text})

        # 3. Parameters for Converse API
        params = {
            "modelId": model,
            "messages": [{"role": "user", "content": content_list}],
            "system": [{"text": system_prompt_modality}],
            "inferenceConfig": {"temperature": 0, "maxTokens": 1000}
        }

        # Add Tool Config only if requested
        if use_tool:
            params["toolConfig"] = {
                "tools": [modality_tool],
                "toolChoice": {"tool": {"name": "ModalityDecision"}}
            }

        # 4. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        for block in output_message.get('content', []):
            if 'toolUse' in block:
                tool_input = block['toolUse'].get('input')
                break

            if 'text' in block:
                text_content = block['text'].strip()
                if not text_content:
                    continue

                # --- NEW: Nova XML Fallback ---
                # Checks for <__parameter=modalities>VALUE</__parameter>
                if "<__parameter=modalities>" in text_content:
                    import re
                    match = re.search(r'<__parameter=modalities>(.*?)</__parameter>', text_content)
                    if match:
                        tool_input = {"modalities": match.group(1).strip()}
                        break

                # --- Standard JSON Fallback (NVIDIA/Mistral) ---
                try:
                    # Clean the text
                    clean_text = text_content.strip('`').replace('json', '', 1).strip()

                    try:
                        # Try parsing as JSON
                        data = json.loads(clean_text)
                        # If JSON is empty or not a dict, fallback to raw text
                        if not data or not isinstance(data, dict):
                            tool_input = {"modalities": clean_text}
                        else:
                            tool_input = data
                    except json.JSONDecodeError:
                        # If parsing fails, fallback to raw text
                        print("JSON parse failed, using raw text")
                        tool_input = {"modalities": clean_text}

                    print(f"Tool input: {tool_input}")
                    break  # exit the loop after first valid content

                except Exception as e:
                    print(f"Unexpected error in text block parsing: {e}")

        print(tool_input)
        if 'modalities' in tool_input:
            if tool_input['modalities'] == 'image_text_table':
                print("Exception case")
                tool_input['modalities'] = random.choice(['image_text'])

        # 6. Validation and Normalization
        if tool_input:
            try:
                validated = ModalityDecision(**tool_input)
                return validated.modalities
            except Exception as e:
                print(f"Modality Validation Error: {e}")
                return "text"

        else:
            return "text"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "text"


def decide_modality_llm_gpt(model, openai_client, system_prompt_modality, user_prompt_modality, question, images_text,
                            images_inputs, paragraphs_text, tables_text):
    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt_modality
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt_modality.format(question=question,
                                                            images_text=images_text,
                                                            paragraphs_text=paragraphs_text,
                                                            tables_text=tables_text)
                    },
                    *[
                        {"type": "input_image", "image_url": f"data:image/{img['image_ext']};base64,{img['image_url']}"}
                        for img in images_inputs
                    ]
                ]
            }
        ],
        text_format=ModalityDecision,
    )

    return response.output_parsed.modalities


# ┌────────────────────────────────────────────────────────────────────────────┐
# │              FUNCTIONS FOR MODALITY DECISION: REDUCED DATA                 │
# └────────────────────────────────────────────────────────────────────────────┘

def decide_modality_reduced_data_amazon(model, bedrock_client, system_prompt_reduced_modality,
                                        user_prompt_reduced_modality, question, available_content, remaining_modalities,
                                        images_inputs, use_tool=True):
    try:
        content_list = []

        # 1. Standardized Image Preprocessing
        for img in images_inputs:
            raw_b64 = img['image_url']
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",")[1]

            image_bytes = base64.b64decode(raw_b64)
            # Standardize format: Bedrock Converse supports png, jpeg, gif, webp
            # img_format = img.get('image_ext', 'png').lower().replace('jpg', 'jpeg')

            content_list.append({
                "image": {
                    "format": img["image_ext"],
                    "source": {"bytes": image_bytes}
                }
            })

        # 2. Unified Text Prompt
        # Standardized to use 'question' - ensure your templates match this key
        formatted_text = user_prompt_reduced_modality.format(question=question, available_content=available_content)
        system_prompt_reduced_modality.format(remaining_modalities=remaining_modalities)

        # Add JSON instructions if not using a tool (NVIDIA use-case)
        if not use_tool:
            formatted_text += """
            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "modalities": string
            }
            """

            system_prompt_reduced_modality += "\nRespond ONLY with a raw JSON object: {\"modalities\": string}"

        content_list.append({"text": formatted_text})

        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

        # 3. Parameters for Converse API
        params = {
            "modelId": model,
            "messages": [{"role": "user", "content": content_list}],
            "system": [{"text": system_prompt_reduced_modality}],
            "inferenceConfig": inference_config
        }

        # Add Tool Config only if requested
        if use_tool:
            params["toolConfig"] = {
                "tools": [modality_tool],
                "toolChoice": {"tool": {"name": "ModalityDecision"}}
            }

        # 4. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        for block in output_message.get('content', []):
            if 'toolUse' in block:
                tool_input = block['toolUse'].get('input')
                break

            if 'text' in block:
                text_content = block['text'].strip()
                if not text_content:
                    continue

                # --- NEW: Nova XML Fallback ---
                # Checks for <__parameter=modalities>VALUE</__parameter>
                if "<__parameter=modalities>" in text_content:
                    import re
                    match = re.search(r'<__parameter=modalities>(.*?)</__parameter>', text_content)
                    if match:
                        tool_input = {"modalities": match.group(1).strip()}
                        break

                # --- Standard JSON Fallback (NVIDIA/Mistral) ---
                try:
                    # Clean the text
                    clean_text = text_content.strip('`').replace('json', '', 1).strip()

                    try:
                        # Try parsing as JSON
                        data = json.loads(clean_text)
                        # If JSON is empty or not a dict, fallback to raw text
                        if not data or not isinstance(data, dict):
                            tool_input = {"modalities": clean_text}
                        else:
                            tool_input = data
                    except json.JSONDecodeError:
                        # If parsing fails, fallback to raw text
                        print("JSON parse failed, using raw text")
                        tool_input = {"modalities": clean_text}

                    print(f"Tool input: {tool_input}")
                    break  # exit the loop after first valid content

                except Exception as e:
                    print(f"Unexpected error in text block parsing: {e}")

        # 6. Validation and Normalization
        if tool_input:
            try:
                validated = ModalityDecision(**tool_input)
                return validated.modalities
            except Exception as e:
                print(f"Modality Validation Error: {e}")
                return None

        print(f"No valid modality JSON found for {model}")
        return None

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return None


def decide_modality_reduced_data_gpt(model, openai_client, system_prompt_reduced_modality, user_prompt_reduced_modality,
                                     question, available_content, remaining_modalities, images_inputs):
    response = openai_client.responses.parse(
        model=model,
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
                            {"type": "input_image",
                             "image_url": f"data:image/{img['image_ext']};base64,{img['image_url']}"}
                            for img in images_inputs
                        ]
                        if images_inputs is not None else []
                    )
                ]
            }
        ],
        text_format=ModalityDecision,
    )

    return response.output_parsed.modalities


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                    CHECK IF IT IS A YES/NO QUESTION                        │
# ├────────────────────────────────────────────────────────────────────────────┤
# │  Validates: Binary Classification (True/False)                             │
# │  Extracts: Confidence Score [0.0 - 1.0]                                    │
# └────────────────────────────────────────────────────────────────────────────┘

def yesnoquestion_amazon(model, bedrock_client, system_prompt_bool_question, question_text, use_tool=True):
    try:
        # 1. Prepare User Content
        user_content = f"QUESTION: {question_text}"

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - is_yes_no: boolean (true if the question is a binary yes/no question)
            - confidence: number (float between 0.0 and 1.0)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "is_yes_no": boolean,
                "confidence": number
            }
            """
            user_content += json_instruction

            # Sync System Prompt to match the tool's required fields
            system_prompt_bool_question += "\nRespond ONLY with a raw JSON object: {\"is_yes_no\": boolean, \"confidence\": number}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": user_content}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_bool_question}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [yes_no_tool],
                "toolChoice": {"tool": {"name": "YesNoQuestion"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Handles <__parameter=is_yes_no> and <__parameter=confidence>
                    if "<__parameter=" in text_content:
                        bool_match = re.search(r'<__parameter=is_yes_no>(.*?)</__parameter>', text_content)
                        conf_match = re.search(r'<__parameter=confidence>(.*?)</__parameter>', text_content)

                        if bool_match:
                            tool_input = {
                                "is_yes_no": bool_match.group(1).lower() == "true",
                                "confidence": float(conf_match.group(1)) if conf_match else 1.0
                            }
                            break

                    # Standard JSON cleaning (Mistral/NVIDIA)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        if tool_input:
            try:
                # Pydantic hydration for strict typing
                validated = YesNoQuestion(**tool_input)
                return validated.is_yes_no, validated.confidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fallback coercion
                is_yn = tool_input.get("is_yes_no", False)
                if isinstance(is_yn, str): is_yn = is_yn.lower() == "true"
                return is_yn, float(tool_input.get("confidence", 0.0))

        else:
            return False, 1.00

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return False, 1.00


def yesnoquestion_gpt(model, openai_client, system_prompt_bool_question, question_text):
    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt_bool_question
            },
            {
                "role": "user",
                "content": f"""QUESTION: {question_text}"""
            }
        ],
        text_format=YesNoQuestion,
    )

    found = response.output_parsed
    return found.is_yes_no, found.confidence


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                   CHECK IF IT IS A COMPARISON QUESTION                     │
# └────────────────────────────────────────────────────────────────────────────┘

def iscomparison_amazon(model, bedrock_client, system_prompt_comparison_question, question_text, use_tool=True):
    try:
        # 1. Prepare User Content
        user_content = f"QUESTION: {question_text}"

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - is_comparison: boolean (true if the question asks to compare entities)
            - num_elements: integer (the number of entities to be compared, 0 if none)
            - confidence: number (float between 0 and 1)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "is_comparison": boolean,
                "num_elements": integer,
                "confidence": number
            }
            """
            user_content += json_instruction

            # Sync System Prompt to enforce the full schema
            system_prompt_comparison_question += (
                "\nRespond ONLY with a raw JSON object: "
                "{\"is_comparison\": boolean, \"num_elements\": integer, \"confidence\": number}"
            )

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": user_content}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_comparison_question}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [is_comparison_tool],
                "toolChoice": {"tool": {"name": "IsComparison"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse (Modern Bedrock behavior)
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                # Standard Tool Use block
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                # Text block (Handles Nova XML or raw JSON fallbacks)
                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Extracts parameters from tags like <__parameter=is_comparison>
                    if "<__parameter=" in text_content:
                        try:
                            # We look for all three required fields in Nova's tag format
                            is_comp_match = re.search(r'<__parameter=is_comparison>(.*?)</__parameter>', text_content)
                            num_match = re.search(r'<__parameter=num_elements>(.*?)</__parameter>', text_content)
                            conf_match = re.search(r'<__parameter=confidence>(.*?)</__parameter>', text_content)

                            if is_comp_match:
                                tool_input = {
                                    "is_comparison": is_comp_match.group(1).lower() == "true",
                                    "num_elements": int(num_match.group(1)) if num_match else 0,
                                    "confidence": float(conf_match.group(1)) if conf_match else 1.0
                                }
                                break
                        except:
                            pass

                    # Standard JSON cleaning (Handles Mistral/NVIDIA/Claude markdown)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        if tool_input:
            try:
                # Hydrate into Pydantic to ensure types (bool, int, float) are correct
                validated = IsComparison(**tool_input)
                return validated.is_comparison, validated.num_elements, validated.confidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                return False, 0, 1.00

        else:
            return False, 0, 1.00

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return False, 0, 1.00


def iscomparison_gpt(model, openai_client, system_prompt_comparison_question, question_text):
    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt_comparison_question
            },
            {
                "role": "user",
                "content": f"""QUESTION: {question_text}"""
            }
        ],
        text_format=IsComparison,
    )

    found = response.output_parsed
    return found.is_comparison, found.num_elements, found.confidence


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                   CHECK IF IT IS A GRAPHICAL QUESTION                      │
# └────────────────────────────────────────────────────────────────────────────┘

def isgraphical_amazon(model, bedrock_client, system_prompt_isgraphical_question, question_text, use_tool=True):
    try:
        # 1. Prepare User Content
        user_content = f"QUESTION: {question_text}"

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - is_graphical: boolean (true if you need to analyse a graphical element to answer a question)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "is_graphical": boolean
            }
            """
            user_content += json_instruction

            # Sync System Prompt to enforce JSON schema
            system_prompt_isgraphical_question += "\nRespond ONLY with a raw JSON object: {\"is_graphical\": boolean}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": user_content}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_isgraphical_question}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [is_graphical_tool],
                "toolChoice": {"tool": {"name": "IsGraphical"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Handles <__parameter=is_graphical> and <__parameter=confidence>
                    if "<__parameter=" in text_content:
                        try:
                            bool_match = re.search(r'<__parameter=is_graphical>(.*?)</__parameter>', text_content)
                            conf_match = re.search(r'<__parameter=confidence>(.*?)</__parameter>', text_content)

                            if bool_match:
                                tool_input = {
                                    "is_graphical": bool_match.group(1).lower().strip() == "true",
                                    "confidence": float(conf_match.group(1)) if conf_match else 1.0
                                }
                                break
                        except:
                            pass

                    # Standard JSON cleaning (Mistral/NVIDIA/Claude markdown)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        if tool_input:
            try:
                # Hydrate into Pydantic to ensure boolean and float types are strictly enforced
                validated = IsGraphical(**tool_input)
                return validated.is_graphical, validated.confidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fallback normalization
                is_g = tool_input.get("is_graphical", False)
                if isinstance(is_g, str): is_g = is_g.lower().strip() == "true"
                return bool(is_g), float(tool_input.get("confidence", 0.0))

        else:
            return False, 1.00

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return False, 1.00


def isgraphical_gpt(model, openai_client, system_prompt_isgraphical_question, question_text):
    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt_isgraphical_question
            },
            {
                "role": "user",
                "content": f"""QUESTION: {question_text}"""
            }
        ],
        text_format=IsGraphical,
    )

    found = response.output_parsed
    return found.is_graphical, found.confidence


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                   ANALYSIS OF IMAGE: CHECK CRITERIA                        │
# └────────────────────────────────────────────────────────────────────────────┘


def analyse_image_criteria_amazon(dataset, model, bedrock_client, system_prompt_image, user_prompt_image_text, criteria,
                                  metadata, image_path, use_tool=True):
    try:
        content_list = []

        image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                            image_path))

        # Strip the data URI header if it exists
        raw_b64 = image64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        image_bytes = base64.b64decode(raw_b64)

        ext = get_image_format(image_path, image_bytes)

        content_list.append({
            "image": {
                "format": ext,
                "source": {"bytes": image_bytes}
            }
        })

        # 2. Unified Text Prompt
        # Standardized to use 'question' - ensure your templates match this key
        formatted_text = user_prompt_image_text.format(metadata=metadata, criteria=criteria)

        # Add JSON instructions if not using a tool (NVIDIA use-case)
        if not use_tool:
            formatted_text += """
            ### Answer by returning: 
            - answer: string (answer with only "yes" or "no" to whether the data contains the criteria)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "answer": string
            }
            """

            system_prompt_image += "\nRespond ONLY with a raw JSON object: {\"answer\": string}"

        content_list.append({"text": formatted_text})

        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

        # 3. Parameters for Converse API
        params = {
            "modelId": model,
            "messages": [{"role": "user", "content": content_list}],
            "system": [{"text": system_prompt_image}],
            "inferenceConfig": inference_config
        }

        # Add Tool Config only if requested
        if use_tool:
            params["toolConfig"] = {
                "tools": [analyse_criteria_tool],
                "toolChoice": {"tool": {"name": "AnswerContainsCriteria"}}
            }

        # 4. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content:
                        continue

                    # Nova XML Shield (Specific to your Nova problem)
                    if "<__parameter=answer>" in text_content:
                        match = re.search(r'<__parameter=answer>(.*?)</__parameter>', text_content)
                        if match:
                            tool_input = {"answer": match.group(1).strip()}
                            break

                    # Standard JSON cleaning
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Try parsing as JSON
                            data = json.loads(clean_text)
                            if not data or not isinstance(data, dict):
                                tool_input = {"answer": clean_text}
                            else:
                                tool_input = data
                        except json.JSONDecodeError:
                            # If parsing fails, fallback to dict with raw text
                            print("JSON parse failed, using raw text")
                            tool_input = {"answer": clean_text}

                        break  # exit the block iteration

                    except Exception as inner_e:
                        print(f"Unexpected inner error: {inner_e}")

        print("Tool input is: ", tool_input)
        # 6. Validation and Return
        if tool_input:
            try:
                validated = AnswerContainsCriteria(**tool_input)
                return validated.answer
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Optional: return a raw guess if pydantic fails
                raw_ans = str(tool_input.get('answer', '')).lower()
                return "yes" if "yes" in raw_ans else "no"
        else:
            return "yes"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "yes"


def analyse_image_criteria_gpt(dataset, model, openai_client,
                               system_prompt_image, user_prompt_image_text, user_prompt_image_image, criteria, metadata,
                               image_path):

    image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                        image_path))

    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                   ANALYSIS OF TEXT: CHECK CRITERIA                         │
# └────────────────────────────────────────────────────────────────────────────┘

def analyse_text_criteria_amazon(model, bedrock_client, system_prompt_text, user_prompt_text, criteria,
                                 metadata, text, use_tool=True):
    try:
        # 1. Format the User Prompt
        formatted_user_text = user_prompt_text.format(
            metadata=metadata,
            text=text,
            criteria=criteria
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            formatted_user_text += """
            ### Answer by returning: 
            - answer: string (answer with only "yes" or "no" to whether the data contains the criteria)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "answer": string
            }
            """

            system_prompt_text += "\nRespond ONLY with a raw JSON object: {\"answer\": string}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        # Increased stability for NVIDIA/Mistral models
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_text}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [analyse_criteria_tool],
                "toolChoice": {"tool": {"name": "AnswerContainsCriteria"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield (Specific to your Nova problem)
                    if "<__parameter=answer>" in text_content:
                        match = re.search(r'<__parameter=answer>(.*?)</__parameter>', text_content)
                        if match:
                            tool_input = {"answer": match.group(1).strip()}
                            break

                    # Standard JSON cleaning
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Try parsing as JSON
                            data = json.loads(clean_text)
                            if not data or not isinstance(data, dict):
                                tool_input = {"answer": clean_text}
                            else:
                                tool_input = data
                        except json.JSONDecodeError:
                            # If parsing fails, fallback to dict with raw text
                            print("JSON parse failed, using raw text")
                            tool_input = {"answer": clean_text}

                        break  # exit the block iteration

                    except Exception as inner_e:
                        print(f"Unexpected inner error: {inner_e}")

        print(f"The tool is: {tool_input}")
        # 6. Validation and Return
        if tool_input:
            try:
                validated = AnswerContainsCriteria(**tool_input)
                return validated.answer
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Optional: return a raw guess if pydantic fails
                raw_ans = str(tool_input.get('answer', '')).lower()
                return "yes" if "yes" in raw_ans else "no"

        else:
            return "yes"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "yes"


def analyse_text_criteria_gpt(model, openai_client, system_prompt_text, user_prompt_text, criteria, metadata, text):
    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                   ANALYSIS OF TABLE: CHECK CRITERIA                        │
# └────────────────────────────────────────────────────────────────────────────┘

def table_general_understanding_amazon(model, bedrock_client, system_prompt_table, user_prompt_table,
                                       table_title, table_name, columns, use_tool=True):
    try:
        # 1. Prepare User Content
        user_content = user_prompt_table.format(
            table_title=table_title,
            table_name=table_name,
            columns=columns
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - description: string (a high-level, accurate summary of what this database table represents based on its title and columns)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "description": string
            }
            """
            user_content += json_instruction

            # Sync System Prompt to enforce the single-key JSON schema
            system_prompt_table += "\nRespond ONLY with a raw JSON object: {\"description\": string}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": user_content}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_table}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [table_description_tool],
                "toolChoice": {"tool": {"name": "TableDescription"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Extracts the description from the custom tag
                    if "<__parameter=description>" in text_content:
                        match = re.search(r'<__parameter=description>(.*?)</__parameter>', text_content, re.DOTALL)
                        if match:
                            tool_input = {"description": match.group(1).strip()}
                            break

                    # Standard JSON cleaning (Mistral/NVIDIA/Claude)
                    try:
                        # Clean the text
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Attempt to parse as JSON
                            data = json.loads(clean_text)
                            # If parsing succeeds but the result is empty or not a dict, fallback to raw text
                            if not data or not isinstance(data, dict):
                                tool_input = {"description": clean_text}
                            else:
                                tool_input = data
                        except json.JSONDecodeError:
                            # If parsing fails, use raw text as fallback
                            print("JSON parse failed, using raw text")
                            tool_input = {"description": clean_text}

                        break  # exit loop after first valid content

                    except Exception as e:
                        print(f"Unexpected error in text block parsing: {e}")

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                # Validate against Pydantic schema
                validated = TableDescription(**tool_input)
                return validated.description
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Fallback to returning whatever string we managed to find
                return tool_input.get("description", str(tool_input))

        else:
            return None

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return None


def table_general_understanding_gtp(model, openai_client, system_prompt_table, user_prompt_table, table_title,
                                    table_name, columns):
    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt_table
            },
            {
                "role": "user",
                "content": user_prompt_table.format(table_title=table_title, table_name=table_name, columns=columns)
            }
        ],
        text_format=TableDescription,
    )

    found = response.output_parsed
    return found.description


def analyse_table_row_criteria_amazon(model, bedrock_client, system_prompt_row, user_prompt_row,
                                      row, criteria, use_tool=True):
    try:
        # 1. Format the User Prompt
        formatted_user_text = user_prompt_row.format(
            row=row,
            criteria=criteria
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - answer: string (answer with only "yes" or "no" to whether the row data contains the criteria)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "answer": string
            }
            """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce JSON schema
            system_prompt_row += "\nRespond ONLY with a raw JSON object: {\"answer\": string}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_row}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [analyse_criteria_tool],
                "toolChoice": {"tool": {"name": "AnswerContainsCriteria"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield (Specific to your Nova problem)
                    if "<__parameter=answer>" in text_content:
                        match = re.search(r'<__parameter=answer>(.*?)</__parameter>', text_content)
                        if match:
                            tool_input = {"answer": match.group(1).strip()}
                            break

                    # Standard JSON cleaning
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Try parsing as JSON
                            data = json.loads(clean_text)

                            if not data or not isinstance(data, dict):
                                tool_input = {"answer": clean_text}
                            else:
                                tool_input = data

                        except json.JSONDecodeError:
                            # If parsing fails, fallback to dict with raw text
                            print("JSON parse failed, using raw text")
                            tool_input = {"answer": clean_text}

                        break  # exit the block iteration

                    except Exception as inner_e:
                        print(f"Unexpected inner error: {inner_e}")

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                validated = AnswerContainsCriteria(**tool_input)
                return validated.answer
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Optional: return a raw guess if pydantic fails
                raw_ans = str(tool_input.get('answer', '')).lower()
                return "yes" if "yes" in raw_ans else "no"

        else:
            return "yes"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "yes"


def analyse_table_row_criteria_gpt(model, openai_client, system_prompt_row, user_prompt_row, row, criteria):
    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                          EXTRACT CRITERIA TEXT                             │
# └────────────────────────────────────────────────────────────────────────────┘

def extract_restricting_criteria_text_amazon(model, bedrock_client, system_restricting_text, user_restricting_text,
                                             question_text, title, text, use_tool=True):
    try:
        # 1. Format the User Prompt
        formatted_user_text = user_restricting_text.format(
            question_text=question_text,
            title=title,
            text=text
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
                    ### Answer by returning: 
                    - is_relevant: boolean (true if the text contains information relevant to the question)
                    - evidence: string or null (the specific textual excerpt that answers the question. Use null if is_relevant is false)

                    ### Response Format: 
                    Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
                    {
                        "is_relevant": boolean,
                        "evidence": string or null
                    }
                    """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce the nullable field
            system_restricting_text += (
                "\nRespond ONLY with a raw JSON object: "
                "{\"is_relevant\": boolean, \"evidence\": string|null}"
            )

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_restricting_text}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [paragraph_extraction_tool],
                "toolChoice": {"tool": {"name": "ParagraphExtraction"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Handles <__parameter=is_relevant> and <__parameter=evidence>
                    if "<__parameter=" in text_content:
                        rel_match = re.search(r'<__parameter=is_relevant>(.*?)</__parameter>', text_content)
                        evid_match = re.search(r'<__parameter=evidence>(.*?)</__parameter>', text_content)

                        if rel_match:
                            tool_input = {
                                "is_relevant": rel_match.group(1).lower() == "true",
                                "evidence": evid_match.group(1).strip() if evid_match else None
                            }
                            break

                    # Standard JSON cleaning (Mistral/NVIDIA)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                # Pydantic hydration ensures keys match and types are correct
                validated = ParagraphExtraction(**tool_input)
                return validated.evidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fix-up for common LLM key naming errors
                is_rel = tool_input.get("is_relevant", False)
                if isinstance(is_rel, str): is_rel = is_rel.lower() == "true"
                return tool_input.get("evidence") or tool_input.get("description")

        else:
            return None

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return None


def extract_restricting_criteria_text_gpt(model, openai_client, system_restricting_text, user_restricting_text,
                                          question_text, title, text):
    response = openai_client.responses.parse(
        model=model,
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
                        "text": user_restricting_text.format(question_text=question_text, title=title, text=text)
                    }
                ]
            }
        ],
        text_format=ParagraphExtraction,
    )

    bridge_element = response.output_parsed
    print(bridge_element)
    return bridge_element.evidence


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                          EXTRACT CRITERIA IMAGE                            │
# └────────────────────────────────────────────────────────────────────────────┘

def extract_restricting_criteria_image_amazon(dataset, model, bedrock_client, system_restricting_image,
                                              user_restricting_image_text, question_text, image_title, image_path,
                                              use_tool=True):
    try:
        content_list = []

        image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                            image_path))

        # Strip the data URI header if it exists
        raw_b64 = image64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        image_bytes = base64.b64decode(raw_b64)
        ext = get_image_format(image_path, image_bytes)

        content_list.append({
            "image": {
                "format": ext,
                "source": {"bytes": image_bytes}
            }
        })

        # 2. Unified Text Prompt
        # Standardized to use 'question' - ensure your templates match this key
        formatted_text = user_restricting_image_text.format(question_text=question_text, image_title=image_title)

        # Add JSON instructions if not using a tool (NVIDIA use-case)
        if not use_tool:
            formatted_text += """### Answer by returning: 
            - is_relevant: boolean (whether the image contains information relevant to the question)
            - description: string (a minimal, precise textual description of the image that directly connects it to the question. Must only include explicitly stated information. Should be null if is_relevant is false)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "is_relevant": bool,
                "description": string
            }
            """

            system_restricting_image += "\nRespond ONLY with a raw JSON object: {\"is_relevant\": bool, \"description\": string}"

        content_list.append({"text": formatted_text})

        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

        # 3. Parameters for Converse API
        params = {
            "modelId": model,
            "messages": [{"role": "user", "content": content_list}],
            "system": [{"text": system_restricting_image}],
            "inferenceConfig": inference_config
        }

        # Add Tool Config only if requested
        if use_tool:
            params["toolConfig"] = {
                "tools": [image_extraction_tool],
                "toolChoice": {"tool": {"name": "ImageExtraction"}}
            }

        # 4. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Handles <__parameter=is_relevant> and <__parameter=evidence>
                    if "<__parameter=" in text_content:
                        rel_match = re.search(r'<__parameter=is_relevant>(.*?)</__parameter>', text_content)
                        evid_match = re.search(r'<__parameter=evidence>(.*?)</__parameter>', text_content)

                        if rel_match:
                            tool_input = {
                                "is_relevant": rel_match.group(1).lower() == "true",
                                "evidence": evid_match.group(1).strip() if evid_match else None
                            }
                            break

                    # Standard JSON cleaning (Mistral/NVIDIA)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                # Pydantic hydration ensures keys match and types are correct
                validated = ImageExtraction(**tool_input)
                return validated.evidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fix-up for common LLM key naming errors
                is_rel = tool_input.get("is_relevant", False)
                if isinstance(is_rel, str): is_rel = is_rel.lower() == "true"
                return tool_input.get("evidence") or tool_input.get("description")

        else:
            return None

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return None


def extract_restricting_criteria_image_gpt(dataset, model, openai_client, system_restricting_image,
                                           user_restricting_image_text, user_prompt_image_image,
                                           question_text, image_title, image_path):

    image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                        image_path))

    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                          EXTRACT CRITERIA TABLE ROW                        │
# └────────────────────────────────────────────────────────────────────────────┘

def extract_restricting_criteria_table_row_amazon(model, bedrock_client, system_restricting_table_row,
                                                  user_restricting_table_row, question_text, document_title, table_name,
                                                  table_description, table_row, use_tool=True):
    try:
        # 1. Format the User Prompt
        formatted_user_text = user_restricting_table_row.format(
            question_text=question_text,
            document_title=document_title,
            table_name=table_name,
            table_description=table_description,
            table_row=table_row
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - is_relevant: boolean (true if this specific row contains information relevant to the question)
            - evidence: string or null (the factual evidence from the row. Use null if is_relevant is false)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "is_relevant": boolean,
                "evidence": "string" or null
            }
            """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce the nullable field
            system_restricting_table_row += (
                "\nRespond ONLY with a raw JSON object: "
                "{\"is_relevant\": boolean, \"evidence\": string|null}"
            )

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_restricting_table_row}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [table_row_extraction_tool],
                "toolChoice": {"tool": {"name": "TableRowExtraction"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Handles <__parameter=is_relevant> and <__parameter=evidence>
                    if "<__parameter=" in text_content:
                        rel_match = re.search(r'<__parameter=is_relevant>(.*?)</__parameter>', text_content)
                        evid_match = re.search(r'<__parameter=evidence>(.*?)</__parameter>', text_content)

                        if rel_match:
                            tool_input = {
                                "is_relevant": rel_match.group(1).lower() == "true",
                                "evidence": evid_match.group(1).strip() if evid_match else None
                            }
                            break

                    # Standard JSON cleaning (Mistral/NVIDIA)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                # Pydantic hydration ensures keys match and types are correct
                validated = TableRowExtraction(**tool_input)
                return validated.evidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fix-up for common LLM key naming errors
                is_rel = tool_input.get("is_relevant", False)
                if isinstance(is_rel, str): is_rel = is_rel.lower() == "true"
                return tool_input.get("evidence") or tool_input.get("description")

        else:
            return None

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return None


def extract_restricting_criteria_table_row_gpt(model, openai_client, system_restricting_table_row,
                                               user_restricting_table_row, question_text, document_title, table_name,
                                               table_description, table_row):
    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                          ANALYSE BRIDGE ELEMENT IMAGE                      │
# └────────────────────────────────────────────────────────────────────────────┘

def analyse_image_bridge_element_amazon(dataset, model, bedrock_client, system_prompt_image_bridge,
                                        user_prompt_image_bridge_text,
                                        question_text, criteria, image_title, image_path, use_tool=True):
    try:
        content_list = []

        image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                            image_path))


        # Strip the data URI header if it exists
        raw_b64 = image64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        image_bytes = base64.b64decode(raw_b64)
        ext = get_image_format(image_path, image_bytes)

        content_list.append({
            "image": {
                "format": ext,
                "source": {"bytes": image_bytes}
            }
        })

        # 2. Unified Text Prompt
        # Standardized to use 'question' - ensure your templates match this key
        formatted_text = user_prompt_image_bridge_text.format(question_text=question_text,
                                                              criteria=criteria,
                                                              image_title=image_title)

        # Add JSON instructions if not using a tool (NVIDIA use-case)
        if not use_tool:
            formatted_text += """### Answer by returning: 
            - answer: string (answer with only "yes" or "no" to whether the image describes a bridge element matching the criteria)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "answer": string
            }
            """

            system_prompt_image_bridge += "\nRespond ONLY with a raw JSON object: {\"answer\": string}"

        content_list.append({"text": formatted_text})

        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

        # 3. Parameters for Converse API
        params = {
            "modelId": model,
            "messages": [{"role": "user", "content": content_list}],
            "system": [{"text": system_prompt_image_bridge}],
            "inferenceConfig": inference_config
        }

        # Add Tool Config only if requested
        if use_tool:
            params["toolConfig"] = {
                "tools": [analyse_criteria_tool],
                "toolChoice": {"tool": {"name": "AnswerContainsCriteria"}}
            }

        # 4. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield (Specific to your Nova problem)
                    if "<__parameter=answer>" in text_content:
                        match = re.search(r'<__parameter=answer>(.*?)</__parameter>', text_content)
                        if match:
                            tool_input = {"answer": match.group(1).strip()}
                            break

                    # Standard JSON cleaning
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Try parsing as JSON
                            data = json.loads(clean_text)
                            print(f"Data is: {data}")
                            if not data or not isinstance(data, dict):
                                tool_input = {"answer": clean_text}
                            else:
                                tool_input = data
                        except json.JSONDecodeError:
                            # If parsing fails, fallback to dict with raw text
                            print("JSON parse failed, using raw text")
                            tool_input = {"answer": clean_text}

                        print(f"Tool input: {tool_input}")
                        break  # exit the block iteration

                    except Exception as inner_e:
                        print(f"Unexpected inner error: {inner_e}")

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                validated = AnswerContainsCriteria(**tool_input)
                return validated.answer
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Optional: return a raw guess if pydantic fails
                raw_ans = str(tool_input.get('answer', '')).lower()
                return "yes" if "yes" in raw_ans else "no"

        else:
            return "yes"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "yes"


def analyse_image_bridge_element_gpt(dataset, model, openai_client, system_prompt_image_bridge, user_prompt_image_bridge_text,
                                     user_prompt_image_image, question_text, criteria, image_title, image_path):

    image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                        image_path))

    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                          ANALYSE BRIDGE ELEMENT TEXT                       │
# └────────────────────────────────────────────────────────────────────────────┘

def analyse_text_bridge_element_amazon(model, bedrock_client, system_prompt_text_bridge, user_prompt_text_bridge,
                                       question_text, criteria, title, text, use_tool=True):
    try:
        # 1. Format the User Prompt
        formatted_user_text = user_prompt_text_bridge.format(
            question_text=question_text,
            criteria=criteria,
            title=title,
            text=text
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - answer: string (answer with only "yes" or "no" to whether the text describes a bridge element matching the criteria)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "answer": string
            }
            """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce JSON schema
            system_prompt_text_bridge += "\nRespond ONLY with a raw JSON object: {\"answer\": string}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_text_bridge}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [analyse_criteria_tool],
                "toolChoice": {"tool": {"name": "AnswerContainsCriteria"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield (Specific to your Nova problem)
                    if "<__parameter=answer>" in text_content:
                        match = re.search(r'<__parameter=answer>(.*?)</__parameter>', text_content)
                        if match:
                            tool_input = {"answer": match.group(1).strip()}
                            break

                    # Standard JSON cleaning
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Try parsing as JSON
                            data = json.loads(clean_text)
                            print(f"Data is: {data}")
                            if not data or not isinstance(data, dict):
                                tool_input = {"answer": clean_text}
                            else:
                                tool_input = data
                        except json.JSONDecodeError:
                            # If parsing fails, fallback to dict with raw text
                            print("JSON parse failed, using raw text")
                            tool_input = {"answer": clean_text}

                        print(f"Tool input: {tool_input}")
                        break  # exit the block iteration

                    except Exception as inner_e:
                        print(f"Unexpected inner error: {inner_e}")

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                validated = AnswerContainsCriteria(**tool_input)
                return validated.answer
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Optional: return a raw guess if pydantic fails
                raw_ans = str(tool_input.get('answer', '')).lower()
                return "yes" if "yes" in raw_ans else "no"

        else:
            return "yes"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "yes"


def analyse_text_bridge_element_gpt(model, openai_client, system_prompt_text_bridge, user_prompt_text_bridge,
                                    question_text, criteria, title, text):
    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                      ANALYSE BRIDGE ELEMENT TABLE ROW                      │
# └────────────────────────────────────────────────────────────────────────────┘

def analyse_table_row_bridge_criteria_amazon(model, bedrock_client, system_prompt_row_bridge, user_prompt_row_bridge,
                                             question_text, criteria, table_row, table_name,
                                             table_description, use_tool=True):
    try:
        # 1. Format the User Prompt with all table context
        formatted_user_text = user_prompt_row_bridge.format(
            question_text=question_text,
            criteria=criteria,
            table_name=table_name,
            table_description=table_description,
            table_row=table_row
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - answer: string (answer with only "yes" or "no" to whether the table row describes a bridge element matching the criteria)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "answer": "yes" or "no"
            }
            """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce JSON schema consistency
            system_prompt_row_bridge += "\nRespond ONLY with a raw JSON object: {\"answer\": string}"

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_prompt_row_bridge}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [analyse_criteria_tool],
                "toolChoice": {"tool": {"name": "AnswerContainsCriteria"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield (Specific to your Nova problem)
                    if "<__parameter=answer>" in text_content:
                        match = re.search(r'<__parameter=answer>(.*?)</__parameter>', text_content)
                        if match:
                            tool_input = {"answer": match.group(1).strip()}
                            break

                    # Standard JSON cleaning
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()

                        try:
                            # Try parsing as JSON
                            data = json.loads(clean_text)
                            print(f"Data is: {data}")
                            if not data or not isinstance(data, dict):
                                tool_input = {"answer": clean_text}
                            else:
                                tool_input = data
                        except json.JSONDecodeError:
                            # If parsing fails, fallback to dict with raw text
                            print("JSON parse failed, using raw text")
                            tool_input = {"answer": clean_text}

                        print(f"Tool input: {tool_input}")
                        break  # exit the block iteration

                    except Exception as inner_e:
                        print(f"Unexpected inner error: {inner_e}")

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                validated = AnswerContainsCriteria(**tool_input)
                return validated.answer
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Optional: return a raw guess if pydantic fails
                raw_ans = str(tool_input.get('answer', '')).lower()
                return "yes" if "yes" in raw_ans else "no"

        else:
            return "yes"

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return "yes"


def analyse_table_row_bridge_criteria_gpt(model, openai_client, system_prompt_row_bridge, user_prompt_row_bridge,
                                          question_text, criteria, table_row, table_name,
                                          table_description):
    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                         CHECK ANSWER IN PARAGRAPH                          │
# └────────────────────────────────────────────────────────────────────────────┘

def check_answer_image_amazon(dataset, model, bedrock_client, system_check_answer_image, user_check_answer_image,
                              answer_class_specific, answer_class_general, question_text,
                              caption_text, image_path, use_tool=True):
    try:
        content_list = []

        image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                            image_path))

        # Strip the data URI header if it exists
        raw_b64 = image64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        image_bytes = base64.b64decode(raw_b64)
        ext = get_image_format(image_path, image_bytes)

        content_list.append({
            "image": {
                "format": ext,
                "source": {"bytes": image_bytes}
            }
        })

        # 2. Unified Text Prompt
        # Standardized to use 'question' - ensure your templates match this key
        formatted_text = user_check_answer_image.format(question_text=question_text,
                                                        answer_class_specific=answer_class_specific,
                                                        answer_class_general=answer_class_general,
                                                        caption_text=caption_text)

        # Add JSON instructions if not using a tool (NVIDIA use-case)
        if not use_tool:
            formatted_text += """### Answer by returning
            - contains: bool (true if the image contains the expected answer entity, false otherwise)
            - entity: string (the answer for the question extracted from the image)
            - match_level: (one of "specific", "general", "none")
            - confidence: value between 0 and 1
            
            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble. 
            {
                "contains": bool, 
                "entity": string,
                "match_level": string (one of: specific, general, none),
                "confidence": float
            }
            """
            system_check_answer_image += "\nRespond ONLY with a raw JSON object: {\"contains\": bool, \"entity\": string, \"match_level\": string, \"confidence\": float}"

        content_list.append({"text": formatted_text})

        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

        # 3. Parameters for Converse API
        params = {
            "modelId": model,
            "messages": [{"role": "user", "content": content_list}],
            "system": [{"text": system_check_answer_image}],
            "inferenceConfig": {"temperature": 0, "maxTokens": 1000}
        }

        # Add Tool Config only if requested
        if use_tool:
            params["toolConfig"] = {
                "tools": [image_contains_answer_tool],
                "toolChoice": {"tool": {"name": "ImageContainsAnswer"}}
            }

        # 4. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Reconstructs dict from individual parameter tags
                    if "<__parameter=" in text_content:
                        try:
                            contains_m = re.search(r'<__parameter=contains>(.*?)</__parameter>', text_content)
                            entity_m = re.search(r'<__parameter=entity>(.*?)</__parameter>', text_content)
                            match_m = re.search(r'<__parameter=match_level>(.*?)</__parameter>', text_content)
                            conf_m = re.search(r'<__parameter=confidence>(.*?)</__parameter>', text_content)

                            if contains_m or entity_m:
                                tool_input = {
                                    "contains": contains_m.group(1).lower() == "true" if contains_m else False,
                                    "entity": entity_m.group(1).strip() if entity_m else "NONE",
                                    "match_level": match_m.group(1).strip().lower() if match_m else "none",
                                    "confidence": float(conf_m.group(1)) if conf_m else 1.0
                                }
                                break
                        except:
                            pass

                    # Standard JSON cleaning (Mistral/NVIDIA/Claude)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        print(f"Tool input is: {tool_input}")
        # 6. Validation and Return
        if tool_input:
            try:
                # Use Pydantic to ensure types (bool, str, Literal, float)
                validated = ImageContainsAnswer(**tool_input)
                return validated.contains, validated.entity, validated.match_level, validated.confidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fallback normalization
                return (
                    str(tool_input.get("contains", "")).lower() == "true",
                    str(tool_input.get("entity", "NONE")),
                    str(tool_input.get("match_level", "none")).lower(),
                    float(tool_input.get("confidence", 0.0))
                )
        else:
            return False, "NONE", "NONE", 1.00

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return False, "NONE", "NONE", 1.00


def check_answer_image_gpt(dataset, model, openai_client, system_check_answer_image, user_check_answer_image,
                           user_prompt_image_image, answer_class_specific, answer_class_general, question_text,
                           caption_text, image_path):

    image64 = encode_image(os.path.join(f"/Users/emanuelemezzi/Desktop/datasetNIPS/{dataset}/final_dataset_images",
                                        image_path))

    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                         CHECK ANSWER IN PARAGRAPH                          │
# └────────────────────────────────────────────────────────────────────────────┘

def check_answer_in_paragraph_amazon(model, bedrock_client, system_check_answer_text, user_check_answer_text,
                                     answer_class, question_text, paragraph_text, contextual_information,
                                     use_tool=True):
    try:
        # 1. Format the User Prompt
        formatted_user_text = user_check_answer_text.format(
            question_text=question_text,
            answer_class=answer_class,
            paragraph_text=paragraph_text,
            contextual_information=contextual_information
        )

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - contains: boolean (true if the paragraph contains the expected answer entity, false otherwise)
            - entity: string (the exact text span matching the answer type, or "NONE")
            - confidence: number (float between 0.0 and 1.0)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "contains": boolean,
                "entity": string,
                "confidence": number
            }
            """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce JSON schema consistency
            system_check_answer_text += (
                "\nRespond ONLY with a raw JSON object: "
                "{\"contains\": boolean, \"entity\": string, \"confidence\": number}"
            )

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": system_check_answer_text}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [paragraph_contains_answer_tool],
                "toolChoice": {"tool": {"name": "ParagraphContainsAnswer"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Reconstructs dict from individual parameter tags
                    if "<__parameter=" in text_content:
                        try:
                            contains_m = re.search(r'<__parameter=contains>(.*?)</__parameter>', text_content)
                            entity_m = re.search(r'<__parameter=entity>(.*?)</__parameter>', text_content)
                            conf_m = re.search(r'<__parameter=confidence>(.*?)</__parameter>', text_content)

                            if contains_m or entity_m:
                                tool_input = {
                                    "contains": contains_m.group(1).lower() == "true" if contains_m else False,
                                    "entity": entity_m.group(1).strip() if entity_m else "NONE",
                                    "confidence": float(conf_m.group(1)) if conf_m else 1.0
                                }
                                break
                        except:
                            pass

                    # Standard JSON cleaning (Mistral/NVIDIA/Claude markdown)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 8. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                # Use Pydantic to ensure types (bool, str, float) are correct
                validated = ParagraphContainsAnswer(**tool_input)
                return validated.contains, validated.entity, validated.confidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fallback normalization
                return (
                    str(tool_input.get("contains", "")).lower() == "true",
                    str(tool_input.get("entity", "NONE")),
                    float(tool_input.get("confidence", 0.0))
                )

        else:
            return False, "NONE", 1.00

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return False, "NONE", 1.00


def check_answer_in_paragraph_gpt(model, openai_client, system_check_answer_text, user_check_answer_text, answer_class,
                                  question_text, paragraph_text, contextual_information):
    response = openai_client.responses.parse(
        model=model,
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


# ┌────────────────────────────────────────────────────────────────────────────┐
# │                         CHECK ANSWER IN ROW                                │
# └────────────────────────────────────────────────────────────────────────────┘

def check_answer_in_row_amazon(model, bedrock_client, system_check_answer_row_cond_criteria,
                               user_check_answer_row_cond_criteria, system_check_answer_row, user_check_answer_row,
                               answer_class_specific, row, question_text, table_description, conditional_criteria,
                               use_tool=True):
    try:
        # 1. Branching Prompt Logic
        if conditional_criteria:
            formatted_user_text = user_check_answer_row_cond_criteria.format(
                question_text=question_text,
                answer_class_specific=answer_class_specific,
                table_description=table_description,
                conditional_criteria=conditional_criteria,
                row=row
            )
            current_system_prompt = system_check_answer_row_cond_criteria
        else:
            formatted_user_text = user_check_answer_row.format(
                question_text=question_text,
                answer_class_specific=answer_class_specific,
                table_description=table_description,
                row=row
            )
            current_system_prompt = system_check_answer_row

        # 2. Add JSON instructions if not using a tool (NVIDIA/Fallback case)
        if not use_tool:
            json_instruction = """
            ### Answer by returning: 
            - contains: boolean (true if the row cell contains the expected answer entity, false otherwise)
            - entity: string (the exact text span from the row matching the answer type, or "NONE")
            - confidence: number (float between 0.0 and 1.0)

            ### Response Format: 
            Respond ONLY with a raw JSON object. Do not include markdown or preamble. 
            {
                "contains": boolean,
                "entity": string,
                "confidence": number
            }
            """
            formatted_user_text += json_instruction

            # Sync System Prompt to enforce JSON schema consistency
            current_system_prompt += (
                "\nRespond ONLY with a raw JSON object: "
                "{\"contains\": boolean, \"entity\": string, \"confidence\": number}"
            )

        # 3. Structure the Messages
        messages = [{
            "role": "user",
            "content": [{"text": formatted_user_text}]
        }]

        # 4. Configure Inference
        inference_config = {"temperature": 0, "maxTokens": 2000}
        if "mistral" in model.lower() or "nvidia" in model.lower():
            inference_config["topP"] = 0.1

            # 5. Prepare Converse Parameters
        params = {
            "modelId": model,
            "messages": messages,
            "system": [{"text": current_system_prompt}],
            "inferenceConfig": inference_config
        }

        if use_tool:
            params["toolConfig"] = {
                "tools": [row_contains_answer_tool],
                "toolChoice": {"tool": {"name": "RowContainsAnswer"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        tool_input = None

        # Case 1: Root-level toolUse
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2 & 3: Content block iteration
        if not tool_input:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text_content = block['text'].strip()
                    if not text_content: continue

                    # Nova XML Shield - Handles multiple parameters in tags
                    if "<__parameter=" in text_content:
                        try:
                            contains_m = re.search(r'<__parameter=contains>(.*?)</__parameter>', text_content)
                            entity_m = re.search(r'<__parameter=entity>(.*?)</__parameter>', text_content)
                            conf_m = re.search(r'<__parameter=confidence>(.*?)</__parameter>', text_content)

                            if contains_m or entity_m:
                                tool_input = {
                                    "contains": contains_m.group(1).lower() == "true" if contains_m else False,
                                    "entity": entity_m.group(1).strip() if entity_m else "NONE",
                                    "confidence": float(conf_m.group(1)) if conf_m else 1.0
                                }
                                break
                        except:
                            pass

                    # Standard JSON cleaning (Mistral/NVIDIA)
                    try:
                        clean_text = text_content.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except:
                        continue

        # 6. Validation and Return
        print(f"Tool input is: {tool_input}")
        if tool_input:
            try:
                # Force types into the Pydantic schema
                validated = RowContainsAnswer(**tool_input)
                return validated.contains, validated.entity, validated.confidence
            except Exception as e:
                print(f"Validation Error for {model}: {e}")
                # Manual fallback if Pydantic fails
                return (
                    str(tool_input.get("contains", "")).lower() == "true",
                    str(tool_input.get("entity", "NONE")),
                    float(tool_input.get("confidence", 0.0))
                )

        else:
            return False, "NONE", 1.00

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return False, "NONE", 1.00


def check_answer_in_row_gpt(model, openai_client, system_check_answer_row_cond_criteria,
                            user_check_answer_row_cond_criteria, system_check_answer_row, user_check_answer_row,
                            answer_class_specific, row, question_text, table_description, conditional_criteria):
    if conditional_criteria:
        system_prompt = system_check_answer_row_cond_criteria
        user_prompt = user_check_answer_row_cond_criteria.format(question_text=question_text,
                                                                 answer_class_specific=answer_class_specific,
                                                                 table_description=table_description,
                                                                 conditional_criteria=conditional_criteria,
                                                                 row=row)

    else:
        system_prompt = system_check_answer_row
        user_prompt = user_check_answer_row.format(question_text=question_text,
                                                   answer_class_specific=answer_class_specific,
                                                   table_description=table_description,
                                                   row=row)

    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        text_format=RowContainsAnswer
    )

    found = response.output_parsed
    return found.contains, found.entity, found.confidence
