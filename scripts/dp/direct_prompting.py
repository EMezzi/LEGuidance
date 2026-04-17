import os
import json
from prompts.dp.direct_prompting import *
from prompts.le.prompt_analyse_criteria import user_prompt_image_image
from utils.utilities import get_question_data, get_question_files, encode_image, detect_media_type_from_bytes
from schemas.pydantic_schemas import DPAnswer
import base64

from botocore.exceptions import ClientError

answer_tool = {
    "toolSpec": {
        "name": "get_answer_dp",
        "description": "Get the answer given the current question and data",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "contains": {
                        "type": "boolean",
                        "description": "Whether the given data contains answer to the question"
                    },
                    "entity": {
                        "type": "string",
                        "description": "The answer to the question if present otherwise NONE"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence value regarding the answer, between 0 and 1"
                    }
                },
                "required": ["contains", "entity", "confidence"],
            }
        },
    }
}


class DPAgent:
    def __init__(self, openai_client, bedrock_client):
        self.openai_client = openai_client
        self.bedrock_client = bedrock_client

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

    def dp_final_answer(self, model, question_text, paragraphs_text, images_text, images_inputs, tables_text):

        # print(f"Model: {model}")
        # print(f"Question text: {question_text}")
        # print(f"Paragraphgs: {paragraphs_text}")
        # print(f"images inputs: {image_inputs}")
        # print(f"Tables: {tables_text}")

        if model == "global.amazon.nova-2-lite-v1:0":

            # 1. Prepare Content List for Nova
            # We combine the formatted text and the images into a single content list
            content_list = []

            # Add the formatted text prompt
            formatted_user_text = user_prompt_dp.format(
                question_text=question_text,
                images_text=images_text,
                paragraphs_text=paragraphs_text,
                tables_text=tables_text
            )
            content_list.append({"text": formatted_user_text})

            # Add images (Assuming images_inputs contains base64 strings)
            for img in images_inputs:
                # Nova expects raw bytes for the 'source' if passing base64
                image_bytes = base64.b64decode(img['image_url'])

                content_list.append({
                    "image": {
                        "format": img['image_ext'].lower(),  # e.g., 'png' or 'jpeg'
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                })

            # 2. Structure the Messages
            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]

            # 3. System Prompt (Nova handles this as a separate parameter in converse)
            system_prompts = [{"text": system_prompt_dp}]

            model_id = model

            # 4. Invoke the model
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig={"temperature": 0, "maxTokens": 1000},
                    toolConfig={
                        "tools": [answer_tool],
                        "toolChoice": {
                            "tool": {"name": "get_answer_dp"}  # Forces the tool call
                        }
                    }
                )

                # Extracting the assistant's message
                output_message = response["output"]["message"]

                print(output_message)
                for block in output_message.get('content', []):
                    if 'toolUse' in block:
                        return block['toolUse']['input']

                return output_message

            except ClientError as err:
                print(f"A client error occurred: {err.response['Error']['Message']}")
                return None

        elif model == "gpt-5.2":
            print("Perfetto gpt: dp")
            try:
                response = self.openai_client.responses.parse(
                    model="gpt-5.2",
                    input=[
                        {
                            "role": "system",
                            "content": system_prompt_dp
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": user_prompt_dp.format(question_text=question_text,
                                                                  images_text=images_text,
                                                                  paragraphs_text=paragraphs_text,
                                                                  tables_text=tables_text)

                                },
                                *[
                                    {
                                        "type": "input_image",
                                        "image_url": user_prompt_image_image.format(image64=img["image_url"])
                                    }

                                    for img in images_inputs
                                ]
                            ]
                        }
                    ],
                    text_format=DPAnswer,
                )

                print(response)
                found = response.output_parsed
                return {"contains": found.contains, "entity": found.entity, "confidence": found.confidence}

            except Exception as e:
                print(f"Converse API Error for Qwen: {str(e)}")
                return None

        elif model == "mistral.mistral-large-3-675b-instruct":

            responses = []

            # 1. Define the Tool Configuration (The JSON Schema)
            # This forces Mistral to output structured data

            # 2. Iterate through images in batches of 3
            for i in range(0, len(images_inputs), 3):
                current_images_text = "\n\n".join(images_text.split("\n\n")[i:i + 3])

                # Prepare text content
                prompt_text = user_prompt_dp.format(
                    question_text=question_text,
                    images_text=current_images_text,
                    paragraphs_text=paragraphs_text,
                    tables_text=tables_text
                )

                # 3. Build the Content List (Text + Decoded Image Bytes)
                content_list = [{"text": prompt_text}]

                for img in images_inputs[i:i + 3]:
                    raw_data = img['image_url']

                    # CRITICAL: Strip the Data URI header if present
                    # Mistral will fail if "data:image/png;base64," is included in the bytes
                    if "," in raw_data:
                        raw_data = raw_data.split(",")[1]

                    try:
                        # Decode base64 string to raw bytes
                        img_bytes = base64.b64decode(raw_data)

                        content_list.append({
                            "image": {
                                "format": "png",  # Ensure this matches: 'png', 'jpeg', 'gif', or 'webp'
                                "source": {"bytes": img_bytes}
                            }
                        })
                    except Exception as decode_err:
                        print(f"Failed to decode image in batch {i}: {decode_err}")

                # 4. Prepare Message structure
                messages = [{"role": "user", "content": content_list}]

                try:
                    # 5. Call Bedrock Converse
                    # Mistral handles 'system' as a separate top-level parameter in Converse
                    response = self.bedrock_client.converse(
                        modelId=model,
                        messages=messages,
                        system=[{"text": system_prompt_dp}],
                        inferenceConfig={"temperature": 0.0, "maxTokens": 1000},
                        toolConfig={
                            "tools": [answer_tool],
                            "toolChoice": {
                                "tool": {"name": "get_answer_dp"}  # Forces the model to use the tool
                            }
                        }
                    )

                    # 6. Extract the Tool Use Output
                    output_message = response['output']['message']

                    for block in output_message.get('content', []):
                        if 'toolUse' in block:
                            print(block['toolUse']['input'])
                            responses.append(block['toolUse']['input'])
                        else:
                            responses.append(output_message)

                except Exception as e:
                    print(f"Error processing batch {i // 3}: {str(e)}")
                    responses.append({"error": str(e), "batch": i // 3})

            return responses

        elif model == "moonshotai.kimi-k2.5":
            try:
                # 2. Format Content for Converse API
                # The content list must strictly follow Bedrock's order/types
                content = []

                # Add Images first (Base64 format differs slightly in Converse)
                for img in images_inputs:
                    # Bedrock Converse expects bytes, not a base64 string
                    image_bytes = base64.b64decode(img['image_url'])

                    # Map extensions to supported formats (png, jpeg, gif, webp)
                    format_ext = img['image_ext'].lower().replace('jpg', 'jpeg')

                    content.append({
                        "image": {
                            "format": format_ext,
                            "source": {"bytes": image_bytes}
                        }
                    })

                # Add Text
                prompt_text = user_prompt_dp.format(
                    question_text=question_text,
                    images_text=images_text,
                    paragraphs_text=paragraphs_text,
                    tables_text=tables_text
                )
                content.append({"text": prompt_text})

                # 3. Assemble Messages and System Prompt
                messages = [{"role": "user", "content": content}]
                system_prompts = [{"text": system_prompt_dp}]

                # 4. Call Converse
                # toolConfig forces the model to use the tool (toolChoice)

                response = self.bedrock_client.converse(
                    modelId=model,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig={"temperature": 0.0, "maxTokens": 1000},
                    toolConfig={
                        "tools": [answer_tool],
                        "toolChoice": {
                            "tool": {"name": "get_answer_dp"}  # Forces the model to use the tool
                        }
                    }
                )

                # 5. Parse Tool Output
                output_message = response['output']['message']
                print(output_message)

                # Look for the toolUse block in the response
                for content_block in output_message.get('content', []):
                    if 'toolUse' in content_block:
                        return content_block['toolUse']['input']

                return output_message

            except Exception as e:
                print(f"Converse API Error for Kimi: {str(e)}")
                return None

        elif model == "nvidia.nemotron-nano-12b-v2":
            try:
                content_list = []

                # 1. Add Images (Converted to the correct dictionary format)
                for img in images_inputs:
                    raw_b64 = img['image_url']
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",")[1]

                    img_bytes = base64.b64decode(raw_b64)
                    img_format = img.get('image_ext', 'png').lower().replace('jpg', 'jpeg')

                    content_list.append({
                        "image": {
                            "format": img_format,
                            "source": {"bytes": img_bytes}
                        }
                    })

                # 2. Build the Text Prompt
                prompt_text = user_prompt_dp.format(
                    question_text=question_text,
                    images_text=images_text,
                    paragraphs_text=paragraphs_text,
                    tables_text=tables_text
                )

                prompt_text += """\n### Answer by returning: 
                - 'contains': boolean (true if the answer is in the data, false otherwise)
                - 'entity': string (the specific answer text, or "NONE" if not found)
                - 'confidence': number (a float between 0 and 1)

                ### Response Format:
                Respond ONLY with a raw JSON object. Do not include markdown, comments, or preamble.
                {
                    "contains": boolean,
                    "entity": string,
                    "confidence": number
                }
                """

                # FIX: Wrap the string in a dictionary with the "text" key
                content_list.append({"text": prompt_text})

                # 3. Prepare System Prompt
                system_content = [{
                    "text": system_prompt_dp + "\nOutput your answer in JSON format. Never talk to the user. Output only raw JSON."}]

                # 4. Call Converse
                response = self.bedrock_client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": content_list}],
                    system=system_content,
                    inferenceConfig={"temperature": 0.0, "maxTokens": 1000}
                )

                # 5. Extract and Parse Response
                output_message = response['output']['message']

                print(output_message)

                return output_message

            except Exception as e:
                print(f"Converse API Error for Nemotron-Nano: {str(e)}")
                return None

        elif model == "qwen.qwen3-vl-235b-a22b":
            try:
                # 1. Prepare User Prompt
                prompt_text = user_prompt_dp.format(
                    question_text=question_text,
                    images_text=images_text,
                    paragraphs_text=paragraphs_text,
                    tables_text=tables_text
                )

                # 2. Build the Content List
                content_list = [{"text": prompt_text}]

                for img in images_inputs:
                    raw_b64 = img['image_url']
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",")[1]

                    # The Converse API needs raw bytes
                    img_bytes = base64.b64decode(raw_b64)

                    content_list.append({
                        "image": {
                            "format": img.get('image_ext', 'png'),  # e.g., 'png' or 'jpeg'
                            "source": {"bytes": img_bytes}
                        }
                    })

                # 4. Invoke Converse
                response = self.bedrock_client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": content_list}],
                    system=[{"text": system_prompt_dp}],
                    inferenceConfig={"temperature": 0, "maxTokens": 1000},
                    toolConfig={
                        "tools": [answer_tool],
                        "toolChoice": {
                            "tool": {"name": "get_answer_dp"}  # Forces the model to use the tool
                        }
                    }
                )

                # 5. Extract the output
                output_message = response['output']['message']

                print(output_message)

                # Search the content blocks for the tool call
                for block in output_message.get('content', []):
                    if 'toolUse' in block:
                        return block['toolUse']['input']

                return output_message

            except Exception as e:
                print(f"Converse API Error for Qwen: {str(e)}")
                return None

        elif model == "us.anthropic.claude-sonnet-4-6":
            try:
                # 2. Build the Content List
                content_list = []

                # Add Images first (Claude processes visual context best when followed by the prompt)
                for img in images_inputs:
                    raw_b64 = img['image_url']
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",")[1]

                    img_bytes = base64.b64decode(raw_b64)
                    # Claude/Bedrock Converse supports: 'png', 'jpeg', 'gif', 'webp'
                    img_format = img.get('image_ext', 'png').lower().replace('jpg', 'jpeg')

                    content_list.append({
                        "image": {
                            "format": img_format,
                            "source": {"bytes": img_bytes}
                        }
                    })

                # Add Text
                prompt_text = user_prompt_dp.format(
                    question_text=question_text,
                    images_text=images_text,
                    paragraphs_text=paragraphs_text,
                    tables_text=tables_text
                )
                content_list.append({"text": prompt_text})

                # 3. Invoke Converse
                response = self.bedrock_client.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": content_list}],
                    system=[{"text": system_prompt_dp}],
                    inferenceConfig={"temperature": 0.0, "maxTokens": 1000},
                    toolConfig={
                        "tools": [answer_tool],
                        "toolChoice": {
                            "tool": {"name": "get_answer_dp"}
                        }
                    }
                )

                # 4. Extract Tool Output
                output_message = response['output']['message']

                for block in output_message.get('content', []):
                    if 'toolUse' in block:
                        return block['toolUse']['input']

                return output_message

            except Exception as e:
                print(f"Converse API Error for Claude: {str(e)}")
                return None


def dp_main(model, dp_agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images,
            answers_dir):
    print(f"Questions dir: {questions_dir}")

    os.makedirs(answers_dir + "/unimodal", exist_ok=True)
    os.makedirs(answers_dir + "/multimodal", exist_ok=True)

    for i, question in enumerate(questions_list):
        print(f"Question {i}: {question}")
        json_question = json.load(open(os.path.join(questions_dir, question), "rb"))
        modalities = json_question["metadata"]["modalities"]
        if len(modalities) == 1:
            unimodal_multimodal = "/unimodal"
        else:
            unimodal_multimodal = "/multimodal"

        if question not in os.listdir(answers_dir + unimodal_multimodal):
            question_data = get_question_data(questions_dir, question)
            question_files = get_question_files(association_dir, question)

            paragraphs_text, images_text, images_inputs, tables_text = DPAgent.data_preparation(
                question_files["text_set"], question_files["image_set"], final_dataset_images,
                question_files["table_set"][0], table_dir)

            print(f"Question text: {question_data['question_text']}")
            answer = dp_agent.dp_final_answer(
                model, question_data["question_text"], paragraphs_text, images_text, images_inputs, tables_text)
            print(answer)

            if answer is None:
                print("ciao")
                json.dump({'final_answer': None}, open(os.path.join(answers_dir + unimodal_multimodal, question), "w"),
                          indent=4)

            elif isinstance(answer, list):
                print("Ma si pazz")
                if any(el is not None for el in answer):
                    json.dump(answer, open(os.path.join(answers_dir + unimodal_multimodal, question), "w"), indent=4)
                else:
                    json.dump({'final_answer': None}, open(os.path.join(answers_dir + unimodal_multimodal, question), "w"), indent=4)
            else:
                print("Perfetto")
                if answer is not None:
                    print("Perfetto")
                    json.dump(answer, open(os.path.join(answers_dir + unimodal_multimodal, question), "w"), indent=4)
