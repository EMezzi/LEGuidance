from schemas.pydantic_schemas import *
from schemas.tools import *
from utils.utilities import save_json_file
import json


def normalize_to_schema(data):
    """Safety net to map unapproved LLM labels to 'Other' or 'other'."""
    ALLOWED_TOPICS = {
        "Film", "Transportation", "Video games", "Industry", "Theater", "Television",
        "Music", "Geography", "Literature", "History", "Economy", "Sports", "Science",
        "Politics", "Buildings", "Other"
    }

    ALLOWED_KINDS = {
        "relation", "award", "league", "season", "time", "role", "qualifier", "domain", "other"
    }

    # 1. Fix Topic: Force title case and check allowed list
    topic_obj = data.get("topic", {})
    if isinstance(topic_obj, dict):
        current_topic = topic_obj.get("question_topic")
        if current_topic not in ALLOWED_TOPICS:
            print("mannazz u cazz")
            data["topic"]["question_topic"] = "Other"

    # 2. Fix Constraint Kinds: Force lower case and check allowed list
    if "constraints" in data and isinstance(data["constraints"], list):
        for constraint in data["constraints"]:
            current_kind = constraint.get("kind", "").lower()
            if current_kind not in ALLOWED_KINDS:
                print("mannazz u crist")
                # Maps "location", "event", "race", "achievement", etc. -> "other"
                constraint["kind"] = "other"
            else:
                constraint["kind"] = current_kind

    # 3. Fix Target: Ensure text is string and type is not null
    target_obj = data.get("target", {})
    if isinstance(target_obj, dict):
        if target_obj.get("type") is None:
            print("mannazz u jeppson")
            data["target"]["type"] = "NONE"

    return data


def extract_criterias_amazon(model, bedrock_client, system_prompt_criteria, user_prompt_criteria, question_text,
                             question, criteria_extraction_dir, use_tool=True):
    try:
        # 1. Format the User Prompt
        # Note: Ensure user_prompt_criteria is defined in your global scope
        formatted_user_text = user_prompt_criteria.format(question_text=question_text)

        # 2. Add JSON instructions if not using a tool (NVIDIA/Mistral case)
        # We mirror the DistinctionCriteria Pydantic schema exactly
        if not use_tool:
            # Defining the allowed values for the prompt
            topics = "Film|Transportation|Video games|Industry|Theater|Television|Music|Geography|Literature|History|Economy|Sports|Science|Politics|Buildings|Other"
            constraint_kinds = "relation|award|league|season|time|role|qualifier|domain|other"
            cardinalities = "single|multiple|unknown"

            json_instruction = f"""
            ### Response Format:
            Respond ONLY with a raw JSON object. Do not include markdown, preamble, or explanations.
        
            ### Field Definitions:
            - expected_answer_type_general: The broad category (e.g., Location).
            - expected_answer_type_specific: The narrow, precise type (e.g., City).

            ### Schema Constraints:
            - topic.question_topic: MUST be one of [{topics}]
            - expected_cardinality: MUST be one of [{cardinalities}]
            - constraints[].kind: MUST be one of [{constraint_kinds}]
            - IMPORTANT: If a constraint is not in the categories indicated write "other".
            - target.type: MUST be a string (e.g., "Person", "Film", "Poster"). If unknown, use "NONE". DO NOT USE NULL.
            - constraints: use an empty list [] if there are no constraints
            - time_constraints: use an empty list [] if there are no time constraints
            - aliases: use an empty list [] if there are no aliases

            ### JSON Structure:
            {{
                "topic": {{ "question_topic": "{topics}" }},
                "expected_answer_type": {{
                    "expected_answer_type_specific": Specific type of the expected answer ("e.g. 'City'"),
                    "expected_answer_type_general": Generic type of the expected answer ("e.g. 'Location'")
                }},
                "expected_cardinality": "single",
                "target": {{
                    "text": "Main entity name",
                    "type": "Entity type or null"
                }},
                "asked_property": "Short predicate (e.g. 'played for')",
                "constraints": [
                    {{
                        "kind": "{constraint_kinds}",
                        "evidence": "phrase from question",
                        "normalized": "cleaned version"
                    }}
                ],
                "time_constraints": [
                    {{
                        "label": "time phrase",
                        "start_year": int or null,
                        "end_year": int or null,
                        "start_date": "YYYY-MM-DD or null",
                        "end_date": "YYYY-MM-DD or null"
                    }}
                ],
                "aliases": [
                    {{
                        "text": "string",
                        "reason": "typo|nickname|abbreviation"
                    }}
                ],
                "rewritten_question": "Minimal rewrite preserving meaning"
            }}
            """
            formatted_user_text += json_instruction

            # Update system prompt to be even stricter
            system_prompt_criteria += (
                f"\nMandatory: Output valid raw JSON object."
                f"Allowed topics: {topics}. Allowed constraint kinds: {constraint_kinds}."
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
            "system": [{"text": system_prompt_criteria}],
            "inferenceConfig": inference_config
        }

        # Use tool only for models that support/prefer it (e.g., Claude, Nova)
        if use_tool:
            params["toolConfig"] = {
                "tools": [criteria_tool],
                "toolChoice": {"tool": {"name": "DistinctionCriteria"}}
            }

        # 6. Invoke Model
        response = bedrock_client.converse(**params)
        output_message = response["output"]["message"]

        print(output_message)

        # 7. Extract and Process Result
        tool_input = None

        # Case 1: root-level toolUse (rare but possible)
        if isinstance(output_message, dict) and 'toolUse' in output_message:
            tool_input = output_message['toolUse'].get('input')

        # Case 2: iterate over content blocks
        else:
            for block in output_message.get('content', []):
                if 'toolUse' in block:
                    tool_input = block['toolUse'].get('input')
                    break

                if 'text' in block:
                    text = block['text'].strip()

                    if not text:
                        continue  # skip empty text blocks

                    try:
                        clean_text = text.strip('`').replace('json', '', 1).strip()
                        tool_input = json.loads(clean_text)
                        break
                    except json.JSONDecodeError:
                        print(f"Skipping non-JSON text block for {model}: {text[:100]}")
                    except Exception as e:
                        print(f"Unexpected parsing error for {model}: {e}")

        # 8. Save and Return
        if tool_input:
            try:
                # This fills in missing [], handles default "single" cardinality,
                # and ensures all Enums are valid.
                tool_input = normalize_to_schema(tool_input)
                validated = DistinctionCriteria(**tool_input)

                # model_dump() ensures the saved JSON has ALL keys, matching GPT's behavior
                final_data = validated.model_dump()

                save_json_file(final_data, question, question_text, criteria_extraction_dir, model)
                return True

            except Exception as pydantic_err:
                print(f"Validation Error for {model}: {pydantic_err}")
                # Optional: Save the raw 'broken' data for debugging
                return False
        else:
            print(f"No valid tool input found for {model}")
            return False

    except Exception as e:
        print(f"Bedrock Converse Error for model {model}: {str(e)}")
        return None


def extract_criterias_gpt(model, openai_client, system_prompt_criteria, user_prompt_criteria, question_text, question,
                          criteria_extraction_dir):
    response = openai_client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt_criteria
            },
            {
                "role": "user",
                "content": user_prompt_criteria.format(question_text=question_text)
            }
        ],
        text_format=DistinctionCriteria,
    )

    distinction_criterias = response.output_parsed
    save_json_file(distinction_criterias, question, question_text, criteria_extraction_dir, model)
