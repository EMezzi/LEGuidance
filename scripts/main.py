import os
import ast
import random
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from criteria_extraction import extract_criterias_main
from entropy_calculation import entropy_calculation_main, Agent


def get_nebula_models(client):
    models = []
    for model in client.models.list().data:
        models.append(model.id)
    return models


if __name__ == "__main__":
    load_dotenv()

    random.seed(42)

    models = ["gpt-5.2", "claude-sonnet-4-5"]

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))

    QUESTIONS_MULTIMODALQA_TRAINING = os.getenv("QUESTIONS_MULTIMODALQA_TRAINING")
    IMAGE_DIR = os.getenv("IMAGE_DIR")
    TEXT_DIR = os.getenv("TEXT_DIR")
    TABLE_DIR = os.getenv("TABLE_DIR")

    ASSOCIATION_DIR = os.getenv("ASSOCIATION_DIR")
    CRITERIA_EXTRACTION_DIR = os.getenv("CRITERIA_EXTRACTION_DIR")
    ANSWERS_DIR_TRAINING = os.getenv("ANSWERS_DIR_TRAINING")

    for model in models:
        if model == "gpt-5.2":
            print("Vai con gpt che ci piace")
            OPENAI_KEY = os.getenv("OPENAI_KEY")
            client = OpenAI(api_key=OPENAI_KEY)
            # extract_criterias_main(model, client)
            agent = Agent(client, os.path.join(os.path.join(CRITERIA_EXTRACTION_DIR, model), "iteration_0"), MODALITIES)
            entropy_calculation_main(model, agent, MODALITIES, QUESTIONS_MULTIMODALQA_TRAINING, ASSOCIATION_DIR,
                                     IMAGE_DIR, TEXT_DIR, TABLE_DIR, os.path.join(ANSWERS_DIR_TRAINING, model))

        elif model == "claude-sonnet-4-5":
            print("Vai con anthropic che ci piace")
            CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic()
            # extract_criterias_main(model, client)
            agent = Agent(client, os.path.join(os.path.join(CRITERIA_EXTRACTION_DIR, model), "iteration_0"), MODALITIES)
            entropy_calculation_main(model, agent, MODALITIES, QUESTIONS_MULTIMODALQA_TRAINING, ASSOCIATION_DIR,
                                     IMAGE_DIR, TEXT_DIR, TABLE_DIR, os.path.join(ANSWERS_DIR_TRAINING, model))

        elif model == "google":
            GOOGLE_KEY = os.getenv("GOOGLE_KEY")
        elif model == "deepseek-r1:8b":
            print("Ci siamo alla singolarità")

            NEBULA_BASE_URL = os.getenv("NEBULA_BASE_URL")
            NEBULA_API_KEY = os.getenv('NEBULA_KEY')
            client = OpenAI(base_url=NEBULA_BASE_URL, api_key=NEBULA_API_KEY)

            get_nebula_models(client)

            configs = {
                "max_tokens": 200,
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "personal_answer",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "country": {"type": "string"},
                                "capital": {"type": "string"}
                            },
                            "required": ["country", "capital"],
                            "additionalProperties": False
                        }
                    }
                }
            }

            prompt_parameters = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an amazing guy"},
                    {"role": "user", "content": "What is your name?"}
                ],
            }

            prompt_parameters.update(configs)

            print("SHOW ME")
            print(prompt_parameters)

            response = client.chat.completions.create(**prompt_parameters)

            extract_criterias_main(model, client)
        elif model == "meta":
            pass
        elif model == "mistral":
            pass
