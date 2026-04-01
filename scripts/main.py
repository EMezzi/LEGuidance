import os
import ast
import json
import boto3
import random
from dotenv import load_dotenv

from openai import OpenAI

from utils.utilities import get_questions

from dp.direct_prompting import dp_main, DPAgent
from cot.chain_of_thought import cot_main, CoTAgent
from pp.planning_prompting import pp_main, PPAgent

from le.criteria_extraction import extract_criterias_main
from le.entropy_calculation import entropy_calculation_main, LEAgent


def get_nebula_models(client):
    models = []
    for model in client.models.list().data:
        models.append(model.id)
    return models


def get_inference_profiles():
    # Initialize the Bedrock client
    # Replace 'us-east-1' with your active region

    client = boto3.client(
        "bedrock",
        region_name="us-west-2",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    bedrock = client

    """
    try:
        # This command ONLY works with the "bedrock" client
        response = bedrock.list_inference_profiles(typeEquals='SYSTEM_DEFINED')

        profiles = response.get('inferenceProfileSummaries', [])

        print(f"{'Profile Name':<40} | {'Inference Profile ID':<40}")
        print("-" * 85)

        for profile in profiles:
            print(f"{profile['inferenceProfileName']:<40} | {profile['inferenceProfileId']:<40}")

    except Exception as e:
        print(f"Error: {e}")

    """

    try:
        # List the inference profiles
        response = bedrock.list_inference_profiles()

        print("Available Inference Profiles:")
        for profile in response.get('inferenceProfileSummaries', []):
            print(
                f"Name: {profile.get('inferenceProfileName')} ID: {profile.get('inferenceProfileId')} Status: {profile.get('status')}")

    except Exception as e:
        print(f"Error: {e}")


def amazon_mistral(client):
    response = client.invoke_model(
        modelId='mistral.mistral-large-3-675b-instruct',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': """Can you explain the features of Amazon Bedrock?
            
            Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
            
            Guidelines: Output only the json without any explanations.
            """}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    response = client.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-6',
        body=json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',  # Required field
            'messages': [
                {
                    'role': 'user',
                    'content': """Can you explain the features of Amazon Bedrock?
                    
                    Guidelines: Output only the json without any explanations.
                    Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
                    """
                }
            ],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    # Amazon
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [{"text": """Can you explain the features of Amazon Bedrock?  
                
                Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
                """}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 1024
        }
    })

    response = client.invoke_model(
        modelId='us.amazon.nova-premier-v1:0',
        body=body
    )

    print(json.loads(response['body'].read()))

    # Nvidia
    response = client.invoke_model(
        modelId='nvidia.nemotron-nano-12b-v2',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': """Can you explain the features of Amazon Bedrock?
            
            Guidelines: Output only the json without any explanations.
            Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
            """}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    # Qwen
    response = client.invoke_model(
        modelId='qwen.qwen3-vl-235b-a22b',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': """Can you explain the features of Amazon Bedrock?
            
            Guidelines: Output only the json without any explanations.
            Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
            """}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    print("Kimi")
    response = client.invoke_model(
        modelId='moonshotai.kimi-k2.5',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': """Can you explain the features of Amazon Bedrock?
            
            Guidelines: Output only the json without any explanations.
            Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
            """}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    print("Gemma")
    response = client.invoke_model(
        modelId='google.gemma-3-27b-it',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': """Can you explain the features of Amazon Bedrock?
            
            Guidelines: Output only the json without any explanations.
            Output Json with: topic (the topic of your explanation), explanation (the actual explanation)
            """}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))


def import_directories(dataset, setting, approach):
    if dataset == "multimodalqa":
        IMAGE_DIR = os.getenv("IMAGE_DIR")
        TEXT_DIR = os.getenv("TEXT_DIR")
        TABLE_DIR = os.getenv("TABLE_DIR")
        FINAL_DATASET_IMAGES = os.getenv("FINAL_DATASET_IMAGES")

        ASSOCIATION_DIR = os.getenv(f"ASSOCIATION_MULTIMODALQA_{setting.upper()}")
        QUESTIONS_DIR = os.getenv(f"QUESTIONS_MULTIMODALQA_{setting.upper()}")
        CRITERIA_DIR = os.getenv(f"CRITERIA_MULTIMODALQA_{setting.upper()}")

        ANSWERS_DIR = "dp"
        if approach == "dp":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_VALIDATION_DP")
        elif approach == "cot":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_VALIDATION_COT")
        elif approach == "pp":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_VALIDATION_PP")
        elif approach == "le":
            ANSWERS_DIR = os.getenv(f"ANSWERS_MULTIMODALQA_{setting.upper()}")

        return IMAGE_DIR, TEXT_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ASSOCIATION_DIR, QUESTIONS_DIR, CRITERIA_DIR, ANSWERS_DIR

    elif dataset == "manymodalqa":
        QUESTIONS_DIR = os.getenv(f"QUESTIONS_MANYMODALQA_{setting.upper()}")
        IMAGE_DIR = os.getenv(f"QUESTIONS_MANYMODALQA_IMAGES")

        CRITERIA_DIR = os.getenv(f"CRITERIA_MANYMODALQA_{setting.upper()}")
        ANSWERS_DIR = os.getenv(f"ANSWERS_MANYMODALQA_{setting.upper()}")

        return QUESTIONS_DIR, IMAGE_DIR, CRITERIA_DIR, ANSWERS_DIR


def import_bedrock_credentials():
    aws_access_key_id = os.getenv("aws_access_key_id")
    aws_secret_access_key = os.getenv("aws_secret_access_key")

    return aws_access_key_id, aws_secret_access_key


if __name__ == "__main__":
    random.seed(42)
    load_dotenv()

    dataset, setting = "multimodalqa", "validation"
    models = ["gpt-5.2",
              "global.amazon.nova-2-lite-v1:0",
              "mistral.mistral-large-3-675b-instruct",
              "moonshotai.kimi-k2.5",
              "nvidia.nemotron-nano-12b-v2",
              "qwen.qwen3-vl-235b-a22b",
              "us.anthropic.claude-sonnet-4-6"]

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))
    OPENAI_KEY = os.getenv("OPENAI_KEY")

    QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ANSWERS_DIR = None, None, None, None, None

    """Import access keys for amazon"""
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = import_bedrock_credentials()
    bedrock_client = boto3.client("bedrock-runtime",
                                  region_name="us-west-2",
                                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    openai_client = OpenAI(api_key=OPENAI_KEY)

    # print(questions_list[:10])

    # get_inference_profiles()
    # amazon_mistral(bedrock_client)

    approaches = ["dp", "pp", "cot", "le"]

    for approach in approaches[1:3]:

        if dataset == "multimodalqa":
            (IMAGE_DIR, TEXT_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ASSOCIATION_DIR, QUESTIONS_DIR, CRITERIA_DIR,
             ANSWERS_DIR) = import_directories(dataset, setting, approach)

        elif dataset == "manymodalqa":
            QUESTIONS_DIR, IMAGE_DIR, CRITERIA_DIR, ANSWERS_DIR = import_directories(dataset, setting, approach)

        # questions_list = get_questions(QUESTIONS_DIR)
        questions_list = ['question_705.json', 'question_2142.json', 'question_2216.json', 'question_705.json',
                          'question_2142.json', 'question_2216.json', 'question_1203.json', 'question_413.json',
                          'question_705.json', 'question_2142.json', 'question_2216.json']

        if approach == "dp":

            print(f"The approach is: {approach}")
            dp_agent = DPAgent(openai_client, bedrock_client)
            for model in models[1:]:
                print(f"Model: {model}")
                dp_main(model, dp_agent, questions_list[:500], QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                        FINAL_DATASET_IMAGES, os.path.join(f"{ANSWERS_DIR}", model))

        elif approach == "cot":

            print(f"The approach is: {approach}")
            cot_agent = CoTAgent(openai_client, bedrock_client)
            for model in models[1:]:
                print(f"Model: {model}")
                cot_main(model, cot_agent, questions_list[:500], QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                         FINAL_DATASET_IMAGES, os.path.join(f"{ANSWERS_DIR}", model))

        elif approach == "pp":

            print(f"The approach is: {approach}")
            pp_agent = PPAgent(openai_client, bedrock_client)
            for model in models[1:]:
                print(f"Model: {model}")
                pp_main(model, pp_agent, questions_list[:500], QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                        FINAL_DATASET_IMAGES, os.path.join(f"{ANSWERS_DIR}", model))

        elif approach == "le":
            print(f"The approach is: {approach}")
            for model in models:
                if model == "gpt-5.2":
                    print(f"CHOSEN MODEL: {model}")
                    print("***********************************\n")
                    OPENAI_KEY = os.getenv("OPENAI_KEY")
                    client = OpenAI(api_key=OPENAI_KEY)
                    extract_criterias_main(model, client)


                    agent = LEAgent(openai_client, os.path.join(os.path.join(CRITERIA_DIR, model), "iteration_0"),
                                    MODALITIES)
                    entropy_calculation_main(model,
                                             agent,
                                             questions_list[2:3],
                                             QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES,
                                             os.path.join(f"{ANSWERS_DIR}/iteration_1", model), dataset, approach,
                                             setting)
