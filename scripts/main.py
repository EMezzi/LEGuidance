import os
import ast
import json
import boto3
import random
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from miscellaneous.utils import dataset_build, get_questions
from reasoning.criteria_extraction import extract_criterias_main
from direct_prompting.dp import dp_main, DirectPrompting
from reasoning.entropy_calculation import entropy_calculation_main, LEAgent
from botocore.exceptions import ClientError


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
    """
    response = client.invoke_model(
        modelId='mistral.mistral-large-3-675b-instruct',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': 'Can you explain the features of Amazon Bedrock?'}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    response = client.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-6',
        body=json.dumps({
            'anthropic_version': 'bedrock-2023-05-31',  # Required field
            'messages': [{'role': 'user', 'content': 'Can you explain the features of Amazon Bedrock?'}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))
    """

    # Amazon
    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Can you explain the features of Amazon Bedrock?"}]
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
            'messages': [{'role': 'user', 'content': 'Can you explain the features of Amazon Bedrock?'}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    # Qwen
    response = client.invoke_model(
        modelId='qwen.qwen3-vl-235b-a22b',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': 'Can you explain the features of Amazon Bedrock?'}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    print("Kimi")
    response = client.invoke_model(
        modelId='moonshotai.kimi-k2.5',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': 'Can you explain the features of Amazon Bedrock?'}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))

    print("Gemma")
    response = client.invoke_model(
        modelId='google.gemma-3-27b-it',
        body=json.dumps({
            'messages': [{'role': 'user', 'content': 'Can you explain the features of Amazon Bedrock?'}],
            'max_tokens': 1024
        })
    )
    print(json.loads(response['body'].read()))


def import_directories(dataset, setting):
    if dataset == "multimodalqa":
        IMAGE_DIR = os.getenv("IMAGE_DIR")
        TEXT_DIR = os.getenv("TEXT_DIR")
        TABLE_DIR = os.getenv("TABLE_DIR")
        FINAL_DATASET_IMAGES = os.getenv("FINAL_DATASET_IMAGES")

        ASSOCIATION_DIR = os.getenv("ASSOCIATION_DIR")

        QUESTIONS_DIR = os.getenv(f"QUESTIONS_MULTIMODALQA_{setting.upper()}")
        CRITERIA_DIR = os.getenv(f"CRITERIA_MULTIMODALQA_{setting.upper()}")
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
    load_dotenv()

    random.seed(42)
    iteration = "iteration_0"

    dataset, setting = "multimodalqa", "training"
    models = ["gpt-5.2"]
    approach = "dp"
    approaches = ["dp", "cot", "le"]

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))

    """Import directories"""
    if dataset == "multimodalqa":
        (IMAGE_DIR, TEXT_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ASSOCIATION_DIR, QUESTIONS_DIR, CRITERIA_DIR,
         ANSWERS_DIR) = import_directories(dataset, setting)

    elif dataset == "manymodalqa":
        QUESTIONS_DIR, IMAGE_DIR, CRITERIA_DIR, ANSWERS_DIR = import_directories(dataset, setting)

    """Import access keys for amazon"""
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = import_bedrock_credentials()
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    get_inference_profiles()
    amazon_mistral(client)

    """Answering process begins"""
    questions_list = get_questions()

    if approach == "dp":
        dp_agent = DirectPrompting()
        for model in models:
            dp_main(model, dp_agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES,
                    os.path.join(f"{ANSWERS_DIR}/iteration_1", model))

    elif approach == "cot":
        cot_agent = ChainOfThought()
        for model in models:
            dp_main(mode)


    for model in models:
        if model == "gpt-5.2":
            print(f"CHOSEN MODEL: {model}")
            print("***********************************\n")
            OPENAI_KEY = os.getenv("OPENAI_KEY")
            client = OpenAI(api_key=OPENAI_KEY)
            # extract_criterias_main(model, client)
            agent = LEAgent(client, os.path.join(os.path.join(CRITERIA_DIR, model), "iteration_0"), MODALITIES)
            entropy_calculation_main(model, agent,
                                     questions_list[0:1],
                                     QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES,
                                     os.path.join(f"{ANSWERS_DIR}/iteration_1", model))

        elif model == "claude-sonnet-4-6":
            pass
        elif model == "mistral-large-3":
            pass
        elif model == "nova-premier-v1:0":
            pass
        elif model == "nvidia.nemotron-nano-12b-v2":
            pass
        elif model == "qwen3-vl-235b-a22b":
            pass
        elif model == "kimi-k2.5":
            pass
        elif model == "gemma-3-27b-it":
            pass


        """
        elif model == "claude-sonnet-4-6":
            print("Vai con anthropic che ci piace")
            CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic()
            # extract_criterias_main(model, client)
            agent = LEAgent(client, os.path.join(os.path.join(CRITERIA_DIR, model), "iteration_0"), MODALITIES)
            entropy_calculation_main(model, agent, MODALITIES, QUESTIONS_DIR, ASSOCIATION_DIR,
                                     IMAGE_DIR, TEXT_DIR, TABLE_DIR, os.path.join(ANSWERS_DIR, model))
        """
