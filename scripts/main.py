import os
import ast
import random
import argparse
from dotenv import load_dotenv

from openai import OpenAI
import boto3

from utils.utilities import get_questions, import_directories

from dp.direct_prompting import dp_main, DPAgent
from cot.chain_of_thought import cot_main, CoTAgent
from pp.planning_prompting import pp_main, PPAgent

from le.criteria_extraction import CriteriasAgent, extract_criterias_main
from le.entropy_calculation import entropy_calculation_main, LEAgent


# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EDMR Experiment Runner")

    parser.add_argument("--dataset", type=str, default="multimodalqa",
                        choices=["multimodalqa", "manymodalqa"])

    parser.add_argument("--approach", type=str, default="all",
                        choices=["dp", "cot", "pp", "le", "all"])

    parser.add_argument("--models", type=str, nargs="+",
                        default=["gpt-5.2"])

    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "bedrock"],
                        help="Which backend to use")

    return parser.parse_args()


# ----------------------------
# Bedrock credentials
# ----------------------------
def import_bedrock_credentials():
    aws_access_key_id = os.getenv("aws_access_key_id")
    aws_secret_access_key = os.getenv("aws_secret_access_key")
    return aws_access_key_id, aws_secret_access_key


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    random.seed(42)
    load_dotenv()

    args = parse_args()

    dataset = args.dataset
    setting = args.setting
    models = args.models

    if args.approach == "all":
        approaches = ["dp", "pp", "cot", "le"]
    else:
        approaches = [args.approach]

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    openai_client = None
    bedrock_client = None

    if args.backend in ["openai"]:
        openai_client = OpenAI(api_key=OPENAI_KEY)

    if args.backend in ["bedrock"]:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY = import_bedrock_credentials()

        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

    for approach in approaches:

        if dataset == "multimodalqa":
            (IMAGE_DIR, TEXT_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ASSOCIATION_DIR, QUESTIONS_DIR, CRITERIA_DIR,
             ANSWERS_DIR) = import_directories(dataset, setting, approach)

        else:
            QUESTIONS_DIR, IMAGE_DIR, CRITERIA_DIR, ANSWERS_DIR = import_directories(dataset, setting, approach)

        questions_list = get_questions(QUESTIONS_DIR)

        # ----------------------------
        # DP
        # ----------------------------
        if approach == "dp":
            print(f"[DP] Running approach")

            agent = DPAgent(openai_client, bedrock_client)

            for model in models:
                print(f"Model: {model}")

                dp_main(model, agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES,
                        os.path.join(ANSWERS_DIR, model))

        # ----------------------------
        # CoT
        # ----------------------------
        elif approach == "cot":
            print(f"[CoT] Running approach")

            agent = CoTAgent(openai_client, bedrock_client)

            for model in models:
                print(f"Model: {model}")

                cot_main(model, agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES,
                         os.path.join(ANSWERS_DIR, model))


        elif approach == "pp":
            print(f"[PP] Running approach")

            agent = PPAgent(openai_client, bedrock_client)

            for model in models:
                print(f"Model: {model}")

                pp_main(model, agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES,
                        os.path.join(ANSWERS_DIR, model))

        elif approach == "le":
            print(f"[LE] Running approach")

            criteria_agent = CriteriasAgent(openai_client, bedrock_client)

            for model in models:
                print(f"Model: {model}")
                print("***********************************")

                # Step 1: extract criteria
                extract_criterias_main(criteria_agent, QUESTIONS_DIR, CRITERIA_DIR, model, questions_list)

                # Step 2: entropy / reasoning
                le_agent = LEAgent(openai_client, bedrock_client, os.path.join(CRITERIA_DIR, model))

                entropy_calculation_main(model, le_agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                                         FINAL_DATASET_IMAGES, os.path.join(ANSWERS_DIR, model), dataset, approach,
                                         setting)
