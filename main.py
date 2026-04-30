import os
import argparse
import random
import logging
from dotenv import load_dotenv

from openai import OpenAI
import boto3

from utils.utilities import get_questions, import_directories

# Approaches
from scripts.dp.direct_prompting import dp_main, DPAgent
from scripts.cot.chain_of_thought import cot_main, CoTAgent
from scripts.pp.planning_prompting import pp_main, PPAgent
from scripts.le.criteria_extraction import CriteriasAgent, extract_criterias_main
from scripts.le.entropy_calculation import entropy_calculation_main, LEAgent


# ----------------------------
# Logging
# ----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EDMA Experiment Runner")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["multimodalqa", "manymodalqa"])

    parser.add_argument("--setting", type=str, default="validation")

    parser.add_argument("--approach", type=str, default="all",
                        choices=["dp", "cot", "pp", "le", "all"])

    parser.add_argument("--models", type=str, nargs="+", required=True)

    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "bedrock"])

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for debugging)")

    return parser.parse_args()


# ----------------------------
# Clients
# ----------------------------
def init_openai():
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OPENAI_KEY not set")
    return OpenAI(api_key=api_key)


def init_bedrock():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key")
    )


# ----------------------------
# Load dataset
# ----------------------------
def load_data(dataset, setting, approach):
    dirs = import_directories(dataset, setting, approach)
    (
        IMAGE_DIR,
        TEXT_DIR,
        TABLE_DIR,
        FINAL_DATASET_IMAGES,
        ASSOCIATION_DIR,
        QUESTIONS_DIR,
        CRITERIA_DIR,
        ANSWERS_DIR,
    ) = dirs

    questions = get_questions(dataset, QUESTIONS_DIR)

    return {
        "IMAGE_DIR": IMAGE_DIR,
        "TEXT_DIR": TEXT_DIR,
        "TABLE_DIR": TABLE_DIR,
        "FINAL_DATASET_IMAGES": FINAL_DATASET_IMAGES,
        "ASSOCIATION_DIR": ASSOCIATION_DIR,
        "QUESTIONS_DIR": QUESTIONS_DIR,
        "CRITERIA_DIR": CRITERIA_DIR,
        "ANSWERS_DIR": ANSWERS_DIR,
        "questions": questions,
    }


# ----------------------------
# Run approaches
# ----------------------------
def run_dp(agent, model, data):
    dp_main(
        data["dataset"],
        model,
        agent,
        data["questions"],
        data["QUESTIONS_DIR"],
        data["ASSOCIATION_DIR"],
        data["TABLE_DIR"],
        data["FINAL_DATASET_IMAGES"],
        os.path.join(data["ANSWERS_DIR"], model),
    )


def run_cot(agent, model, data):
    cot_main(
        data["dataset"],
        model,
        agent,
        data["questions"],
        data["QUESTIONS_DIR"],
        data["ASSOCIATION_DIR"],
        data["TABLE_DIR"],
        data["FINAL_DATASET_IMAGES"],
        os.path.join(data["ANSWERS_DIR"], model),
    )


def run_pp(agent, model, data):
    pp_main(
        data["dataset"],
        model,
        agent,
        data["questions"],
        data["QUESTIONS_DIR"],
        data["ASSOCIATION_DIR"],
        data["TABLE_DIR"],
        data["FINAL_DATASET_IMAGES"],
        os.path.join(data["ANSWERS_DIR"], model),
    )


def run_le(openai_client, bedrock_client, model, data):
    criteria_agent = CriteriasAgent(openai_client, bedrock_client)

    # Step 1 (optional for submission reproducibility)
    extract_criterias_main(criteria_agent, data["QUESTIONS_DIR"], data["CRITERIA_DIR"], model, data["questions"])

    le_agent = LEAgent(
        openai_client,
        bedrock_client,
        os.path.join(data["CRITERIA_DIR"], model),
    )

    entropy_calculation_main(
        model,
        le_agent,
        data["questions"],
        data["QUESTIONS_DIR"],
        data["ASSOCIATION_DIR"],
        data["TABLE_DIR"],
        data["FINAL_DATASET_IMAGES"],
        os.path.join(data["ANSWERS_DIR"], model),
        data["dataset"],
        "le",
        data["setting"],
    )


# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    load_dotenv()

    args = parse_args()

    random.seed(42)

    logging.info(f"Running dataset={args.dataset}, approach={args.approach}")

    # Init clients
    openai_client = init_openai() if args.backend == "openai" else None
    bedrock_client = init_bedrock() if args.backend == "bedrock" else None

    approaches = ["dp", "cot", "pp", "le"] if args.approach == "all" else [args.approach]

    for approach in approaches:
        logging.info(f"Approach: {approach}")

        data = load_data(args.dataset, args.setting, approach)
        data["dataset"] = args.dataset
        data["setting"] = args.setting

        if args.limit:
            data["questions"] = data["questions"][:args.limit]

        for model in args.models:
            logging.info(f"Model: {model}")

            if approach == "dp":
                agent = DPAgent(openai_client, bedrock_client)
                run_dp(agent, model, data)

            elif approach == "cot":
                agent = CoTAgent(openai_client, bedrock_client)
                run_cot(agent, model, data)

            elif approach == "pp":
                agent = PPAgent(openai_client, bedrock_client)
                run_pp(agent, model, data)

            elif approach == "le":
                run_le(openai_client, bedrock_client, model, data)


if __name__ == "__main__":
    main()
