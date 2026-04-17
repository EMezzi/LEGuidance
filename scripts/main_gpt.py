import os
import ast
import random
from dotenv import load_dotenv

from openai import OpenAI

from utils.utilities import get_questions, import_directories

from dp.direct_prompting import dp_main, DPAgent
from cot.chain_of_thought import cot_main, CoTAgent
from pp.planning_prompting import pp_main, PPAgent

from le.criteria_extraction import CriteriasAgent
from le.entropy_calculation import entropy_calculation_main, LEAgent


if __name__ == "__main__":
    random.seed(42)
    load_dotenv()

    dataset, setting = "multimodalqa", "validation"
    models = ["gpt-5.2"]

    MODALITIES = ast.literal_eval(os.getenv("MODALITIES", "[]"))
    OPENAI_KEY = os.getenv("OPENAI_KEY")

    QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ANSWERS_DIR, CRITERIA_DIR = (None, None, None,
                                                                                                  None, None, None)

    openai_client = OpenAI(api_key=OPENAI_KEY)
    bedrock_client = None

    approaches = ["dp", "pp", "cot", "le"]

    for approach in approaches:

        if dataset == "multimodalqa":
            (IMAGE_DIR, TEXT_DIR, TABLE_DIR, FINAL_DATASET_IMAGES, ASSOCIATION_DIR, QUESTIONS_DIR, CRITERIA_DIR,
             ANSWERS_DIR) = import_directories(dataset, setting, approach)

        elif dataset == "manymodalqa":
            QUESTIONS_DIR, IMAGE_DIR, CRITERIA_DIR, ANSWERS_DIR = import_directories(dataset, setting, approach)

        questions_list = get_questions(QUESTIONS_DIR)

        if approach == "dp":

            print(f"The approach is: {approach}")
            dp_agent = DPAgent(openai_client, bedrock_client)
            for model in models:
                print(f"Model: {model}")
                dp_main(model, dp_agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                        FINAL_DATASET_IMAGES, os.path.join(f"{ANSWERS_DIR}", model))

        elif approach == "cot":

            print(f"The approach is: {approach}")
            cot_agent = CoTAgent(openai_client, bedrock_client)
            for model in models:
                print(f"Model: {model}")
                cot_main(model, cot_agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                         FINAL_DATASET_IMAGES, os.path.join(f"{ANSWERS_DIR}", model))

        elif approach == "pp":

            print(f"The approach is: {approach}")
            pp_agent = PPAgent(openai_client, bedrock_client)
            for model in models:
                print(f"Model: {model}")
                pp_main(model, pp_agent, questions_list, QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                        FINAL_DATASET_IMAGES, os.path.join(f"{ANSWERS_DIR}", model))

        elif approach == "le":
            print(QUESTIONS_DIR, CRITERIA_DIR)
            c_agent = CriteriasAgent(openai_client, bedrock_client)
            print(f"The approach is: {approach}")
            for model in models:
                print(f"Model: {model}")
                print("***********************************\n")
                # extract_criterias_main(c_agent, QUESTIONS_DIR, CRITERIA_DIR, model, questions_list)

                agent = LEAgent(openai_client, bedrock_client, os.path.join(os.path.join(CRITERIA_DIR, model)))
                entropy_calculation_main(model,
                                         agent,
                                         questions_list,
                                         QUESTIONS_DIR, ASSOCIATION_DIR, TABLE_DIR,
                                         FINAL_DATASET_IMAGES,
                                         os.path.join(ANSWERS_DIR, model),
                                         dataset, approach, setting)
