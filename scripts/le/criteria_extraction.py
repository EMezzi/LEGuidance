import os
import sys
from scripts.le.functions_extract_criteria import *
from prompts.le.prompt_criteria_extraction import *

sys.path.append(os.path.dirname(__file__))


class CriteriasAgent:
    def __init__(self, openai_client, bedrock_client):
        self.openai_client = openai_client
        self.bedrock_client = bedrock_client

    def extract_criterias(self, model, question_text, question, criteria_extraction_dir):
        # Add tho the prompt the generation of a set that contains the entire information brought by the sentence.

        if (model == "global.amazon.nova-2-lite-v1:0" or model == "mistral.mistral-large-3-675b-instruct" or
                model == "moonshotai.kimi-k2.5" or model == "qwen.qwen3-vl-235b-a22b" or
                model == "us.anthropic.claude-sonnet-4-6"):
            extract_criterias_amazon(model, self.bedrock_client, system_prompt_criteria, user_prompt_criteria,
                                     question_text, question, criteria_extraction_dir, use_tool=True)

        elif model == "nvidia.nemotron-nano-12b-v2":
            extract_criterias_amazon(model, self.bedrock_client, system_prompt_criteria, user_prompt_criteria,
                                     question_text, question, criteria_extraction_dir, use_tool=False)
        elif model == "gpt-5.2":
            print(question_text)
            extract_criterias_gpt(model, self.openai_client, system_prompt_criteria, user_prompt_criteria,
                                  question_text, question, criteria_extraction_dir)


def extract_criterias_main(ca, questions_dir, criteria_dir, model, questions):
    # The loop is to repeat the criteria extraction multiple times

    os.makedirs(os.path.join(criteria_dir, model), exist_ok=True)

    for i, question in enumerate(questions):
        print("Question: ", i, question)
        if question not in os.listdir(f"{criteria_dir}/{model}/"):
            print("Not present between criterias. So let's recompute it.")
            json_question = json.load(open(os.path.join(questions_dir, question), "rb"))
            question_text = json_question["question"]
            ca.extract_criterias(model, question_text, question, criteria_dir)
