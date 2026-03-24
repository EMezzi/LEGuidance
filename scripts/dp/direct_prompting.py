import os
import json
from miscellaneous.prompts.direct_prompting import dp_prompt
from miscellaneous.utils import get_question_data, get_question_files


class DirectPrompting:
    def data_preparation(self, question, images, texts, table, table_dir, final_dataset_images):
        pass

    def dp_function(self, model):
        if model == "gpt-5.2":
            pass
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


def dp_main(model, agent, questions_list, questions_dir, association_dir, table_dir, final_dataset_images):
    for question in questions_list:
        question_data = get_question_data(questions_dir, question)
        question_files = get_question_files(association_dir, question)

        priority_modalities = agent.dp_function(question_data["question_text"],
                                                question_files["image_set"],
                                                question_files["text_set"],
                                                question_files["table_set"][0],
                                                table_dir, final_dataset_images).split('_')

        json.dump({"step_1": priority_modalities},
                  open(os.path.join(f"../results/modalities_predicted/{model}/{question}"), "w"), indent=4)
