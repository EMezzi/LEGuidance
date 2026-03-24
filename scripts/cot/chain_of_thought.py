
class ChainOfThought:
    def data_preparation(self, question):
        images_text = "\n\n".join([
            f"{i + 1}. Title: {img['title']}"
            for i, img in enumerate(images)
        ])

        image_inputs = []
        for img in images:
            image64 = encode_image(os.path.join(final_dataset_images, img["path"]))
            image_inputs.append({
                "title": img["title"],
                "image_url": f"data:image/jpeg;base64,{image64}"
            })

        paragraphs_text = "\n\n".join([
            f"{i + 1}. Title: {p['title']}\nContent: {p['text']}"
            for i, p in enumerate(texts)
        ])

        json_table = json.load(open(os.path.join(table_dir, table["json"]), "rb"))

        tables_text = f"""
            Table Title: {json_table["title"]}
            Table name: {json_table["table"]["table_name"]}
            Content: {json_table["table"]}
            """

    def cot_function(self, model):
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
        agent.data_preparation(question)
        agent.dp_function(model)
