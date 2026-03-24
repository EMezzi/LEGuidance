dp_prompt = """

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

response = self.client.responses.parse(
    model="gpt-5.2",
    input=[
        {
            "role": "system",
            "content": system_prompt_modality
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_prompt_modality.format(question=question,
                                                        images_text=images_text,
                                                        paragraphs_text=paragraphs_text,
                                                        tables_text=tables_text)
                },
                *[
                    {"type": "input_image", "image_url": img["image_url"]} for img in image_inputs
                ]
            ]
        }
    ],
    text_format=ModalityDecision,
)

return response.output_parsed.modalities
"""