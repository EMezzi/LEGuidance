system_prompt_text_bridge = """You are a reasoning assistant.

You are given: 
- A QUESTION
- Two paragraphs: PARAGRAPH 1 AND PARAGRAPH 2

Your task is to determine whether any entity, concept, or fact mentioned in PARAGRAPH 1 is also present in PARAGRAPH 2.

Rules:
1) Use only the text provided.
2) Do NOT require full fact matching.
3) If any shared entity, name, or concept appears in both paragraphs, respond "yes".

Respond ONLY with "yes" or "no".
"""

user_prompt_text_bridge = """QUESTION:
{question_text}

PARAGRAPH 1: 
{criteria}

TITLE PARAGRAPH 2: 
{title}

PARAGRAPH 2:
{text}

Task:
Determine whether any entity, concept, or fact mentioned in PARAGRAPH 1 is also present in PARAGRAPH 2.

Answer "yes" or "no".
"""


system_prompt_image_bridge = """You are a reasoning assistant.

You are given: 
- A QUESTION
- One PARAGRAPH, one IMAGE, one IMAGE TITLE.

Your task is to determine whether any entity, concept, or fact mentioned in PARAGRAPH is also present in IMAGE or IMAGE TITLE.

Rules:
1) Use only the text and image provided.
2) Do NOT require full fact matching.
3) If any shared entity, name, or concept appears in the paragraph and in the image title or in the paragraph and in the image, respond "yes".

Respond ONLY with "yes" or "no"."""

user_prompt_image_bridge_text = """QUESTION: 
{question_text}

PARAGRAPH: 
{criteria}

IMAGE TITLE: 
{image_title}

Task:
Determine whether any entity, concept, or fact mentioned in PARAGRAPH is also present in the IMAGE TITLE or in the IMAGE.

Answer "yes" or "no".
"""

system_prompt_row_bridge = """You are a reasoning assistant.

You are given: 
- A QUESTION
- One PARAGRAPH, one TABLE NAME, one TABLE DESCRIPTION, one TABLE ROW 

Your task is to determine whether any entity, concept, or fact mentioned in the PARAGRAPH is also present in the TABLE ROW.

Rules:
1) Use only the text provided.
2) To understand the context consider the TABLE DESCRIPTION and TABLE NAME.
2) Do NOT require full fact matching.
3) If any shared entity, name, or concept appears in both the paragraph and the table row, respond "yes".

Respond ONLY with "yes" or "no".
"""

user_prompt_row_bridge = """QUESTION:
{question_text}

PARAGRAPH: 
{criteria}

TABLE NAME: 
{table_name}

TABLE DESCRIPTION:
{table_description}

TABLE ROW: 
{table_row}

Task:
Determine whether any entity, concept, or fact mentioned in PARAGRAPH is also present in TABLE ROW.

Answer "yes" or "no".
"""