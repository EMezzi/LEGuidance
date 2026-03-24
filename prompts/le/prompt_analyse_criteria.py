"*******IMAGE*******"""

system_prompt_image = """You are a visual reasoning assistant.

Your task is to determine whether a given IMAGE or its TITLE contains evidence of any facts expressed in the CRITERIA.

Rules:
1) Only use what is directly visible in the image and the provided metadata/title text.
2) Do NOT use external knowledge or assumptions.
3) The image only needs to contain evidence of **any of the provided facts**, not the full original criteria.
4) If at least one fact is present in the image or metadata, respond "yes". Otherwise, respond "no".

Respond ONLY with "yes" or "no".
"""

user_prompt_image_text = """Given:
- Image TITLE: '{metadata}'
- CRITERIA: {criteria}

Task:
Determine whether the image or its metadata/title contains visual evidence of any facts expressed in the criteria.

Answer "yes" if at least one fact is present, otherwise answer "no".
"""

user_prompt_image_image = """data:image/jpeg;base64,{image64}"""

"*******TEXT*******"""
system_prompt_text = """You are a reasoning assistant.

Your task is to determine whether any of the concepts contained in CRITERIA are also contained in PARAGRAPH or its TITLE.

Rules:
1) Use only information explicitly stated in CRITERIA, the PARAGRAPH and its TITLE.
2) Do NOT use external knowledge, assumptions, or inference beyond what is written.
4) If at least one fact is explicitly supported by the PARAGRAPH, or METADATA/TITLE, respond "yes".

Respond ONLY with "yes" or "no"."""

user_prompt_text = """TITLE: 
{metadata}

PARAGRAPH:
{text}

CRITERIA: 
{criteria}

Task:
Determine whether the content of the PARAGRAPH or TITLE contains any concept expressed in the CRITERIA.

Answer "yes" or "no".
"""


"""*****TABLE - ROW*****"""

system_prompt_table = """You are a data understanding assistant. 
Your task is to infer the general topic and purpose of a table based only on its metadata and column names. 
Do NOT invent specific data values or row-level details.
Produce a concise, high-level description that applies to all rows.
"""

user_prompt_table = """
Table title: '''{table_title}'''

Table name: '''{table_name}'''

Column names: '''{columns}'''

Generate a short description of what this table is about.
"""


system_prompt_row = """You are a reasoning assistant.

Your task it to determine whether any of the concepts contained in CRITERIA also contained in TABLE ROW.

Rules:
1) Use only information explicitly stated in the table row.
2) Do NOT use external knowledge, assumptions, or inference beyond what is written.
3) If at least one fact is explicitly supported by the row, title, or description, respond "yes".

Respond ONLY with "yes" or "no".
"""

user_prompt_row = """TABLE ROW:
{row}

CRITERIA:
{criteria}

Task:
Determine whether the content of the TABLE ROW contains any concept expressed in the CRITERIA.

Answer "yes" or "no".
"""