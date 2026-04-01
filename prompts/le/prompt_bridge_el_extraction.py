"""****TEXT****"""

system_restricting_text = """You are a reasoning module for a multimodal question answering system.

Your task is to extract relevant information from this PARAGRAPH in relation to the QUESTION received. 

You are given:
- The QUESTION
- The PARAGRAPH TITLE
- The PARAGRAPH

Step 1 - Question analysis: 
- Given the QUESTION, the PARAGRAPH TITLE, and the PARAGRAPH check if the information in the PARAGRAPH is related to the QUESTION.

Step 2 - Evidence extraction: 
- If the PARAGRAPH contains information related to the QUESTION, create a short textual description of the PARAGRAPH taking into account the QUESTION. The description must be minimal and contain the entity/value that connects the PARAGRAPH to the QUESTION.

Guidelines:
- In the description only include information explicitly present in the paragraph.
- Do not infer unsupported facts.
- Be concise and precise.
"""

user_restricting_text = """QUESTION:
{question_text}

PARAGRAPH TITLE:
{title}

PARAGRAPH: 
{text}

Provide the relevant information according to the instructions.
"""

"""****IMAGE****"""

system_restricting_image = """You are a reasoning module for a multimodal question answering system.

Your task is to extract relevant information from this IMAGE in relation to the QUESTION received. 

You are given:
- The QUESTION
- The IMAGE TITLE
- The IMAGE

Step 1 - Question analysis: 
- Given the QUESTION, the IMAGE TITLE, and the IMAGE check if the information in the IMAGE is related to the QUESTION.

Step 2 - Evidence extraction: 
- If the IMAGE contains information related to the QUESTION, create a short textual description of the IMAGE taking into account the QUESTION. The description must be minimal and contain the entity/value that connects the IMAGE to the QUESTION.

Guidelines:
- In the description only include information explicitly present in the image.
- Do not infer unsupported facts.
- Be concise and precise.
"""

user_restricting_image_text = """QUESTION:
{question_text}

IMAGE TITLE:
{image_title}

Provide the relevant information according to the instructions.
"""

"""****TABLE****"""
system_restricting_table_row = """You are a reasoning module for a multimodal question answering system.

Your task is to extract relevant information from this TABLE ROW in relation to the QUESTION received. 

You are given: 
- The QUESTION
- The DOCUMENT TITLE of the document containing the table
- The TABLE NAME
- The TABLE description
- The TABLE ROW (with all cell values)

Step 1 - Question analysis: 
- Given the QUESTION, the TABLE NAME, the TABLE DESCRIPTION, and the TABLE ROW check if the information in the TABLE ROW is related to the QUESTION

Step 2 - Evidence extraction: 
- If the TABLE ROW contains information related to the QUESTION, create a short textual description of the TABLE ROW taking into account the QUESTION. The description must be minimal and contain the entity/value that connects the TABLE ROW to the QUESTION

Guidelines:
- In the description only include information explicitly present in the table row.
- Do not infer unsupported facts.
- Be concise and precise.
"""

user_restricting_table_row = """QUESTION:
{question_text}

DOCUMENT TITLE:
{document_title}

TABLE NAME:
{table_name}

TABLE DESCRIPTION: 
{table_description}

TABLE ROW: 
{table_row}

Provide the relevant information according to the instructions.
"""