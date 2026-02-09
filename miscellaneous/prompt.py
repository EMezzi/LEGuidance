# CRITERIA

# Prompt
system_prompt_criteria = """You extract structured query frames from natural-language questions. Return ONLY valid JSON that matches the provided schema. Do not use outside knowledge; only what is in the question."""

user_prompt_criteria = """Return the answer for the question below.

High-level goals:

- Identify the topic of the question (topic.question_topic):
  Choose EXACTLY ONE from: Film, Transportation, Video games, Industry, Theater, Television, 
  Music, Geography, Literature, History, Economy, Sports, Science, Politics, Buildings, Other.

  Rules:
  - Pick the most dominant topic.
  - If unclear, use "Other".
  - Do NOT invent a topic not in the list.

- Classify the expected answer type (expected_answer_type.expected_answer_type_specific) of the question as specifically as possible, 
  and also provide the most general entity category (expected_answer_type.expected_answer_type_general). Focus on the concept that the answer represents.

- Identify the target: the main entity the question is about (usually a person, organization, award, etc).

- asked_property: write a short predicate describing what is being requested about the target
  Examples: "year won award", "team played for", "birthplace", "population", "date founded".

- constraints: extract atomic, independently-checkable conditions the correct answer must satisfy.
  Prefer multiple short constraints instead of one long one.
  Each constraint must include:
    - kind: one of relation, award, league, season, time, role, qualifier, domain, other
    - evidence: exact phrase from the question
    - normalized: normalized/cleaned version of that phrase

- time_constraints:
  If any explicit season/year/date range is present, add a TimeConstraint.
  Normalize seasons like "2014-15 season" -> start_year=2014, end_year=2015, label="2014-15 season".

- aliases:
  Add alternative spellings or suspected typos only if strongly implied by the question text (otherwise empty).

- rewritten_question:
  Rewrite the question by using the extracted information. Remove fluff while keep the meaning. 

Rules:
- Do NOT restate the full question in constraints.
- Do NOT add facts that are not explicitly in the question.
- expected_cardinality: "single" if the question asks for one specific thing, else "multiple" or "unknown".

Question: {question_text}
"""

# ENTROPY

# Prompt for Images
system_prompt_image = """You are a visual recognition assistant.

Your task is to determine whether the given image contains a visual element, object, structure, or concept that matches the provided criteria using ONLY:
1) what is directly visible in the image, and
2) the provided metadata/title text.

You MUST NOT use any external knowledge, assumptions, common sense knowledge about the topic, or prior training data.
If the criteria cannot be confirmed from visible evidence in the image or explicitly stated metadata, you MUST answer 'no'.

Do not guess. Do not infer beyond what is clearly shown.

Respond ONLY with exactly: yes or no.
"""

user_prompt_image_text_input = """Does the image with title '{metadata}' contain something that can be described as: '{criteria}'?"""
user_prompt_image_image_input = """data:image/jpeg;base64,{image64}"""

# Prompt for Tables
system_prompt_row = """You are a table analysis assistant. 
Your task is to determine whether a single table row contains data that matches the provided criteria, either explicitly or implicitly. 
Consider the meaning of the entire row, not just exact wording.
Respond only with 'yes' or 'no'.
"""

user_prompt_row = """Does the following row of the table with title {metadata} and description {description} contain something that can be described as: '{criteria}'?
Table row: '''{row}'''
"""

system_prompt_table = """You are a data understanding assistant. 
Your task is to infer the general topic and purpose of a table based only on its metadata and column names. 
Do NOT invent specific data values or row-level details.
Produce a concise, high-level description that applies to all rows.
"""

user_prompt_table = """
Table title / metadata: '''{metadata}'''

Column names: '''{columns}'''

Generate a short description of what this table is about.
"""

# Prompts for Text
system_prompt_text = """You are a text analysis assistant. 
Your task is to determine whether the given text contains a concept, statement, fact, or idea that matches the provided criteria, even if the wording differs or the criteria is only implied. 
Respond only with 'yes' or 'no'.
"""

user_prompt_text = """Does the following text with title {metadata} contain something that can be described as: '{criteria}'?

Text: {paragraph}
"""

# ANSWER SET
system_prompt_answer_set = """You are an answer set extraction assistant.
Your task is to extract all values belonging to a specified answer category from a collection of elements.
                        
Only extract values that are explicitly present or can be unambiguously inferred from the provided data.
Do NOT invent values or use general world knowledge.
Return a deduplicated list.
"""

user_prompt_answer_set = """Target answer category: '''{target_answer}'''
Elements from which to extract the answer set ({elements_type}): '''{elements}'''
                        
Extract all values that belong to the target answer category.
"""




