# CRITERIA

# Prompt
system_prompt_criteria = """You extract structured query frames from natural-language questions. Return ONLY valid 
JSON that matches the provided schema. Do not use outside knowledge; only what is in the question."""

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
