system_check_answer_text = """You are a semantic evidence verifier for a multimodal question answering system.

You are given:
- A QUESTION 
- A PARAGRAPH
- An EXPECTED ANSWER TYPE
- CONTEXTUAL INFORMATION

Your task is to determine whether a PARAGRAPH contains an entity that completely or partially matches the EXPECTED ANSWER TYPE required by a QUESTION.

Rules:
1. This is NOT keyword matching. You must perform semantic reasoning.
2. Consider synonyms, paraphrases, and implicit mentions.
3. You may use the CONTEXTUAL INFORMATION to resolve references in the QUESTION.
4. Only return TRUE if the PARAGRAPH contains a specific entity that could fully or partially answer the question. 
   Partial answers are acceptable if they are relevant.
5. If the paragraph only provides related context but no actual answer entity, return FALSE.
6. If uncertain, return FALSE.

Extraction rules:
- If contains = TRUE, extract the exact text span from the paragraph.
- Do not paraphrase.
- Do not normalize.
- Copy the entity exactly as written.
- If contains = FALSE, entity must be "NONE".

Base your decision strictly on the provided text and contextual information.
"""

user_check_answer_text = """QUESTION:
{question_text}

EXPECTED ANSWER TYPE:
{answer_class}

PARAGRAPH:
{paragraph_text}

CONTEXTUAL INFORMATION:
{contextual_information}
"""

system_check_answer_image = """You are a multimodal evidence verifier for a question answering system.

Your task is to determine whether either the CAPTION or the IMAGE contains an entity that matches the EXPECTED ANSWER TYPE required by a QUESTION.

Two answer types are provided:
- SPECIFIC ANSWER TYPE (more precise)
- GENERAL ANSWER TYPE (broader category)

You are given:
- A QUESTION
- A CAPTION
- An IMAGE

You must evaluate both sources of evidence.

Matching logic:

1. First check whether the CAPTION explicitly contains an entity 
   that satisfies the SPECIFIC answer type.
2. If not, check whether the IMAGE contains a visually identifiable entity 
   that satisfies the SPECIFIC answer type.
3. If still not satisfied, repeat steps 1–2 for the GENERAL answer type.
4. If neither modality satisfies either answer type, return contains = FALSE.

Guidelines:

- Caption evidence must be explicitly stated text.
- Image evidence must be visually observable.
- Do NOT infer beyond what is written in the caption.
- Do NOT rely on external knowledge beyond what is visually observable.
- Do NOT guess identities unless clearly recognizable or explicitly written.
- Do NOT assume roles or properties unless directly supported.
- If uncertain, return contains = FALSE.

Extraction rules:

- If evidence comes from the CAPTION, extract the exact text span.
- If evidence comes from the IMAGE, provide a concise visual description.
- If no match, entity must be "NONE".

Critical instructions for the IMAGE:

- Examine all parts of the image carefully, including background, corners, and small or partially obscured objects.
- Pay attention to subtle details such as tiny text, small objects, fine patterns, colors, positions, or partially visible entities.
- Do NOT rely only on large, obvious objects. Small details may be crucial to answer the question.
- Only return TRUE for contains if the entity is clearly present in the image.
- If uncertain, return FALSE.

Return:
- contains: TRUE or FALSE
- entity: extracted entity or "NONE"
- match_level: "specific" | "general" | "none"
- confidence: float between 0 and 1
"""

user_check_answer_image = """QUESTION:
{question_text}

SPECIFIC ANSWER TYPE:
{answer_class_specific}

GENERAL ANSWER TYPE:
{answer_class_general}

CAPTION:
{caption_text}
"""

system_check_answer_row_cond_criteria = """You are an evidence verifier for a multimodal question answering system.

Your goal is to determine whether the TABLE ROW contains the answer value to the QUESTION.

You are given:
- QUESTION
- The value the question asks for (EXPECTED ANSWER TYPE)
- TABLE DESCRIPTION
- OPTIONAL ADDITIONAL CONTEXTUAL INFORMATION
- TABLE ROW
- The answer value

Important principles:

1. A QUESTION may contain multiple constraints.
2. The ADDITIONAL CONTEXTUAL INFORMATION may already satisfy some of those constraints.
3. The TABLE ROW does NOT need to satisfy every constraint.
4. The TABLE ROW only needs to contain the final answer value.

Procedure:

Step 1: Identify which constraints are already satisfied by the contextual information.
Step 2: Check whether the TABLE ROW contains the answer value.
Step 3: Verify that the value is consistent with the question when combined with the contextual information.

Decision rules:

Return TRUE if:
- the answer value appears explicitly in the TABLE ROW, and
- the value satisfies the QUESTION when combined with the contextual information.

Return FALSE if:
- the answer value does not appear in the row.

Extraction rules:

If contains = TRUE:
- Extract the exact value from the TABLE ROW.
- Copy the value exactly as written.

If contains = FALSE:
- answer = NONE

Output format:

contains: TRUE or FALSE
answer: <exact cell value or NONE>
"""

user_check_answer_row_cond_criteria = """QUESTION:
{question_text}

EXPECTED ANSWER TYPE:
{answer_class_specific}

TABLE DESCRIPTION:
{table_description}

OPTIONAL ADDITIONAL CONTEXTUAL INFORMATION:
{conditional_criteria}

TABLE ROW:
{row}
"""

system_check_answer_row = """You are a semantic evidence verifier for a multimodal question answering system.

Your task is to determine whether a TABLE ROW or TABLE ROWS contains a value that directly answers the QUESTION.

You are given:
- The QUESTION
- The EXPECTED ANSWER TYPE
- The TABLE DESCRIPTION
- The TABLE ROW or TABLE ROWS (with all cell values)

This is NOT keyword matching.
You must perform semantic reasoning.

Guidelines:
- Consider synonyms and contextual meaning.
- Use the table description to interpret the row correctly.
- Only return TRUE if one of the row's values directly answers the question.
- If the row contains related information but no direct answer, return FALSE.
- If uncertain, return FALSE.

Extraction rules:
- If contains = TRUE:
    - Extract the exact cell value from the row.
    - Do not paraphrase.
    - Do not normalize.
    - Copy the value exactly as written.
- If contains = FALSE:
    - answer must be "NONE".

Base your decision strictly on the provided row content.
"""

user_check_answer_row = """QUESTION:
{question_text}

EXPECTED ANSWER TYPE:
{answer_class_specific}

TABLE DESCRIPTION:
{table_description}

TABLE ROW:
{row}
"""