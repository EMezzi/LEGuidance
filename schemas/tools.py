# Tool criteria Extraction

criteria_tool = {
    "toolSpec": {
        "name": "DistinctionCriteria",
        "description": "Extracts structured distinction criteria and constraints from a natural language question.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "object",
                        "properties": {
                            "question_topic": {
                                "type": "string",
                                "enum": ["Film", "Transportation", "Video games", "Industry", "Theater", "Television",
                                         "Music", "Geography", "Literature", "History", "Economy", "Sports", "Science",
                                         "Politics", "Buildings", "Other"],
                                "description": "Topic of the question"
                            }
                        },
                        "required": ["question_topic"]
                    },
                    "expected_answer_type": {
                        "type": "object",
                        "properties": {
                            "expected_answer_type_specific": {"type": "string",
                                                              "description": "Specific type of the expected answer"},
                            "expected_answer_type_general": {"type": "string",
                                                             "description": "Generic type of the expected answer"}
                        },
                        "required": ["expected_answer_type_specific", "expected_answer_type_general"]
                    },
                    "expected_cardinality": {
                        "type": "string",
                        "enum": ["single", "multiple", "unknown"],
                        "default": "single"
                    },
                    "target": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Main entity the question is about."},
                            "type": {"type": "string", "description": "Type of target entity."}
                        },
                        "required": ["text"]
                    },
                    "asked_property": {
                        "type": "string",
                        "description": "Short predicate describing what is asked (e.g., 'team played for')."
                    },
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string",
                                         "enum": ["relation", "award", "league", "season", "time", "role", "qualifier",
                                                  "domain", "other"]},
                                "evidence": {"type": "string", "description": "Exact phrase from the question."},
                                "normalized": {"type": "string",
                                               "description": "Cleaned/normalized version of evidence."}
                            },
                            "required": ["kind", "evidence", "normalized"]
                        }
                    },
                    "time_constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Time phrase from the question."},
                                "start_year": {"type": "integer"},
                                "end_year": {"type": "integer"},
                                "start_date": {"type": "string", "description": "ISO-8601 (YYYY-MM-DD)"},
                                "end_date": {"type": "string", "description": "ISO-8601 (YYYY-MM-DD)"}
                            },
                            "required": ["label"]
                        }
                    },
                    "aliases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "reason": {"type": "string", "description": "e.g. typo, nickname, abbreviation"}
                            },
                            "required": ["text", "reason"]
                        }
                    },
                    "rewritten_question": {
                        "type": "string",
                        "description": "Minimal rewrite preserving meaning."
                    }
                },
                "required": [
                    "topic",
                    "expected_answer_type",
                    "expected_cardinality",
                    "target",
                    "asked_property",
                    "constraints",
                    "time_constraints",
                    "aliases",
                    "rewritten_question"
                ]
            }
        }
    }
}

# Tool to decide modality
modality_tool = {
    "toolSpec": {
        "name": "ModalityDecision",
        "description": "Predicts the best combination of modalities (image, text, table) to answer a specific question.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "modalities": {
                        "type": "string",
                        "enum": [
                            "image",
                            "text",
                            "table",
                            "image_text",
                            "image_table",
                            "text_table"
                        ],
                        "description": "The predicted combination of modalities required for the task."
                    }
                },
                "required": ["modalities"]
            }
        }
    }
}

# YesNoTool
yes_no_tool = {
    "toolSpec": {
        "name": "YesNoQuestion",
        "description": "Determines if a question is a binary Yes/No question and provides a confidence score.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_yes_no": {
                        "type": "boolean",
                        "description": "Whether the question is a yes or no question"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score between 0 and 1"
                    }
                },
                "required": ["is_yes_no", "confidence"]
            }
        }
    }
}

# Is comparison tool
is_comparison_tool = {
    "toolSpec": {
        "name": "IsComparison",
        "description": "Determines if a question requires comparing multiple entities and identifies the count of elements involved.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_comparison": {
                        "type": "boolean",
                        "description": "Whether the question requires a comparison"
                    },
                    "num_elements": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of elements/entities that must be compared."
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score of the classification from 0.0 to 1.0."
                    }
                },
                "required": ["is_comparison", "num_elements", "confidence"]
            }
        }
    }
}

# Is graphical tool
is_graphical_tool = {
    "toolSpec": {
        "name": "IsGraphical",
        "description": "Determines if a question requires analyzing visual/graphical elements (images, charts, diagrams) to be answered.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_graphical": {
                        "type": "boolean",
                        "description": "Whether you need to analyse a graphical element to answer a question."
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score of the visual requirement detection."
                    }
                },
                "required": ["is_graphical", "confidence"]
            }
        }
    }
}

# Analyse criteria text
analyse_criteria_tool = {
    "toolSpec": {
        "name": "AnswerContainsCriteria",
        "description": "Determines if a specific piece of data meets the required criteria and returns a yes or no answer.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Answer yes or no to whether the data contains the criteria"
                    }
                },
                "required": ["answer"]
            }
        }
    }
}


# Analyse table
table_description_tool = {
    "toolSpec": {
        "name": "TableDescription",
        "description": "Generates a high-level summary of a database table based on its schema or sample data.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "High level description of the table"
                    }
                },
                "required": ["description"]
            }
        }
    }
}


# Extract restricting criteria paragraph

paragraph_extraction_tool = {
    "toolSpec": {
        "name": "ParagraphExtraction",
        "description": "Analyzes a paragraph for relevance to a question and extracts precise evidence if found.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_relevant": {
                        "type": "boolean",
                        "description": "Whether the paragraph contains information relevant to the question."
                    },
                    "evidence": {
                        "type": ["string", "null"],
                        "description": (
                            "A minimal, precise textual description of the paragraph that directly "
                            "connects it to the question. Must only include explicitly stated information. "
                            "Should be null if is_relevant is False."
                        )
                    }
                },
                "required": ["is_relevant"]
            }
        }
    }
}

image_extraction_tool = {
    "toolSpec": {
        "name": "ImageExtraction",
        "description": "Determines if an image is relevant to a question and provides textual evidence.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_relevant": {
                        "type": "boolean",
                        "description": "Whether the image contains information relevant to the question."
                    },
                    "evidence": {
                        "type": "string",
                        "description": "A minimal, precise textual description of the image that directly connects it to the question. Must only include explicitly stated information. Should be null if is_relevant is False."
                    }
                },
                "required": ["is_relevant"]
            }
        }
    }
}

# Extract restricting criteria table row
table_row_extraction_tool = {
    "toolSpec": {
        "name": "TableRowExtraction",
        "description": "Evaluates a specific row from a table to determine if it answers a question and extracts the relevant factual evidence.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "is_relevant": {
                        "type": "boolean",
                        "description": "Whether the table row contains information relevant to the question."
                    },
                    "evidence": {
                        "type": ["string", "null"],
                        "description": (
                            "A minimal, precise textual description of the table row that directly "
                            "connects it to the question. Must only include explicitly stated information. "
                            "Should be null if is_relevant is False."
                        )
                    }
                },
                "required": ["is_relevant"]
            }
        }
    }
}

# Image contains answer
image_contains_answer_tool = {
    "toolSpec": {
        "name": "ImageContainsAnswer",
        "description": "Analyzes an image to determine if it contains an entity that matches the expected answer type and extracts that entity.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "contains": {
                        "type": "boolean",
                        "description": "Whether the image describes an entity matching the expected answer type."
                    },
                    "entity": {
                        "type": "string",
                        "description": "The answer for the question extracted from the image."
                    },
                    "match_level": {
                        "type": "string",
                        "enum": ["specific", "general", "none"],
                        "description": "The level of match between the image content and the expected answer type."
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "The confidence score of the extraction, between 0 and 1."
                    }
                },
                "required": ["contains", "entity", "match_level", "confidence"]
            }
        }
    }
}

# Contains answer paragraph
paragraph_contains_answer_tool = {
    "toolSpec": {
        "name": "ParagraphContainsAnswer",
        "description": "Checks a paragraph for a specific entity type and extracts the matching text span and confidence score.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "contains": {
                        "type": "boolean",
                        "description": "Whether the paragraph contains an element matching the expected answer type"
                    },
                    "entity": {
                        "type": "string",
                        "description": "The exact text span from the paragraph that matches the expected answer type, or NONE"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "The confidence score for the extraction.",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["contains", "entity", "confidence"]
            }
        }
    }
}

# Contains answer row
row_contains_answer_tool = {
    "toolSpec": {
        "name": "RowContainsAnswer",
        "description": "Analyzes a table row to identify a specific entity type, extracting the text span and a confidence score.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "contains": {
                        "type": "boolean",
                        "description": "Whether the row contains an element matching the expected answer type"
                    },
                    "entity": {
                        "type": "string",
                        "description": "The exact text span from the row cell that matches the expected answer type, or NONE"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "The confidence score for the extraction.",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["contains", "entity", "confidence"]
            }
        }
    }
}