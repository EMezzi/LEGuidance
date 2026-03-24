# CRITERIA

# Json Schemas
json_schema_extraction_criteria = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "topic",
        "expected_answer_type",
        "expected_cardinality",
        "target",
        "asked_property",
        "constraints",
        "time_constraints",
        "aliases",
        "rewritten_question",
        "original_question"
    ],
    "properties": {
        "topic": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "question_topic"
            ],
            "properties": {
                "question_topic": {
                    "type": "string",
                    "enum": [
                        "Film",
                        "Transportation",
                        "Video games",
                        "Industry",
                        "Theater",
                        "Television",
                        "Music",
                        "Geography",
                        "Literature",
                        "History",
                        "Economy",
                        "Sports",
                        "Science",
                        "Politics",
                        "Buildings",
                        "Other"
                    ]
                }
            }
        },
        "expected_answer_type": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "expected_answer_type_specific",
                "expected_answer_type_general"
            ],
            "properties": {
                "expected_answer_type_specific": {
                    "type": "string",
                    "minLength": 1
                },
                "expected_answer_type_general": {
                    "type": "string",
                    "minLength": 1
                }
            }
        },
        "expected_cardinality": {
            "type": "string",
            "enum": [
                "single",
                "multiple",
                "unknown"
            ]
        },
        "target": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "text",
                "type"
            ],
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "person",
                        "organization",
                        "place",
                        "event",
                        "award",
                        "work",
                        "other"
                    ]
                }
            }
        },
        "asked_property": {
            "type": "string",
            "minLength": 1
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind",
                    "evidence",
                    "normalized"
                ],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "relation",
                            "award",
                            "league",
                            "season",
                            "time",
                            "role",
                            "qualifier",
                            "domain",
                            "other"
                        ]
                    },
                    "evidence": {
                        "type": "string",
                        "minLength": 1
                    },
                    "normalized": {
                        "type": "string",
                        "minLength": 1
                    }
                }
            }
        },
        "time_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "label",
                    "start_year",
                    "end_year"
                ],
                "properties": {
                    "label": {
                        "type": "string",
                        "minLength": 1
                    },
                    "start_year": {
                        "type": [
                            "integer",
                            "null"
                        ],
                    },
                    "end_year": {
                        "type": [
                            "integer",
                            "null"
                        ],
                    },
                    "start_date": {
                        "type": [
                            "string",
                            "null"
                        ],
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                    },
                    "end_date": {
                        "type": [
                            "string",
                            "null"
                        ],
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
                    }
                }
            }
        },
        "aliases": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "rewritten_question": {
            "type": "string",
            "minLength": 1
        },
        "original_question": {
            "type": "string",
            "minLength": 1
        }
    }
}

json_schema_check_criteria = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "description": "Answer yes or no to whether the data contains the criteria",
            "enum": ["yes", "no"]
        }
    },
    "required": ["answer"],
}


# TABLE DESCRIPTON JSON SCHEMA
json_schema_table_description = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "description": {
            "type": "string",
            "description": "High level description of the table"
        }
    },
    "required": ["description"]
}

# ANSWER SET JSON SCHEMA AND PROMPTS

json_schema_answer_set = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_set": {
            "type": "array",
            "description": "Answer set for that type data modality",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["answer_set"]
}

