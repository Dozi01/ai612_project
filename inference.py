import os
import json
from utils import load_schema, create_schema_prompt, post_process_sql, post_process_score
from utils import write_json as write_label

from llm_openai import OpenAIModel
import prompts

from scoring.utils import SQLEvaluator
from scoring.scorer import Scorer

import datetime

# Directory paths for database, results and scoring program
DB_ID = "mimic_iv"
BASE_DATA_DIR = "data"
RESULT_DIR = "results"
SCORING_DIR = "scoring"

data_split = "valid"
model_name = "gpt-4o"
temperature = 0.6
# File paths for the dataset and labels
TABLES_PATH = os.path.join("database", "tables.json")  # JSON containing database schema

DATA_PATH = os.path.join(BASE_DATA_DIR, f"{data_split}_data.json")  # JSON file for validation data
LABEL_PATH = os.path.join(
    BASE_DATA_DIR, f"{data_split}_label.json"
)  # JSON file for validation labels (for evaluation)

DB_PATH = os.path.join("data", DB_ID, f"{DB_ID}.sqlite")  # Database path

# Load data
with open(os.path.join(DATA_PATH), "r") as f:
    dataset = json.load(f)

with open(os.path.join(LABEL_PATH), "r") as f:
    labels = json.load(f)

# Choose data between valid_data and test_data
data = dataset["data"]

# Load SQL assumptions for MIMIC-IV
assumptions = open("database/mimic_iv_assumption.txt", "r").read()

db_schema, primary_key, foreign_key = load_schema(TABLES_PATH)

table_columns, sql_assumptions = create_schema_prompt(
    DB_ID, db_schema, primary_key, foreign_key, assumptions
)
table_prompt = table_columns + sql_assumptions

# Load api key from json file
with open("sample_submission_chatgpt_api_key.json", "r") as f:
    openai_api_key = json.load(f)["key"]
# new_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIModel(
    model_name=model_name, temperature=temperature, api_key=openai_api_key, async_mode=True
)
evaluator = SQLEvaluator(data_dir="database", dataset=DB_ID)

# Answerable Classification
threshold = 30  # Threshold for answerable classification

classification_result_dict = []
batch_prompts = [
    [
        {
            "role": "system",
            "content": prompts.answerable_classification_system_prompt.format(
                SQL_TABLES=table_columns, SQL_ASSUMPTIONS=sql_assumptions
            ),
        },
        {
            "role": "user",
            "content": prompts.answerable_classification_user_prompt.format(
                USER_QUESTION=sample["question"]
            ),
        },
    ]
    for sample in data
]

responses = model.batch_forward_chatcompletion(batch_prompts)

classification_result_dict = {sample["id"]: response for sample, response in zip(data, responses)}
classification_score_dict = {
    sample["id"]: post_process_score(response) for sample, response in zip(data, responses)
}

sql_prompts = [
    [
        {
            "role": "system",
            "content": prompts.sql_generation_system_prompt.format(
                SQL_TABLES=table_columns, SQL_ASSUMPTIONS=sql_assumptions
            ),
        },
        {
            "role": "user",
            "content": prompts.sql_generation_user_prompt.format(USER_QUESTION=sample["question"]),
        },
    ]
    for sample in data
]

sql_responses = model.batch_forward_chatcompletion(sql_prompts)

# SQL 결과 저장
result_dict = {sample["id"]: response for sample, response in zip(data, sql_responses)}

MAX_RETRY = 5
retry_count_dict = {}
sql_result_dict = {}

for id, response in result_dict.items():
    generated_sql = post_process_sql(response)
    retry_count = 0

    while retry_count < MAX_RETRY:
        sql_result = evaluator.execute(db_id=DB_ID, sql=generated_sql, is_gold_sql=False)

        if len(sql_result) > 3 and "Error" not in sql_result:
            result_dict[id] = generated_sql
            break

        print(
            f"[Retry {retry_count+1}] SQL Execution Failed for ID : {id}\nSQL_RESULT: {sql_result}\nGENERATED_SQL: {generated_sql}\n\n"
        )

        retry_prompt = [
            {
                "role": "system",
                "content": prompts.sql_repair_system_prompt.format(
                    SQL_TABLES=table_columns, SQL_ASSUMPTIONS=sql_assumptions
                ),
            },
            {
                "role": "user",
                "content": prompts.sql_repair_user_prompt.format(
                    USER_QUESTION=next(sample["question"] for sample in data if sample["id"] == id),
                    SQL_QUERY=generated_sql,
                    ERROR_MESSAGE=sql_result,
                ),
            },
        ]

        new_response = model.generate(retry_prompt)
        generated_sql = post_process_sql(new_response)

        retry_count += 1

    retry_count_dict[id] = retry_count
    sql_result_dict[id] = sql_result
    # Abstain for the query that retry 5 times but still not answerable and answerable score is less than 30.
    if retry_count == MAX_RETRY and classification_score_dict[id] <= 30:
        print(
            f"Abstain for ID {id} because the answerable score is less than 30 and retry 5 times but still not answerable."
        )
        result_dict[id] = "Null" #TODO Check this null is correctly abstain the query.


############### SAVE AND EVALUATION ###############
os.makedirs(RESULT_DIR, exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SCORING_OUTPUT_DIR = os.path.join(RESULT_DIR, f"result_{current_time}.json")  # The file to submit
write_label(SCORING_OUTPUT_DIR, result_dict)

with open(LABEL_PATH, "r") as f:
    gold_labels = json.load(f)

scorer = Scorer(data=dataset, predictions=result_dict, gold_labels=gold_labels, score_dir="results")

score = scorer.get_scores()

sql_result_dict = {}
for id in result_dict.keys():
    sql_result = evaluator(db_id=DB_ID, pred_sql=result_dict[id], gold_sql=gold_labels[id])
    sql_result_dict[id] = sql_result

log_dict = {
    "score": score,
    "data_split": data_split,
    "model": model_name,
    "temperature": temperature,
    "logs": [
        {
            "id": id,
            "question": next(sample["question"] for sample in data if sample["id"] == id),
            "generated_sql": result_dict[id],
            "gt_sql_query": gold_labels[id],
            "retry_count": retry_count_dict[id],
            "classification_result": classification_result_dict[id],
            "classification_score": classification_score_dict[id],
            "sql_result": sql_result_dict[id]["pred_answer"],
            "gt_sql_result": sql_result_dict[id]["gold_answer"],
            "is_correct": sql_result_dict[id]["is_correct"],
        }
        for id in result_dict.keys()
    ],
}

os.makedirs(RESULT_DIR, exist_ok=True)
SCORING_OUTPUT_DIR = os.path.join(RESULT_DIR, f"logs_{current_time}.json")  # The file to submit
write_label(SCORING_OUTPUT_DIR, log_dict)
